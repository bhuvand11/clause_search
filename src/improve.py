"""Self-improvement loop. Runs on the train split only.

Each iteration:
  1. diagnose  sort the current config's failures into buckets
  2. propose   pick the untried change that targets the biggest failure buckets
  3. evaluate  run the changed config on train
  4. decide    accept or reject it (bootstrap test on the primary metric + guardrails)
  5. log       append everything to experiments/log.jsonl

Usage:
    python -m src.improve                       # start from configs/baseline.yaml
    python -m src.improve --max-iters 5
"""

import argparse
import copy
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.data import load_benchmark, load_unanswerable
from src.evaluate import evaluate
from src.pipeline import Pipeline, load_config

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "experiments" / "log.jsonl"

# --- Objective -------------------------------------------------------------
# Primary: Recall@8. A reviewer reads about 8 passages; missing a clause is the
# costliest error in legal review, so coverage of the gold text comes first.
# Guardrails: a change is rejected if any of these breaks.
PRIMARY = "recall@8"
K = 8
MIN_PRECISION_RATIO = 0.9     # precision@8 must stay >= 90% of the current config's
MAX_FALSE_ABSTENTION = 0.05   # at most 5% of real questions may get "no answer"
MAX_ABSTENTION_DROP = 1       # correct abstentions may drop by at most 1 query
P_ACCEPT = 0.95               # bootstrap: change must win in >= 95% of resamples

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DENSE = {"method": "dense", "model": "BAAI/bge-small-en-v1.5", "query_prefix": BGE_PREFIX}
HYBRID = {**DENSE, "method": "hybrid", "bm25_weight": 0.5}
RERANK = {"enabled": True, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "candidates": 20}


# --- Proposals -------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    targets: tuple[str, ...]          # failure buckets this change is meant to fix
    rationale: str
    apply: Callable[[dict], dict]     # returns a changed copy of the config
    applies_to: Callable[[dict], bool] = lambda c: True


def with_(cfg: dict, section: str, **values) -> dict:
    new = copy.deepcopy(cfg)
    new.setdefault(section, {}).update(values)
    return new


def replace(cfg: dict, section: str, value) -> dict:
    new = copy.deepcopy(cfg)
    new[section] = copy.deepcopy(value)
    return new


CANDIDATES = [
    Candidate("strip_title", ("wrong_section",),
              "Party names in the title only appear on the cover page and pull results there; rank on the question only.",
              lambda c: with_(c, "query", strip_title=True),
              lambda c: not c.get("query", {}).get("strip_title")),
    Candidate("route_to_contract", ("wrong_document", "wrong_section"),
              "Every query names its contract; search only the contract whose text best matches the title.",
              lambda c: with_(c, "routing", enabled=True, top_docs=1),
              lambda c: not c.get("routing", {}).get("enabled")),
    Candidate("dense_retrieval", ("wrong_section", "wrong_document"),
              "Clause wording differs from question wording; match by meaning instead of shared words.",
              lambda c: replace(c, "retrieval", DENSE),
              lambda c: c["retrieval"]["method"] != "dense"),
    Candidate("hybrid_retrieval", ("wrong_section", "wrong_document"),
              "Combine exact keyword matches (BM25) with meaning matches (dense) via rank fusion.",
              lambda c: replace(c, "retrieval", HYBRID),
              lambda c: c["retrieval"]["method"] != "hybrid"),
    Candidate("cross_encoder_rerank", ("wrong_section",),
              "Re-score the top 20 with a model that reads question and passage together.",
              lambda c: replace(c, "rerank", RERANK),
              lambda c: not c.get("rerank", {}).get("enabled")),
    Candidate("sentence_chunking", ("partial_span",),
              "Fixed windows cut clauses mid-sentence; cut only at sentence/line boundaries.",
              lambda c: replace(c, "chunking", {"method": "sentence", "size": c["chunking"]["size"]}),
              lambda c: c["chunking"]["method"] != "sentence"),
    Candidate("chunk_overlap", ("partial_span",),
              "Answers that straddle a chunk boundary are split; overlap windows by 20%.",
              lambda c: with_(c, "chunking", overlap=c["chunking"]["size"] // 5),
              lambda c: c["chunking"]["method"] == "fixed" and not c["chunking"].get("overlap")),
    Candidate("larger_chunks", ("partial_span",),
              "Many gold spans are only partly covered; make passages 50% longer.",
              lambda c: with_(c, "chunking", size=int(c["chunking"]["size"] * 1.5))),
    Candidate("smaller_chunks", ("wrong_section",),
              "Shorter passages let the ranker isolate the clause instead of mixing it with unrelated text.",
              lambda c: with_(c, "chunking", size=int(c["chunking"]["size"] * 0.6))),
    Candidate("rerank_more_candidates", ("wrong_section",),
              "Some gold passages sit just outside the reranked top 20; rerank the top 40.",
              lambda c: with_(c, "rerank", candidates=40),
              lambda c: c.get("rerank", {}).get("enabled") and c["rerank"].get("candidates", 20) < 40),
    Candidate("hybrid_favour_dense", ("wrong_section",),
              "Give the dense ranking more weight than BM25 in the fusion (0.3 / 0.7).",
              lambda c: with_(c, "retrieval", bm25_weight=0.3),
              lambda c: c["retrieval"]["method"] == "hybrid" and c["retrieval"].get("bm25_weight", 0.5) > 0.3),
    Candidate("route_top2_contracts", ("wrong_document",),
              "Hedge against routing mistakes by searching the two best-matching contracts.",
              lambda c: with_(c, "routing", top_docs=2),
              lambda c: c.get("routing", {}).get("enabled") and c["routing"].get("top_docs", 1) < 2),
    Candidate("calibrated_abstention", ("missed_abstention", "false_abstention"),
              "The system never says 'no answer'; learn title-match and passage-score thresholds on train.",
              lambda c: with_(c, "abstention", calibrate=True),
              lambda c: not c.get("abstention", {}).get("calibrate")),
]


# --- Diagnosis -------------------------------------------------------------

def diagnose(result: dict) -> dict:
    """Count failures by cause, at the cut-off a user sees (top K)."""
    buckets = dict.fromkeys(["ok", "wrong_document", "wrong_section", "partial_span",
                             "false_abstention", "missed_abstention"], 0)
    for r in result["per_query"]:
        if r["abstained"]:
            buckets["false_abstention"] += 1
        elif r[f"doc_hit@{K}"] == 0:
            buckets["wrong_document"] += 1
        elif r[f"recall@{K}"] == 0:
            buckets["wrong_section"] += 1
        elif r[f"recall@{K}"] < 0.999:
            buckets["partial_span"] += 1
        else:
            buckets["ok"] += 1
    buckets["missed_abstention"] = sum(not u["abstained"] for u in result["unanswerable"])
    return buckets


def propose(config: dict, buckets: dict, tried: dict) -> Candidate | None:
    """Rule-based: the applicable candidate whose target buckets hold the most failures.

    A rejected candidate may be retried once after the config has changed (the
    context it was rejected in no longer holds); ties go to list order.
    """
    best, best_score = None, 0
    for cand in CANDIDATES:
        if not cand.applies_to(config) or tried.get(cand.name, 0) >= 2:
            continue
        if tried.get(cand.name) == 1 and tried.get("_since_accept", {}).get(cand.name):
            continue  # already tried against the current config
        score = sum(buckets[t] for t in cand.targets)
        if score > best_score:
            best, best_score = cand, score
    return best


# --- Abstention calibration ------------------------------------------------

def calibrate(result: dict) -> tuple[float | None, float | None]:
    """Thresholds that catch the most unanswerable queries with <= 5% false abstention.

    A query abstains if its title match < doc threshold or its best passage
    score < score threshold. Candidate thresholds are the observed scores.
    """
    def arrays(rows):
        doc = np.array([r["doc_score"] if r["doc_score"] is not None else np.inf for r in rows])
        top = np.array([r["ranked"][0]["score"] if r["ranked"] else -np.inf for r in rows])
        return doc, top

    b_doc, b_top = arrays(result["per_query"])
    u_doc, u_top = arrays(result["unanswerable"])
    max_false = int(MAX_FALSE_ABSTENTION * len(b_doc))
    doc_options = [None] + sorted(set(b_doc[np.isfinite(b_doc)]))
    top_options = [None] + sorted(set(b_top[np.isfinite(b_top)]))

    best = (-1, 0, None, None)  # (caught, -false, doc_thr, top_thr)
    for dt_ in doc_options:
        b_d = b_doc < dt_ if dt_ is not None else np.zeros(len(b_doc), bool)
        u_d = u_doc < dt_ if dt_ is not None else np.zeros(len(u_doc), bool)
        for tt in top_options:
            b_abs = b_d | (b_top < tt) if tt is not None else b_d
            false = int(b_abs.sum())
            if false > max_false:
                break  # higher score thresholds only abstain more
            caught = int((u_d | (u_top < tt) if tt is not None else u_d).sum())
            if (caught, -false) > best[:2]:
                best = (caught, -false, dt_, tt)
    return (None if best[2] is None else float(best[2]),
            None if best[3] is None else float(best[3]))


# --- Evaluation and decision ----------------------------------------------

def run_config(config: dict, corpus, queries, unanswerable) -> tuple[dict, dict]:
    """Evaluate a config on train; if calibration is on, fit thresholds first."""
    pipeline = Pipeline(config, corpus)
    result = evaluate(pipeline, queries, unanswerable)
    if config.get("abstention", {}).get("calibrate"):
        doc_thr, score_thr = calibrate(result)
        config = with_(config, "abstention", min_doc_score=doc_thr, min_score=score_thr)
        pipeline.abstention = config["abstention"]
        result = evaluate(pipeline, queries, unanswerable)
    return config, result


def bootstrap_p_better(new: np.ndarray, old: np.ndarray, n: int = 1000, seed: int = 0) -> float:
    """Share of resamples (of the same queries) in which the new config has the higher mean."""
    rng = np.random.default_rng(seed)
    delta = new - old
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    return float((delta[idx].mean(axis=1) > 0).mean())


def decide(cur: dict, new: dict) -> tuple[bool, str, dict]:
    s_cur, s_new = cur["summary"], new["summary"]
    per = lambda res: np.array([r[PRIMARY] for r in res["per_query"]])
    delta = s_new[PRIMARY] - s_cur[PRIMARY]
    p_better = bootstrap_p_better(per(new), per(cur))
    caught_cur = sum(u["abstained"] for u in cur["unanswerable"])
    caught_new = sum(u["abstained"] for u in new["unanswerable"])
    stats = {"delta_primary": delta, "p_better": p_better, "caught_before": caught_cur, "caught_after": caught_new}

    if s_new[f"precision@{K}"] < MIN_PRECISION_RATIO * s_cur[f"precision@{K}"]:
        return False, f"guardrail: precision@{K} fell more than 10%", stats
    if s_new["false_abstention_rate"] > MAX_FALSE_ABSTENTION:
        return False, "guardrail: false abstention above 5%", stats
    if caught_new < caught_cur - MAX_ABSTENTION_DROP:
        return False, "guardrail: correct abstentions dropped", stats
    if p_better >= P_ACCEPT and delta > 0:
        return True, f"{PRIMARY} improved in {p_better:.0%} of bootstrap resamples", stats
    if abs(delta) < 1e-9 and caught_new > caught_cur:
        return True, f"abstention improved ({caught_cur} -> {caught_new} caught) with {PRIMARY} unchanged", stats
    return False, f"{PRIMARY} improvement not reliable ({p_better:.0%} of resamples < {P_ACCEPT:.0%})", stats


# --- Main loop -------------------------------------------------------------

def log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def headline(s: dict) -> dict:
    keys = ["recall@1", "recall@8", "recall@64", "precision@1", "precision@8", "doc_hit@8",
            "correct_abstention_rate", "false_abstention_rate"]
    return {k: round(s[k], 4) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "configs" / "final.yaml"))
    args = parser.parse_args()

    queries, corpus = load_benchmark()
    queries = [q for q in queries if q["split"] == "train"]
    unanswerable = [u for u in load_unanswerable() if u["split"] == "train"]
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    t = time.time()
    config, current = run_config(load_config(args.config), corpus, queries, unanswerable)
    buckets = diagnose(current)
    log({"run_id": run_id, "iteration": 0, "change": "baseline", "proposer": "human",
         "rationale": "Simplest reasonable pipeline", "config": config, "metrics": headline(current["summary"]),
         "diagnosis": buckets, "decision": "accepted", "reason": "starting point", "seconds": round(time.time() - t, 1)})
    print(f"[0] baseline  {PRIMARY}={current['summary'][PRIMARY]:.4f}  diagnosis={buckets}")

    tried = {"_since_accept": {}}
    for it in range(1, args.max_iters + 1):
        cand = propose(config, buckets, tried)
        if cand is None:
            print("No applicable proposals left.")
            break
        tried[cand.name] = tried.get(cand.name, 0) + 1
        tried["_since_accept"][cand.name] = True

        t = time.time()
        new_config, new = run_config(cand.apply(config), corpus, queries, unanswerable)
        accepted, reason, stats = decide(current, new)
        new_buckets = diagnose(new)
        log({"run_id": run_id, "iteration": it, "change": cand.name,
             "proposer": "rule:" + "+".join(cand.targets), "rationale": cand.rationale,
             "diagnosis_before": buckets, "diagnosis_after": new_buckets, "config": new_config,
             "metrics": headline(new["summary"]), **stats,
             "decision": "accepted" if accepted else "rejected", "reason": reason,
             "seconds": round(time.time() - t, 1)})
        print(f"[{it}] {cand.name:24} {PRIMARY} {current['summary'][PRIMARY]:.4f} -> {new['summary'][PRIMARY]:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}: {reason}")
        if accepted:
            config, current, buckets = new_config, new, new_buckets
            tried["_since_accept"] = {}

    header = (f"# Generated by `python -m src.improve` (run {run_id}) from {Path(args.config).name}.\n"
              f"# Train {PRIMARY} = {current['summary'][PRIMARY]:.4f}. See experiments/log.jsonl for every iteration.\n")
    Path(args.out).write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Final config written to {args.out}")


if __name__ == "__main__":
    main()
