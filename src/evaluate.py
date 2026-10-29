"""Evaluate a pipeline config on the train or test split.

Retrieval metrics (precision/recall/doc-hit @k) score the ranked list, ignoring
abstention, so they measure ranking quality. Abstention is measured separately
on benchmark queries (should answer) and unanswerable queries (should not).

Usage:
    python -m src.evaluate --config configs/baseline.yaml --split train
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from src.data import load_benchmark, load_unanswerable
from src.metrics import doc_hit, precision_recall
from src.pipeline import Pipeline, load_config

KS = [1, 2, 4, 8, 16, 32, 64]
TEST_LOG = Path(__file__).resolve().parent.parent / "experiments" / "test_runs.log"


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval of the mean, by resampling queries with replacement."""
    v = np.asarray(values, dtype=float)
    means = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(pipeline: Pipeline, queries: list[dict], unanswerable: list[dict]) -> dict:
    per_query = []
    latencies = []
    for q in queries:
        t = time.perf_counter()
        ranked, reason = pipeline.retrieve(q["query"], max(KS))
        latencies.append(time.perf_counter() - t)
        gold = [(s["file_path"], *s["span"]) for s in q["snippets"]]
        row = {"query": q["query"], "gold": gold, "ranked": ranked, "doc_score": pipeline.route(q["query"])[1],
               "abstained": reason is not None, "abstain_reason": reason}
        for k in KS:
            retrieved = [(r["file_path"], r["start"], r["end"]) for r in ranked[:k]]
            row[f"precision@{k}"], row[f"recall@{k}"] = precision_recall(retrieved, gold)
            row[f"doc_hit@{k}"] = doc_hit(retrieved, gold)
        per_query.append(row)

    unans = []
    for u in unanswerable:
        ranked, reason = pipeline.retrieve(u["query"], pipeline.config["top_k"])
        unans.append({"id": u["id"], "category": u["category"], "query": u["query"],
                      "doc_score": pipeline.route(u["query"])[1],
                      "abstained": reason is not None, "abstain_reason": reason, "ranked": ranked})

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {"n_queries": len(per_query), "n_unanswerable": len(unans)}
    for k in KS:
        for m in ("precision", "recall", "doc_hit"):
            summary[f"{m}@{k}"] = mean([r[f"{m}@{k}"] for r in per_query])
    summary["false_abstention_rate"] = mean([r["abstained"] for r in per_query])
    summary["correct_abstention_rate"] = mean([u["abstained"] for u in unans])
    summary["abstention_by_category"] = {
        c: f"{sum(u['abstained'] for u in unans if u['category'] == c)}/{sum(u['category'] == c for u in unans)}"
        for c in sorted({u["category"] for u in unans})
    }
    summary["ci95"] = {m: bootstrap_ci([r[m] for r in per_query])
                       for m in ("recall@1", "recall@8", "recall@64", "precision@8")}
    summary["latency_ms"] = {"mean": 1000 * mean(latencies), "p95": 1000 * float(np.percentile(latencies, 95))}
    return {"summary": summary, "per_query": per_query, "unanswerable": unans}


def run(config_path: str, split: str) -> dict:
    config = load_config(config_path)
    queries, corpus = load_benchmark()
    queries = [q for q in queries if q["split"] == split]
    unanswerable = [u for u in load_unanswerable() if u["split"] == split]
    if split == "test":
        TEST_LOG.parent.mkdir(exist_ok=True)
        with TEST_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{config_path}\n")
    return evaluate(Pipeline(config, corpus), queries, unanswerable)


def print_summary(s: dict) -> None:
    print(f"Benchmark queries: {s['n_queries']}   Unanswerable queries: {s['n_unanswerable']}")
    print(f"{'k':>4} {'Precision':>10} {'Recall':>8} {'DocHit':>8}")
    for k in KS:
        print(f"{k:>4} {100*s[f'precision@{k}']:>9.2f}% {100*s[f'recall@{k}']:>7.2f}% {100*s[f'doc_hit@{k}']:>7.2f}%")
    print(f"Correct abstention (unanswerable): {100*s['correct_abstention_rate']:.1f}%  {s['abstention_by_category']}")
    print(f"False abstention (benchmark):      {100*s['false_abstention_rate']:.1f}%")
    print("95% CI: " + "  ".join(f"{m} [{100*lo:.1f}, {100*hi:.1f}]" for m, (lo, hi) in s["ci95"].items()))
    print(f"Latency per query: mean {s['latency_ms']['mean']:.0f} ms, p95 {s['latency_ms']['p95']:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out", help="optional path to save the summary and per-query results as JSON")
    args = parser.parse_args()
    result = run(args.config, args.split)
    print_summary(result["summary"])
    if args.out:
        for row in result["per_query"] + result["unanswerable"]:
            row["ranked"] = row["ranked"][:8]  # keep the saved file small
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
