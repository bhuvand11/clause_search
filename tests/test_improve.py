import numpy as np

from src.improve import bootstrap_p_better, calibrate, decide, diagnose, propose


def row(recall=0.0, doc_hit=1.0, abstained=False, doc_score=1.0, top=0.0, precision=0.1):
    return {"recall@8": recall, "doc_hit@8": doc_hit, "precision@8": precision, "abstained": abstained,
            "doc_score": doc_score, "ranked": [{"score": top}]}


def result(rows, unans, **summary):
    s = {"recall@8": np.mean([r["recall@8"] for r in rows]), "precision@8": 0.1,
         "false_abstention_rate": np.mean([r["abstained"] for r in rows]), **summary}
    return {"per_query": rows, "unanswerable": unans, "summary": s}


def test_diagnose_buckets():
    rows = [row(abstained=True), row(doc_hit=0), row(recall=0), row(recall=0.5), row(recall=1.0)]
    unans = [{"abstained": False}, {"abstained": True}]
    assert diagnose(result(rows, unans)) == {"ok": 1, "wrong_document": 1, "wrong_section": 1, "partial_span": 1,
                                             "false_abstention": 1, "missed_abstention": 1}


def test_bootstrap_clear_win_and_no_difference():
    old = np.zeros(100)
    assert bootstrap_p_better(np.full(100, 0.2), old) == 1.0
    assert bootstrap_p_better(old, old) == 0.0


def test_calibrate_catches_unknown_titles_without_false_abstention():
    rows = [row(doc_score=0.95 + i / 1000, top=-2 + i / 100) for i in range(40)]
    unans = [{"doc_score": 0.2, "ranked": [{"score": 3.0}]},   # unknown contract
             {"doc_score": None, "ranked": [{"score": -9.0}]}]  # off-topic, low passage score
    doc_thr, score_thr = calibrate(result(rows, unans))
    assert 0.2 < doc_thr < 0.95 and -9.0 < score_thr < -2.0
    observed = {r["doc_score"] for r in rows} | {r["ranked"][0]["score"] for r in rows}
    assert doc_thr not in observed and score_thr not in observed  # thresholds sit between scores


def test_decide_rejects_precision_guardrail_and_accepts_clear_win():
    cur = result([row(recall=0.2) for _ in range(50)], [])
    worse_precision = result([row(recall=0.9) for _ in range(50)], [], **{"precision@8": 0.05})
    assert decide(cur, worse_precision)[0] is False
    better = result([row(recall=0.9) for _ in range(50)], [])
    assert decide(cur, better)[0] is True


def test_propose_picks_candidate_targeting_biggest_bucket():
    config = {"chunking": {"method": "fixed", "size": 500}, "retrieval": {"method": "bm25"}}
    buckets = {"ok": 0, "wrong_document": 0, "wrong_section": 100, "partial_span": 5,
               "false_abstention": 0, "missed_abstention": 25}
    assert propose(config, buckets, {"_since_accept": {}}).name == "strip_title"
    tried = {"strip_title": 1, "_since_accept": {"strip_title": True}}
    assert propose(config, buckets, tried).name == "route_to_contract"
