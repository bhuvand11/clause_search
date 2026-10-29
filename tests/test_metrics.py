import pytest

from src.metrics import doc_hit, merge_spans, precision_recall


def authors_precision_recall(retrieved, gold):
    """Direct copy of the logic in LegalBench-RAG run_benchmark.py, for comparison."""
    ret_len = sum(e - s for _, s, e in retrieved)
    gold_len = sum(e - s for _, s, e in gold)
    overlap = 0
    for rf, rs, re_ in retrieved:
        for gf, gs, ge in gold:
            if rf == gf:
                overlap += max(0, min(re_, ge) - max(rs, gs))
    return (overlap / ret_len if ret_len else 0), (overlap / gold_len if gold_len else 0)


def test_exact_match_is_perfect():
    assert precision_recall([("a", 10, 20)], [("a", 10, 20)]) == (1.0, 1.0)


def test_partial_overlap():
    # returned 0-100, gold 50-150: 50 shared chars
    assert precision_recall([("a", 0, 100)], [("a", 50, 150)]) == (0.5, 0.5)


def test_wrong_document_counts_against_precision():
    p, r = precision_recall([("a", 0, 100), ("b", 0, 100)], [("a", 0, 100)])
    assert (p, r) == (0.5, 1.0)


def test_empty_retrieval_scores_zero():
    assert precision_recall([], [("a", 0, 10)]) == (0.0, 0.0)


def test_overlapping_retrieved_spans_not_double_counted():
    p, r = precision_recall([("a", 0, 100), ("a", 50, 150)], [("a", 0, 150)])
    assert (p, r) == (1.0, 1.0)


@pytest.mark.parametrize("retrieved", [
    [("a", 0, 500), ("a", 500, 1000), ("b", 0, 500)],
    [("b", 0, 500)],
    [("a", 900, 1400), ("a", 100, 600)],
])
def test_matches_authors_code_for_disjoint_chunks(retrieved):
    gold = [("a", 120, 480), ("a", 950, 1100)]
    assert precision_recall(retrieved, gold) == pytest.approx(authors_precision_recall(retrieved, gold))


def test_merge_spans():
    assert merge_spans([(5, 8), (0, 3), (2, 4)]) == [(0, 4), (5, 8)]


def test_doc_hit():
    assert doc_hit([("b", 0, 1), ("a", 0, 1)], [("a", 5, 9)]) == 1.0
    assert doc_hit([("b", 0, 1)], [("a", 5, 9)]) == 0.0
