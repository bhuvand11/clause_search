"""Character-level precision and recall, as in LegalBench-RAG (run_benchmark.py).

For one query, R = characters returned, G = gold characters:
    precision = |R & G| / |R|      recall = |R & G| / |G|
Characters returned from the wrong document count in |R| but never in |R & G|.
Overlapping returned spans are merged first, so no character is counted twice;
for non-overlapping chunks this gives exactly the authors' numbers.
"""


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _by_file(items) -> dict[str, list[tuple[int, int]]]:
    out = {}
    for fp, start, end in items:
        out.setdefault(fp, []).append((start, end))
    return {fp: merge_spans(spans) for fp, spans in out.items()}


def precision_recall(retrieved, gold) -> tuple[float, float]:
    """retrieved, gold: lists of (file_path, start, end). Empty retrieval scores 0, 0."""
    ret, gt = _by_file(retrieved), _by_file(gold)
    ret_len = sum(e - s for spans in ret.values() for s, e in spans)
    gold_len = sum(e - s for spans in gt.values() for s, e in spans)
    overlap = 0
    for fp, spans in ret.items():
        for rs, re_ in spans:
            for gs, ge in gt.get(fp, []):
                overlap += max(0, min(re_, ge) - max(rs, gs))
    precision = overlap / ret_len if ret_len else 0.0
    recall = overlap / gold_len if gold_len else 0.0
    return precision, recall


def doc_hit(retrieved, gold) -> float:
    """1.0 if any returned passage is from the gold document, else 0.0."""
    gold_docs = {fp for fp, _, _ in gold}
    return float(any(fp in gold_docs for fp, _, _ in retrieved))
