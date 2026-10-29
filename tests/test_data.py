import pytest

from src.data import assign_split, mini, nfc, verify_offsets


def test_assign_split_is_deterministic_and_ignores_surrounding_whitespace():
    q = "Consider the X agreement; What is the governing law?"
    assert assign_split(q) == assign_split(q) == assign_split(f"  {q}\n")


def test_assign_split_puts_about_30_percent_in_test():
    splits = [assign_split(f"question {i}") for i in range(2000)]
    assert 0.25 < splits.count("test") / len(splits) < 0.35


def test_mini_keeps_n_and_groups_queries_by_document():
    tests = [{"snippets": [{"file_path": f"doc{i % 7}.txt"}], "query": str(i)} for i in range(300)]
    picked = mini(tests)
    assert len(picked) == 194 and picked == mini(tests)
    # Sorting by a per-document key keeps each document's queries together.
    docs = [t["snippets"][0]["file_path"] for t in picked]
    assert len({d for i, d in enumerate(docs) if i == 0 or d != docs[i - 1]}) == len(set(docs))


def test_nfc_normalises_decomposed_characters():
    assert nfc("Rodé") == "Rodé"


def test_verify_offsets_catches_mismatch():
    corpus = {"d.txt": "hello world"}
    good = [{"snippets": [{"file_path": "d.txt", "span": (6, 11), "answer": "world"}]}]
    verify_offsets(good, corpus)
    bad = [{"snippets": [{"file_path": "d.txt", "span": (0, 5), "answer": "world"}]}]
    with pytest.raises(ValueError):
        verify_offsets(bad, corpus)
