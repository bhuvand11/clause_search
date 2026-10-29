import pytest

from src.chunking import chunk_fixed, chunk_sentences

CONTRACT = ("1. Term. This Agreement lasts one year.\n\n2. Law. Texas law governs. "
            "Venue is Tarrant County.\n3. Assignment. No assignment without consent.")


def test_sentence_chunks_are_contiguous_and_cover_text():
    spans = chunk_sentences(CONTRACT, size=60)
    assert spans[0][0] == 0 and spans[-1][1] == len(CONTRACT)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(e - s <= 60 for s, e in spans)


def test_sentence_chunks_do_not_cut_sentences():
    for s, e in chunk_sentences(CONTRACT, size=60):
        assert CONTRACT[s:e].rstrip()[-1] in ".:;"


def test_sentence_longer_than_size_is_split():
    text = "a" * 250 + ". Short."
    spans = chunk_sentences(text, size=100)
    assert all(e - s <= 100 for s, e in spans) and spans[-1][1] == len(text)


def test_fixed_chunks_cover_text_without_gaps():
    text = "x" * 1234
    spans = chunk_fixed(text, size=500)
    assert spans == [(0, 500), (500, 1000), (1000, 1234)]


def test_fixed_chunks_with_overlap():
    spans = chunk_fixed("x" * 1000, size=500, overlap=100)
    assert spans == [(0, 500), (400, 900), (800, 1000)]


def test_short_text_is_one_chunk():
    assert chunk_fixed("abc", size=500) == [(0, 3)]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_fixed("abc", size=10, overlap=10)
