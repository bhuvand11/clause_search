import pytest

from src.chunking import chunk_fixed


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
