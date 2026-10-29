"""Split a contract into passages, keeping each passage's character offsets.

A chunk is a (start, end) pair into the original text, so every result can be
traced back to exact characters in the source document.
"""


def chunk_fixed(text: str, size: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Fixed-size windows of `size` characters, each starting `size - overlap` after the last."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    spans = []
    for start in range(0, len(text), step):
        spans.append((start, min(start + size, len(text))))
        if start + size >= len(text):
            break
    return spans


CHUNKERS = {"fixed": chunk_fixed}


def chunk_document(text: str, method: str, **params) -> list[tuple[int, int]]:
    return CHUNKERS[method](text, **params)
