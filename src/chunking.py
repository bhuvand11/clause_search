"""Split a contract into passages, keeping each passage's character offsets.

A chunk is a (start, end) pair into the original text, so every result can be
traced back to exact characters in the source document.
"""

import re


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


# A boundary is after a line break, or after . ; : followed by whitespace.
BOUNDARY_RE = re.compile(r"\n+|(?<=[.;:])\s+")


def chunk_sentences(text: str, size: int) -> list[tuple[int, int]]:
    """Pack whole sentences/lines into chunks of at most `size` characters.

    Chunks never cut a sentence unless that sentence alone is longer than
    `size`, in which case it is split into fixed windows.
    """
    cuts = sorted({0, len(text), *(m.end() for m in BOUNDARY_RE.finditer(text))})
    units = list(zip(cuts, cuts[1:]))
    spans = []
    start = None
    for u_start, u_end in units:
        if u_end - u_start > size:
            if start is not None:
                spans.append((start, u_start))
                start = None
            spans += [(u_start + s, u_start + e) for s, e in chunk_fixed(text[u_start:u_end], size)]
        elif start is None:
            start = u_start
        elif u_end - start > size:
            spans.append((start, u_start))
            start = u_start
    if start is not None:
        spans.append((start, len(text)))
    return spans


CHUNKERS = {"fixed": chunk_fixed, "sentence": chunk_sentences}


def chunk_document(text: str, method: str, **params) -> list[tuple[int, int]]:
    return CHUNKERS[method](text, **params)
