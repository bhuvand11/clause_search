"""Score every chunk against a query.

Each retriever is built once from the list of chunk texts and returns one score
per chunk for a query (higher = more relevant).
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Keyword retrieval: rewards chunks that share rare query words."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))


RETRIEVERS = {"bm25": BM25Retriever}
