"""The search pipeline: config in, ranked passages (or nothing) out.

Every choice (chunking, retriever, how many results, when to abstain) comes
from a YAML config so the improvement loop can change it without code edits.
"""

from pathlib import Path

import numpy as np
import yaml

from src.chunking import chunk_document
from src.retrieval import RETRIEVERS


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class Pipeline:
    def __init__(self, config: dict, corpus: dict[str, str]):
        self.config = config
        self.corpus = corpus

        # Chunk every document and remember where each chunk came from.
        chunk_cfg = dict(config["chunking"])
        method = chunk_cfg.pop("method")
        self.chunks = []  # list of (file_path, start, end)
        for fp, text in corpus.items():
            for start, end in chunk_document(text, method, **chunk_cfg):
                self.chunks.append((fp, start, end))
        texts = [corpus[fp][s:e] for fp, s, e in self.chunks]

        ret_cfg = dict(config["retrieval"])
        self.retriever = RETRIEVERS[ret_cfg.pop("method")](texts, **ret_cfg)

    def rank(self, query: str, k: int) -> list[dict]:
        """Top-k chunks by score, ignoring the abstention rule."""
        scores = self.retriever.scores(query)
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            {"file_path": self.chunks[i][0], "start": self.chunks[i][1],
             "end": self.chunks[i][2], "score": float(scores[i])}
            for i in top
        ]

    def should_abstain(self, results: list[dict]) -> bool:
        threshold = self.config.get("abstention", {}).get("min_score")
        if threshold is None or not results:
            return not results
        return results[0]["score"] < threshold

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """What a user sees: top-k passages with text, or [] if nothing is relevant enough."""
        results = self.rank(query, k or self.config["top_k"])
        if self.should_abstain(results):
            return []
        for r in results:
            r["text"] = self.corpus[r["file_path"]][r["start"]:r["end"]]
        return results
