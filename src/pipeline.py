"""The search pipeline: config in, ranked passages (or nothing) out.

Every choice (chunking, retriever, routing, reranking, when to abstain) comes
from a YAML config so the improvement loop can change it without code edits.

Flow for one query:
  1. parse   "Consider <contract title>; <question>"  ->  title, question
  2. route   pick the contract(s) whose text best matches the title
  3. rank    score chunks of those contracts against the question
  4. rerank  (optional) re-score the top candidates with a cross-encoder
  5. abstain if the contract match or the best passage is too weak
"""

import re
from pathlib import Path

import numpy as np
import yaml

from src.chunking import chunk_document
from src.retrieval import RETRIEVERS, DocRouter, Reranker

TITLE_RE = re.compile(r"^\s*Consider (?:the )?(.+?);\s*(.+)$", re.S | re.I)


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_query(query: str) -> tuple[str | None, str]:
    """Split a benchmark-style query into (contract title, question)."""
    m = TITLE_RE.match(query)
    return (m[1].strip(), m[2].strip()) if m else (None, query.strip())


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
        self.texts = [corpus[fp][s:e] for fp, s, e in self.chunks]
        self.chunk_doc = np.array([fp for fp, _, _ in self.chunks])

        ret_cfg = dict(config["retrieval"])
        self.retriever = RETRIEVERS[ret_cfg.pop("method")](self.texts, **ret_cfg)

        self.routing = config.get("routing", {})
        self.router = DocRouter(corpus) if self.routing.get("enabled") else None

        self.rerank_cfg = config.get("rerank", {})
        self.reranker = Reranker(self.rerank_cfg["model"]) if self.rerank_cfg.get("enabled") else None

        self.abstention = config.get("abstention", {})

    def retrieve(self, query: str, k: int) -> tuple[list[dict], str | None]:
        """Top-k passages and an abstention reason (None means: answer).

        The ranked list is returned even when abstaining, so evaluation can
        score ranking quality separately from the abstention decision.
        """
        title, question = parse_query(query)
        use_question_only = self.config.get("query", {}).get("strip_title", False)
        text = question if (use_question_only and title) else query

        scores = self.retriever.scores(text).astype(float)
        doc_score = None
        if self.router and title:
            ranked_docs = self.router.rank(title)
            doc_score = ranked_docs[0][1]
            allowed = [fp for fp, _ in ranked_docs[: self.routing.get("top_docs", 1)]]
            scores[~np.isin(self.chunk_doc, allowed)] = -np.inf

        n = self.rerank_cfg.get("candidates", 20) if self.reranker else 0
        top = [i for i in np.argsort(-scores, kind="stable")[:max(k, n)] if np.isfinite(scores[i])]
        final = [scores[i] for i in top]
        if self.reranker and top:
            # Re-score only the first `candidates`; the rest keep their retriever order after them.
            head, tail = top[:n], top[n:]
            new = self.reranker.scores(question, [self.texts[i] for i in head])
            order = np.argsort(-np.array(new), kind="stable")
            top = [head[j] for j in order] + tail
            final = [new[j] for j in order] + final[n:]

        results = [
            {"file_path": self.chunks[i][0], "start": self.chunks[i][1], "end": self.chunks[i][2],
             "score": float(s)}
            for i, s in zip(top[:k], final[:k])
        ]
        return results, self._abstain_reason(results, doc_score)

    def _abstain_reason(self, results: list[dict], doc_score: float | None) -> str | None:
        if not results:
            return "no passages"
        min_doc = self.abstention.get("min_doc_score")
        if min_doc is not None and doc_score is not None and doc_score < min_doc:
            return f"no indexed contract matches the title (match {doc_score:.2f} < {min_doc})"
        min_score = self.abstention.get("min_score")
        if min_score is not None and results[0]["score"] < min_score:
            return f"no passage is relevant enough (best {results[0]['score']:.2f} < {min_score})"
        return None

    def search(self, query: str, k: int | None = None) -> tuple[list[dict], str | None]:
        """What a user sees: top-k passages with text, or [] plus the reason for abstaining."""
        results, reason = self.retrieve(query, k or self.config["top_k"])
        if reason:
            return [], reason
        for r in results:
            r["text"] = self.corpus[r["file_path"]][r["start"]:r["end"]]
        return results, None
