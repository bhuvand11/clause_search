"""Score chunks (and whole documents) against a query.

Chunk retrievers are built once from the list of chunk texts and return one
score per chunk for a query (higher = more relevant). The DocRouter picks which
contract a query is about. The Reranker re-scores a short candidate list.
"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_MODELS = {}  # loaded models, shared across pipelines in one process


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _load(kind: str, name: str):
    if (kind, name) not in _MODELS:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        cls = SentenceTransformer if kind == "embed" else CrossEncoder
        _MODELS[(kind, name)] = cls(name, device="cpu")
    return _MODELS[(kind, name)]


class BM25Retriever:
    """Keyword retrieval: rewards chunks that share rare query words."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))


class DenseRetriever:
    """Meaning-based retrieval: cosine similarity between neural embeddings.

    Chunk embeddings are cached on disk, keyed by model and chunk texts, so
    re-running a config with the same chunks does not re-embed.
    """

    def __init__(self, texts: list[str], model: str = "BAAI/bge-small-en-v1.5", query_prefix: str = ""):
        self.model = _load("embed", model)
        self.query_prefix = query_prefix
        key = hashlib.sha1((model + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = CACHE_DIR / f"emb_{key}.npy"
        if path.exists():
            self.embeddings = np.load(path)
        else:
            self.embeddings = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                                show_progress_bar=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, self.embeddings)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        return self.embeddings @ q


class HybridRetriever:
    """Reciprocal rank fusion of BM25 and dense: each adds weight / (60 + rank)."""

    def __init__(self, texts: list[str], bm25_weight: float = 0.5, rrf_k: int = 60, **dense_params):
        self.bm25 = BM25Retriever(texts)
        self.dense = DenseRetriever(texts, **dense_params)
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def _ranks(self, scores: np.ndarray) -> np.ndarray:
        ranks = np.empty(len(scores))
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        return ranks

    def scores(self, query: str) -> np.ndarray:
        rb = self._ranks(self.bm25.scores(query))
        rd = self._ranks(self.dense.scores(query))
        w = self.bm25_weight
        return w / (self.rrf_k + rb) + (1 - w) / (self.rrf_k + rd)


RETRIEVERS = {"bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever}


class Reranker:
    """Cross-encoder: reads query and passage together and outputs a relevance score.

    Scores are cached on disk, keyed by a hash of (model, query, passage), so the
    improvement loop never scores the same pair twice.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = _load("cross", model)
        self.name = model
        self.path = CACHE_DIR / f"rerank_{hashlib.sha1(model.encode()).hexdigest()[:8]}.json"
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _key(self, query: str, text: str) -> str:
        return hashlib.sha1(f"{query}\x00{text}".encode("utf-8")).hexdigest()

    def scores(self, query: str, texts: list[str]) -> list[float]:
        keys = [self._key(query, t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if missing:
            preds = self.model.predict([(query, t) for _, t in missing], batch_size=32)
            self.cache.update({k: float(s) for (k, _), s in zip(missing, preds)})
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
        return [self.cache[k] for k in keys]


class DocRouter:
    """Picks the contract a title refers to.

    Score = share of the title's IDF weight found in the document (text plus
    file name). Rare words like company names dominate; words in every contract
    ("agreement", "the") count for almost nothing. A title whose distinctive
    words appear nowhere scores near 0, which signals an unknown contract.
    """

    def __init__(self, corpus: dict[str, str]):
        self.doc_ids = list(corpus)
        self.vocab = []
        for fp, text in corpus.items():
            name = re.sub(r"([a-z])([A-Z])", r"\1 \2", Path(fp).stem).replace("_", " ")
            self.vocab.append(set(tokenize(text)) | set(tokenize(name)))
        df = Counter(t for v in self.vocab for t in v)
        n = len(self.vocab)
        self.idf = lambda t: math.log((n + 1) / (df.get(t, 0) + 0.5))

    def rank(self, title: str) -> list[tuple[str, float]]:
        terms = set(tokenize(title))
        total = sum(self.idf(t) for t in terms) or 1.0
        scored = [(fp, sum(self.idf(t) for t in terms if t in vocab) / total)
                  for fp, vocab in zip(self.doc_ids, self.vocab)]
        return sorted(scored, key=lambda x: -x[1])
