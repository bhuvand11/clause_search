"""Download, load and split the LegalBench-RAG-mini CUAD data.

Only the 194 mini queries and the contracts they point to are downloaded.
Every downloaded file is checked against the dataset's SHA256SUMS, and every
gold span is checked so that document[start:end] == answer.
"""

import hashlib
import json
import random
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/awinml/legalbench-rag/resolve/main"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --- Selection and split functions, used exactly as given by the benchmark ---

def mini(tests, n=194):
    if len(tests) <= n:
        return tests

    def key(t):
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


# --- Download ---

def nfc(path: str) -> str:
    # One CUAD file_path is in Unicode NFD while the file name is NFC.
    return unicodedata.normalize("NFC", path)


def _fetch(rel_path: str) -> bytes:
    url = f"{HF_BASE}/{urllib.parse.quote(rel_path)}"
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def download(data_dir: Path = DATA_DIR) -> None:
    """Fetch cuad.json and the corpus files used by the mini queries."""
    checksums = {}
    for line in _fetch("SHA256SUMS").decode("utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        checksums[nfc(rel)] = digest

    _download_file("benchmarks/cuad.json", data_dir, checksums)
    raw = (data_dir / "benchmarks" / "cuad.json").read_text(encoding="utf-8")
    tests = json.loads(raw)["tests"]

    doc_paths = sorted({nfc(s["file_path"]) for t in mini(tests) for s in t["snippets"]})
    for fp in doc_paths:
        _download_file(f"corpus/{fp}", data_dir, checksums)
    print(f"Ready: cuad.json + {len(doc_paths)} contracts in {data_dir}")


def _download_file(rel: str, data_dir: Path, checksums: dict) -> None:
    dest = data_dir / rel
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == checksums[rel]:
        return
    content = _fetch(rel)
    if hashlib.sha256(content).hexdigest() != checksums[rel]:
        raise ValueError(f"Checksum mismatch for {rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


# --- Loading ---

def load_corpus(file_paths, data_dir: Path = DATA_DIR) -> dict:
    """Read each contract in default text mode with UTF-8, as the offsets assume."""
    return {
        fp: (data_dir / "corpus" / fp).read_text(encoding="utf-8")
        for fp in sorted(set(file_paths))
    }


def load_benchmark(data_dir: Path = DATA_DIR):
    """Return (queries, corpus) for the 194 CUAD mini queries.

    Each query is a dict with keys: query, split, snippets (file_path, span, answer).
    File paths are NFC-normalised so they match the files on disk.
    """
    raw = (data_dir / "benchmarks" / "cuad.json").read_text(encoding="utf-8")
    tests = json.loads(raw)["tests"]
    queries = []
    for t in mini(tests):
        snippets = [
            {"file_path": nfc(s["file_path"]), "span": tuple(s["span"]), "answer": s["answer"]}
            for s in t["snippets"]
        ]
        queries.append({"query": t["query"], "split": assign_split(t["query"]), "snippets": snippets})
    corpus = load_corpus([s["file_path"] for q in queries for s in q["snippets"]], data_dir)
    verify_offsets(queries, corpus)
    return queries, corpus


def contract_titles(queries) -> dict[str, str]:
    """Readable name per contract, taken from the 'Consider the <title>;' part of its queries."""
    titles = {}
    for q in queries:
        head = q["query"].split(";", 1)[0]
        titles.setdefault(q["snippets"][0]["file_path"], head.removeprefix("Consider ").removeprefix("the "))
    return titles


def verify_offsets(queries, corpus) -> None:
    for q in queries:
        for s in q["snippets"]:
            start, end = s["span"]
            if corpus[s["file_path"]][start:end] != s["answer"]:
                raise ValueError(f"Offset mismatch in {s['file_path']} at {s['span']}")


def load_unanswerable(path: Path = Path(__file__).resolve().parent.parent / "queries" / "unanswerable.json"):
    """Return the hand-written unanswerable queries, each with its split."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [{**item, "split": assign_split(item["query"])} for item in items]


if __name__ == "__main__":
    download()
    queries, corpus = load_benchmark()
    n_snip = sum(len(q["snippets"]) for q in queries)
    n_train = sum(q["split"] == "train" for q in queries)
    print(f"{len(queries)} queries, {n_snip} snippets, {len(corpus)} contracts, "
          f"{sum(map(len, corpus.values())):,} chars. All offsets verified.")
    print(f"Split: train={n_train}, test={len(queries) - n_train}")
