# Clause Search

A clause search system for commercial contracts. Given a natural-language question, it returns a
ranked list of the passages most likely to answer it, each with its source document and exact
character offsets, or returns nothing and says why when no passage is relevant enough.

It is built and evaluated on the 194 CUAD queries of LegalBench-RAG-mini (29 contracts), plus 35
hand-written unanswerable queries. It has a web app, a CLI, an evaluation script and a
self-improvement loop that tunes the pipeline on the train split. Results and analysis are in
[REPORT.md](REPORT.md).

## Setup

Requires **Python 3.11** (tested on 3.11.8, Windows 11, CPU only; no GPU needed).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~2 GB smaller
pip install -r requirements.txt
```

Download the data (about 7 MB: `benchmarks/cuad.json` and the 29 contracts the mini queries use,
each checked against the dataset's SHA256SUMS, every gold span checked against its answer text):

```bash
python -m src.data
```

The first run also downloads two small models from Hugging Face into your user cache
(`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`; about 220 MB together).
No API keys are needed; the pipeline does not call any hosted LLM.

## How to run

| Task | Command |
|---|---|
| Web app | `streamlit run app.py` then open http://localhost:8501 |
| CLI | `python -m src.cli "Can this agreement be assigned?" --contract "Cano Petroleum"` |
| List contracts | `python -m src.cli --list-contracts` |
| Evaluate a config | `python -m src.evaluate --config configs/final.yaml --split train` |
| Improvement loop | `python -m src.improve` |
| Tests | `python -m pytest` |

The CLI also accepts benchmark-style questions that name the contract, for example
`python -m src.cli "Consider the Gas Franchise Agreement between the Town of Vinton and Roanoke Gas Company; When does it expire?"`,
and `--json` for machine-readable output.

## Architecture

```
question ──► parse ──► route ──► retrieve ──► rerank ──► abstain? ──► ranked passages
             title +   pick the   score chunks  re-score    title match       (file, start, end,
             question  contract   of that       top 20 with  or best passage    score, text)
                       by title   contract      a cross-     too weak → []
                                  (dense)       encoder      + reason
```

| File | Role |
|---|---|
| `src/data.py` | Downloads and verifies the data; selects the mini queries with the benchmark's `mini()`; splits with `assign_split()`; loads the unanswerable set |
| `src/chunking.py` | Splits contracts into passages that keep their character offsets (fixed windows or sentence-packed) |
| `src/retrieval.py` | BM25, dense (bge-small), hybrid (reciprocal rank fusion), cross-encoder reranker, and the contract router |
| `src/pipeline.py` | Builds a pipeline from a config and runs parse → route → retrieve → rerank → abstain |
| `src/metrics.py` | Character-level precision and recall as in the LegalBench-RAG code, plus document hit rate |
| `src/evaluate.py` | Scores a config on a split: P/R@k for k in 1–64, abstention both ways, 95% bootstrap CIs, latency. Logs every test-split run |
| `src/improve.py` | The self-improvement loop: diagnose, propose, evaluate, accept/reject, log |
| `src/cli.py`, `app.py` | Command-line and Streamlit interfaces over the same pipeline |
| `.streamlit/config.toml` | The web app's theme |
| `configs/baseline.yaml` | The starting pipeline (fixed 500-char chunks + BM25) |
| `configs/final.yaml` | The pipeline chosen by the loop (written by `src.improve`) |
| `queries/unanswerable.json` | 35 labelled queries the indexed contracts cannot answer |
| `experiments/log.jsonl` | Every loop iteration: change, rationale, proposer, metrics, decision |
| `experiments/test_runs.log` | One line per evaluation on the test split |
| `experiments/*_train.json`, `*_test.json` | Saved evaluation outputs behind the numbers in the report |
| `tests/` | Unit tests for data, chunking, metrics, pipeline, loop logic and CLI |

**Configuration.** Every pipeline choice (chunking method and size, retriever, routing, reranking,
abstention thresholds, `top_k`) lives in a YAML file under `configs/`. The loop edits copies of the
config, never the code. Downloaded data and caches (embeddings, reranker scores) live in `data/`,
which is not committed and is rebuilt by the commands above.

## Reproducing results

Run these in order from the repo root. Times are on a laptop CPU (Intel i5-11320H, 4 cores / 8 threads).

| Step | Command | Time |
|---|---|---|
| 1. Data | `python -m src.data` | < 1 min |
| 2. Improvement loop (writes `configs/final.yaml`, appends to `experiments/log.jsonl`) | `python -m src.improve` | ~40 min cold, ~5 min with caches |
| 3. Train metrics | `python -m src.evaluate --config configs/baseline.yaml --split train --out experiments/baseline_train.json` and the same with `final` | seconds (baseline), ~10 min cold (final) |
| 4. Test metrics | `python -m src.evaluate --config configs/baseline.yaml --split test --out experiments/baseline_test.json` and the same with `final` | seconds (baseline), ~3 min (final) |

Step 4 appends to `experiments/test_runs.log`. Results are deterministic: no random sampling is
used at inference, the bootstrap uses a fixed seed, and the router's arithmetic does not depend
on Python's hash seed.

## Assumptions

Where the brief left room for interpretation:

1. **Metric definition.** Precision and recall follow the authors' `run_benchmark.py`: characters
   returned from the wrong document count in the precision denominator. Overlapping returned spans
   are merged before scoring so no character counts twice; for non-overlapping chunks this equals
   the authors' numbers exactly (checked in `tests/test_metrics.py`). k ∈ {1, 2, 4, 8, 16, 32, 64}
   as in the paper.
2. **Ranking vs abstention.** P/R@k score the ranked list as if the system always answered, so they
   measure ranking quality and are comparable with the paper. Abstention is measured separately:
   correct abstention on unanswerable queries and false abstention on benchmark queries.
3. **The contract name in the query is input, not ground truth.** Every benchmark query starts with
   "Consider <contract title>;". The pipeline uses that text to route to a contract. It never uses
   the gold `file_path` or spans at inference. In the app, a user can pick the contract instead.
4. **What counts as unanswerable.** A query is unanswerable if no passage in the 29 indexed
   contracts contains the requested information. A clause from a different contract does not
   answer a question about a named contract. Each missing-clause query was checked two ways: CUAD's
   lawyers annotated no such clause for that contract, and a keyword search of its text finds none.
5. **Split of the unanswerable set.** It is split with the same `assign_split()` (25 train, 10 test).
   Abstention thresholds are fitted on the train portion only.
6. **Unicode file names.** One CUAD `file_path` is NFD while its file name is NFC; all paths are
   NFC-normalised.
7. **Repository visibility.** The brief says both "private" (submission) and "public" (deliverables);
   the repository is private with collaborator access for the reviewers.
8. **Counting test runs.** Each `python -m src.evaluate --split test` invocation counts as one run and
   is logged automatically.

## AI tool usage

Claude Code (Anthropic) was used as a pair programmer: to read the dataset card and the
LegalBench-RAG source code, to write first drafts of modules, tests and documentation, and to
debug. Design decisions (metrics, objective and guardrails, the unanswerable set, which components
to try) were discussed and reviewed by me, and I can explain and modify every part of the code.
