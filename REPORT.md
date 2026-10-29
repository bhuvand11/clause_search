# Evaluation Report: Clause Search over CUAD Contracts

## Summary

I built a clause search system over the 29 CUAD contracts in LegalBench-RAG-mini. A question
first routes to the contract it names. The system then retrieves 500-character passages from
that contract with a dense embedding model (bge-small), reranks the top 20 with a cross-encoder,
and abstains when either the contract match or the best passage score is too weak. Every choice
came out of a logged improvement loop that ran on the train split only. On the held-out test
split (54 benchmark queries), the final pipeline reaches **Recall@8 = 64.1%** (95% CI 52.6–74.5)
and **Recall@64 = 93.4%**, up from 12.3% and 36.4% for the BM25 baseline. It correctly says "not
found" for **8 of 10** unanswerable test queries (all 6 questions about contracts that are not in
the corpus) and wrongly abstains on **1 of 54** real queries. The test split was evaluated twice
in total: once for the baseline and once for the final configuration, after that configuration
was frozen in commit `d87f81d`. Everything runs locally on a CPU with no LLM. Mean latency is
about 0.5 s per query.

## System design

```
question → parse (title; question) → route to contract → dense retrieval → cross-encoder rerank → abstain? → passages + offsets
```

| Decision | Chosen | Alternatives tried or considered | Why |
|---|---|---|---|
| Passages | Fixed 500-char windows, no overlap | Sentence-packed chunks, 20% overlap, 300- and 750-char windows | Sentence chunks and overlap moved Recall@8 by +1 to +2 pts, but neither won the bootstrap test. 750-char chunks raised recall but cut precision@8 by about a quarter (guardrail). 300-char chunks lost 11 pts of recall. |
| Which contract | Route by title: the share of the title's IDF weight found in each contract | Search all contracts; search the top-2 contracts | Every benchmark query names its contract ("Consider the …;"). Routing is 140/140 correct on train and doubled Recall@8 on its own. A low title match is also a strong "not in the corpus" signal. |
| Query text | Question only (title removed) after routing | Full query | Party names occur mostly on the cover page, so with the full query the top hit was the cover page for 119/140 train queries. |
| Retriever | Dense, `BAAI/bge-small-en-v1.5` (33M params) | BM25; hybrid BM25+dense with reciprocal rank fusion | Dense gave the largest single gain (+25 pts Recall@8). Hybrid was worse than dense in both tries: inside a single contract, BM25's keyword hits add noise. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20 | No reranker; reranking the top 40 | +7 pts Recall@8. Reranking 40 was worse (−3 pts): the web-search reranker promotes topical but wrong passages when given more of them. |
| Abstention | Two thresholds fitted on train: title match < 0.82, or best reranker score < −7.74 | Never abstain; a single threshold | Caught 13/25 train unanswerable queries at 2.9% false abstention. Thresholds sit at midpoints between observed scores (see the bug note under Iteration history). |
| Interfaces | Streamlit app with a contract picker, and a CLI | — | A real user picks a contract rather than typing "Consider the …". Both interfaces call the same `Pipeline.search`. |

Not implemented: cross-reference and definition resolution (for example "as defined in
Section 4.2"), a legal-domain reranker, and clause-boundary expansion. These are in the next steps.

## LLM use

No LLM is used anywhere in the pipeline or the loop. The two models are small local encoders: a
33M-parameter embedder and a 22M-parameter cross-encoder. They run on a laptop CPU, they need no
API keys, and no contract text leaves the machine. That matters if this were run on confidential
JioStar agreements, because a hosted LLM would send every queried contract passage to a third
party. The brief centres on retrieval, so I spent the time on a measurable non-LLM pipeline
first. The clearest place an LLM could pay off is the remaining weakness described below:
deciding whether a passage really is the requested clause, which is the missing-clause
abstention problem. That would be an optional stage. If added, I would prefer a local model for
the privacy reason above, and I would cache its outputs so evaluation stays reproducible.

## Evaluation setup

- **Data.** The 194 CUAD queries of LegalBench-RAG-mini, selected with the benchmark's `mini()`.
  They cover 29 contracts (1,211,773 characters, matching the paper's Table 3). All 272 gold spans
  were verified so that `document[start:end] == answer`.
- **Split.** The brief's `assign_split()` puts **140 train / 54 test** benchmark queries. The
  unanswerable set splits **25 train / 10 test**: 12 out-of-corpus (6/6), 16 missing-clause
  (14/2) and 7 off-topic (5/2), given as train/test. The set was committed (`3b0583b`) before the
  improvement loop existed and has not been edited since.
- **Retrieval metrics.** Character-level precision and recall at k ∈ {1, 2, 4, 8, 16, 32, 64}, as in
  the authors' `run_benchmark.py`: characters from the wrong document count against precision.
  These are computed on the ranked list regardless of abstention. Doc-hit@k (the right contract
  appears in the top k) is added for diagnosis. 95% CIs come from 1,000 bootstrap resamples of
  queries.
- **Abstention.** *Correct abstention* is the share of unanswerable queries that return nothing.
  *False abstention* is the share of benchmark queries that return nothing.
- **Primary metric: Recall@8.** A reviewer will read about eight passages, and in legal review a
  missed clause costs more than an extra paragraph to read. **Guardrails:** precision@8 must stay
  at or above 90% of the current config's value, which blocks the "just return longer passages"
  shortcut; false abstention must stay at or below 5%; correct abstentions may not drop by more
  than one query.
- **Note on precision.** Gold spans are short (median ~300 chars) compared with 8 × 500 = 4,000
  returned characters, so precision@8 is capped at about 10% even for a perfect ranker. Precision
  is therefore a guardrail, not the target.
- **Use of test.** Test was run exactly twice (`experiments/test_runs.log`): baseline and final,
  back to back, after the final config was committed. No changes were made afterwards.

## Results

Rows show precision / recall (%). Final = `configs/final.yaml`. n is the number of benchmark
queries.

| k | Baseline train (n=140) | Final train (n=140) | Baseline test (n=54) | **Final test (n=54)** |
|---|---|---|---|---|
| 1 | 0.00 / 0.00 | 24.24 / 33.95 | 0.00 / 0.00 | **24.31 / 32.93** |
| 2 | 0.46 / 0.59 | 17.82 / 46.03 | 0.28 / 0.57 | **16.86 / 44.07** |
| 4 | 0.83 / 3.01 | 11.89 / 58.95 | 0.55 / 6.12 | **11.64 / 53.13** |
| 8 | 1.06 / 8.87 | 7.68 / 69.61 | 1.49 / 12.32 | **7.44 / 64.07** |
| 16 | 1.15 / 16.75 | 4.71 / 78.34 | 1.25 / 18.54 | **4.06 / 71.52** |
| 32 | 0.96 / 27.97 | 3.00 / 87.00 | 0.87 / 24.19 | **2.54 / 82.40** |
| 64 | 0.66 / 36.57 | 2.14 / 96.17 | 0.69 / 36.43 | **1.72 / 93.39** |

| Abstention | Baseline train | Final train | Baseline test | **Final test** |
|---|---|---|---|---|
| Correct, all unanswerable | 0/25 | 13/25 (52%) | 0/10 | **8/10 (80%)** |
| – out of corpus | 0/6 | 6/6 | 0/6 | **6/6** |
| – missing clause | 0/14 | 4/14 | 0/2 | **2/2** |
| – off topic | 0/5 | 3/5 | 0/2 | **0/2** |
| False (benchmark queries) | 0/140 | 4/140 (2.9%) | 0/54 | **1/54 (1.9%)** |

| Other | Final train | Final test |
|---|---|---|
| Recall@8, 95% CI | 63.0–76.1 | 52.6–74.5 |
| Recall@1, 95% CI | 27.4–41.0 | 21.9–43.0 |
| Doc-hit@1 | 100% | 100% |
| Latency per query (mean / p95) | — (cached reranker scores) | 500 / 684 ms on CPU |

**Train vs test gap.** Recall@8 drops 5.5 points from train to test (69.6 → 64.1) and
Recall@64 drops 2.8 (96.2 → 93.4). Recall@1 and precision barely move. The confidence intervals
overlap heavily, so most of the gap could be sampling noise at n=54. Some optimism on train is
expected anyway: the loop picked the best of 16 candidates on train, so the train figure carries
selection bias. Every accepted change was a standard component (routing, a stock embedder, a
stock reranker), not a tuned number, which limits how far this could overfit. Abstention is
different: its two thresholds were *fitted* on the 25 train unanswerable queries. The test
abstention rate (8/10) looks better than train (13/25), but that is not evidence of improvement.
Only 2 missing-clause queries landed in test. Missing-clause is the hard category, where train
shows 4/14, so the realistic expectation is about 30% on that type.

**Context from the paper.** On CUAD-mini, the paper's naive method (500-char chunks,
`text-embedding-3-large`, searching all contracts) reports R@8 = 31.7% and R@64 = 74.7%. Our
final system is far higher, but the comparison is not like-for-like. Most of our gain comes from
routing on the contract title in the query, which the paper's pipeline does not exploit.

## Iteration history

**How the loop works** (`python -m src.improve`, train only):

1. Evaluate the current config.
2. Diagnose every benchmark query at k=8 into *wrong document*, *wrong section* (right contract,
   recall 0), *partial span*, *false abstention* or *ok*. Count unanswerable queries answered
   anyway as *missed abstention*.
3. Propose the applicable candidate, from a fixed list of 13, whose target buckets hold the
   most failures. Each candidate carries a written rationale.
4. Evaluate it, re-fitting abstention thresholds if calibration is on.
5. Accept only if the guardrails hold **and** either Recall@8 is higher in ≥95% of 1,000 paired
   bootstrap resamples of the train queries, or Recall@8 is unchanged and more unanswerable
   queries are caught. The bootstrap guards against accepting noise at n=140.
6. Log everything to `experiments/log.jsonl`.

A rejected candidate may be retried once after a later change is accepted, because a change can
fail in one context and help in another. Gold spans are used only for scoring, never at
inference. All proposals are rule-based (the `proposer` field); no LLM proposer was used.

**Official run `20261029-221942`** (6 accepted, 10 rejected):

| # | Change | Proposed for | R@8 | P@8 | R@1 | Wins in bootstrap | Decision |
|---|---|---|---|---|---|---|---|
| 0 | baseline (BM25, 500 fixed) | — | 8.9 | 1.06 | 0.0 | — | start |
| 1 | strip title from query | wrong section | 10.3 | 1.25 | 1.1 | 72% | rejected: not reliable |
| 2 | route to contract | wrong doc + section | 17.2 | 2.05 | 0.0 | 100% | **accepted** |
| 3 | strip title (retry) | wrong section | 37.9 | 4.45 | 12.3 | 100% | **accepted** |
| 4 | dense retrieval | wrong section | 62.7 | 6.98 | 26.4 | 100% | **accepted** |
| 5 | sentence chunking | partial span | 63.9 | 8.54 | 25.7 | 64% | rejected: not reliable |
| 6 | 20% chunk overlap | partial span | 65.0 | 7.86 | 29.8 | 82% | rejected: not reliable |
| 7 | larger chunks (750) | partial span | 67.1 | 5.24 | 26.5 | 92% | rejected: precision guardrail |
| 8 | hybrid BM25+dense | wrong section | 58.1 | 6.57 | 25.4 | 4% | rejected: worse |
| 9 | cross-encoder rerank | wrong section | 69.6 | 7.68 | 34.0 | 100% | **accepted** |
| 10 | sentence chunking (retry) | partial span | 68.5 | 8.90 | 36.7 | 33% | rejected |
| 11 | chunk overlap (retry) | partial span | 69.3 | 8.15 | 33.4 | 47% | rejected |
| 12 | larger chunks (retry) | partial span | 74.3 | 5.65 | 35.0 | 97% | rejected: precision guardrail |
| 13 | calibrated abstention | missed abstention | 69.6 | 7.68 | 34.0 | — | **accepted**: 0 → 13 caught |
| 14 | hybrid (retry) | wrong section | 66.1 | 7.23 | 32.8 | 1% | rejected: worse |
| 15 | smaller chunks (300) | wrong section | 58.0 | 9.97 | 24.8 | 0% | rejected: worse |
| 16 | rerank top 40 | wrong section | 66.6 | 7.31 | 32.5 | 1% | rejected: worse |

What this shows:

- **Order matters.** Stripping the title was useless across all contracts but worth +21 points
  once routing was on.
- **The precision guardrail did its job.** Twice it blocked larger chunks, which raise recall
  only by returning more text.
- **Partial span is the largest remaining bucket.** It covers 43 of 140 queries, and no
  chunk-level change fixed it reliably.

**Earlier runs, kept in the log.** Run `20261029-213706` was a 2-iteration dry run that tested the
loop mechanics. Run `20261029-213728` made the same 16 decisions, but it was superseded by a bug
that I found while checking that a standalone evaluation reproduced the loop's numbers:

- The router summed IDF weights over a Python `set`, whose iteration order depends on the
  per-process hash seed. One contract's title match therefore differed in the 16th digit
  between processes.
- Calibration had placed the threshold exactly on that value, so 5 real queries flipped to
  "no answer" (false abstention 2.9% in the loop, 6.4% standalone).

I fixed both causes: the router now sums in sorted order, and thresholds now sit at midpoints
between observed scores. I then re-ran the loop before touching test.

## Error analysis

These are concrete failures of the final config, drawn from train (tr) and test (te) and grouped
by cause. Counts are at k=8 across both splits: 31 queries with recall 0, 60 partial, and 5
false abstentions.

**1. The clause doesn't use the category's words (vocabulary gap).** CUAD questions name an
abstract category. The clause states the concrete term.
- *tr: Florida Chemical / Flotek supply, "volume restrictions?"* The gold clause is "'Maximum
  Quantity' means [***] pounds of Terpene Product per Year". The best reranker score was
  −8.12, so the system **abstained** on a question that has an answer.
- *te: Papa John's endorsement, "volume restrictions?"* The gold is "At least eight (8) 'Service
  Days'… Up to four (4) production days". Same pattern, also a **false abstention**.
- *tr: Gas Franchise (Vinton), "revenue or profit-sharing?"* The gold is "Franchise Fee shall be
  a percentage share of… gas sales". The top passage was the cover page, and the system
  **abstained**.
- *tr: GE Capital / Duckwall-Alco agency, "audit rights?"* The gold is "books and records shall be
  open for inspection and examination by Lessor". None of the top 8 contained it.
- *te: ISA / Rubicon alliance, "covenant not to sue?"* The gold is "Rubicon shall not contest
  ownership by ISA". The reranker preferred an "independent contractors" clause.

*Diagnosis:* the embedder and reranker match surface topic. Category names like "volume
restriction", "covenant not to sue" and "audit rights" need a glossary of the concrete forms
they take: maximum quantities, service-day caps, "shall not contest", "open for inspection".

**2. Long contracts with many look-alike clauses.**
- *te: Depomed / King promotion (173k chars).* Five of the 28 imperfect test queries come from this one
  contract. They include "non-compete", where the gold is "King shall not promote… any product
  containing metformin", and "covenant not to sue", where the top hit is a clause that literally
  contains the words "covenant not to sue" but is about a different grant.
- *tr: Tate & Lyle / Igene JV.* The same generic passage ("nothing contained in this Agreement will
  require or obligate either Party…") ranks first for three different questions: MFN, assignment
  and minimum commitment.

*Diagnosis:* in a 100k+ character contract, many 500-char windows share legal vocabulary. A
general-domain cross-encoder cannot tell "mentions X" from "is the X clause". Generic
boilerplate becomes a hub that ranks high for many questions.

**3. The answer is long or split across sections.**
- *te: Depomed / King, "minimum commitment?"* The gold is 6 snippets totalling 2,275 chars
  ("at least [***] PDEs per calendar year…"), so Recall@8 is 0.12. Even a perfect 8-passage
  answer would struggle to cover six separate sections.
- *tr: Premier Biomedical IP agreement, "licenses granted?"* The gold is 3 snippets, 1,891 chars,
  and recall at 8 is 0.

*Diagnosis:* the metric rewards covering every gold character. A ranker that finds one of the
six pieces scores low. Clause-level expansion, which returns the whole numbered section, would
help.

**4. Chunk boundaries and the cover page.**
- *tr: Excite / Vitamin Shoppe sponsorship, "governing law?"* The gold span (58,417–58,720)
  starts near the end of chunk 58,000–58,500. The top hit is that chunk, so recall is only 0.27
  even though the right clause was found.
- *tr: Glamis / Western Copper, "right of first offer?"* The top hit is the cover page, whose title
  is literally "NON COMPETITION AGREEMENT AND RIGHT OF FIRST OFFER". The operative clause
  ranks lower.

*Diagnosis:* fixed windows cut clauses. Overlap and sentence chunking were tried and helped
slightly (+1 to +2 pts), but not reliably. Expanding a hit to its section boundaries is the
better next step.

**5. Abstention misses on topical neighbours.**
- *tr: Cano Petroleum "Honey Hole", "can the Show be streamed online?" (missing clause).* The top
  passage is "The Company shall produce no less than forty (40) original episodes of the Show"
  at score −1.27, comfortably above the threshold. It is on topic, but it is not an answer.
- *tr: "What is the capital of Japan?" (off topic).* This retrieved "the symbols 'JPY' or '¥' shall
  refer to the lawful currency of Japan". This adversarial case was planted in the
  unanswerable set and fooled the system as predicted.
- *te: "Recommended dosage of the COVID-19 treatment under the Premier Biomedical agreement?"*
  This scored 0.99, higher than many true answers, because the contract does discuss the
  COVID-19 treatment.
- *te: "Glu Mobile's total revenue in 2007?"* This returned the Glu signature block.

*Diagnosis:* the reranker was trained on web search (MS MARCO) and scores topical relevance, not
"this passage answers the question". Questions without a contract title skip routing entirely,
so its strong out-of-corpus signal is lost. That is why off-topic questions mentioning corpus
entities get through.

**6. Questionable gold labels** (not excuses, but worth knowing when reading the numbers):
- *tr: deltathree / RSL "change of control?"* The gold span is the term clause ("shall continue
  for a term of three (3) years…").
- *te: MJ Syndicated JV "anti-assignment?"* The gold is "Title to all Joint Venture Property will
  remain in the name of the Joint Venture".

Neither reads as an answer to its question. CUAD annotations follow a category guide, and a few
labels are surprising.

## Limitations and next steps

**What doesn't work yet:**
- **Missing-clause detection is weak.** On train it catches 4 of 14. On test it caught 2 of 2,
  but that is too few to count as evidence. A legal user asking "is there an insurance clause?"
  can get on-topic but non-answering passages instead of "no".
- **Answers are often only partly covered**, which is 60 of the 194 queries at k=8.
- **No cross-references or definitions.** "As defined in Section 4.2" is not followed, and
  defined terms are not expanded.
- **Routing assumes the user names the contract.** Generic questions with no contract search the
  whole corpus with no "not in the corpus" signal. The app mitigates this with a contract picker.
- **Small evaluation sets.** 54 test queries give a ±11-point CI on Recall@8. The 10 test
  unanswerable queries are too few to estimate abstention reliably.
- **Speed.** About 0.5 s per query on a laptop CPU, and indexing takes about 5 minutes per
  chunking configuration. That is fine for 29 contracts. Thousands of contracts would need a
  vector index (for example FAISS) and a GPU or a batch indexing job.

**Next steps, in order of expected value:**
1. **Clause expansion.** Grow each hit to its numbered section. This targets the partial-span and
   boundary failures directly, and it is cheap to try as a new loop candidate.
2. **A category glossary or query rewriting.** Expand "volume restriction" into its concrete
   forms. This could be a hand-written glossary for CUAD's 35 categories, or an LLM rewrite
   with cached outputs. It targets the vocabulary-gap failures, including the false abstentions.
3. **An answer-verification step for abstention.** Use an NLI model or a local LLM that judges
   "does this passage contain the requested clause?". Calibrate it on a larger unanswerable set,
   at least 100 queries, so the test estimate means something.
4. **A legal-domain reranker.** Fine-tune on the CUAD contracts that are not in the mini set, so
   there is no leakage into evaluation.
5. **Before legal users rely on it:** keep a human in the loop. Always show the full section
   and its offsets so a lawyer can verify the source. Report "not found" as "not found in these
   passages", never "the contract has no such clause". Re-evaluate on JioStar's own agreement
   types (licensing windows, territories, sublicensing), which CUAD under-represents. Keep all
   models local for confidentiality.
