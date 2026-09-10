# CruxBot

Hybrid-retrieval RAG over 338,433 rock climbing documents — routes, forum
threads, accident reports, and gear reviews — with a retrieval evaluation
harness that measures the retriever separately from the generator.

> **Status: in progress.** Two-stage retrieval, the indexing pipeline, the test
> suite (253 tests), and CI are in place. The labelled retrieval benchmark and
> the hosted demo are not yet built. This README documents what exists; sections
> marked *(planned)* do not.

---

## Why this repo exists

Most RAG demos cannot tell you whether their retrieval works. They report
end-to-end answer quality, which conflates the retriever, the prompt, and the
model — so "we added hybrid search and it got better" is unfalsifiable.

This project takes the opposite position:

- **The retriever is measured on its own** — Recall@k, MRR, nDCG@k against a
  labelled query set, with each ablation changing exactly one variable. *(planned)*
- **Metrics are named for what they measure.** The predecessor project reported
  a "90% pass rate" that, read closely, only checked whether an answer avoided a
  refusal phrase. It measured false-refusal rate, which is a real and useful
  number — but it is not accuracy, and it is not labelled as accuracy here.
- **Known failure modes are pinned as tests**, not omitted. See
  `TestKnownLimitations` in [tests/test_intent.py](tests/test_intent.py).

---

## Architecture

```
Query
  │
  ├─► intent.detect()          rule-based content-type prior
  ├─► grades.expand_query()    YDS ↔ French ↔ V ↔ Font normalization
  │
  ▼
Stage 1 — recall over 384k chunks
  ├─ dense   (bge-small-en-v1.5 + ChromaDB)  ─┐
  └─ sparse  (BM25 inverted index)           ─┴─► rrf_merge() ─► ~50 candidates
                                                                      │
                                                                      ▼
Stage 2 — precision over ~50            cross-encoder rerank ─► top 5
                                                                      │
                                                                      ▼
  prompts.build()  ─►  LLMProvider  ─►  answer + citations
                       (Ollama / hosted API)
```

### Design notes

**Why the core has no dependencies.** `grades`, `intent`, `fusion`, `urls`,
`prompts`, and the BM25 index import nothing outside the standard library. The
embedding model and vector store are injected behind Protocols, so hybrid
retrieval is tested against in-memory doubles. The whole suite runs in under a
second without installing `torch` — which is why CI can run on every push.

**Why a hand-written BM25 instead of `rank_bm25`.** `rank_bm25.get_scores`
scores every document in the corpus on every query: at 382k documents that is
382k operations per query term, whether or not the term appears anywhere. An
inverted index makes the cost proportional to the postings actually touched,
which for a typical term is well under 1% of the corpus. It also removes a
dependency and keeps the module testable in CI.

**Why the sparse index stores postings as packed arrays.** It holds term
statistics only — text for a chunk found by BM25 alone is hydrated from the
vector store in one batched lookup after fusion. That alone was not enough:
the first build, with no text stored, still came to 681 MB, *larger* than the
predecessor's 633 MB cache. The cost was the representation. 26.5M postings
held as small Python objects pickle to roughly 27 bytes each; the same data as
two parallel `array("i")` per term is 8, and unpickles as raw bytes rather than
26.5M constructor calls.

| | objects | packed arrays |
| --- | --- | --- |
| on disk | 681 MB | **277 MB** |
| load | 6.1 s | **0.35 s** |
| build | 113 s | **21 s** |
| filtered search | 159 ms | **38 ms** |

The search figure is a separate fix: content-type positions are now precomputed
at build time. Deriving them per query meant scanning all 384k documents on
every filtered search, which made filtering *cost* rather than save time.

**A measured limit of ChromaDB.** Metadata filtering does not use the HNSW
index — an unfiltered query returns in 2 ms, the same query with a
`content_type` filter takes 390 ms. This is now the dominant term in retrieval
latency, and it is the concrete form of the ceiling described above rather than
a hypothetical one.

**Why ChromaDB.** 382k vectors × 384 dims fits comfortably in memory on a
single node, and Chroma needs no separate service to operate. This is a
deliberate ceiling: at roughly 10M vectors, or once multi-replica reads matter,
the right move is pgvector or Qdrant. Chroma is not being defended as a
production vector store — it is being used where its limits do not bind.

**Why a small embedding model.** Under a two-stage architecture the first stage
is judged on Recall@50 — did the right chunk reach the candidate pool at all —
not on ranking it first, which is the cross-encoder's job. That weakens the case
for a larger bi-encoder considerably. Measured on an M4, `bge-small` indexes
384k chunks at 238/s against `bge-base`'s 91/s: 27 minutes versus 67, and a
smaller index. Whether the reranker actually absorbs the quality difference is a
question for the benchmark, not for taste — the comparison is one row of the
table below.

**Why an LLM provider Protocol.** A self-hosted Llama 3 is right for privacy
and zero marginal query cost; a hosted API is right for a public demo that must
not require a GPU. Both sit behind [`LLMProvider`](src/cruxbot/llm/base.py) so
evaluation can hold the retriever fixed and swap only the generator.

**What incremental indexing requires.** Chunk ids are derived from document
ids, so re-indexing a changed document overwrites its chunks instead of
appending a second copy. Full and incremental runs are therefore the same
command over a different input file. Only the dense index is incremental; BM25's
statistics are corpus-global, so it is rebuilt — minutes at this size.

---

## Layout

```
src/cruxbot/
├── grades.py           # grade normalization       (pure, no deps)
├── intent.py           # query intent detection    (pure, no deps)
├── fusion.py           # reciprocal rank fusion    (pure, no deps)
├── urls.py             # citation quality checks   (pure, no deps)
├── prompts.py          # prompt construction       (pure, no deps)
├── types.py            # Chunk / Source / Answer
├── config.py           # env-driven settings
├── llm/
│   ├── base.py         # LLMProvider Protocol
│   └── ollama.py       # local backend
├── indexing/
│   ├── chunking.py     # per-content-type splitting (pure, no deps)
│   ├── corpus.py       # streaming JSON / JSONL loader
│   └── pipeline.py     # chunk -> embed -> upsert, full or incremental
└── retrieval/
    ├── sparse.py       # BM25 inverted index       (pure, no deps)
    ├── dense.py        # Embedder / VectorStore Protocols + Chroma backend
    ├── rerank.py       # cross-encoder second stage
    └── hybrid.py       # intent routing, fusion, hydration, rerank
scripts/build_index.py  # CLI: full and incremental index builds
tests/                  # 253 tests; no torch, no network
```

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
make install        # core + pytest + ruff, no torch
make check          # lint + tests
```

The full stack (embeddings, vector store, serving) installs separately:

```bash
make install-all
```

Building the index:

```bash
# full build: 338,433 documents -> 384,315 chunks
python scripts/build_index.py --corpus data/unified.jsonl

# incremental: same command, a smaller input, no BM25 rebuild
python scripts/build_index.py --corpus data/incoming/reddit-latest.jsonl --skip-sparse
```

Tests that need a built index or a running model are marked `integration` and
excluded by default.

---

## Evaluation *(planned)*

The harness will report, per retrieval configuration:

| Configuration                    | Recall@50 | Recall@10 | nDCG@10 | p50 latency |
| -------------------------------- | --------- | --------- | ------- | ----------- |
| dense only (bge-small)           |           |           |         |             |
| BM25 only                        |           |           |         |             |
| hybrid (RRF)                     |           |           |         |             |
| hybrid + cross-encoder rerank    |           |           |         |             |
| dense only (bge-base)            |           |           |         |             |

Recall@50 is reported separately because it measures the first stage on its own
terms: whether the right chunk reached the pool the reranker sees.

Answer quality is scored separately by an LLM judge on relevance, groundedness,
citation quality, and completeness — reported alongside, never merged into, the
retrieval numbers.

---

## Acknowledgements

This project began as **CS 6120 (Natural Language Processing), Northeastern
University**, built by a team of four:
[zongyang078/CruxBot](https://github.com/zongyang078/CruxBot).

In the original repository:

- **Sherwin Vahidimowlavi** wrote the chunking and embedding pipelines and built
  the initial 382k-vector ChromaDB index.
- **Lingyun Xiao** wrote the FastAPI backend, the Docker and GCP deployment, and
  the LLM-as-Judge evaluation framework.
- **Linxuan Li** wrote the initial dense-retrieval logic and RAG orchestration.
- **Zongyang Li** (this repository's author) wrote the data collection and
  cleaning pipeline for all six sources, the hybrid BM25 + dense + RRF
  retrieval, query intent detection, grade normalization, the anti-hallucination
  prompt design, and the 50-query evaluation suite.

This repository is a solo rebuild. It carries forward the collected dataset and
the retrieval logic, and replaces the chunking, embedding, evaluation, and
serving layers with new implementations. Where an idea originated in a
teammate's work, it is credited above; where their code has been replaced, the
replacement is mine.

---

## License

MIT — see [LICENSE](LICENSE).

Source data is used under the terms of each provider: OpenBeta (CC0), American
Alpine Club (CC0), Kaggle datasets (per-author copyright), Reddit (API ToS).
