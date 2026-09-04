# CruxBot

Hybrid-retrieval RAG over 338,433 rock climbing documents — routes, forum
threads, accident reports, and gear reviews — with a retrieval evaluation
harness that measures the retriever separately from the generator.

> **Status: in progress.** The core retrieval logic, test suite, and CI are in
> place. The reranker, labelled retrieval benchmark, and hosted demo are not yet
> built. This README documents what exists; sections marked *(planned)* do not.

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
Hybrid retrieval
  ├─ dense   (bge-base-en-v1.5 + ChromaDB)     ─┐
  └─ sparse  (BM25)                            ─┴─► fusion.rrf_merge()
                                                        │
                                                        ▼
                                              cross-encoder rerank (planned)
                                                        │
                                                        ▼
  prompts.build()  ─►  LLMProvider  ─►  answer + citations
                       (Ollama / hosted API)
```

### Design notes

**Why the pure core has no dependencies.** `grades`, `intent`, `fusion`,
`urls`, and `prompts` import nothing outside the standard library. Retrieval
correctness, grade conversion, and prompt construction are therefore testable
in seconds without installing `torch` — which is why CI is fast enough to run
on every push.

**Why ChromaDB.** 382k vectors × 384 dims fits comfortably in memory on a
single node, and Chroma needs no separate service to operate. This is a
deliberate ceiling: at roughly 10M vectors, or once multi-replica reads matter,
the right move is pgvector or Qdrant. Chroma is not being defended as a
production vector store — it is being used where its limits do not bind.

**Why an LLM provider Protocol.** A self-hosted Llama 3 is right for privacy
and zero marginal query cost; a hosted API is right for a public demo that must
not require a GPU. Both sit behind [`LLMProvider`](src/cruxbot/llm/base.py) so
evaluation can hold the retriever fixed and swap only the generator.

---

## Layout

```
src/cruxbot/
├── grades.py       # grade normalization      (pure, no deps)
├── intent.py       # query intent detection   (pure, no deps)
├── fusion.py       # reciprocal rank fusion   (pure, no deps)
├── urls.py         # citation quality checks  (pure, no deps)
├── prompts.py      # prompt construction      (pure, no deps)
├── types.py        # Chunk / Source / Answer
├── config.py       # env-driven settings
├── llm/            # provider Protocol + Ollama backend
└── retrieval/      # dense, sparse, hybrid    (planned)
tests/              # unit tests for every pure module
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

Tests that need a built index or a running model are marked `integration` and
excluded by default.

---

## Evaluation *(planned)*

The harness will report, per retrieval configuration:

| Configuration          | Recall@10 | MRR | nDCG@10 | p50 latency |
| ---------------------- | --------- | --- | ------- | ----------- |
| dense only             |           |     |         |             |
| BM25 only              |           |     |         |             |
| hybrid (RRF)           |           |     |         |             |
| hybrid + cross-encoder |           |     |         |             |

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
