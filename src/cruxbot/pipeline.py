"""End-to-end question answering: retrieve, build a prompt, generate.

The pipeline owns the orchestration and nothing else. Retrieval, prompt
construction, and generation each live behind their own seam, so the evaluation
harness can hold two of them fixed and vary the third -- which is the whole
reason the retrieval numbers in the benchmark mean anything.

Timing is broken down per stage rather than reported as one figure. A 1.2
second retrieval and a 15 second generation are the same total as the reverse,
and they call for entirely different work.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cruxbot import prompts
from cruxbot.types import Answer, Chunk, Source

if TYPE_CHECKING:
    from cruxbot.llm.base import LLMProvider
    from cruxbot.retrieval.hybrid import HybridRetriever


@dataclass(slots=True)
class Retrieved:
    """Retrieval output, before any language model has seen it.

    Returned on its own by `search`, which is the endpoint that costs nothing
    to serve: no tokens, no provider, no per-query spend. Most of what this
    system does well is visible here.
    """

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    n_dense: int = 0
    n_sparse: int = 0
    n_reranked: int = 0

    @property
    def sources(self) -> list[Source]:
        return [Source.from_chunk(chunk) for chunk in self.chunks]


class Pipeline:
    """Answer questions from the climbing corpus."""

    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        # Optional: a deployment that only serves retrieval needs no provider,
        # and should not fail to start for want of one.
        self.provider = provider
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None, **kwargs: Any) -> Retrieved:
        """Retrieve passages without generating anything."""
        result = self.retriever.retrieve(query, top_k=top_k or self.top_k, **kwargs)
        return Retrieved(
            query=query,
            chunks=result.chunks,
            retrieval_ms=result.elapsed_ms - result.rerank_ms,
            rerank_ms=result.rerank_ms,
            n_dense=result.n_dense,
            n_sparse=result.n_sparse,
            n_reranked=result.n_reranked,
        )

    def _require_provider(self) -> LLMProvider:
        if self.provider is None:
            raise RuntimeError(
                "This pipeline was built without an LLM provider and can only serve "
                "retrieval. Pass one to Pipeline(...) to generate answers."
            )
        return self.provider

    def answer(self, query: str, top_k: int | None = None, **kwargs: Any) -> Answer:
        """Retrieve and generate a complete answer."""
        provider = self._require_provider()
        retrieved = self.search(query, top_k=top_k, **kwargs)

        started = time.perf_counter()
        text = provider.complete(prompts.build(query, retrieved.chunks))
        generation_ms = (time.perf_counter() - started) * 1000

        return Answer(
            query=query,
            text=text,
            sources=retrieved.sources,
            retrieval_ms=retrieved.retrieval_ms + retrieved.rerank_ms,
            generation_ms=generation_ms,
            latency_ms=retrieved.retrieval_ms + retrieved.rerank_ms + generation_ms,
        )

    def answer_stream(
        self, query: str, top_k: int | None = None, **kwargs: Any
    ) -> tuple[Retrieved, Iterator[str]]:
        """Retrieve, then return the sources and a lazy token stream.

        Sources come back before the first token so a caller can render
        citations while the answer is still being written -- and so a slow or
        failing generator cannot swallow retrieval results that are already in
        hand.
        """
        provider = self._require_provider()
        retrieved = self.search(query, top_k=top_k, **kwargs)
        return retrieved, provider.stream(prompts.build(query, retrieved.chunks))
