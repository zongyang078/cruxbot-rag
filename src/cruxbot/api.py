"""HTTP service over the retrieval pipeline.

Two endpoints, split along the line that matters operationally:

  POST /search   Retrieval only. No model tokens, no provider, no per-query
                 spend, ~1.2 s. This is most of what the system does, and it
                 can be served to anyone.
  POST /answer   Retrieval plus generation, streamed. Costs money per call and
                 depends on a provider being configured, so it degrades to a
                 clear 503 rather than a stack trace when one is not.

Both return the per-stage timing breakdown. A caller that only ever sees a
total cannot tell a slow retriever from a slow generator, and the two are
fixed by different work.

The index and models load once at startup, not per request: a bi-encoder, a
cross-encoder, and a 384k-vector store take tens of seconds to bring up, which
is a startup cost, not a request cost.

Requires the serving extra:  pip install -e ".[rag,serve]"
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cruxbot.config import settings
from cruxbot.llm.base import ProviderError
from cruxbot.pipeline import Pipeline, Retrieved

logger = logging.getLogger(__name__)

# Populated at startup. A module-level handle rather than a factory per request
# because the models behind it are the expensive part.
_pipeline: Pipeline | None = None
# Why generation is unavailable, when it is. Reported by /health so that a
# misconfiguration is distinguishable from a deliberate retrieval-only setup.
_provider_error: str | None = None


def build_pipeline() -> Pipeline:
    """Load the index, models, and (if configured) an LLM provider."""
    from cruxbot.llm.base import get_provider
    from cruxbot.retrieval.dense import ChromaStore, DenseRetriever, SentenceTransformerEmbedder
    from cruxbot.retrieval.hybrid import HybridRetriever
    from cruxbot.retrieval.rerank import CrossEncoderReranker
    from cruxbot.retrieval.sparse import BM25Index

    embedder = SentenceTransformerEmbedder()
    store = ChromaStore()
    bm25 = BM25Index.load(settings.bm25_cache)
    reranker = CrossEncoderReranker()

    provider = None
    if os.getenv("CRUXBOT_DISABLE_LLM", "").lower() not in {"1", "true", "yes"}:
        try:
            provider = get_provider()
        except Exception as exc:  # noqa: BLE001 - retrieval must still serve
            # A missing or misconfigured provider is not a startup failure --
            # the search endpoint has to come up regardless. But it must be
            # said out loud: swallowing this silently meant a container built
            # without the anthropic package reported "no provider configured",
            # which is indistinguishable from not configuring one.
            global _provider_error
            _provider_error = f"{type(exc).__name__}: {exc}"
            logger.warning("LLM provider unavailable, serving retrieval only: %s", _provider_error)

    return Pipeline(
        HybridRetriever(DenseRetriever(embedder, store), bm25, reranker=reranker),
        provider,
        top_k=settings.top_k,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = build_pipeline()
    yield
    _pipeline = None


app = FastAPI(
    title="CruxBot",
    description="Hybrid-retrieval RAG over 338k rock climbing documents.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pipeline() -> Pipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Index is still loading.")
    return _pipeline


# --- Schemas --------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceOut(BaseModel):
    chunk_id: str
    text: str
    score: float
    url: str = ""
    # False when the URL is a site-level link rather than a document. Surfaced
    # rather than hidden: some sources in this corpus have no per-post URL, and
    # presenting those as citations would be misleading.
    url_is_specific: bool = True
    content_type: str = ""
    grade: str = ""
    location: str = ""


class Timing(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float = 0.0
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    sources: list[SourceOut]
    timing: Timing
    candidates: dict[str, int]


class AnswerRequest(SearchRequest):
    stream: bool = True


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceOut]
    timing: Timing


def _sources(retrieved: Retrieved) -> list[SourceOut]:
    return [
        SourceOut(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=round(chunk.score, 4),
            url=chunk.source_url,
            url_is_specific=chunk.is_citable,
            content_type=chunk.content_type,
            grade=chunk.grade,
            location=chunk.location,
        )
        for chunk in retrieved.chunks
    ]


def _timing(retrieved: Retrieved, generation_ms: float = 0.0) -> Timing:
    return Timing(
        retrieval_ms=round(retrieved.retrieval_ms, 1),
        rerank_ms=round(retrieved.rerank_ms, 1),
        generation_ms=round(generation_ms, 1),
        total_ms=round(retrieved.retrieval_ms + retrieved.rerank_ms + generation_ms, 1),
    )


# --- Endpoints ------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness, with generation probed rather than assumed.

    The provider is contacted, not merely checked for existence: a configured
    but unreachable backend is exactly the state a health check exists to
    catch, and reporting it as healthy defeats the purpose.
    """
    ready = _pipeline is not None
    provider = _pipeline.provider if ready else None
    body: dict[str, Any] = {
        "status": "ok" if ready else "loading",
        "retrieval": ready,
        "generation": bool(provider and provider.available()),
        "provider": provider.name if provider else None,
    }
    if not body["generation"] and _provider_error:
        body["generation_error"] = _provider_error
    return body


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Retrieve passages. No model tokens are spent."""
    retrieved = get_pipeline().search(request.query, top_k=request.top_k)
    return SearchResponse(
        query=retrieved.query,
        sources=_sources(retrieved),
        timing=_timing(retrieved),
        candidates={
            "dense": retrieved.n_dense,
            "sparse": retrieved.n_sparse,
            "reranked": retrieved.n_reranked,
        },
    )


@app.post("/answer")
def answer(request: AnswerRequest):
    """Retrieve and generate an answer, streamed by default.

    Streaming is the default because generation dominates end-to-end latency;
    holding the whole answer back until it is finished wastes the seconds the
    user could have spent reading it.
    """
    pipeline = get_pipeline()
    if pipeline.provider is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider is configured. /search still works.",
        )

    # One retrieval either way. `answer_stream` hands back the passages before
    # generation begins, so the non-streaming branch drains the same generator
    # rather than calling `answer()` and retrieving a second time.
    retrieved, tokens = pipeline.answer_stream(request.query, top_k=request.top_k)

    if not request.stream:
        started = time.perf_counter()
        try:
            text = "".join(tokens)
        except ProviderError as exc:
            # Configured but unreachable. A 500 here would read as a bug in
            # this service rather than a missing dependency.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        generation_ms = (time.perf_counter() - started) * 1000
        return AnswerResponse(
            query=request.query,
            answer=text.strip(),
            sources=_sources(retrieved),
            timing=_timing(retrieved, generation_ms),
        )

    def events():
        # Sources first, so a client can render citations while the answer is
        # still being written.
        yield _sse("sources", {"sources": [s.model_dump() for s in _sources(retrieved)]})
        yield _sse("timing", _timing(retrieved).model_dump())
        try:
            for token in tokens:
                yield _sse("token", {"text": token})
        except ProviderError as exc:
            # The status line is already sent, so this cannot become a 503;
            # the client learns about it through the event stream instead.
            yield _sse("error", {"detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the client needs to hear about it
            yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
