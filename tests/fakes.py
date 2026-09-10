"""In-memory doubles for the retrieval Protocols.

The embedder and vector store are Protocols, so retrieval logic can be tested
against small in-memory doubles. Nothing here imports torch or chromadb, which
is what keeps the suite runnable on every push.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


class FakeEmbedder:
    """Records what it was asked to embed; returns a fixed vector.

    Single and batch encoding are recorded separately so a test can assert
    that indexing takes the batch path.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.batch_calls = 0

    def encode(self, text: str) -> Sequence[float]:
        self.seen.append(text)
        return [0.0, 1.0, 0.0]

    def encode_batch(self, texts: Sequence[str]) -> list[Sequence[float]]:
        self.batch_calls += 1
        return [[0.0, 1.0, 0.0] for _ in texts]


class FakeStore:
    """An in-memory vector store returning a scripted result list."""

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents or []
        self.queries: list[dict[str, Any]] = []
        self.get_calls: list[list[str]] = []

    def query(
        self,
        embedding: Sequence[float],
        n_results: int,
        content_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append({"n_results": n_results, "content_types": content_types})
        matches = [
            doc
            for doc in self.documents
            if content_types is None or doc.get("metadata", {}).get("content_type") in content_types
        ]
        return matches[:n_results]

    def get(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        self.get_calls.append(list(chunk_ids))
        wanted = set(chunk_ids)
        return {doc["chunk_id"]: doc for doc in self.documents if doc["chunk_id"] in wanted}


class FakeLLM:
    """Returns a fixed completion and records the prompts it was given."""

    name = "fake-llm"

    def __init__(self, response: str = "an answer", reachable: bool = True) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.stream_calls = 0
        # Configured-but-unreachable is a distinct state from not configured,
        # and the one a health check exists to catch.
        self.reachable = reachable

    def available(self, timeout: float = 2.0) -> bool:
        return self.reachable

    def complete(self, prompt: str, *, timeout: float = 120.0) -> str:
        self.prompts.append(prompt)
        return self.response

    def stream(self, prompt: str, *, timeout: float = 120.0):
        self.prompts.append(prompt)
        self.stream_calls += 1
        # Tokens keep their trailing whitespace, so joining them reproduces the
        # response exactly -- a real provider's tokens do the same, and a fake
        # that drops it would hide off-by-one spacing bugs in the caller.
        yield from re.findall(r"\S+\s*", self.response) or [self.response]


class FakeReranker:
    """Scores texts from a lookup table; unknown texts score 0.

    Records call count and how many texts it saw, so a test can assert that
    the whole candidate pool was scored in a single batch rather than the
    final top_k one at a time.
    """

    name = "fake-reranker"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []
        self.calls = 0
        self.n_scored = 0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        self.queries.append(query)
        self.n_scored += len(texts)
        return [self.scores.get(text, 0.0) for text in texts]


class RecordingStore:
    """A write-only vector store that keeps what was upserted.

    Upsert semantics are modelled faithfully -- writing an id that already
    exists replaces it -- so a test can distinguish overwriting from
    duplicating when the same corpus is indexed twice.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[list[str]] = []

    @property
    def upserted(self) -> list[dict[str, Any]]:
        return list(self.rows.values())

    def upsert(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, str]],
    ) -> None:
        self.upsert_calls.append(list(ids))
        for chunk_id, text, embedding, metadata in zip(
            ids, texts, embeddings, metadatas, strict=True
        ):
            self.rows[chunk_id] = {
                "id": chunk_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(
        self,
        embedding: Sequence[float],
        n_results: int,
        content_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def get(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return {cid: self.rows[cid] for cid in chunk_ids if cid in self.rows}


def make_doc(
    chunk_id: str,
    text: str = "",
    content_type: str = "route",
    source_url: str = "",
    score: float = 0.9,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text or f"passage {chunk_id}",
        "score": score,
        "metadata": {"content_type": content_type, "source_url": source_url},
    }
