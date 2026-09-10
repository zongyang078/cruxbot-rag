"""In-memory doubles for the retrieval Protocols.

The embedder and vector store are Protocols, so retrieval logic can be tested
against small in-memory doubles. Nothing here imports torch or chromadb, which
is what keeps the suite runnable on every push.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class FakeEmbedder:
    """Records what it was asked to embed; returns a fixed vector."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str) -> Sequence[float]:
        self.seen.append(text)
        return [0.0, 1.0, 0.0]


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
