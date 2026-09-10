"""Dense vector retrieval.

The embedding model and the vector store are injected rather than constructed
here. Two consequences that matter:

- This module can be unit tested without installing torch or chromadb; the
  tests supply small fakes satisfying the Protocols below.
- Swapping the vector store (Chroma is adequate at 382k vectors, but not the
  right answer at 10M) touches one class, not the retrieval logic.

Heavy imports live inside the concrete implementations, so importing this
module costs nothing until a real backend is constructed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from cruxbot import grades


class Embedder(Protocol):
    """Turns text into a vector."""

    def encode(self, text: str) -> Sequence[float]: ...


class VectorStore(Protocol):
    """A queryable collection of embedded documents."""

    def query(
        self,
        embedding: Sequence[float],
        n_results: int,
        content_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest neighbours, best first.

        Each result carries at least "chunk_id", "text", "score", "metadata".
        """
        ...

    def get(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Fetch documents by id, keyed by id.

        The sparse index ranks by term statistics and does not store document
        bodies, so a chunk found only by BM25 needs its text hydrated from here
        before it can go into a prompt. Missing ids are simply absent from the
        result rather than raising.
        """
        ...


class DenseRetriever:
    """Retrieve chunks by embedding similarity."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        top_k: int = 20,
        content_types: Sequence[str] | None = None,
        expand_grades: bool = True,
    ) -> list[dict[str, Any]]:
        """Embed the query and return its nearest neighbours.

        When `expand_grades` is set, a query naming a climbing grade is
        rewritten to mention its equivalents before embedding. A bi-encoder has
        no way to know that 7a+ and 5.12a denote the same difficulty, so
        without this the query and the document land in unrelated regions of
        the vector space.
        """
        text = grades.expand_query(query) if expand_grades else query
        embedding = self.embedder.encode(text)
        return self.store.query(embedding, n_results=top_k, content_types=content_types)


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        from cruxbot.config import settings

        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> Sequence[float]:
        return self._model.encode(text).tolist()


class ChromaStore:
    """Vector store backed by a persistent ChromaDB collection."""

    def __init__(self, path: str | None = None, collection: str | None = None) -> None:
        import chromadb

        from cruxbot.config import settings

        self.path = str(path or settings.chroma_path)
        self.collection_name = collection or settings.collection
        self._collection = chromadb.PersistentClient(path=self.path).get_collection(
            self.collection_name
        )

    def count(self) -> int:
        return int(self._collection.count())

    def iter_documents(self, batch_size: int = 5000):
        """Yield (ids, documents, metadatas) batches across the whole collection.

        Used to build the sparse index from the same corpus the dense index
        holds, so the two retrievers can never disagree about what exists.
        """
        total = self.count()
        offset = 0
        while offset < total:
            batch = self._collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            yield batch["ids"], batch["documents"], batch["metadatas"]
            offset += batch_size

    def get(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        batch = self._collection.get(ids=ids, include=["documents", "metadatas"])
        return {
            chunk_id: {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata or {},
            }
            for chunk_id, document, metadata in zip(
                batch["ids"], batch["documents"], batch["metadatas"], strict=True
            )
        }

    def query(
        self,
        embedding: Sequence[float],
        n_results: int,
        content_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if content_types:
            kwargs["where"] = (
                {"content_type": content_types[0]}
                if len(content_types) == 1
                else {"content_type": {"$in": list(content_types)}}
            )

        result = self._collection.query(**kwargs)
        if not result["ids"] or not result["ids"][0]:
            return []

        return [
            {
                "chunk_id": chunk_id,
                "text": document,
                # Chroma returns cosine distance; smaller is closer. Negate so
                # that every retriever in this package reports "higher is
                # better" and fusion does not need to know the difference.
                "score": -distance,
                "metadata": metadata or {},
            }
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
