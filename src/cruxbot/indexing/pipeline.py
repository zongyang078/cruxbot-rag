"""Build the dense and sparse indexes from a stream of documents.

Full and incremental indexing are the same code path: the builder consumes any
iterable of documents and upserts by deterministic chunk id, so re-running it
over a changed subset overwrites exactly those chunks. Nothing here knows
whether it was handed the whole corpus or one day of new Reddit posts.

Only the dense index is genuinely incremental. BM25 statistics -- document
frequency and average document length -- are corpus-global, so the sparse
index is rebuilt rather than patched. At this corpus size a rebuild is minutes,
which is cheap enough that keeping it correct beats keeping it incremental.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from cruxbot.indexing.chunking import chunk_document
from cruxbot.retrieval.dense import Embedder, VectorStore
from cruxbot.retrieval.sparse import BM25Index
from cruxbot.types import Document


@dataclass(slots=True)
class IndexStats:
    """What one indexing run did."""

    documents: int = 0
    chunks: int = 0
    batches: int = 0
    skipped_empty: int = 0
    elapsed_s: float = 0.0
    by_content_type: dict[str, int] = field(default_factory=dict)

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.elapsed_s if self.elapsed_s else 0.0


def batched(items: Iterable[Document], size: int) -> Iterator[list[Document]]:
    """Group an iterable into lists of at most `size`.

    Written as a generator over the input stream so that indexing never needs
    the whole corpus in memory at once.
    """
    batch: list[Document] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class IndexBuilder:
    """Chunk, embed, and upsert documents into a vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        batch_size: int = 256,
        on_progress: Callable[[IndexStats], None] | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size
        self.on_progress = on_progress

    def build(self, documents: Iterable[Document]) -> IndexStats:
        """Index every document, returning what was written.

        Chunking happens lazily inside the batching generator, so a corpus
        larger than memory streams through rather than being materialized.
        """
        stats = IndexStats()
        started = time.perf_counter()

        for batch in batched(self._chunk_stream(documents, stats), self.batch_size):
            self._write(batch)
            stats.chunks += len(batch)
            stats.batches += 1
            stats.elapsed_s = time.perf_counter() - started
            if self.on_progress:
                self.on_progress(stats)

        stats.elapsed_s = time.perf_counter() - started
        return stats

    def _chunk_stream(self, documents: Iterable[Document], stats: IndexStats) -> Iterator[Document]:
        for document in documents:
            chunks = chunk_document(document)
            if not chunks:
                stats.skipped_empty += 1
                continue
            stats.documents += 1
            key = document.content_type or "unknown"
            stats.by_content_type[key] = stats.by_content_type.get(key, 0) + len(chunks)
            yield from chunks

    def _write(self, chunks: list[Document]) -> None:
        texts = [c.text for c in chunks]
        self.store.upsert(
            ids=[c.chunk_id for c in chunks],
            texts=texts,
            embeddings=self.embedder.encode_batch(texts),
            metadatas=[c.index_metadata() for c in chunks],
        )


def build_sparse_index(documents: Iterable[Document]) -> BM25Index:
    """Build the BM25 index over the same chunks the dense index holds.

    Takes chunks rather than whole documents so the two indexes address the
    identical units -- if they disagreed, fusion would be merging rankings over
    different things and a chunk_id from one could not be resolved in the other.
    """
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str]] = []

    for document in documents:
        for chunk in chunk_document(document):
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(chunk.index_metadata())

    return BM25Index.build(ids, texts, metadatas)
