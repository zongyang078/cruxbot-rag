"""Shared data types for the retrieval and generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from cruxbot import urls


@dataclass(slots=True)
class Document:
    """One record in the unified corpus, or one chunk of it.

    Chunks are Documents too: they carry the same fields plus `chunk_index`,
    so a single type flows from ingestion through chunking into the index.
    """

    doc_id: str
    text: str
    content_type: str = ""
    title: str = ""
    source: str = ""
    source_url: str = ""
    grade: str = ""
    location: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Set once the document has been split.
    chunk_index: int | None = None
    total_chunks: int | None = None
    parent_doc_id: str = ""

    @property
    def chunk_id(self) -> str:
        """Stable identifier for this chunk within the index.

        Derived from the parent document id rather than assigned at write
        time, so re-chunking a document that changed produces the same ids and
        overwrites its previous chunks. Without this, every re-ingest would
        append a fresh copy of the document alongside the stale one.
        """
        if self.chunk_index is None:
            return self.doc_id
        return f"{self.parent_doc_id or self.doc_id}_c{self.chunk_index}"

    def as_chunk(self, text: str, index: int, total: int) -> Document:
        """Return a copy of this document representing one of its chunks."""
        return replace(
            self,
            text=text,
            chunk_index=index,
            total_chunks=total,
            parent_doc_id=self.parent_doc_id or self.doc_id,
        )

    def index_metadata(self) -> dict[str, str]:
        """Flatten to the string-valued metadata a vector store will accept.

        Chroma rejects nested values, so nested `metadata` is dropped here
        rather than at the call site. Empty fields are omitted so that a filter
        on a missing key does not match the empty string.
        """
        fields = {
            "content_type": self.content_type,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "grade": self.grade,
            "location": self.location,
            "parent_doc_id": self.parent_doc_id or self.doc_id,
        }
        return {key: str(value) for key, value in fields.items() if value}


@dataclass(slots=True)
class Chunk:
    """A retrieved passage, plus the metadata needed to cite it."""

    text: str
    chunk_id: str
    source_url: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_citable(self) -> bool:
        """True when this chunk's URL points at a specific document."""
        return urls.is_specific(self.source_url)

    @property
    def content_type(self) -> str:
        return str(self.metadata.get("content_type", ""))

    @property
    def grade(self) -> str:
        return str(self.metadata.get("grade", ""))

    @property
    def location(self) -> str:
        return str(self.metadata.get("location", ""))


@dataclass(slots=True)
class Source:
    """A citation surfaced to the user alongside an answer."""

    url: str
    snippet: str
    is_specific: bool

    @classmethod
    def from_chunk(cls, chunk: Chunk, snippet_chars: int = 160) -> Source:
        return cls(
            url=chunk.source_url,
            snippet=chunk.text[:snippet_chars],
            is_specific=chunk.is_citable,
        )


@dataclass(slots=True)
class Answer:
    """The result of one end-to-end query."""

    query: str
    text: str
    sources: list[Source] = field(default_factory=list)
    intent: str | None = None
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
