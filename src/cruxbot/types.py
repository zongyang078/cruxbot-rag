"""Shared data types for the retrieval and generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cruxbot import urls


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
