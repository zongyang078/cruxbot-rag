"""Chunk / Source / Answer data types."""

from __future__ import annotations

from cruxbot.types import Answer, Chunk, Source


class TestChunk:
    def test_citable_when_url_is_document_level(self) -> None:
        c = Chunk(text="t", chunk_id="1", source_url="https://example.com/route/5")
        assert c.is_citable is True

    def test_not_citable_when_url_is_site_level(self) -> None:
        c = Chunk(text="t", chunk_id="1", source_url="https://www.mountainproject.com/forum")
        assert c.is_citable is False

    def test_not_citable_without_a_url(self) -> None:
        assert Chunk(text="t", chunk_id="1").is_citable is False

    def test_metadata_accessors_default_to_empty_strings(self) -> None:
        c = Chunk(text="t", chunk_id="1")
        assert c.content_type == ""
        assert c.grade == ""
        assert c.location == ""

    def test_metadata_accessors_coerce_to_str(self) -> None:
        # Chroma returns numeric metadata unchanged; prompt building assumes str.
        c = Chunk(text="t", chunk_id="1", metadata={"grade": 5.10})
        assert isinstance(c.grade, str)


class TestSource:
    def test_from_chunk_truncates_the_snippet(self) -> None:
        c = Chunk(text="x" * 500, chunk_id="1")
        assert len(Source.from_chunk(c, snippet_chars=100).snippet) == 100

    def test_from_chunk_carries_citation_quality(self) -> None:
        c = Chunk(text="t", chunk_id="1", source_url="https://www.reddit.com")
        assert Source.from_chunk(c).is_specific is False


class TestAnswer:
    def test_defaults_are_independent_between_instances(self) -> None:
        a, b = Answer(query="q1", text="t1"), Answer(query="q2", text="t2")
        a.sources.append(Source(url="u", snippet="s", is_specific=True))
        assert b.sources == []

    def test_carries_a_latency_breakdown(self) -> None:
        # Latency was unmeasured in the original system; it is a first-class
        # field here so the evaluation harness can report it per stage.
        answer = Answer(query="q", text="t", retrieval_ms=120.0, generation_ms=4200.0)
        assert answer.retrieval_ms + answer.generation_ms == 4320.0
