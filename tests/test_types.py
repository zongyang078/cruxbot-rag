"""Chunk / Source / Answer data types.

Only the behaviour these types add is tested. `is_citable` delegates to
`urls.is_specific`, which `test_urls.py` covers in detail, so what is checked
here is that the delegation happens -- not the URL rules again.
"""

from __future__ import annotations

from cruxbot.types import Chunk, Source


class TestChunk:
    def test_citability_follows_the_url_rules(self) -> None:
        document_level = Chunk(text="t", chunk_id="1", source_url="https://example.com/route/5")
        site_level = Chunk(
            text="t", chunk_id="1", source_url="https://www.mountainproject.com/forum"
        )
        assert document_level.is_citable is True
        assert site_level.is_citable is False

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
