"""Document chunking."""

from __future__ import annotations

import pytest

from cruxbot.indexing.chunking import (
    CHARS_PER_TOKEN,
    CHUNK_CONFIG,
    chunk_document,
    split_text,
)
from cruxbot.types import Document


class TestSplitTextInvariants:
    def test_empty_text_gives_no_chunks(self) -> None:
        assert split_text("", 100, 10) == []
        assert split_text("   \n  ", 100, 10) == []

    def test_short_text_stays_whole(self) -> None:
        assert split_text("a short passage", 100, 10) == ["a short passage"]

    def test_never_exceeds_chunk_size(self) -> None:
        text = " ".join(f"word{i}" for i in range(500))
        assert all(len(chunk) <= 100 for chunk in split_text(text, 100, 20))

    def test_holds_even_without_any_separator(self) -> None:
        # A single unbroken token longer than the limit must still be split,
        # or the size guarantee silently fails on minified or CJK text.
        assert all(len(chunk) <= 50 for chunk in split_text("x" * 500, 50, 10))

    def test_loses_no_content(self) -> None:
        text = "alpha. beta. gamma. delta. epsilon. zeta. eta. theta."
        joined = "".join(split_text(text, 20, 0))
        for word in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"):
            assert word in joined

    def test_is_deterministic(self) -> None:
        text = "sentence one. sentence two. sentence three. " * 20
        assert split_text(text, 100, 20) == split_text(text, 100, 20)


class TestSplitTextBoundaries:
    def test_prefers_paragraph_breaks(self) -> None:
        text = "First paragraph here.\n\nSecond paragraph here."
        assert split_text(text, 30, 0) == ["First paragraph here.", "Second paragraph here."]

    def test_falls_back_to_sentence_breaks(self) -> None:
        # No paragraph break available, so it must split at ". " rather than
        # mid-word.
        chunks = split_text("First sentence here. Second sentence here.", 25, 0)
        assert all(not chunk.endswith(("Firs", "Secon")) for chunk in chunks)

    def test_overlap_repeats_trailing_context(self) -> None:
        # A sentence spanning a boundary should be retrievable from either
        # side, so consecutive chunks must share text.
        chunks = split_text(" ".join(f"w{i}" for i in range(200)), 100, 30)
        assert len(chunks) > 1
        assert chunks[0][-10:] in chunks[1]

    def test_overlap_yields_to_a_fragment_that_fills_the_chunk(self) -> None:
        # Text with no usable separator arrives pre-split at exactly the size
        # limit, leaving no room for carried context. Overlap must be dropped
        # rather than pushing the chunk over the limit.
        assert all(len(c) <= 40 for c in split_text("x" * 400, 40, 15))

    def test_zero_overlap_produces_disjoint_chunks(self) -> None:
        chunks = split_text("alpha. beta. gamma. delta.", 15, 0)
        assert sum(len(c) for c in chunks) <= len("alpha. beta. gamma. delta.") + len(chunks)


class TestSplitTextValidation:
    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            split_text("text", 0, 0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            split_text("text", 100, -1)

    def test_rejects_overlap_at_or_above_chunk_size(self) -> None:
        # Overlap >= chunk_size never advances, so the splitter would loop
        # forever or emit infinite duplicate chunks.
        with pytest.raises(ValueError, match="must be smaller than chunk_size"):
            split_text("text", 100, 100)


def doc(**overrides: object) -> Document:
    defaults: dict[str, object] = {
        "doc_id": "abc123",
        "text": "A short route description.",
        "content_type": "route",
        "source_url": "https://example.com/route/1",
        "grade": "5.10a",
    }
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


class TestChunkDocument:
    def test_short_document_becomes_one_chunk(self) -> None:
        chunks = chunk_document(doc())
        assert len(chunks) == 1
        assert chunks[0].total_chunks == 1

    def test_empty_document_is_dropped(self) -> None:
        assert chunk_document(doc(text="")) == []
        assert chunk_document(doc(text="   ")) == []

    def test_long_document_is_split(self) -> None:
        assert len(chunk_document(doc(text="word " * 2000))) > 1

    def test_chunk_ids_are_deterministic(self) -> None:
        # Re-running ingestion must overwrite a document's chunks, not append
        # a second copy beside them.
        first = [c.chunk_id for c in chunk_document(doc(text="word " * 2000))]
        second = [c.chunk_id for c in chunk_document(doc(text="word " * 2000))]
        assert first == second

    def test_chunk_ids_derive_from_the_parent(self) -> None:
        chunks = chunk_document(doc(doc_id="xyz", text="word " * 2000))
        assert chunks[0].chunk_id == "xyz_c0"
        assert chunks[1].chunk_id == "xyz_c1"

    def test_chunks_inherit_parent_metadata(self) -> None:
        chunk = chunk_document(doc(text="word " * 2000))[0]
        assert chunk.source_url == "https://example.com/route/1"
        assert chunk.grade == "5.10a"
        assert chunk.content_type == "route"

    def test_chunks_record_their_position(self) -> None:
        chunks = chunk_document(doc(text="word " * 2000))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.total_chunks == len(chunks) for c in chunks)

    def test_parent_id_survives_rechunking(self) -> None:
        chunk = chunk_document(doc(text="word " * 2000))[0]
        assert chunk_document(chunk)[0].parent_doc_id == "abc123"

    @pytest.mark.parametrize("content_type", sorted(CHUNK_CONFIG))
    def test_respects_the_size_configured_for_each_type(self, content_type: str) -> None:
        limit = CHUNK_CONFIG[content_type][0] * CHARS_PER_TOKEN
        chunks = chunk_document(doc(content_type=content_type, text="word " * 4000))
        assert all(len(c.text) <= limit for c in chunks)

    def test_unknown_content_type_uses_the_default(self) -> None:
        assert chunk_document(doc(content_type="podcast", text="word " * 2000))


class TestIndexMetadata:
    def test_flattens_to_strings(self) -> None:
        meta = doc().index_metadata()
        assert all(isinstance(v, str) for v in meta.values())

    def test_omits_empty_fields(self) -> None:
        # A filter on a missing key must not match documents where the field
        # is simply absent.
        assert "location" not in doc(location="").index_metadata()

    def test_drops_nested_values(self) -> None:
        # Chroma rejects nested metadata; dropping it here rather than at the
        # call site means every backend gets an acceptable dict.
        meta = doc(metadata={"stars": {"avg": 2.9}}).index_metadata()
        assert all(isinstance(v, str) for v in meta.values())

    def test_records_the_parent_document(self) -> None:
        chunk = chunk_document(doc(text="word " * 2000))[1]
        assert chunk.index_metadata()["parent_doc_id"] == "abc123"
