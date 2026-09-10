"""Index building: corpus loading, batching, and upsert behaviour."""

from __future__ import annotations

import json

import pytest

from cruxbot.indexing.corpus import load_documents, to_document, to_jsonl
from cruxbot.indexing.pipeline import IndexBuilder, batched, build_sparse_index
from cruxbot.types import Document
from tests.fakes import FakeEmbedder, RecordingStore


def docs(n: int, text: str = "a short passage", content_type: str = "route") -> list[Document]:
    return [Document(doc_id=f"d{i}", text=text, content_type=content_type) for i in range(n)]


class TestToDocument:
    def test_promotes_known_fields(self) -> None:
        document = to_document(
            {"doc_id": "x", "text": "t", "content_type": "route", "grade": "5.10a"}
        )
        assert document.doc_id == "x"
        assert document.grade == "5.10a"

    def test_keeps_unknown_fields_in_metadata(self) -> None:
        # Dropping them would lose lat/lng and per-source extras that later
        # filtering may need.
        document = to_document({"doc_id": "x", "text": "t", "lat": "37.7", "pitches": 4})
        assert document.metadata == {"lat": "37.7", "pitches": 4}

    def test_missing_fields_become_empty_strings(self) -> None:
        document = to_document({"doc_id": "x"})
        assert document.text == ""
        assert document.grade == ""

    def test_null_values_become_empty_strings(self) -> None:
        # The cleaning pipeline emits nulls for absent fields; `or ""` has to
        # catch those, not just missing keys.
        assert to_document({"doc_id": "x", "grade": None}).grade == ""


class TestLoadDocuments:
    def test_reads_a_json_array(self, tmp_path) -> None:
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps([{"doc_id": "a", "text": "one"}]))
        assert [d.doc_id for d in load_documents(path)] == ["a"]

    def test_reads_jsonl(self, tmp_path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"doc_id": "a", "text": "one"}\n{"doc_id": "b", "text": "two"}\n')
        assert [d.doc_id for d in load_documents(path)] == ["a", "b"]

    def test_skips_blank_jsonl_lines(self, tmp_path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"doc_id": "a", "text": "one"}\n\n')
        assert len(list(load_documents(path))) == 1

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="No corpus at"):
            list(load_documents(tmp_path / "absent.json"))

    def test_non_list_json_raises(self, tmp_path) -> None:
        path = tmp_path / "corpus.json"
        path.write_text('{"doc_id": "a"}')
        with pytest.raises(ValueError, match="expected a list"):
            list(load_documents(path))

    def test_jsonl_conversion_round_trips(self, tmp_path) -> None:
        source = tmp_path / "corpus.json"
        source.write_text(json.dumps([{"doc_id": "a", "text": "one", "lat": "37.7"}]))

        destination = tmp_path / "corpus.jsonl"
        assert to_jsonl(source, destination) == 1

        document = next(iter(load_documents(destination)))
        assert document.doc_id == "a"
        assert document.metadata["lat"] == "37.7"


class TestBatched:
    def test_splits_into_full_batches(self) -> None:
        assert [len(b) for b in batched(docs(10), 4)] == [4, 4, 2]

    def test_empty_input_yields_nothing(self) -> None:
        assert list(batched([], 4)) == []

    def test_is_lazy(self) -> None:
        # The corpus is larger than memory once expanded into chunks, so
        # batching must not materialize the input.
        consumed = 0

        def counting():
            nonlocal consumed
            for document in docs(100):
                consumed += 1
                yield document

        next(iter(batched(counting(), 4)))
        assert consumed == 4


class TestIndexBuilder:
    def test_writes_every_chunk(self) -> None:
        store = RecordingStore()
        stats = IndexBuilder(FakeEmbedder(), store, batch_size=4).build(docs(10))
        assert stats.documents == 10
        assert stats.chunks == 10
        assert len(store.upserted) == 10

    def test_upserts_by_deterministic_chunk_id(self) -> None:
        # Indexing the same corpus twice must overwrite, not duplicate. This is
        # what makes a daily incremental run safe to repeat.
        store = RecordingStore()
        builder = IndexBuilder(FakeEmbedder(), store, batch_size=4)
        builder.build(docs(5))
        builder.build(docs(5))
        assert len({row["id"] for row in store.upserted}) == 5

    def test_batches_respect_the_configured_size(self) -> None:
        store = RecordingStore()
        IndexBuilder(FakeEmbedder(), store, batch_size=3).build(docs(7))
        assert [len(call) for call in store.upsert_calls] == [3, 3, 1]

    def test_embeds_in_batches_not_one_at_a_time(self) -> None:
        # Per-item encoding is roughly an order of magnitude slower on this
        # corpus, so the builder must use the batch path.
        embedder = FakeEmbedder()
        IndexBuilder(embedder, RecordingStore(), batch_size=4).build(docs(8))
        assert embedder.batch_calls == 2
        assert embedder.seen == []

    def test_skips_documents_with_no_text(self) -> None:
        store = RecordingStore()
        stats = IndexBuilder(FakeEmbedder(), store).build(
            [Document(doc_id="empty", text="   "), Document(doc_id="ok", text="real text")]
        )
        assert stats.skipped_empty == 1
        assert stats.documents == 1

    def test_long_documents_produce_multiple_chunks(self) -> None:
        store = RecordingStore()
        stats = IndexBuilder(FakeEmbedder(), store).build(
            [Document(doc_id="long", text="word " * 2000, content_type="article")]
        )
        assert stats.documents == 1
        assert stats.chunks > 1

    def test_records_chunks_per_content_type(self) -> None:
        stats = IndexBuilder(FakeEmbedder(), RecordingStore()).build(
            docs(3, content_type="route") + docs(2, content_type="article")
        )
        assert stats.by_content_type["route"] == 3
        assert stats.by_content_type["article"] == 2

    def test_writes_index_metadata(self) -> None:
        store = RecordingStore()
        IndexBuilder(FakeEmbedder(), store).build(
            [Document(doc_id="d", text="t", content_type="route", source_url="https://x/1")]
        )
        assert store.upserted[0]["metadata"]["source_url"] == "https://x/1"

    def test_reports_progress_per_batch(self) -> None:
        seen: list[int] = []
        IndexBuilder(
            FakeEmbedder(),
            RecordingStore(),
            batch_size=3,
            on_progress=lambda s: seen.append(s.chunks),
        ).build(docs(7))
        assert seen == [3, 6, 7]

    def test_empty_corpus_is_not_an_error(self) -> None:
        stats = IndexBuilder(FakeEmbedder(), RecordingStore()).build([])
        assert stats.chunks == 0
        assert stats.chunks_per_second == 0.0


class TestBuildSparseIndex:
    def test_indexes_chunks_not_documents(self) -> None:
        # The two indexes must address identical units, or a chunk_id from the
        # sparse side cannot be resolved on the dense side.
        index = build_sparse_index(
            [Document(doc_id="d", text="word " * 2000, content_type="article")]
        )
        assert index.n_docs > 1
        assert all(doc_id.startswith("d_c") for doc_id in index.doc_ids)

    def test_is_searchable(self) -> None:
        index = build_sparse_index(
            [
                Document(
                    doc_id="d", text="hangboard training protocol", content_type="forum_discussion"
                )
            ]
        )
        assert index.search("hangboard")[0]["chunk_id"] == "d_c0"

    def test_carries_content_type_for_filtering(self) -> None:
        index = build_sparse_index(
            [Document(doc_id="d", text="hangboard", content_type="forum_discussion")]
        )
        assert index.search("hangboard", content_types=["forum_discussion"])
        assert not index.search("hangboard", content_types=["route"])
