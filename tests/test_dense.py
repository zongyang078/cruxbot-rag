"""Dense retrieval."""

from __future__ import annotations

from cruxbot.retrieval.dense import DenseRetriever
from tests.fakes import FakeEmbedder, FakeStore


class TestDenseRetriever:
    def test_embeds_the_query(self, embedder: FakeEmbedder, store: FakeStore) -> None:
        DenseRetriever(embedder, store).search("how do I train?")
        assert embedder.seen == ["how do I train?"]

    def test_expands_grades_before_embedding(
        self, embedder: FakeEmbedder, store: FakeStore
    ) -> None:
        # A bi-encoder cannot know that 7a+ and 5.12a are the same difficulty,
        # so the equivalence has to be made lexical before encoding.
        DenseRetriever(embedder, store).search("a 5.11a route")
        assert "6c" in embedder.seen[0]

    def test_grade_expansion_can_be_disabled(
        self, embedder: FakeEmbedder, store: FakeStore
    ) -> None:
        DenseRetriever(embedder, store).search("a 5.11a route", expand_grades=False)
        assert embedder.seen == ["a 5.11a route"]

    def test_query_without_a_grade_is_unmodified(
        self, embedder: FakeEmbedder, store: FakeStore
    ) -> None:
        DenseRetriever(embedder, store).search("best belay device")
        assert embedder.seen == ["best belay device"]

    def test_passes_top_k_to_the_store(self, embedder: FakeEmbedder, store: FakeStore) -> None:
        DenseRetriever(embedder, store).search("q", top_k=7)
        assert store.queries[0]["n_results"] == 7

    def test_passes_content_type_filter_through(
        self, embedder: FakeEmbedder, store: FakeStore
    ) -> None:
        results = DenseRetriever(embedder, store).search("q", content_types=["article"])
        assert [r["chunk_id"] for r in results] == ["a1"]

    def test_returns_store_results_unchanged(
        self, embedder: FakeEmbedder, store: FakeStore
    ) -> None:
        results = DenseRetriever(embedder, store).search("q", top_k=2)
        assert [r["chunk_id"] for r in results] == ["r1", "r2"]

    def test_empty_store_returns_nothing(self, embedder: FakeEmbedder) -> None:
        assert DenseRetriever(embedder, FakeStore([])).search("q") == []
