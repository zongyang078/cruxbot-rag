"""Hybrid retrieval: intent routing, fusion, hydration, backfill."""

from __future__ import annotations

from cruxbot import intent
from cruxbot.retrieval.dense import DenseRetriever
from cruxbot.retrieval.hybrid import HybridRetriever
from cruxbot.retrieval.sparse import BM25Index
from tests.fakes import FakeEmbedder, FakeStore, make_doc


def build(store: FakeStore, sparse: BM25Index | None = None, **kwargs) -> HybridRetriever:
    return HybridRetriever(DenseRetriever(FakeEmbedder(), store), sparse, **kwargs)


class TestIntentRouting:
    def test_is_off_by_default(self, store: FakeStore) -> None:
        # Measured against the labelled benchmark, the intent prior produced no
        # quality difference and a 79% higher p95 latency. It stays available
        # so the ablation can reproduce that row, not because it is on.
        result = build(store).retrieve("Recommend a route in Yosemite")
        assert result.intent is None
        assert result.content_types is None

    def test_detected_intent_narrows_content_types_when_enabled(self, store: FakeStore) -> None:
        result = build(store).retrieve("Recommend a route in Yosemite", use_intent=True)
        assert result.intent == intent.ROUTE
        assert result.content_types == [intent.ROUTE]

    def test_ambiguous_query_applies_no_filter(self, store: FakeStore) -> None:
        result = build(store).retrieve("hello there", use_intent=True)
        assert result.intent is None
        assert result.content_types is None

    def test_explicit_content_types_override_intent(self, store: FakeStore) -> None:
        result = build(store).retrieve(
            "Recommend a route in Yosemite", use_intent=True, content_types=["article"]
        )
        assert result.content_types == ["article"]

    def test_an_explicit_filter_applies_without_intent_detection(self, store: FakeStore) -> None:
        # top_k=1 so the filter's own result stands; asking for more would
        # trigger the backfill, which is a different behaviour tested below.
        result = build(store).retrieve("anything", content_types=["article"], top_k=1)
        assert result.content_types == ["article"]
        assert [c.chunk_id for c in result.chunks] == ["a1"]


class TestFusion:
    def test_works_without_a_sparse_index(self, store: FakeStore) -> None:
        result = build(store, sparse=None).retrieve("q", top_k=2)
        assert result.n_sparse == 0
        assert len(result.chunks) == 2

    def test_combines_both_retrievers(self, store: FakeStore) -> None:
        sparse = BM25Index.build(
            ["f1"], ["hangboard training"], [{"content_type": "forum_discussion"}]
        )
        result = build(store, sparse).retrieve("hangboard", use_intent=False, top_k=4)
        assert result.n_dense > 0
        assert result.n_sparse > 0

    def test_a_chunk_found_by_both_is_ranked_first(self) -> None:
        store = FakeStore([make_doc("only_dense"), make_doc("both")])
        sparse = BM25Index.build(["both"], ["hangboard"], [{"content_type": "route"}])
        result = build(store, sparse).retrieve("hangboard", use_intent=False, top_k=2)
        assert result.chunks[0].chunk_id == "both"

    def test_weights_are_forwarded_to_fusion(self) -> None:
        store = FakeStore([make_doc("dense_only")])
        sparse = BM25Index.build(["sparse_only"], ["hangboard"], [{"content_type": "route"}])

        favor_sparse = build(store, sparse, dense_weight=1.0, sparse_weight=5.0)
        result = favor_sparse.retrieve("hangboard", use_intent=False, top_k=2)
        assert result.chunks[0].chunk_id == "sparse_only"


class TestHydration:
    def test_sparse_only_hits_get_their_text_filled_in(self) -> None:
        # BM25 stores term statistics, not document bodies. Without hydration
        # this chunk reaches the prompt as an empty context block.
        store = FakeStore([make_doc("s1", text="the real passage text")])
        sparse = BM25Index.build(["s1"], ["hangboard"], [{"content_type": "route"}])

        result = build(store, sparse).retrieve("hangboard", use_intent=False, top_k=1)
        assert result.chunks[0].text == "the real passage text"

    def test_no_lookup_when_every_hit_already_has_text(self, store: FakeStore) -> None:
        build(store).retrieve("q", top_k=2)
        assert store.get_calls == []

    def test_hydration_is_a_single_batched_call(self) -> None:
        store = FakeStore([make_doc("s1"), make_doc("s2")])
        sparse = BM25Index.build(
            ["s1", "s2"], ["hangboard", "hangboard"], [{"content_type": "route"}] * 2
        )
        build(store, sparse).retrieve("hangboard", use_intent=False, top_k=2)
        assert len(store.get_calls) <= 1


class TestBackfill:
    def test_starved_filter_is_backfilled(self) -> None:
        # Only one article exists, but five chunks were asked for. Answering
        # from one passage is worse than widening the pool.
        store = FakeStore(
            [make_doc("a1", content_type="article")]
            + [make_doc(f"r{i}", content_type="route") for i in range(5)]
        )
        result = build(store).retrieve(
            "What causes rappelling accidents?", use_intent=True, top_k=3
        )
        assert len(result.chunks) == 3

    def test_backfill_does_not_duplicate(self) -> None:
        store = FakeStore([make_doc("a1", content_type="article")])
        result = build(store).retrieve(
            "What causes rappelling accidents?", use_intent=True, top_k=5
        )
        assert len({c.chunk_id for c in result.chunks}) == len(result.chunks)

    def test_no_backfill_when_unfiltered(self, store: FakeStore) -> None:
        result = build(store).retrieve("hello there", top_k=10)
        assert len(result.chunks) == 4


class TestResultMetadata:
    def test_maps_source_url_onto_the_chunk(self, store: FakeStore) -> None:
        result = build(store).retrieve("q", top_k=1)
        assert result.chunks[0].source_url == "https://example.com/route/1"
        assert result.chunks[0].is_citable is True

    def test_records_elapsed_time(self, store: FakeStore) -> None:
        assert build(store).retrieve("q").elapsed_ms >= 0

    def test_records_candidate_counts_per_retriever(self, store: FakeStore) -> None:
        result = build(store).retrieve("hello there", top_k=2)
        assert result.n_dense == 4
        assert result.n_sparse == 0

    def test_candidate_pool_is_wider_than_top_k(self, store: FakeStore) -> None:
        # Fusion can only promote a chunk that some retriever surfaced, so the
        # candidate pool has to exceed the number of chunks finally returned.
        build(store).retrieve("q", top_k=5)
        assert store.queries[0]["n_results"] >= 50

    def test_explicit_candidate_k_is_honoured(self, store: FakeStore) -> None:
        build(store).retrieve("q", top_k=2, candidate_k=11)
        assert store.queries[0]["n_results"] == 11

    def test_empty_store_returns_no_chunks(self) -> None:
        assert build(FakeStore([])).retrieve("q").chunks == []
