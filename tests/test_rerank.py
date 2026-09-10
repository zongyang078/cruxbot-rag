"""Cross-encoder reranking, and its integration into hybrid retrieval."""

from __future__ import annotations

from cruxbot.retrieval.dense import DenseRetriever
from cruxbot.retrieval.hybrid import HybridRetriever
from cruxbot.retrieval.rerank import rerank
from tests.fakes import FakeEmbedder, FakeReranker, FakeStore, make_doc


def hits(*specs: tuple[str, str]) -> list[dict[str, object]]:
    return [{"chunk_id": cid, "text": text} for cid, text in specs]


class TestRerank:
    def test_empty_input_gives_empty_output(self) -> None:
        assert rerank(FakeReranker({}), "q", []) == []

    def test_reorders_by_score(self) -> None:
        reranker = FakeReranker({"weak text": 0.1, "strong text": 0.9})
        result = rerank(reranker, "q", hits(("a", "weak text"), ("b", "strong text")))
        assert [h["chunk_id"] for h in result] == ["b", "a"]

    def test_annotates_the_score(self) -> None:
        result = rerank(FakeReranker({"text": 0.7}), "q", hits(("a", "text")))
        assert result[0]["rerank_score"] == 0.7

    def test_truncates_to_top_k(self) -> None:
        reranker = FakeReranker({"a": 0.1, "b": 0.5, "c": 0.9})
        result = rerank(reranker, "q", hits(("x", "a"), ("y", "b"), ("z", "c")), top_k=2)
        assert [h["chunk_id"] for h in result] == ["z", "y"]

    def test_does_not_mutate_the_input(self) -> None:
        original = hits(("a", "text"))
        rerank(FakeReranker({"text": 0.5}), "q", original)
        assert "rerank_score" not in original[0]

    def test_passes_the_query_to_the_model(self) -> None:
        # A cross-encoder conditions the document score on the query; passing
        # the wrong query silently reduces it to a document-quality prior.
        reranker = FakeReranker({"text": 0.5})
        rerank(reranker, "how do I train fingers?", hits(("a", "text")))
        assert reranker.queries == ["how do I train fingers?"]

    def test_scores_all_candidates_in_one_call(self) -> None:
        reranker = FakeReranker({"a": 0.1, "b": 0.2, "c": 0.3})
        rerank(reranker, "q", hits(("x", "a"), ("y", "b"), ("z", "c")))
        assert reranker.calls == 1

    def test_textless_hits_sort_last_but_survive(self) -> None:
        # A missing body means hydration failed. Dropping the chunk would hide
        # the bug and shrink the context; ranking it last surfaces it instead.
        reranker = FakeReranker({"real text": 0.5})
        result = rerank(reranker, "q", hits(("empty", ""), ("full", "real text")))
        assert [h["chunk_id"] for h in result] == ["full", "empty"]

    def test_all_textless_does_not_call_the_model(self) -> None:
        reranker = FakeReranker({})
        result = rerank(reranker, "q", hits(("a", ""), ("b", "")))
        assert reranker.calls == 0
        assert len(result) == 2

    def test_ties_break_deterministically(self) -> None:
        reranker = FakeReranker({"a": 0.5, "b": 0.5})
        result = rerank(reranker, "q", hits(("first", "a"), ("second", "b")))
        assert [h["chunk_id"] for h in result] == ["first", "second"]


class TestHybridWithReranker:
    def _store(self) -> FakeStore:
        return FakeStore(
            [
                make_doc("weak", text="barely related passage"),
                make_doc("strong", text="exactly what was asked"),
            ]
        )

    def test_reranker_overrides_fusion_order(self) -> None:
        # "weak" is returned first by the store, so without reranking it wins.
        reranker = FakeReranker({"barely related passage": 0.1, "exactly what was asked": 0.9})
        retriever = HybridRetriever(
            DenseRetriever(FakeEmbedder(), self._store()), reranker=reranker
        )
        result = retriever.retrieve("q", top_k=1, use_intent=False)
        assert result.chunks[0].chunk_id == "strong"

    def test_without_a_reranker_fusion_order_stands(self) -> None:
        retriever = HybridRetriever(DenseRetriever(FakeEmbedder(), self._store()))
        result = retriever.retrieve("q", top_k=1, use_intent=False)
        assert result.chunks[0].chunk_id == "weak"

    def test_reranks_the_pool_not_just_top_k(self) -> None:
        # The point of the second stage is to promote a candidate the first
        # stage ranked low, so it must see more than the final top_k.
        reranker = FakeReranker({})
        HybridRetriever(DenseRetriever(FakeEmbedder(), self._store()), reranker=reranker).retrieve(
            "q", top_k=1, use_intent=False
        )
        assert reranker.n_scored > 1

    def test_records_rerank_timing_and_pool_size(self) -> None:
        reranker = FakeReranker({})
        result = HybridRetriever(
            DenseRetriever(FakeEmbedder(), self._store()), reranker=reranker
        ).retrieve("q", top_k=1, use_intent=False)
        assert result.n_reranked == 2
        assert result.rerank_ms >= 0

    def test_no_reranker_reports_no_rerank_work(self) -> None:
        result = HybridRetriever(DenseRetriever(FakeEmbedder(), self._store())).retrieve(
            "q", top_k=1, use_intent=False
        )
        assert result.n_reranked == 0
        assert result.rerank_ms == 0.0

    def test_chunk_score_reflects_the_final_stage(self) -> None:
        reranker = FakeReranker({"exactly what was asked": 0.9})
        result = HybridRetriever(
            DenseRetriever(FakeEmbedder(), self._store()), reranker=reranker
        ).retrieve("q", top_k=1, use_intent=False)
        assert result.chunks[0].score == 0.9

    def test_pool_size_is_configurable(self) -> None:
        reranker = FakeReranker({})
        store = FakeStore([make_doc(f"d{i}", text=f"passage {i}") for i in range(20)])
        HybridRetriever(
            DenseRetriever(FakeEmbedder(), store), reranker=reranker, rerank_pool=5
        ).retrieve("q", top_k=2, use_intent=False)
        assert reranker.n_scored == 5
