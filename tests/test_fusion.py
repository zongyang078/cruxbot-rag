"""Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from cruxbot.fusion import DEFAULT_K, rrf_merge


def doc(chunk_id: str, text: str = "") -> dict[str, object]:
    return {"chunk_id": chunk_id, "text": text or f"text for {chunk_id}"}


class TestRRFMerge:
    def test_empty_input_gives_empty_output(self) -> None:
        assert rrf_merge([]) == []
        assert rrf_merge([[], []]) == []

    def test_single_list_preserves_order(self) -> None:
        ranked = [doc("a"), doc("b"), doc("c")]
        merged = rrf_merge([ranked])
        assert [d["chunk_id"] for d in merged] == ["a", "b", "c"]

    def test_document_in_both_lists_outranks_one_in_either(self) -> None:
        # "b" is 2nd in both lists; "a" and "x" are 1st in one list each.
        # Appearing in both should win: 2/(60+1) > 1/(60+0).
        dense = [doc("a"), doc("b")]
        sparse = [doc("x"), doc("b")]
        merged = rrf_merge([dense, sparse])
        assert merged[0]["chunk_id"] == "b"

    def test_deduplicates_across_lists(self) -> None:
        merged = rrf_merge([[doc("a"), doc("b")], [doc("b"), doc("a")]])
        assert len(merged) == 2

    def test_annotates_rrf_score(self) -> None:
        merged = rrf_merge([[doc("a")]])
        assert merged[0]["rrf_score"] == pytest.approx(1 / DEFAULT_K)

    def test_scores_sum_across_lists(self) -> None:
        merged = rrf_merge([[doc("a")], [doc("a")]])
        assert merged[0]["rrf_score"] == pytest.approx(2 / DEFAULT_K)

    def test_weights_shift_the_ranking(self) -> None:
        dense = [doc("dense_top")]
        sparse = [doc("sparse_top")]

        favor_dense = rrf_merge([dense, sparse], weights=[2.0, 1.0])
        assert favor_dense[0]["chunk_id"] == "dense_top"

        favor_sparse = rrf_merge([dense, sparse], weights=[1.0, 2.0])
        assert favor_sparse[0]["chunk_id"] == "sparse_top"

    def test_mismatched_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="weights has length"):
            rrf_merge([[doc("a")], [doc("b")]], weights=[1.0])

    def test_does_not_mutate_input_documents(self) -> None:
        original = doc("a")
        rrf_merge([[original]])
        assert "rrf_score" not in original

    def test_ties_break_deterministically(self) -> None:
        # Two documents at rank 0 of two lists have identical scores; the
        # result must still be stable across runs.
        lists = [[doc("first")], [doc("second")]]
        assert rrf_merge(lists) == rrf_merge(lists)
        assert rrf_merge(lists)[0]["chunk_id"] == "first"

    def test_falls_back_to_text_when_chunk_id_is_missing(self) -> None:
        merged = rrf_merge([[{"text": "same passage"}], [{"text": "same passage"}]])
        assert len(merged) == 1
        assert merged[0]["rrf_score"] == pytest.approx(2 / DEFAULT_K)

    def test_larger_k_flattens_rank_advantage(self) -> None:
        ranked = [doc("a"), doc("b")]
        tight = rrf_merge([ranked], k=1)
        loose = rrf_merge([ranked], k=1000)

        tight_gap = tight[0]["rrf_score"] - tight[1]["rrf_score"]
        loose_gap = loose[0]["rrf_score"] - loose[1]["rrf_score"]
        assert loose_gap < tight_gap
