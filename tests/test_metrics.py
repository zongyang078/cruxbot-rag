"""Ranking metrics.

Every expected value here is hand-computed. Ranking metrics are easy to get
subtly wrong in ways that still look plausible -- an off-by-one in the log
discount, or an IDCG that ignores the cutoff -- and a self-consistent test
would not catch either.
"""

from __future__ import annotations

import math

import pytest

from cruxbot.evaluation.metrics import (
    aggregate,
    dcg_at_k,
    evaluate_query,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    threshold,
)

# Three relevant documents, graded. "d9" and below are not relevant.
GRADED = {"a": 3.0, "b": 2.0, "c": 1.0}
BINARY = {"a": 1.0, "b": 1.0, "c": 1.0}


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], BINARY, 3) == 1.0

    def test_none_retrieved(self) -> None:
        assert recall_at_k(["x", "y"], BINARY, 2) == 0.0

    def test_partial(self) -> None:
        assert recall_at_k(["a", "x", "y"], BINARY, 3) == pytest.approx(1 / 3)

    def test_respects_the_cutoff(self) -> None:
        # "b" sits at rank 4 and must not count towards recall@3.
        assert recall_at_k(["a", "x", "y", "b"], BINARY, 3) == pytest.approx(1 / 3)

    def test_uses_graded_gain(self) -> None:
        # Finding the grade-3 document recovers 3 of 6 available gain.
        assert recall_at_k(["a"], GRADED, 1) == pytest.approx(0.5)

    def test_no_relevant_documents_scores_zero(self) -> None:
        # Not 1.0: a query with no labels says nothing about the retriever.
        assert recall_at_k(["a"], {}, 5) == 0.0

    def test_negative_gains_are_ignored(self) -> None:
        assert recall_at_k(["a"], {"a": 1.0, "bad": -5.0}, 1) == 1.0


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], BINARY, 2) == 1.0

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x"], BINARY, 2) == 0.5

    def test_divides_by_k_not_by_results_returned(self) -> None:
        # Only one result came back, but the user asked for five slots; the
        # other four were empty and that is a real shortcoming.
        assert precision_at_k(["a"], BINARY, 5) == pytest.approx(0.2)

    def test_zero_k_is_not_a_division_error(self) -> None:
        assert precision_at_k(["a"], BINARY, 0) == 0.0

    def test_grading_is_thresholded(self) -> None:
        # Precision asks whether the result was worth showing, not how good.
        assert precision_at_k(["c"], GRADED, 1) == 1.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "x"], BINARY) == 1.0

    def test_third_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], BINARY) == pytest.approx(1 / 3)

    def test_none_relevant(self) -> None:
        assert reciprocal_rank(["x", "y"], BINARY) == 0.0

    def test_only_the_first_hit_counts(self) -> None:
        assert reciprocal_rank(["x", "a", "b"], BINARY) == 0.5

    def test_respects_an_optional_cutoff(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], BINARY, k=2) == 0.0


class TestDCG:
    def test_matches_the_definition(self) -> None:
        # gain/log2(rank+1): 3/log2(2) + 2/log2(3) + 1/log2(4)
        expected = 3 / 1.0 + 2 / math.log2(3) + 1 / 2.0
        assert dcg_at_k(["a", "b", "c"], GRADED, 3) == pytest.approx(expected)

    def test_first_position_is_undiscounted(self) -> None:
        assert dcg_at_k(["a"], GRADED, 1) == 3.0

    def test_irrelevant_documents_contribute_nothing(self) -> None:
        assert dcg_at_k(["x", "y"], GRADED, 2) == 0.0


class TestNDCG:
    def test_ideal_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], GRADED, 3) == pytest.approx(1.0)

    def test_reversed_ranking_scores_less(self) -> None:
        assert ndcg_at_k(["c", "b", "a"], GRADED, 3) < 1.0

    def test_order_matters_even_with_the_same_documents(self) -> None:
        good = ndcg_at_k(["a", "b", "c"], GRADED, 3)
        bad = ndcg_at_k(["c", "a", "b"], GRADED, 3)
        assert good > bad

    def test_ideal_is_capped_at_k(self) -> None:
        # With three relevant documents but k=1, retrieving the best one is a
        # perfect result at that cutoff. An IDCG that ignored k would score
        # this at 3/6 and understate the retriever.
        assert ndcg_at_k(["a"], GRADED, 1) == pytest.approx(1.0)

    def test_no_relevant_documents_scores_zero(self) -> None:
        assert ndcg_at_k(["a"], {}, 5) == 0.0

    def test_normalizes_across_queries_with_different_label_counts(self) -> None:
        # Both retrievers put every relevant document in the ideal order; the
        # scores must match despite one query having more labels.
        few = ndcg_at_k(["a"], {"a": 1.0}, 10)
        many = ndcg_at_k(["a", "b", "c"], BINARY, 10)
        assert few == pytest.approx(many)


class TestEvaluateQuery:
    def test_reports_every_metric_at_every_k(self) -> None:
        scores = evaluate_query(["a"], BINARY, ks=(1, 5))
        assert set(scores) == {
            "mrr",
            "recall@1",
            "precision@1",
            "ndcg@1",
            "recall@5",
            "precision@5",
            "ndcg@5",
        }

    def test_values_are_consistent_with_the_individual_functions(self) -> None:
        scores = evaluate_query(["x", "a"], GRADED, ks=(2,), min_grade=1.0)
        assert scores["mrr"] == 0.5
        assert scores["recall@2"] == pytest.approx(3 / 6)

    def test_binary_metrics_respect_the_threshold(self) -> None:
        # "c" is grade 1: on topic, does not answer. Finding it is not a
        # retrieval success at the default bar.
        assert evaluate_query(["c"], GRADED, ks=(5,))["recall@5"] == 0.0
        assert evaluate_query(["c"], GRADED, ks=(5,), min_grade=1.0)["recall@5"] > 0

    def test_ndcg_stays_graded_regardless_of_the_threshold(self) -> None:
        # Thresholding before nDCG would discard exactly the information it
        # exists to use -- that a 3 outranks a 2.
        strict = evaluate_query(["a", "b", "c"], GRADED, ks=(3,))["ndcg@3"]
        loose = evaluate_query(["a", "b", "c"], GRADED, ks=(3,), min_grade=1.0)["ndcg@3"]
        assert strict == pytest.approx(loose) == pytest.approx(1.0)


class TestAggregate:
    def test_averages_each_metric(self) -> None:
        assert aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}]) == {"recall@5": 0.5}

    def test_empty_input_gives_empty_output(self) -> None:
        assert aggregate([]) == {}

    def test_missing_metrics_count_as_zero(self) -> None:
        # A query that errored contributes a zero rather than being dropped,
        # so a configuration cannot improve its average by failing.
        assert aggregate([{"mrr": 1.0}, {}]) == {"mrr": 0.5}

    def test_is_macro_averaged(self) -> None:
        # Each query counts once regardless of how many labels it has.
        assert aggregate([{"recall@5": 1.0}, {"recall@5": 1.0}, {"recall@5": 0.0}]) == {
            "recall@5": pytest.approx(2 / 3)
        }


class TestThreshold:
    def test_drops_grades_below_the_bar(self) -> None:
        assert threshold(GRADED, 2.0) == {"a": 3.0, "b": 2.0}

    def test_keeps_gains_at_full_value(self) -> None:
        # Not binarized to 1.0: nDCG downstream still needs the grade.
        assert threshold(GRADED, 2.0)["a"] == 3.0

    def test_a_bar_of_one_keeps_every_positive_grade(self) -> None:
        assert threshold(GRADED, 1.0) == GRADED

    def test_a_bar_above_every_grade_empties_the_map(self) -> None:
        assert threshold(GRADED, 4.0) == {}
