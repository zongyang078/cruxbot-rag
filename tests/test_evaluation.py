"""The labelled query set and the ablation runner."""

from __future__ import annotations

import json

import pytest

from cruxbot.evaluation.dataset import (
    LabeledQuery,
    load_queries,
    pool,
    save_queries,
    usable,
)
from cruxbot.evaluation.runner import (
    RetrievalConfig,
    format_table,
    run_all,
    run_config,
    summarize,
)


def labeled(query_id: str, relevant: dict[str, float], category: str = "route") -> LabeledQuery:
    return LabeledQuery(
        query_id=query_id, query=f"query {query_id}", category=category, relevance=relevant
    )


def ranked(*chunk_ids: str) -> RetrievalConfig:
    """A configuration that always returns the same ranking."""
    return RetrievalConfig(name="fixed", retrieve=lambda q, k: list(chunk_ids)[:k])


class TestLabeledQuery:
    def test_counts_only_positive_grades(self) -> None:
        assert labeled("q1", {"a": 2.0, "b": 0.0}).n_relevant() == 1

    def test_usable_requires_a_relevant_passage(self) -> None:
        assert labeled("q1", {"a": 2.0}).is_usable() is True
        assert labeled("q1", {"a": 0.0}).is_usable() is False
        assert labeled("q1", {}).is_usable() is False

    def test_a_grade_one_passage_does_not_make_a_query_usable(self) -> None:
        # Grade 1 is "on topic but does not address the question". A query whose
        # pool holds only those would score every configuration zero, so the
        # usability bar must match the one the metrics apply.
        assert labeled("q1", {"a": 1.0}).is_usable() is False
        assert labeled("q1", {"a": 1.0}).is_usable(min_grade=1.0) is True

    def test_round_trips_through_a_dict(self) -> None:
        original = LabeledQuery("q1", "text", "route", {"a": 2.0}, ["hybrid"])
        assert LabeledQuery.from_dict(original.to_dict()) == original

    def test_from_dict_tolerates_missing_fields(self) -> None:
        query = LabeledQuery.from_dict({"query_id": "q1", "query": "text"})
        assert query.relevance == {}
        assert query.pooled_from == []

    def test_from_dict_coerces_grades_to_float(self) -> None:
        # JSON round-trips integers as ints; metrics arithmetic wants floats.
        query = LabeledQuery.from_dict({"query_id": "q1", "query": "t", "relevance": {"a": 2}})
        assert isinstance(query.relevance["a"], float)


class TestUsable:
    def test_filters_out_unusable_queries(self) -> None:
        queries = [labeled("q1", {"a": 2.0}), labeled("q2", {})]
        # A query with no relevant passage scores every configuration zero, so
        # including it would depress all of them equally and inform nothing.
        assert [q.query_id for q in usable(queries)] == ["q1"]


class TestPersistence:
    def test_round_trips_through_jsonl(self, tmp_path) -> None:
        path = tmp_path / "queries.jsonl"
        queries = [labeled("q1", {"a": 2.0}), labeled("q2", {"b": 2.0})]
        assert save_queries(queries, path) == 2
        assert load_queries(path) == queries

    def test_creates_parent_directories(self, tmp_path) -> None:
        save_queries([labeled("q1", {"a": 2.0})], tmp_path / "nested" / "q.jsonl")
        assert (tmp_path / "nested" / "q.jsonl").exists()

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="No query set at"):
            load_queries(tmp_path / "absent.jsonl")

    def test_skips_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "queries.jsonl"
        path.write_text(json.dumps({"query_id": "q1", "query": "t"}) + "\n\n")
        assert len(load_queries(path)) == 1


class TestPool:
    def test_unions_across_configurations(self) -> None:
        assert set(pool({"dense": ["a", "b"], "sparse": ["b", "c"]}, depth=2)) == {"a", "b", "c"}

    def test_respects_the_depth_cutoff(self) -> None:
        assert pool({"dense": ["a", "b", "c"]}, depth=2) == ["a", "b"]

    def test_deduplicates_while_preserving_first_appearance(self) -> None:
        assert pool({"x": ["a", "b"], "y": ["b", "a"]}, depth=2) == ["a", "b"]

    def test_empty_input_gives_an_empty_pool(self) -> None:
        assert pool({}, depth=10) == []


class TestRunConfig:
    def test_scores_a_perfect_retriever(self) -> None:
        report = run_config(ranked("a"), [labeled("q1", {"a": 2.0})], ks=(1,))
        assert report.metrics["recall@1"] == 1.0
        assert report.metrics["mrr"] == 1.0

    def test_scores_a_retriever_that_misses(self) -> None:
        report = run_config(ranked("x"), [labeled("q1", {"a": 2.0})], ks=(1,))
        assert report.metrics["recall@1"] == 0.0

    def test_skips_unusable_queries(self) -> None:
        queries = [labeled("q1", {"a": 2.0}), labeled("q2", {})]
        assert run_config(ranked("a"), queries, ks=(1,)).n_queries == 1

    def test_retrieves_deep_enough_for_the_largest_k(self) -> None:
        # Retrieving fewer results than a metric's cutoff would report a
        # ceiling imposed by the harness rather than by the retriever.
        seen: list[int] = []

        def record(query: str, k: int) -> list[str]:
            seen.append(k)
            return ["a"]

        run_config(RetrievalConfig("c", record), [labeled("q1", {"a": 2.0})], ks=(5, 50))
        assert seen == [50]

    def test_a_failing_query_scores_zero_rather_than_voiding_the_run(self) -> None:
        # Dropping it would let a configuration improve its average by erroring
        # on the queries it handles badly.
        def explode(query: str, k: int) -> list[str]:
            raise RuntimeError("index unavailable")

        report = run_config(
            RetrievalConfig("c", explode),
            [labeled("q1", {"a": 2.0}), labeled("q2", {"b": 2.0})],
            ks=(1,),
        )
        assert report.n_errors == 2
        assert report.n_queries == 2
        assert report.metrics["recall@1"] == 0.0
        assert "index unavailable" in report.results[0].error

    def test_records_latency_percentiles(self) -> None:
        report = run_config(ranked("a"), [labeled("q1", {"a": 2.0})], ks=(1,))
        assert report.latency_p50_ms >= 0
        assert report.latency_p95_ms >= report.latency_p50_ms

    def test_breaks_results_down_by_category(self) -> None:
        queries = [
            labeled("q1", {"a": 2.0}, category="route"),
            labeled("q2", {"x": 2.0}, category="safety"),
        ]
        report = run_config(ranked("a"), queries, ks=(1,))
        assert report.by_category["route"]["recall@1"] == 1.0
        assert report.by_category["safety"]["recall@1"] == 0.0

    def test_empty_query_set_is_not_an_error(self) -> None:
        report = run_config(ranked("a"), [], ks=(1,))
        assert report.n_queries == 0
        assert report.metrics == {}


class TestRunAll:
    def test_evaluates_every_configuration_on_the_same_queries(self) -> None:
        queries = [labeled("q1", {"a": 2.0})]
        configs = [
            RetrievalConfig("hits", lambda q, k: ["a"]),
            RetrievalConfig("misses", lambda q, k: ["x"]),
        ]
        reports = run_all(configs, queries, ks=(1,))
        assert [r.name for r in reports] == ["hits", "misses"]
        assert reports[0].metrics["recall@1"] == 1.0
        assert reports[1].metrics["recall@1"] == 0.0


class TestFormatTable:
    def _reports(self):
        queries = [labeled("q1", {"a": 2.0})]
        configs = [
            RetrievalConfig("hits", lambda q, k: ["a"]),
            RetrievalConfig("misses", lambda q, k: ["x"]),
        ]
        return run_all(configs, queries, ks=(10,))

    def test_lists_every_configuration(self) -> None:
        table = format_table(self._reports(), columns=("recall@10",))
        assert "hits" in table
        assert "misses" in table

    def test_bolds_the_best_value_per_column(self) -> None:
        table = format_table(self._reports(), columns=("recall@10",))
        assert "**1.000**" in table

    def test_does_not_bold_a_column_where_everything_scored_zero(self) -> None:
        # Bolding a tie at zero would present total failure as a winner.
        configs = [RetrievalConfig("a", lambda q, k: ["x"])]
        table = format_table(run_all(configs, [labeled("q1", {"z": 2.0})], ks=(10,)))
        assert "**" not in table

    def test_includes_latency_columns(self) -> None:
        table = format_table(self._reports(), columns=("recall@10",))
        assert "p50 ms" in table
        assert "p95 ms" in table

    def test_empty_input_is_handled(self) -> None:
        assert format_table([]) == "(no results)"


class TestSummarize:
    def test_is_json_serializable(self) -> None:
        queries = [labeled("q1", {"a": 2.0})]
        reports = run_all([RetrievalConfig("c", lambda q, k: ["a"])], queries, ks=(10,))
        # Reports carry per-query detail; the summary must stay writable to disk.
        assert json.loads(json.dumps(summarize(reports)))["configurations"][0]["name"] == "c"
