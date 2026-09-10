"""Relevance judging.

No test here reaches the network. The judge's prompt construction, its
grade-to-chunk matching, and its abstention handling are all pure; the API call
itself is one thin method covered by the smoke run, not by CI.
"""

from __future__ import annotations

import pytest

from cruxbot.evaluation.judge import (
    GRADE_DEFINITIONS,
    MAX_PASSAGE_CHARS,
    Agreement,
    Judgement,
    PassageGrade,
    agreement,
    build_system_prompt,
    build_user_prompt,
    collect,
    sample_for_review,
    to_relevance,
)


class TestPrompts:
    def test_system_prompt_states_every_grade(self) -> None:
        prompt = build_system_prompt()
        for grade, text in GRADE_DEFINITIONS.items():
            assert str(grade) in prompt
            assert text in prompt

    def test_system_prompt_warns_against_topical_credit(self) -> None:
        # The failure mode that matters: grading a route description as
        # relevant to a training question because both concern climbing.
        assert "merely because both concern" in build_system_prompt()

    def test_user_prompt_numbers_passages_from_zero(self) -> None:
        prompt = build_user_prompt("q", ["first", "second"])
        assert "[0] first" in prompt
        assert "[1] second" in prompt

    def test_user_prompt_includes_the_query(self) -> None:
        assert "Question: how do I train?" in build_user_prompt("how do I train?", ["p"])

    def test_long_passages_are_truncated(self) -> None:
        # Thirty untruncated passages would not fit in one request.
        prompt = build_user_prompt("q", ["x" * 10_000])
        assert prompt.count("x") == MAX_PASSAGE_CHARS

    def test_empty_pool_still_renders(self) -> None:
        assert "Question: q" in build_user_prompt("q", [])


class TestCollect:
    def test_matches_grades_to_chunk_ids_by_index(self) -> None:
        result = collect(
            ["a", "b"], [PassageGrade(index=1, grade=3), PassageGrade(index=0, grade=1)]
        )
        assert {j.chunk_id: j.grade for j in result} == {"a": 1, "b": 3}

    def test_preserves_input_order(self) -> None:
        result = collect(["a", "b", "c"], [PassageGrade(index=2, grade=3)])
        assert [j.chunk_id for j in result] == ["a", "b", "c"]

    def test_an_ungraded_passage_abstains_rather_than_scoring_zero(self) -> None:
        # "Nobody looked" and "judged irrelevant" are different claims, and
        # only one of them belongs in a label set as a zero.
        result = collect(["a", "b"], [PassageGrade(index=0, grade=2)], note="not graded")
        assert result[1].abstained is True
        assert result[1].note == "not graded"

    def test_a_graded_passage_carries_no_note(self) -> None:
        result = collect(["a"], [PassageGrade(index=0, grade=2)], note="not graded")
        assert result[0].note == ""

    def test_out_of_range_indices_are_ignored(self) -> None:
        # A judge that invents an index must not corrupt a real one.
        result = collect(["a"], [PassageGrade(index=7, grade=3)])
        assert result[0].abstained is True

    def test_empty_pool_gives_no_judgements(self) -> None:
        assert collect([], []) == []


class TestToRelevance:
    def test_keeps_graded_passages(self) -> None:
        assert to_relevance([Judgement("a", 2), Judgement("b", 0)]) == {"a": 2.0, "b": 0.0}

    def test_omits_abstentions(self) -> None:
        # An abstention recorded as 0.0 would assert irrelevance the judge
        # never claimed.
        assert to_relevance([Judgement("a", 2), Judgement("b", None)]) == {"a": 2.0}

    def test_grades_are_floats(self) -> None:
        assert isinstance(to_relevance([Judgement("a", 2)])["a"], float)


class TestAgreement:
    def test_perfect_agreement(self) -> None:
        result = agreement({"a": 3.0, "b": 0.0}, {"a": 3.0, "b": 0.0})
        assert result.exact == 1.0
        assert result.within_one == 1.0
        assert result.binary == 1.0

    def test_adjacent_grades_count_as_within_one_but_not_exact(self) -> None:
        # Graders argue about 2 versus 3 constantly; that is not the kind of
        # disagreement that invalidates a label set.
        result = agreement({"a": 3.0}, {"a": 2.0})
        assert result.exact == 0.0
        assert result.within_one == 1.0
        assert result.binary == 1.0

    def test_binary_agreement_is_what_recall_depends_on(self) -> None:
        # Judge says relevant, human says not: this one does matter.
        result = agreement({"a": 1.0}, {"a": 0.0})
        assert result.within_one == 1.0
        assert result.binary == 0.0

    def test_only_shared_passages_are_compared(self) -> None:
        result = agreement({"a": 3.0, "b": 1.0}, {"a": 3.0})
        assert result.n == 1
        assert result.exact == 1.0

    def test_no_overlap_reports_nothing_rather_than_perfection(self) -> None:
        result = agreement({"a": 3.0}, {"b": 3.0})
        assert result == Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_reports_both_means_to_expose_systematic_bias(self) -> None:
        # A judge that grades a full point high everywhere still shows perfect
        # within-one agreement; the means are what reveal it.
        result = agreement({"a": 3.0, "b": 2.0}, {"a": 2.0, "b": 1.0})
        assert result.judge_mean == 2.5
        assert result.human_mean == 1.5


class TestSampleForReview:
    def test_returns_at_most_the_requested_size(self) -> None:
        relevance = {f"c{i}": float(i % 4) for i in range(40)}
        assert len(sample_for_review(relevance, size=8)) <= 8

    def test_is_stratified_across_grades(self) -> None:
        # A uniform sample of a mostly-zero label set is mostly zeros, and
        # confirms only that the judge recognises irrelevance.
        relevance = {f"c{i}": 0.0 for i in range(50)} | {"rare3": 3.0, "rare2": 2.0}
        sampled = set(sample_for_review(relevance, size=9))
        assert "rare3" in sampled
        assert "rare2" in sampled

    def test_is_deterministic_for_a_given_seed(self) -> None:
        relevance = {f"c{i}": float(i % 4) for i in range(40)}
        assert sample_for_review(relevance, 8, seed=1) == sample_for_review(relevance, 8, seed=1)

    def test_different_seeds_can_differ(self) -> None:
        relevance = {f"c{i}": float(i % 4) for i in range(40)}
        a = sample_for_review(relevance, 8, seed=1)
        b = sample_for_review(relevance, 8, seed=99)
        assert a != b or len(a) < 8

    def test_empty_label_set_gives_an_empty_sample(self) -> None:
        assert sample_for_review({}, size=10) == []


class TestPassageGradeSchema:
    def test_rejects_a_grade_outside_the_scale(self) -> None:
        # The schema is the guard: a judge returning 7 should fail validation
        # rather than silently enter the label set.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PassageGrade(index=0, grade=7)

    def test_accepts_every_valid_grade(self) -> None:
        for grade in GRADE_DEFINITIONS:
            assert PassageGrade(index=0, grade=grade).grade == grade
