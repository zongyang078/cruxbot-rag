"""Grade normalization: the correctness of route retrieval rests on this."""

from __future__ import annotations

import pytest

from cruxbot import grades


class TestExtractGrade:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Recommend a 5.10 sport climbing route in California", "5.10"),
            ("Find me a 5.8 sport route in Joshua Tree", "5.8"),
            ("What are some classic 5.12a routes?", "5.12a"),
            ("Best bouldering problems around V4 in Bishop", "V4"),
            ("Anything at V10 nearby?", "V10"),
            ("Looking for 7a+ sport routes", "7a+"),
            ("I want to project 6b this season", "6b"),
        ],
    )
    def test_finds_grade(self, query: str, expected: str) -> None:
        assert grades.extract_grade(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "How should I train finger strength?",
            "What is the best belay device?",
            "",
            "Tell me about Yosemite",
        ],
    )
    def test_returns_none_without_a_grade(self, query: str) -> None:
        assert grades.extract_grade(query) is None

    def test_yds_takes_precedence_over_french(self) -> None:
        # "5.10a" contains no French grade, but a naive pattern order could
        # read the trailing "0a" or similar. YDS must win.
        assert grades.extract_grade("a 5.10a route") == "5.10a"

    def test_v_scale_normalizes_to_uppercase(self) -> None:
        assert grades.extract_grade("something around v7") == "V7"


class TestEquivalents:
    def test_always_includes_input(self) -> None:
        for grade in ("5.11a", "V4", "7a", "5.9"):
            assert grade in grades.equivalents(grade)

    def test_yds_maps_to_french(self) -> None:
        assert "6c" in grades.equivalents("5.11a")

    def test_yds_includes_its_base(self) -> None:
        assert "5.11" in grades.equivalents("5.11a")

    def test_bare_yds_fans_out_to_letter_grades(self) -> None:
        result = grades.equivalents("5.10")
        for suffix in ("a", "b", "c", "d"):
            assert f"5.10{suffix}" in result

    def test_bare_yds_below_ten_does_not_fan_out(self) -> None:
        # 5.9 has no letter subdivision in common usage.
        result = grades.equivalents("5.9")
        assert not any(g.startswith("5.9") and g != "5.9" for g in result)

    def test_v_scale_maps_to_font(self) -> None:
        assert "6B" in grades.equivalents("V4")

    def test_french_maps_back_to_yds(self) -> None:
        assert "5.11a" in grades.equivalents("6c")

    def test_ambiguous_french_returns_all_yds_grades(self) -> None:
        # Both 5.11d and 5.12a are cited as ~7a+. A lossy dict inversion drops
        # one of them; retrieval then misses half the relevant routes.
        result = grades.equivalents("7a+")
        assert "5.11d" in result
        assert "5.12a" in result

    def test_lookup_is_case_insensitive(self) -> None:
        assert grades.equivalents("6A") == grades.equivalents("6a")
        assert grades.equivalents("v4") == grades.equivalents("V4")

    def test_unknown_grade_degrades_to_itself(self) -> None:
        assert grades.equivalents("5.15d") == ["5.15", "5.15d"]

    def test_output_is_sorted_and_deduplicated(self) -> None:
        result = grades.equivalents("5.10")
        assert result == sorted(result)
        assert len(result) == len(set(result))


class TestExpandQuery:
    def test_appends_equivalents(self) -> None:
        expanded = grades.expand_query("Find me a 5.11a route")
        assert expanded.startswith("Find me a 5.11a route")
        assert "6c" in expanded

    def test_unchanged_without_a_grade(self) -> None:
        query = "How do I train finger strength?"
        assert grades.expand_query(query) == query

    def test_does_not_repeat_a_grade_already_present(self) -> None:
        expanded = grades.expand_query("5.11a or 6c please")
        assert expanded.count("6c") == 1
