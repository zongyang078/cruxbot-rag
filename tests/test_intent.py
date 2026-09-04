"""Query intent detection, including the cases where it is known to be wrong."""

from __future__ import annotations

import pytest

from cruxbot import intent


class TestDetect:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Recommend a multi-pitch route in Yosemite", intent.ROUTE),
            ("What are some classic routes in Red River Gorge?", intent.ROUTE),
            ("How should I train finger strength for climbing?", intent.FORUM),
            ("What is the best hangboard routine?", intent.FORUM),
            ("What causes rappelling accidents?", intent.ARTICLE),
            ("Common belaying mistakes that lead to accidents", intent.ARTICLE),
            ("La Sportiva vs Scarpa climbing shoes", intent.GEAR),
            ("What crash pad should I buy?", intent.GEAR),
        ],
    )
    def test_classifies_clear_queries(self, query: str, expected: str) -> None:
        assert intent.detect(query) == expected

    @pytest.mark.parametrize(
        "query",
        ["", "hello", "What is the stock price of Apple today?"],
    )
    def test_returns_none_when_nothing_matches(self, query: str) -> None:
        assert intent.detect(query) is None

    def test_returns_none_when_two_intents_are_comparably_strong(self) -> None:
        # Deliberately mixes route and gear vocabulary. Narrowing the candidate
        # pool on a coin flip is worse than not narrowing it at all.
        assert intent.detect("What rope and harness for a trad route?") is None

    def test_is_case_insensitive(self) -> None:
        assert intent.detect("RECOMMEND A ROUTE IN YOSEMITE") == intent.ROUTE

    def test_word_boundaries_are_respected(self) -> None:
        # "climb" must not match inside "climbing" alone -- otherwise every
        # query in this corpus scores as a route query.
        scores = intent.score("climbing")
        assert scores[intent.ROUTE] == 0


class TestKnownLimitations:
    """Failure modes we accept, measure, and report -- rather than hide.

    These are pinned as tests so that a future change to the classifier makes
    the change in behaviour visible in the diff.
    """

    def test_out_of_domain_query_with_route_keyword_is_misrouted(self) -> None:
        # The H06 case from the original evaluation. "Mars" is a planet, but
        # "routes" is a route keyword, so the rule-based classifier commits.
        # Surface-keyword matching cannot fix this; semantic classification can.
        assert intent.detect("What climbing routes are on Mars?") == intent.ROUTE

    def test_place_names_outside_the_keyword_list_are_invisible(self) -> None:
        # The location list is hand-curated, so a real crag that is not on it
        # contributes no route signal.
        assert intent.score("Anything good in Ceuse?")[intent.ROUTE] == 0


class TestScore:
    def test_exposes_per_type_counts(self) -> None:
        scores = intent.score("How should I train for a multi-pitch route?")
        assert scores[intent.FORUM] > 0
        assert scores[intent.ROUTE] > 0

    def test_covers_every_classifiable_type(self) -> None:
        keys = set(intent.score("anything"))
        assert keys == {intent.ROUTE, intent.FORUM, intent.ARTICLE, intent.GEAR}


class TestContentTypesFor:
    def test_none_intent_means_no_filter(self) -> None:
        assert intent.content_types_for(None) is None

    def test_route_stays_narrow(self) -> None:
        assert intent.content_types_for(intent.ROUTE) == [intent.ROUTE]

    @pytest.mark.parametrize(
        ("detected", "also_expected"),
        [
            (intent.FORUM, intent.REDDIT),
            (intent.GEAR, intent.FORUM),
            (intent.ARTICLE, intent.FORUM),
        ],
    )
    def test_soft_intents_widen_to_adjacent_types(self, detected: str, also_expected: str) -> None:
        # Training advice lives in Reddit threads as much as in MP forums;
        # a hard filter on one type loses real answers.
        types = intent.content_types_for(detected)
        assert types is not None
        assert detected in types
        assert also_expected in types
