"""Source URL quality checks.

Citation quality was the weakest dimension in the original evaluation (2.78/5).
Most of that is a data limitation, but part of it was this check letting
site-level URLs through as if they were real citations.
"""

from __future__ import annotations

import pytest

from cruxbot import urls


class TestIsSpecific:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.mountainproject.com/route/110149834/access-denied",
            "https://www.reddit.com/r/climbing/comments/abc123/title/",
            "https://publications.americanalpineclub.org/articles/12345",
            "https://openbeta.io/climb/some-route-uuid",
        ],
    )
    def test_document_level_urls_are_specific(self, url: str) -> None:
        assert urls.is_specific(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.mountainproject.com/forum",
            "https://www.mountainproject.com",
            "https://www.reddit.com",
            "https://openbeta.io",
        ],
    )
    def test_site_level_urls_are_not_specific(self, url: str) -> None:
        assert urls.is_specific(url) is False

    @pytest.mark.parametrize("url", ["", "   ", None])
    def test_missing_url_is_not_specific(self, url: str | None) -> None:
        assert urls.is_specific(url or "") is False

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.mountainproject.com/forum/",
            "http://www.mountainproject.com/forum",
            "https://www.mountainproject.com/forum?page=2",
            "  https://www.mountainproject.com/forum  ",
            "https://WWW.MountainProject.com/forum",
        ],
    )
    def test_generic_urls_are_caught_despite_surface_variation(self, url: str) -> None:
        # A raw string denylist misses every one of these, so the pipeline
        # reports a site-level link as a genuine citation.
        assert urls.is_specific(url) is False


class TestCanonical:
    def test_strips_scheme_and_trailing_slash(self) -> None:
        assert urls.canonical("https://example.com/a/") == "example.com/a"

    def test_lowercases_host_but_preserves_path_case(self) -> None:
        # Hosts are case-insensitive; paths are not.
        assert urls.canonical("https://EXAMPLE.com/Path") == "example.com/Path"

    def test_drops_query_and_fragment(self) -> None:
        assert urls.canonical("https://example.com/a?b=1#frag") == "example.com/a"

    def test_handles_url_without_a_scheme(self) -> None:
        assert urls.canonical("example.com/a") == "example.com/a"

    def test_empty_input_gives_empty_output(self) -> None:
        assert urls.canonical("") == ""
