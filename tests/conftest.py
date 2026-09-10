"""Fixtures built on the in-memory doubles in `tests.fakes`."""

from __future__ import annotations

import pytest

from tests.fakes import FakeEmbedder, FakeStore, make_doc


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore(
        [
            make_doc("r1", content_type="route", source_url="https://example.com/route/1"),
            make_doc("r2", content_type="route"),
            make_doc("f1", content_type="forum_discussion"),
            make_doc("a1", content_type="article"),
        ]
    )
