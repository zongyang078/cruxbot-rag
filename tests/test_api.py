"""The HTTP service.

The pipeline is swapped for one built on in-memory doubles, so these tests
exercise routing, schemas, streaming, and degradation without loading an index
or reaching a model.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi", reason="requires the serve extra")
from fastapi.testclient import TestClient  # noqa: E402

from cruxbot import api  # noqa: E402
from cruxbot.pipeline import Pipeline  # noqa: E402
from cruxbot.retrieval.dense import DenseRetriever  # noqa: E402
from cruxbot.retrieval.hybrid import HybridRetriever  # noqa: E402
from tests.fakes import FakeEmbedder, FakeLLM, FakeStore, make_doc  # noqa: E402


def make_pipeline(provider: FakeLLM | None = None) -> Pipeline:
    store = FakeStore(
        [
            make_doc(
                "r1",
                text="Reed's Pinnacle has several 5.11a sport routes.",
                source_url="https://example.com/route/1",
            ),
            make_doc(
                "f1",
                text="Hangboard protocol.",
                content_type="forum_discussion",
                source_url="https://www.mountainproject.com/forum",
            ),
        ]
    )
    return Pipeline(HybridRetriever(DenseRetriever(FakeEmbedder(), store)), provider)


# Distinguishes "caller did not specify a provider" from "caller asked for
# none". A default-constructed instance here would be shared across every test.
_DEFAULT = object()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """A client whose app is already 'started' with a faked pipeline."""

    def _client(provider: FakeLLM | None | object = _DEFAULT):
        if provider is _DEFAULT:
            provider = FakeLLM("Reed's Pinnacle [1].")
        monkeypatch.setattr(api, "_pipeline", make_pipeline(provider))
        return TestClient(api.app)

    return _client


class TestHealth:
    def test_reports_ready(self, client) -> None:
        body = client().get("/health").json()
        assert body["status"] == "ok"
        assert body["retrieval"] is True

    def test_distinguishes_retrieval_from_generation(self, client) -> None:
        # A deployment with no provider is degraded, not down; the difference
        # has to be visible to whatever is watching it.
        body = client(provider=None).get("/health").json()
        assert body["retrieval"] is True
        assert body["generation"] is False

    def test_a_configured_but_unreachable_provider_is_not_healthy(self, client) -> None:
        # Constructing a provider proves nothing: it records a URL. Reporting
        # generation as available on that basis is wrong in exactly the case
        # anyone checks health for.
        body = client(provider=FakeLLM("x", reachable=False)).get("/health").json()
        assert body["generation"] is False
        assert body["retrieval"] is True


class TestSearch:
    def test_returns_ranked_sources(self, client) -> None:
        body = client().post("/search", json={"query": "5.11a routes", "top_k": 2}).json()
        assert [s["chunk_id"] for s in body["sources"]] == ["r1", "f1"]

    def test_reports_per_stage_timing(self, client) -> None:
        timing = client().post("/search", json={"query": "q"}).json()["timing"]
        assert timing["retrieval_ms"] >= 0
        assert timing["generation_ms"] == 0.0

    def test_reports_candidate_counts_per_retriever(self, client) -> None:
        body = client().post("/search", json={"query": "q"}).json()
        assert body["candidates"]["dense"] == 2
        assert body["candidates"]["sparse"] == 0

    def test_flags_a_site_level_url_as_not_specific(self, client) -> None:
        # The corpus has sources with no per-post URL. Presenting those as
        # citations without saying so would mislead the caller.
        body = client().post("/search", json={"query": "q", "top_k": 2}).json()
        by_id = {s["chunk_id"]: s for s in body["sources"]}
        assert by_id["r1"]["url_is_specific"] is True
        assert by_id["f1"]["url_is_specific"] is False

    def test_works_without_a_provider(self, client) -> None:
        # Retrieval is the endpoint that costs nothing; it must not depend on
        # a generator being configured.
        assert client(provider=None).post("/search", json={"query": "q"}).status_code == 200

    def test_rejects_an_empty_query(self, client) -> None:
        assert client().post("/search", json={"query": ""}).status_code == 422

    def test_rejects_an_out_of_range_top_k(self, client) -> None:
        assert client().post("/search", json={"query": "q", "top_k": 99}).status_code == 422

    def test_returns_503_before_the_index_finishes_loading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api, "_pipeline", None)
        response = TestClient(api.app).post("/search", json={"query": "q"})
        assert response.status_code == 503
        assert "loading" in response.json()["detail"]


class TestAnswer:
    def test_non_streaming_returns_the_whole_answer(self, client) -> None:
        body = client().post("/answer", json={"query": "q", "stream": False}).json()
        assert body["answer"] == "Reed's Pinnacle [1]."
        assert body["sources"]

    def test_non_streaming_retrieves_only_once(self, client, monkeypatch) -> None:
        # An earlier version called answer() and search() separately, paying
        # for retrieval twice on every non-streaming request.
        pipeline = make_pipeline(FakeLLM("answer"))
        calls: list[str] = []
        original = pipeline.search

        def counting(*args, **kwargs):
            calls.append("search")
            return original(*args, **kwargs)

        monkeypatch.setattr(pipeline, "search", counting)
        monkeypatch.setattr(api, "_pipeline", pipeline)
        TestClient(api.app).post("/answer", json={"query": "q", "stream": False})
        assert len(calls) == 1

    def test_streaming_sends_sources_before_tokens(self, client) -> None:
        # A client should be able to render citations while the answer is still
        # being written.
        with client().stream("POST", "/answer", json={"query": "q"}) as response:
            events = [
                line[len("event: ") :]
                for line in response.iter_lines()
                if line.startswith("event: ")
            ]
        assert events[0] == "sources"
        assert "token" in events
        assert events[-1] == "done"

    def test_streaming_tokens_reassemble_into_the_answer(self, client) -> None:
        # Parsed as SSE rather than by grepping the payload: the sources event
        # also carries "text" fields, and matching on those would silently
        # concatenate passage bodies into the answer.
        tokens: list[str] = []
        event = ""
        with client().stream("POST", "/answer", json={"query": "q"}) as response:
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[len("event: ") :]
                elif line.startswith("data: ") and event == "token":
                    tokens.append(json.loads(line[len("data: ") :])["text"])

        assert "".join(tokens).strip() == "Reed's Pinnacle [1]."

    def test_without_a_provider_it_returns_503_and_says_what_still_works(self, client) -> None:
        response = client(provider=None).post("/answer", json={"query": "q"})
        assert response.status_code == 503
        assert "/search still works" in response.json()["detail"]
