"""End-to-end orchestration, with every dependency faked."""

from __future__ import annotations

import pytest

from cruxbot.pipeline import Pipeline
from cruxbot.retrieval.dense import DenseRetriever
from cruxbot.retrieval.hybrid import HybridRetriever
from tests.fakes import FakeEmbedder, FakeLLM, FakeStore, make_doc


def pipeline(provider: FakeLLM | None = None, docs: list | None = None) -> Pipeline:
    store = FakeStore(
        docs
        if docs is not None
        else [
            make_doc(
                "r1",
                text="Reed's Pinnacle has several 5.11a sport routes.",
                source_url="https://example.com/route/1",
            ),
            make_doc("r2", text="The Lombard Classic is a 5.11a sport climb."),
        ]
    )
    return Pipeline(HybridRetriever(DenseRetriever(FakeEmbedder(), store)), provider)


class TestSearch:
    def test_returns_chunks(self) -> None:
        assert len(pipeline().search("5.11a in Yosemite", top_k=2).chunks) == 2

    def test_needs_no_provider(self) -> None:
        # Retrieval is the endpoint that costs nothing to serve; requiring a
        # provider to start would make a free deployment impossible.
        assert pipeline(provider=None).search("q").chunks

    def test_breaks_timing_down_by_stage(self) -> None:
        # One total figure cannot distinguish slow retrieval from slow
        # generation, and the two call for different work.
        result = pipeline().search("q")
        assert result.retrieval_ms >= 0
        assert result.rerank_ms == 0.0

    def test_reports_candidate_counts(self) -> None:
        result = pipeline().search("q")
        assert result.n_dense == 2
        assert result.n_sparse == 0

    def test_exposes_citable_sources(self) -> None:
        sources = pipeline().search("q", top_k=1).sources
        assert sources[0].url == "https://example.com/route/1"
        assert sources[0].is_specific is True

    def test_top_k_overrides_the_default(self) -> None:
        assert len(pipeline().search("q", top_k=1).chunks) == 1

    def test_empty_corpus_returns_nothing(self) -> None:
        assert pipeline(docs=[]).search("q").chunks == []


class TestAnswer:
    def test_generates_from_retrieved_context(self) -> None:
        provider = FakeLLM("Reed's Pinnacle has 5.11a routes [1].")
        result = pipeline(provider).answer("5.11a in Yosemite")
        assert result.text == "Reed's Pinnacle has 5.11a routes [1]."

    def test_prompt_carries_the_retrieved_passages(self) -> None:
        provider = FakeLLM("answer")
        pipeline(provider).answer("q")
        assert "Reed's Pinnacle" in provider.prompts[0]

    def test_prompt_carries_the_question(self) -> None:
        provider = FakeLLM("answer")
        pipeline(provider).answer("how do I train?")
        assert "Question: how do I train?" in provider.prompts[0]

    def test_returns_sources_alongside_the_text(self) -> None:
        result = pipeline(FakeLLM("answer")).answer("q", top_k=1)
        assert result.sources[0].url == "https://example.com/route/1"

    def test_separates_retrieval_from_generation_time(self) -> None:
        result = pipeline(FakeLLM("answer")).answer("q")
        assert result.latency_ms == pytest.approx(result.retrieval_ms + result.generation_ms)

    def test_without_a_provider_it_fails_with_a_usable_message(self) -> None:
        with pytest.raises(RuntimeError, match="can only serve retrieval"):
            pipeline(provider=None).answer("q")


class TestAnswerStream:
    def test_yields_the_tokens(self) -> None:
        _, stream = pipeline(FakeLLM("one two three")).answer_stream("q")
        assert "".join(stream) == "one two three"

    def test_sources_are_available_before_generation_starts(self) -> None:
        # A caller should be able to render citations while the answer is still
        # being written, and a stalled generator must not withhold results that
        # are already in hand. The stream is lazy: the provider is not touched
        # until the first token is pulled.
        provider = FakeLLM("answer")
        retrieved, stream = pipeline(provider).answer_stream("q", top_k=1)

        assert retrieved.sources[0].url == "https://example.com/route/1"
        assert provider.stream_calls == 0

        assert next(iter(stream)) == "answer"
        assert provider.stream_calls == 1

    def test_without_a_provider_it_fails_before_retrieving(self) -> None:
        with pytest.raises(RuntimeError, match="can only serve retrieval"):
            pipeline(provider=None).answer_stream("q")
