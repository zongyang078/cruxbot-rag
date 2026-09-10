"""LLM provider selection and the Anthropic backend.

No test here reaches the network. The Anthropic SDK client is replaced with a
stub, which is enough to cover the parts that are this project's code: request
shape, refusal handling, error translation, and reachability.
"""

from __future__ import annotations

import pytest

from cruxbot.llm.base import ProviderError, get_provider


class StubBlock:
    def __init__(self, text: str, block_type: str = "text") -> None:
        self.text = text
        self.type = block_type


class StubResponse:
    def __init__(self, text: str = "an answer", stop_reason: str = "end_turn") -> None:
        self.content = [StubBlock(text)]
        self.stop_reason = stop_reason


class StubMessages:
    def __init__(self, response: StubResponse, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, response: StubResponse, error: Exception | None = None) -> None:
        self.messages = StubMessages(response, error)
        self.models = self

    def with_options(self, **kwargs):
        return self

    def retrieve(self, model_id: str):
        return {"id": model_id}


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch):
    """An AnthropicProvider whose SDK client is a stub."""
    pytest.importorskip("anthropic")
    from cruxbot.llm.anthropic import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def _provider(response: StubResponse | None = None, error: Exception | None = None):
        instance = AnthropicProvider()
        instance._client = StubClient(response or StubResponse(), error)
        return instance

    return _provider


class TestGetProvider:
    def test_resolves_ollama(self) -> None:
        assert get_provider("ollama").name == "ollama"

    def test_resolves_anthropic_by_either_name(self, monkeypatch) -> None:
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert get_provider("anthropic").name == get_provider("claude").name

    def test_is_case_insensitive(self) -> None:
        assert get_provider("OLLAMA").name == "ollama"

    def test_an_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(ProviderError, match="'ollama', 'anthropic'"):
            get_provider("gpt-9")


class TestAnthropicProvider:
    def test_requires_a_key_and_says_what_to_do_instead(self, monkeypatch) -> None:
        pytest.importorskip("anthropic")
        from cruxbot.llm.anthropic import AnthropicProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="CRUXBOT_LLM_PROVIDER=ollama"):
            AnthropicProvider()

    def test_model_is_configurable_by_environment(self, monkeypatch) -> None:
        pytest.importorskip("anthropic")
        from cruxbot.llm.anthropic import AnthropicProvider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("CRUXBOT_ANTHROPIC_MODEL", "claude-haiku-4-5")
        assert AnthropicProvider().name == "claude-haiku-4-5"

    def test_complete_returns_the_text(self, provider) -> None:
        assert provider(StubResponse("Remain in Light [1].")).complete("q") == (
            "Remain in Light [1]."
        )

    def test_request_carries_the_prompt_and_effort(self, provider) -> None:
        instance = provider()
        instance.complete("what routes?")
        call = instance._client.messages.calls[0]
        assert call["messages"] == [{"role": "user", "content": "what routes?"}]
        # Answering from supplied context is not a reasoning task; deliberating
        # over it is latency the caller pays for.
        assert call["output_config"] == {"effort": "low"}

    def test_a_refusal_is_raised_rather_than_returned_as_text(self, provider) -> None:
        # A safety decline arrives as a normal 200. Returning its content as an
        # answer would present a refusal as though it were grounded output.
        with pytest.raises(ProviderError, match="declined"):
            provider(StubResponse("", stop_reason="refusal")).complete("q")

    def test_api_errors_become_provider_errors(self, provider) -> None:
        # The service maps ProviderError to a 503; an SDK exception escaping
        # here would surface as a 500 and read as a bug in this service.
        import anthropic

        error = anthropic.APIError("boom", request=None, body=None)  # type: ignore[arg-type]
        with pytest.raises(ProviderError, match="Anthropic request failed"):
            provider(error=error).complete("q")

    def test_available_is_true_when_the_model_resolves(self, provider) -> None:
        assert provider().available() is True

    def test_available_is_false_when_the_client_raises(self, provider, monkeypatch) -> None:
        instance = provider()
        monkeypatch.setattr(
            instance._client, "retrieve", lambda model_id: (_ for _ in ()).throw(OSError("down"))
        )
        assert instance.available() is False
