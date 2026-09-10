"""Anthropic backend: a hosted model, reached over the API.

The counterpart to the Ollama backend, behind the same Protocol. Having both
is not redundancy for its own sake:

- A self-hosted Llama 3 costs nothing per query and sends nothing to a third
  party, which is the right answer for bulk evaluation and for anyone running
  this on their own machine.
- A hosted model needs no GPU and no local install, which is the only answer
  for a deployed service. "The user has Ollama running" is not a deployment
  story.

Evaluation can hold the retriever fixed and swap only this, which is what makes
a generator comparison mean anything.

Requires the eval or serve extra:  pip install -e ".[eval]"
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from cruxbot.llm.base import ProviderError

DEFAULT_MODEL = "claude-opus-5"

# A grounded answer over five retrieved passages is deliberately short -- it
# should cite, not expound. Thinking tokens count against this budget, so the
# ceiling is set well above the expected answer length rather than at it.
DEFAULT_MAX_TOKENS = 4096

# Answering from supplied context is not a reasoning problem. Low effort keeps
# the model from deliberating over passages it was handed, which is latency the
# user pays for and tokens the operator pays for.
DEFAULT_EFFORT = "low"


class AnthropicProvider:
    """Generate text with a hosted Claude model."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        import anthropic

        self.name = model or os.getenv("CRUXBOT_ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.effort = effort

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Set it, or use the Ollama provider "
                "with CRUXBOT_LLM_PROVIDER=ollama."
            )
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key)

    def _request_kwargs(self, prompt: str) -> dict[str, object]:
        return {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": prompt}],
        }

    def available(self, timeout: float = 2.0) -> bool:
        """Check the credential and the model, without generating anything.

        Listing models costs no tokens, where a probe generation would spend
        them every time something checked health.
        """
        try:
            self._client.with_options(timeout=timeout, max_retries=0).models.retrieve(self.name)
        except Exception:  # noqa: BLE001 - any failure means "not usable right now"
            return False
        return True

    def complete(self, prompt: str, *, timeout: float = 120.0) -> str:
        client = self._client.with_options(timeout=timeout)
        try:
            response = client.messages.create(**self._request_kwargs(prompt))
        except self._anthropic.APIError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        # A safety decline arrives as a normal 200 with a refusal stop reason,
        # so it has to be checked before reading content rather than caught.
        if response.stop_reason == "refusal":
            raise ProviderError("The model declined to answer this request.")

        return "".join(block.text for block in response.content if block.type == "text").strip()

    def stream(self, prompt: str, *, timeout: float = 120.0) -> Iterator[str]:
        client = self._client.with_options(timeout=timeout)
        try:
            with client.messages.stream(**self._request_kwargs(prompt)) as stream:
                yield from stream.text_stream
                if stream.get_final_message().stop_reason == "refusal":
                    raise ProviderError("The model declined to answer this request.")
        except self._anthropic.APIError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
