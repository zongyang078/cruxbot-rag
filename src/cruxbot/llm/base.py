"""LLM provider interface.

The pipeline talks to this Protocol, never to a vendor SDK directly. Two
reasons, both of which showed up in the original version of this project:

1. The demo and the local development environment want different backends. A
   self-hosted Llama 3 on a GPU VM is the right answer for privacy and for
   zero marginal cost per query; a hosted API is the right answer for a public
   demo that must not require a GPU to try.
2. Evaluation needs to hold the retriever fixed while swapping the generator.
   That is only cheap if the generator is behind one seam.

Implementations live in sibling modules and are resolved by `get_provider`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal surface the pipeline needs from a text generator."""

    name: str

    def complete(self, prompt: str, *, timeout: float = 120.0) -> str:
        """Return the full completion. Used by batch evaluation."""
        ...

    def stream(self, prompt: str, *, timeout: float = 120.0) -> Iterator[str]:
        """Yield completion text incrementally. Used by the UI."""
        ...

    def available(self, timeout: float = 2.0) -> bool:
        """Whether the backend can actually be reached right now.

        Constructing a provider proves nothing -- it only records a URL and a
        model name. A health endpoint that reports "generation: true" on that
        basis is wrong in the one case anybody checks it for.
        """
        ...


class ProviderError(RuntimeError):
    """Raised when a backend is unreachable or returns an unusable response."""


def get_provider(name: str | None = None, **kwargs: object) -> LLMProvider:
    """Construct a provider by name.

    Defaults to the value of `CRUXBOT_LLM_PROVIDER`, else "ollama".
    """
    from cruxbot.config import settings

    resolved = (name or settings.llm_provider).lower()

    if resolved == "ollama":
        from cruxbot.llm.ollama import OllamaProvider

        return OllamaProvider(**kwargs)  # type: ignore[arg-type]

    if resolved in {"anthropic", "claude"}:
        from cruxbot.llm.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)  # type: ignore[arg-type]

    raise ProviderError(f"Unknown LLM provider {resolved!r}. Available: 'ollama', 'anthropic'.")
