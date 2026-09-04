"""Ollama backend: a locally hosted model, spoken to over its HTTP API.

Uses the stdlib rather than a vendor SDK -- the surface is two endpoints, and
avoiding the dependency keeps the container image small enough to matter on a
GPU VM where the model weights already dominate.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from cruxbot.config import settings
from cruxbot.llm.base import ProviderError


class OllamaProvider:
    """Generate text with a model served by a local Ollama instance."""

    name = "ollama"

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = url or settings.ollama_url
        self.model = model or settings.ollama_model

    def _request(self, prompt: str, *, stream: bool, timeout: float):
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": stream}).encode()
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self.url}: {exc.reason}. Is `ollama serve` running?"
            ) from exc

    def complete(self, prompt: str, *, timeout: float = 120.0) -> str:
        with self._request(prompt, stream=False, timeout=timeout) as response:
            try:
                body = json.loads(response.read())
            except json.JSONDecodeError as exc:
                raise ProviderError("Ollama returned a non-JSON response") from exc
        if "response" not in body:
            raise ProviderError(f"Unexpected Ollama response shape: {sorted(body)}")
        return str(body["response"]).strip()

    def stream(self, prompt: str, *, timeout: float = 120.0) -> Iterator[str]:
        with self._request(prompt, stream=True, timeout=timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Ollama emits one JSON object per line; a partial line at a
                    # chunk boundary is recoverable, so skip rather than abort.
                    continue
                if token := event.get("response"):
                    yield token
                if event.get("done"):
                    return
