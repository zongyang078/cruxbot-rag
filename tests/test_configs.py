"""The ablation set must state its own variables, not inherit them.

Every config in `build_all` differs from its neighbour in exactly one variable,
which only holds if each one sets that variable explicitly. A config that reads
a production default instead becomes a duplicate of its neighbour the moment
the default changes -- and reports the variable's effect as zero rather than
failing.
"""

from __future__ import annotations

from typing import Any

import pytest

from cruxbot.evaluation.configs import build_all


class RecordingHybrid:
    """Stands in for HybridRetriever and records the kwargs each spec passes."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *_args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    def retrieve(self, _query: str, **kwargs: Any) -> Any:
        RecordingHybrid.calls.append({**self.init_kwargs, **kwargs})
        return type("Result", (), {"chunks": []})()


class StubBM25:
    """`sparse` bypasses HybridRetriever and hits the index directly."""

    def search(self, _query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return []


@pytest.fixture
def intent_flags(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool | None]:
    """Map each config name to the `use_intent` it passes, or None if omitted."""
    import cruxbot.retrieval.hybrid as hybrid_module

    monkeypatch.setattr(hybrid_module, "HybridRetriever", RecordingHybrid)

    flags: dict[str, bool | None] = {}
    for spec in build_all(dense=object(), bm25=StubBM25(), reranker=object()):
        RecordingHybrid.calls = []
        spec.build({})("a query", 10)
        if RecordingHybrid.calls:
            flags[spec.name] = RecordingHybrid.calls[0].get("use_intent")
    return flags


class TestIntentIsExplicit:
    def test_intent_configs_turn_the_prior_on(self, intent_flags: dict[str, bool | None]) -> None:
        assert intent_flags["hybrid+intent"] is True
        assert intent_flags["hybrid+intent+rerank"] is True

    def test_non_intent_configs_turn_the_prior_off(
        self, intent_flags: dict[str, bool | None]
    ) -> None:
        assert intent_flags["hybrid"] is False
        assert intent_flags["hybrid+rerank"] is False
        assert intent_flags["dense"] is False

    def test_no_config_relies_on_the_default(self, intent_flags: dict[str, bool | None]) -> None:
        # The regression this file exists for: `use_intent` defaulted to False
        # when intent was turned off in production, which silently made
        # hybrid+intent a copy of hybrid.
        omitted = sorted(name for name, flag in intent_flags.items() if flag is None)
        assert not omitted, f"configs leaving use_intent to the default: {omitted}"
