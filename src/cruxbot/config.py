"""Runtime configuration, read from the environment.

Every path and endpoint the pipeline touches is resolved here rather than
hardcoded at its use site -- the original version had "data/chroma" written as
a default argument in four separate modules, which meant the Docker image and
the test suite disagreed about where the index lived.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    path = Path(raw)
    return path if path.is_absolute() else _REPO_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration for one process."""

    # Storage
    chroma_path: Path = field(
        default_factory=lambda: _env_path("CRUXBOT_CHROMA_PATH", "data/chroma")
    )
    collection: str = field(default_factory=lambda: os.getenv("CRUXBOT_COLLECTION", "cruxbot"))
    bm25_cache: Path = field(
        default_factory=lambda: _env_path("CRUXBOT_BM25_CACHE", "data/bm25.pkl")
    )

    # Models
    # bge-small rather than bge-base: under a two-stage architecture the first
    # stage is judged on Recall@50, not Precision@5, and the cross-encoder
    # recovers the ordering. Measured on an M4, small indexes 384k chunks at
    # 238/s against base's 91/s -- 27 minutes versus 67 -- for a difference the
    # reranker is expected to absorb. The retrieval benchmark decides whether
    # that expectation holds.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("CRUXBOT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("CRUXBOT_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )

    # Generation
    llm_provider: str = field(default_factory=lambda: os.getenv("CRUXBOT_LLM_PROVIDER", "ollama"))
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))

    # Retrieval defaults
    top_k: int = field(default_factory=lambda: int(os.getenv("CRUXBOT_TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(os.getenv("CRUXBOT_CANDIDATE_K", "50")))


settings = Settings()
