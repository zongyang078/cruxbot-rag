"""BM25 keyword retrieval over an inverted index.

Why not `rank_bm25`: its `get_scores` computes a score for every document in
the corpus on every query. At 382k documents that is 382k floating point
operations per query term regardless of how many documents actually contain
the term -- and for a corpus this size, a typical query term appears in well
under 1% of documents.

An inverted index makes query cost proportional to the length of the postings
lists actually touched. The implementation below uses only the standard
library, which keeps the module testable in CI without installing torch.

BM25 (Robertson & Walker, SIGIR 1994):

    score(d, q) = sum over t in q of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Term frequency saturation. Above k1, additional occurrences of a term in one
# document add progressively less. 1.5 is the usual default.
K1 = 1.5
# Length normalization. 0 disables it; 1 fully normalizes by document length.
B = 0.75

# Bump when the tokenizer or index layout changes, so stale caches are rejected
# rather than silently loaded -- the original implementation had no such guard
# and would happily load an index built against a different corpus.
INDEX_VERSION = 1

# Keep alphanumerics, plus internal dots and hyphens so that climbing grades
# ("5.10a") and hyphenated terms ("multi-pitch") survive tokenization intact.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into BM25 terms."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class Posting:
    """One document's occurrence count for one term."""

    doc_index: int
    term_frequency: int


class BM25Index:
    """An inverted index supporting BM25 ranking with optional filtering.

    Documents are addressed by an integer position assigned at build time;
    `doc_ids` maps those positions back to the caller's identifiers.
    """

    __slots__ = ("postings", "doc_ids", "doc_lengths", "doc_meta", "avgdl", "n_docs")

    def __init__(self) -> None:
        self.postings: dict[str, list[Posting]] = {}
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.doc_meta: list[dict[str, str]] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, str]] | None = None,
    ) -> BM25Index:
        """Build an index from parallel sequences of ids, texts, and metadata."""
        if len(doc_ids) != len(texts):
            raise ValueError(f"{len(doc_ids)} ids but {len(texts)} texts")
        if metadatas is not None and len(metadatas) != len(doc_ids):
            raise ValueError(f"{len(doc_ids)} ids but {len(metadatas)} metadatas")

        index = cls()
        index.doc_ids = list(doc_ids)
        index.doc_meta = list(metadatas) if metadatas is not None else [{} for _ in doc_ids]

        accumulator: dict[str, list[Posting]] = defaultdict(list)
        total_length = 0

        for position, text in enumerate(texts):
            tokens = tokenize(text)
            index.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, frequency in counts.items():
                accumulator[term].append(Posting(position, frequency))

        index.postings = dict(accumulator)
        index.n_docs = len(index.doc_ids)
        index.avgdl = total_length / index.n_docs if index.n_docs else 0.0
        return index

    # -- querying ----------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Inverse document frequency, in the smoothed BM25+ form.

        The `1 +` keeps the value positive for terms appearing in more than
        half the corpus; the unsmoothed form goes negative there, which would
        let a common term actively penalize a document that contains it.
        """
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, allowed: set[int] | None = None) -> dict[int, float]:
        """Score every document containing at least one query term.

        Args:
            query: Raw query text; tokenized with the same tokenizer as the corpus.
            allowed: If given, only these document positions are scored. Passing
                the filter down here rather than filtering afterwards means a
                content-type-restricted query does not pay to score the rest of
                the corpus.

        Returns:
            Document position -> score, for scoring documents only. Documents
            sharing no term with the query are absent rather than scored zero.
        """
        scores: dict[int, float] = defaultdict(float)

        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for posting in postings:
                if allowed is not None and posting.doc_index not in allowed:
                    continue
                tf = posting.term_frequency
                length_norm = 1 - B + B * self.doc_lengths[posting.doc_index] / self.avgdl
                scores[posting.doc_index] += idf * tf * (K1 + 1) / (tf + K1 * length_norm)

        return dict(scores)

    def search(
        self,
        query: str,
        top_k: int = 20,
        content_types: Iterable[str] | None = None,
    ) -> list[dict[str, object]]:
        """Return the top_k highest scoring documents, best first.

        Ties are broken by document position so results are reproducible.
        """
        allowed = self._positions_for_content_types(content_types)
        if allowed is not None and not allowed:
            return []

        scores = self.score(query, allowed=allowed)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

        return [
            {
                "chunk_id": self.doc_ids[position],
                "score": score,
                "metadata": self.doc_meta[position],
            }
            for position, score in ranked
        ]

    def _positions_for_content_types(self, content_types: Iterable[str] | None) -> set[int] | None:
        if content_types is None:
            return None
        wanted = set(content_types)
        return {
            position
            for position, meta in enumerate(self.doc_meta)
            if meta.get("content_type") in wanted
        }

    # -- persistence -------------------------------------------------------

    def fingerprint(self) -> tuple[int, int, str]:
        """Identify the corpus this index was built from.

        Compared on load so that an index built against a different or
        re-generated corpus is rebuilt instead of silently reused.
        """
        first = self.doc_ids[0] if self.doc_ids else ""
        last = self.doc_ids[-1] if self.doc_ids else ""
        return (INDEX_VERSION, self.n_docs, f"{first}|{last}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"fingerprint": self.fingerprint(), "index": self}, handle)

    @classmethod
    def load(cls, path: str | Path, expect: tuple[int, int, str] | None = None) -> BM25Index | None:
        """Load a cached index, or None if it is missing, corrupt, or stale.

        Returning None rather than raising lets the caller treat a stale cache
        as a cache miss and rebuild.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            index = payload["index"]
        except Exception:
            return None
        if not isinstance(index, cls):
            return None
        if expect is not None and payload.get("fingerprint") != expect:
            return None
        return index
