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
from array import array
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

# Term frequency saturation. Above k1, additional occurrences of a term in one
# document add progressively less. 1.5 is the usual default.
K1 = 1.5
# Length normalization. 0 disables it; 1 fully normalizes by document length.
B = 0.75

# Bump when the tokenizer or index layout changes, so stale caches are rejected
# rather than silently loaded -- the original implementation had no such guard
# and would happily load an index built against a different corpus.
INDEX_VERSION = 2

# Terms appearing in more than this fraction of the corpus are dropped at build
# time. Their IDF is already near zero, so they barely affect ranking, but they
# own the longest postings lists: in this corpus "a", "in", "is", "the" and
# "climbing" alone account for millions of postings. Pruning by observed
# document frequency rather than a hardcoded stopword list keeps the rule
# corpus-driven -- "climbing" is a stopword here and a keyword elsewhere.
MAX_DF_RATIO = 0.5

# Document frequency is only meaningful once there are enough documents to
# estimate it from. Below this, pruning is disabled -- otherwise a ten-document
# corpus loses every term shared by more than five of them, which is most of
# the vocabulary that matters.
MIN_DOCS_FOR_PRUNING = 1000

# Keep alphanumerics, plus internal dots and hyphens so that climbing grades
# ("5.10a") and hyphenated terms ("multi-pitch") survive tokenization intact.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into BM25 terms."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An inverted index supporting BM25 ranking with optional filtering.

    Documents are addressed by an integer position assigned at build time;
    `doc_ids` maps those positions back to the caller's identifiers.
    """

    __slots__ = (
        "postings",
        "doc_ids",
        "doc_lengths",
        "doc_meta",
        "by_content_type",
        "avgdl",
        "n_docs",
    )

    def __init__(self) -> None:
        # term -> (document positions, term frequencies), as parallel arrays.
        # Storing these as `array("i")` rather than a list of small objects
        # matters at this scale: 26M postings as dataclass instances pickle to
        # 681 MB and take seconds to load, where the same data as packed
        # integers is a fraction of that and unpickles as raw bytes.
        self.postings: dict[str, tuple[array, array]] = {}
        self.doc_ids: list[str] = []
        self.doc_lengths: array = array("i")
        self.doc_meta: list[dict[str, str]] = []
        # Precomputed at build time. Deriving it per query meant scanning all
        # 384k documents on every filtered search.
        self.by_content_type: dict[str, frozenset[int]] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, str]] | None = None,
        max_df_ratio: float = MAX_DF_RATIO,
    ) -> BM25Index:
        """Build an index from parallel sequences of ids, texts, and metadata.

        Terms present in more than `max_df_ratio` of documents are omitted; see
        MAX_DF_RATIO for why.
        """
        if len(doc_ids) != len(texts):
            raise ValueError(f"{len(doc_ids)} ids but {len(texts)} texts")
        if metadatas is not None and len(metadatas) != len(doc_ids):
            raise ValueError(f"{len(doc_ids)} ids but {len(metadatas)} metadatas")

        index = cls()
        index.doc_ids = list(doc_ids)
        index.doc_meta = list(metadatas) if metadatas is not None else [{} for _ in doc_ids]
        index.n_docs = len(index.doc_ids)

        accumulator: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total_length = 0

        for position, text in enumerate(texts):
            tokens = tokenize(text)
            index.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, frequency in counts.items():
                accumulator[term].append((position, frequency))

        index.avgdl = total_length / index.n_docs if index.n_docs else 0.0

        df_cutoff = (
            index.n_docs * max_df_ratio if index.n_docs >= MIN_DOCS_FOR_PRUNING else float("inf")
        )
        index.postings = {
            term: (
                array("i", [p for p, _ in entries]),
                array("i", [f for _, f in entries]),
            )
            for term, entries in accumulator.items()
            if len(entries) <= df_cutoff
        }

        by_type: dict[str, set[int]] = defaultdict(set)
        for position, meta in enumerate(index.doc_meta):
            if content_type := meta.get("content_type"):
                by_type[content_type].add(position)
        index.by_content_type = {k: frozenset(v) for k, v in by_type.items()}

        return index

    # -- querying ----------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Inverse document frequency, in the smoothed BM25+ form.

        The `1 +` keeps the value positive for terms appearing in more than
        half the corpus; the unsmoothed form goes negative there, which would
        let a common term actively penalize a document that contains it.
        """
        entry = self.postings.get(term)
        if entry is None:
            return 0.0
        df = len(entry[0])
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
        lengths, avgdl = self.doc_lengths, self.avgdl

        for term in set(tokenize(query)):
            entry = self.postings.get(term)
            if entry is None:
                continue
            idf = self._idf(term)
            positions, frequencies = entry
            for position, tf in zip(positions, frequencies, strict=True):
                if allowed is not None and position not in allowed:
                    continue
                length_norm = 1 - B + B * lengths[position] / avgdl
                scores[position] += idf * tf * (K1 + 1) / (tf + K1 * length_norm)

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
        positions: set[int] = set()
        for content_type in content_types:
            positions |= self.by_content_type.get(content_type, frozenset())
        return positions

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
            # A pickle written by an older layout may not even be loadable --
            # it can reference classes that no longer exist. Treat any failure
            # as a cache miss so the caller rebuilds.
            return None
        if not isinstance(index, cls):
            return None

        fingerprint = payload.get("fingerprint")
        # The version is checked unconditionally, not just when the caller
        # supplies a fingerprint: an index written by a different tokenizer or
        # layout is wrong regardless of whether the corpus matches.
        if not isinstance(fingerprint, tuple) or fingerprint[0] != INDEX_VERSION:
            return None
        if expect is not None and fingerprint != expect:
            return None
        return index
