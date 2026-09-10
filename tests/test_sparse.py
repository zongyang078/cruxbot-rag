"""BM25 inverted index.

This is the most algorithmically involved code in the repo, so it is tested
against hand-computable cases rather than only for self-consistency.
"""

from __future__ import annotations

import math

import pytest

from cruxbot.retrieval.sparse import K1, B, BM25Index, tokenize


class TestTokenize:
    def test_lowercases(self) -> None:
        assert tokenize("Yosemite VALLEY") == ["yosemite", "valley"]

    def test_preserves_grades(self) -> None:
        # "5.10a" must survive as one token; splitting on the dot would make
        # the sparse index unable to match the single most important term in a
        # route query.
        assert "5.10a" in tokenize("a 5.10a route")

    def test_preserves_hyphenated_terms(self) -> None:
        assert "multi-pitch" in tokenize("a multi-pitch climb")

    def test_drops_punctuation(self) -> None:
        assert tokenize("hello, world!") == ["hello", "world"]

    def test_empty_text_gives_no_tokens(self) -> None:
        assert tokenize("") == []


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(
        doc_ids=["d0", "d1", "d2", "d3"],
        texts=[
            "sport route in yosemite",
            "trad route in yosemite valley",
            "bouldering in bishop",
            "hangboard training for finger strength",
        ],
        metadatas=[
            {"content_type": "route"},
            {"content_type": "route"},
            {"content_type": "route"},
            {"content_type": "forum_discussion"},
        ],
    )


class TestBuild:
    def test_records_corpus_statistics(self, index: BM25Index) -> None:
        assert index.n_docs == 4
        assert index.doc_lengths == [4, 5, 3, 5]
        assert index.avgdl == pytest.approx(17 / 4)

    def test_builds_postings_only_for_present_terms(self, index: BM25Index) -> None:
        assert {p.doc_index for p in index.postings["yosemite"]} == {0, 1}
        assert "kayaking" not in index.postings

    def test_counts_repeated_terms(self) -> None:
        idx = BM25Index.build(["d0"], ["rope rope rope"])
        assert idx.postings["rope"][0].term_frequency == 3

    def test_rejects_mismatched_input_lengths(self) -> None:
        with pytest.raises(ValueError, match="2 ids but 1 texts"):
            BM25Index.build(["a", "b"], ["only one"])

    def test_rejects_mismatched_metadata(self) -> None:
        with pytest.raises(ValueError, match="1 ids but 2 metadatas"):
            BM25Index.build(["a"], ["t"], [{}, {}])

    def test_metadata_defaults_to_empty(self) -> None:
        idx = BM25Index.build(["d0"], ["text"])
        assert idx.doc_meta == [{}]

    def test_empty_corpus_does_not_divide_by_zero(self) -> None:
        idx = BM25Index.build([], [])
        assert idx.avgdl == 0.0
        assert idx.search("anything") == []


class TestScore:
    def test_only_scores_documents_containing_a_query_term(self, index: BM25Index) -> None:
        scores = index.score("hangboard")
        assert set(scores) == {3}

    def test_unknown_term_scores_nothing(self, index: BM25Index) -> None:
        assert index.score("kayaking") == {}

    def test_matches_the_bm25_formula(self) -> None:
        # Two documents, one term. Hand-computed so a refactor that changes the
        # scoring silently would fail here rather than pass on self-consistency.
        idx = BM25Index.build(["d0", "d1"], ["rope rope", "harness"])

        df, n_docs = 1, 2
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf, doc_len, avgdl = 2, 2, 1.5
        expected = idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len / avgdl))

        assert idx.score("rope")[0] == pytest.approx(expected)

    def test_rarer_terms_score_higher(self) -> None:
        # "yosemite" appears in 1 of 3 docs, "route" in all 3. A document
        # matching the rare term should outrank one matching the common term.
        idx = BM25Index.build(
            ["rare", "common_a", "common_b"],
            ["yosemite route", "route", "route"],
        )
        scores = idx.score("yosemite route")
        assert scores[0] > scores[1]

    def test_term_frequency_saturates(self) -> None:
        # Ten occurrences must not score five times a document with two.
        idx = BM25Index.build(["few", "many"], ["rope rope", "rope " * 10])
        scores = idx.score("rope")
        assert scores[1] < 5 * scores[0]

    def test_idf_stays_positive_for_ubiquitous_terms(self) -> None:
        # The unsmoothed IDF goes negative past 50% document frequency, which
        # would let a common term penalize documents that contain it.
        idx = BM25Index.build([f"d{i}" for i in range(10)], ["rope"] * 10)
        assert all(score > 0 for score in idx.score("rope").values())

    def test_allowed_set_restricts_scoring(self, index: BM25Index) -> None:
        assert set(index.score("route", allowed={0})) == {0}

    def test_empty_allowed_set_scores_nothing(self, index: BM25Index) -> None:
        assert index.score("route", allowed=set()) == {}


class TestSearch:
    def test_returns_best_first(self, index: BM25Index) -> None:
        results = index.search("yosemite valley")
        assert results[0]["chunk_id"] == "d1"

    def test_respects_top_k(self, index: BM25Index) -> None:
        assert len(index.search("route in yosemite", top_k=1)) == 1

    def test_carries_metadata_through(self, index: BM25Index) -> None:
        assert index.search("hangboard")[0]["metadata"] == {"content_type": "forum_discussion"}

    def test_filters_by_content_type(self, index: BM25Index) -> None:
        results = index.search("in", content_types=["forum_discussion"])
        assert [r["chunk_id"] for r in results] == []

        results = index.search("hangboard", content_types=["forum_discussion"])
        assert [r["chunk_id"] for r in results] == ["d3"]

    def test_unmatched_content_type_returns_nothing(self, index: BM25Index) -> None:
        assert index.search("route", content_types=["gear_review"]) == []

    def test_no_match_returns_empty_list(self, index: BM25Index) -> None:
        assert index.search("kayaking") == []

    def test_is_deterministic(self, index: BM25Index) -> None:
        assert index.search("route in yosemite") == index.search("route in yosemite")


class TestPersistence:
    def test_round_trips(self, index: BM25Index, tmp_path) -> None:
        path = tmp_path / "bm25.pkl"
        index.save(path)
        loaded = BM25Index.load(path, expect=index.fingerprint())
        assert loaded is not None
        assert loaded.search("yosemite") == index.search("yosemite")

    def test_creates_parent_directories(self, index: BM25Index, tmp_path) -> None:
        index.save(tmp_path / "nested" / "dir" / "bm25.pkl")
        assert (tmp_path / "nested" / "dir" / "bm25.pkl").exists()

    def test_missing_file_is_a_cache_miss(self, tmp_path) -> None:
        assert BM25Index.load(tmp_path / "absent.pkl") is None

    def test_corrupt_file_is_a_cache_miss(self, tmp_path) -> None:
        path = tmp_path / "bm25.pkl"
        path.write_bytes(b"not a pickle")
        assert BM25Index.load(path) is None

    def test_stale_index_is_rejected(self, index: BM25Index, tmp_path) -> None:
        # The original implementation cached the index with no validity key, so
        # rebuilding the corpus left a stale index that loaded without warning
        # and returned ids that no longer existed.
        path = tmp_path / "bm25.pkl"
        index.save(path)

        rebuilt = BM25Index.build(["x0", "x1"], ["different corpus", "entirely"])
        assert BM25Index.load(path, expect=rebuilt.fingerprint()) is None

    def test_fingerprint_changes_with_corpus_size(self, index: BM25Index) -> None:
        smaller = BM25Index.build(index.doc_ids[:2], ["a", "b"])
        assert smaller.fingerprint() != index.fingerprint()
