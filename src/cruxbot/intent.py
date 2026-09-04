"""Rule-based query intent detection.

The knowledge base is heavily skewed toward route entries (~60% of documents).
Without a content-type prior, a naive retriever answers "how do I train finger
strength" with route descriptions, because routes dominate the index. Detecting
intent lets the retriever restrict its candidate pool to the content types that
can actually answer the question.

This is deliberately rule-based rather than model-based: it is inspectable,
costs nothing at query time, and its failure modes are enumerable (see
`tests/test_intent.py`). Its known weakness -- that it keys on surface
keywords and so misreads "What climbing routes are on Mars?" as a genuine
route query -- is measured rather than hidden.

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

import re

# Content types present in the unified corpus.
ROUTE = "route"
FORUM = "forum_discussion"
ARTICLE = "article"
GEAR = "gear_review"
REDDIT = "reddit_post"

# Patterns must tolerate inflection. The original keyword list was written with
# bare \b...\b anchors, so `\baccident\b` did not match "accidents" and
# `\broute\b` did not match "routes" -- meaning the single most common phrasing
# of a route query ("what routes are in...") scored zero on route intent. Plural
# and gerund forms are spelled out explicitly rather than by a loose \w* suffix,
# which would make `\bcam\b` match "campus" and "camping".
#
# `\bclimb\b` is deliberately NOT widened to "climbing": that word appears in
# nearly every query to this system, so it carries no discriminative signal and
# would push every query toward the majority content type.
# fmt: off
_PATTERNS: dict[str, tuple[str, ...]] = {
    ROUTE: (
        r"\broutes?\b", r"\bclimbs?\b", r"\bcrags?\b", r"\bwalls?\b",
        r"\bpitch(?:es)?\b", r"\bmulti-?pitch\b", r"\bboulder(?:s|ing)?\b",
        r"\btrad\b", r"\bsport\b", r"\bareas?\b", r"\bclassics?\b",
        r"\byosemite\b", r"\bjoshua tree\b", r"\bred river gorge\b",
        r"\bbishop\b", r"\bsmith rock\b", r"\bel potrero\b",
        r"\brumney\b", r"\bnew river gorge\b", r"\bindian creek\b",
        r"\bhueco\b", r"\butah\b", r"\bcolorado\b", r"\bcalifornia\b",
    ),
    FORUM: (
        r"\btrain(?:s|ing|ed)?\b", r"\bworkouts?\b", r"\bhangboard(?:ing)?\b",
        r"\bcampus(?:ing)?\b", r"\bfootwork\b", r"\btechniques?\b",
        r"\bendurance\b", r"\bstrength\b", r"\bplateau(?:ing|ed)?\b",
        r"\bflexib(?:le|ility)\b", r"\bstretch(?:es|ing)?\b",
        r"\binjur(?:y|ies|ed)\b", r"\bpreven(?:t|ts|tion)\b",
        r"\brecover(?:y|ing)?\b", r"\btips?\b", r"\badvice\b",
        r"\bhow (?:do|should|can|to)\b",
    ),
    ARTICLE: (
        r"\baccidents?\b", r"\bfatal(?:ity|ities)?\b", r"\bdeaths?\b",
        r"\bfall(?:s|ing|en)?\b", r"\bsafety\b", r"\brescues?\b",
        r"\brappel(?:s|ling|led|ing)?\b", r"\bbelay(?:s|ing|ed)?\b",
        r"\banchors?\b", r"\bprotection\b", r"\brockfall\b", r"\bknots?\b",
        r"\bfree solo(?:ing)?\b", r"\brisks?\b", r"\bdanger(?:ous)?\b",
    ),
    GEAR: (
        r"\bgear\b", r"\bequipment\b", r"\bshoes?\b", r"\bharness(?:es)?\b",
        r"\bhelmets?\b", r"\bropes?\b", r"\bcarabiners?\b", r"\bquickdraws?\b",
        r"\bbelay devices?\b", r"\batc\b", r"\bgrigri\b", r"\bcrash pads?\b",
        r"\bchalk\b", r"\bcams?\b", r"\bnuts?\b", r"\breviews?\b", r"\bbuy\b",
        r"\brecommend(?:s|ation|ations)?\b", r"\bbest\b.*\bfor\b",
        r"\bcompari?sons?\b", r"\bla sportiva\b", r"\bscarpa\b",
        r"\bblack diamond\b", r"\bpetzl\b",
    ),
}
# fmt: on

_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    label: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for label, patterns in _PATTERNS.items()
}

# When an intent is detected, these are the content types worth retrieving from.
# Intent is a prior, not a hard filter: training questions are often answered in
# Reddit threads, and gear opinions live in forum posts as much as in reviews.
_EXPANSIONS: dict[str, list[str]] = {
    ROUTE: [ROUTE],
    FORUM: [FORUM, REDDIT],
    GEAR: [GEAR, FORUM],
    ARTICLE: [ARTICLE, FORUM],
}

# A single matched keyword is enough to commit when nothing else matches; when
# two intents compete, the winner must be at least this many times stronger.
_DOMINANCE_RATIO = 2


def score(query: str) -> dict[str, int]:
    """Return the number of matching patterns per content type.

    Exposed separately from `detect` so that evaluation and debugging can see
    *why* a query was routed the way it was.
    """
    return {
        label: sum(1 for pattern in patterns if pattern.search(query))
        for label, patterns in _COMPILED.items()
    }


def detect(query: str) -> str | None:
    """Classify a query's intent, or return None when the signal is ambiguous.

    Returning None is a real outcome, not a failure: it means "do not narrow
    the candidate pool", which is the safe default.
    """
    scores = {label: n for label, n in score(query).items() if n > 0}
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_label, best_score = ranked[0]

    if len(ranked) == 1:
        return best_label
    if best_score >= _DOMINANCE_RATIO * ranked[1][1]:
        return best_label
    return None


def content_types_for(intent: str | None) -> list[str] | None:
    """Map a detected intent to the content types to retrieve from.

    Returns None when no filter should be applied.
    """
    if intent is None:
        return None
    return _EXPANSIONS.get(intent)
