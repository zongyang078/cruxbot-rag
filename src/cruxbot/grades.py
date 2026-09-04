"""Climbing grade normalization.

Climbing grades are not a single scale. A query for "7a routes" (French sport)
should match documents graded "5.11d" (YDS), and a query for "V4" should match
"6B" (Fontainebleau). Retrieval that treats these as unrelated strings silently
loses recall on a large fraction of route queries.

This module converts between:
  - YDS       (US free climbing)      5.5 .. 5.14d
  - French    (sport)                 4a .. 9a
  - V-scale   (US bouldering)         V0 .. V16
  - Font      (Fontainebleau boulder) 4 .. 8C+

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

import re

# --- YDS <-> French -------------------------------------------------------
# Note: the mapping is not injective. Both 5.11d and 5.12a are commonly cited
# as ~7a+. The reverse table therefore maps each French grade to *all* YDS
# grades it corresponds to, rather than silently keeping whichever key happened
# to be inserted last (a bug in the original implementation).
# fmt: off
YDS_TO_FRENCH: dict[str, str] = {
    "5.5": "4a", "5.6": "4b", "5.7": "4c", "5.8": "5a", "5.9": "5b",
    "5.10a": "6a", "5.10b": "6a+", "5.10c": "6b", "5.10d": "6b+",
    "5.11a": "6c", "5.11b": "6c+", "5.11c": "7a", "5.11d": "7a+",
    "5.12a": "7a+", "5.12b": "7b", "5.12c": "7b+", "5.12d": "7c",
    "5.13a": "7c+", "5.13b": "8a", "5.13c": "8a+", "5.13d": "8b",
    "5.14a": "8b+", "5.14b": "8c", "5.14c": "8c+", "5.14d": "9a",
}
# fmt: on

FRENCH_TO_YDS: dict[str, list[str]] = {}
for _yds, _fr in YDS_TO_FRENCH.items():
    FRENCH_TO_YDS.setdefault(_fr, []).append(_yds)

# --- V-scale <-> Font -----------------------------------------------------
# fmt: off
V_TO_FONT: dict[str, str] = {
    "V0": "4", "V1": "5", "V2": "5+", "V3": "6A", "V4": "6B",
    "V5": "6C", "V6": "7A", "V7": "7A+", "V8": "7B+", "V9": "7C",
    "V10": "7C+", "V11": "8A", "V12": "8A+", "V13": "8B", "V14": "8B+",
    "V15": "8C", "V16": "8C+",
}
# fmt: on
FONT_TO_V: dict[str, str] = {v: k for k, v in V_TO_FONT.items()}

# --- Patterns -------------------------------------------------------------
_YDS_RE = re.compile(r"^5\.(\d{1,2})([a-d])?$", re.IGNORECASE)
_FRENCH_RE = re.compile(r"^([4-9])([a-c])(\+?)$", re.IGNORECASE)
_V_RE = re.compile(r"^V(\d{1,2})$", re.IGNORECASE)

# Ordered by specificity: a query containing "5.10a" must not be read as "V1".
#
# The French pattern ends in a negative lookahead rather than \b. A trailing \b
# after an optional "+" can never match -- "+" and the following space are both
# non-word characters, so there is no boundary between them -- which makes the
# regex backtrack and silently return "7a" for a query about "7a+". That is a
# full grade off, and it is wrong in the direction that matters: it widens the
# search to easier routes than the climber asked for.
_QUERY_PATTERNS = (
    re.compile(r"\b(5\.\d{1,2}[a-d]?)\b", re.IGNORECASE),
    re.compile(r"\b(V\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b([4-9][a-c]\+?)(?!\w)", re.IGNORECASE),
)


def normalize(grade: str) -> str:
    """Canonicalize casing so lookups are case-insensitive.

    YDS and French are lowercase by convention ("5.10a", "7a+"); V-scale is
    uppercase ("V4"). The original implementation matched case-insensitively
    but then did a case-sensitive dict lookup, so "6A" silently returned no
    equivalents.
    """
    g = grade.strip()
    if _V_RE.match(g):
        return g.upper()
    return g.lower()


def extract_grade(query: str) -> str | None:
    """Return the first climbing grade mentioned in a free-text query, if any."""
    for pattern in _QUERY_PATTERNS:
        m = pattern.search(query)
        if m:
            return normalize(m.group(1))
    return None


def equivalents(grade: str) -> list[str]:
    """Return every grade string worth searching for, given one grade.

    Always includes the input itself. Results are sorted for determinism so
    that callers (and snapshot tests) see a stable ordering.
    """
    g = normalize(grade)
    out: set[str] = {g}

    if m := _YDS_RE.match(g):
        number, letter = m.group(1), m.group(2)
        base = f"5.{number}"
        out.add(base)

        if fr := YDS_TO_FRENCH.get(g):
            out.add(fr)

        # A bare "5.11" should fan out to 5.11a-d, since routes are catalogued
        # with letter suffixes. Only meaningful from 5.10 up, where the letter
        # subdivision exists.
        if letter is None and int(number) >= 10:
            for suffix in ("a", "b", "c", "d"):
                sub = f"{base}{suffix}"
                out.add(sub)
                if fr := YDS_TO_FRENCH.get(sub):
                    out.add(fr)

    elif _FRENCH_RE.match(g):
        for yds in FRENCH_TO_YDS.get(g, []):
            out.add(yds)
            if m := _YDS_RE.match(yds):
                out.add(f"5.{m.group(1)}")

    elif _V_RE.match(g):
        if font := V_TO_FONT.get(g):
            out.add(font)

    return sorted(out)


def expand_query(query: str) -> str:
    """Append grade equivalents to a query so sparse retrieval can match them.

    Returns the query unchanged when it mentions no grade, or when the grade
    has no known equivalents beyond itself.
    """
    grade = extract_grade(query)
    if not grade:
        return query

    extras = [g for g in equivalents(grade) if g.lower() not in query.lower()]
    if not extras:
        return query
    return f"{query} (grades: {', '.join(extras)})"
