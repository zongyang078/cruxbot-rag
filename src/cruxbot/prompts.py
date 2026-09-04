"""Prompt construction.

Kept pure and separate from the LLM client so that prompt changes are testable
without a model running, and so that an evaluation run can diff the exact
prompt text between system versions.

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

from cruxbot import grades
from cruxbot.types import Chunk

SYSTEM = """\
You are CruxBot, an expert rock climbing assistant.

INSTRUCTIONS:
- FIRST, determine whether the question is about rock climbing (routes, \
training, gear, safety, accidents, technique, or climbing areas). If it is not \
about climbing at all, respond exactly: "I don't have data on this in my \
climbing knowledge base. I can only answer questions about rock climbing."
- Otherwise, answer using ONLY the context below.
- Cite sources by their bracket number, e.g. [1], [2].
- If the context is relevant but incomplete, give what it supports and state \
what is missing. Do not refuse outright when partial information exists.
- If the context contains no relevant climbing information, say: "I don't have \
data on this in my climbing knowledge base."
- NEVER invent facts, routes, grades, or URLs that are absent from the context.
- A passing mention of a non-climbing topic (a climber mentioning food, say) is \
not evidence for a non-climbing question.
- If a source is marked as a general page, note that the specific post link is \
unavailable."""

MAX_LOCATION_CHARS = 80


def format_chunk(chunk: Chunk, index: int) -> str:
    """Render one retrieved chunk as a numbered, attributed context block."""
    header: list[str] = []
    if chunk.content_type:
        header.append(f"Type: {chunk.content_type}")
    if chunk.grade:
        header.append(f"Grade: {chunk.grade}")
    if location := chunk.location:
        if len(location) > MAX_LOCATION_CHARS:
            location = location[:MAX_LOCATION_CHARS] + "..."
        header.append(f"Location: {location}")
    if chunk.source_url:
        note = "" if chunk.is_citable else " (general page, not a specific post)"
        header.append(f"URL: {chunk.source_url}{note}")

    if header:
        return f"[{index}] ({' | '.join(header)})\n{chunk.text}"
    return f"[{index}] {chunk.text}"


def grade_note(query: str) -> str:
    """A hint listing equivalent grades, when the query names one.

    Returns an empty string when the query names no grade, so callers can
    concatenate unconditionally.
    """
    grade = grades.extract_grade(query)
    if not grade:
        return ""
    equivalents = grades.equivalents(grade)
    if len(equivalents) <= 1:
        return ""
    return (
        f"\nNote: {grade} is approximately equivalent to "
        f"{', '.join(g for g in equivalents if g != grade)}. "
        "Treat these as the same difficulty when matching routes.\n"
    )


def build(query: str, chunks: Sequence[Chunk]) -> str:
    """Assemble the full prompt for one query."""
    context = "\n\n".join(format_chunk(c, i) for i, c in enumerate(chunks, 1))
    return f"{SYSTEM}\n{grade_note(query)}\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
