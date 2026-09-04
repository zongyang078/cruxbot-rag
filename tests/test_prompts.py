"""Prompt construction.

Anti-hallucination in this system is largely a property of the prompt, so the
prompt is treated as code: pinned, and changed deliberately.
"""

from __future__ import annotations

from cruxbot import prompts
from cruxbot.types import Chunk


def chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "text": "Access Denied is a 5.10b/c sport route with four pitches.",
        "chunk_id": "abc_c0",
        "source_url": "https://www.mountainproject.com/route/110149834/access-denied",
        "metadata": {"content_type": "route", "grade": "5.10b/c", "location": "El Potrero Chico"},
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


class TestFormatChunk:
    def test_includes_the_passage_text(self) -> None:
        assert "Access Denied" in prompts.format_chunk(chunk(), 1)

    def test_is_numbered_for_citation(self) -> None:
        assert prompts.format_chunk(chunk(), 3).startswith("[3]")

    def test_surfaces_metadata_the_model_needs(self) -> None:
        rendered = prompts.format_chunk(chunk(), 1)
        assert "Type: route" in rendered
        assert "Grade: 5.10b/c" in rendered
        assert "El Potrero Chico" in rendered

    def test_truncates_a_long_location(self) -> None:
        long_location = " > ".join(["Region"] * 40)
        rendered = prompts.format_chunk(chunk(metadata={"location": long_location}), 1)
        assert "..." in rendered
        assert len(rendered) < len(long_location)

    def test_flags_a_non_specific_url(self) -> None:
        rendered = prompts.format_chunk(
            chunk(source_url="https://www.mountainproject.com/forum"), 1
        )
        assert "general page" in rendered

    def test_does_not_flag_a_specific_url(self) -> None:
        assert "general page" not in prompts.format_chunk(chunk(), 1)

    def test_bare_chunk_still_renders(self) -> None:
        rendered = prompts.format_chunk(
            Chunk(text="just text", chunk_id="x", source_url="", metadata={}), 1
        )
        assert rendered == "[1] just text"


class TestGradeNote:
    def test_empty_when_the_query_names_no_grade(self) -> None:
        assert prompts.grade_note("How do I train finger strength?") == ""

    def test_lists_equivalents_when_a_grade_is_named(self) -> None:
        note = prompts.grade_note("Find me a 5.11a route")
        assert "6c" in note
        assert "5.11a" in note

    def test_does_not_list_the_grade_as_its_own_equivalent(self) -> None:
        note = prompts.grade_note("Find me a 5.11a route")
        equivalents_clause = note.split("equivalent to")[1]
        assert "5.11a" not in equivalents_clause


class TestBuild:
    def test_contains_the_question(self) -> None:
        assert "Question: Where to climb?" in prompts.build("Where to climb?", [chunk()])

    def test_contains_the_refusal_instruction(self) -> None:
        assert "I don't have data on this" in prompts.build("q", [chunk()])

    def test_forbids_inventing_sources(self) -> None:
        assert "NEVER invent" in prompts.build("q", [chunk()])

    def test_numbers_every_chunk(self) -> None:
        built = prompts.build("q", [chunk(chunk_id="a"), chunk(chunk_id="b")])
        assert "[1]" in built
        assert "[2]" in built

    def test_handles_zero_retrieved_chunks(self) -> None:
        # An empty index or an over-narrow filter must not crash the pipeline;
        # the model should still receive its refusal instruction.
        built = prompts.build("q", [])
        assert "I don't have data on this" in built
        assert "Question: q" in built

    def test_is_deterministic(self) -> None:
        chunks = [chunk()]
        assert prompts.build("q", chunks) == prompts.build("q", chunks)
