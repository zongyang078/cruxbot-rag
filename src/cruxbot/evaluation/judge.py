"""Grade the relevance of retrieved passages to a query.

Judging turns the pool built in `dataset` into labels, and it is the part of
the harness most able to quietly invalidate everything downstream: labels that
are noise produce metrics that are noise, and the metrics still look like
numbers. Three things guard against that.

**The judge sees a whole query's pool in one request.** Not for cost -- the
whole label set is a couple of dollars either way -- but because a grader that
sees thirty candidates together can calibrate across them, where thirty
independent calls each have to guess at an absolute scale. It also removes
twenty-nine repetitions of the rubric.

**Grades come back as a validated schema, not as prose to be parsed.** A regex
hunting for a digit in free text is a silent failure waiting to happen: it
cannot distinguish "2" the grade from "2" in the passage.

**Agreement with human labels is measured, not assumed.** `agreement` compares
judged grades against a hand-labelled sample so the judge's reliability is a
reported number rather than a hope.

Requires the eval extra:  pip install -e ".[eval]"
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from cruxbot.evaluation.metrics import DEFAULT_MIN_GRADE

# Grades are ordinal, and the wording carries the work. A judge given only
# "relevant" or "irrelevant" collapses "mentions the topic" into "answers the
# question" -- which is the distinction nDCG exists to reward.
GRADE_DEFINITIONS: dict[int, str] = {
    3: "Directly answers the question. A climber would need nothing else.",
    2: "Useful but partial. Addresses the question without fully answering it.",
    1: "On topic but does not address the question. Background at best.",
    0: "Irrelevant to the question.",
}

SYSTEM = """\
You are grading how well retrieved passages answer a rock climbing question, \
for a retrieval evaluation. Grade each passage independently on this scale:

{scale}

Judge only whether the passage answers THIS question. Do not reward a passage \
for being well written, and do not penalise one for being an excerpt. A route \
description is not relevant to a training question merely because both concern \
climbing.

Return a grade for every passage, identified by its index."""

# Long enough to judge on, short enough that thirty fit in one request.
MAX_PASSAGE_CHARS = 1200

DEFAULT_MODEL = "claude-opus-5"


class PassageGrade(BaseModel):
    """One graded passage."""

    index: int = Field(description="The passage's index as given in the prompt")
    grade: int = Field(ge=0, le=3, description="Relevance grade from 0 to 3")


class GradeResponse(BaseModel):
    """The judge's grades for one query's pool."""

    grades: list[PassageGrade]


@dataclass(slots=True)
class Judgement:
    """One graded (query, passage) pair."""

    chunk_id: str
    grade: int | None
    note: str = ""

    @property
    def abstained(self) -> bool:
        """True when no grade came back for this passage."""
        return self.grade is None


class RelevanceJudge(Protocol):
    """Grades a query's candidate passages."""

    name: str

    def grade(self, query: str, passages: Mapping[str, str]) -> list[Judgement]:
        """Return one judgement per passage, in the order given."""
        ...


def build_system_prompt() -> str:
    scale = "\n".join(
        f"  {grade} = {text}" for grade, text in sorted(GRADE_DEFINITIONS.items(), reverse=True)
    )
    return SYSTEM.format(scale=scale)


def build_user_prompt(query: str, passages: Sequence[str]) -> str:
    """Render one query and its numbered candidate passages."""
    blocks = "\n\n".join(
        f"[{i}] {passage[:MAX_PASSAGE_CHARS]}" for i, passage in enumerate(passages)
    )
    return f"Question: {query}\n\nPassages:\n\n{blocks}"


def collect(
    chunk_ids: Sequence[str], graded: Iterable[PassageGrade], note: str = ""
) -> list[Judgement]:
    """Match returned grades back to chunk ids, positionally.

    A passage the judge skipped, or referred to by an out-of-range index,
    becomes an abstention rather than a zero. The two are different claims: one
    says the passage is irrelevant, the other says nobody looked.
    """
    by_index = {g.index: g.grade for g in graded if 0 <= g.index < len(chunk_ids)}
    return [
        Judgement(chunk_id, by_index.get(i), note if i not in by_index else "")
        for i, chunk_id in enumerate(chunk_ids)
    ]


class AnthropicJudge:
    """Relevance judge backed by the Anthropic API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        api_key: str | None = None,
    ) -> None:
        import anthropic

        # Relevance grading is classification, not reasoning: low effort keeps
        # the judge from spending thinking tokens deliberating over a scale it
        # was handed.
        self.name = model
        self.effort = effort
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def grade(self, query: str, passages: Mapping[str, str]) -> list[Judgement]:
        chunk_ids = list(passages)
        if not chunk_ids:
            return []

        try:
            response = self._client.messages.parse(
                model=self.name,
                max_tokens=4096,
                system=build_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": build_user_prompt(query, [passages[c] for c in chunk_ids]),
                    }
                ],
                output_format=GradeResponse,
                output_config={"effort": self.effort},
            )
        except Exception as exc:  # noqa: BLE001 - one query must not void the run
            return collect(chunk_ids, [], note=f"{type(exc).__name__}: {exc}")

        parsed = response.parsed_output
        if parsed is None:
            return collect(chunk_ids, [], note="no parsed output")
        return collect(chunk_ids, parsed.grades, note="not graded")


def to_relevance(judgements: Iterable[Judgement]) -> dict[str, float]:
    """Collapse judgements into the relevance map the metrics expect.

    Abstentions are omitted. A passage the judge did not grade was never
    judged, and the label set should say so rather than assert irrelevance.
    """
    return {j.chunk_id: float(j.grade) for j in judgements if j.grade is not None}


@dataclass(slots=True)
class Agreement:
    """How closely judged grades track human ones on a shared sample."""

    n: int
    exact: float
    within_one: float
    binary: float
    judge_mean: float
    human_mean: float


def agreement(
    judged: Mapping[str, float],
    human: Mapping[str, float],
    min_grade: float = DEFAULT_MIN_GRADE,
) -> Agreement:
    """Compare judged grades against human grades on the same passages.

    Three levels, because they answer different questions. Exact agreement is
    the strictest and usually the least informative -- graders argue about 2
    versus 3 constantly. Within-one tolerates that. `binary` collapses to
    relevant-or-not, which is what Recall actually depends on, and is therefore
    the figure that matters most for these metrics.

    `min_grade` must be the threshold the metrics use, or `binary` answers a
    question nobody asked. It previously collapsed at `> 0`, which scores the
    boundary between "off topic" and "on topic" -- easy, and not the one Recall
    draws. Recall's boundary is grade 1 versus 2, "on topic" versus "actually
    answers", which is where graders genuinely disagree; measuring the easy
    boundary instead reported a flatteringly high number.
    """
    shared = sorted(set(judged) & set(human))
    if not shared:
        return Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    exact = sum(1 for k in shared if judged[k] == human[k])
    within = sum(1 for k in shared if abs(judged[k] - human[k]) <= 1)
    binary = sum(
        1 for k in shared if (judged[k] >= min_grade) == (human[k] >= min_grade)
    )

    return Agreement(
        n=len(shared),
        exact=exact / len(shared),
        within_one=within / len(shared),
        binary=binary / len(shared),
        judge_mean=sum(judged[k] for k in shared) / len(shared),
        human_mean=sum(human[k] for k in shared) / len(shared),
    )


def sample_for_review(relevance: Mapping[str, float], size: int, seed: int = 0) -> list[str]:
    """Pick a stratified sample of judged passages for a human to re-grade.

    Stratified by grade, because a uniform sample of a label set that is mostly
    zeros is mostly zeros, and would confirm only that the judge can recognise
    irrelevance -- the disagreements worth finding live at the boundaries.
    """
    import random

    by_grade: dict[float, list[str]] = {}
    for chunk_id, grade in sorted(relevance.items()):
        by_grade.setdefault(grade, []).append(chunk_id)
    if not by_grade:
        return []

    rng = random.Random(seed)
    per_stratum = max(1, size // len(by_grade))

    picked: list[str] = []
    for grade in sorted(by_grade):
        candidates = by_grade[grade]
        picked.extend(rng.sample(candidates, min(per_stratum, len(candidates))))
    return picked[:size]
