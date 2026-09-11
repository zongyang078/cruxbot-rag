"""Validate the relevance judge against human grades.

Every retrieval number in this project rests on 1,103 machine-assigned
relevance labels. Without a check, "Recall@50 is 0.95" means "a model said so",
and the obvious question -- how do you know the labels are right? -- has no
answer. This measures that.

Two steps:

    python scripts/human_eval.py prepare --size 60
    # fill in the GRADE lines in benchmarks/human_eval.md, then:
    python scripts/human_eval.py score

`prepare` writes a blind sheet: the judge's own grade is withheld, so seeing it
cannot anchor the human grade. The sample is stratified across grades, because
a uniform sample of a label set that is 41% zeros is mostly zeros, and would
confirm only that the judge recognises irrelevance -- the disagreements worth
finding are at the boundaries.

Requires:  pip install -e ".[rag,eval]"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from cruxbot.config import settings
from cruxbot.evaluation.dataset import load_queries
from cruxbot.evaluation.judge import GRADE_DEFINITIONS, agreement

SHEET = Path("benchmarks/human_eval.md")
KEY = Path("benchmarks/human_eval_key.json")
PASSAGE_CHARS = 900

_ITEM_RE = re.compile(r"^### (\S+)\s*$", re.MULTILINE)
_GRADE_RE = re.compile(r"^GRADE:\s*([0-3])?\s*$", re.MULTILINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Write a blind grading sheet")
    prep.add_argument("--labels", type=Path, default=Path("benchmarks/labeled.jsonl"))
    prep.add_argument("--size", type=int, default=60)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    prep.add_argument("--collection", default=settings.collection)

    sub.add_parser("score", help="Read the filled sheet and report agreement")
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace) -> int:
    from cruxbot.retrieval.dense import ChromaStore

    queries = {q.query_id: q for q in load_queries(args.labels)}

    # Stratify globally rather than per query. Sampling within each query and
    # taking one from each returns only the lowest grade present -- 39 of 40
    # items came back graded 0, which would confirm nothing beyond the judge's
    # ability to recognise irrelevance. Grades are the axis that matters, so
    # they are the axis to balance on.
    pairs: dict[float, list[tuple[str, str]]] = {}
    for query_id, query in sorted(queries.items()):
        for chunk_id, grade in sorted(query.relevance.items()):
            pairs.setdefault(grade, []).append((query_id, chunk_id))

    rng = random.Random(args.seed)
    per_grade = max(1, args.size // max(len(pairs), 1))
    picked: list[tuple[str, str]] = []
    for grade in sorted(pairs):
        candidates = pairs[grade]
        picked.extend(rng.sample(candidates, min(per_grade, len(candidates))))

    # Interleave so the sheet alternates between grades instead of running
    # forty irrelevant passages before the first good one.
    rng.shuffle(picked)

    # The same chunk can be pooled for more than one query; the text lookup
    # rejects duplicate ids.
    seen: set[str] = set()
    picked = [p for p in picked if not (p[1] in seen or seen.add(p[1]))][: args.size]
    n_queries = len({query_id for query_id, _ in picked})

    store = ChromaStore(str(args.chroma_path), args.collection)
    texts = store.get([chunk_id for _, chunk_id in picked])

    scale = "\n".join(
        f"- **{g}** — {t}" for g, t in sorted(GRADE_DEFINITIONS.items(), reverse=True)
    )
    lines = [
        "# Relevance grading sheet",
        "",
        f"{len(picked)} (query, passage) pairs sampled across the benchmark, stratified",
        "by the judge's grade. **The judge's grades are not shown** — they are held in",
        "`human_eval_key.json` and compared only after this sheet is filled in.",
        "",
        "Each item is one retrieved passage, quoted verbatim from the corpus. **These are",
        "not generated answers.** No model wrote them and nothing was summarised or",
        "tidied up; most are forum posts, accident reports, or route descriptions. Grade",
        "the information, not the prose — a blunt one-line forum reply that answers the",
        "question beats a well-written article that does not.",
        "",
        "The same query appears several times, paired with a different passage each time",
        f"({n_queries} queries across these {len(picked)} pairs). Judge each pair on its"
        " own; do not rank a passage against the others for its query.",
        "",
        "Sources are withheld deliberately. Knowing a passage came from an AAC accident",
        "report rather than a forum thread would anchor the grade on authority instead of",
        "on whether it answers the question.",
        "",
        "Fill in each `GRADE:` line with 0, 1, 2, or 3:",
        "",
        scale,
        "",
        "Ask one thing only: if a climber read this passage and nothing else, how much of",
        "*that* question is answered? A route description is not relevant to a training",
        "question merely because both concern climbing.",
        "",
        "---",
        "",
    ]

    key: dict[str, float] = {}
    for i, (query_id, chunk_id) in enumerate(picked, 1):
        record = texts.get(chunk_id)
        if not record or not record.get("text"):
            continue
        key[f"{query_id}|{chunk_id}"] = queries[query_id].relevance[chunk_id]
        passage = " ".join(record["text"].split())[:PASSAGE_CHARS]
        lines += [
            f"### {query_id}|{chunk_id}",
            "",
            f"**Q{i}. {queries[query_id].query}**",
            "",
            f"> {passage}",
            "",
            "GRADE: ",
            "",
        ]

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text("\n".join(lines), encoding="utf-8")
    KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(f"{len(key)} pairs -> {SHEET}")
    print(f"Judge grades withheld in {KEY}")
    print("\nFill in the GRADE lines, then:  python scripts/human_eval.py score")
    return 0


def score(_args: argparse.Namespace) -> int:
    if not SHEET.exists() or not KEY.exists():
        print("error: run `prepare` first", file=sys.stderr)
        return 1

    text = SHEET.read_text(encoding="utf-8")
    judged = {k: float(v) for k, v in json.loads(KEY.read_text()).items()}

    ids = _ITEM_RE.findall(text)
    grades = _GRADE_RE.findall(text)
    if len(ids) != len(grades):
        print(
            f"error: {len(ids)} items but {len(grades)} GRADE lines -- "
            "a line was probably reformatted",
            file=sys.stderr,
        )
        return 1

    human = {i: float(g) for i, g in zip(ids, grades, strict=True) if g}
    blank = len(ids) - len(human)
    if not human:
        print("error: no grades filled in", file=sys.stderr)
        return 1

    result = agreement(judged, human)
    print(f"\n{result.n} pairs graded by hand" + (f" ({blank} left blank)" if blank else ""))
    print(f"  exact            {result.exact:.1%}")
    print(f"  within one grade {result.within_one:.1%}")
    print(f"  relevant or not  {result.binary:.1%}   <- what Recall depends on")
    print(f"\n  mean grade: judge {result.judge_mean:.2f}, human {result.human_mean:.2f}")
    if abs(result.judge_mean - result.human_mean) > 0.3:
        direction = "higher" if result.judge_mean > result.human_mean else "lower"
        print(f"  note: the judge grades systematically {direction} than you do")

    disagreements = sorted(
        ((k, judged[k], human[k]) for k in human if abs(judged[k] - human[k]) > 1),
        key=lambda kv: -abs(kv[1] - kv[2]),
    )
    if disagreements:
        print(f"\n  {len(disagreements)} pairs differ by more than one grade:")
        for key, j, h in disagreements[:10]:
            print(f"    {key.split('|')[0]}  judge {j:.0f} vs you {h:.0f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return prepare(args) if args.command == "prepare" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
