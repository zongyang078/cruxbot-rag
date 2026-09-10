"""Read the unified corpus into `Document` objects.

The unified file is a JSON array of records produced by the collection and
cleaning pipeline. JSON Lines is also accepted and preferred for anything
large: a 373 MB array has to be parsed in full before the first record is
available, whereas JSONL streams one record at a time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cruxbot.types import Document

# Fields promoted out of the raw record onto the Document; anything else is
# kept in `metadata` rather than silently dropped.
_PROMOTED = frozenset(
    {"doc_id", "text", "content_type", "title", "source", "source_url", "grade", "location"}
)


def to_document(record: dict[str, Any]) -> Document:
    """Convert one raw record into a Document.

    Missing fields become empty strings rather than None, so that downstream
    code can treat every field as a string without guarding each access.
    """
    return Document(
        doc_id=str(record.get("doc_id", "")),
        text=str(record.get("text") or ""),
        content_type=str(record.get("content_type") or ""),
        title=str(record.get("title") or ""),
        source=str(record.get("source") or ""),
        source_url=str(record.get("source_url") or ""),
        grade=str(record.get("grade") or ""),
        location=str(record.get("location") or ""),
        metadata={k: v for k, v in record.items() if k not in _PROMOTED},
    )


def load_documents(path: str | Path) -> Iterator[Document]:
    """Yield documents from a .json array or a .jsonl stream.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if a .json file does not contain a list.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No corpus at {path}")

    if path.suffix == ".jsonl":
        yield from _load_jsonl(path)
    else:
        yield from _load_json_array(path)


def _load_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line := line.strip():
                yield to_document(json.loads(line))


def _load_json_array(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} holds {type(records).__name__}, expected a list")
    for record in records:
        yield to_document(record)


def to_jsonl(source: str | Path, destination: str | Path) -> int:
    """Rewrite a JSON array corpus as JSON Lines, returning the record count.

    Worth doing once for a large corpus: it makes every later pass streamable
    and drops peak memory during indexing from gigabytes to kilobytes.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with destination.open("w", encoding="utf-8") as out:
        for document in _load_json_array(Path(source)):
            record = {
                "doc_id": document.doc_id,
                "text": document.text,
                "content_type": document.content_type,
                "title": document.title,
                "source": document.source,
                "source_url": document.source_url,
                "grade": document.grade,
                "location": document.location,
                **document.metadata,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
