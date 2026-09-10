"""Split documents into retrieval units.

Chunk size is a retrieval parameter, not a formatting detail. Too large and a
chunk's embedding averages several topics, so it matches everything weakly;
too small and a passage loses the context that made it an answer. The corpus
here spans two orders of magnitude in document length -- a route description
is a paragraph, an AAC accident report is several pages -- so the size is set
per content type rather than globally.

Chunk ids are derived deterministically from the parent document id, so
re-chunking a document that changed produces the same ids and overwrites its
previous chunks instead of accumulating duplicates alongside them. That is
what makes incremental re-indexing safe to run repeatedly.

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

from cruxbot.types import Document

# Chunk sizes are specified in tokens but enforced in characters, since running
# a tokenizer over 338k documents to make a splitting decision is not worth it.
# Four characters per token is the usual approximation for English prose.
CHARS_PER_TOKEN = 4

# (chunk_tokens, overlap_tokens) per content type. Longer, more discursive
# sources get larger chunks and more overlap: an accident report builds context
# across paragraphs, whereas a gear review states its verdict in a sentence.
CHUNK_CONFIG: dict[str, tuple[int, int]] = {
    "route": (300, 50),
    "forum_discussion": (400, 75),
    "reddit_post": (400, 75),
    "article": (500, 100),
    "gear_review": (350, 50),
}
DEFAULT_CONFIG = (400, 75)

# Tried in order. Earlier separators preserve more structure, so text is only
# broken at a weaker boundary when a stronger one leaves a piece too large.
# The empty string is the last resort: split mid-word rather than emit a chunk
# that exceeds the limit.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "")


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: Sequence[str] = SEPARATORS,
) -> list[str]:
    """Split text into overlapping chunks of at most `chunk_size` characters.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of trailing context repeated at the start of the
            next chunk, so a sentence spanning a boundary is retrievable from
            either side.
        separators: Boundary candidates, strongest first.

    Raises:
        ValueError: if chunk_size is not positive, or overlap is negative or
            not smaller than chunk_size (which would never advance).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap {overlap} must be smaller than chunk_size {chunk_size}")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _split_to_pieces(text, chunk_size, list(separators))
    return _merge(pieces, chunk_size, overlap)


def _split_to_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into fragments no longer than chunk_size, preferring strong
    boundaries. Fragments are not yet merged into final chunks."""
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:
        # Out of boundaries: hard-split so the size guarantee always holds.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, rest = separators[0], separators[1:]
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    if len(parts) == 1:
        # This separator does not occur; try a weaker one.
        return _split_to_pieces(text, chunk_size, rest)

    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Put the separator back, except after the final part, so that
        # rejoining chunks reproduces the original text.
        restored = part + separator if i < len(parts) - 1 else part
        if not restored:
            continue
        if len(restored) <= chunk_size:
            pieces.append(restored)
        else:
            pieces.extend(_split_to_pieces(restored, chunk_size, rest))
    return pieces


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack fragments into chunks, carrying `overlap` characters over.

    The carried overlap competes with the next fragment for the same budget.
    When a fragment already fills the chunk on its own -- which is exactly what
    happens to text with no usable separator, since it arrives pre-split at the
    size limit -- the overlap is trimmed to whatever still fits, down to none.
    Carrying it unconditionally would emit chunks longer than `chunk_size`.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            carry = current[-overlap:] if overlap else ""
            if len(carry) + len(piece) > chunk_size:
                carry = carry[len(carry) - max(chunk_size - len(piece), 0) :]
            current = carry
        current += piece

    if stripped := current.strip():
        chunks.append(stripped)
    return [c for c in chunks if c]


def chunk_document(document: Document) -> list[Document]:
    """Split one document into chunk documents, inheriting its metadata.

    A document short enough to fit in one chunk is returned as a single chunk
    rather than passed through unchanged, so that every downstream stage sees
    the same shape regardless of source length.
    """
    text = (document.text or "").strip()
    if not text:
        return []

    chunk_tokens, overlap_tokens = CHUNK_CONFIG.get(document.content_type, DEFAULT_CONFIG)
    pieces = split_text(
        text,
        chunk_size=chunk_tokens * CHARS_PER_TOKEN,
        overlap=overlap_tokens * CHARS_PER_TOKEN,
    )

    return [
        document.as_chunk(text=piece, index=index, total=len(pieces))
        for index, piece in enumerate(pieces)
    ]
