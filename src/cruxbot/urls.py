"""Source URL quality checks.

Not every document in the corpus carries a usable citation. The Mountain
Project forum dump (99k entries, the primary source for training advice) was
collected from a Kaggle mirror that did not preserve per-post URLs, so every
one of those entries shares a single site-level link. Presenting that to a user
as a citation is misleading: the link is real, but it does not lead to the
passage the answer was drawn from.

Rather than hide this, the pipeline labels such URLs so the UI can render them
differently and the evaluation harness can score citation quality honestly.

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Site- or section-level URLs that carry no document-level information.
# fmt: off
GENERIC_URLS: frozenset[str] = frozenset({
    "mountainproject.com",
    "www.mountainproject.com",
    "mountainproject.com/forum",
    "www.mountainproject.com/forum",
    "reddit.com",
    "www.reddit.com",
    "openbeta.io",
    "www.openbeta.io",
    "publications.americanalpineclub.org",
})
# fmt: on


def canonical(url: str) -> str:
    """Reduce a URL to `host/path` with no scheme, trailing slash, or query.

    The original implementation compared raw strings against a denylist, so
    `https://www.mountainproject.com/forum/` (trailing slash) and
    `http://mountainproject.com/forum?page=2` both slipped through as if they
    were specific citations.
    """
    if not url:
        return ""
    parsed = urlparse(url.strip() if "//" in url else f"//{url.strip()}", scheme="https")
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{host}{path}" if path else host


def is_specific(url: str) -> bool:
    """True when the URL identifies a specific document, not just a site.

    An empty or unparseable URL is not specific.
    """
    key = canonical(url)
    if not key:
        return False
    return key not in GENERIC_URLS
