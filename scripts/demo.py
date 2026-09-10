#!/usr/bin/env python3
"""Ask the running service a question and render the answer readably.

    python scripts/demo.py "classic 5.11a sport routes in Yosemite"
    python scripts/demo.py --search-only "how do I train finger strength"

Retrieval always runs. Generation runs only when `/health` reports a reachable
provider, so this works unchanged against a retrieval-only deployment.

Deliberately stdlib-only: the point of this script is to be the first thing
someone runs after `docker compose up`, and needing to install something first
would defeat that.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.request

DIM, BOLD, CYAN, YELLOW, BLUE, GREEN, RESET = (
    "\033[2m",
    "\033[1m",
    "\033[36m",
    "\033[33m",
    "\033[34m",
    "\033[32m",
    "\033[0m",
)

WRAP = 68
SNIPPET_CHARS = 150


def post(host: str, path: str, payload: dict, stream: bool = False, timeout: float = 300.0):
    request = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    return response if stream else json.loads(response.read())


def health(host: str) -> dict:
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=3) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError) as exc:
        print(f"No service at {host}: {exc}\nStart one with:  docker compose up", file=sys.stderr)
        raise SystemExit(1) from exc


def show_retrieval(result: dict) -> None:
    """Print sources with their scores and the per-stage timing.

    The breakdown is the point. A single total cannot say whether a slow query
    was slow to retrieve or slow to rerank, and those are fixed differently.
    """
    timing, candidates = result["timing"], result["candidates"]
    print(
        f"  {DIM}{candidates['dense']} dense + {candidates['sparse']} sparse"
        f" -> reranked {candidates['reranked']}"
        f" -> top {len(result['sources'])}{RESET}"
    )
    print(
        f"  {DIM}retrieval {timing['retrieval_ms']:.0f}ms"
        f" + rerank {timing['rerank_ms']:.0f}ms{RESET}\n"
    )

    for i, source in enumerate(result["sources"], 1):
        head = f"  {CYAN}[{i}]{RESET} {BOLD}{source['score']:.3f}{RESET}  {source['content_type']}"
        if source["grade"]:
            head += f"  {YELLOW}{source['grade']}{RESET}"
        print(head)

        body = " ".join(source["text"].split())[:SNIPPET_CHARS]
        for line in textwrap.wrap(body, WRAP):
            print(f"      {DIM}{line}{RESET}")

        if url := source["url"]:
            # Marked rather than hidden: some sources in this corpus have no
            # per-post URL, and showing those as citations would mislead.
            note = "" if source["url_is_specific"] else f"  {DIM}(site-level){RESET}"
            print(f"      {BLUE}{url[:WRAP]}{RESET}{note}")
        print()


class StreamWrapper:
    """Wrap streamed text to a width without altering it.

    Token boundaries do not align with word boundaries -- "Here" and "'s"
    arrive separately -- so splitting each token on whitespace and rejoining
    inserts spaces that were never in the answer. Instead, characters are
    buffered until a whitespace character proves a word is complete, and only
    then is a line break considered. Whitespace from the model is reproduced
    exactly; the only thing added is a newline when a line would overrun.
    """

    def __init__(self, width: int, indent: str = "    ") -> None:
        self.width = width
        self.indent = indent
        self.column = len(indent)
        self.pending = ""

    def feed(self, text: str) -> None:
        self.pending += text
        while (index := self._next_break()) is not None:
            self._emit(self.pending[:index], self.pending[index])
            self.pending = self.pending[index + 1 :]

    def close(self) -> None:
        if self.pending:
            self._emit(self.pending, "")
            self.pending = ""

    def _next_break(self) -> int | None:
        for i, char in enumerate(self.pending):
            if char.isspace():
                return i
        return None

    def _emit(self, word: str, separator: str) -> None:
        if separator == "\n":
            print(word)
            print(self.indent, end="", flush=True)
            self.column = len(self.indent)
            return
        if word and self.column + len(word) > self.width:
            print("\n" + self.indent, end="")
            self.column = len(self.indent)
        print(word + separator, end="", flush=True)
        self.column += len(word) + len(separator)


def stream_answer(host: str, query: str, top_k: int) -> None:
    """Print the answer as it arrives, wrapped."""
    print(f"  {GREEN}>{RESET} ", end="", flush=True)
    wrapper = StreamWrapper(WRAP)
    event = ""

    with post(host, "/answer", {"query": query, "top_k": top_k}, stream=True) as response:
        for raw in response:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "token":
                wrapper.feed(json.loads(line[6:])["text"])
            elif line.startswith("data: ") and event == "error":
                print(f"\n  {json.loads(line[6:])['detail']}", end="")
    wrapper.close()
    print("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("query", nargs="*", default=[])
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--search-only", action="store_true", help="Skip generation")
    args = parser.parse_args(argv)

    query = " ".join(args.query) or "classic 5.11a sport routes in Yosemite"
    status = health(args.host)

    print(f"\n{BOLD}?{RESET} {query}\n")
    show_retrieval(post(args.host, "/search", {"query": query, "top_k": args.top_k}))

    if args.search_only:
        return 0
    if not status.get("generation"):
        reason = status.get("generation_error", "no provider configured")
        print(f"  {DIM}Retrieval only — {reason}{RESET}\n")
        return 0

    stream_answer(args.host, query, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
