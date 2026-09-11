"""
Reddit Climbing Data Collection Script
========================================
Collects posts and comments from climbing subreddits using
Reddit's public JSON endpoints (no API key required).

Subreddits: r/climbing, r/bouldering, r/climbharder
Sorting: top posts by year/all time + hot posts

Usage:
    python -u scripts/03_reddit_collect.py

Note: Reddit rate-limits unauthenticated requests.
      We use a 2-second delay between requests to be respectful.
"""

import json
import os
import time
from datetime import datetime

import requests

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = "data/reddit"
DELAY = 2  # Seconds between requests (be respectful to Reddit)
USER_AGENT = "cruxbot/1.0 (academic NLP project)"

SUBREDDITS = [
    "climbing",
    "bouldering",
    "climbharder",
]

# How many pages to fetch per subreddit per sort category
# Each page returns up to 100 posts
PAGES_PER_CATEGORY = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_reddit(url, params=None):
    """Fetch a Reddit JSON endpoint with retry logic."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                # Rate limited, wait and retry
                wait = int(resp.headers.get("Retry-After", 10))
                print(f" [rate limited, waiting {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f" [err: {e}]", end="", flush=True)
                return None
    return None


def fetch_comments(subreddit, post_id):
    """Fetch top comments for a specific post."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = fetch_reddit(url, params={"limit": 20, "sort": "top"})
    time.sleep(DELAY)

    if not data or not isinstance(data, list) or len(data) < 2:
        return []

    comments = []
    for child in data[1].get("data", {}).get("children", []):
        if child.get("kind") != "t1":
            continue
        cdata = child.get("data", {})
        body = cdata.get("body", "")
        if not body or body == "[deleted]" or body == "[removed]":
            continue
        comments.append(
            {
                "body": body,
                "score": cdata.get("score", 0),
                "author": cdata.get("author", ""),
            }
        )

    return comments


def fetch_subreddit_posts(subreddit, sort="top", time_filter="all", max_pages=PAGES_PER_CATEGORY):
    """
    Fetch posts from a subreddit with pagination.

    Args:
        subreddit: Subreddit name (without r/)
        sort: "top", "hot", or "new"
        time_filter: "all", "year", "month" (only used with sort=top)
        max_pages: Number of pages to fetch (100 posts per page)
    """
    posts = []
    after = None

    for _page in range(max_pages):
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {"limit": 100, "raw_json": 1}
        if after:
            params["after"] = after
        if sort == "top":
            params["t"] = time_filter

        data = fetch_reddit(url, params)
        time.sleep(DELAY)

        if not data:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            if child.get("kind") != "t3":
                continue
            pdata = child.get("data", {})

            title = pdata.get("title", "")
            selftext = pdata.get("selftext", "")
            score = pdata.get("score", 0)
            num_comments = pdata.get("num_comments", 0)
            post_id = pdata.get("id", "")
            permalink = pdata.get("permalink", "")
            created = pdata.get("created_utc", 0)

            # Skip low-quality posts
            if score < 2:
                continue

            # Skip image-only posts with no text
            if not selftext and not title:
                continue

            posts.append(
                {
                    "post_id": post_id,
                    "subreddit": subreddit,
                    "title": title,
                    "selftext": selftext,
                    "score": score,
                    "num_comments": num_comments,
                    "permalink": permalink,
                    "created_utc": created,
                    "url": f"https://www.reddit.com{permalink}",
                }
            )

        # Get next page cursor
        after = data.get("data", {}).get("after")
        if not after:
            break

        print(".", end="", flush=True)

    return posts


def collect_subreddit(subreddit):
    """Collect posts and top comments from a subreddit."""
    all_posts = []

    # Fetch top posts (all time)
    print("\n    top/all", end="", flush=True)
    posts = fetch_subreddit_posts(subreddit, sort="top", time_filter="all")
    all_posts.extend(posts)
    print(f" ({len(posts)})", end="", flush=True)

    # Fetch top posts (past year)
    print(" | top/year", end="", flush=True)
    posts = fetch_subreddit_posts(subreddit, sort="top", time_filter="year")
    all_posts.extend(posts)
    print(f" ({len(posts)})", end="", flush=True)

    # Fetch hot posts
    print(" | hot", end="", flush=True)
    posts = fetch_subreddit_posts(subreddit, sort="hot", max_pages=5)
    all_posts.extend(posts)
    print(f" ({len(posts)})", end="", flush=True)

    # Deduplicate by post_id
    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["post_id"] not in seen:
            seen.add(p["post_id"])
            unique_posts.append(p)

    print(f" | unique: {len(unique_posts)}", end="", flush=True)

    # Fetch comments for top posts (those with most comments)
    print("\n    Fetching comments for top posts", end="", flush=True)
    top_posts = sorted(unique_posts, key=lambda x: x["score"], reverse=True)

    # Only fetch comments for top 200 posts to save time
    fetch_limit = min(200, len(top_posts))
    for i, post in enumerate(top_posts[:fetch_limit]):
        comments = fetch_comments(subreddit, post["post_id"])
        post["top_comments"] = comments

        if (i + 1) % 20 == 0:
            print(f" [{i + 1}/{fetch_limit}]", end="", flush=True)

    # Build text entries for RAG
    entries = []
    for post in unique_posts:
        # Build combined text
        text_parts = [f"Title: {post['title']}"]
        if post["selftext"]:
            text_parts.append(post["selftext"])

        # Add top comments
        comments = post.get("top_comments", [])
        if comments:
            comment_texts = [
                c["body"] for c in sorted(comments, key=lambda x: x["score"], reverse=True)[:5]
            ]
            text_parts.append("Top comments: " + " | ".join(comment_texts))

        entries.append(
            {
                "name": post["title"],
                "text": "\n".join(text_parts),
                "subreddit": post["subreddit"],
                "score": post["score"],
                "num_comments": post["num_comments"],
                "created_utc": post["created_utc"],
                "source": "reddit",
                "source_url": post["url"],
                "content_type": "reddit_post",
            }
        )

    return entries


def main():
    print("=" * 60)
    print("🗣️ Reddit Climbing Data Collection")
    print(f"   Subreddits: {', '.join('r/' + s for s in SUBREDDITS)}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_entries = []

    for i, subreddit in enumerate(SUBREDDITS):
        print(f"\n  [{i + 1}/{len(SUBREDDITS)}] r/{subreddit}...")
        entries = collect_subreddit(subreddit)
        all_entries.extend(entries)
        print(f"\n    ✅ {len(entries)} posts collected")

    # Save
    print("\n💾 Saving data...")

    json_path = os.path.join(OUTPUT_DIR, "reddit_climbing.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(json_path) / 1024 / 1024
    print(f"  JSON: {json_path} ({len(all_entries):,} posts, {size:.1f} MB)")

    # Summary
    print("\n📊 Summary:")
    print(f"  Total posts: {len(all_entries):,}")

    for sub in SUBREDDITS:
        count = sum(1 for e in all_entries if e["subreddit"] == sub)
        print(f"    r/{sub:<20} {count:>5,} posts")

    with_text = sum(1 for e in all_entries if len(e.get("text", "")) > 100)
    print(f"\n  With substantial text (>100 chars): {with_text:,}")

    avg_score = sum(e["score"] for e in all_entries) / max(len(all_entries), 1)
    print(f"  Average post score: {avg_score:.0f}")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print(f"Data: {OUTPUT_DIR}/")
    print("All data collection complete! Ready for chunking & embedding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
