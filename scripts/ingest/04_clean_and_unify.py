"""
CruxBot — Data Cleaning & Schema Unification
==============================================
Cleans all collected data and normalizes into a unified schema
ready for chunking and embedding.

Input:  data/openbeta/, data/kaggle_8a/, data/reddit/
Output: data/unified/cruxbot_unified.json

Cleaning steps:
  1. Remove low-value entries (too short, no name, etc.)
  2. Deduplicate routes across OpenBeta and Kaggle MP
  3. Clean text (HTML, special chars, whitespace)
  4. Build text field for entries lacking descriptions
  5. Normalize all entries to unified schema

Usage:
    python -u scripts/04_clean_and_unify.py
"""

import hashlib
import json
import os
import re
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = "data/unified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum text length thresholds
MIN_TEXT_ROUTE = 20  # Routes can be short (just name + grade)
MIN_TEXT_FORUM = 50  # Forum posts need some substance
MIN_TEXT_ARTICLE = 100  # Articles should have meaningful body
MIN_TEXT_REDDIT = 30  # Reddit posts at least need a title


# ============================================================
# Text cleaning utilities
# ============================================================
def clean_text(text):
    """Clean a text string: remove HTML, normalize whitespace, strip junk."""
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Remove URLs (keep them in source_url field instead)
    # text = re.sub(r'https?://\S+', '', text)

    # Remove excessive special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,;:!?\'\"()\-/°+#@&]", " ", text)

    # Normalize whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_dmca_notice(text):
    """Check if text is a DMCA/copyright complaint instead of real content."""
    dmca_phrases = [
        "did not authorize",
        "copied from mountain project",
        "copyright violation",
        "do not copy",
        "unauthorized reproduction",
        "removed at the request",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in dmca_phrases)


def generate_doc_id(source, identifier):
    """Generate a unique document ID from source and identifier."""
    raw = f"{source}:{identifier}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ============================================================
# Step 1: Load all data
# ============================================================
def load_all_data():
    """Load all collected data files."""
    print("\n📂 Loading all data sources...")

    data = {}

    sources = {
        "openbeta": "data/openbeta/openbeta_routes.json",
        "kaggle_routes": "data/kaggle_8a/kaggle_routes.json",
        "kaggle_forums": "data/kaggle_8a/kaggle_forums.json",
        "kaggle_reviews": "data/kaggle_8a/kaggle_reviews.json",
        "aac_articles": "data/kaggle_8a/kaggle_aac_articles.json",
        "reddit": "data/reddit/reddit_climbing.json",
    }

    for name, path in sources.items():
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data[name] = json.load(f)
            print(f"  {name:<20} {len(data[name]):>8,} entries")
        else:
            print(f"  {name:<20} ⚠️ not found: {path}")
            data[name] = []

    return data


# ============================================================
# Step 2: Deduplicate routes (OpenBeta vs Kaggle MP)
# ============================================================
def deduplicate_routes(openbeta_routes, kaggle_routes):
    """
    Deduplicate routes between OpenBeta and Kaggle MP.
    Strategy: Keep Kaggle MP version (has descriptions + URLs),
    enrich with OpenBeta GPS/area_path where missing.
    Return: (merged_kaggle_routes, unique_openbeta_routes)
    """
    print("\n🔄 Deduplicating routes...")
    print(f"  OpenBeta: {len(openbeta_routes):,}")
    print(f"  Kaggle MP: {len(kaggle_routes):,}")

    # Build a lookup of OpenBeta routes by normalized name
    ob_by_name = {}
    for r in openbeta_routes:
        name = r.get("name", "").lower().strip()
        if name:
            # Store the first occurrence (or one with most info)
            if name not in ob_by_name or r.get("description", ""):
                ob_by_name[name] = r

    # Enrich Kaggle routes with OpenBeta data
    matched = 0
    for kr in kaggle_routes:
        name = kr.get("name", "").lower().strip()
        if name in ob_by_name:
            ob = ob_by_name[name]
            matched += 1
            # Fill in missing GPS from OpenBeta
            if (not kr.get("lat") or kr["lat"] == "nan") and ob.get("lat"):
                kr["lat"] = str(ob["lat"])
                kr["lng"] = str(ob["lng"])
            # Add area_path from OpenBeta
            if ob.get("area_path"):
                kr["area_path"] = ob["area_path"]

    # Find OpenBeta routes NOT in Kaggle (unique to OpenBeta)
    kaggle_names = set(r.get("name", "").lower().strip() for r in kaggle_routes)
    unique_ob = [
        r for r in openbeta_routes if r.get("name", "").lower().strip() not in kaggle_names
    ]

    print(f"  Matched (deduplicated): {matched:,}")
    print(f"  Unique to OpenBeta: {len(unique_ob):,}")
    print(
        f"  After dedup: {len(kaggle_routes):,} (Kaggle) + {len(unique_ob):,} (OpenBeta-only)"
        f" = {len(kaggle_routes) + len(unique_ob):,}"
    )

    return kaggle_routes, unique_ob


# ============================================================
# Step 3: Clean and normalize each source
# ============================================================
def normalize_openbeta_routes(routes):
    """Normalize OpenBeta route entries."""
    entries = []
    removed = {"dmca": 0, "no_name": 0, "no_grade": 0}

    for r in routes:
        name = clean_text(r.get("name", ""))
        if not name:
            removed["no_name"] += 1
            continue

        desc = clean_text(r.get("description", ""))
        if is_dmca_notice(desc):
            desc = ""
            removed["dmca"] += 1

        protection = clean_text(r.get("protection", ""))
        if is_dmca_notice(protection):
            protection = ""

        grade = r.get("grade", "") or ""
        route_type = r.get("type", "unknown")
        state = r.get("state", "")
        area_path = r.get("area_path", "")
        location = f"{area_path}" if area_path else state

        # Build text from structured fields
        text_parts = []
        if name and grade:
            text_parts.append(f"{name} is a {grade} {route_type} climbing route")
        elif name:
            text_parts.append(f"{name} is a {route_type} climbing route")
        if location:
            text_parts.append(f"located in {location}")
        if r.get("first_ascent") and r["first_ascent"] != "unknown":
            text_parts.append(f"First ascent: {clean_text(r['first_ascent'])}")
        if desc:
            text_parts.append(desc)
        if protection and protection != "NA":
            text_parts.append(f"Protection: {protection}")

        text = ". ".join(text_parts)

        entries.append(
            {
                "doc_id": generate_doc_id("openbeta", r.get("uuid", name)),
                "title": name,
                "text": text,
                "content_type": "route",
                "source": "openbeta",
                "source_url": r.get("source_url", "https://openbeta.io"),
                "grade": grade,
                "route_type": route_type,
                "location": location,
                "lat": r.get("lat"),
                "lng": r.get("lng"),
                "metadata": {
                    "state": state,
                    "area_path": area_path,
                    "first_ascent": r.get("first_ascent", ""),
                    "uuid": r.get("uuid", ""),
                },
            }
        )

    print(f"  OpenBeta: {len(entries):,} kept (DMCA removed: {removed['dmca']})")
    return entries


def normalize_kaggle_routes(routes):
    """Normalize Kaggle MP route entries."""
    entries = []
    removed = 0

    for r in routes:
        name = clean_text(r.get("name", ""))
        if not name:
            removed += 1
            continue

        desc = clean_text(r.get("description", ""))
        protection = clean_text(r.get("protection", ""))
        grade = r.get("grade", "") or ""
        route_type = r.get("type", "") or ""
        location = clean_text(r.get("location", ""))
        stars = r.get("avg_stars", "")
        pitches = r.get("pitches", "")
        length = r.get("length", "")

        # Build text
        text_parts = []
        if name:
            type_str = f" {route_type}" if route_type else ""
            grade_str = f" {grade}" if grade else ""
            text_parts.append(f"{name} is a{grade_str}{type_str} climbing route")
        if location:
            text_parts.append(f"located in {location}")
        if stars and str(stars) != "nan" and str(stars) != "":
            text_parts.append(f"rated {stars} stars")
        if pitches and str(pitches) != "nan" and str(pitches) != "":
            text_parts.append(f"{pitches} pitches")
        if length and str(length) != "nan" and str(length) != "":
            text_parts.append(f"{length} feet long")
        if desc:
            text_parts.append(desc)
        if protection:
            text_parts.append(f"Protection: {protection}")

        text = ". ".join(text_parts)

        source_url = r.get("source_url", "")
        if not source_url or source_url == "nan":
            source_url = "https://www.mountainproject.com/"

        entries.append(
            {
                "doc_id": generate_doc_id("kaggle_mp", source_url or name),
                "title": name,
                "text": text,
                "content_type": "route",
                "source": "mountain_project",
                "source_url": source_url,
                "grade": grade,
                "route_type": route_type,
                "location": location,
                "lat": str(r.get("lat", "")) if str(r.get("lat", "")) != "nan" else "",
                "lng": str(r.get("lng", "")) if str(r.get("lng", "")) != "nan" else "",
                "metadata": {
                    "avg_stars": str(stars) if str(stars) != "nan" else "",
                    "pitches": str(pitches) if str(pitches) != "nan" else "",
                    "length": str(length) if str(length) != "nan" else "",
                    "area_path": r.get("area_path", ""),
                },
            }
        )

    print(f"  Kaggle Routes: {len(entries):,} kept (removed: {removed})")
    return entries


def normalize_forums(posts):
    """Normalize forum discussion posts."""
    entries = []
    removed = 0

    for p in posts:
        text = clean_text(p.get("text", ""))
        if len(text) < MIN_TEXT_FORUM:
            removed += 1
            continue

        title = clean_text(p.get("name", ""))

        entries.append(
            {
                "doc_id": generate_doc_id("forum", title + text[:50]),
                "title": title,
                "text": text,
                "content_type": "forum_discussion",
                "source": "mountain_project_forum",
                "source_url": p.get("source_url", "https://www.mountainproject.com/forum"),
                "grade": "",
                "route_type": "",
                "location": "",
                "lat": "",
                "lng": "",
                "metadata": {
                    "num_likes": p.get("num_likes", 0),
                    "post_date": p.get("post_date", ""),
                },
            }
        )

    print(f"  Forums: {len(entries):,} kept (removed short: {removed})")
    return entries


def normalize_reviews(reviews):
    """Normalize gear review entries."""
    entries = []
    removed = 0

    for r in reviews:
        text = clean_text(r.get("text", ""))
        if len(text) < MIN_TEXT_FORUM:
            removed += 1
            continue

        title = clean_text(r.get("name", ""))

        entries.append(
            {
                "doc_id": generate_doc_id("review", title + text[:50]),
                "title": title,
                "text": text,
                "content_type": "gear_review",
                "source": r.get("source", "mountain_project_forum"),
                "source_url": r.get("source_url", ""),
                "grade": "",
                "route_type": "",
                "location": "",
                "lat": "",
                "lng": "",
                "metadata": {
                    "num_likes": r.get("num_likes", 0),
                },
            }
        )

    print(f"  Reviews: {len(entries):,} kept (removed short: {removed})")
    return entries


def normalize_aac(articles):
    """Normalize AAC article entries."""
    entries = []
    removed = 0

    for a in articles:
        text = clean_text(a.get("text", ""))
        if len(text) < MIN_TEXT_ARTICLE:
            removed += 1
            continue

        title = clean_text(a.get("name", ""))
        location = clean_text(a.get("location", ""))

        entries.append(
            {
                "doc_id": generate_doc_id("aac", a.get("source_url", title)),
                "title": title,
                "text": text,
                "content_type": "article",
                "source": "american_alpine_club",
                "source_url": a.get("source_url", ""),
                "grade": "",
                "route_type": "",
                "location": location,
                "lat": "",
                "lng": "",
                "metadata": {
                    "article_type": a.get("article_type", ""),
                    "author": a.get("author", ""),
                    "publication": a.get("publication", ""),
                    "publication_year": a.get("publication_year", ""),
                    "climb_year": a.get("climb_year", ""),
                },
            }
        )

    print(f"  AAC: {len(entries):,} kept (removed short: {removed})")
    return entries


def normalize_reddit(posts):
    """Normalize Reddit post entries."""
    entries = []
    removed = 0

    for p in posts:
        text = clean_text(p.get("text", ""))
        if len(text) < MIN_TEXT_REDDIT:
            removed += 1
            continue

        title = clean_text(p.get("name", ""))
        source_url = p.get("source_url", "")

        entries.append(
            {
                "doc_id": generate_doc_id("reddit", source_url or title),
                "title": title,
                "text": text,
                "content_type": "reddit_post",
                "source": "reddit",
                "source_url": source_url,
                "grade": "",
                "route_type": "",
                "location": "",
                "lat": "",
                "lng": "",
                "metadata": {
                    "subreddit": p.get("subreddit", ""),
                    "score": p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "created_utc": p.get("created_utc", ""),
                },
            }
        )

    print(f"  Reddit: {len(entries):,} kept (removed short: {removed})")
    return entries


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("=" * 60)
    print("🧹 CruxBot Data Cleaning & Unification")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Load
    data = load_all_data()

    # Step 2: Deduplicate routes
    kaggle_routes, unique_ob = deduplicate_routes(data["openbeta"], data["kaggle_routes"])

    # Step 3: Clean and normalize each source
    print("\n🧹 Cleaning and normalizing...")
    all_entries = []

    # Routes (deduplicated)
    all_entries.extend(normalize_kaggle_routes(kaggle_routes))
    all_entries.extend(normalize_openbeta_routes(unique_ob))

    # Forums
    all_entries.extend(normalize_forums(data["kaggle_forums"]))

    # Reviews
    all_entries.extend(normalize_reviews(data["kaggle_reviews"]))

    # AAC Articles
    all_entries.extend(normalize_aac(data["aac_articles"]))

    # Reddit
    all_entries.extend(normalize_reddit(data["reddit"]))

    # Step 4: Final dedup by doc_id
    print("\n🔍 Final deduplication by doc_id...")
    seen = set()
    unique_entries = []
    for e in all_entries:
        if e["doc_id"] not in seen:
            seen.add(e["doc_id"])
            unique_entries.append(e)

    print(f"  Before: {len(all_entries):,} -> After: {len(unique_entries):,}")
    all_entries = unique_entries

    # Step 5: Save
    print("\n💾 Saving unified dataset...")

    json_path = os.path.join(OUTPUT_DIR, "cruxbot_unified.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(json_path) / 1024 / 1024
    print(f"  {json_path} ({len(all_entries):,} entries, {size:.1f} MB)")

    # Step 6: Summary
    print("\n📊 Unified Dataset Summary:")
    print(f"  Total entries: {len(all_entries):,}")

    # By content type
    type_counts = {}
    for e in all_entries:
        ct = e["content_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print("\n  By content type:")
    for ct, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ct:<25} {c:>8,}")

    # By source
    source_counts = {}
    for e in all_entries:
        s = e["source"]
        source_counts[s] = source_counts.get(s, 0) + 1
    print("\n  By source:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<30} {c:>8,}")

    # Text length stats
    lengths = [len(e["text"]) for e in all_entries]
    avg_len = sum(lengths) / max(len(lengths), 1)
    short = sum(1 for n in lengths if n < 100)
    medium = sum(1 for n in lengths if 100 <= n < 500)
    long_text = sum(1 for n in lengths if n >= 500)
    print("\n  Text length distribution:")
    print(f"    Short (<100 chars):   {short:>8,}")
    print(f"    Medium (100-500):     {medium:>8,}")
    print(f"    Long (500+):          {long_text:>8,}")
    print(f"    Average:              {avg_len:>8,.0f} chars")

    # Citation readiness
    with_url = sum(1 for e in all_entries if e.get("source_url", "").startswith("http"))
    print("\n  Citation readiness:")
    print(
        f"    With clickable URL: {with_url:,} ({with_url / max(len(all_entries), 1) * 100:.1f}%)"
    )

    # Schema fields
    print("\n  Unified schema fields:")
    print(f"    {list(all_entries[0].keys())}")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print(f"Unified data: {json_path}")
    print("Next: run chunking & embedding pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
