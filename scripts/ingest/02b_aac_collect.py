"""
American Alpine Club (AAC) Articles - Processing Script
=========================================================
Processes the AAC articles dataset from Kaggle.
Contains 28k articles: climbing reports, accident analyses, new route records.

Dataset: iantonopoulos/american-alpine-club-articles
License: CC0-1.0

Usage:
    python -u scripts/02b_aac_collect.py
"""

import json
import os
from datetime import datetime

import pandas as pd

# ============================================================
# Configuration
# ============================================================
RAW_PATH = "data/kaggle_8a/raw/aac/articles.csv"
OUTPUT_DIR = "data/kaggle_8a"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("🏔️ AAC Articles Processing")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check file exists
    if not os.path.exists(RAW_PATH):
        print(f"\n❌ File not found: {RAW_PATH}")
        print("   Download first:")
        print("   kaggle datasets download -d iantonopoulos/american-alpine-club-articles \\")
        print("       -p data/kaggle_8a/raw/aac --unzip")
        return

    # Load data
    print("\n📖 Loading articles...")
    df = pd.read_csv(RAW_PATH, encoding="utf-8", on_bad_lines="skip")
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Article type distribution
    print("\n  Article types:")
    for t, c in df["type"].value_counts().head(10).items():
        print(f"    {t:<40} {c:>6,}")

    # Process into RAG-ready format
    print("\n⚙️ Processing articles...")
    entries = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        body = str(row.get("body", "")).strip()
        location = str(row.get("location", "")).strip()
        article_type = str(row.get("type", "")).strip()
        url = str(row.get("url", "")).strip()
        author = str(row.get("author", "")).strip()
        pub = str(row.get("publication", "")).strip()
        pub_year = row.get("publication_year", "")
        climb_year = row.get("climb_year", "")

        # Skip entries with no meaningful body text
        if body == "nan" or not body or len(body) < 50:
            continue

        # Clean NaN values
        if title == "nan":
            title = ""
        if location == "nan":
            location = ""
        if article_type == "nan":
            article_type = ""
        if url == "nan":
            url = ""
        if author == "nan":
            author = ""
        if pub == "nan":
            pub = ""

        # Build text for RAG
        text_parts = []
        if title:
            text_parts.append(title)
        if location:
            text_parts.append(f"Location: {location}")
        if article_type:
            text_parts.append(f"Type: {article_type}")
        if author:
            text_parts.append(f"Author: {author}")
        text_parts.append(body)

        entries.append(
            {
                "name": title,
                "article_type": article_type,
                "location": location,
                "author": author,
                "publication": pub,
                "publication_year": str(pub_year) if str(pub_year) != "nan" else "",
                "climb_year": str(climb_year) if str(climb_year) != "nan" else "",
                "text": "\n".join(text_parts),
                "body_length": len(body),
                "source": "american_alpine_club",
                "source_url": url,
                "content_type": "article",
            }
        )

    print(f"  Processed: {len(entries):,} articles")

    # Stats
    avg_len = sum(e["body_length"] for e in entries) / max(len(entries), 1)
    print(f"  Average body length: {avg_len:,.0f} characters")

    with_location = sum(1 for e in entries if e["location"])
    print(f"  With location: {with_location:,} ({with_location / max(len(entries), 1) * 100:.1f}%)")

    # Type breakdown
    type_counts = {}
    for e in entries:
        t = e["article_type"] or "unknown"
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\n  Type breakdown:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<40} {c:>6,}")

    # Save
    print("\n💾 Saving data...")

    json_path = os.path.join(OUTPUT_DIR, "kaggle_aac_articles.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(json_path) / 1024 / 1024
    print(f"  JSON: {json_path} ({len(entries):,} articles, {size:.1f} MB)")

    csv_path = os.path.join(OUTPUT_DIR, "kaggle_aac_preview.csv")
    preview = [
        {
            "name": e["name"],
            "article_type": e["article_type"],
            "location": e["location"],
            "publication_year": e["publication_year"],
            "body_length": e["body_length"],
            "source_url": e["source_url"],
        }
        for e in entries
    ]
    pd.DataFrame(preview).to_csv(csv_path, index=False)
    print(f"  CSV:  {csv_path}")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print(f"Data: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
