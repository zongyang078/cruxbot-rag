"""
Kaggle Mountain Project Dataset - Processing Script
=====================================================
Processes the already-downloaded Mountain Project dataset from Kaggle.
Dataset: pdegner/mountain-project-rotues-and-forums

Files:
  mp_routes.csv        - 116k climbing routes with descriptions & URLs
  discussion_forum.csv - 177k forum discussion posts
  review_forum.csv     - 12k gear review posts
  trailspace.csv       - 1k gear ratings & reviews

Usage:
    python -u scripts/02_kaggle_8a_collect.py

Note: Run download first if not already done:
    kaggle datasets download -d pdegner/mountain-project-rotues-and-forums \
        -p data/kaggle_8a/raw --unzip
"""

import json
import os
from datetime import datetime

import pandas as pd

# ============================================================
# Configuration
# ============================================================
RAW_DIR = "data/kaggle_8a/raw"
OUTPUT_DIR = "data/kaggle_8a"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Step 1: Process route data
# ============================================================
def process_routes():
    """Process mp_routes.csv into RAG-ready format."""
    print("\n⛰️ Processing routes...")

    df = pd.read_csv(os.path.join(RAW_DIR, "mp_routes.csv"), encoding="utf-8", on_bad_lines="skip")
    print(f"  Raw rows: {len(df):,}")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Build structured entries
    entries = []
    for _, row in df.iterrows():
        name = str(row.get("Route", "")).strip()
        location = str(row.get("Location", "")).strip()
        url = str(row.get("URL", "")).strip()
        stars = row.get("Avg Stars", "")
        route_type = str(row.get("Route Type", "")).strip()
        rating = str(row.get("Rating", "")).strip()
        pitches = row.get("Pitches", "")
        length = row.get("Length", "")
        lat = row.get("Area Latitude", "")
        lng = row.get("Area Longitude", "")
        desc = str(row.get("desc", "")).strip()
        protection = str(row.get("protection", "")).strip()

        # Clean up NaN values
        if desc == "nan":
            desc = ""
        if protection == "nan":
            protection = ""
        if route_type == "nan":
            route_type = ""

        # Build text for RAG
        text_parts = []
        if name:
            text_parts.append(f"{name} is a {rating} {route_type} route")
        if location:
            text_parts.append(f"located in {location}")
        if stars and str(stars) != "nan":
            text_parts.append(f"with an average rating of {stars} stars")
        if pitches and str(pitches) != "nan":
            text_parts.append(f"{pitches} pitches")
        if length and str(length) != "nan":
            text_parts.append(f"{length} feet long")
        if desc:
            text_parts.append(f"Description: {desc}")
        if protection:
            text_parts.append(f"Protection: {protection}")

        text = ". ".join(text_parts)

        entries.append(
            {
                "name": name,
                "grade": rating,
                "type": route_type,
                "location": location,
                "avg_stars": str(stars) if str(stars) != "nan" else "",
                "pitches": str(pitches) if str(pitches) != "nan" else "",
                "length": str(length) if str(length) != "nan" else "",
                "lat": str(lat) if str(lat) != "nan" else "",
                "lng": str(lng) if str(lng) != "nan" else "",
                "description": desc,
                "protection": protection,
                "text": text,
                "source": "mountain_project",
                "source_url": url if url != "nan" else "",
                "content_type": "route",
            }
        )

    # Filter out empty entries
    entries = [e for e in entries if e["name"]]

    with_desc = sum(1 for e in entries if e["description"])
    print(f"  Processed: {len(entries):,} routes ({with_desc:,} with description)")

    return entries


# ============================================================
# Step 2: Process forum discussions
# ============================================================
def process_forums():
    """Process discussion_forum.csv into RAG-ready format."""
    print("\n💬 Processing forum discussions...")

    df = pd.read_csv(
        os.path.join(RAW_DIR, "discussion_forum.csv"), encoding="utf-8", on_bad_lines="skip"
    )
    print(f"  Raw rows: {len(df):,}")

    df.columns = df.columns.str.strip()

    entries = []
    for _, row in df.iterrows():
        topic = str(row.get("topic", "")).strip()
        text = str(row.get("text", "")).strip()
        post_date = str(row.get("post_date", "")).strip()
        num_likes = row.get("num_likes", 0)

        if text == "nan" or not text or len(text) < 20:
            continue

        # Only keep posts with some engagement or length
        try:
            likes = int(float(num_likes)) if str(num_likes) != "nan" else 0
        except (ValueError, TypeError):
            likes = 0

        entries.append(
            {
                "name": topic,
                "text": f"Topic: {topic}. {text}",
                "post_date": post_date if post_date != "nan" else "",
                "num_likes": likes,
                "source": "mountain_project_forum",
                "source_url": "https://www.mountainproject.com/forum",
                "content_type": "forum_discussion",
            }
        )

    # Sort by likes and keep top posts for quality
    entries.sort(key=lambda x: x["num_likes"], reverse=True)
    print(
        f"  Processed: {len(entries):,} posts "
        f"(top liked: {entries[0]['num_likes'] if entries else 0})"
    )

    return entries


# ============================================================
# Step 3: Process gear reviews
# ============================================================
def process_reviews():
    """Process review_forum.csv and trailspace.csv."""
    print("\n🔧 Processing gear reviews...")

    entries = []

    # Review forum
    df = pd.read_csv(
        os.path.join(RAW_DIR, "review_forum.csv"), encoding="utf-8", on_bad_lines="skip"
    )
    df.columns = df.columns.str.strip()

    for _, row in df.iterrows():
        topic = str(row.get("topic", "")).strip()
        text = str(row.get("text", "")).strip()
        num_likes = row.get("num_likes", 0)

        if text == "nan" or not text or len(text) < 20:
            continue

        try:
            likes = int(float(num_likes)) if str(num_likes) != "nan" else 0
        except (ValueError, TypeError):
            likes = 0

        entries.append(
            {
                "name": topic,
                "text": f"Gear Review - {topic}: {text}",
                "num_likes": likes,
                "source": "mountain_project_forum",
                "source_url": "https://www.mountainproject.com/forum",
                "content_type": "gear_review",
            }
        )

    print(f"  Review forum: {len(entries):,} posts")

    # Trailspace reviews
    df2 = pd.read_csv(
        os.path.join(RAW_DIR, "trailspace.csv"), encoding="utf-8", on_bad_lines="skip"
    )
    df2.columns = df2.columns.str.strip()

    ts_count = 0
    for _, row in df2.iterrows():
        brand = str(row.get("brand", "")).strip()
        model = str(row.get("model", "")).strip()
        rating = row.get("rating", "")
        rating_text = str(row.get("rating_text", "")).strip()

        if rating_text == "nan" or not rating_text or len(rating_text) < 20:
            continue

        name = f"{brand} {model}".strip()
        rating_str = f" (rated {rating}/5)" if str(rating) != "nan" else ""

        entries.append(
            {
                "name": name,
                "text": f"Gear Review - {name}{rating_str}: {rating_text}",
                "num_likes": 0,
                "source": "trailspace",
                "source_url": "https://www.trailspace.com/",
                "content_type": "gear_review",
            }
        )
        ts_count += 1

    print(f"  Trailspace: {ts_count:,} reviews")
    print(f"  Total gear reviews: {len(entries):,}")

    return entries


# ============================================================
# Step 4: Save all processed data
# ============================================================
def save_data(routes, forums, reviews):
    """Save all processed data to JSON and CSV."""
    print("\n💾 Saving data...")

    datasets = {
        "kaggle_routes.json": routes,
        "kaggle_forums.json": forums,
        "kaggle_reviews.json": reviews,
    }

    for filename, data in datasets.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {filename}: {len(data):,} entries ({size:.1f} MB)")

    # Also save a combined CSV for quick inspection
    all_data = []
    for entry in routes + forums + reviews:
        all_data.append(
            {
                "name": entry.get("name", ""),
                "content_type": entry.get("content_type", ""),
                "text": entry.get("text", "")[:500],  # Truncate for CSV
                "source": entry.get("source", ""),
                "source_url": entry.get("source_url", ""),
            }
        )

    csv_path = os.path.join(OUTPUT_DIR, "kaggle_all_preview.csv")
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Preview CSV: {csv_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("🏔️ Kaggle Mountain Project Dataset Processing")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists(os.path.join(RAW_DIR, "mp_routes.csv")):
        print("\n❌ Raw data not found. Download first:")
        print("   kaggle datasets download -d pdegner/mountain-project-rotues-and-forums \\")
        print("       -p data/kaggle_8a/raw --unzip")
        return

    # Process each data source
    routes = process_routes()
    forums = process_forums()
    reviews = process_reviews()

    # Save
    save_data(routes, forums, reviews)

    # Summary
    total = len(routes) + len(forums) + len(reviews)
    print("\n📊 Summary:")
    print(f"  Routes:     {len(routes):>8,}")
    print(f"  Forums:     {len(forums):>8,}")
    print(f"  Reviews:    {len(reviews):>8,}")
    print(f"  Total:      {total:>8,}")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print(f"Data: {OUTPUT_DIR}/")
    print("Next: run 03_reddit_collect.py for Reddit data")
    print("=" * 60)


if __name__ == "__main__":
    main()
