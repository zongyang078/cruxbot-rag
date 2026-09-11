# CruxBot — Data Collection & Processing Documentation

## 1. Project Overview

CruxBot is a Retrieval-Augmented Generation (RAG) system for rock climbing. This document describes all data sources, collection methods, processing pipeline, unified schema, and sample data.

**Final dataset: 338,433 unified entries from 6 sources across 5 content types.**

---

## 2. Data Sources Summary

| # | Source | Raw File | Entries | File Size | Collection Method | License |
|---|--------|----------|---------|-----------|-------------------|---------|
| 1 | OpenBeta | `openbeta_routes.json` | 204,463 | 115 MB | GraphQL API | CC0 |
| 2 | Mountain Project (Kaggle) | `mp_routes.csv` | 116,700 | 70 MB | Kaggle CLI download | Copyright-authors |
| 3 | MP Forums (Kaggle) | `discussion_forum.csv` | 141,837 | 58 MB | Kaggle CLI download | Copyright-authors |
| 4 | AAC Articles (Kaggle) | `articles.csv` | 27,828 | 78 MB | Kaggle CLI download | CC0 |
| 5 | Reddit | `reddit_climbing.json` | 2,519 | 1.6 MB | Public JSON endpoints | Reddit API ToS |
| 6 | Gear Reviews (Kaggle) | `review_forum.csv` + `trailspace.csv` | 10,774 | 5.8 MB | Kaggle CLI download | Copyright-authors |

---

## 3. Detailed Source Descriptions

### 3.1 OpenBeta (204,463 routes)

**What it is:** OpenBeta is a 501(c)(3) nonprofit that maintains an open-source climbing route database under CC0 license. Data originates from user contributions on Mountain Project and is maintained independently after MP's API was deprecated in 2020.

**Collection method:**
- **API:** GraphQL endpoint at `https://api.openbeta.io` (no API key required)
- **Strategy:** BFS (breadth-first search) traversal using UUID-based `area(uuid)` queries
  1. Query `USA` to get all 49 states with UUIDs
  2. Query each state to get regions with UUIDs
  3. For each region, query children and their climbs
  4. If a child has `totalClimbs > 0` but 0 direct climbs, queue its UUID for deeper traversal
- **Concurrency:** 5 parallel threads (ThreadPoolExecutor), 0.5s delay per request
- **Dedup:** By route UUID
- **Script:** `scripts/01_openbeta_collect.py`

**Raw fields (17):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `name` | string | `"Moonlight Buttress"` | Route name |
| `uuid` | string | `"147bff6c-11a2-..."` | OpenBeta unique ID |
| `grade` | string | `"5.12d"` | Best available grade (YDS/V-scale/French) |
| `grade_yds` | string | `"5.12d"` | Yosemite Decimal System grade |
| `grade_vscale` | string | `null` | V-scale (bouldering) |
| `grade_french` | string | `"7c"` | French sport climbing grade |
| `type` | string | `"trad"` | Route type (sport/trad/boulder/tr/alpine/ice/mixed/aid) |
| `description` | string | `""` | Route description (only 0.6% populated) |
| `protection` | string | `""` | Gear/protection notes |
| `location_info` | string | `""` | Additional location details |
| `first_ascent` | string | `"unknown"` | First ascent info |
| `lat` | float | `37.2716` | Latitude |
| `lng` | float | `-112.9478` | Longitude |
| `state` | string | `"Utah"` | US state |
| `area_path` | string | `"Utah > Zion National Park > ..."` | Full area hierarchy |
| `source_url` | string | `"https://openbeta.io"` | Citation URL |
| `source` | string | `"openbeta"` | Source identifier |

**Key stats:**
- 47 US states covered
- Type distribution: bouldering 75,749 | sport 62,750 | trad 61,529 | top-rope 19,124 | alpine 6,240
- Only 0.6% have text descriptions — mostly structured data
- 100% have GPS coordinates

**How it's used in RAG:** Structured fields are composed into natural language text for embedding (e.g., "Moonlight Buttress is a 5.12d trad climbing route located in Utah > Zion National Park"). Provides geographic coverage and route metadata for location-based queries.

---

### 3.2 Mountain Project Routes via Kaggle (116,700 routes)

**What it is:** A Kaggle dataset (`pdegner/mountain-project-rotues-and-forums`) containing scraped Mountain Project data. Unlike OpenBeta, this dataset has rich text descriptions for 99.8% of routes plus clickable Mountain Project URLs.

**Collection method:**
- Kaggle CLI: `kaggle datasets download -d pdegner/mountain-project-rotues-and-forums --unzip`
- Processing: Pandas CSV parsing, text field composition
- **Script:** `scripts/02_kaggle_8a_collect.py`

**Raw fields (14):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `Route` | string | `"Access Denied"` | Route name |
| `Location` | string | `"El Mirador > El Potrero Chico > ..."` | Full location hierarchy |
| `URL` | string | `"https://www.mountainproject.com/route/110149834/..."` | Mountain Project URL |
| `Avg Stars` | float | `2.9` | Average user rating (1-4 stars) |
| `Route Type` | string | `"Sport"` | Climbing style |
| `Rating` | string | `"5.10b/c"` | Difficulty grade |
| `Pitches` | int | `4` | Number of pitches |
| `Length` | float | `350.0` | Route length in feet |
| `Area Latitude` | float | `25.95044` | Latitude |
| `Area Longitude` | float | `-100.47755` | Longitude |
| `desc` | string | `"This is a really great route..."` | Full text description |
| `protection` | string | `"12 draws + 60m Rope"` | Gear/protection beta |
| `num_votes` | int | `15` | Number of user votes |

**Key stats:**
- 99.8% have text descriptions (high RAG value)
- 100% have clickable Mountain Project URLs (perfect for citations)
- Covers international routes (not limited to US)
- Dataset from ~2020 (static snapshot)

**How it's used in RAG:** Primary source for route queries. Rich descriptions provide detailed beta (move-by-move info, gear lists, approach instructions). URLs serve as clickable citations in RAG responses.

---

### 3.3 Mountain Project Forums via Kaggle (141,837 posts)

**What it is:** Forum discussion posts from Mountain Project covering training, gear, technique, partner finding, and route beta discussions.

**Collection method:**
- Same Kaggle download as Source 3.2
- Parsed from `discussion_forum.csv`
- **Script:** `scripts/02_kaggle_8a_collect.py`

**Raw fields (8):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | string | `"In need of an XXS harness"` | Discussion topic/title |
| `page_num` | int | `1` | Page number in thread |
| `post_num` | int | `0` | Post position in page |
| `text` | string | `"One of my climbing partners..."` | Post body text |
| `join_date` | string | `"Jul 2012"` | Author's join date |
| `post_date` | string | `"Jul 10 2012"` | Post date |
| `num_likes` | int | `10000` | Number of likes |
| `pred_label` | string | `"gear"` | Predicted topic label (in labeled_forums.csv) |

**Key stats:**
- Time range: April 2009 – September 2019
- Average post length: ~400 characters
- Topics: training, gear, technique, route beta, partner finding, safety

**How it's used in RAG:** Answers subjective/experience-based questions ("What harness fits a small person?", "How do I train for overhangs?"). Community-sourced advice with real-world experience.

---

### 3.4 American Alpine Club Articles via Kaggle (27,828 articles)

**What it is:** Published articles from the American Alpine Club's two journals: the American Alpine Journal (AAJ) and Accidents in North American Climbing (ANAM). Contains accident reports, expedition records, feature articles, and book reviews spanning nearly a century.

**Collection method:**
- Kaggle CLI: `kaggle datasets download -d iantonopoulos/american-alpine-club-articles --unzip`
- Parsed from `articles.csv`
- **Script:** `scripts/02b_aac_collect.py`

**Raw fields (10):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `url` | string | `"https://publications.americanalpineclub.org/..."` | Article URL |
| `type` | string | `"Accident Reports"` | Article category |
| `publication` | string | `"ANAM"` | Journal name (AAJ or ANAM) |
| `title` | string | `"Fatal Lead Fall on Rock"` | Article title |
| `location` | string | `"Kentucky, Red River Gorge, ..."` | Incident/expedition location |
| `body` | string | `(full article text)` | Complete article body |
| `climb_year` | int | `2019` | Year of climb/incident |
| `link_to_pdf` | string | `(URL)` | Link to PDF version |
| `author` | string | `"Wolfe County Search and Rescue"` | Author name |
| `publication_year` | int | `2020` | Year published |

**Key stats:**
- Average body length: 2,685 characters (long-form, high RAG value)
- Article types: Climbs & Expeditions 17,188 | Accident Reports 5,186 | Feature Articles 1,489
- Year range: 1929–2023
- 17.6% include specific location metadata
- CC0 license

**How it's used in RAG:** Uniquely valuable for safety queries ("What are common lead climbing accidents?", "How do I safely clean an anchor?"). The 5,186 accident reports provide expert analysis unavailable elsewhere. Expedition records answer questions about remote/alpine climbing.

---

### 3.5 Reddit (2,519 posts)

**What it is:** Recent posts and top comments from climbing subreddits, filling the 2020–2026 recency gap that other sources don't cover.

**Collection method:**
- **API:** Reddit's public JSON endpoints (no API key, no OAuth required)
  - URL pattern: `https://www.reddit.com/r/{subreddit}/{sort}.json?t={time}&limit=100`
  - Pagination via `after` cursor
- **Subreddits:** r/climbing (1,203), r/bouldering (1,316), r/climbharder (0 — blocked by rate limit)
- **Sorting:** top/all-time + top/year + hot
- **Comments:** Top 5 comments fetched for top 200 posts per subreddit
- **Rate limiting:** ~10 requests/minute for unauthenticated access, auto-retry with backoff
- **Script:** `scripts/03_reddit_collect.py`

**Raw fields (9):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `title` | string | `"How to improve overhang climbing"` | Post title |
| `selftext` | string | `"I've been stuck at V4..."` | Post body (empty for image posts) |
| `score` | int | `923` | Upvote count |
| `num_comments` | int | `142` | Number of comments |
| `subreddit` | string | `"climbing"` | Source subreddit |
| `permalink` | string | `"/r/climbing/comments/..."` | Reddit permalink |
| `created_utc` | float | `1711300800` | Unix timestamp |
| `url` | string | `"https://www.reddit.com/r/..."` | Full URL |
| `top_comments` | list | `[{"body": "...", "score": 45}]` | Top 5 comments with scores |

**Key stats:**
- Average post score: 923 upvotes (high-quality content only)
- 1,297 posts with substantial text (>100 chars)
- Time range: 2024–2026 (fills recency gap)

**How it's used in RAG:** Provides current community knowledge. Answers trend-sensitive questions ("What are the best climbing shoes in 2025?", "Any new routes at Red River Gorge?"). Top comments add diverse perspectives and practical advice.

---

### 3.6 Gear Reviews (10,774 reviews)

**What it is:** Equipment reviews from Mountain Project forums and Trailspace.com covering shoes, ropes, harnesses, protection devices, and other climbing gear.

**Collection method:**
- Parsed from `review_forum.csv` (9,676 posts) + `trailspace.csv` (1,098 reviews)
- **Script:** `scripts/02_kaggle_8a_collect.py`

**Raw fields (review_forum.csv — 8 fields):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | string | `"Plaquette/Guide Style Device"` | Review topic |
| `text` | string | `"Hi all, I am in the market..."` | Review text |
| `num_likes` | int | `5` | Number of likes |
| `post_date` | string | `"Sep 2017"` | Post date |

**Raw fields (trailspace.csv — 4 fields):**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `brand` | string | `"la-sportiva"` | Gear brand |
| `model` | string | `"makalu"` | Product model |
| `rating` | float | `5.0` | Rating out of 5 |
| `rating_text` | string | `"Best boots I have ever owned..."` | Review text |

**How it's used in RAG:** Answers gear-specific questions ("What's the best belay device for multipitch?", "La Sportiva vs Scarpa shoes?"). User reviews provide practical, experience-based recommendations.

---

## 4. Data Processing Pipeline

### 4.1 Pipeline Overview

```
Raw Data (6 sources, 504k entries)
    │
    ▼
Step 1: Deduplication
    - OpenBeta vs Kaggle MP: 76.6% route name overlap
    - Keep Kaggle version (has descriptions + URLs)
    - Enrich with OpenBeta GPS data
    │
    ▼
Step 2: Cleaning
    - Remove HTML tags and entities
    - Remove DMCA/copyright complaint text (7 found)
    - Filter entries below minimum text length
    - Normalize whitespace
    │
    ▼
Step 3: Schema Unification
    - Map all sources to unified 12-field schema
    - Build text field for routes lacking descriptions
    - Preserve original metadata in nested dict
    │
    ▼
Step 4: Final Dedup by doc_id
    - MD5 hash of source + identifier
    │
    ▼
Unified Dataset (338,433 entries, 369 MB)
```

**Script:** `scripts/04_clean_and_unify.py`

### 4.2 Cleaning Rules

| Rule | Affected Source | Entries Removed |
|------|----------------|-----------------|
| Text < 50 chars | Forums | 556 |
| Text < 30 chars | Reddit | 147 |
| DMCA complaints | OpenBeta | 7 |
| Short reviews < 50 chars | Gear Reviews | 10 |
| Duplicate routes (name match) | OpenBeta vs Kaggle | 94,511 merged |
| Duplicate doc_id | All | 46,410 |

### 4.3 Route Text Composition

OpenBeta routes lack text descriptions (only 0.6% have them). Structured fields are composed into natural language:

**Before (raw OpenBeta):**
```json
{
  "name": "Rockocco",
  "grade": "5.6",
  "type": "trad",
  "state": "California",
  "area_path": "California > Central Coast > Santa Barbara > San Ysidro",
  "description": ""
}
```

**After (unified):**
```json
{
  "title": "Rockocco",
  "text": "Rockocco is a 5.6 trad climbing route. located in California > Central Coast > Santa Barbara > San Ysidro",
  "content_type": "route",
  "source_url": "https://openbeta.io"
}
```

---

## 5. Unified Schema

Every entry in the final dataset follows this schema:

```json
{
  "doc_id":       "f83c6348003e",
  "title":        "Access Denied",
  "text":         "Access Denied is a 5.10b/c Sport climbing route...",
  "content_type": "route",
  "source":       "mountain_project",
  "source_url":   "https://www.mountainproject.com/route/110149834/access-denied",
  "grade":        "5.10b/c",
  "route_type":   "Sport",
  "location":     "El Mirador > El Potrero Chico > Nuevo Leon > ...",
  "lat":          "25.95044",
  "lng":          "-100.47755",
  "metadata":     {"avg_stars": "2.9", "pitches": "4", "length": "350.0"}
}
```

### 5.1 Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `doc_id` | string | Yes | Unique 12-char MD5 hash (source + identifier). Used for dedup and chunk tracing. |
| `title` | string | Yes | Human-readable title. Route name, post title, or article title. |
| `text` | string | Yes | Full text for embedding. The primary field used for vector search. Composed from description + structured fields for routes, or raw text for forums/articles. |
| `content_type` | string | Yes | One of: `route`, `forum_discussion`, `article`, `gear_review`, `reddit_post`. Used for metadata filtering during retrieval. |
| `source` | string | Yes | One of: `openbeta`, `mountain_project`, `mountain_project_forum`, `american_alpine_club`, `reddit`, `trailspace`. Used for citation attribution. |
| `source_url` | string | Yes | Clickable URL to original content. Used for the citation link in RAG responses. **100% populated.** |
| `grade` | string | No | Climbing difficulty (e.g., "5.10b", "V4", "7a"). Only populated for routes. Empty for forums/articles. |
| `route_type` | string | No | Climbing style (e.g., "sport", "trad", "boulder"). Only for routes. |
| `location` | string | No | Geographic location as text. Area hierarchy for routes, incident location for AAC articles. |
| `lat` | string | No | GPS latitude. Populated for most routes. |
| `lng` | string | No | GPS longitude. Populated for most routes. |
| `metadata` | dict | Yes | Source-specific fields preserved as key-value pairs. Varies by source (see below). |

### 5.2 Metadata by Content Type

**route:**
```json
{"avg_stars": "3.5", "pitches": "4", "length": "350.0", "state": "Utah", "area_path": "Utah > Zion", "first_ascent": "1971", "uuid": "..."}
```

**forum_discussion:**
```json
{"num_likes": 42, "post_date": "Jul 10 2012"}
```

**article:**
```json
{"article_type": "Accident Reports", "author": "Wolfe County SAR", "publication": "ANAM", "publication_year": "2020", "climb_year": "2019"}
```

**gear_review:**
```json
{"num_likes": 5}
```

**reddit_post:**
```json
{"subreddit": "climbing", "score": 923, "num_comments": 142, "created_utc": "1711300800"}
```

---

## 6. Sample Entries (Real Data)

### 6.1 Route (Mountain Project)
```json
{
  "doc_id": "f83c6348003e",
  "title": "Access Denied",
  "text": "Access Denied is a 5.10b/c Sport climbing route. located in El Mirador > El Potrero Chico > Nuevo Leon > Northern Mexico. rated 2.9 stars. 4 pitches. 350.0 feet long. This is a really great route with awesome exposure and a really cool summit. It climbs obvious dihedrals and good face climbing up to the ridge. P1: 5.10a - 11 bolts P2: 5.10a - 9 bolts P3: 5.10c - 9 bolts - crux. P4: 5.9 - 8 bolts - Jugs to the summit.",
  "content_type": "route",
  "source": "mountain_project",
  "source_url": "https://www.mountainproject.com/route/110149834/access-denied"
}
```

### 6.2 Forum Discussion (Mountain Project Forum)
```json
{
  "doc_id": "7538211655bb",
  "title": "In need of an XXS harness for a friend",
  "text": "Topic: In need of an XXS harness for a friend. One of my climbing partners is currently using an XS Mammut Vision climbing harness that was the smallest women's harness we could find at the time. When completely tightened it still is a little loose on her waist and legs, she's like 5' 100lbs. Things have been fine until this weekend when she took a slightly penduluming lead fall...",
  "content_type": "forum_discussion",
  "source": "mountain_project_forum",
  "source_url": "https://www.mountainproject.com/forum"
}
```

### 6.3 Article (American Alpine Club)
```json
{
  "doc_id": "d11522fbc4e0",
  "title": "The Great Game: A Difficult New Route up Koyo Zom, 51 Years After the First Ascent",
  "text": "The Great Game: A Difficult New Route up Koyo Zom, 51 Years After the First Ascent. Location: Pakistan, Hindu Raj, Yarkhun Valley. Type: Feature Article. Author: Tom Livingstone. Koyo Zom (6,877 meters) from the Yarkhun Valley. The northwest face is near the right skyline, with a steep icefield leading to a difficult mixed headwall...",
  "content_type": "article",
  "source": "american_alpine_club",
  "source_url": "https://publications.americanalpineclub.org/articles/13201215543"
}
```

### 6.4 Gear Review (Mountain Project Forum)
```json
{
  "doc_id": "c5ae8e362b0e",
  "title": "Plaquette/Guide Style Device",
  "text": "Gear Review - Plaquette/Guide Style Device: Hi all, I am in the market for a new plaquette style device. I currently have a Reverso 4 but I am contemplating between a Pivot, ATC Guide, or Reverso. I know they all operate essentially the same. When updating my device I am looking for the device that has the least friction when bringing up a second in Guide Mode...",
  "content_type": "gear_review",
  "source": "mountain_project_forum",
  "source_url": "https://www.mountainproject.com/forum"
}
```

### 6.5 Reddit Post
```json
{
  "doc_id": "35d00232c914",
  "title": "ICU nurse Alex Pretti, killed by ICE agents in Minneapolis today",
  "text": "Title: ICU nurse Alex Pretti, killed by ICE agents in Minneapolis today. Top comments: Swarmed by at minimum 4 agents, was on the ground when shot...",
  "content_type": "reddit_post",
  "source": "reddit",
  "source_url": "https://www.reddit.com/r/climbing/comments/..."
}
```

---

## 7. Final Dataset Statistics

```
Total entries:          338,433
Total file size:        369 MB (JSON)

By content type:
  route                 202,598  (59.9%)
  forum_discussion       99,173  (29.3%)
  article                27,828  ( 8.2%)
  gear_review             6,462  ( 1.9%)
  reddit_post             2,372  ( 0.7%)

By source:
  mountain_project       116,700  (34.5%)
  mountain_project_forum 104,763  (30.9%)
  openbeta                85,898  (25.4%)
  american_alpine_club    27,828  ( 8.2%)
  reddit                   2,372  ( 0.7%)
  trailspace                 872  ( 0.3%)

Text length distribution:
  Short  (<100 chars):    15,835  ( 4.7%)
  Medium (100-500 chars): 223,364  (66.0%)
  Long   (500+ chars):    99,234  (29.3%)
  Average length:         597 chars

Citation readiness:
  With clickable URL:     338,433  (100.0%)
```

---

## 8. File Structure

```
data/
├── openbeta/
│   ├── openbeta_routes.json          # 204,463 routes from GraphQL API
│   └── openbeta_routes.csv
├── kaggle_8a/
│   ├── raw/                          # Original downloaded files
│   │   ├── mp_routes.csv             # 116,700 MP routes
│   │   ├── discussion_forum.csv      # 177,172 forum posts (raw)
│   │   ├── labeled_forums.csv        # 177,172 labeled posts
│   │   ├── review_forum.csv          # 12,309 gear review posts
│   │   ├── trailspace.csv            # 1,099 Trailspace reviews
│   │   └── aac/
│   │       └── articles.csv          # 27,831 AAC articles
│   ├── kaggle_routes.json            # Processed routes (116,700)
│   ├── kaggle_forums.json            # Processed forums (141,837)
│   ├── kaggle_reviews.json           # Processed reviews (10,774)
│   └── kaggle_aac_articles.json      # Processed AAC (27,828)
├── reddit/
│   └── reddit_climbing.json          # 2,519 Reddit posts
└── unified/
    └── cruxbot_unified.json          # 338,433 unified entries (FINAL)
```
