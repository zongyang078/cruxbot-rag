"""
OpenBeta GraphQL API - Batch Data Collection Script (v4)
=========================================================
Strategy: Use area(uuid) queries to traverse the area tree via BFS.
Each query fetches one area's children and their climbs.
If a child has routes but no direct climbs, queue it for deeper traversal.

Usage:
    python -u scripts/01_openbeta_collect.py

Endpoint: https://api.openbeta.io (no API key required)
License:  CC0 (public domain)
"""

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

# ============================================================
# Configuration
# ============================================================
ENDPOINT = "https://api.openbeta.io"
OUTPUT_DIR = "data/openbeta"
DELAY = 0.3  # Seconds between API requests per thread
MAX_WORKERS = 5  # Concurrent requests (conservative for free public API)
WAVE_PAUSE = 0.5  # Seconds to pause between BFS waves

os.makedirs(OUTPUT_DIR, exist_ok=True)


def query_api(query_string):
    """Send a GraphQL POST request with retry logic."""
    for attempt in range(3):
        try:
            resp = requests.post(
                ENDPOINT,
                json={"query": query_string},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if not resp.ok:
                raise Exception(f"{resp.status_code} {resp.reason}: {resp.text[:500]}")
            data = resp.json()
            if "errors" in data:
                print(f"\n  [warn] {str(data['errors'])[:100]}", end="", flush=True)
            return data
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                raise e


# ============================================================
# Step 1: Get all states and regions with UUIDs
# ============================================================
def get_state_regions():
    """Fetch USA > State > Region tree with UUIDs."""
    print("\n📍 Step 1: Fetching area tree (USA > States > Regions)...")

    query = """
    {
      areas(filter: {area_name: {match: "USA", exactMatch: true}}) {
        children {
          area_name
          uuid
          totalClimbs
          children {
            area_name
            uuid
            totalClimbs
          }
        }
      }
    }
    """
    result = query_api(query)
    usa_list = result.get("data", {}).get("areas", [])

    states = []
    for usa in usa_list:
        for state in usa.get("children", []):
            regions = []
            for region in state.get("children", []):
                if region.get("totalClimbs", 0) > 0:
                    regions.append(
                        {
                            "name": region["area_name"],
                            "uuid": region["uuid"],
                            "totalClimbs": region["totalClimbs"],
                        }
                    )
            if regions:
                states.append(
                    {
                        "name": state["area_name"],
                        "uuid": state["uuid"],
                        "totalClimbs": state.get("totalClimbs", 0),
                        "regions": regions,
                    }
                )

    states.sort(key=lambda x: x["totalClimbs"], reverse=True)
    total_regions = sum(len(s["regions"]) for s in states)
    total_routes = sum(s["totalClimbs"] for s in states)

    print(f"  Found {len(states)} states, {total_regions} regions, {total_routes:,} total routes")
    for s in states[:10]:
        print(f"    {s['name']:<25} {s['totalClimbs']:>6,} routes  ({len(s['regions'])} regions)")

    return states


# ============================================================
# Step 2: Query a single area by UUID, get children + climbs
# ============================================================
def query_area_by_uuid(uuid):
    """
    Query one area by UUID. Returns its children (with uuid, totalClimbs)
    and each child's direct climbs.
    """
    query = f"""
    {{
      area(uuid: "{uuid}") {{
        area_name
        uuid
        totalClimbs
        children {{
          area_name
          uuid
          totalClimbs
          climbs {{
            name
            uuid
            fa
            grades {{ yds vscale french }}
            type {{ sport trad bouldering tr alpine ice mixed aid }}
            content {{ description protection location }}
            metadata {{ lat lng mp_id }}
          }}
        }}
      }}
    }}
    """
    return query_api(query)


# ============================================================
# Step 3: BFS traversal to collect all climbs for a state
# ============================================================
def fetch_area(uuid, path, state_name):
    """Fetch one area and return (climbs, next_queue_items)."""
    time.sleep(DELAY)
    try:
        result = query_area_by_uuid(uuid)
    except Exception as e:
        print(f"x[{e}]", end="", flush=True)
        return [], []

    area = result.get("data", {}).get("area")
    if not area:
        return [], []

    climbs = []
    next_items = []

    for child in area.get("children", []):
        child_uuid = child.get("uuid", "")
        child_name = child.get("area_name", "")
        child_total = child.get("totalClimbs", 0)
        child_climbs = child.get("climbs", [])
        child_path = f"{path} > {child_name}"

        for climb in child_climbs:
            grades = climb.get("grades", {}) or {}
            ctype = climb.get("type", {}) or {}
            content = climb.get("content", {}) or {}
            meta = climb.get("metadata", {}) or {}

            type_list = [k for k, v in ctype.items() if v]
            grade = grades.get("yds") or grades.get("vscale") or grades.get("french") or ""
            mp_id = meta.get("mp_id", "")
            source_url = (
                f"https://www.mountainproject.com/route/{mp_id}" if mp_id else "https://openbeta.io"
            )

            climbs.append(
                {
                    "name": climb.get("name", ""),
                    "uuid": climb.get("uuid", ""),
                    "grade": grade,
                    "grade_yds": grades.get("yds", ""),
                    "grade_vscale": grades.get("vscale", ""),
                    "grade_french": grades.get("french", ""),
                    "type": ", ".join(type_list) if type_list else "unknown",
                    "description": content.get("description", "") or "",
                    "protection": content.get("protection", "") or "",
                    "location_info": content.get("location", "") or "",
                    "first_ascent": climb.get("fa", "") or "",
                    "lat": meta.get("lat"),
                    "lng": meta.get("lng"),
                    "state": state_name,
                    "area_path": child_path,
                    "source_url": source_url,
                    "source": "openbeta",
                }
            )

        if child_total > 0 and len(child_climbs) == 0 and child_uuid:
            next_items.append((child_uuid, child_path))

    return climbs, next_items


def collect_state(state):
    """
    Collect all climbs for a state using concurrent wave BFS.
    Each wave processes all queued areas in parallel (up to MAX_WORKERS),
    then pauses briefly before the next wave.
    """
    all_climbs = []
    state_name = state["name"]

    wave = [(region["uuid"], f"{state_name} > {region['name']}") for region in state["regions"]]
    visited = set()
    api_calls = 0

    while wave:
        # Deduplicate within wave
        wave = [(u, p) for u, p in wave if u not in visited]
        for u, _ in wave:
            visited.add(u)

        next_wave = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_area, u, p, state_name): (u, p) for u, p in wave}
            for future in as_completed(futures):
                climbs, next_items = future.result()
                all_climbs.extend(climbs)
                next_wave.extend(next_items)
                api_calls += 1
                if api_calls % 20 == 0:
                    print(f" [{api_calls} calls, {len(all_climbs)} climbs]", end="", flush=True)

        wave = next_wave
        if wave:
            time.sleep(WAVE_PAUSE)

    return all_climbs, api_calls


# ============================================================
# Main pipeline
# ============================================================
def collect_all(max_states=None):
    """
    Collect route data for US states.

    Args:
        max_states: Limit to top N states by route count (None = all).
    """
    print("=" * 60)
    print("🧗 OpenBeta Data Collection (v4 - UUID-based BFS)")
    print(f"   Endpoint: {ENDPOINT}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    states = get_state_regions()

    if max_states:
        states = states[:max_states]
        names = ", ".join(s["name"] for s in states)
        print(f"\n⚡ Test mode: top {max_states} state(s): {names}")

    # Collect state by state
    all_routes = []
    total_api_calls = 0

    print("\n📥 Step 2: Collecting routes via BFS traversal...")

    for i, state in enumerate(states):
        print(
            f"\n  [{i + 1}/{len(states)}] {state['name']} ({state['totalClimbs']:,} routes)...",
            end="",
            flush=True,
        )

        climbs, calls = collect_state(state)
        all_routes.extend(climbs)
        total_api_calls += calls
        print(f" ✅ {len(climbs):,} climbs ({calls} API calls)")

    # Deduplicate by UUID
    seen = set()
    unique = []
    for r in all_routes:
        uid = r.get("uuid", "")
        if uid and uid in seen:
            continue
        if uid:
            seen.add(uid)
        unique.append(r)

    print(f"\n  Dedup: {len(all_routes):,} -> {len(unique):,} unique")
    print(f"  Total API calls: {total_api_calls}")
    all_routes = unique

    # Save
    print("\n💾 Step 3: Saving data...")

    json_path = os.path.join(OUTPUT_DIR, "openbeta_routes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_routes, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(json_path) / 1024 / 1024
    print(f"  JSON: {json_path} ({len(all_routes):,} routes, {file_size:.1f} MB)")

    csv_path = os.path.join(OUTPUT_DIR, "openbeta_routes.csv")
    if all_routes:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_routes[0].keys())
            writer.writeheader()
            writer.writerows(all_routes)
        print(f"  CSV:  {csv_path}")

    # Summary
    print("\n📊 Summary:")
    print(f"  Total routes: {len(all_routes):,}")

    type_counts = {}
    for r in all_routes:
        for t in r["type"].split(", "):
            type_counts[t] = type_counts.get(t, 0) + 1
    print("\n  Type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<15} {c:>6,}")

    with_desc = sum(1 for r in all_routes if r.get("description", "").strip())
    print(f"\n  With description: {with_desc:,} ({with_desc / max(len(all_routes), 1) * 100:.1f}%)")

    state_counts = {}
    for r in all_routes:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
    print("\n  Per state:")
    for s, c in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<25} {c:>6,}")

    return all_routes


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    # Test with 1 state first. Change to None for full collection.
    routes = collect_all(max_states=None)

    print("\n" + "=" * 60)
    print("✅ Done!")
    print(f"Data: {OUTPUT_DIR}/")
    print("Next: set max_states=None for full US, then run 02_kaggle_8a_collect.py")
    print("=" * 60)
