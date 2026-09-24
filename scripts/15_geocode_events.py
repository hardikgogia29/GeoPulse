
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

GEOSEARCH = "https://geosearch.planninglabs.nyc/v2/search"

# "Astoria Park: Soccer-01" / "Pier 42: Nike Field-Soccer-02" -> drop the facility tag
FACILITY = re.compile(r":\s*[^:]*?-\d+\s*$")
BETWEEN = re.compile(r"^(?P<street>.+?)\s+between\s+(?P<cross>.+?)(?:\s+and\s+.+)?$", re.I)
DEAD_END = re.compile(r"\s+(dead\s*end|dead-end)\s*$", re.I)


def normalise(location: str) -> str:
    """Turn a permit location string into something a geocoder can resolve."""
    text = (location or "").strip()
    text = FACILITY.sub("", text)
    text = DEAD_END.sub("", text)
    match = BETWEEN.match(text)
    if match:
        # an intersection is far more resolvable than a segment description
        text = f"{match.group('street').strip()} & {match.group('cross').strip()}"
    return re.sub(r"\s+", " ", text).strip(" ,-")


def geocode(session: requests.Session, query: str, borough: str | None) -> dict:
    text = f"{query}, {borough}" if borough else query
    try:
        response = session.get(GEOSEARCH, params={"text": text, "size": 1}, timeout=25)
        if response.status_code != 200:
            return {"status": f"http_{response.status_code}"}
        features = response.json().get("features") or []
        if not features:
            return {"status": "no_match"}
        feature = features[0]
        props = feature["properties"]
        lng, lat = feature["geometry"]["coordinates"]
        got_borough = props.get("borough")
        # a confidently wrong borough is worse than nothing: it would attribute the
        # event to the wrong cell and quietly turn the ablation into noise
        if borough and got_borough and got_borough.strip().lower() != borough.strip().lower():
            return {"status": "borough_mismatch", "lat": lat, "lng": lng,
                    "matched_borough": got_borough, "label": props.get("label")}
        return {"status": "ok", "lat": lat, "lng": lng, "label": props.get("label"),
                "confidence": props.get("confidence"), "match_type": props.get("match_type"),
                "layer": props.get("layer"), "matched_borough": got_borough}
    except Exception as exc:  # noqa: BLE001 - a failed lookup is just unresolved
        return {"status": f"error_{type(exc).__name__}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("geocode", cfg)
    external = resolve_path(cfg, "paths.external", mkdir=True)
    source = external / "event_locations.parquet"
    out_path = external / "event_geocodes.parquet"
    if not source.exists():
        log.error("missing %s - run scripts/04_prepare_events.py", source)
        return 1

    locations = pl.read_parquet(source).sort("n_event_rows", descending=True)
    done: pl.DataFrame | None = None
    if out_path.exists() and not args.force:
        done = pl.read_parquet(out_path)
        seen = set(zip(done["event_location"].to_list(), done["event_borough"].to_list()))
        locations = locations.filter(
            ~pl.struct(["event_location", "event_borough"]).map_elements(
                lambda s: (s["event_location"], s["event_borough"]) in seen,
                return_dtype=pl.Boolean,
            )
        )
        log.info("resuming: %s already geocoded, %s remaining",
                 f"{done.height:,}", f"{locations.height:,}")
    if args.limit:
        locations = locations.head(args.limit)
    if locations.is_empty():
        log.info("nothing to do")
        return 0

    rows = locations.to_dicts()
    log.info("geocoding %s strings with %d workers", f"{len(rows):,}", args.workers)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers,
                                            pool_maxsize=args.workers * 2)
    session.mount("https://", adapter)

    started = time.perf_counter()
    results: list[dict] = []

    def work(row: dict) -> dict:
        query = normalise(row["event_location"])
        outcome = geocode(session, query, row["event_borough"]) if query else {"status": "empty"}
        return {"event_location": row["event_location"],
                "event_borough": row["event_borough"],
                "n_event_rows": row["n_event_rows"],
                "query": query, **outcome}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(work, rows), 1):
            results.append(result)
            if index % 2000 == 0:
                elapsed = time.perf_counter() - started
                log.info("  %s/%s (%.0f%%) %.1f min elapsed, eta %.1f min",
                         f"{index:,}", f"{len(rows):,}", 100 * index / len(rows),
                         elapsed / 60, elapsed / index * (len(rows) - index) / 60)

    frame = pl.DataFrame(results)
    if done is not None:
        frame = pl.concat([done, frame], how="diagonal_relaxed")
    frame.write_parquet(out_path, compression="zstd")

    total_rows = int(frame["n_event_rows"].sum())
    by_status = frame.group_by("status").agg(
        [pl.len().alias("strings"), pl.col("n_event_rows").sum().alias("event_rows")]
    ).sort("event_rows", descending=True)
    log.info("geocoded %s strings in %.1f min -> %s",
             f"{frame.height:,}", (time.perf_counter() - started) / 60, out_path)
    for row in by_status.to_dicts():
        log.info("  %-18s %6s strings  %9s event rows (%.1f%%)",
                 row["status"], f"{row['strings']:,}", f"{row['event_rows']:,}",
                 100 * row["event_rows"] / total_rows)
    resolved = frame.filter(pl.col("status") == "ok")
    log.info("RESOLVED: %.1f%% of event rows have usable coordinates",
             100 * int(resolved["n_event_rows"].sum()) / total_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
