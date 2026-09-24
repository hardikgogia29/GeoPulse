
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.timezone import localize_to_utc  # noqa: E402
from src.data.traffic import (  # noqa: E402
    SocrataClient,
    add_link_geometry_summary,
    day_chunks,
    fetch_link_registry,
    fetch_observations,
    parse_timestamp,
)
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402


def sample_days(start: date, end: date, n: int = 8) -> list[date]:
    span = (end - start).days
    return sorted({start + timedelta(days=round(i * span / (n - 1))) for i in range(n)})


def build_registry(client, cfg, log, start: date, end: date) -> int:
    out_dir = resolve_path(cfg, "paths.spatial", mkdir=True)
    log.info("building traffic link registry from sampled windows")
    registry = fetch_link_registry(client, cfg, sample_days(start, end), log)
    if registry.is_empty():
        log.error("link registry came back empty")
        return 1
    registry = add_link_geometry_summary(registry)
    out_path = out_dir / "traffic_links.parquet"
    registry.write_parquet(out_path, compression="zstd")
    log.info("link registry: %d (link, geometry) rows covering %d distinct links -> %s",
             registry.height, registry["link_id"].n_unique(), out_path)
    changed = registry.group_by("link_id").len().filter(pl.col("len") > 1)
    if changed.height:
        log.info("links whose geometry changed during the window: %d", changed.height)
    missing = registry.filter(pl.col("mid_lat").is_null())
    if missing.height:
        log.warning("links with unparseable link_points geometry: %d", missing.height)
    log.info("borough mix: %s", registry["borough"].value_counts(sort=True).to_dicts())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true",
                        help="only fetch the configured 7-day dev window")
    parser.add_argument("--registry-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="refetch days already written")
    parser.add_argument("--start", default=None, help="override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="override end date (YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("traffic", cfg)
    tz = cfg.dotted("time.timezone")

    if args.dev_sample:
        start = date.fromisoformat(cfg.dotted("dev_sample.start"))
        end = date.fromisoformat(cfg.dotted("dev_sample.end"))
    else:
        start = date.fromisoformat(args.start or cfg.dotted("time.start_date"))
        end = date.fromisoformat(args.end or cfg.dotted("time.end_date"))

    client = SocrataClient(cfg, log)

    if args.registry_only:
        return build_registry(client, cfg, log,
                              date.fromisoformat(cfg.dotted("time.start_date")),
                              date.fromisoformat(cfg.dotted("time.end_date")))

    out_root = resolve_path(cfg, "paths.raw", "traffic", mkdir=True)
    report_dir = resolve_path(cfg, "paths.reports", mkdir=True)
    manifest_path = report_dir / "traffic_manifest.json"
    manifest: dict = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if not (resolve_path(cfg, "paths.spatial") / "traffic_links.parquet").exists():
        rc = build_registry(client, cfg, log,
                            date.fromisoformat(cfg.dotted("time.start_date")),
                            date.fromisoformat(cfg.dotted("time.end_date")))
        if rc:
            return rc

    chunks = day_chunks(start, end, cfg.dotted("traffic.chunk_days"))
    page_limit = cfg.dotted("traffic.page_limit")
    log.info("fetching %d day-chunks: %s .. %s", len(chunks), start, end)

    started = time.perf_counter()
    total_rows = 0
    for i, (lo, hi) in enumerate(chunks, 1):
        key = lo.isoformat()
        if not args.force and key in manifest:
            total_rows += manifest[key]["rows"]
            continue

        t0 = time.perf_counter()
        raw = fetch_observations(client, cfg, lo, hi)
        if raw.height >= page_limit:
            # a truncated page would silently lose readings - fail loudly instead
            log.error("day %s returned %d rows == page_limit; raise traffic.page_limit "
                      "or lower traffic.chunk_days", key, raw.height)
            return 2

        if raw.height:
            frame = raw.with_columns(parse_timestamp("data_as_of"))
            unparsed = int(frame["data_as_of"].null_count())
            frame = frame.drop_nulls(subset=["data_as_of"]).with_columns(
                localize_to_utc(pl.col("data_as_of"), tz).alias("data_as_of")
            ).with_columns(
                pl.col("data_as_of").dt.convert_time_zone(tz).dt.strftime("%Y%m").alias("month")
            )
            for (month,), part in frame.partition_by("month", as_dict=True).items():
                target_dir = out_root / f"month={month}"
                target_dir.mkdir(parents=True, exist_ok=True)
                part.drop("month").write_parquet(
                    target_dir / f"part-{key}.parquet", compression="zstd"
                )
            rows = frame.height
        else:
            rows, unparsed = 0, 0
            log.warning("day %s returned no rows", key)

        manifest[key] = {"rows": rows, "unparsed": unparsed,
                         "seconds": round(time.perf_counter() - t0, 1)}
        total_rows += rows
        if i % 10 == 0 or i == len(chunks) or rows == 0:
            elapsed = time.perf_counter() - started
            rate = elapsed / i
            log.info("[%d/%d] %s rows=%s total=%s elapsed=%.1fmin eta=%.1fmin",
                     i, len(chunks), key, f"{rows:,}", f"{total_rows:,}",
                     elapsed / 60, rate * (len(chunks) - i) / 60)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("traffic fetch complete: %s observations across %d days in %.1f min -> %s",
             f"{total_rows:,}", len(chunks), (time.perf_counter() - started) / 60, out_root)
    log.info("manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
