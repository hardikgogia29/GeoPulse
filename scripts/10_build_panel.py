
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.panel import (  # noqa: E402
    build_panel,
    scaled_thresholds,
    create_trip_regions_view,
    dev_grid_bounds,
    grid_bounds,
    region_metadata,
    select_active_regions,
)
from src.spatial.h3_indexer import load_h3_extension, make_indexer  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--spatial", default="h3", help="config overlay: h3 or s2")
    parser.add_argument("--resolution", type=int, default=None,
                        help="override spatial.resolution (Phase 5 sweep)")
    parser.add_argument("--warmup-days", type=int, default=8,
                        help="dev-sample only: extra history so lag features have inputs")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.spatial)
    log = get_logger("panel", cfg)

    if args.resolution is not None:
        from src.utils.config import Config

        # S2 reads `level`, H3 reads `resolution` - set both so either indexer sees it
        cfg = Config({**cfg, "spatial": {**cfg["spatial"],
                                         "resolution": args.resolution,
                                         "level": args.resolution}})
    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"
    log.info("spatial system: %s", indexer)

    if args.dev_sample:
        trips_path = resolve_path(cfg, "paths.dev_sample") / "trips_clean.parquet"
        out_root = resolve_path(cfg, "paths.dev_sample", mkdir=True) / f"panel_{tag}.parquet"
        meta_root = resolve_path(cfg, "paths.dev_sample", mkdir=True)
        lo, hi = dev_grid_bounds(cfg, warmup_days=args.warmup_days)
        partition = False
    else:
        trips_path = resolve_path(cfg, "paths.interim") / "trips_clean.parquet"
        out_root = resolve_path(cfg, "paths.processed", mkdir=True) / f"panel_{tag}"
        meta_root = resolve_path(cfg, "paths.spatial", mkdir=True)
        lo, hi = grid_bounds(cfg)
        partition = True
    if not trips_path.exists():
        log.error("missing %s - run scripts/02_clean.py first", trips_path)
        return 1

    interval = cfg.dotted("time.interval_minutes")
    n_bins = int((hi - lo).total_seconds() // (interval * 60))
    log.info("grid: %s .. %s UTC = %s bins of %d min", lo, hi, f"{n_bins:,}", interval)

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(args.temp_dir).as_posix()}'")
    load_h3_extension(con)
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet('{trips_path.as_posix()}')")

    with timed(log, "assign regions"):
        n_coords = create_trip_regions_view(con, indexer, "trips")
    log.info("distinct coordinates indexed: %s (joined onto trips, not evaluated per row)",
             f"{n_coords:,}")

    min_trips, min_days, scale = scaled_thresholds(cfg, lo, hi)
    if scale < 1.0:
        log.warning("window is %.1f%% of the project range - active-region thresholds "
                    "scaled to >=%s trips / >=%s active days (full-run values are "
                    "%s / %s)", 100 * scale, min_trips, min_days,
                    cfg.dotted("active_region.min_trips_total"),
                    cfg.dotted("active_region.min_active_days"))
    with timed(log, "select active regions"):
        activity = select_active_regions(con, cfg, lo, hi, min_trips, min_days)
    n_active = con.execute("SELECT count(*) FROM active_regions").fetchone()[0]
    if n_active == 0:
        log.error("no active regions selected - thresholds too strict for this window")
        return 3
    covered = con.execute(
        "SELECT sum(total_trips) FROM active_regions"
    ).fetchone()[0] or 0
    total_endpoints = int(activity["total_trips"].sum())
    log.info("regions touched: %s | active (>=%s trips, >=%s days): %s | "
             "endpoint coverage %.2f%%",
             f"{activity.height:,}", min_trips, min_days, f"{n_active:,}",
             100 * covered / max(total_endpoints, 1))

    with timed(log, "region metadata"):
        meta = region_metadata(con, indexer)
    meta_path = meta_root / f"regions_{tag}.parquet"
    meta.write_parquet(meta_path, compression="zstd")
    log.info("region metadata -> %s", meta_path)
    log.info("area km2: median=%.4f | trips/region: median=%s max=%s",
             float(meta["area_km2"].median()),
             f"{int(meta['total_trips'].median()):,}",
             f"{int(meta['total_trips'].max()):,}")

    expected_rows = n_active * n_bins
    log.info("expected panel rows: %s x %s = %s",
             f"{n_active:,}", f"{n_bins:,}", f"{expected_rows:,}")

    with timed(log, f"build panel -> {out_root.name}"):
        build_panel(con, cfg, lo, hi, out_root, partition_by_month=partition)

    scan = pl.scan_parquet(
        str(out_root / "**" / "*.parquet") if partition else str(out_root)
    )
    summary = scan.select(
        [
            pl.len().alias("rows"),
            pl.col("region_id").n_unique().alias("regions"),
            pl.col("ts").n_unique().alias("bins"),
            pl.col("pickups").sum().alias("total_pickups"),
            pl.col("dropoffs").sum().alias("total_dropoffs"),
            (pl.col("pickups") == 0).mean().alias("zero_pickup_share"),
            (pl.col("dropoffs") == 0).mean().alias("zero_dropoff_share"),
            pl.col("dst_unreliable").sum().alias("dst_unreliable_bins"),
            pl.col("pickup_h1").null_count().alias("null_pickup_h1"),
        ]
    ).collect(engine="streaming").to_dicts()[0]

    log.info("panel rows: %s (expected %s) %s",
             f"{summary['rows']:,}", f"{expected_rows:,}",
             "OK" if summary["rows"] == expected_rows else "MISMATCH")
    log.info("regions=%s bins=%s", f"{summary['regions']:,}", f"{summary['bins']:,}")
    log.info("pickups=%s dropoffs=%s", f"{summary['total_pickups']:,}",
             f"{summary['total_dropoffs']:,}")
    log.info("zero-demand share: pickups %.2f%% dropoffs %.2f%%",
             100 * summary["zero_pickup_share"], 100 * summary["zero_dropoff_share"])
    log.info("dst_unreliable bins flagged: %s", f"{summary['dst_unreliable_bins']:,}")
    log.info("rows with null pickup_h1 (past the horizon at the end): %s",
             f"{summary['null_pickup_h1']:,}")

    stats = {
        "spatial_system": indexer.name, "resolution": indexer.resolution,
        "grid_start_utc": str(lo), "grid_end_utc": str(hi), "bins": n_bins,
        "regions_touched": activity.height, "regions_active": n_active,
        "min_trips_total": min_trips, "min_active_days": min_days,
        "threshold_scale": round(scale, 6), "distinct_coords": n_coords,
        "endpoint_coverage_pct": round(100 * covered / max(total_endpoints, 1), 4),
        **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in summary.items()},
    }
    report = resolve_path(cfg, "paths.reports", mkdir=True) / (
        f"panel_{tag}{'_dev' if args.dev_sample else ''}.json"
    )
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("panel stats -> %s", report)

    if summary["rows"] != expected_rows:
        log.error("panel is not dense - expected %s rows", f"{expected_rows:,}")
        return 2
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
