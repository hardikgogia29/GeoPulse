
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.basic import (  # noqa: E402
    all_lag_columns,
    build_features_sql,
    create_holiday_table,
    feature_columns,
    lag_columns,
    max_materialised_lag,
    target_columns,
)
from src.models.splits import load_splits, validate_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--spatial", default="h3", help="config overlay: h3 or s2")
    parser.add_argument("--resolution", type=int, default=None,
                        help="override the spatial resolution / S2 level")
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--batch-rows", type=int, default=15_000_000,
                        help="approx panel rows per region batch")
    args = parser.parse_args()

    cfg = load_config(args.spatial)
    if args.resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: args.resolution,
                                          "resolution": args.resolution}})
    log = get_logger("features", cfg)
    from src.spatial.h3_indexer import make_indexer as _mk
    _idx = _mk(cfg)
    tag = f"{_idx.name}{_idx.resolution}"

    problems = validate_splits(cfg)
    if problems:
        for problem in problems:
            log.error("split problem: %s", problem)
        return 1
    splits = load_splits(cfg)
    log.info("splits: %s", " | ".join(
        f"{s.name} {s.start}..{s.end} ({s.days}d)" for s in splits.values()))

    if args.dev_sample:
        panel_path = resolve_path(cfg, "paths.dev_sample") / f"panel_{tag}.parquet"
        out_path = resolve_path(cfg, "paths.dev_sample", mkdir=True) / f"features_{tag}.parquet"
        partition = False
    else:
        panel_path = resolve_path(cfg, "paths.processed") / f"panel_{tag}"
        out_path = resolve_path(cfg, "paths.processed", mkdir=True) / f"features_{tag}"
        partition = True
    if not (panel_path.exists()):
        log.error("missing panel at %s - run scripts/10_build_panel.py", panel_path)
        return 1

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(args.temp_dir).as_posix()}'")

    source = (
        f"read_parquet('{(panel_path / '**' / '*.parquet').as_posix()}', hive_partitioning=true)"
        if partition
        else f"read_parquet('{panel_path.as_posix()}')"
    )
    con.execute(f"CREATE VIEW panel AS SELECT * EXCLUDE (month) FROM {source}"
                if partition else f"CREATE VIEW panel AS SELECT * FROM {source}")

    n_holidays = create_holiday_table(con, cfg)
    log.info("public holidays in window: %d", n_holidays)

    lags = all_lag_columns(cfg)
    model_lags = lag_columns(cfg)
    features = feature_columns(cfg)
    targets = target_columns(cfg)
    log.info("model features: %d (%d lags + %d calendar/cyclical + region_id)",
             len(features), len(model_lags), len(features) - len(model_lags) - 1)
    log.info("plus %d lag columns materialised for the Seasonal Naive baseline only: %s",
             len(lags) - len(model_lags),
             ", ".join(c for c in lags if c not in set(model_lags)))

    panel_rows = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    max_lag = max_materialised_lag(cfg)
    regions = con.execute("SELECT count(DISTINCT region_id) FROM panel").fetchone()[0]
    log.info("panel rows %s | dropping first %d steps per region as lag warm-up (%s rows)",
             f"{panel_rows:,}", max_lag, f"{regions * max_lag:,}")

    tz = cfg.dotted("time.timezone")
    month_col = f", strftime(ts AT TIME ZONE '{tz}', '%Y%m') AS month" if partition else ""
    copy_options = (
        "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (month), OVERWRITE_OR_IGNORE 1)"
        if partition
        else "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
    )
    select_list = f"""
              SELECT region_id, ts, pickups, dropoffs, net_flow, dst_unreliable,
                     {', '.join(targets)},
                     {', '.join(lags)},
                     hour, minute, day_of_week, day_of_month, week_of_year, month_of_year,
                     is_weekend, is_holiday, is_business_day, is_morning_rush, is_evening_rush,
                     tod_sin, tod_cos, dow_sin, dow_cos, doy_sin, doy_cos
                     {month_col}
              FROM feats
              ORDER BY region_id, ts
    """

    # Batch by region. Lags are PARTITION BY region_id, so a region batch is exact -
    # it only bounds what must be materialised at once. One pass over the whole panel
    # spills tens of GB and risks running the disk out; batches stay in memory.
    all_regions = [
        row[0] for row in con.execute(
            "SELECT DISTINCT region_id FROM panel ORDER BY region_id"
        ).fetchall()
    ]
    rows_per_region = panel_rows / max(len(all_regions), 1)
    if partition:
        batch_size = max(
            1, min(len(all_regions), int(args.batch_rows // max(rows_per_region, 1)))
        )
    else:
        # the dev output is a single file, so batching would make each batch overwrite
        # the last; it is small enough to do in one pass anyway
        batch_size = len(all_regions)
    batches = [all_regions[i:i + batch_size] for i in range(0, len(all_regions), batch_size)]
    log.info("region batches: %d x up to %d regions (~%s rows each)",
             len(batches), batch_size, f"{int(batch_size * rows_per_region):,}")

    if out_path.exists() and partition:
        import shutil

        shutil.rmtree(out_path)

    with timed(log, f"build features -> {out_path.name}"):
        for index, batch in enumerate(batches, 1):
            quoted = ", ".join(f"'{r}'" for r in batch)
            con.execute(
                f"CREATE OR REPLACE VIEW feats AS "
                f"{build_features_sql(cfg, 'panel', f'region_id IN ({quoted})')}"
            )
            if partition:
                options = (
                    "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (month), "
                    f"OVERWRITE_OR_IGNORE 1, FILENAME_PATTERN 'b{index:03d}_{{uuid}}')"
                )
                destination = out_path.as_posix()
            else:
                options = copy_options
                destination = out_path.as_posix()
            con.execute(f"COPY ({select_list}) TO '{destination}' {options}")
            if index % 5 == 0 or index == len(batches):
                log.info("  batch %d/%d done", index, len(batches))

    scan = pl.scan_parquet(
        str(out_path / "**" / "*.parquet") if partition else str(out_path)
    )
    summary = scan.select(
        [
            pl.len().alias("rows"),
            pl.col("region_id").n_unique().alias("regions"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("ts").max().alias("last_ts"),
            *[pl.col(c).null_count().alias(f"null_{c}") for c in lags[:1] + lags[-1:]],
            pl.col("pickup_h1").null_count().alias("null_pickup_h1"),
        ]
    ).collect(engine="streaming").to_dicts()[0]

    expected = panel_rows - regions * max_lag
    log.info("feature rows: %s (expected %s) %s", f"{summary['rows']:,}", f"{expected:,}",
             "OK" if summary["rows"] == expected else "MISMATCH")
    log.info("regions=%s | ts %s .. %s", f"{summary['regions']:,}",
             summary["first_ts"], summary["last_ts"])
    log.info("null lags after warm-up drop: %s=%s %s=%s",
             lags[0], summary[f"null_{lags[0]}"], lags[-1], summary[f"null_{lags[-1]}"])

    stats = {"tag": tag, "features": features, "targets": targets,
             "n_features": len(features), "panel_rows": panel_rows,
             "warmup_rows_dropped": regions * max_lag,
             **{k: (str(v) if hasattr(v, "isoformat") else v) for k, v in summary.items()}}
    report = resolve_path(cfg, "paths.reports", mkdir=True) / (
        f"features_{tag}{'_dev' if args.dev_sample else ''}.json")
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("feature stats -> %s", report)

    if summary["rows"] != expected:
        return 2
    if summary[f"null_{lags[-1]}"] != 0:
        log.error("longest lag still has nulls after the warm-up drop")
        return 3
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
