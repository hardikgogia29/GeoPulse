
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.advanced import (  # noqa: E402
    EVENT_COLUMNS, STATION_COLUMNS, TRAFFIC_COLUMNS, WEATHER_BASE,
    demand_recent_exprs, family_columns, neighbor_columns, rolling_trend_exprs,
    seasonal_exprs, target_calendar_exprs, weather_derived_exprs,
)
from src.features.basic import CALENDAR_COLUMNS, CYCLICAL_COLUMNS, calendar_sql, create_holiday_table, target_columns  # noqa: E402
from src.spatial.h3_indexer import load_h3_extension, make_indexer  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def batch_sql(cfg, indexer, regions_sql: str, neighbors_sql: str, interval: int) -> str:
    """Panel rows for one region batch, with neighbours and external context joined."""
    tz = cfg.dotted("time.timezone")
    stats = cfg.dotted("advanced_features.neighbor_stats")
    nb_aggs = ",\n                 ".join(
        f"{stat}(p2.{target})::FLOAT AS nb_{target}_{stat}"
        for target in ("pickups", "dropoffs") for stat in stats
    )
    event_cols = ", ".join(
        f"coalesce(e.{c}, 0) AS {c}" for c in EVENT_COLUMNS if c != "event_present"
    )
    traffic_cols = ", ".join(
        f"t.{c}" for c in TRAFFIC_COLUMNS if c not in ("traffic_missing",)
    )
    station_cols = ", ".join(f"s.{c}" for c in STATION_COLUMNS)
    weather_cols = ", ".join(f"w.{c}" for c in WEATHER_BASE)
    return f"""
    WITH batch AS (
      SELECT * FROM panel WHERE region_id IN ({regions_sql})
    ),
    edges AS (
      -- built in Python from indexer.neighbors() so H3 and S2 share one code path.
      -- At ring 1 this is exactly h3_grid_ring(r, 1): grid_disk(r, 1) minus self.
      SELECT region_id, neighbor_id FROM region_edges
      WHERE region_id IN ({regions_sql})
    ),
    neighbor_panel AS (
      -- restrict the scan to the batch's neighbours; without this every batch
      -- re-reads all 104M panel rows
      SELECT region_id, ts, pickups, dropoffs FROM panel
      WHERE region_id IN ({neighbors_sql})
    ),
    neighbor_agg AS (
      SELECT e.region_id, p2.ts,
             {nb_aggs},
             count(*)::INTEGER AS neighbor_count
      FROM edges e
      JOIN neighbor_panel p2 ON p2.region_id = e.neighbor_id
      GROUP BY e.region_id, p2.ts
    )
    SELECT b.region_id, b.ts, b.pickups, b.dropoffs, b.net_flow, b.dst_unreliable,
           {', '.join('b.' + c for c in target_columns(cfg))},
           {calendar_sql(cfg, 'b.ts')},
           (extract('hour' FROM (b.ts AT TIME ZONE '{tz}')) * 4
            + extract('minute' FROM (b.ts AT TIME ZONE '{tz}')) / {interval})::SMALLINT
             AS slot_of_day,
           {weather_cols},
           (w.weather_timestamp IS NULL) AS weather_missing,
           coalesce(e.event_count, 0) > 0 AS event_present,
           {event_cols},
           {traffic_cols},
           (t.region_id IS NULL) AS traffic_missing,
           {station_cols},
           {', '.join('n.' + c for c in neighbor_columns(cfg) if c.startswith('nb_') or c == 'neighbor_count')}
    FROM batch b
    LEFT JOIN holidays_tbl h
      ON h.holiday_date = (b.ts AT TIME ZONE '{tz}')::DATE
    LEFT JOIN weather w
      ON w.weather_timestamp = date_trunc('hour', b.ts + INTERVAL {interval} MINUTE)
    LEFT JOIN events e   ON e.region_id = b.region_id AND e.ts = b.ts
    LEFT JOIN traffic t  ON t.region_id = b.region_id AND t.ts = b.ts
    LEFT JOIN station s
      ON s.region_id = b.region_id
     AND s.month = strftime(b.ts AT TIME ZONE '{tz}', '%Y%m')
    LEFT JOIN neighbor_agg n ON n.region_id = b.region_id AND n.ts = b.ts
    ORDER BY b.region_id, b.ts
    """


def add_series_features(frame: pl.DataFrame, cfg) -> pl.DataFrame:
    """Everything that is a function of the region's own history."""
    frame = frame.with_columns(demand_recent_exprs(cfg))
    frame = frame.with_columns(seasonal_exprs(cfg))
    frame = frame.with_columns(target_calendar_exprs(cfg))
    frame = frame.with_columns(rolling_trend_exprs(cfg))
    frame = frame.with_columns(weather_derived_exprs(cfg))
    # spatial gradient: own demand relative to the neighbourhood
    frame = frame.with_columns(
        [
            (pl.col(target) - pl.col(f"nb_{target}_mean")).cast(pl.Float32)
            .alias(f"{target}_vs_neighbor_mean")
            for target in ("pickups", "dropoffs")
        ]
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--spatial", default="h3")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=3_000_000)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.spatial)
    if args.resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: args.resolution,
                                         "resolution": args.resolution}})
    log = get_logger("features4", cfg)
    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"
    interval = cfg.dotted("time.interval_minutes")

    processed = resolve_path(cfg, "paths.processed")
    external = resolve_path(cfg, "paths.external")
    if args.dev_sample:
        panel_path = resolve_path(cfg, "paths.dev_sample") / f"panel_{tag}.parquet"
        out_path = resolve_path(cfg, "paths.dev_sample", mkdir=True) / f"features4_{tag}.parquet"
        partition = False
    else:
        panel_path = processed / f"panel_{tag}"
        out_path = processed / f"features4_{tag}"
        partition = True
    if not panel_path.exists():
        log.error("missing panel at %s", panel_path)
        return 1

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(args.temp_dir).as_posix()}'")
    if indexer.name == "h3":
        load_h3_extension(con)

    source = (f"read_parquet('{(panel_path / '**' / '*.parquet').as_posix()}', hive_partitioning=true)"
              if partition else f"read_parquet('{panel_path.as_posix()}')")
    con.execute(f"CREATE VIEW panel AS SELECT * {'EXCLUDE (month)' if partition else ''} FROM {source}")
    con.execute(f"CREATE VIEW weather AS SELECT * FROM read_parquet('{(external / 'weather_hourly.parquet').as_posix()}')")
    missing_external = []
    for name, stem in (("events", "event_bins"), ("traffic", "traffic_bins"),
                       ("station", "station_network")):
        path = processed / f"{stem}_{tag}.parquet"
        if path.exists():
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
            continue
        # scripts/16_map_external.py is H3-only, so an S2 run has no external mapping.
        # Borrow a sibling tag's SCHEMA with zero rows: the LEFT JOINs then produce the
        # same NULL/0 defaults an empty mapping would, families E/F/H are honestly
        # unusable for this tag, and A-D+G are still computed identically to H3.
        siblings = sorted(processed.glob(f"{stem}_*.parquet"))
        if not siblings:
            log.error("missing %s with no sibling to take a schema from - run "
                      "scripts/16_map_external.py", path)
            return 1
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM "
                    f"read_parquet('{siblings[0].as_posix()}') WHERE false")
        missing_external.append(name)
    if missing_external:
        log.warning("no %s mapping for %s - families E/F/H will be empty. The final "
                    "feature set is %s, which does not use them.",
                    "/".join(missing_external), tag,
                    "+".join(cfg.dotted("advanced_features.final_families")))
    create_holiday_table(con, cfg)

    all_regions = [r[0] for r in con.execute(
        "SELECT DISTINCT region_id FROM panel ORDER BY region_id").fetchall()]
    panel_rows = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    rows_per_region = panel_rows / max(len(all_regions), 1)
    batch_size = max(1, min(len(all_regions), int(args.batch_rows // max(rows_per_region, 1))))
    batches = [all_regions[i:i + batch_size] for i in range(0, len(all_regions), batch_size)]
    region_set = set(all_regions)
    ring = cfg.dotted("advanced_features.neighbor_ring")
    edge_rows = [(r, n) for r in all_regions
                 for n in indexer.neighbors(r, ring) if n in region_set]
    con.register("region_edges", pl.DataFrame(
        edge_rows, schema={"region_id": pl.Utf8, "neighbor_id": pl.Utf8}, orient="row"))
    log.info("panel %s rows, %s regions -> %d batches of up to %d regions",
             f"{panel_rows:,}", f"{len(all_regions):,}", len(batches), batch_size)

    families = family_columns(cfg)
    n_features = sum(len(v) for v in families.values()) + 1
    log.info("target feature count: %d across families %s",
             n_features, {k: len(v) for k, v in families.items()})

    if out_path.exists():
        shutil.rmtree(out_path) if partition else out_path.unlink()

    tz = cfg.dotted("time.timezone")
    written = 0
    started = time.perf_counter()
    with timed(log, f"build features4 -> {out_path.name}"):
        for index, batch in enumerate(batches, 1):
            quoted = ", ".join(f"'{r}'" for r in batch)
            neighbours = {n for r in batch for n in indexer.neighbors(r)} & region_set
            neighbours_sql = ", ".join(f"'{r}'" for r in sorted(neighbours)) or "''"
            frame = pl.from_arrow(
                con.execute(batch_sql(cfg, indexer, quoted, neighbours_sql, interval)).arrow()
            )
            frame = add_series_features(frame, cfg)
            frame = frame.with_columns(
                pl.col("ts").dt.convert_time_zone(tz).dt.strftime("%Y%m").alias("month")
            )
            if partition:
                # NOT polars' partitioned write: it names every file 00000000.parquet
                # and rewrites the directory, so each batch silently replaced the last.
                # DuckDB's FILENAME_PATTERN gives each batch its own files.
                con.register("batch_out", frame)
                con.execute(
                    f"""
                    COPY (SELECT * FROM batch_out) TO '{out_path.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (month),
                     OVERWRITE_OR_IGNORE 1, FILENAME_PATTERN 'b{index:03d}_{{uuid}}')
                    """
                )
                con.unregister("batch_out")
            else:
                frame.drop("month").write_parquet(out_path, compression="zstd")
            written += frame.height
            if index % 5 == 0 or index == len(batches):
                elapsed = time.perf_counter() - started
                log.info("  batch %d/%d  %s rows  %.1f min elapsed, eta %.1f min",
                         index, len(batches), f"{written:,}", elapsed / 60,
                         elapsed / index * (len(batches) - index) / 60)

    scan = pl.scan_parquet(str(out_path / "**" / "*.parquet") if partition else str(out_path))
    schema = scan.collect_schema()
    on_disk = scan.select(
        [pl.len().alias("rows"), pl.col("region_id").n_unique().alias("regions")]
    ).collect(engine="streaming").to_dicts()[0]
    log.info("wrote %s rows, %d columns -> %s", f"{written:,}", len(schema.names()), out_path)
    log.info("on disk: %s rows across %s regions", f"{on_disk['rows']:,}",
             f"{on_disk['regions']:,}")
    # a partitioned writer that reuses filenames silently drops earlier batches, so
    # verify what landed rather than trusting what was handed to the writer
    if on_disk["rows"] != written or on_disk["regions"] != len(all_regions):
        log.error("MISMATCH: expected %s rows across %s regions - batches were lost",
                  f"{written:,}", f"{len(all_regions):,}")
        return 3

    missing = [c for fam in families.values() for c in fam if c not in schema.names()]
    if missing:
        log.error("missing feature columns: %s", missing[:20])
        return 2
    log.info("all %d family columns present", n_features)

    report = resolve_path(cfg, "paths.reports", mkdir=True) / (
        f"features4_{tag}{'_dev' if args.dev_sample else ''}.json")
    report.write_text(json.dumps(
        {"tag": tag, "rows": written, "columns": len(schema.names()),
         "families": {k: len(v) for k, v in families.items()},
         "family_columns": families}, indent=2), encoding="utf-8")
    log.info("stats -> %s", report)
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
