
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path, source_path  # noqa: E402
from src.utils.geo import haversine_km  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402

AGG_SQL = """
WITH appearances AS (
  SELECT start_station_id AS station_id, start_station_name AS station_name,
         start_lat AS lat, start_lng AS lng, started_at AS ts, 1 AS is_pickup
  FROM trips WHERE start_station_id IS NOT NULL
  UNION ALL
  SELECT end_station_id, end_station_name, end_lat, end_lng, ended_at, 0
  FROM trips WHERE end_station_id IS NOT NULL
)
SELECT
  station_id,
  mode(station_name)                 AS station_name,
  count(DISTINCT station_name)       AS n_distinct_names,
  median(lat)                        AS lat,
  median(lng)                        AS lng,
  min(ts)                            AS first_seen,
  max(ts)                            AS last_seen,
  count(*)                           AS appearances,
  sum(is_pickup)                     AS pickups,
  count(*) - sum(is_pickup)          AS dropoffs,
  count(DISTINCT ts::DATE)           AS active_days,
  quantile_cont(lat, 0.01)           AS lat_p01,
  quantile_cont(lat, 0.99)           AS lat_p99,
  quantile_cont(lng, 0.01)           AS lng_p01,
  quantile_cont(lng, 0.99)           AS lng_p99,
  min(lat) AS lat_min, max(lat) AS lat_max,
  min(lng) AS lng_min, max(lng) AS lng_max
FROM appearances
GROUP BY station_id
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("stations", cfg)

    if args.dev_sample:
        trips_path = resolve_path(cfg, "paths.dev_sample") / "trips_clean.parquet"
        out_path = resolve_path(cfg, "paths.dev_sample", mkdir=True) / "station_registry.parquet"
    else:
        trips_path = resolve_path(cfg, "paths.interim") / "trips_clean.parquet"
        out_path = resolve_path(cfg, "paths.spatial", mkdir=True) / "station_registry.parquet"
    if not trips_path.exists():
        log.error("missing %s - run scripts/02_clean.py first", trips_path)
        return 1

    con = duckdb.connect()
    con.execute(f"SET TimeZone='{cfg.dotted('time.timezone')}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(args.temp_dir).as_posix()}'")
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet('{trips_path.as_posix()}')")

    with timed(log, "aggregate station appearances"):
        registry = pl.from_arrow(con.execute(AGG_SQL).arrow())

    threshold = cfg.dotted("station_registry.coord_jump_km_threshold")
    min_rides = cfg.dotted("station_registry.min_rides_for_registry")

    robust_km = haversine_km(
        registry["lat_p01"], registry["lng_p01"], registry["lat_p99"], registry["lng_p99"]
    )
    max_km = haversine_km(
        registry["lat_min"], registry["lng_min"], registry["lat_max"], registry["lng_max"]
    )
    registry = registry.with_columns(
        [
            pl.Series("coord_spread_km_robust", robust_km).round(4),
            pl.Series("coord_spread_km_max", max_km).round(4),
        ]
    ).with_columns(
        (pl.col("coord_spread_km_robust") > threshold).alias("coord_anomaly")
    )

    before = registry.height
    registry = registry.filter(pl.col("appearances") >= min_rides).sort(
        "appearances", descending=True
    )
    log.info("stations: %s (dropped %s below min_rides_for_registry=%s)",
             f"{registry.height:,}", f"{before - registry.height:,}", min_rides)

    # Cross-check against the published roster snapshot - reported, never substituted.
    roster_path = source_path(cfg, "station_roster_csv")
    if roster_path.exists():
        roster = pl.read_csv(roster_path, schema_overrides={"id": pl.String}).rename(
            {"id": "station_id", "name": "roster_name",
             "latitude": "roster_lat", "longitude": "roster_lng"}
        )
        joined = registry.join(roster, on="station_id", how="left")
        matched = joined.filter(pl.col("roster_lat").is_not_null())
        log.info("roster cross-check: %s of %s registry stations present in the roster snapshot",
                 f"{matched.height:,}", f"{registry.height:,}")
        if matched.height:
            delta = haversine_km(
                matched["lat"], matched["lng"], matched["roster_lat"], matched["roster_lng"]
            )
            log.info("  roster-vs-observed coord delta km: median=%.4f p95=%.4f max=%.4f",
                     float(pl.Series(delta).median()),
                     float(pl.Series(delta).quantile(0.95)),
                     float(pl.Series(delta).max()))
        registry = joined.with_columns(
            pl.col("roster_lat").is_not_null().alias("in_roster_snapshot")
        )
        log.info("roster stations absent from 2023-2024 trip activity: %s",
                 f"{roster.height - matched.height:,}")

    registry.write_parquet(out_path, compression="zstd")

    anomalies = registry.filter(pl.col("coord_anomaly"))
    log.info("coordinate anomalies (> %.2f km robust spread): %s stations",
             threshold, f"{anomalies.height:,}")
    if anomalies.height:
        for row in anomalies.sort("coord_spread_km_robust", descending=True).head(10).iter_rows(named=True):
            log.info("  %-10s %-38s spread_robust=%.2fkm spread_max=%.2fkm rides=%s",
                     row["station_id"], (row["station_name"] or "")[:38],
                     row["coord_spread_km_robust"], row["coord_spread_km_max"],
                     f"{row['appearances']:,}")
    log.info("multi-name stations (renamed over time): %s",
             f"{registry.filter(pl.col('n_distinct_names') > 1).height:,}")
    log.info("wrote %s stations -> %s", f"{registry.height:,}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
