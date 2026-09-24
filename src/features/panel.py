
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import polars as pl

from src.data.timezone import dst_unreliable_utc_hours, find_dst_transitions


def grid_bounds(cfg) -> tuple[datetime, datetime]:
    """UTC [start, end) instants covering the configured local date range."""
    tz = cfg.dotted("time.timezone")
    start_local = f"{cfg.dotted('time.start_date')}T00:00:00"
    end_local = (
        datetime.fromisoformat(cfg.dotted("time.end_date")) + timedelta(days=1)
    ).strftime("%Y-%m-%dT%H:%M:%S")
    frame = pl.select(
        [
            pl.lit(start_local).str.to_datetime().dt.replace_time_zone(tz)
            .dt.convert_time_zone("UTC").alias("lo"),
            pl.lit(end_local).str.to_datetime().dt.replace_time_zone(tz)
            .dt.convert_time_zone("UTC").alias("hi"),
        ]
    )
    return frame["lo"].item(), frame["hi"].item()


def dev_grid_bounds(cfg, warmup_days: int = 0) -> tuple[datetime, datetime]:
    tz = cfg.dotted("time.timezone")
    start = datetime.fromisoformat(cfg.dotted("dev_sample.start")) - timedelta(days=warmup_days)
    end = datetime.fromisoformat(cfg.dotted("dev_sample.end")) + timedelta(days=1)
    frame = pl.select(
        [
            pl.lit(start.strftime("%Y-%m-%dT%H:%M:%S")).str.to_datetime()
            .dt.replace_time_zone(tz).dt.convert_time_zone("UTC").alias("lo"),
            pl.lit(end.strftime("%Y-%m-%dT%H:%M:%S")).str.to_datetime()
            .dt.replace_time_zone(tz).dt.convert_time_zone("UTC").alias("hi"),
        ]
    )
    return frame["lo"].item(), frame["hi"].item()


def create_trip_regions_view(con: duckdb.DuckDBPyConnection, indexer, trips_view: str) -> int:
    """Attach start/end region ids to every trip, in-database.

    Citi Bike snaps coordinates to station locations, so 79M trips contain only a few
    thousand distinct (lat, lng) pairs. Indexing those once into a lookup table and
    joining beats calling the H3 function 158M times by orders of magnitude - and it
    is exact, not an approximation. Returns the lookup size.
    """
    index_expr = indexer.sql_index_expr("lat", "lng")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE coord_pairs AS
        SELECT DISTINCT start_lat AS lat, start_lng AS lng FROM {trips_view}
        UNION
        SELECT DISTINCT end_lat, end_lng FROM {trips_view}
        """
    )
    if index_expr is not None:
        con.execute(
            f"CREATE OR REPLACE TABLE coord_regions AS "
            f"SELECT lat, lng, {index_expr} AS region_id FROM coord_pairs"
        )
    else:
        # no in-database indexer (S2): index the few thousand distinct coordinates in
        # Python and register the lookup. Same result, computed elsewhere.
        import polars as pl

        coords = pl.from_arrow(con.execute("SELECT lat, lng FROM coord_pairs").arrow())
        coords = coords.with_columns(
            pl.struct(["lat", "lng"])
            .map_elements(lambda s: indexer.index(s["lat"], s["lng"]), return_dtype=pl.String)
            .alias("region_id")
        )
        con.register("coord_regions_py", coords)
        con.execute("CREATE OR REPLACE TABLE coord_regions AS SELECT * FROM coord_regions_py")
        con.unregister("coord_regions_py")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS coord_regions_pk ON coord_regions(lat, lng)")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW trip_regions AS
        SELECT t.started_at, t.ended_at,
               s.region_id AS start_region,
               e.region_id AS end_region,
               t.start_station_id, t.end_station_id
        FROM {trips_view} t
        JOIN coord_regions s ON s.lat = t.start_lat AND s.lng = t.start_lng
        JOIN coord_regions e ON e.lat = t.end_lat   AND e.lng = t.end_lng
        """
    )
    return con.execute("SELECT count(*) FROM coord_regions").fetchone()[0]


def scaled_thresholds(cfg, lo: datetime, hi: datetime) -> tuple[int, int, float]:
    """Activity thresholds, rescaled when the window is shorter than the project.

    `min_trips_total` / `min_active_days` are calibrated for the full two years. On
    the 7-day dev sample no region could ever clear 180 active days, so the dev run
    would select nothing and prove nothing. Scaling by the window length keeps the
    dev sample a real smoke test of the same code path. Returns (trips, days, scale).
    """
    full_days = (
        datetime.fromisoformat(cfg.dotted("time.end_date"))
        - datetime.fromisoformat(cfg.dotted("time.start_date"))
    ).days + 1
    window_days = max((hi - lo).total_seconds() / 86400, 1)
    scale = min(window_days / full_days, 1.0)
    min_trips = max(1, round(cfg.dotted("active_region.min_trips_total") * scale))
    min_days = max(1, round(cfg.dotted("active_region.min_active_days") * scale))
    return min_trips, min_days, scale


def select_active_regions(
    con: duckdb.DuckDBPyConnection,
    cfg,
    lo: datetime,
    hi: datetime,
    min_trips: int | None = None,
    min_days: int | None = None,
) -> pl.DataFrame:
    """Regions defined by observed trip activity, never by a station roster.

    A region qualifies on total trips AND on the number of distinct days it saw
    activity, so a single busy week cannot promote a region that is otherwise dead.
    """
    if min_trips is None or min_days is None:
        min_trips, min_days, _ = scaled_thresholds(cfg, lo, hi)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE region_activity AS
        WITH endpoints AS (
          SELECT start_region AS region_id, started_at AS ts, 1 AS is_pickup,
                 start_station_id AS station_id
          FROM trip_regions
          WHERE started_at >= TIMESTAMPTZ '{lo.isoformat()}'
            AND started_at <  TIMESTAMPTZ '{hi.isoformat()}'
          UNION ALL
          SELECT end_region, ended_at, 0, end_station_id
          FROM trip_regions
          WHERE ended_at >= TIMESTAMPTZ '{lo.isoformat()}'
            AND ended_at <  TIMESTAMPTZ '{hi.isoformat()}'
        )
        SELECT region_id,
               count(*)                        AS total_trips,
               sum(is_pickup)                  AS total_pickups,
               count(*) - sum(is_pickup)       AS total_dropoffs,
               count(DISTINCT ts::DATE)        AS active_days,
               count(DISTINCT station_id)      AS station_count,
               min(ts)                         AS first_seen,
               max(ts)                         AS last_seen
        FROM endpoints
        WHERE region_id IS NOT NULL
        GROUP BY region_id
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE active_regions AS
        SELECT * FROM region_activity
        WHERE total_trips >= {min_trips} AND active_days >= {min_days}
        """
    )
    return pl.from_arrow(con.execute("SELECT * FROM region_activity ORDER BY total_trips DESC").arrow())


def region_metadata(con: duckdb.DuckDBPyConnection, indexer) -> pl.DataFrame:
    """Geometry for the active regions - small enough for the Python H3 API."""
    active = pl.from_arrow(con.execute("SELECT * FROM active_regions").arrow())
    centroids = [indexer.centroid(r) for r in active["region_id"]]
    return active.with_columns(
        [
            pl.Series("centroid_lat", [c[0] for c in centroids]),
            pl.Series("centroid_lng", [c[1] for c in centroids]),
            pl.Series("area_km2", [indexer.area_km2(r) for r in active["region_id"]]),
            pl.Series("n_neighbors", [len(indexer.neighbors(r)) for r in active["region_id"]]),
            pl.lit(indexer.name).alias("spatial_system"),
            pl.lit(indexer.resolution).alias("resolution"),
        ]
    ).sort("total_trips", descending=True)


def dst_unreliable_sql(cfg) -> str:
    """A SQL predicate marking bins inside a DST-distorted UTC hour."""
    transitions = find_dst_transitions(
        cfg.dotted("time.timezone"),
        datetime.fromisoformat(cfg.dotted("time.start_date")),
        datetime.fromisoformat(cfg.dotted("time.end_date")),
    )
    hours = dst_unreliable_utc_hours(transitions)
    if not hours:
        return "FALSE"
    clauses = []
    for hour in hours:
        lo = hour["utc_hour_start"].astimezone(timezone.utc)
        hi = lo + timedelta(hours=1)
        clauses.append(
            f"(ts >= TIMESTAMPTZ '{lo.isoformat()}' AND ts < TIMESTAMPTZ '{hi.isoformat()}')"
        )
    return " OR ".join(clauses)


def build_panel(
    con: duckdb.DuckDBPyConnection,
    cfg,
    lo: datetime,
    hi: datetime,
    out_path: Path,
    partition_by_month: bool = True,
) -> None:
    """Build the dense panel with targets and write it to Parquet."""
    interval = cfg.dotted("time.interval_minutes")
    horizons = cfg.dotted("time.horizons")
    tz = cfg.dotted("time.timezone")

    target_cols = ",\n          ".join(
        f"lead(pickups, {h}) OVER w AS pickup_h{h},\n          "
        f"lead(dropoffs, {h}) OVER w AS dropoff_h{h}"
        for h in horizons
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW panel_dense AS
        WITH bins AS (
          SELECT unnest(generate_series(
            TIMESTAMPTZ '{lo.isoformat()}',
            TIMESTAMPTZ '{hi.isoformat()}' - INTERVAL {interval} MINUTE,
            INTERVAL {interval} MINUTE)) AS ts
        ),
        grid AS (
          SELECT r.region_id, b.ts FROM active_regions r CROSS JOIN bins b
        ),
        pickup_counts AS (
          SELECT start_region AS region_id,
                 time_bucket(INTERVAL {interval} MINUTE, started_at,
                             TIMESTAMPTZ '{lo.isoformat()}') AS ts,
                 count(*) AS pickups
          FROM trip_regions
          WHERE started_at >= TIMESTAMPTZ '{lo.isoformat()}'
            AND started_at <  TIMESTAMPTZ '{hi.isoformat()}'
            AND start_region IS NOT NULL
          GROUP BY 1, 2
        ),
        dropoff_counts AS (
          SELECT end_region AS region_id,
                 time_bucket(INTERVAL {interval} MINUTE, ended_at,
                             TIMESTAMPTZ '{lo.isoformat()}') AS ts,
                 count(*) AS dropoffs
          FROM trip_regions
          WHERE ended_at >= TIMESTAMPTZ '{lo.isoformat()}'
            AND ended_at <  TIMESTAMPTZ '{hi.isoformat()}'
            AND end_region IS NOT NULL
          GROUP BY 1, 2
        )
        SELECT g.region_id, g.ts,
               coalesce(p.pickups, 0)::INTEGER  AS pickups,
               coalesce(d.dropoffs, 0)::INTEGER AS dropoffs
        FROM grid g
        LEFT JOIN pickup_counts  p ON p.region_id = g.region_id AND p.ts = g.ts
        LEFT JOIN dropoff_counts d ON d.region_id = g.region_id AND d.ts = g.ts
        """
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    copy_options = (
        f"(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (month), OVERWRITE_OR_IGNORE 1)"
        if partition_by_month
        else "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
    )
    month_col = (
        f", strftime(ts AT TIME ZONE '{tz}', '%Y%m') AS month" if partition_by_month else ""
    )

    con.execute(
        f"""
        COPY (
          SELECT region_id, ts, pickups, dropoffs,
                 (dropoffs - pickups)::INTEGER AS net_flow,
                 ({dst_unreliable_sql(cfg)}) AS dst_unreliable,
                 {target_cols}
                 {month_col}
          FROM panel_dense
          WINDOW w AS (PARTITION BY region_id ORDER BY ts)
          -- (region, time) order, not (time, region): every downstream feature is a
          -- per-region time series (lags out to a week, rolling windows), so this is
          -- the access pattern Phase 3-4 actually reads in.
          ORDER BY region_id, ts
        ) TO '{out_path.as_posix()}' {copy_options}
        """
    )
