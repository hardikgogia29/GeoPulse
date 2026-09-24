
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spatial.h3_indexer import load_h3_extension, make_indexer  # noqa: E402
from src.models.splits import load_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-event-hours", type=int, default=24)
    parser.add_argument("--memory-limit", default="9GB")
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()

    cfg = load_config("h3")
    if args.resolution is not None:
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], "resolution": args.resolution}})
    log = get_logger("external", cfg)
    indexer = make_indexer(cfg)
    tag = f"h3{indexer.resolution}"
    interval = cfg.dotted("time.interval_minutes")

    processed = resolve_path(cfg, "paths.processed", mkdir=True)
    spatial = resolve_path(cfg, "paths.spatial")
    external = resolve_path(cfg, "paths.external")
    interim = resolve_path(cfg, "paths.interim")

    regions_path = spatial / f"regions_{tag}.parquet"
    if not regions_path.exists():
        log.error("missing %s - run scripts/10_build_panel.py", regions_path)
        return 1

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(args.temp_dir).as_posix()}'")
    load_h3_extension(con)
    con.execute(f"CREATE VIEW regions AS SELECT * FROM read_parquet('{regions_path.as_posix()}')")

    # ---------------------------------------------------------------- EVENTS
    geocodes_path = external / "event_geocodes.parquet"
    if geocodes_path.exists():
        with timed(log, "map events -> (region, bin)"):
            con.execute(
                f"CREATE VIEW geocodes AS SELECT * FROM read_parquet('{geocodes_path.as_posix()}') "
                f"WHERE status = 'ok' AND lat IS NOT NULL"
            )
            con.execute(
                f"""
                CREATE OR REPLACE TABLE event_cells AS
                SELECT event_location, event_borough,
                       h3_latlng_to_cell_string(lat, lng, {indexer.resolution}) AS home_cell
                FROM geocodes
                """
            )
            # containing cell (ring 0) + first ring: a permit on a cell edge affects both
            con.execute(
                """
                CREATE OR REPLACE TABLE event_regions AS
                SELECT ec.event_location, ec.event_borough,
                       unnest(h3_grid_disk(ec.home_cell, 1)) AS region_id,
                       ec.home_cell
                FROM event_cells ec
                """
            )
            con.execute(
                "CREATE OR REPLACE TABLE event_regions AS "
                "SELECT er.*, (er.region_id = er.home_cell) AS is_home "
                "FROM event_regions er "
                "WHERE er.region_id IN (SELECT region_id FROM regions)"
            )
            matched = con.execute("SELECT count(*) FROM event_regions").fetchone()[0]
            log.info("  event locations mapped into %s (region, location) pairs", f"{matched:,}")

            events_path = external / "events.parquet"
            con.execute(
                f"CREATE VIEW events_raw AS SELECT * FROM read_parquet('{events_path.as_posix()}')"
            )
            con.execute(
                f"""
                CREATE OR REPLACE TABLE event_region_bins AS
                WITH bounded AS (
                  SELECT e.event_start, e.event_type,
                         least(e.event_end,
                               e.event_start + INTERVAL {args.max_event_hours} HOUR) AS event_end,
                         er.region_id, er.is_home
                  FROM events_raw e
                  JOIN event_regions er
                    ON er.event_location = e.event_location
                   AND er.event_borough IS NOT DISTINCT FROM e.event_borough
                ),
                expanded AS (
                  SELECT region_id, is_home, event_start, event_end, event_type,
                         unnest(generate_series(
                           time_bucket(INTERVAL {interval} MINUTE, event_start),
                           time_bucket(INTERVAL {interval} MINUTE, event_end),
                           INTERVAL {interval} MINUTE)) AS ts
                  FROM bounded
                )
                SELECT region_id, ts,
                       count(*) FILTER (WHERE is_home)            AS event_count,
                       count(*) FILTER (WHERE NOT is_home)        AS event_count_ring,
                       count(DISTINCT event_type)                 AS event_attendee_types,
                       count(*) FILTER (
                         WHERE ts < event_start + INTERVAL {cfg.dotted("advanced_features.event_recent_minutes")} MINUTE
                       )                                          AS event_started_recently,
                       count(*) FILTER (
                         WHERE ts >= event_end - INTERVAL {cfg.dotted("advanced_features.event_ending_minutes")} MINUTE
                       )                                          AS event_ending_soon
                FROM expanded
                GROUP BY region_id, ts
                """
            )
            out = processed / f"event_bins_{tag}.parquet"
            con.execute(
                f"COPY (SELECT * FROM event_region_bins ORDER BY region_id, ts) "
                f"TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            stats = con.execute(
                "SELECT count(*), count(DISTINCT region_id), sum(event_count) FROM event_region_bins"
            ).fetchone()
            log.info("  event bins: %s rows across %s regions, %s home-cell event-bins -> %s",
                     f"{stats[0]:,}", f"{stats[1]:,}", f"{stats[2]:,}", out.name)
    else:
        log.warning("no geocodes at %s - run scripts/15_geocode_events.py", geocodes_path)

    # --------------------------------------------------------------- TRAFFIC
    links_path = spatial / "traffic_links.parquet"
    traffic_path = interim / "traffic_clean.parquet"
    if links_path.exists() and traffic_path.exists():
        with timed(log, "map traffic -> (region, bin)"):
            # A link is a LINE, not a point. Attributing it only to the cell holding
            # its midpoint threw away most of its extent: 18 of 128 links landed in an
            # active region. Exploding the polyline and taking every cell any vertex
            # falls in is both more accurate and fairer to the traffic ablation - a
            # strawman feature that loses is not a result.
            con.execute(
                f"""
                CREATE OR REPLACE TABLE link_regions AS
                WITH vertices AS (
                  SELECT link_id,
                         unnest(str_split(trim(regexp_replace(link_points, '\\s+', ' ', 'g')), ' '))
                           AS point
                  FROM read_parquet('{links_path.as_posix()}')
                  WHERE link_points IS NOT NULL
                ),
                parsed AS (
                  SELECT link_id,
                         try_cast(str_split(point, ',')[1] AS DOUBLE) AS lat,
                         try_cast(str_split(point, ',')[2] AS DOUBLE) AS lng
                  FROM vertices
                )
                SELECT DISTINCT link_id,
                       h3_latlng_to_cell_string(lat, lng, {indexer.resolution}) AS region_id
                FROM parsed
                WHERE lat IS NOT NULL AND lng IS NOT NULL
                """
            )
            covered_links, covered_regions = con.execute(
                "SELECT count(DISTINCT link_id), count(DISTINCT region_id) FROM link_regions "
                "WHERE region_id IN (SELECT region_id FROM regions)"
            ).fetchone()
            total_links = con.execute("SELECT count(DISTINCT link_id) FROM link_regions").fetchone()[0]
            n_regions = con.execute("SELECT count(*) FROM regions").fetchone()[0]
            log.info("  %d of %d traffic links touch an active region, covering %d of %d "
                     "regions (%.1f%%)", covered_links, total_links, covered_regions,
                     n_regions, 100 * covered_regions / n_regions)

            con.execute(
                f"CREATE VIEW traffic_raw AS SELECT * FROM read_parquet('{traffic_path.as_posix()}')"
            )
            train = load_splits(cfg)["train"]
            tz = cfg.dotted("time.timezone")
            # free-flow reference from TRAIN only - using the whole series would leak
            # validation/test conditions into a training feature
            con.execute(
                f"""
                CREATE OR REPLACE TABLE link_freeflow AS
                SELECT link_id,
                       quantile_cont(speed, 0.85)       AS free_flow_speed,
                       quantile_cont(travel_time, 0.15) AS free_flow_travel_time
                FROM traffic_raw
                WHERE (data_as_of AT TIME ZONE '{tz}')::DATE
                        BETWEEN DATE '{train.start}' AND DATE '{train.end}'
                GROUP BY link_id
                """
            )
            con.execute(
                f"""
                CREATE OR REPLACE TABLE traffic_region_bins AS
                WITH binned AS (
                  SELECT lr.region_id,
                         time_bucket(INTERVAL {interval} MINUTE, t.data_as_of) AS ts,
                         t.link_id, avg(t.speed) AS speed, avg(t.travel_time) AS travel_time
                  FROM traffic_raw t
                  JOIN link_regions lr ON lr.link_id = t.link_id
                  GROUP BY 1, 2, 3
                )
                SELECT b.region_id, b.ts,
                       avg(b.speed)::FLOAT                      AS traffic_speed_mean,
                       min(b.speed)::FLOAT                      AS traffic_speed_min,
                       count(DISTINCT b.link_id)                AS traffic_link_count,
                       avg(b.speed / nullif(f.free_flow_speed, 0))::FLOAT
                                                                AS congestion_index,
                       avg(b.travel_time / nullif(f.free_flow_travel_time, 0))::FLOAT
                                                                AS travel_time_index
                FROM binned b
                LEFT JOIN link_freeflow f ON f.link_id = b.link_id
                GROUP BY b.region_id, b.ts
                """
            )
            out = processed / f"traffic_bins_{tag}.parquet"
            con.execute(
                f"""
                COPY (
                  SELECT *, traffic_speed_mean
                            - lag(traffic_speed_mean) OVER (PARTITION BY region_id ORDER BY ts)
                            AS traffic_change_15m
                  FROM traffic_region_bins ORDER BY region_id, ts
                ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            stats = con.execute(
                "SELECT count(*), count(DISTINCT region_id) FROM traffic_region_bins"
            ).fetchone()
            log.info("  traffic bins: %s rows across %s regions -> %s",
                     f"{stats[0]:,}", f"{stats[1]:,}", out.name)
    else:
        log.warning("traffic inputs missing - skipping traffic mapping")

    # ------------------------------------------------------- STATION / NETWORK
    registry_path = spatial / "station_registry.parquet"
    trips_path = interim / "trips_clean.parquet"
    if registry_path.exists():
        with timed(log, "station/network context -> (region, month)"):
            tz = cfg.dotted("time.timezone")
            con.execute(
                f"""
                CREATE OR REPLACE TABLE station_cells AS
                SELECT station_id, first_seen, appearances,
                       h3_latlng_to_cell_string(lat, lng, {indexer.resolution}) AS region_id
                FROM read_parquet('{registry_path.as_posix()}')
                """
            )
            months = con.execute(
                f"""
                SELECT DISTINCT strftime(ts AT TIME ZONE '{tz}', '%Y%m') AS month
                FROM read_parquet('{(resolve_path(cfg, "paths.processed") / f"panel_{tag}" / "**" / "*.parquet").as_posix()}',
                                  hive_partitioning=true)
                ORDER BY 1
                """
            ).fetchall()
            month_list = ", ".join(f"'{m[0]}'" for m in months)
            con.execute(
                f"""
                CREATE OR REPLACE TABLE station_network AS
                WITH months AS (SELECT unnest([{month_list}]) AS month),
                     grid AS (SELECT r.region_id, m.month, r.area_km2
                              FROM regions r CROSS JOIN months m)
                SELECT g.region_id, g.month,
                       count(s.station_id)                          AS active_station_count,
                       (count(s.station_id) / g.area_km2)::FLOAT    AS station_density_km2,
                       coalesce(sum(s.appearances), 0)              AS region_hist_trip_volume,
                       g.area_km2::FLOAT                            AS region_area_km2
                FROM grid g
                LEFT JOIN station_cells s
                  ON s.region_id = g.region_id
                 -- a station counts only once it has actually appeared: comparing
                 -- first_seen to the START of the month keeps future stations out
                 AND strftime(s.first_seen AT TIME ZONE '{tz}', '%Y%m') <= g.month
                GROUP BY g.region_id, g.month, g.area_km2
                """
            )
            out = processed / f"station_network_{tag}.parquet"
            con.execute(
                f"COPY (SELECT * FROM station_network ORDER BY region_id, month) "
                f"TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            stats = con.execute(
                "SELECT count(*), min(active_station_count), max(active_station_count) "
                "FROM station_network"
            ).fetchone()
            log.info("  station/network: %s (region, month) rows, stations per region %d..%d -> %s",
                     f"{stats[0]:,}", stats[1], stats[2], out.name)

    log.info("external context ready in %s", processed)
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
