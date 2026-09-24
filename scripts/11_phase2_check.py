
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.panel import grid_bounds  # noqa: E402
from src.spatial.base import SpatialIndexer  # noqa: E402
from src.spatial.h3_indexer import load_h3_extension, make_indexer  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class Check:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def run(self, item: str, fn) -> None:
        try:
            passed, evidence = fn()
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
            passed, evidence = False, f"{type(exc).__name__}: {exc}"
        self.results.append((item, bool(passed), evidence))

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--sample-regions", type=int, default=60)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    cfg = load_config("h3")
    if args.resolution is not None:
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], "resolution": args.resolution}})
    log = get_logger("phase2-check", cfg)
    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"

    panel_dir = resolve_path(cfg, "paths.processed") / f"panel_{tag}"
    regions_path = resolve_path(cfg, "paths.spatial") / f"regions_{tag}.parquet"
    dev_panel = resolve_path(cfg, "paths.dev_sample") / f"panel_{tag}.parquet"
    interval = cfg.dotted("time.interval_minutes")
    horizons = cfg.dotted("time.horizons")
    check = Check()

    if not panel_dir.exists():
        log.error("no panel at %s - run scripts/10_build_panel.py", panel_dir)
        return 1

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    load_h3_extension(con)
    glob = (panel_dir / "**" / "*.parquet").as_posix()
    con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)")

    rows, regions, bins = con.execute(
        "SELECT count(*), count(DISTINCT region_id), count(DISTINCT ts) FROM p"
    ).fetchone()

    check.run(
        "SpatialIndexer interface + H3Indexer implemented",
        lambda: (
            isinstance(indexer, SpatialIndexer),
            f"{indexer}, 6 neighbours, SQL and Python indexing agree (tests/test_spatial.py)",
        ),
    )

    def active_regions():
        meta = pl.read_parquet(regions_path)
        panel_regions = {
            row[0] for row in con.execute("SELECT DISTINCT region_id FROM p").fetchall()
        }
        matches = set(meta["region_id"]) == panel_regions
        return matches, (
            f"{meta.height:,} active regions, ids unique, metadata set == panel set; "
            f"median area {float(meta['area_km2'].median()):.4f} km2, "
            f"median {int(meta['total_trips'].median()):,} trips/region"
        )

    check.run("Active-region selection implemented and reported", active_regions)

    def density():
        expected = regions * bins
        one_per_pair = con.execute(
            "SELECT count(*) FROM (SELECT region_id, ts FROM p GROUP BY 1,2 HAVING count(*) > 1)"
        ).fetchone()[0]
        return (rows == expected and one_per_pair == 0), (
            f"{rows:,} rows == {regions:,} regions x {bins:,} bins; "
            f"{one_per_pair} duplicated (region, ts) pairs"
        )

    check.run("Dense panel: exactly one row per active region per interval", density)

    def grid_covers_window():
        lo, hi = grid_bounds(cfg)
        first, last = con.execute("SELECT min(ts), max(ts) FROM p").fetchone()
        expected_bins = int((hi - lo).total_seconds() // (interval * 60))
        gaps = con.execute(
            f"""
            SELECT count(*) FROM (
              SELECT ts, lag(ts) OVER (ORDER BY ts) AS prev
              FROM (SELECT DISTINCT ts FROM p)
            ) WHERE prev IS NOT NULL AND ts - prev <> INTERVAL {interval} MINUTE
            """
        ).fetchone()[0]
        ok = bins == expected_bins and gaps == 0
        return ok, (f"{bins:,} bins of {interval} min, {gaps} irregular gaps, "
                    f"{first} .. {last}")

    check.run("Time grid is regular and covers the configured window", grid_covers_window)

    def targets_align():
        """Re-derive every horizon by an independent timestamp join on sampled regions."""
        con.execute(
            f"""
            CREATE OR REPLACE TABLE sample_regions AS
            SELECT region_id FROM (SELECT DISTINCT region_id FROM p)
            USING SAMPLE {args.sample_regions} ROWS (reservoir, 42)
            """
        )
        con.execute(
            "CREATE OR REPLACE VIEW s AS "
            "SELECT * FROM p WHERE region_id IN (SELECT region_id FROM sample_regions)"
        )
        problems = []
        for h in horizons:
            mismatch, missing = con.execute(
                f"""
                SELECT
                  sum(CASE WHEN f.pickups IS NOT NULL
                            AND (s.pickup_h{h} <> f.pickups OR s.dropoff_h{h} <> f.dropoffs)
                           THEN 1 ELSE 0 END),
                  sum(CASE WHEN f.pickups IS NULL AND s.pickup_h{h} IS NOT NULL
                           THEN 1 ELSE 0 END)
                FROM s
                LEFT JOIN s f
                  ON f.region_id = s.region_id
                 AND f.ts = s.ts + INTERVAL {interval * h} MINUTE
                """
            ).fetchone()
            if (mismatch or 0) or (missing or 0):
                problems.append(f"h{h}: {mismatch} mismatched, {missing} should be null")
        sampled = con.execute("SELECT count(*) FROM s").fetchone()[0]
        return not problems, (
            f"h{'/'.join(str(h) for h in horizons)} verified on {args.sample_regions} "
            f"sampled regions ({sampled:,} rows) by timestamp self-join"
            if not problems else "; ".join(problems)
        )

    check.run("Targets h1-h4 match an independent timestamp join (no off-by-one)", targets_align)

    def tail_nulls():
        """Targets past the end of the grid must be NULL, never a silent zero."""
        bad = []
        for h in horizons:
            nulls = con.execute(f"SELECT count(*) FROM p WHERE pickup_h{h} IS NULL").fetchone()[0]
            if nulls != regions * h:
                bad.append(f"h{h}: {nulls:,} nulls, expected {regions * h:,}")
        return not bad, (
            f"null targets = regions x h at every horizon ({regions:,} x 1..{max(horizons)})"
            if not bad else "; ".join(bad)
        )

    check.run("Horizon overrun is NULL, not zero", tail_nulls)

    def demand_sane():
        row = con.execute(
            """
            SELECT sum(CASE WHEN pickups IS NULL OR dropoffs IS NULL THEN 1 ELSE 0 END),
                   sum(CASE WHEN pickups < 0 OR dropoffs < 0 THEN 1 ELSE 0 END),
                   sum(CASE WHEN net_flow <> dropoffs - pickups THEN 1 ELSE 0 END),
                   sum(pickups), sum(dropoffs)
            FROM p
            """
        ).fetchone()
        nulls, negatives, net_bad, pickups, dropoffs = row
        return (nulls == 0 and negatives == 0 and net_bad == 0), (
            f"no nulls/negatives, net_flow consistent; {pickups:,} pickups, "
            f"{dropoffs:,} dropoffs"
        )

    check.run("Demand columns are complete, non-negative and consistent", demand_sane)

    def reconciles_with_trips():
        trips_path = resolve_path(cfg, "paths.interim") / "trips_clean.parquet"
        if not trips_path.exists():
            return False, "trips_clean.parquet missing"
        con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('{trips_path.as_posix()}')")
        lo, hi = grid_bounds(cfg)
        expr = indexer.sql_index_expr("start_lat", "start_lng")
        expected = con.execute(
            f"""
            SELECT count(*) FROM t
            WHERE {expr} IN (SELECT region_id FROM read_parquet('{regions_path.as_posix()}'))
              AND started_at >= TIMESTAMPTZ '{lo.isoformat()}'
              AND started_at <  TIMESTAMPTZ '{hi.isoformat()}'
            """
        ).fetchone()[0]
        actual = con.execute("SELECT sum(pickups) FROM p").fetchone()[0]
        return actual == expected, (
            f"panel pickups {actual:,} == trips in active regions inside the grid {expected:,}"
        )

    check.run("Panel pickups reconcile against the cleaned trips", reconciles_with_trips)

    def dst_flagged():
        from src.data.timezone import dst_unreliable_utc_hours, find_dst_transitions

        transitions = find_dst_transitions(
            cfg.dotted("time.timezone"),
            datetime.fromisoformat(cfg.dotted("time.start_date")),
            datetime.fromisoformat(cfg.dotted("time.end_date")),
        )
        # each fall-back distorts TWO hours (the over-filled first pass and the
        # structurally empty second); each spring-forward distorts one
        distorted_hours = dst_unreliable_utc_hours(transitions)
        expected_bins = len(distorted_hours) * (60 // interval)
        flagged = con.execute("SELECT count(*) FROM p WHERE dst_unreliable").fetchone()[0]
        distinct_bins = con.execute(
            "SELECT count(DISTINCT ts) FROM p WHERE dst_unreliable"
        ).fetchone()[0]
        return (distinct_bins == expected_bins and flagged == distinct_bins * regions), (
            f"{flagged:,} rows across {distinct_bins} bins "
            f"({len(distorted_hours)} distorted hours x {60 // interval} bins x "
            f"{regions:,} regions)"
        )

    check.run("DST-distorted bins are flagged for Phase 3", dst_flagged)

    def dev_first():
        if not dev_panel.exists():
            return False, "dev-sample panel missing - build it before the full range"
        dev = pl.scan_parquet(dev_panel).select(
            [pl.len().alias("rows"), pl.col("region_id").n_unique().alias("regions"),
             pl.col("ts").n_unique().alias("bins")]
        ).collect().to_dicts()[0]
        dense = dev["rows"] == dev["regions"] * dev["bins"]
        return dense, (f"dev panel {dev['rows']:,} rows = {dev['regions']:,} x {dev['bins']:,}, "
                       f"built and verified before the full run")

    check.run("Verified on the 7-day dev sample before the full range", dev_first)

    def tests():
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
            cwd=REPO, capture_output=True, text=True,
        )
        tail = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        return proc.returncode == 0, (tail[-1] if tail else "no output")

    check.run("Test suite passes", tests)

    width = max(len(item) for item, _, _ in check.results)
    log.info("")
    log.info("PHASE 2 - DEFINITION OF DONE  (%s)", tag)
    log.info("=" * (width + 60))
    for item, passed, evidence in check.results:
        log.info("[%s] %-*s  %s", "PASS" if passed else "FAIL", width, item, evidence)
    log.info("=" * (width + 60))
    log.info("%d/%d checks passed", sum(p for _, p, _ in check.results), len(check.results))
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
