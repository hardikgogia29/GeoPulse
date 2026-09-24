
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.timezone import localize_to_utc  # noqa: E402
from src.utils.config import load_config, resolve_path, source_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

RENAME = {
    "Event ID": "event_id",
    "Event Name": "event_name",
    "Start Date/Time": "start_raw",
    "End Date/Time": "end_raw",
    "Event Agency": "event_agency",
    "Event Type": "event_type",
    "Event Borough": "event_borough",
    "Event Location": "event_location",
    "Event Street Side": "event_street_side",
    "Street Closure Type": "street_closure_type",
    "Community Board": "community_board",
    "Police Precinct": "police_precinct",
}
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("events", cfg)

    src = source_path(cfg, "events_csv")
    out_dir = resolve_path(cfg, "paths.external", mkdir=True)
    out_path = out_dir / "events.parquet"
    locations_path = out_dir / "event_locations.parquet"
    if out_path.exists() and not args.force:
        log.info("already present: %s - use --force to rebuild", out_path)
        return 0

    tz = cfg.dotted("time.timezone")
    start_date = cfg.dotted("time.start_date")
    end_date = cfg.dotted("time.end_date")
    keep_years = {str(y) for y in range(int(start_date[:4]) - 1, int(end_date[:4]) + 2)}

    log.info("scanning %s (%.0f MB)", src.name, src.stat().st_size / 1e6)
    lazy = (
        pl.scan_csv(src, infer_schema_length=0, ignore_errors=True)
        .rename(RENAME)
        # cheap pre-filter on the MM/DD/YYYY year substring before any date parsing
        .filter(pl.col("start_raw").str.slice(6, 4).is_in(list(keep_years)))
    )
    df = lazy.collect(engine="streaming")
    log.info("rows with a start year in %s: %s", sorted(keep_years), f"{df.height:,}")

    df = df.with_columns(
        [
            pl.col("start_raw").str.to_datetime(format=TS_FORMAT, strict=False).alias("start_naive"),
            pl.col("end_raw").str.to_datetime(format=TS_FORMAT, strict=False).alias("end_naive"),
        ]
    )
    unparsed = int(df["start_naive"].null_count() + df["end_naive"].null_count())
    if unparsed:
        log.warning("dropping %d rows with unparseable start/end timestamps", unparsed)
    df = df.drop_nulls(subset=["start_naive", "end_naive"])

    df = df.with_columns(
        [
            localize_to_utc(pl.col("start_naive"), tz).alias("event_start"),
            localize_to_utc(pl.col("end_naive"), tz).alias("event_end"),
        ]
    ).drop(["start_raw", "end_raw", "start_naive", "end_naive"])

    # compare in UTC - event_start/event_end are stored tz-aware UTC
    window_start = (pl.lit(f"{start_date}T00:00:00").str.to_datetime()
                    .dt.replace_time_zone(tz).dt.convert_time_zone("UTC"))
    window_end = (pl.lit(f"{end_date}T23:59:59").str.to_datetime()
                  .dt.replace_time_zone(tz).dt.convert_time_zone("UTC"))
    before = df.height
    df = df.filter((pl.col("event_end") >= window_start) & (pl.col("event_start") <= window_end))
    log.info("overlapping %s..%s: %s rows (dropped %s)", start_date, end_date,
             f"{df.height:,}", f"{before - df.height:,}")

    before = df.height
    df = df.unique(subset=["event_id", "event_location", "event_start", "event_end"])
    log.info("after dedupe on (event_id, location, start, end): %s rows (dropped %s)",
             f"{df.height:,}", f"{before - df.height:,}")

    df = df.with_columns(
        ((pl.col("event_end") - pl.col("event_start")).dt.total_minutes()).alias("duration_minutes")
    )

    # An event that ends before it starts is a data error, not a short event.
    if cfg.dotted("events.drop_end_before_start"):
        before = df.height
        df = df.filter(pl.col("duration_minutes") >= 0)
        log.info("rule end_before_start: dropped %s rows (%.4f%%)",
                 f"{before - df.height:,}", 100 * (before - df.height) / max(before, 1))

    # Multi-week permits are background conditions, not demand spikes. Flag, don't drop -
    # Phase 4 decides whether to treat them as events at all.
    long_running = cfg.dotted("events.long_running_minutes")
    df = df.with_columns(
        (pl.col("duration_minutes") > long_running).alias("is_long_running")
    )
    log.info("flag is_long_running (> %s min): %s rows (%.2f%%)",
             f"{long_running:,}", f"{int(df['is_long_running'].sum()):,}",
             100 * float(df["is_long_running"].mean()))
    log.info("duration_minutes: median=%s p95=%s max=%s",
             int(df["duration_minutes"].median()),
             int(df["duration_minutes"].quantile(0.95)),
             int(df["duration_minutes"].max()))

    df = df.sort("event_start")

    df.write_parquet(out_path, compression="zstd")
    log.info("wrote %s events -> %s", f"{df.height:,}", out_path)

    locations = (
        df.group_by(["event_location", "event_borough"])
        .agg(pl.len().alias("n_event_rows"))
        .sort("n_event_rows", descending=True)
    )
    locations.write_parquet(locations_path, compression="zstd")
    log.info("distinct (location, borough) strings to geocode in Phase 4: %s -> %s",
             f"{locations.height:,}", locations_path)

    log.info("distinct events: %s", f"{df['event_id'].n_unique():,}")
    log.info("borough mix: %s", df["event_borough"].value_counts(sort=True).head(8).to_dicts())
    log.info("top event types: %s", df["event_type"].value_counts(sort=True).head(8).to_dicts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
