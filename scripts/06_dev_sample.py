
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402


def _local_window(cfg, warmup_days: int) -> tuple[datetime, datetime, datetime, str]:
    tz = cfg.dotted("time.timezone")
    start = datetime.fromisoformat(cfg.dotted("dev_sample.start"))
    end = datetime.fromisoformat(cfg.dotted("dev_sample.end")) + timedelta(days=1)
    warmup = start - timedelta(days=warmup_days)
    return warmup, start, end, tz


def _to_utc(naive: datetime, tz: str) -> datetime:
    return (
        pl.select(pl.lit(naive).dt.replace_time_zone(tz).dt.convert_time_zone("UTC"))
        .item()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-days", type=int, default=8,
                        help="extra days of weather/event history for lag features")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("devsample", cfg)

    out_dir = resolve_path(cfg, "paths.dev_sample", mkdir=True)
    interim = resolve_path(cfg, "paths.interim")
    external = resolve_path(cfg, "paths.external")
    spatial = resolve_path(cfg, "paths.spatial")

    warmup, start, end, tz = _local_window(cfg, args.warmup_days)
    warmup_utc, start_utc, end_utc = (_to_utc(t, tz) for t in (warmup, start, end))
    log.info("dev window (local): %s .. %s | warm-up from %s",
             start.date(), (end - timedelta(days=1)).date(), warmup.date())

    manifest: dict = {
        "window_local": {"start": str(start), "end": str(end), "warmup_start": str(warmup)},
        "window_utc": {"start": str(start_utc), "end": str(end_utc), "warmup_start": str(warmup_utc)},
        "timezone": tz,
        "warmup_days": args.warmup_days,
    }

    trips_src = interim / "trips_clean.parquet"
    if not trips_src.exists():
        log.error("missing %s - run scripts/02_clean.py first", trips_src)
        return 1
    # Trips carry the warm-up prefix too: the longest Phase 3 lag is one week, so a
    # bare 7-day slice leaves every row with null lags and zero usable training rows.
    trips = (
        pl.scan_parquet(trips_src)
        .filter((pl.col("started_at") >= warmup_utc) & (pl.col("started_at") < end_utc))
        .collect(engine="streaming")
    )
    trips.write_parquet(out_dir / "trips_clean.parquet", compression="zstd")
    manifest["trips_rows"] = trips.height
    log.info("trips: %s rows (incl. %d warm-up days) -> %s", f"{trips.height:,}",
             args.warmup_days, out_dir / "trips_clean.parquet")

    weather_src = external / "weather_hourly.parquet"
    if weather_src.exists():
        weather = (
            pl.scan_parquet(weather_src)
            .filter(
                (pl.col("weather_timestamp") >= warmup_utc)
                & (pl.col("weather_timestamp") < end_utc)
            )
            .collect()
        )
        weather.write_parquet(out_dir / "weather_hourly.parquet", compression="zstd")
        manifest["weather_rows"] = weather.height
        log.info("weather: %s rows (incl. %d warm-up days)", f"{weather.height:,}", args.warmup_days)
    else:
        log.warning("no weather file at %s - run scripts/03_fetch_weather.py", weather_src)

    events_src = external / "events.parquet"
    if events_src.exists():
        events = (
            pl.scan_parquet(events_src)
            .filter((pl.col("event_end") >= warmup_utc) & (pl.col("event_start") < end_utc))
            .collect()
        )
        events.write_parquet(out_dir / "events.parquet", compression="zstd")
        manifest["event_rows"] = events.height
        log.info("events: %s rows overlapping the window", f"{events.height:,}")
    else:
        log.warning("no events file at %s - run scripts/04_prepare_events.py", events_src)

    traffic_src = interim / "traffic_clean.parquet"
    if traffic_src.exists():
        traffic = (
            pl.scan_parquet(traffic_src)
            .filter((pl.col("data_as_of") >= warmup_utc) & (pl.col("data_as_of") < end_utc))
            .collect(engine="streaming")
        )
        traffic.write_parquet(out_dir / "traffic_clean.parquet", compression="zstd")
        manifest["traffic_rows"] = traffic.height
        log.info("traffic: %s readings (incl. %d warm-up days)",
                 f"{traffic.height:,}", args.warmup_days)
    else:
        log.warning("no cleaned traffic at %s - run scripts/08_clean_traffic.py", traffic_src)

    links_src = spatial / "traffic_links.parquet"
    if links_src.exists():
        links = pl.read_parquet(links_src)
        links.write_parquet(out_dir / "traffic_links.parquet", compression="zstd")
        manifest["traffic_link_rows"] = links.height
        log.info("traffic links: %s", f"{links.height:,}")

    registry_src = spatial / "station_registry.parquet"
    if registry_src.exists():
        active = set(trips["start_station_id"].drop_nulls().unique().to_list()) | set(
            trips["end_station_id"].drop_nulls().unique().to_list()
        )
        registry = pl.read_parquet(registry_src).filter(pl.col("station_id").is_in(list(active)))
        registry.write_parquet(out_dir / "station_registry.parquet", compression="zstd")
        manifest["station_rows"] = registry.height
        log.info("stations active in the window: %s", f"{registry.height:,}")
    else:
        log.warning("no registry at %s - run scripts/05_station_registry.py", registry_src)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("dev sample ready: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
