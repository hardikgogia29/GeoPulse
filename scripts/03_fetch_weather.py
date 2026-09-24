
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

TIMEOUT_SECONDS = 180


def fetch(url: str, params: dict, log) -> dict:
    """GET with a graceful retry that drops hourly variables the archive rejects."""
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    if response.status_code == 400:
        reason = response.json().get("reason", "")
        log.warning("Open-Meteo rejected the request: %s", reason)
        bad = [v for v in params["hourly"].split(",") if v in reason]
        if not bad:
            response.raise_for_status()
        kept = [v for v in params["hourly"].split(",") if v not in bad]
        log.warning("retrying without unsupported variables: %s", ", ".join(bad))
        params = {**params, "hourly": ",".join(kept)}
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("weather", cfg)

    out_path = resolve_path(cfg, "paths.external", mkdir=True) / "weather_hourly.parquet"
    if out_path.exists() and not args.force:
        existing = pl.read_parquet(out_path)
        log.info("already present: %s (%d rows) - use --force to refetch", out_path, existing.height)
        return 0

    params = {
        "latitude": cfg.dotted("weather.latitude"),
        "longitude": cfg.dotted("weather.longitude"),
        "start_date": cfg.dotted("time.start_date"),
        "end_date": cfg.dotted("time.end_date"),
        "hourly": ",".join(cfg.dotted("weather.hourly_variables")),
        "timezone": "UTC",
    }
    log.info("fetching %s .. %s from %s", params["start_date"], params["end_date"],
             cfg.dotted("weather.archive_url"))
    payload = fetch(cfg.dotted("weather.archive_url"), params, log)

    hourly = payload["hourly"]
    df = pl.DataFrame(hourly).with_columns(
        pl.col("time").str.to_datetime(format="%Y-%m-%dT%H:%M", time_unit="us")
        .dt.replace_time_zone("UTC").alias("weather_timestamp")
    ).drop("time")
    df = df.select(["weather_timestamp"] + [c for c in df.columns if c != "weather_timestamp"])
    df = df.sort("weather_timestamp")

    expected_hours = df["weather_timestamp"].dt.truncate("1h").n_unique()
    null_share = {c: round(df[c].null_count() / df.height, 4) for c in df.columns}

    df.write_parquet(out_path, compression="zstd")
    log.info("wrote %s rows -> %s", f"{df.height:,}", out_path)
    log.info("range: %s .. %s (%d distinct hours)",
             df["weather_timestamp"].min(), df["weather_timestamp"].max(), expected_hours)
    log.info("variables: %s", ", ".join(c for c in df.columns if c != "weather_timestamp"))
    for col, share in null_share.items():
        if share > 0:
            log.warning("  %-24s %.2f%% null", col, share * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
