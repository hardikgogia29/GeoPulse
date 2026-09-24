
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.splits import load_splits  # noqa: E402
from src.spatial.h3_indexer import make_indexer  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402

CALENDAR_FEATURES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
                     "is_weekend", "is_holiday"]
WEATHER_FEATURES = ["temperature_2m", "precipitation", "rain", "snowfall",
                    "wind_speed_10m", "cloud_cover"]
STATIC_FEATURES = ["area_km2", "total_trips", "station_count", "centroid_lat", "centroid_lng"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", default="h3")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.spatial)
    if args.resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: args.resolution,
                                         "resolution": args.resolution}})
    log = get_logger("bundle", cfg)
    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"

    panel_dir = resolve_path(cfg, "paths.processed") / f"panel_{tag}"
    regions_path = resolve_path(cfg, "paths.spatial") / f"regions_{tag}.parquet"
    if not panel_dir.exists():
        log.error("missing panel at %s", panel_dir)
        return 1
    out_dir = Path(args.out) if args.out else (
        resolve_path(cfg, "paths.processed") / f"deep_bundle_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with timed(log, "read panel"):
        panel = pl.scan_parquet(str(panel_dir / "**" / "*.parquet")).select(
            ["region_id", "ts", "pickups", "dropoffs", "dst_unreliable"]
        ).collect(engine="streaming")

    regions = sorted(panel["region_id"].unique().to_list())
    timestamps = panel["ts"].unique().sort()
    n_regions, n_steps = len(regions), len(timestamps)
    log.info("grid: %s timesteps x %s regions", f"{n_steps:,}", f"{n_regions:,}")

    region_index = {r: i for i, r in enumerate(regions)}
    ts_index = {t: i for i, t in enumerate(timestamps.to_list())}

    with timed(log, "pivot demand -> [T, N, 2]"):
        ordered = panel.with_columns([
            pl.col("region_id").replace_strict(region_index).alias("ri"),
            pl.col("ts").replace_strict(ts_index).alias("ti"),
        ]).sort(["ti", "ri"])
        demand = np.zeros((n_steps, n_regions, 2), dtype=np.int16)
        demand[ordered["ti"].to_numpy(), ordered["ri"].to_numpy(), 0] = \
            ordered["pickups"].to_numpy().astype(np.int16)
        demand[ordered["ti"].to_numpy(), ordered["ri"].to_numpy(), 1] = \
            ordered["dropoffs"].to_numpy().astype(np.int16)
        # the DST-distorted bins are flagged so the training loop can mask them
        dst_mask = np.zeros(n_steps, dtype=bool)
        flagged = ordered.filter(pl.col("dst_unreliable"))["ti"].unique().to_numpy()
        dst_mask[flagged] = True
    del panel, ordered

    # ---- calendar (known in advance) and weather (observed at forecast_time)
    tz = cfg.dotted("time.timezone")
    interval = cfg.dotted("time.interval_minutes")
    ts_frame = pl.DataFrame({"ts": timestamps})
    local = pl.col("ts").dt.convert_time_zone(tz)
    minutes = local.dt.hour() * 60 + local.dt.minute()
    import holidays as holidays_lib

    holiday_set = set(holidays_lib.country_holidays(
        cfg.dotted("calendar.holiday_country"), subdiv=cfg.dotted("calendar.holiday_subdiv"),
        years=range(2023, 2025)))
    calendar = ts_frame.select([
        (2 * np.pi * minutes / 1440).sin().alias("tod_sin"),
        (2 * np.pi * minutes / 1440).cos().alias("tod_cos"),
        (2 * np.pi * local.dt.weekday() / 7).sin().alias("dow_sin"),
        (2 * np.pi * local.dt.weekday() / 7).cos().alias("dow_cos"),
        (2 * np.pi * local.dt.ordinal_day() / 365.25).sin().alias("doy_sin"),
        (2 * np.pi * local.dt.ordinal_day() / 365.25).cos().alias("doy_cos"),
        (local.dt.weekday() >= 6).cast(pl.Float32).alias("is_weekend"),
        local.dt.date().is_in(list(holiday_set)).cast(pl.Float32).alias("is_holiday"),
    ]).to_numpy().astype(np.float32)

    weather_raw = pl.read_parquet(
        resolve_path(cfg, "paths.external") / "weather_hourly.parquet")
    join_hour = ts_frame.select(
        (pl.col("ts") + pl.duration(minutes=interval)).dt.truncate("1h").alias("weather_timestamp")
    )
    weather = (
        join_hour.join(weather_raw, on="weather_timestamp", how="left")
        .select(WEATHER_FEATURES).fill_null(0.0).to_numpy().astype(np.float32)
    )

    # ---- graph: first-ring adjacency, both directions, restricted to active regions
    with timed(log, "build adjacency"):
        pairs = []
        active = set(regions)
        for region in regions:
            for neighbour in indexer.neighbors(region):
                if neighbour in active:
                    pairs.append((region_index[region], region_index[neighbour]))
        edges = np.array(sorted(set(pairs)), dtype=np.int32).T
    log.info("graph: %s directed edges, mean degree %.2f",
             f"{edges.shape[1]:,}", edges.shape[1] / n_regions)

    meta_frame = pl.read_parquet(regions_path).sort("region_id")
    meta_frame = meta_frame.filter(pl.col("region_id").is_in(regions))
    order = [region_index[r] for r in meta_frame["region_id"].to_list()]
    static = np.zeros((n_regions, len(STATIC_FEATURES)), dtype=np.float32)
    static[order] = meta_frame.select(STATIC_FEATURES).to_numpy().astype(np.float32)

    # ---- split boundaries as index ranges on the time axis
    splits = load_splits(cfg)
    local_dates = ts_frame.select(local.dt.date().alias("d"))["d"].to_numpy()
    bounds = {}
    for name, split in splits.items():
        mask = (local_dates >= np.datetime64(split.start)) & \
               (local_dates <= np.datetime64(split.end))
        idx = np.flatnonzero(mask)
        bounds[name] = [int(idx[0]), int(idx[-1]) + 1]
    log.info("split index ranges: %s", bounds)

    # ---- scaling stats on TRAIN ONLY
    train_lo, train_hi = bounds["train"]
    train_demand = demand[train_lo:train_hi].astype(np.float32)
    scaling = {
        "demand_mean": train_demand.mean(axis=(0, 1)).tolist(),
        "demand_std": (train_demand.std(axis=(0, 1)) + 1e-6).tolist(),
        "weather_mean": weather[train_lo:train_hi].mean(axis=0).tolist(),
        "weather_std": (weather[train_lo:train_hi].std(axis=0) + 1e-6).tolist(),
        "static_mean": static.mean(axis=0).tolist(),
        "static_std": (static.std(axis=0) + 1e-6).tolist(),
    }
    del train_demand

    np.save(out_dir / "demand.npy", demand)
    np.save(out_dir / "calendar.npy", calendar)
    np.save(out_dir / "weather.npy", weather)
    np.save(out_dir / "edges.npy", edges)
    np.save(out_dir / "node_static.npy", static)
    np.save(out_dir / "dst_mask.npy", dst_mask)
    (out_dir / "meta.json").write_text(json.dumps({
        "tag": tag, "spatial_system": indexer.name, "resolution": indexer.resolution,
        "n_steps": n_steps, "n_regions": n_regions,
        "interval_minutes": interval, "timezone": tz,
        "horizons": cfg.dotted("time.horizons"),
        "calendar_features": CALENDAR_FEATURES,
        "weather_features": WEATHER_FEATURES,
        "static_features": STATIC_FEATURES,
        "region_ids": regions,
        "first_ts": str(timestamps[0]), "last_ts": str(timestamps[-1]),
        "split_index": bounds,
        "scaling_fitted_on": "train",
        "scaling": scaling,
    }, indent=2), encoding="utf-8")

    total_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    log.info("bundle -> %s (%.0f MB)", out_dir, total_mb)
    for f in sorted(out_dir.iterdir()):
        log.info("  %-18s %8.1f MB", f.name, f.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
