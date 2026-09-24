
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.features.advanced import cumulative_feature_sets
from src.models import baseline
from src.utils.config import Config, load_config, resolve_path

TARGET_KINDS = {"pickup": "pickups", "dropoff": "dropoffs"}

#: the columns the Phase 8 contract promises, in order
OUTPUT_COLUMNS = [
    "region_id", "forecast_time", "horizon",
    "predicted_pickups", "predicted_dropoffs",
    "projected_inventory", "shortage", "surplus",
]


class PredictionError(RuntimeError):
    """Raised when a request cannot be served truthfully rather than approximately."""


def _config(spatial_system: str, resolution: int | None) -> Config:
    cfg = load_config(spatial_system, "lightgbm")
    if resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: resolution,
                                         "resolution": resolution}})
    return cfg


def _tag(cfg) -> str:
    from src.spatial.h3_indexer import make_indexer

    indexer = make_indexer(cfg)
    return f"{indexer.name}{indexer.resolution}"


def available_models(spatial_system: str = "h3",
                     resolution: int | None = None) -> list[str]:
    """Model names `predict` can actually serve, judged from artifacts on disk."""
    cfg = _config(spatial_system, resolution)
    tag = _tag(cfg)
    models_dir = resolve_path(cfg, "paths.models")
    names = []
    if (models_dir / f"lgbm_final_{tag}_pickup_h1.txt").exists():
        names.append("lightgbm_final")
    names += [f"seasonal_naive_{v}" for v in baseline.seasonal_lag_steps(cfg)]
    return names


def predict(spatial_system: str,
            model_name: str,
            forecast_time: datetime,
            horizon: int,
            *,
            resolution: int | None = None,
            safety_pct: float | None = None,
            target_pct: float | None = None) -> pl.DataFrame:
    """Forecast every active region for one instant, with the operational layer attached.

    Parameters
    ----------
    spatial_system : "h3" or "s2"
    model_name     : one of `available_models()`
    forecast_time  : tz-aware instant on a bin boundary
    horizon        : steps ahead; must be one of `time.horizons`
    """
    cfg = _config(spatial_system, resolution)
    tag = _tag(cfg)
    interval = cfg.dotted("time.interval_minutes")
    tz = cfg.dotted("time.timezone")
    horizons = cfg.dotted("time.horizons")

    if horizon not in horizons:
        raise PredictionError(f"horizon {horizon} not in trained horizons {horizons}")
    if forecast_time.tzinfo is None:
        raise PredictionError("forecast_time must be timezone-aware")
    step = timedelta(minutes=interval)
    epoch = datetime(1970, 1, 1, tzinfo=forecast_time.tzinfo)
    if (forecast_time - epoch) % step != timedelta(0):
        raise PredictionError(
            f"forecast_time {forecast_time} is not on a {interval}-minute bin boundary")

    ts = forecast_time - step  # forecast_time is the END of bin ts
    processed = resolve_path(cfg, "paths.processed")
    models_dir = resolve_path(cfg, "paths.models")

    ops = cfg.dotted("operations")
    if safety_pct is None:
        safety_pct = ops["safety_inventory_pcts"][1]
    if target_pct is None:
        target_pct = ops["target_inventory_pct"]

    steps = [h for h in horizons if 1 <= h <= horizon]

    # --------------------------------------------------------- features and models
    season = None
    features: list[str] = []
    boosters: dict[tuple[str, int], lgb.Booster] = {}
    if model_name == "lightgbm_final":
        table = processed / f"features4_{tag}"
        families = cfg.dotted("advanced_features.final_families")
        features = cumulative_feature_sets(cfg)[families[-1]]
        needed = ["region_id", "ts", *features]
        for kind in TARGET_KINDS:
            for h in steps:
                path = models_dir / f"lgbm_final_{tag}_{kind}_h{h}.txt"
                if not path.exists():
                    raise PredictionError(f"missing model artifact {path}")
                boosters[(kind, h)] = lgb.Booster(model_file=str(path))
    elif model_name.startswith("seasonal_naive_"):
        variant = model_name[len("seasonal_naive_"):]
        seasons = baseline.seasonal_lag_steps(cfg)
        if variant not in seasons:
            raise PredictionError(
                f"unknown seasonal variant {variant!r}; have {list(seasons)}")
        season = seasons[variant]
        table = processed / f"features_{tag}"
        needed = ["region_id", "ts"] + [
            baseline.lag_column_for(source, season, h)
            for source in TARGET_KINDS.values() for h in steps
        ]
    else:
        raise PredictionError(
            f"unknown model {model_name!r}; available: "
            f"{available_models(spatial_system, resolution)}")

    table = Path(table)
    if not table.exists():
        raise PredictionError(f"missing feature table {table} - build it first")

    pattern = str(table / "**" / "*.parquet")
    frame = (pl.scan_parquet(pattern)
             .filter(pl.col("ts") == ts)
             .select(sorted(set(needed)))
             .collect(engine="streaming")
             .sort("region_id"))
    if frame.height == 0:
        raise PredictionError(
            f"no feature row at ts={ts} (forecast_time={forecast_time}) in {table}")
    regions = frame["region_id"].to_list()

    # ------------------------------------------------------------------ predictions
    per_horizon: dict[int, dict[str, np.ndarray]] = {}
    if model_name == "lightgbm_final":
        # region_idx has to be built exactly as training built it: a dense rank over
        # the table's full sorted region list, never over this one timestamp's slice.
        all_regions = sorted(pl.scan_parquet(pattern).select("region_id").unique()
                             .collect(engine="streaming")["region_id"].to_list())
        mapping = {r: i for i, r in enumerate(all_regions)}
        inputs = ["region_idx"] + [c for c in features if c != "region_id"]
        design = (frame.with_columns(
            pl.col("region_id").replace_strict(mapping).alias("region_idx"))
            .select(inputs).to_numpy().astype("float32", copy=False))
        for h in steps:
            per_horizon[h] = {kind: np.clip(boosters[(kind, h)].predict(design), 0, None)
                              for kind in TARGET_KINDS}
    else:
        for h in steps:
            per_horizon[h] = {
                kind: np.clip(frame[baseline.lag_column_for(source, season, h)]
                              .fill_null(0).to_numpy().astype("float64"), 0, None)
                for kind, source in TARGET_KINDS.items()
            }

    # -------------------------------------------------------------- inventory layer
    capacity = np.full(len(regions), np.nan)
    inventory = np.full(len(regions), np.nan)
    inventory_path = processed / f"region_daily_inventory_{tag}.parquet"
    estimated = inventory_path.exists()
    if estimated:
        day = (pl.DataFrame({"ts": [ts]})
               .with_columns(pl.col("ts").dt.convert_time_zone(tz).dt.date()
                             .alias("d"))["d"][0])
        inv = (pl.read_parquet(inventory_path)
               .filter(pl.col("day") == day)
               .select(["region_id", "capacity", "start_inventory"]))
        lookup = {row["region_id"]: row for row in inv.to_dicts()}
        for i, region in enumerate(regions):
            row = lookup.get(region)
            if row is not None:
                capacity[i] = row["capacity"]
                inventory[i] = row["start_inventory"]
        # no inventory row for this day: report forecasts and leave the operational
        # columns null rather than inventing a capacity
        estimated = not np.isnan(capacity).all()

    cum_pick = np.zeros(len(regions))
    cum_drop = np.zeros(len(regions))
    for h in steps:
        cum_pick = cum_pick + per_horizon[h]["pickup"]
        cum_drop = cum_drop + per_horizon[h]["dropoff"]

    projected = np.clip(inventory + cum_drop - cum_pick, 0, capacity)
    safety = safety_pct * capacity
    target = target_pct * capacity

    return pl.DataFrame({
        "region_id": regions,
        "forecast_time": [forecast_time] * len(regions),
        "horizon": [horizon] * len(regions),
        "predicted_pickups": per_horizon[horizon]["pickup"],
        "predicted_dropoffs": per_horizon[horizon]["dropoff"],
        "projected_inventory": projected,
        "shortage": np.clip(safety - projected, 0, None),
        "surplus": np.clip(projected - target, 0, None),
        "cumulative_predicted_pickups": cum_pick,
        "cumulative_predicted_dropoffs": cum_drop,
        "capacity": capacity,
        "model": [model_name] * len(regions),
        "spatial_system": [tag] * len(regions),
        "inventory_estimated": [estimated] * len(regions),
    }).select(OUTPUT_COLUMNS + ["cumulative_predicted_pickups",
                                "cumulative_predicted_dropoffs", "capacity",
                                "model", "spatial_system", "inventory_estimated"])
