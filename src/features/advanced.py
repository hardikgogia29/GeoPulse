
from __future__ import annotations

import polars as pl

# family key -> human label, in the cumulative order the ablation adds them
FAMILY_ORDER = [
    ("A", "demand_recent"),
    ("B", "calendar_seasonal"),
    ("C", "rolling_trend"),
    ("D", "weather"),
    ("E", "events"),
    ("F", "traffic"),
    ("G", "spatial_neighbor"),
    ("H", "station_network"),
]

TARGETS = ["pickups", "dropoffs"]


# --------------------------------------------------------------------- family A
def demand_recent_exprs(cfg) -> list[pl.Expr]:
    """Lag 0 (the just-completed bin) plus the plain lags from Phase 3."""
    exprs: list[pl.Expr] = []
    for target in TARGETS:
        # lag 0: the current bin is complete at forecast_time, so this is the single
        # most informative legal input. Phase 3 omitted it and was poorer for it.
        exprs.append(pl.col(target).alias(f"{target}_lag_0"))
        for lag in cfg.dotted("features.basic_lags"):
            exprs.append(pl.col(target).shift(lag).over("region_id").alias(f"{target}_lag_{lag}"))
    return exprs


def demand_recent_columns(cfg) -> list[str]:
    return [
        f"{t}_lag_{lag}"
        for t in TARGETS
        for lag in [0, *cfg.dotted("features.basic_lags")]
    ]


# --------------------------------------------------------------------- family B
def seasonal_exprs(cfg) -> list[pl.Expr]:
    """Horizon-aligned seasonal slots + expanding per-slot expectations.

    The seasonal value for `t + h` is the demand one season before *that* instant,
    so the shift is `season - h`, not `season` - the same alignment the Seasonal
    Naive baseline uses.
    """
    exprs: list[pl.Expr] = []
    horizons = cfg.dotted("time.horizons")
    for name, season in cfg.dotted("advanced_features.seasonal_slot_lags").items():
        for h in horizons:
            offset = season - h
            for target in TARGETS:
                exprs.append(
                    pl.col(target).shift(offset).over("region_id")
                    .alias(f"{target}_seasonal_{name}_h{h}")
                )

    # expanding mean for this region at this weekday+slot, PAST days only.
    # shift(1) inside the group is what excludes the current row.
    key = ["region_id", "day_of_week", "slot_of_day"]
    min_periods = cfg.dotted("advanced_features.expanding_min_periods")
    for target in TARGETS:
        history = pl.col(target).shift(1)
        count = history.cum_count().over(key)
        mean = (history.cum_sum().over(key) / count)
        exprs.append(
            pl.when(count >= min_periods).then(mean).otherwise(None)
            .alias(f"{target}_slot_expanding_mean")
        )
    return exprs


def target_calendar_exprs(cfg) -> list[pl.Expr]:
    """Calendar of the TARGET time, which is known in advance - not leakage."""
    exprs: list[pl.Expr] = []
    interval = cfg.dotted("time.interval_minutes")
    tz = cfg.dotted("time.timezone")
    morning = cfg.dotted("calendar.morning_rush")
    evening = cfg.dotted("calendar.evening_rush")
    for h in cfg.dotted("time.horizons"):
        local = (pl.col("ts") + pl.duration(minutes=interval * h)).dt.convert_time_zone(tz)
        minutes = local.dt.hour() * 60 + local.dt.minute()
        exprs += [
            (2 * 3.141592653589793 * minutes / 1440.0).sin().cast(pl.Float32)
            .alias(f"target_tod_sin_h{h}"),
            (2 * 3.141592653589793 * minutes / 1440.0).cos().cast(pl.Float32)
            .alias(f"target_tod_cos_h{h}"),
            local.dt.weekday().cast(pl.Int8).alias(f"target_weekday_h{h}"),
            local.dt.hour().cast(pl.Int8).alias(f"target_hour_h{h}"),
            (
                ((local.dt.hour() >= morning["start_hour"]) & (local.dt.hour() < morning["end_hour"]))
                | ((local.dt.hour() >= evening["start_hour"]) & (local.dt.hour() < evening["end_hour"]))
            ).alias(f"target_is_rush_h{h}"),
        ]
    return exprs


def calendar_seasonal_columns(cfg) -> list[str]:
    from src.features.basic import CALENDAR_COLUMNS, CYCLICAL_COLUMNS

    columns = [*CALENDAR_COLUMNS, *CYCLICAL_COLUMNS]
    for name, season in cfg.dotted("advanced_features.seasonal_slot_lags").items():
        for h in cfg.dotted("time.horizons"):
            columns += [f"{t}_seasonal_{name}_h{h}" for t in TARGETS]
    columns += [f"{t}_slot_expanding_mean" for t in TARGETS]
    for h in cfg.dotted("time.horizons"):
        columns += [f"target_tod_sin_h{h}", f"target_tod_cos_h{h}",
                    f"target_weekday_h{h}", f"target_hour_h{h}", f"target_is_rush_h{h}"]
    return columns


# --------------------------------------------------------------------- family C
def rolling_trend_exprs(cfg) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    windows = cfg.dotted("advanced_features.rolling_windows")
    stats = cfg.dotted("advanced_features.rolling_stats")
    for target in TARGETS:
        for window in windows:
            if "mean" in stats:
                exprs.append(pl.col(target).rolling_mean(window, min_samples=1)
                             .over("region_id").cast(pl.Float32)
                             .alias(f"{target}_roll_mean_{window}"))
            if "std" in stats:
                exprs.append(pl.col(target).rolling_std(window, min_samples=2)
                             .over("region_id").cast(pl.Float32)
                             .alias(f"{target}_roll_std_{window}"))
            if "sum" in stats:
                exprs.append(pl.col(target).rolling_sum(window, min_samples=1)
                             .over("region_id").cast(pl.Int32)
                             .alias(f"{target}_roll_sum_{window}"))
        for span in cfg.dotted("advanced_features.ewm_spans"):
            exprs.append(pl.col(target).ewm_mean(span=span, ignore_nulls=True)
                         .over("region_id").cast(pl.Float32)
                         .alias(f"{target}_ewm_{span}"))
        for step in cfg.dotted("advanced_features.trend_steps"):
            exprs.append((pl.col(target) - pl.col(target).shift(step).over("region_id"))
                         .cast(pl.Int32).alias(f"{target}_delta_{step}"))
        # recent hour vs the hour before it
        exprs.append(
            (pl.col(target).rolling_mean(4, min_samples=1).over("region_id")
             - pl.col(target).shift(4).rolling_mean(4, min_samples=1).over("region_id"))
            .cast(pl.Float32).alias(f"{target}_hour_over_hour")
        )

    # net flow / inventory pressure
    for window in cfg.dotted("advanced_features.netflow_windows"):
        exprs.append(pl.col("net_flow").rolling_sum(window, min_samples=1)
                     .over("region_id").cast(pl.Int32).alias(f"net_flow_roll_sum_{window}"))
    exprs.append(pl.col("net_flow").alias("net_flow_lag_0"))
    exprs.append(pl.col("net_flow").shift(1).over("region_id").alias("net_flow_lag_1"))
    exprs.append(
        ((pl.col("dropoffs").rolling_sum(4, min_samples=1).over("region_id") + 1)
         / (pl.col("pickups").rolling_sum(4, min_samples=1).over("region_id") + 1))
        .cast(pl.Float32).alias("dropoff_pickup_ratio_1h")
    )
    return exprs


def rolling_trend_columns(cfg) -> list[str]:
    columns: list[str] = []
    windows = cfg.dotted("advanced_features.rolling_windows")
    stats = cfg.dotted("advanced_features.rolling_stats")
    for target in TARGETS:
        for window in windows:
            columns += [f"{target}_roll_{stat}_{window}" for stat in stats]
        columns += [f"{target}_ewm_{span}" for span in cfg.dotted("advanced_features.ewm_spans")]
        columns += [f"{target}_delta_{step}" for step in cfg.dotted("advanced_features.trend_steps")]
        columns.append(f"{target}_hour_over_hour")
    columns += [f"net_flow_roll_sum_{w}" for w in cfg.dotted("advanced_features.netflow_windows")]
    columns += ["net_flow_lag_0", "net_flow_lag_1", "dropoff_pickup_ratio_1h"]
    return columns


# --------------------------------------------------------------------- family D
WEATHER_BASE = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
    "rain", "snowfall", "snow_depth", "wind_speed_10m", "wind_gusts_10m", "cloud_cover",
]


def weather_derived_exprs(cfg) -> list[pl.Expr]:
    thresholds = cfg.dotted("weather")
    return [
        (pl.col("rain") > thresholds["rain_mm_threshold"]).alias("is_raining"),
        (pl.col("snowfall") > thresholds["snow_cm_threshold"]).alias("is_snowing"),
        (pl.col("temperature_2m") < thresholds["freezing_temp_c"]).alias("is_freezing"),
        (pl.col("wind_speed_10m") > thresholds["high_wind_kmh"]).alias("high_wind"),
        (pl.col("temperature_2m") - pl.col("temperature_2m").shift(4).over("region_id"))
        .cast(pl.Float32).alias("temp_change_1h"),
        (pl.col("temperature_2m") - pl.col("temperature_2m").shift(12).over("region_id"))
        .cast(pl.Float32).alias("temp_change_3h"),
    ]


def weather_columns() -> list[str]:
    return [
        *WEATHER_BASE, "is_raining", "is_snowing", "is_freezing", "high_wind",
        "temp_change_1h", "temp_change_3h", "weather_missing",
    ]


# ----------------------------------------------------------- families E / F / G / H
EVENT_COLUMNS = [
    "event_present", "event_count", "event_count_ring", "event_started_recently",
    "event_ending_soon", "event_attendee_types",
]
TRAFFIC_COLUMNS = [
    "traffic_speed_mean", "traffic_speed_min", "traffic_link_count",
    "congestion_index", "travel_time_index", "traffic_change_15m", "traffic_missing",
]
STATION_COLUMNS = [
    "active_station_count", "station_density_km2", "region_hist_trip_volume",
    "region_area_km2",
]


def neighbor_columns(cfg) -> list[str]:
    return [
        f"nb_{target}_{stat}"
        for target in TARGETS
        for stat in cfg.dotted("advanced_features.neighbor_stats")
    ] + [f"{target}_vs_neighbor_mean" for target in TARGETS] + ["neighbor_count"]


def family_columns(cfg) -> dict[str, list[str]]:
    """Column list per ablation family, in cumulative A-H order."""
    return {
        "A": demand_recent_columns(cfg),
        "B": calendar_seasonal_columns(cfg),
        "C": rolling_trend_columns(cfg),
        "D": weather_columns(),
        "E": EVENT_COLUMNS,
        "F": TRAFFIC_COLUMNS,
        "G": neighbor_columns(cfg),
        "H": STATION_COLUMNS,
    }


def cumulative_feature_sets(cfg) -> dict[str, list[str]]:
    """A, A+B, A+B+C, ... - what the ablation actually trains on."""
    per_family = family_columns(cfg)
    sets: dict[str, list[str]] = {}
    running: list[str] = ["region_id"]
    for key, _ in FAMILY_ORDER:
        running = running + per_family[key]
        sets[key] = list(running)
    return sets
