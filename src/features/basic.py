
from __future__ import annotations

from datetime import date

import duckdb
import holidays

CALENDAR_COLUMNS = [
    "hour", "minute", "day_of_week", "day_of_month", "week_of_year", "month_of_year",
    "is_weekend", "is_holiday", "is_business_day", "is_morning_rush", "is_evening_rush",
]
CYCLICAL_COLUMNS = [
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]


def create_holiday_table(con: duckdb.DuckDBPyConnection, cfg) -> int:
    """Materialise public holidays for the project window (Python `holidays`)."""
    start = date.fromisoformat(cfg.dotted("time.start_date"))
    end = date.fromisoformat(cfg.dotted("time.end_date"))
    calendar = holidays.country_holidays(
        cfg.dotted("calendar.holiday_country"),
        subdiv=cfg.dotted("calendar.holiday_subdiv"),
        years=range(start.year, end.year + 1),
    )
    days = sorted(d for d in calendar if start <= d <= end)
    con.execute("CREATE OR REPLACE TABLE holidays_tbl (holiday_date DATE, holiday_name VARCHAR)")
    con.executemany(
        "INSERT INTO holidays_tbl VALUES (?, ?)",
        [(d, calendar.get(d)) for d in days],
    )
    return len(days)


def calendar_sql(cfg, ts_col: str = "ts") -> str:
    """Calendar + cyclical features derived from the LOCAL wall clock at `ts`."""
    tz = cfg.dotted("time.timezone")
    morning = cfg.dotted("calendar.morning_rush")
    evening = cfg.dotted("calendar.evening_rush")
    local = f"({ts_col} AT TIME ZONE '{tz}')"
    minutes_of_day = f"(extract('hour' FROM {local}) * 60 + extract('minute' FROM {local}))"
    return f"""
        extract('hour'    FROM {local})::TINYINT   AS hour,
        extract('minute'  FROM {local})::TINYINT   AS minute,
        extract('isodow'  FROM {local})::TINYINT   AS day_of_week,
        extract('day'     FROM {local})::TINYINT   AS day_of_month,
        extract('week'    FROM {local})::TINYINT   AS week_of_year,
        -- named month_of_year, not month: the panel is hive-partitioned on a
        -- `month` column and the two would collide on read
        extract('month'   FROM {local})::TINYINT   AS month_of_year,
        (extract('isodow' FROM {local}) >= 6)      AS is_weekend,
        (h.holiday_date IS NOT NULL)               AS is_holiday,
        (extract('isodow' FROM {local}) < 6
         AND h.holiday_date IS NULL)               AS is_business_day,
        (extract('hour' FROM {local}) >= {morning['start_hour']}
         AND extract('hour' FROM {local}) < {morning['end_hour']})  AS is_morning_rush,
        (extract('hour' FROM {local}) >= {evening['start_hour']}
         AND extract('hour' FROM {local}) < {evening['end_hour']})  AS is_evening_rush,
        sin(2 * pi() * {minutes_of_day} / 1440.0)::FLOAT  AS tod_sin,
        cos(2 * pi() * {minutes_of_day} / 1440.0)::FLOAT  AS tod_cos,
        sin(2 * pi() * extract('isodow' FROM {local}) / 7.0)::FLOAT   AS dow_sin,
        cos(2 * pi() * extract('isodow' FROM {local}) / 7.0)::FLOAT   AS dow_cos,
        sin(2 * pi() * extract('dayofyear' FROM {local}) / 365.25)::FLOAT AS doy_sin,
        cos(2 * pi() * extract('dayofyear' FROM {local}) / 365.25)::FLOAT AS doy_cos
    """


def lag_columns(cfg) -> list[str]:
    """Lag columns offered to the MODEL (Phase 3 basic lags only)."""
    return [
        f"{target}_lag_{lag}"
        for target in cfg.dotted("features.lag_targets")
        for lag in cfg.dotted("features.basic_lags")
    ]


def baseline_lag_columns(cfg) -> list[str]:
    """Extra lags the Seasonal Naive baseline needs, which the model does not use.

    The naive forecast for `t + h` is the value one season before *that* instant, so
    it needs `lag_{season - h}` (95, not 96, for one-day seasonality at h=1). Scoring
    the baseline off `lag_96` instead would misalign it by up to an hour and hand
    LightGBM an unearned win - the opposite of what a gate is for. These columns are
    materialised but deliberately kept OUT of `feature_columns`.
    """
    from src.models.baseline import required_lags

    extra = sorted(required_lags(cfg) - set(cfg.dotted("features.basic_lags")))
    return [
        f"{target}_lag_{lag}"
        for target in cfg.dotted("features.lag_targets")
        for lag in extra
    ]


def all_lag_columns(cfg) -> list[str]:
    return lag_columns(cfg) + baseline_lag_columns(cfg)


def max_materialised_lag(cfg) -> int:
    """Longest lag actually built - drives how many warm-up rows must be dropped."""
    return max(int(column.rpartition("_lag_")[2]) for column in all_lag_columns(cfg))


def lag_sql(cfg) -> str:
    """Strictly backward-looking lags over the dense per-region series."""
    parts = []
    for column in all_lag_columns(cfg):
        target, _, lag = column.rpartition("_lag_")
        parts.append(f"lag({target}, {lag}) OVER w AS {column}")
    return ",\n        ".join(parts)


def feature_columns(cfg) -> list[str]:
    """Model input columns, in a stable order (region_id is a categorical)."""
    return ["region_id", *CALENDAR_COLUMNS, *CYCLICAL_COLUMNS, *lag_columns(cfg)]


def target_columns(cfg) -> list[str]:
    return [
        f"{kind}_h{h}"
        for h in cfg.dotted("time.horizons")
        for kind in ("pickup", "dropoff")
    ]


def build_features_sql(cfg, panel_view: str, region_filter: str | None = None) -> str:
    """Panel -> feature table. `ts` is the forecast time; nothing reads past it.

    `region_filter` restricts the pass to a batch of regions. Lags are computed
    `PARTITION BY region_id`, so they never cross a region boundary and batching is
    **exact**, not an approximation - it just bounds how much has to be materialised
    at once. A single pass over 104M rows with 34 lag windows plus a global sort
    spills tens of GB; batching keeps each pass in memory and makes the higher
    resolutions of the Phase 5 sweep feasible at all.
    """
    max_lag = max_materialised_lag(cfg)
    targets = ", ".join(target_columns(cfg))
    where = f"WHERE {region_filter}" if region_filter else ""
    return f"""
    WITH lagged AS (
      SELECT region_id, ts, pickups, dropoffs, net_flow, dst_unreliable,
             {targets},
             {lag_sql(cfg)},
             row_number() OVER w AS _step
      FROM {panel_view}
      {where}
      WINDOW w AS (PARTITION BY region_id ORDER BY ts)
    )
    SELECT l.* EXCLUDE (_step),
           {calendar_sql(cfg, "l.ts")}
    FROM lagged l
    LEFT JOIN holidays_tbl h
      ON h.holiday_date = (l.ts AT TIME ZONE '{cfg.dotted("time.timezone")}')::DATE
    WHERE l._step > {max_lag}
    """
