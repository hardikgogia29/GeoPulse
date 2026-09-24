
from __future__ import annotations

import numpy as np
import polars as pl


def estimate_capacity(daily_flow: pl.DataFrame, quantile: float,
                      min_capacity: int, volume_deciles: int) -> pl.DataFrame:
    """Per-station capacity from the q95 daily cumulative-flow range.

    Stations with too few observed days get the median capacity of their trip-volume
    decile rather than a flat constant - a quiet station in a busy corridor is not
    the same size as a quiet station in the outer boroughs.
    """
    capacity = (
        daily_flow.group_by("station_id")
        .agg([
            pl.col("flow_range").quantile(quantile).alias("capacity_raw"),
            pl.col("flow_range").len().alias("observed_days"),
            pl.col("day_trips").sum().alias("total_trips"),
        ])
        .with_columns(
            pl.col("capacity_raw").ceil().cast(pl.Int32).clip(min_capacity, None)
            .alias("capacity")
        )
    )
    capacity = capacity.with_columns(
        pl.col("total_trips").qcut(volume_deciles, labels=[str(i) for i in range(volume_deciles)],
                                   allow_duplicates=True).alias("volume_decile")
    )
    decile_median = (
        capacity.filter(pl.col("observed_days") >= 30)
        .group_by("volume_decile")
        .agg(pl.col("capacity").median().alias("decile_capacity"))
    )
    return (
        capacity.join(decile_median, on="volume_decile", how="left")
        .with_columns(
            pl.when(pl.col("observed_days") >= 30)
            .then(pl.col("capacity"))
            .otherwise(pl.col("decile_capacity").fill_null(pl.col("capacity")))
            .cast(pl.Int32).alias("capacity"),
            pl.when(pl.col("observed_days") >= 30)
            .then(pl.lit("flow_q95"))
            .otherwise(pl.lit("volume_decile_median"))
            .alias("capacity_source"),
        )
        .drop("decile_capacity")
    )


def reconstruct_start_inventory(min_flow: np.ndarray, max_flow: np.ndarray,
                                capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feasible starting level per station-day, plus the infeasibility residual.

    Returns ``(start_inventory, fit_error)``. `fit_error` is 0 where a feasible
    interval exists and the size of the violation where it does not - which is the
    signature of a day staff actually rebalanced.
    """
    lower = np.maximum(0.0, -min_flow)
    upper = np.minimum(capacity, capacity - max_flow)
    target = capacity / 2.0

    feasible = lower <= upper
    start = np.where(feasible, np.clip(target, lower, upper), 0.0)
    fit_error = np.zeros_like(start)

    # infeasible days: pick the level minimising below-zero + overflow, breaking ties
    # toward half capacity. With a piecewise-linear objective the optimum is at one of
    # the two interval ends, so a closed-form comparison replaces an optimiser.
    if (~feasible).any():
        idx = ~feasible
        candidate_low = np.maximum(0.0, -min_flow[idx])
        candidate_high = np.minimum(capacity[idx], capacity[idx] - max_flow[idx])
        candidates = np.stack([candidate_low, candidate_high,
                               np.clip(target[idx], 0, capacity[idx])])
        below = np.maximum(0.0, -(candidates + min_flow[idx]))
        over = np.maximum(0.0, candidates + max_flow[idx] - capacity[idx])
        distance = np.abs(candidates - target[idx]) * 1e-3
        cost = below + over + distance
        best = np.argmin(cost, axis=0)
        chosen = np.take_along_axis(candidates, best[None, :], axis=0)[0]
        start[idx] = np.clip(chosen, 0, capacity[idx])
        fit_error[idx] = np.take_along_axis(below + over, best[None, :], axis=0)[0]
    return start, fit_error


def daily_flow_table(con, trips_view: str, tz: str) -> str:
    """SQL producing one row per (station, local day) with the cumulative-flow range."""
    return f"""
    WITH endpoints AS (
      SELECT start_station_id AS station_id, started_at AS ts, -1 AS delta
      FROM {trips_view} WHERE start_station_id IS NOT NULL
      UNION ALL
      SELECT end_station_id, ended_at, 1
      FROM {trips_view} WHERE end_station_id IS NOT NULL
    ),
    marked AS (
      SELECT station_id, (ts AT TIME ZONE '{tz}')::DATE AS day, ts, delta,
             sum(delta) OVER (
               PARTITION BY station_id, (ts AT TIME ZONE '{tz}')::DATE
               ORDER BY ts ROWS UNBOUNDED PRECEDING
             ) AS cumulative
      FROM endpoints
    )
    SELECT station_id, day,
           min(cumulative)::DOUBLE AS min_flow,
           max(cumulative)::DOUBLE AS max_flow,
           (max(cumulative) - min(cumulative))::DOUBLE AS flow_range,
           count(*) AS day_trips,
           last(cumulative ORDER BY ts)::DOUBLE AS end_flow
    FROM marked
    GROUP BY station_id, day
    """
