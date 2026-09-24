
from __future__ import annotations

import numpy as np


def seasonal_lag_steps(cfg) -> dict[str, int]:
    return dict(cfg.dotted("seasonal_naive.seasonal_naive_lags"))


def required_lags(cfg) -> set[int]:
    """Lag columns the seasonal-naive baseline needs, given the horizons."""
    horizons = cfg.dotted("time.horizons")
    return {
        season - h
        for season in seasonal_lag_steps(cfg).values()
        for h in horizons
        if season - h > 0
    }


def lag_column_for(target: str, season_steps: int, horizon: int) -> str:
    """The lag column holding demand at the same slot one season before `t + h`."""
    offset = season_steps - horizon
    if offset <= 0:
        raise ValueError(f"season {season_steps} must exceed horizon {horizon}")
    return f"{target}_lag_{offset}"


def predict(frame, target: str, season_steps: int, horizon: int) -> np.ndarray:
    """Seasonal-naive prediction as a float array, clipped at zero."""
    column = lag_column_for(target, season_steps, horizon)
    return np.clip(frame[column].to_numpy().astype("float64"), 0, None)
