"""Tests for the Phase 8 standardized prediction interface.

The contract tests run everywhere; the end-to-end test skips itself when the trained
artifacts are not on disk, so a fresh clone still gets a green suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.serving.predict import (
    OUTPUT_COLUMNS, PredictionError, available_models, predict,
)
from src.utils.config import load_config, resolve_path

UTC = timezone.utc


def _tag(spatial="h3", resolution=8):
    from src.spatial.h3_indexer import make_indexer
    from src.utils.config import Config

    cfg = load_config(spatial, "lightgbm")
    key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
    cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: resolution,
                                     "resolution": resolution}})
    indexer = make_indexer(cfg)
    return cfg, f"{indexer.name}{indexer.resolution}"


# ------------------------------------------------------------------ contract tests
def test_forecast_time_must_be_timezone_aware():
    with pytest.raises(PredictionError, match="timezone-aware"):
        predict("h3", "lightgbm_final", datetime(2024, 11, 1, 12, 0), 1, resolution=8)


def test_forecast_time_must_sit_on_a_bin_boundary():
    """A 7-minute offset is a caller bug; rounding it silently would move the target."""
    off_grid = datetime(2024, 11, 1, 12, 7, tzinfo=UTC)
    with pytest.raises(PredictionError, match="bin boundary"):
        predict("h3", "lightgbm_final", off_grid, 1, resolution=8)


def test_unknown_horizon_is_rejected():
    when = datetime(2024, 11, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(PredictionError, match="not in trained horizons"):
        predict("h3", "lightgbm_final", when, 99, resolution=8)


def test_unknown_model_is_rejected_and_lists_what_exists():
    when = datetime(2024, 11, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(PredictionError, match="unknown model"):
        predict("h3", "no_such_model", when, 1, resolution=8)


def test_seasonal_naive_variant_is_validated():
    when = datetime(2024, 11, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(PredictionError, match="unknown seasonal variant"):
        predict("h3", "seasonal_naive_same_century", when, 1, resolution=8)


def test_available_models_always_offers_the_baselines():
    names = available_models("h3", 8)
    assert any(n.startswith("seasonal_naive_") for n in names)


# --------------------------------------------------------------- end-to-end smoke
def _feature_table_exists(cfg, tag, stem="features4"):
    return (resolve_path(cfg, "paths.processed") / f"{stem}_{tag}").exists()


@pytest.mark.parametrize("model", ["seasonal_naive_same_week", "lightgbm_final"])
def test_predict_returns_the_promised_schema(model):
    cfg, tag = _tag()
    stem = "features_" if model.startswith("seasonal") else "features4_"
    if not (resolve_path(cfg, "paths.processed") / f"{stem}{tag}").exists():
        pytest.skip(f"{stem}{tag} not built")
    if model not in available_models("h3", 8):
        pytest.skip(f"{model} not trained for {tag}")

    # a mid-morning weekday inside TEST, on the bin grid
    when = datetime(2024, 11, 6, 14, 0, tzinfo=UTC)
    frame = predict("h3", model, when, 4, resolution=8)

    for column in OUTPUT_COLUMNS:
        assert column in frame.columns, f"contract column {column} missing"
    assert frame.height > 0
    assert frame["horizon"].unique().to_list() == [4]
    assert (frame["predicted_pickups"] >= 0).all()
    assert (frame["predicted_dropoffs"] >= 0).all()
    assert frame["region_id"].n_unique() == frame.height, "one row per region"


def test_projection_and_flags_are_internally_consistent():
    """shortage and surplus must follow from projected_inventory, not float around it."""
    cfg, tag = _tag()
    if not _feature_table_exists(cfg, tag):
        pytest.skip(f"features4_{tag} not built")
    if "lightgbm_final" not in available_models("h3", 8):
        pytest.skip("lightgbm_final not trained")

    when = datetime(2024, 11, 6, 14, 0, tzinfo=UTC)
    frame = predict("h3", "lightgbm_final", when, 4, resolution=8,
                    safety_pct=0.10, target_pct=0.50)

    if not frame["inventory_estimated"][0]:
        # no inventory layer yet: the operational columns must be null, never zero,
        # so a caller cannot read "no shortage" out of "unknown"
        assert frame["projected_inventory"].is_null().all() or \
            frame["projected_inventory"].is_nan().all()
        pytest.skip("inventory layer not built - operational columns correctly null")

    work = frame.drop_nulls(["projected_inventory", "capacity"])
    assert work.height > 0
    expected_shortage = (0.10 * work["capacity"] - work["projected_inventory"]).clip(0)
    expected_surplus = (work["projected_inventory"] - 0.50 * work["capacity"]).clip(0)
    assert (work["shortage"] - expected_shortage).abs().max() < 1e-6
    assert (work["surplus"] - expected_surplus).abs().max() < 1e-6
    # a region cannot be simultaneously short and in surplus
    assert ((work["shortage"] > 0) & (work["surplus"] > 0)).sum() == 0
    # projection is physically bounded
    assert (work["projected_inventory"] >= 0).all()
    assert (work["projected_inventory"] <= work["capacity"]).all()


def test_cumulative_horizons_accumulate():
    """Horizon 4's cumulative demand must be >= horizon 1's - it contains it."""
    cfg, tag = _tag()
    if not _feature_table_exists(cfg, tag) or "lightgbm_final" not in available_models("h3", 8):
        pytest.skip("artifacts not built")
    when = datetime(2024, 11, 6, 14, 0, tzinfo=UTC)
    h1 = predict("h3", "lightgbm_final", when, 1, resolution=8).sort("region_id")
    h4 = predict("h3", "lightgbm_final", when, 4, resolution=8).sort("region_id")
    assert (h4["cumulative_predicted_pickups"]
            >= h1["cumulative_predicted_pickups"] - 1e-9).all()
