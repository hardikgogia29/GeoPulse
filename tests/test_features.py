"""Leakage tests for the Phase 3 feature table.

The rule: **every feature at row `(r, t)` must be computable from information
available at `t`**. Two families need checking, and they fail in different ways.

* **Lags** must look strictly backwards. They are produced with SQL `LAG` over a
  dense grid; here every one is re-derived by an independent timestamp join, so a
  sign error, an off-by-one, or a density bug cannot slip through.
* **Calendar** features describe `t` itself, which is known ahead of time - not
  leakage - but they must actually describe `t` in *local* time, and the rush-hour
  and holiday flags must match the configured windows rather than drifting.

A separate test asserts the split boundaries never overlap, because a leaky split
would flatter every model in every later phase.
"""

from datetime import date, timedelta

import holidays
import polars as pl
import pytest

from src.features.basic import CALENDAR_COLUMNS, CYCLICAL_COLUMNS, lag_columns
from src.models.splits import load_splits, validate_splits


@pytest.fixture(scope="module")
def cfg_h3():
    from src.utils.config import load_config

    return load_config("h3")


@pytest.fixture(scope="module")
def features(cfg_h3):
    from src.utils.config import resolve_path

    path = resolve_path(cfg_h3, "paths.dev_sample") / "features_h39.parquet"
    if not path.exists():
        pytest.skip("dev features not built - run scripts/12_build_features.py --dev-sample")
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def panel(cfg_h3):
    from src.utils.config import resolve_path

    path = resolve_path(cfg_h3, "paths.dev_sample") / "panel_h39.parquet"
    if not path.exists():
        pytest.skip("dev panel not built")
    return pl.read_parquet(path, columns=["region_id", "ts", "pickups", "dropoffs"])


# --------------------------------------------------------------------------- lags


@pytest.mark.parametrize("target", ["pickups", "dropoffs"])
def test_every_lag_matches_an_independent_timestamp_join(features, panel, cfg_h3, target):
    """`{target}_lag_k[r,t]` must equal `{target}[r, t - k*interval]`, joined by time."""
    interval = cfg_h3.dotted("time.interval_minutes")
    for lag in cfg_h3.dotted("features.basic_lags"):
        column = f"{target}_lag_{lag}"
        shifted = panel.select(
            [
                pl.col("region_id"),
                (pl.col("ts") + pl.duration(minutes=interval * lag)).alias("ts"),
                pl.col(target).alias("expected"),
            ]
        )
        joined = features.select(["region_id", "ts", column]).join(
            shifted, on=["region_id", "ts"], how="left"
        )
        # after the warm-up drop every row must have a real history value
        assert joined["expected"].null_count() == 0, f"{column}: history missing"
        mismatched = joined.filter(pl.col(column) != pl.col("expected"))
        assert mismatched.height == 0, f"{column}: {mismatched.height} rows differ"


def test_lags_are_backwards_not_forwards(features, panel, cfg_h3):
    """A sign error would make `lag_k` equal the value k steps AHEAD. Rule it out."""
    interval = cfg_h3.dotted("time.interval_minutes")
    lag = cfg_h3.dotted("features.basic_lags")[2]
    lead = panel.select(
        [
            pl.col("region_id"),
            (pl.col("ts") - pl.duration(minutes=interval * lag)).alias("ts"),
            pl.col("pickups").alias("future_value"),
        ]
    )
    joined = features.select(["region_id", "ts", f"pickups_lag_{lag}"]).join(
        lead, on=["region_id", "ts"], how="inner"
    )
    equal_to_future = joined.filter(
        pl.col(f"pickups_lag_{lag}") == pl.col("future_value")
    ).height
    # they coincide whenever both are 0, which is common in a sparse panel - so
    # require that they are not *systematically* identical
    assert equal_to_future < joined.height, "lag column equals the future value everywhere"


def test_no_lag_column_is_a_disguised_target(features, cfg_h3):
    """A lag must never equal the current bin's demand for the same target."""
    for target in cfg_h3.dotted("features.lag_targets"):
        for lag in cfg_h3.dotted("features.basic_lags"):
            identical = (features[f"{target}_lag_{lag}"] == features[target]).all()
            assert not identical, f"{target}_lag_{lag} is identical to {target}"


def test_lags_have_no_nulls_after_warmup(features, cfg_h3):
    for column in lag_columns(cfg_h3):
        assert features[column].null_count() == 0, column


# ----------------------------------------------------------------------- calendar


def test_calendar_matches_an_independent_local_time_computation(features, cfg_h3):
    tz = cfg_h3.dotted("time.timezone")
    sample = features.select(["ts", *CALENDAR_COLUMNS]).head(50_000)
    local = sample.select(pl.col("ts").dt.convert_time_zone(tz).alias("local"))["local"]
    assert (sample["hour"] == local.dt.hour()).all()
    assert (sample["minute"] == local.dt.minute()).all()
    assert (sample["day_of_week"] == local.dt.weekday()).all()  # polars: Mon=1..Sun=7
    assert (sample["day_of_month"] == local.dt.day()).all()
    assert (sample["month_of_year"] == local.dt.month()).all()
    assert (sample["is_weekend"] == (local.dt.weekday() >= 6)).all()


def test_rush_hour_flags_follow_the_configured_windows(features, cfg_h3):
    morning = cfg_h3.dotted("calendar.morning_rush")
    evening = cfg_h3.dotted("calendar.evening_rush")
    expected_morning = (features["hour"] >= morning["start_hour"]) & (
        features["hour"] < morning["end_hour"]
    )
    expected_evening = (features["hour"] >= evening["start_hour"]) & (
        features["hour"] < evening["end_hour"]
    )
    assert (features["is_morning_rush"] == expected_morning).all()
    assert (features["is_evening_rush"] == expected_evening).all()
    # and the two windows must not overlap
    assert not (features["is_morning_rush"] & features["is_evening_rush"]).any()


def test_holiday_flag_matches_the_holidays_library(features, cfg_h3):
    tz = cfg_h3.dotted("time.timezone")
    calendar = holidays.country_holidays(
        cfg_h3.dotted("calendar.holiday_country"),
        subdiv=cfg_h3.dotted("calendar.holiday_subdiv"),
        years=range(
            date.fromisoformat(cfg_h3.dotted("time.start_date")).year,
            date.fromisoformat(cfg_h3.dotted("time.end_date")).year + 1,
        ),
    )
    local_dates = (
        features.select(pl.col("ts").dt.convert_time_zone(tz).dt.date().alias("d"))["d"]
        .unique()
        .to_list()
    )
    flags = (
        features.select(
            [pl.col("ts").dt.convert_time_zone(tz).dt.date().alias("d"), pl.col("is_holiday")]
        )
        .group_by("d")
        .agg(pl.col("is_holiday").max())
    )
    lookup = dict(zip(flags["d"].to_list(), flags["is_holiday"].to_list()))
    for day in local_dates:
        assert lookup[day] == (day in calendar), day


def test_business_day_is_weekday_and_not_holiday(features):
    expected = (~features["is_weekend"]) & (~features["is_holiday"])
    assert (features["is_business_day"] == expected).all()


def test_cyclical_encodings_are_on_the_unit_circle(features):
    for sin_col, cos_col in [
        ("tod_sin", "tod_cos"), ("dow_sin", "dow_cos"), ("doy_sin", "doy_cos")
    ]:
        radius = (features[sin_col] ** 2 + features[cos_col] ** 2).to_numpy()
        assert abs(radius.min() - 1.0) < 1e-4, sin_col
        assert abs(radius.max() - 1.0) < 1e-4, sin_col
    for column in CYCLICAL_COLUMNS:
        assert features[column].null_count() == 0


def test_midnight_and_noon_map_to_the_expected_phase(features, cfg_h3):
    """Sanity-check the time-of-day encoding rather than trusting the formula."""
    tz = cfg_h3.dotted("time.timezone")
    tagged = features.select(
        [
            pl.col("ts").dt.convert_time_zone(tz).dt.hour().alias("h"),
            pl.col("ts").dt.convert_time_zone(tz).dt.minute().alias("m"),
            "tod_sin", "tod_cos",
        ]
    )
    midnight = tagged.filter((pl.col("h") == 0) & (pl.col("m") == 0))
    if midnight.height:
        assert abs(midnight["tod_sin"][0]) < 1e-5
        assert abs(midnight["tod_cos"][0] - 1.0) < 1e-5
    noon = tagged.filter((pl.col("h") == 12) & (pl.col("m") == 0))
    if noon.height:
        assert abs(noon["tod_sin"][0]) < 1e-5
        assert abs(noon["tod_cos"][0] + 1.0) < 1e-5


# ------------------------------------------------------------------------- splits


def test_splits_are_valid(cfg_h3):
    assert validate_splits(cfg_h3) == []


def test_splits_do_not_overlap_and_cover_contiguously(cfg_h3):
    splits = load_splits(cfg_h3)
    train, validate, test = splits["train"], splits["validate"], splits["test"]
    assert train.end < validate.start < validate.end < test.start <= test.end
    assert validate.start - train.end == timedelta(days=1)
    assert test.start - validate.end == timedelta(days=1)


def test_split_predicates_partition_the_rows(features, cfg_h3):
    """Every row lands in at most one split, and the predicates are mutually exclusive."""
    import duckdb

    splits = load_splits(cfg_h3)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("f", features.select("ts"))
    counts = {
        name: con.execute(f"SELECT count(*) FROM f WHERE {split.sql(cfg_h3)}").fetchone()[0]
        for name, split in splits.items()
    }
    overlap = con.execute(
        f"SELECT count(*) FROM f WHERE ({splits['train'].sql(cfg_h3)}) "
        f"AND ({splits['validate'].sql(cfg_h3)})"
    ).fetchone()[0]
    assert overlap == 0
    assert sum(counts.values()) <= features.height


def test_targets_are_not_present_among_the_feature_inputs(cfg_h3):
    """The model input list must never contain a target column."""
    from src.features.basic import feature_columns, target_columns

    assert not set(feature_columns(cfg_h3)) & set(target_columns(cfg_h3))
    for column in feature_columns(cfg_h3):
        assert not column.startswith(("pickup_h", "dropoff_h")), column
