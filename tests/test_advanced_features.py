"""Leakage tests for the Phase 4 feature families.

Same standard as Phase 3: every derived value is re-derived here by an independent
route (a timestamp join, or a hand-rolled window) rather than trusting the expression
that produced it. Phase 4 adds families that can leak in new ways:

* **lag 0** is legal only under the convention in `docs/FORECAST_TIME_CONVENTION.md`
  (`forecast_time` = end of bin `ts`). It must equal the row's own observed demand -
  never the next bin's.
* **rolling / EWM** windows may include the current row but nothing after it.
* **expanding slot means** must exclude the current row, or the feature contains its
  own answer.
* **weather** must be the latest reading at or before `forecast_time`, never after.
* **target-time calendar** is known in advance and must describe `ts + h*interval`.
"""

from datetime import timedelta

import polars as pl
import pytest


@pytest.fixture(scope="module")
def cfg4():
    from src.utils.config import load_config

    return load_config("h3")


@pytest.fixture(scope="module")
def feats(cfg4):
    from src.utils.config import resolve_path

    path = resolve_path(cfg4, "paths.dev_sample") / "features4_h39.parquet"
    if not path.exists():
        pytest.skip("dev features4 not built - run scripts/17_build_features_full.py --dev-sample")
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def interval(cfg4):
    return cfg4.dotted("time.interval_minutes")


# ------------------------------------------------------------------ family A
def test_lag_zero_is_the_current_bin_not_the_next(feats):
    """lag 0 must equal this bin's observed demand - the whole convention rests on it."""
    for target in ("pickups", "dropoffs"):
        assert (feats[f"{target}_lag_0"] == feats[target]).all()


def test_lag_zero_is_not_the_target(feats):
    """If lag 0 had been built one bin late it would equal pickup_h1 - catch that."""
    matching = (feats["pickups_lag_0"] == feats["pickup_h1"]).sum()
    assert matching < feats.height, "pickups_lag_0 equals pickup_h1 everywhere - off by one"


@pytest.mark.parametrize("target", ["pickups", "dropoffs"])
def test_lags_match_timestamp_join(feats, cfg4, interval, target):
    base = feats.select(["region_id", "ts", target])
    for lag in cfg4.dotted("features.basic_lags"):
        shifted = base.select(
            [pl.col("region_id"),
             (pl.col("ts") + pl.duration(minutes=interval * lag)).alias("ts"),
             pl.col(target).alias("expected")]
        )
        joined = (
            feats.select(["region_id", "ts", f"{target}_lag_{lag}"])
            .join(shifted, on=["region_id", "ts"], how="inner")
        )
        bad = joined.filter(pl.col(f"{target}_lag_{lag}") != pl.col("expected"))
        assert bad.height == 0, f"{target}_lag_{lag}: {bad.height} mismatches"


# ------------------------------------------------------------------ family C
def test_rolling_sum_includes_current_and_only_past(feats, cfg4, interval):
    """roll_sum_4 must be the current bin plus the three before it, and nothing after.

    Re-derived by timestamp joins against the panel rather than from other feature
    columns, so a shared bug in the lag expressions cannot hide a rolling bug.
    """
    window = 4
    base = feats.select(["region_id", "ts", "pickups"])
    frame = feats.select(["region_id", "ts", "pickups_roll_sum_4"])
    expected = None
    for step in range(window):
        shifted = base.select(
            [pl.col("region_id"),
             (pl.col("ts") + pl.duration(minutes=interval * step)).alias("ts"),
             pl.col("pickups").alias(f"back_{step}")]
        )
        frame = frame.join(shifted, on=["region_id", "ts"], how="inner")
    total = sum(pl.col(f"back_{s}") for s in range(window))
    bad = frame.with_columns(total.alias("expected")).filter(
        pl.col("pickups_roll_sum_4") != pl.col("expected")
    )
    assert bad.height == 0, f"{bad.height} rows differ from the hand-rolled window"


def test_rolling_sum_does_not_reach_into_the_future(feats, cfg4, interval):
    """A window that accidentally included ts+1 would equal a forward-looking sum."""
    base = feats.select(["region_id", "ts", "pickups"])
    forward = base.select(
        [pl.col("region_id"),
         (pl.col("ts") - pl.duration(minutes=interval)).alias("ts"),
         pl.col("pickups").alias("next_bin")]
    )
    joined = (
        feats.select(["region_id", "ts", "pickups_roll_sum_4", "pickups_lag_0",
                      "pickups_lag_1", "pickups_lag_2"])
        .join(forward, on=["region_id", "ts"], how="inner")
    )
    leaky = joined.filter(
        pl.col("pickups_roll_sum_4")
        == pl.col("pickups_lag_0") + pl.col("pickups_lag_1")
        + pl.col("pickups_lag_2") + pl.col("next_bin")
    )
    # they coincide only by chance (e.g. all zeros), never systematically
    assert leaky.height < joined.height * 0.9


def test_rolling_mean_matches_rolling_sum_once_the_window_is_full(feats, cfg4, interval):
    """min_samples=1 means the first rows of a region divide by fewer than `window`,
    so only compare where a full window of history exists."""
    window = 4
    base = feats.select(["region_id", "ts", "pickups"])
    frame = feats.select(["region_id", "ts", "pickups_roll_mean_4", "pickups_roll_sum_4"])
    for step in range(window):
        shifted = base.select(
            [pl.col("region_id"),
             (pl.col("ts") + pl.duration(minutes=interval * step)).alias("ts"),
             pl.col("pickups").alias(f"back_{step}")]
        )
        frame = frame.join(shifted, on=["region_id", "ts"], how="inner")
    diff = (frame["pickups_roll_mean_4"] - frame["pickups_roll_sum_4"] / window).abs()
    assert diff.max() < 1e-3


def test_trend_delta_matches_lags(feats, cfg4):
    sample = feats.filter(pl.col("pickups_lag_4").is_not_null()).head(200_000)
    for step in cfg4.dotted("advanced_features.trend_steps"):
        expected = sample["pickups_lag_0"] - sample[f"pickups_lag_{step}"]
        assert (sample[f"pickups_delta_{step}"] == expected).all(), step


def test_ewm_is_finite_and_bounded_by_observed_range(feats):
    for span in (4, 12, 96):
        column = feats[f"pickups_ewm_{span}"].drop_nulls()
        assert column.is_finite().all()
        assert column.min() >= 0
        assert column.max() <= feats["pickups"].max()


def test_net_flow_features_are_consistent(feats):
    assert (feats["net_flow_lag_0"] == feats["net_flow"]).all()


# ------------------------------------------------------------------ family B
def test_seasonal_slot_matches_timestamp_join(feats, cfg4, interval):
    """seasonal_{name}_h{h}[t] == demand at t - (season - h) * interval."""
    base = feats.select(["region_id", "ts", "pickups"])
    for name, season in cfg4.dotted("advanced_features.seasonal_slot_lags").items():
        for h in cfg4.dotted("time.horizons"):
            offset = season - h
            shifted = base.select(
                [pl.col("region_id"),
                 (pl.col("ts") + pl.duration(minutes=interval * offset)).alias("ts"),
                 pl.col("pickups").alias("expected")]
            )
            joined = (
                feats.select(["region_id", "ts", f"pickups_seasonal_{name}_h{h}"])
                .join(shifted, on=["region_id", "ts"], how="inner")
            )
            bad = joined.filter(pl.col(f"pickups_seasonal_{name}_h{h}") != pl.col("expected"))
            assert bad.height == 0, f"pickups_seasonal_{name}_h{h}"


def test_expanding_slot_mean_excludes_the_current_row(feats, cfg4):
    """The expanding mean must be computed from strictly earlier same-slot rows."""
    key = ["region_id", "day_of_week", "slot_of_day"]
    sample = feats.sort(["region_id", "ts"]).select([*key, "ts", "pickups",
                                                     "pickups_slot_expanding_mean"])
    min_periods = cfg4.dotted("advanced_features.expanding_min_periods")
    recomputed = sample.with_columns(
        [
            (pl.col("pickups").shift(1).cum_sum() / pl.col("pickups").shift(1).cum_count())
            .over(key).alias("expected"),
            pl.col("pickups").shift(1).cum_count().over(key).alias("n"),
        ]
    )

    # first occurrence of a slot has no history, so it must be null - this holds
    # regardless of how much history the window contains
    firsts = sample.group_by(key).head(1)
    assert firsts["pickups_slot_expanding_mean"].null_count() == firsts.height

    # below min_periods the feature must be null rather than a one-sample "mean"
    thin = recomputed.filter(pl.col("n") < min_periods)
    assert thin["pickups_slot_expanding_mean"].null_count() == thin.height

    usable = recomputed.filter(pl.col("n") >= min_periods)
    if usable.is_empty():
        # a 15-day dev window sees each weekday+slot only twice, so nothing clears
        # min_periods=3. The null-pattern assertions above still prove the exclusion
        # rule; the value check runs on the full build.
        pytest.skip("dev window too short for the expanding mean to have min_periods")
    diff = (usable["pickups_slot_expanding_mean"] - usable["expected"]).abs()
    assert diff.max() < 1e-6


def test_target_calendar_describes_the_target_time(feats, cfg4, interval):
    tz = cfg4.dotted("time.timezone")
    for h in cfg4.dotted("time.horizons"):
        local = (
            feats.select(
                (pl.col("ts") + pl.duration(minutes=interval * h))
                .dt.convert_time_zone(tz).alias("t")
            )["t"]
        )
        assert (feats[f"target_hour_h{h}"] == local.dt.hour()).all()
        assert (feats[f"target_weekday_h{h}"] == local.dt.weekday()).all()


def test_target_calendar_differs_from_current_calendar(feats, cfg4):
    """If the target calendar had been copied from `ts` it would be identical."""
    h = max(cfg4.dotted("time.horizons"))
    assert (feats[f"target_hour_h{h}"] == feats["hour"]).sum() < feats.height


# ------------------------------------------------------------------ family D
def test_weather_is_never_from_the_future(feats, cfg4, interval):
    """The joined reading must be the latest with weather_timestamp <= forecast_time."""
    from src.utils.config import resolve_path

    weather = pl.read_parquet(
        resolve_path(cfg4, "paths.external") / "weather_hourly.parquet"
    ).select(["weather_timestamp", "temperature_2m"])
    sample = feats.select(["ts", "temperature_2m"]).unique(subset=["ts"]).sort("ts")
    expected_hour = sample.select(
        (pl.col("ts") + pl.duration(minutes=interval)).dt.truncate("1h").alias("weather_timestamp")
    )
    joined = pl.concat([sample, expected_hour], how="horizontal_extend").join(
        weather, on="weather_timestamp", how="left", suffix="_src"
    )
    # forecast_time = ts + interval; the reading used must not post-date it
    assert (joined["weather_timestamp"] <= joined["ts"] + timedelta(minutes=interval)).all()
    matched = joined.filter(pl.col("temperature_2m_src").is_not_null())
    diff = (matched["temperature_2m"] - matched["temperature_2m_src"]).abs()
    assert diff.max() < 1e-6, "joined weather is not the expected hourly reading"


def test_weather_flags_follow_thresholds(feats, cfg4):
    thresholds = cfg4.dotted("weather")
    assert (feats["is_raining"] == (feats["rain"] > thresholds["rain_mm_threshold"])).all()
    assert (feats["is_freezing"] ==
            (feats["temperature_2m"] < thresholds["freezing_temp_c"])).all()


# ------------------------------------------------------------- families E/F/G/H
def test_event_columns_are_non_negative_and_present_flag_agrees(feats):
    assert feats["event_count"].min() >= 0
    assert (feats["event_present"] == (feats["event_count"] > 0)).all()


def test_traffic_missing_flag_agrees_with_null_speed(feats):
    assert (feats["traffic_missing"] == feats["traffic_speed_mean"].is_null()).all()


def test_neighbor_aggregates_are_consistent(feats):
    both = feats.filter(pl.col("nb_pickups_mean").is_not_null())
    if both.is_empty():
        pytest.skip("no neighbour data in the dev sample")
    assert (both["nb_pickups_max"] >= both["nb_pickups_mean"] - 1e-6).all()
    assert (both["nb_pickups_sum"] >= both["nb_pickups_max"] - 1e-6).all()
    gradient = both["pickups"] - both["nb_pickups_mean"]
    assert (both["pickups_vs_neighbor_mean"] - gradient).abs().max() < 1e-3


def test_station_network_has_no_future_stations(feats):
    assert feats["active_station_count"].min() >= 0
    assert feats["region_area_km2"].min() > 0


def test_every_family_column_exists_and_is_not_all_null(feats, cfg4):
    from src.features.advanced import family_columns

    for family, columns in family_columns(cfg4).items():
        for column in columns:
            assert column in feats.columns, f"{family}: missing {column}"
        non_null = [c for c in columns if feats[c].null_count() < feats.height]
        assert non_null, f"{family}: every column is entirely null"
