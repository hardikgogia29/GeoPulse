"""Phase 1 data-validation gate, run against the produced artifacts.

Runs on the dev sample by default; the same assertions run on the full two-year
file too (skipped automatically if it has not been built yet).
"""

from datetime import datetime

import duckdb
import polars as pl
import pytest

from src.data.clean import OUTPUT_COLUMNS, verify_sorted

TZ = "America/New_York"


def _con(path):
    con = duckdb.connect()
    con.execute(f"SET TimeZone='{TZ}'")
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet('{path.as_posix()}')")
    return con


@pytest.fixture(params=["dev", "full"])
def trips_path(request, dev_trips_path, cfg):
    if request.param == "dev":
        return dev_trips_path
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.interim") / "trips_clean.parquet"
    if not path.exists():
        pytest.skip("full clean not built yet")
    return path


def test_schema_is_exactly_the_declared_output(trips_path):
    schema = pl.scan_parquet(trips_path).collect_schema()
    assert list(schema.names()) == OUTPUT_COLUMNS


def test_timestamps_are_sorted(trips_path):
    result = verify_sorted(trips_path)
    assert result["rowgroup_stat_violations"] == 0
    assert result["pairwise_inversions"] == 0
    assert result["is_sorted"] is True


def test_no_duplicate_ride_ids(trips_path):
    con = _con(trips_path)
    total, distinct = con.execute("SELECT count(*), count(DISTINCT ride_id) FROM trips").fetchone()
    assert total == distinct, f"{total - distinct} duplicate ride_ids survived cleaning"


def test_no_nulls_in_required_columns(trips_path):
    con = _con(trips_path)
    required = [
        "ride_id", "started_at", "ended_at", "ride_duration",
        "start_lat", "start_lng", "end_lat", "end_lng",
    ]
    sql = ", ".join(f"sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}" for c in required)
    row = con.execute(f"SELECT {sql} FROM trips").fetchone()
    assert all(value == 0 for value in row), dict(zip(required, row))


def test_durations_within_configured_bounds(trips_path, cfg):
    con = _con(trips_path)
    low = cfg.dotted("cleaning.min_duration_seconds")
    high = cfg.dotted("cleaning.max_duration_seconds")
    bad = con.execute(
        f"SELECT count(*) FROM trips WHERE ride_duration < {low} OR ride_duration > {high}"
    ).fetchone()[0]
    assert bad == 0


def test_end_strictly_after_start(trips_path):
    con = _con(trips_path)
    assert con.execute("SELECT count(*) FROM trips WHERE ended_at <= started_at").fetchone()[0] == 0


def test_coordinates_inside_service_bbox(trips_path, cfg):
    con = _con(trips_path)
    bbox = cfg.dotted("cleaning.bbox")
    bad = con.execute(
        f"""
        SELECT count(*) FROM trips
        WHERE start_lat NOT BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
           OR start_lng NOT BETWEEN {bbox['lng_min']} AND {bbox['lng_max']}
           OR end_lat   NOT BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
           OR end_lng   NOT BETWEEN {bbox['lng_min']} AND {bbox['lng_max']}
        """
    ).fetchone()[0]
    assert bad == 0


def test_all_rides_inside_project_window(trips_path, cfg):
    con = _con(trips_path)
    bad = con.execute(
        f"""
        SELECT count(*) FROM trips
        WHERE started_at::DATE < DATE '{cfg.dotted("time.start_date")}'
           OR started_at::DATE > DATE '{cfg.dotted("time.end_date")}'
        """
    ).fetchone()[0]
    assert bad == 0


def test_timestamps_are_timezone_aware_utc(trips_path):
    schema = pl.scan_parquet(trips_path).collect_schema()
    for column in ("started_at", "ended_at"):
        assert schema[column].time_zone == "UTC", f"{column} must be stored as tz-aware UTC"


def test_dst_distortion_matches_documented_behaviour(cfg):
    """DST handling must be exactly as documented - no surprises, no silent merge.

    Storing UTC means the fall-back hour does not collapse into its neighbour, but
    it also means the second pass through the repeated local hour is unreachable
    from a naive timestamp. That hour is therefore empty by construction, and the
    first pass is over-filled. Assert both, so the distortion stays visible and
    Phase 2 can flag those hours instead of modelling them as real demand.
    """
    from src.data.timezone import dst_unreliable_utc_hours, find_dst_transitions
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.interim") / "trips_clean.parquet"
    if not path.exists():
        pytest.skip("full clean not built yet")
    con = _con(path)
    transitions = find_dst_transitions(
        cfg.dotted("time.timezone"),
        datetime.fromisoformat(cfg.dotted("time.start_date")),
        datetime.fromisoformat(cfg.dotted("time.end_date")),
    )
    for hour in dst_unreliable_utc_hours(transitions):
        start_iso = hour["utc_hour_start"].isoformat()
        rides = con.execute(
            f"""
            SELECT count(*) FROM trips
            WHERE started_at >= TIMESTAMPTZ '{start_iso}'
              AND started_at <  TIMESTAMPTZ '{start_iso}' + INTERVAL 1 HOUR
            """
        ).fetchone()[0]
        if hour["effect"] == "empty":
            assert rides == 0, f"{start_iso} should be empty by construction, got {rides}"
        elif hour["effect"] == "over_filled":
            assert rides > 0, f"{start_iso} should hold both passes of the repeated hour"


def test_station_registry_is_consistent(cfg):
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.spatial") / "station_registry.parquet"
    if not path.exists():
        pytest.skip("station registry not built yet")
    registry = pl.read_parquet(path)
    assert registry["station_id"].n_unique() == registry.height
    assert registry["station_id"].null_count() == 0
    bbox = cfg.dotted("cleaning.bbox")
    assert registry["lat"].min() >= bbox["lat_min"]
    assert registry["lat"].max() <= bbox["lat_max"]
    assert registry["lng"].min() >= bbox["lng_min"]
    assert registry["lng"].max() <= bbox["lng_max"]
    assert (registry["first_seen"] <= registry["last_seen"]).all()
    assert (registry["appearances"] == registry["pickups"] + registry["dropoffs"]).all()


def test_events_are_clean_and_timezone_aware(cfg):
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.external") / "events.parquet"
    if not path.exists():
        pytest.skip("events not prepared yet")
    events = pl.read_parquet(path)
    schema = events.collect_schema()
    for column in ("event_start", "event_end"):
        assert schema[column].time_zone == "UTC"
    assert events["event_start"].is_sorted()
    # an event ending before it starts is a data error and must not survive
    assert (events["event_end"] >= events["event_start"]).all()
    assert events["duration_minutes"].min() >= 0
    assert events["event_start"].null_count() == 0
    assert events["event_end"].null_count() == 0
    # multi-week permits are flagged, not dropped - they are background, not spikes
    assert "is_long_running" in events.columns
    threshold = cfg.dotted("events.long_running_minutes")
    flagged = events.filter(pl.col("duration_minutes") > threshold)
    assert bool(flagged["is_long_running"].all())
    assert not bool(
        events.filter(pl.col("duration_minutes") <= threshold)["is_long_running"].any()
    )


def test_event_location_inventory_matches_events(cfg):
    """Events carry no coordinates; Phase 4 must geocode this exact set of strings."""
    from src.utils.config import resolve_path

    external = resolve_path(cfg, "paths.external")
    if not (external / "event_locations.parquet").exists():
        pytest.skip("events not prepared yet")
    events = pl.read_parquet(external / "events.parquet")
    locations = pl.read_parquet(external / "event_locations.parquet")
    assert set(locations.columns) >= {"event_location", "event_borough", "n_event_rows"}
    assert locations["n_event_rows"].sum() == events.height
    assert locations.height == events.select(
        ["event_location", "event_borough"]
    ).unique().height


def test_weather_grid_is_complete_and_hourly(cfg):
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.external") / "weather_hourly.parquet"
    if not path.exists():
        pytest.skip("weather not fetched yet")
    weather = pl.read_parquet(path)
    start = datetime.fromisoformat(cfg.dotted("time.start_date"))
    end = datetime.fromisoformat(cfg.dotted("time.end_date"))
    expected_hours = int((end - start).days + 1) * 24
    assert weather.height == expected_hours
    assert weather["weather_timestamp"].n_unique() == expected_hours
    assert weather["weather_timestamp"].is_sorted()
    gaps = weather["weather_timestamp"].diff().drop_nulls().unique().to_list()
    assert len(gaps) == 1 and gaps[0].total_seconds() == 3600
