"""Panel correctness: density, target alignment, and no leakage.

The targets are produced with `LEAD`, which is only correct on a gap-free grid. These
tests re-derive them by an independent timestamp join, so a density bug or an
off-by-one in the horizon cannot pass unnoticed - that would silently corrupt every
model from Phase 3 onward.
"""

from datetime import timedelta

import duckdb
import polars as pl
import pytest


def _panel_path(cfg):
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.dev_sample") / "panel_h39.parquet"
    if not path.exists():
        pytest.skip("dev panel not built - run scripts/10_build_panel.py --dev-sample")
    return path


@pytest.fixture(scope="module")
def panel(cfg_module):
    return pl.read_parquet(_panel_path(cfg_module))


@pytest.fixture(scope="module")
def cfg_module():
    from src.utils.config import load_config

    return load_config("h3")


def test_panel_is_dense(panel):
    regions = panel["region_id"].n_unique()
    bins = panel["ts"].n_unique()
    assert panel.height == regions * bins, "panel must be a full region x bin grid"
    # and exactly one row per pair
    assert panel.select(["region_id", "ts"]).unique().height == panel.height


def test_bins_are_evenly_spaced(panel, cfg_module):
    interval = cfg_module.dotted("time.interval_minutes")
    stamps = panel["ts"].unique().sort()
    gaps = stamps.diff().drop_nulls().unique().to_list()
    assert len(gaps) == 1
    assert gaps[0] == timedelta(minutes=interval)


def test_demand_is_non_negative_and_never_null(panel):
    for column in ("pickups", "dropoffs"):
        assert panel[column].null_count() == 0
        assert panel[column].min() >= 0


def test_net_flow_is_consistent(panel):
    assert (panel["net_flow"] == panel["dropoffs"] - panel["pickups"]).all()


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_targets_match_an_independent_timestamp_join(panel, cfg_module, horizon):
    """`pickup_h{h}[r,t]` must equal `pickups[r, t + h*interval]` - re-derived here
    by joining on the timestamp itself rather than trusting row order."""
    interval = cfg_module.dotted("time.interval_minutes")
    shifted = panel.select(
        [
            pl.col("region_id"),
            (pl.col("ts") - pl.duration(minutes=interval * horizon)).alias("ts"),
            pl.col("pickups").alias("expected_pickup"),
            pl.col("dropoffs").alias("expected_dropoff"),
        ]
    )
    joined = panel.join(shifted, on=["region_id", "ts"], how="left")
    mismatched_pickup = joined.filter(
        pl.col(f"pickup_h{horizon}") != pl.col("expected_pickup")
    )
    mismatched_dropoff = joined.filter(
        pl.col(f"dropoff_h{horizon}") != pl.col("expected_dropoff")
    )
    assert mismatched_pickup.height == 0, mismatched_pickup.head(3).to_dicts()
    assert mismatched_dropoff.height == 0, mismatched_dropoff.head(3).to_dicts()

    # where the horizon runs past the end of the grid the target must be null,
    # never silently zero - a zero would teach the model a phantom demand collapse
    tail = joined.filter(pl.col("expected_pickup").is_null())
    assert tail.height == panel["region_id"].n_unique() * horizon
    assert tail[f"pickup_h{horizon}"].null_count() == tail.height


def test_targets_are_strictly_in_the_future(panel, cfg_module):
    """Leakage guard: a target must never be derivable from the current bin."""
    interval = cfg_module.dotted("time.interval_minutes")
    # h1 target equals the NEXT bin's pickups, so it must differ from the current
    # bin's pickups somewhere - if they were identical the shift never happened
    same = (panel["pickup_h1"] == panel["pickups"]).sum()
    assert same < panel.height, "pickup_h1 is identical to pickups - shift not applied"

    # and the h-step targets must be ordered in time: h2 equals h1 shifted by one bin
    shifted_h1 = panel.select(
        [
            pl.col("region_id"),
            (pl.col("ts") - pl.duration(minutes=interval)).alias("ts"),
            pl.col("pickup_h1").alias("h1_next"),
        ]
    )
    joined = panel.join(shifted_h1, on=["region_id", "ts"], how="left")
    check = joined.filter(pl.col("h1_next").is_not_null())
    assert (check["pickup_h2"] == check["h1_next"]).all()


def test_panel_totals_reconcile_against_the_trips(cfg_module):
    """Every pickup in an active region inside the grid must appear exactly once."""
    from src.utils.config import resolve_path

    panel_path = _panel_path(cfg_module)
    trips_path = resolve_path(cfg_module, "paths.dev_sample") / "trips_clean.parquet"
    regions_path = resolve_path(cfg_module, "paths.dev_sample") / "regions_h39.parquet"
    if not (trips_path.exists() and regions_path.exists()):
        pytest.skip("dev artifacts missing")

    from src.spatial.h3_indexer import H3Indexer, load_h3_extension

    indexer = H3Indexer(cfg_module.dotted("spatial.resolution"))
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    load_h3_extension(con)
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{trips_path.as_posix()}')")
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{regions_path.as_posix()}')")
    con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{panel_path.as_posix()}')")

    lo, hi = con.execute("SELECT min(ts), max(ts) FROM p").fetchone()
    expr = indexer.sql_index_expr("start_lat", "start_lng")
    expected = con.execute(
        f"""
        SELECT count(*) FROM t
        WHERE {expr} IN (SELECT region_id FROM r)
          AND started_at >= '{lo}'::TIMESTAMPTZ
          AND started_at <  '{hi}'::TIMESTAMPTZ + INTERVAL 15 MINUTE
        """
    ).fetchone()[0]
    actual = con.execute("SELECT sum(pickups) FROM p").fetchone()[0]
    assert actual == expected, f"panel has {actual} pickups, trips imply {expected}"


def test_region_metadata_matches_the_panel(cfg_module):
    from src.utils.config import resolve_path

    panel_path = _panel_path(cfg_module)
    regions_path = resolve_path(cfg_module, "paths.dev_sample") / "regions_h39.parquet"
    if not regions_path.exists():
        pytest.skip("region metadata missing")
    regions = pl.read_parquet(regions_path)
    panel_regions = set(pl.read_parquet(panel_path, columns=["region_id"])["region_id"].unique())
    assert set(regions["region_id"]) == panel_regions
    assert regions["region_id"].n_unique() == regions.height
    assert regions["area_km2"].min() > 0
    assert regions["centroid_lat"].is_between(40.4, 41.1).all()
    assert regions["centroid_lng"].is_between(-74.4, -73.6).all()
