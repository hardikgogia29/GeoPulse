"""Traffic ingest + cleaning: rule attribution, de-duplication, geometry parsing."""

from datetime import date

import duckdb
import polars as pl
import pytest

from src.data import traffic_clean as tc
from src.data.clean import drop_reason_case, drop_waterfall, verify_sorted
from src.data.traffic import add_link_geometry_summary, day_chunks

TZ = "America/New_York"

# (label, expected drop_reason, row literal)
# columns: link_id, data_as_of, speed, travel_time, status
CASES = [
    ("ok", None, "('L1', TIMESTAMP '2023-06-01 10:00:00', 34.5, 120.0, '0')"),
    ("ok2", None, "('L2', TIMESTAMP '2023-06-01 10:00:00', 12.0, 300.0, '0')"),
    ("null_ts", "null_timestamp", "('L3', NULL, 20.0, 100.0, '0')"),
    ("null_link", "null_link_id", "(NULL, TIMESTAMP '2023-06-01 10:00:00', 20.0, 100.0, '0')"),
    ("before", "outside_project_window", "('L4', TIMESTAMP '2022-12-31 10:00:00', 20.0, 100.0, '0')"),
    ("after", "outside_project_window", "('L5', TIMESTAMP '2025-02-01 10:00:00', 20.0, 100.0, '0')"),
    # the feed's own invalid marker: reported as -101 with a filler speed of 0
    ("bad_status", "invalid_status", "('L6', TIMESTAMP '2023-06-01 10:00:00', 0.0, 0.0, '-101')"),
    ("zero_speed", "speed_out_of_range", "('L7', TIMESTAMP '2023-06-01 10:00:00', 0.0, 100.0, '0')"),
    ("fast", "speed_out_of_range", "('L8', TIMESTAMP '2023-06-01 10:00:00', 150.0, 100.0, '0')"),
    ("no_tt", "travel_time_nonpositive", "('L9', TIMESTAMP '2023-06-01 10:00:00', 30.0, 0.0, '0')"),
    # same link + same instant twice: de-duplicated before the rule pass
    ("dup_a", None, "('LD', TIMESTAMP '2023-06-01 11:00:00', 30.0, 90.0, '0')"),
    ("dup_b", None, "('LD', TIMESTAMP '2023-06-01 11:00:00', 31.0, 91.0, '0')"),
]


@pytest.fixture()
def con(cfg):
    connection = duckdb.connect()
    connection.execute(f"SET TimeZone='{TZ}'")
    values = ",\n".join(row for _, _, row in CASES)
    connection.execute(
        f"""
        CREATE VIEW raw AS SELECT
          link_id,
          data_as_of_naive AT TIME ZONE '{TZ}' AS data_as_of,
          speed, travel_time, status
        FROM (VALUES {values}) AS t(link_id, data_as_of_naive, speed, travel_time, status)
        """
    )
    yield connection
    connection.close()


def test_rules_attribute_each_row_correctly(con, cfg):
    dup = tc.build_deduped_view(con, "raw", "deduped")
    assert dup == {"rule": "duplicate_link_timestamp", "rows_removed": 1}

    case = drop_reason_case(tc.rule_params(cfg), tc.DROP_RULES)
    rows = con.execute(f"SELECT link_id, {case} AS reason FROM deduped").fetchall()
    got = {link_id: reason for link_id, reason in rows if link_id is not None}
    assert got["L1"] is None
    assert got["L4"] == "outside_project_window"
    assert got["L5"] == "outside_project_window"
    assert got["L6"] == "invalid_status"
    assert got["L7"] == "speed_out_of_range"
    assert got["L8"] == "speed_out_of_range"
    assert got["L9"] == "travel_time_nonpositive"
    assert got["LD"] is None
    # the null-link row survives as a NULL key and is caught by its own rule
    null_link = [reason for link_id, reason in rows if link_id is None]
    assert null_link == ["null_link_id"]


def test_waterfall_reconciles_against_raw(con, cfg):
    raw_total = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    dup = tc.build_deduped_view(con, "raw", "deduped")
    case = drop_reason_case(tc.rule_params(cfg), tc.DROP_RULES)
    con.execute(f"CREATE VIEW flagged AS SELECT *, {case} AS drop_reason FROM deduped")
    rows = drop_waterfall(con, "flagged", raw_total, pre_rules=[dup], rules=tc.DROP_RULES)
    waterfall = {row["rule"]: row["rows_removed"] for row in rows}
    assert waterfall["TOTAL REMOVED"] + waterfall["KEPT"] == raw_total


def test_invalid_status_rows_are_not_averaged_in(con, cfg):
    """A status=-101 row carries speed 0; keeping it would bias every mean down."""
    tc.build_deduped_view(con, "raw", "deduped")
    case = drop_reason_case(tc.rule_params(cfg), tc.DROP_RULES)
    mean_kept = con.execute(
        f"SELECT avg(speed) FROM (SELECT *, {case} AS drop_reason FROM deduped) "
        f"WHERE drop_reason IS NULL"
    ).fetchone()[0]
    assert mean_kept > 10, "invalid filler rows leaked into the kept set"


def test_day_chunks_cover_the_window_exactly():
    chunks = day_chunks(date(2023, 1, 1), date(2023, 1, 5), 1)
    assert len(chunks) == 5
    assert chunks[0] == (date(2023, 1, 1), date(2023, 1, 2))
    assert chunks[-1] == (date(2023, 1, 5), date(2023, 1, 6))
    # windows must tile without gaps or overlap
    for (_, hi), (lo, _) in zip(chunks, chunks[1:]):
        assert hi == lo

    wide = day_chunks(date(2023, 1, 1), date(2023, 1, 10), 4)
    assert wide[0] == (date(2023, 1, 1), date(2023, 1, 5))
    assert wide[-1][1] == date(2023, 1, 11)


def test_link_geometry_parsing():
    registry = pl.DataFrame(
        {
            "link_id": ["A", "B"],
            "link_points": [
                "40.7081105,-73.99944 40.7084705,-73.99884 40.70868,-73.998331",
                "not-a-coordinate",
            ],
        }
    )
    out = add_link_geometry_summary(registry)
    first = out.filter(pl.col("link_id") == "A").to_dicts()[0]
    assert first["n_vertices"] == 3
    assert first["start_lat"] == pytest.approx(40.7081105)
    assert first["end_lng"] == pytest.approx(-73.998331)
    assert first["mid_lat"] == pytest.approx(40.7084705)
    second = out.filter(pl.col("link_id") == "B").to_dicts()[0]
    assert second["n_vertices"] == 0
    assert second["mid_lat"] is None


def _traffic_artifact(cfg, kind: str):
    from src.utils.config import resolve_path

    path = resolve_path(cfg, f"paths.{kind}") / "traffic_clean.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not built yet - run scripts/08_clean_traffic.py")
    return path


@pytest.mark.parametrize("kind", ["dev_sample", "interim"])
def test_cleaned_traffic_artifact_is_valid(cfg, kind):
    path = _traffic_artifact(cfg, kind)
    con = duckdb.connect()
    con.execute(f"SET TimeZone='{TZ}'")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path.as_posix()}')")

    schema = pl.scan_parquet(path).collect_schema()
    assert list(schema.names()) == tc.OUTPUT_COLUMNS
    assert schema["data_as_of"].time_zone == "UTC"

    low = cfg.dotted("traffic.min_speed_mph")
    high = cfg.dotted("traffic.max_speed_mph")
    bad = con.execute(
        f"SELECT count(*) FROM t WHERE speed < {low} OR speed > {high} "
        f"OR travel_time <= 0 OR link_id IS NULL OR data_as_of IS NULL"
    ).fetchone()[0]
    assert bad == 0

    total, distinct = con.execute(
        "SELECT count(*), count(DISTINCT (link_id, data_as_of)) FROM t"
    ).fetchone()
    assert total == distinct, "duplicate (link_id, data_as_of) survived cleaning"

    result = verify_sorted(path, column="data_as_of")
    assert result["is_sorted"] is True
    assert result["pairwise_inversions"] == 0


def test_traffic_link_registry_geometry_is_usable(cfg):
    from src.utils.config import resolve_path

    path = resolve_path(cfg, "paths.spatial") / "traffic_links.parquet"
    if not path.exists():
        pytest.skip("traffic link registry not built yet")
    links = pl.read_parquet(path)
    assert links.height > 0
    assert links["link_id"].null_count() == 0
    located = links.drop_nulls(["mid_lat", "mid_lng"])
    assert located.height == links.height, "every link needs a parseable midpoint for Phase 4"
    # midpoints must land in the NYC region, not at (0, 0)
    assert located["mid_lat"].min() > 40.4
    assert located["mid_lat"].max() < 41.1
    assert located["mid_lng"].min() > -74.4
    assert located["mid_lng"].max() < -73.6
