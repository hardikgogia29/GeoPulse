"""Each cleaning rule must catch exactly the rows it is meant to, and nothing else.

Phase 1 forbids silent drops, so the waterfall must also reconcile: rows removed
per rule + rows kept == rows in.
"""

from types import SimpleNamespace

import duckdb
import pytest

from src.data.clean import (
    DROP_RULES,
    build_deduped_view,
    drop_waterfall,
    flagged_view_sql,
    verify_sorted,
    write_sorted_clean,
)

TZ = "America/New_York"

# (ride_id, expected drop_reason, SQL row literal)
# columns: ride_id, started_at, ended_at, ride_duration, start_lat, start_lng,
#          end_lat, end_lng, start_station_id, end_station_id
CASES = [
    ("r_ok", None, "('r_ok', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:20:00', 1200, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_rt", None, "('r_rt', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:20:00', 1200, 40.75, -73.98, 40.75, -73.98, 'A', 'A')"),
    ("r_ms", None, "('r_ms', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:20:00', 1200, 40.75, -73.98, 40.76, -73.97, NULL, 'B')"),
    ("r_nt", "null_timestamp", "('r_nt', NULL, TIMESTAMP '2023-06-01 10:20:00', 1200, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    # regression: a NULL ride_id must survive de-duplication and be dropped by a
    # NAMED rule. `WHERE ride_id NOT IN (...)` is NULL for a NULL ride_id, which
    # silently discarded this row before - exactly the kind of unattributed drop
    # Phase 1 forbids.
    (None, "null_ride_id", "(NULL, TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:20:00', 1200, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_bw", "outside_project_window", "('r_bw', TIMESTAMP '2022-12-31 23:00:00', TIMESTAMP '2022-12-31 23:20:00', 1200, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_aw", "outside_project_window", "('r_aw', TIMESTAMP '2025-01-01 00:30:00', TIMESTAMP '2025-01-01 00:50:00', 1200, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_eb", "end_before_start", "('r_eb', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 09:00:00', -3600, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_ts", "duration_too_short", "('r_ts', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:00:30', 30, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_tl", "duration_too_long", "('r_tl', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-03 10:00:00', 172800, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_nc", "missing_coords", "('r_nc', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:20:00', 1200, NULL, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_ob", "coords_out_of_bbox", "('r_ob', TIMESTAMP '2023-06-01 10:00:00', TIMESTAMP '2023-06-01 10:20:00', 1200, 34.05, -118.24, 40.76, -73.97, 'A', 'B')"),
    # the same ride id twice: de-duplication keeps one, before the rule pass
    ("r_dup", None, "('r_dup', TIMESTAMP '2023-06-01 09:00:00', TIMESTAMP '2023-06-01 09:20:00', 1200, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
    ("r_dup", None, "('r_dup', TIMESTAMP '2023-06-01 09:00:00', TIMESTAMP '2023-06-01 09:25:00', 1500, 40.75, -73.98, 40.76, -73.97, 'A', 'B')"),
]

EXPECTED_REASON = {ride_id: reason for ride_id, reason, _ in CASES}


@pytest.fixture()
def ctx(cfg):
    con = duckdb.connect()
    con.execute(f"SET TimeZone='{TZ}'")
    values = ",\n".join(row for _, _, row in CASES)
    con.execute(
        f"""
        CREATE TABLE raw AS
        SELECT * FROM (VALUES {values}) AS t(
          ride_id, started_at_naive, ended_at_naive, ride_duration,
          start_lat, start_lng, end_lat, end_lng, start_station_id, end_station_id
        )
        """
    )
    # store timestamps the way the ingest stage does: tz-aware, UTC
    con.execute(
        f"""
        CREATE VIEW raw_tz AS SELECT
          ride_id,
          'classic_bike' AS rideable_type,
          started_at_naive AT TIME ZONE '{TZ}' AS started_at,
          ended_at_naive   AT TIME ZONE '{TZ}' AS ended_at,
          ride_duration,
          start_station_id, 'start name' AS start_station_name, start_lat, start_lng,
          end_station_id,   'end name'   AS end_station_name,   end_lat,   end_lng,
          'member' AS member_casual
        FROM raw
        """
    )
    dup_entry = build_deduped_view(con, "raw_tz", "deduped")
    raw_total = con.execute("SELECT count(*) FROM raw_tz").fetchone()[0]
    con.execute(f"CREATE VIEW flagged AS {flagged_view_sql(cfg, 'deduped')}")
    yield SimpleNamespace(con=con, dup_entry=dup_entry, raw_total=raw_total)
    con.close()


def test_each_row_gets_its_expected_reason(ctx):
    rows = ctx.con.execute("SELECT ride_id, drop_reason FROM flagged").fetchall()
    by_ride: dict[str, list] = {}
    for ride_id, reason in rows:
        by_ride.setdefault(ride_id, []).append(reason)

    for ride_id, expected in EXPECTED_REASON.items():
        assert len(by_ride[ride_id]) == 1, f"{ride_id} appears {len(by_ride[ride_id])} times"
        assert by_ride[ride_id][0] == expected, ride_id


def test_duplicate_ride_id_is_resolved_before_the_rule_pass(ctx):
    assert ctx.dup_entry == {"rule": "duplicate_ride_id", "rows_removed": 1}
    assert ctx.con.execute("SELECT count(*) FROM deduped WHERE ride_id = 'r_dup'").fetchone()[0] == 1
    # the survivor is the deterministic pick: earliest started_at, then earliest ended_at
    kept = ctx.con.execute("SELECT ride_duration FROM deduped WHERE ride_id = 'r_dup'").fetchone()[0]
    assert kept == 1200


def test_waterfall_reconciles(ctx):
    rows = drop_waterfall(ctx.con, "flagged", ctx.raw_total, pre_rules=[ctx.dup_entry])
    waterfall = {row["rule"]: row["rows_removed"] for row in rows}
    removed = waterfall["duplicate_ride_id"] + sum(waterfall[name] for name, _ in DROP_RULES)
    assert removed == waterfall["TOTAL REMOVED"]
    # must reconcile against the RAW row count, not the post-dedupe count
    assert waterfall["TOTAL REMOVED"] + waterfall["KEPT"] == ctx.raw_total


def test_flags_not_drops(ctx):
    """Missing station id and round trips are flagged, never removed."""
    kept = ctx.con.execute(
        "SELECT ride_id, "
        "(start_station_id IS NULL OR end_station_id IS NULL) AS missing_station_id, "
        "(start_station_id IS NOT NULL AND start_station_id = end_station_id) AS is_round_trip "
        "FROM flagged WHERE drop_reason IS NULL"
    ).fetchall()
    flags = {ride_id: (missing, round_trip) for ride_id, missing, round_trip in kept}
    assert flags["r_ms"] == (True, False)
    assert flags["r_rt"] == (False, True)
    assert flags["r_ok"] == (False, False)


def test_written_output_is_sorted_and_verifier_catches_unsorted(ctx, tmp_path):
    out = tmp_path / "clean.parquet"
    write_sorted_clean(ctx.con, "flagged", out, row_group_size=2)
    result = verify_sorted(out)
    assert result["is_sorted"] is True
    assert result["pairwise_inversions"] == 0

    unsorted = tmp_path / "unsorted.parquet"
    ctx.con.execute(
        f"COPY (SELECT * FROM read_parquet('{out.as_posix()}') ORDER BY started_at DESC) "
        f"TO '{unsorted.as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 2)"
    )
    bad = verify_sorted(unsorted)
    assert bad["is_sorted"] is False
    assert bad["pairwise_inversions"] > 0
