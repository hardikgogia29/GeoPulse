
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

DUPLICATE_RULE = "duplicate_ride_id"

# Order matters: it defines which rule gets the blame when a row violates several.
# De-duplication is applied ahead of these (see `build_deduped_view`).
DROP_RULES: list[tuple[str, str]] = [
    # a row with no ride id cannot be de-duplicated or traced back to a trip
    ("null_ride_id", "ride_id IS NULL"),
    ("null_timestamp", "started_at IS NULL OR ended_at IS NULL"),
    (
        "outside_project_window",
        "started_at::DATE < DATE '{start_date}' OR started_at::DATE > DATE '{end_date}'",
    ),
    ("end_before_start", "ended_at <= started_at"),
    ("duration_too_short", "ride_duration < {min_duration_seconds}"),
    ("duration_too_long", "ride_duration > {max_duration_seconds}"),
    (
        "missing_coords",
        "start_lat IS NULL OR start_lng IS NULL OR end_lat IS NULL OR end_lng IS NULL",
    ),
    (
        "coords_out_of_bbox",
        "start_lat NOT BETWEEN {lat_min} AND {lat_max} "
        "OR start_lng NOT BETWEEN {lng_min} AND {lng_max} "
        "OR end_lat NOT BETWEEN {lat_min} AND {lat_max} "
        "OR end_lng NOT BETWEEN {lng_min} AND {lng_max}",
    ),
]

OUTPUT_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "ride_duration",
    "start_station_id",
    "start_station_name",
    "start_lat",
    "start_lng",
    "end_station_id",
    "end_station_name",
    "end_lat",
    "end_lng",
    "member_casual",
    "missing_station_id",
    "is_round_trip",
]

SELECT_LIST = """
    ride_id, rideable_type, started_at, ended_at, ride_duration,
    start_station_id, start_station_name, start_lat, start_lng,
    end_station_id,   end_station_name,   end_lat,   end_lng,
    member_casual,
    (start_station_id IS NULL OR end_station_id IS NULL) AS missing_station_id,
    (start_station_id IS NOT NULL
     AND start_station_id = end_station_id)              AS is_round_trip
"""


def rule_params(cfg) -> dict:
    bbox = cfg.dotted("cleaning.bbox")
    return {
        "start_date": cfg.dotted("time.start_date"),
        "end_date": cfg.dotted("time.end_date"),
        "min_duration_seconds": cfg.dotted("cleaning.min_duration_seconds"),
        "max_duration_seconds": cfg.dotted("cleaning.max_duration_seconds"),
        "lat_min": bbox["lat_min"],
        "lat_max": bbox["lat_max"],
        "lng_min": bbox["lng_min"],
        "lng_max": bbox["lng_max"],
    }


def drop_reason_case(params: dict, rules: list[tuple[str, str]]) -> str:
    """A CASE expression labelling each row with the FIRST rule it violates.

    Generic over the rule list so the traffic table reuses the same machinery.
    """
    branches = [
        f"    WHEN {condition.format(**params)} THEN '{name}'" for name, condition in rules
    ]
    return "CASE\n" + "\n".join(branches) + "\n    ELSE NULL END"


def build_deduped_view(
    con: duckdb.DuckDBPyConnection, source_view: str, target_view: str = "deduped"
) -> dict:
    """Materialise the rows that share a ride id, keep one of each, expose the rest.

    Returns the waterfall entry for the de-duplication rule. Only rows whose ride id
    is genuinely repeated are materialised, so this stays cheap even though it is
    logically a global operation.
    """
    # Rows with a NULL ride_id are deliberately excluded from duplicate detection:
    # SQL's `NOT IN` yields NULL - not TRUE - when the probed value is NULL, so such
    # a row would be filtered out silently with no rule to blame. They bypass
    # de-duplication and reach the named rules instead. Keeping `_dup_rows` free of
    # NULLs also makes the `NOT IN` below safe.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE _dup_rows AS
        SELECT * FROM {source_view}
        WHERE ride_id IS NOT NULL
          AND ride_id IN (
            SELECT ride_id FROM {source_view}
            WHERE ride_id IS NOT NULL
            GROUP BY 1 HAVING count(*) > 1
          )
        """
    )
    total, distinct = con.execute(
        "SELECT count(*), count(DISTINCT ride_id) FROM _dup_rows"
    ).fetchone()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT * FROM {source_view}
        WHERE ride_id IS NULL
           OR ride_id NOT IN (SELECT ride_id FROM _dup_rows)
        UNION ALL BY NAME
        SELECT * EXCLUDE (_rn) FROM (
          SELECT *, row_number() OVER (
            PARTITION BY ride_id ORDER BY started_at NULLS LAST, ended_at NULLS LAST
          ) AS _rn
          FROM _dup_rows
        ) WHERE _rn = 1
        """
    )
    return {"rule": DUPLICATE_RULE, "rows_removed": int(total - distinct)}


def flagged_view_sql(cfg, source_view: str) -> str:
    """A view of `source_view` with `drop_reason` attached."""
    return (
        f"SELECT *, {drop_reason_case(rule_params(cfg), DROP_RULES)} AS drop_reason "
        f"FROM {source_view}"
    )


def drop_waterfall(
    con: duckdb.DuckDBPyConnection,
    flagged_view: str,
    raw_total: int,
    pre_rules: list[dict] | None = None,
    rules: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Rows removed per rule, in rule order, plus the surviving count.

    `pre_rules` carries removals applied before the rule pass (de-duplication), so
    the table still reconciles against the raw row count.
    """
    rules = rules if rules is not None else DROP_RULES
    counts = dict(
        con.execute(
            f"SELECT coalesce(drop_reason, '__kept__') AS reason, count(*) "
            f"FROM {flagged_view} GROUP BY 1"
        ).fetchall()
    )
    kept = int(counts.pop("__kept__", 0))
    rows: list[dict] = []
    for entry in pre_rules or []:
        rows.append({**entry, "pct_of_raw": round(100 * entry["rows_removed"] / raw_total, 4)})
    for name, _ in rules:
        removed = int(counts.pop(name, 0))
        rows.append(
            {"rule": name, "rows_removed": removed, "pct_of_raw": round(100 * removed / raw_total, 4)}
        )
    for name, removed in counts.items():  # defensive: an unexpected reason must still show up
        rows.append(
            {
                "rule": name,
                "rows_removed": int(removed),
                "pct_of_raw": round(100 * removed / raw_total, 4),
            }
        )
    total_removed = sum(row["rows_removed"] for row in rows)
    rows.append(
        {
            "rule": "TOTAL REMOVED",
            "rows_removed": total_removed,
            "pct_of_raw": round(100 * total_removed / raw_total, 4),
        }
    )
    rows.append(
        {"rule": "KEPT", "rows_removed": kept, "pct_of_raw": round(100 * kept / raw_total, 4)}
    )
    return rows


def write_sorted_clean(
    con: duckdb.DuckDBPyConnection,
    flagged_view: str,
    out_path: Path,
    row_group_size: int = 1_000_000,
) -> None:
    """Single-shot sorted write. Fine for the dev sample and tests; for the full
    two years use :func:`write_sorted_clean_by_month`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
          SELECT {SELECT_LIST}
          FROM {flagged_view}
          WHERE drop_reason IS NULL
          ORDER BY started_at
        ) TO '{out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size})
        """
    )


def months_present(con: duckdb.DuckDBPyConnection, view: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT month FROM {view} WHERE month IS NOT NULL ORDER BY month"
        ).fetchall()
    ]


def write_sorted_clean_by_month(
    con: duckdb.DuckDBPyConnection,
    flagged_view: str,
    out_path: Path,
    months: list[str],
    part_dir: Path,
    row_group_size: int = 1_000_000,
    progress=None,
) -> list[Path]:
    """Sort each month independently, then concatenate the parts in month order.

    The raw data is partitioned by the *local* month of `started_at`, so month
    boundaries line up exactly with the sort key: concatenating month-sorted parts
    in ascending month order is globally sorted by construction. `verify_sorted`
    checks that claim against the written file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for month in months:
        part = part_dir / f"trips_clean_{month}.parquet"
        con.execute(
            f"""
            COPY (
              SELECT {SELECT_LIST}
              FROM {flagged_view}
              WHERE month = '{month}' AND drop_reason IS NULL
              ORDER BY started_at
            ) TO '{part.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size})
            """
        )
        rows = pq.ParquetFile(part).metadata.num_rows
        if rows == 0:
            part.unlink()  # a month that cleaning emptied contributes nothing
        else:
            parts.append(part)
        if progress:
            progress(month, rows)

    file_list = ", ".join(f"'{p.as_posix()}'" for p in parts)
    con.execute("SET preserve_insertion_order=true")
    con.execute(
        f"""
        COPY (SELECT * FROM read_parquet([{file_list}]))
        TO '{out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size})
        """
    )
    return parts


def verify_sorted(path: Path, column: str = "started_at") -> dict:
    """Prove the written file is chronologically sorted - two independent checks.

    1. Parquet row-group statistics: every row group's min >= the previous max.
    2. A full scan of the sort column in file order, counting inversions.

    `column` lets the traffic table (`data_as_of`) reuse this unchanged.
    """
    parquet = pq.ParquetFile(path)
    meta = parquet.metadata
    col_index = meta.schema.names.index(column)

    prev_max = None
    rowgroup_violations = 0
    for group in range(meta.num_row_groups):
        stats = meta.row_group(group).column(col_index).statistics
        if stats is None:
            continue
        if prev_max is not None and stats.min < prev_max:
            rowgroup_violations += 1
        prev_max = stats.max

    # Stream the column in batches so 79M timestamps are never materialised at once,
    # and compare with numpy - building Python datetime objects for every row turns a
    # 20-second check into a multi-minute one.
    inversions = 0
    first_value = last_value = last_value_raw = None
    for batch in parquet.iter_batches(batch_size=2_000_000, columns=[column]):
        arrow_column = batch.column(column)
        values = arrow_column.to_numpy(zero_copy_only=False)
        if values.size == 0:
            continue
        if first_value is None:
            first_value = arrow_column[0].as_py()  # keep the tz-aware value for the report
        if last_value_raw is not None and values[0] < last_value_raw:
            inversions += 1
        inversions += int((np.diff(values) < np.timedelta64(0, "ns")).sum())
        last_value_raw = values[-1]
        last_value = arrow_column[-1].as_py()

    return {
        "rows": int(meta.num_rows),
        "row_groups": int(meta.num_row_groups),
        "rowgroup_stat_violations": rowgroup_violations,
        "pairwise_inversions": inversions,
        "is_sorted": rowgroup_violations == 0 and inversions == 0,
        f"first_{column}": str(first_value),
        f"last_{column}": str(last_value),
        "file_size_mb": round(path.stat().st_size / 1e6, 1),
    }
