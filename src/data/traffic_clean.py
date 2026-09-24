
from __future__ import annotations

import duckdb

DUPLICATE_RULE = "duplicate_link_timestamp"

DROP_RULES: list[tuple[str, str]] = [
    ("null_timestamp", "data_as_of IS NULL"),
    ("null_link_id", "link_id IS NULL"),
    (
        "outside_project_window",
        "data_as_of::DATE < DATE '{start_date}' OR data_as_of::DATE > DATE '{end_date}'",
    ),
    ("invalid_status", "status IS NULL OR status <> '{valid_status}'"),
    ("speed_out_of_range", "speed IS NULL OR speed < {min_speed} OR speed > {max_speed}"),
    ("travel_time_nonpositive", "travel_time IS NULL OR travel_time <= 0"),
]

OUTPUT_COLUMNS = ["link_id", "data_as_of", "speed", "travel_time"]

SELECT_LIST = "link_id, data_as_of, speed, travel_time"


def rule_params(cfg) -> dict:
    return {
        "start_date": cfg.dotted("time.start_date"),
        "end_date": cfg.dotted("time.end_date"),
        "valid_status": cfg.dotted("traffic.valid_status"),
        "min_speed": cfg.dotted("traffic.min_speed_mph"),
        "max_speed": cfg.dotted("traffic.max_speed_mph"),
    }


def build_deduped_view(
    con: duckdb.DuckDBPyConnection, source_view: str, target_view: str = "traffic_deduped"
) -> dict:
    """Collapse repeated `(link_id, data_as_of)` readings, keeping one of each.

    Only the repeated keys are materialised, so this stays cheap on 23.8M rows.
    """
    # NULL keys are excluded from duplicate detection on purpose. SQL's `NOT IN`
    # yields NULL - not TRUE - when the probed key is NULL, so a null-key row would
    # be filtered out silently, with no rule to blame. Instead those rows bypass
    # de-duplication and are caught by the named `null_link_id` / `null_timestamp`
    # rules below. Keeping `_traffic_dup_rows` null-free also makes the `NOT IN`
    # below safe.
    key_not_null = "link_id IS NOT NULL AND data_as_of IS NOT NULL"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE _traffic_dup_rows AS
        SELECT * FROM {source_view}
        WHERE {key_not_null}
          AND (link_id, data_as_of) IN (
            SELECT link_id, data_as_of FROM {source_view}
            WHERE {key_not_null}
            GROUP BY 1, 2 HAVING count(*) > 1
          )
        """
    )
    total, distinct = con.execute(
        "SELECT count(*), count(DISTINCT (link_id, data_as_of)) FROM _traffic_dup_rows"
    ).fetchone()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT * FROM {source_view}
        WHERE NOT ({key_not_null})
           OR (link_id, data_as_of) NOT IN (
             SELECT link_id, data_as_of FROM _traffic_dup_rows
           )
        UNION ALL BY NAME
        SELECT * EXCLUDE (_rn) FROM (
          SELECT *, row_number() OVER (
            PARTITION BY link_id, data_as_of ORDER BY speed NULLS LAST, travel_time NULLS LAST
          ) AS _rn
          FROM _traffic_dup_rows
        ) WHERE _rn = 1
        """
    )
    return {"rule": DUPLICATE_RULE, "rows_removed": int(total - distinct)}


def profile_raw(con: duckdb.DuckDBPyConnection, view: str, cfg) -> dict:
    """Profile the raw traffic observations before anything is removed."""
    report: dict = {}
    report["total_rows"] = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
    report["distinct_links"] = con.execute(
        f"SELECT count(DISTINCT link_id) FROM {view}"
    ).fetchone()[0]

    cur = con.execute(
        f"""
        SELECT
          sum(CASE WHEN data_as_of IS NULL THEN 1 ELSE 0 END)   AS null_timestamp,
          sum(CASE WHEN link_id IS NULL THEN 1 ELSE 0 END)      AS null_link_id,
          sum(CASE WHEN speed IS NULL THEN 1 ELSE 0 END)        AS null_speed,
          sum(CASE WHEN travel_time IS NULL THEN 1 ELSE 0 END)  AS null_travel_time,
          min(data_as_of) AS first_reading, max(data_as_of) AS last_reading
        FROM {view}
        """
    )
    report["completeness"] = dict(zip([d[0] for d in cur.description], cur.fetchone()))

    report["status_mix"] = [
        {"status": row[0], "rows": row[1], "mean_speed": row[2]}
        for row in con.execute(
            f"SELECT status, count(*), round(avg(speed), 3) FROM {view} "
            f"GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    ]

    cur = con.execute(
        f"""
        SELECT min(speed) AS min, quantile_cont(speed, 0.25) AS p25,
               quantile_cont(speed, 0.50) AS median, quantile_cont(speed, 0.75) AS p75,
               quantile_cont(speed, 0.99) AS p99, max(speed) AS max, avg(speed) AS mean
        FROM {view} WHERE status = '{cfg.dotted("traffic.valid_status")}'
        """
    )
    report["speed_distribution_valid_only"] = dict(
        zip([d[0] for d in cur.description], cur.fetchone())
    )

    report["readings_per_day"] = [
        {"day": str(row[0]), "rows": row[1], "links": row[2]}
        for row in con.execute(
            f"SELECT data_as_of::DATE AS day, count(*), count(DISTINCT link_id) "
            f"FROM {view} GROUP BY 1 ORDER BY 1"
        ).fetchall()
    ]
    return report


def missing_days(report: dict, cfg) -> list[str]:
    """Calendar days in the project window with no traffic readings at all.

    The feed has real multi-day outages; Phase 4 must treat these as missing rather
    than as free-flowing traffic.
    """
    from datetime import date, timedelta

    present = {row["day"] for row in report["readings_per_day"]}
    start = date.fromisoformat(cfg.dotted("time.start_date"))
    end = date.fromisoformat(cfg.dotted("time.end_date"))
    out: list[str] = []
    cursor = start
    while cursor <= end:
        if cursor.isoformat() not in present:
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out
