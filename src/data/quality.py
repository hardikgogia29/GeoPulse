
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from src.data.timezone import dst_unreliable_utc_hours, find_dst_transitions

RAW_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def profile_raw(con: duckdb.DuckDBPyConnection, view: str, cfg) -> dict:
    """Compute the full Phase-1 data-quality profile for `view`."""
    bbox = cfg.dotted("cleaning.bbox")
    report: dict = {}

    report["total_rows"] = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]

    null_sql = ", ".join(f"sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}" for c in RAW_COLUMNS)
    report["null_counts"] = _rows(con, f"SELECT {null_sql} FROM {view}")[0]

    report["distinct_ride_ids"] = con.execute(
        f"SELECT count(DISTINCT ride_id) FROM {view}"
    ).fetchone()[0]
    report["duplicate_ride_id_extra_rows"] = report["total_rows"] - report["distinct_ride_ids"]

    report["invalid_timestamps"] = _rows(
        con,
        f"""
        SELECT
          sum(CASE WHEN started_at IS NULL THEN 1 ELSE 0 END)  AS null_started_at,
          sum(CASE WHEN ended_at   IS NULL THEN 1 ELSE 0 END)  AS null_ended_at,
          sum(CASE WHEN ended_at <= started_at THEN 1 ELSE 0 END) AS end_le_start,
          sum(CASE WHEN ended_at  = started_at THEN 1 ELSE 0 END) AS end_eq_start
        FROM {view}
        """,
    )[0]

    report["duration_distribution_seconds"] = _rows(
        con,
        f"""
        SELECT
          min(ride_duration) AS min,
          quantile_cont(ride_duration, 0.01)  AS p01,
          quantile_cont(ride_duration, 0.25)  AS p25,
          quantile_cont(ride_duration, 0.50)  AS median,
          quantile_cont(ride_duration, 0.75)  AS p75,
          quantile_cont(ride_duration, 0.99)  AS p99,
          quantile_cont(ride_duration, 0.999) AS p999,
          max(ride_duration) AS max,
          avg(ride_duration) AS mean,
          sum(CASE WHEN ride_duration < 60 THEN 1 ELSE 0 END)    AS under_60s,
          sum(CASE WHEN ride_duration > 86400 THEN 1 ELSE 0 END) AS over_24h
        FROM {view} WHERE ride_duration IS NOT NULL
        """,
    )[0]

    report["coordinate_ranges"] = _rows(
        con,
        f"""
        SELECT min(start_lat) AS start_lat_min, max(start_lat) AS start_lat_max,
               min(start_lng) AS start_lng_min, max(start_lng) AS start_lng_max,
               min(end_lat)   AS end_lat_min,   max(end_lat)   AS end_lat_max,
               min(end_lng)   AS end_lng_min,   max(end_lng)   AS end_lng_max
        FROM {view}
        """,
    )[0]

    report["coords_outside_bbox"] = con.execute(
        f"""
        SELECT count(*) FROM {view}
        WHERE start_lat NOT BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
           OR start_lng NOT BETWEEN {bbox['lng_min']} AND {bbox['lng_max']}
           OR end_lat   NOT BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
           OR end_lng   NOT BETWEEN {bbox['lng_min']} AND {bbox['lng_max']}
        """
    ).fetchone()[0]

    report["rides_per_month"] = _rows(
        con,
        f"""
        SELECT strftime(started_at, '%Y-%m') AS month, count(*) AS rides
        FROM {view} WHERE started_at IS NOT NULL GROUP BY 1 ORDER BY 1
        """,
    )

    report["rides_per_day_summary"] = _rows(
        con,
        f"""
        WITH d AS (
          SELECT started_at::DATE AS day, count(*) AS rides
          FROM {view} WHERE started_at IS NOT NULL GROUP BY 1
        )
        SELECT count(*) AS n_days, min(rides) AS min, quantile_cont(rides, 0.5) AS median,
               avg(rides) AS mean, max(rides) AS max,
               min(day) AS first_day, max(day) AS last_day
        FROM d
        """,
    )[0]

    report["category_mix"] = {
        "rideable_type": _rows(
            con, f"SELECT rideable_type, count(*) AS n FROM {view} GROUP BY 1 ORDER BY 2 DESC"
        ),
        "member_casual": _rows(
            con, f"SELECT member_casual, count(*) AS n FROM {view} GROUP BY 1 ORDER BY 2 DESC"
        ),
    }

    report["station_completeness"] = _rows(
        con,
        f"""
        SELECT
          sum(CASE WHEN start_station_id IS NULL THEN 1 ELSE 0 END) AS null_start_station_id,
          sum(CASE WHEN end_station_id   IS NULL THEN 1 ELSE 0 END) AS null_end_station_id,
          sum(CASE WHEN start_station_id IS NULL OR end_station_id IS NULL THEN 1 ELSE 0 END)
            AS null_either_station_id,
          count(DISTINCT start_station_id) AS distinct_start_station_ids,
          count(DISTINCT end_station_id)   AS distinct_end_station_ids
        FROM {view}
        """,
    )[0]

    report["dst_hour_profile"] = _dst_hour_profile(con, view, cfg)

    report["outside_project_window"] = con.execute(
        f"""
        SELECT count(*) FROM {view}
        WHERE started_at::DATE < DATE '{cfg.dotted("time.start_date")}'
           OR started_at::DATE > DATE '{cfg.dotted("time.end_date")}'
        """
    ).fetchone()[0]

    return report


def _dst_hour_profile(con: duckdb.DuckDBPyConnection, view: str, cfg) -> list[dict]:
    """Ride counts per UTC hour around each DST transition, with the expected effect.

    Makes the DST distortion visible in the report instead of leaving it implicit:
    the over-filled hour, the structurally empty hour, and their neighbours.
    """
    transitions = find_dst_transitions(
        cfg.dotted("time.timezone"),
        datetime.fromisoformat(cfg.dotted("time.start_date")),
        datetime.fromisoformat(cfg.dotted("time.end_date")),
    )
    # Bucket in EXPLICIT UTC. `date_trunc('hour', started_at)` truncates in the
    # session timezone, which for America/New_York collapses both passes through the
    # repeated fall-back hour onto one local label and shifts the counts by an hour -
    # the very distortion this table exists to show. `AT TIME ZONE 'UTC'` converts to
    # a naive UTC timestamp first, so the buckets are real UTC hours.
    def _naive_utc(value):
        return value.replace(tzinfo=None) if value.tzinfo is None else value.astimezone(
            timezone.utc
        ).replace(tzinfo=None)

    effects = {
        _naive_utc(h["utc_hour_start"]): h["effect"]
        for h in dst_unreliable_utc_hours(transitions)
    }

    ranges = []
    for transition in transitions:
        instant = transition.utc_instant.replace(minute=0, second=0, microsecond=0)
        ranges.append((instant - timedelta(hours=3), instant + timedelta(hours=3)))
    where = " OR ".join(
        f"(started_at >= TIMESTAMPTZ '{lo.isoformat()}' AND started_at < TIMESTAMPTZ '{hi.isoformat()}')"
        for lo, hi in ranges
    )
    rows = _rows(
        con,
        f"""
        SELECT date_trunc('hour', started_at AT TIME ZONE 'UTC') AS utc_hour,
               count(*) AS rides
        FROM {view} WHERE started_at IS NOT NULL AND ({where})
        GROUP BY 1 ORDER BY 1
        """,
    )
    seen = {_naive_utc(row["utc_hour"]) for row in rows}
    for hour_start in effects:
        if hour_start not in seen:
            rows.append({"utc_hour": hour_start, "rides": 0})
    for row in rows:
        row["utc_hour"] = _naive_utc(row["utc_hour"])
        row["expected_effect"] = effects.get(row["utc_hour"], "-")
    return sorted(rows, key=lambda r: r["utc_hour"])


def _fmt(value) -> str:
    if isinstance(value, float):
        # small percentages are the interesting ones here - 0.00 hides whether a rule
        # removed 178 rows or none at all
        if value != 0 and abs(value) < 0.01:
            return f"{value:,.4f}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(rows: list[dict]) -> str:
    if not rows:
        return "_(none)_\n"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(row[h]) for h in headers) + " |")
    return "\n".join(out) + "\n"


# public alias: other report writers (traffic) render tables the same way
render_table = _table


def render_markdown(report: dict, drop_waterfall: list[dict] | None, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Raw rows profiled: **{report['total_rows']:,}**")
    lines.append("")

    lines += ["## Nulls per raw column", "", _table([report["null_counts"]])]
    lines += [
        "## Ride id uniqueness",
        "",
        _table(
            [
                {
                    "distinct_ride_ids": report["distinct_ride_ids"],
                    "duplicate_extra_rows": report["duplicate_ride_id_extra_rows"],
                }
            ]
        ),
    ]
    lines += ["## Timestamp validity", "", _table([report["invalid_timestamps"]])]
    lines += [
        "## Ride duration (seconds, raw)",
        "",
        _table([report["duration_distribution_seconds"]]),
    ]
    lines += ["## Coordinate ranges", "", _table([report["coordinate_ranges"]])]
    lines += [
        "",
        f"Rows with any coordinate outside the configured service bbox: "
        f"**{report['coords_outside_bbox']:,}**",
        f"Rows with `started_at` outside the project window: "
        f"**{report['outside_project_window']:,}**",
        "",
    ]
    lines += ["## Station id completeness", "", _table([report["station_completeness"]])]
    lines += [
        "## DST transition hours (UTC)",
        "",
        "Citi Bike publishes naive local wall-clock time, which cannot say which pass "
        "through the repeated fall-back hour a ride belongs to. Under the "
        "`ambiguous=\"earliest\"` policy the distortion is deterministic: the first "
        "(EDT) pass absorbs both hours' rides (`over_filled`), the second (EST) pass "
        "is unreachable and therefore empty by construction, and spring-forward "
        "timestamps that cannot exist are shifted one hour forward (`shifted_in`). "
        "Four hours across 2023-2024 are affected; Phase 2 must flag these bins "
        "rather than model them as observed demand.",
        "",
        _table(report["dst_hour_profile"]),
    ]
    lines += ["## Rides per day", "", _table([report["rides_per_day_summary"]])]
    lines += ["## Rides per month", "", _table(report["rides_per_month"])]
    lines += ["## Rideable type", "", _table(report["category_mix"]["rideable_type"])]
    lines += ["## Membership", "", _table(report["category_mix"]["member_casual"])]

    if drop_waterfall is not None:
        lines += [
            "## Cleaning waterfall",
            "",
            "Each row is attributed to the **first** rule it violates, so the counts sum "
            "exactly to the rows removed. No row is dropped without a named rule.",
            "",
            _table(drop_waterfall),
        ]
    return "\n".join(lines) + "\n"


def save_report(report: dict, markdown: str, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
