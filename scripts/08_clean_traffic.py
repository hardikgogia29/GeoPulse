
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import traffic_clean as tc  # noqa: E402
from src.data.clean import drop_reason_case, drop_waterfall, verify_sorted  # noqa: E402
from src.data.quality import render_table as _table, save_report  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def render_markdown(report: dict, waterfall: list[dict], gaps: list[str], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Raw traffic observations profiled: **{report['total_rows']:,}** "
                 f"across **{report['distinct_links']}** road links")
    lines += ["", "## Completeness", "", _table([report["completeness"]])]
    lines += [
        "## Feed status codes",
        "",
        "`0` is a valid reading. Anything else means the sensor reported nothing; "
        "those rows carry `speed = 0` and are removed rather than averaged in.",
        "",
        _table(report["status_mix"]),
    ]
    lines += [
        "## Speed distribution (valid readings only, mph)",
        "",
        _table([report["speed_distribution_valid_only"]]),
    ]
    lines += ["## Cleaning waterfall", "", _table(waterfall)]
    lines += [
        "## Feed outages",
        "",
        f"Calendar days in the project window with **no** traffic readings: "
        f"**{len(gaps)}** of {len(report['readings_per_day']) + len(gaps)}.",
        "",
        "These are real gaps in the published feed, not a fetch failure. Phase 4 must "
        "treat them as missing data (with a `traffic_missing` flag), never as "
        "free-flowing traffic.",
        "",
    ]
    if gaps:
        lines += ["```", "\n".join(gaps), "```", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("traffic-clean", cfg)

    raw_root = resolve_path(cfg, "paths.raw", "traffic")
    if not any(raw_root.glob("month=*/*.parquet")):
        log.error("no traffic parquet under %s - run scripts/07_fetch_traffic.py first", raw_root)
        return 1

    if args.dev_sample:
        out_path = resolve_path(cfg, "paths.dev_sample", mkdir=True) / "traffic_clean.parquet"
        suffix = "dev_sample"
    else:
        out_path = resolve_path(cfg, "paths.interim", mkdir=True) / "traffic_clean.parquet"
        suffix = "full"
    report_dir = resolve_path(cfg, "paths.reports", mkdir=True)

    con = duckdb.connect()
    con.execute(f"SET TimeZone='{cfg.dotted('time.timezone')}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(args.temp_dir).as_posix()}'")

    glob_path = (raw_root / "**" / "*.parquet").as_posix()
    con.execute(
        f"CREATE VIEW raw_all AS SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true)"
    )
    if args.dev_sample:
        start, end = cfg.dotted("dev_sample.start"), cfg.dotted("dev_sample.end")
        con.execute(
            f"CREATE VIEW raw AS SELECT * FROM raw_all "
            f"WHERE data_as_of::DATE BETWEEN DATE '{start}' AND DATE '{end}'"
        )
        log.info("dev-sample mode: %s .. %s", start, end)
    else:
        con.execute("CREATE VIEW raw AS SELECT * FROM raw_all")

    with timed(log, "profile raw traffic"):
        report = tc.profile_raw(con, "raw", cfg)
    raw_total = report["total_rows"]
    log.info("raw observations: %s across %d links",
             f"{raw_total:,}", report["distinct_links"])
    for row in report["status_mix"]:
        log.info("  status=%-6s %12s rows  mean_speed=%s",
                 row["status"], f"{row['rows']:,}", row["mean_speed"])

    gaps = tc.missing_days(report, cfg) if not args.dev_sample else []
    if gaps:
        log.warning("feed outages: %d days with no readings at all (e.g. %s)",
                    len(gaps), ", ".join(gaps[:5]))

    with timed(log, "de-duplicate (link_id, data_as_of)"):
        dup_entry = tc.build_deduped_view(con, "raw", "traffic_deduped")
    log.info("duplicate (link, timestamp) rows removed: %s", f"{dup_entry['rows_removed']:,}")

    case = drop_reason_case(tc.rule_params(cfg), tc.DROP_RULES)
    con.execute(
        f"CREATE VIEW traffic_flagged AS "
        f"SELECT *, {case} AS drop_reason FROM traffic_deduped"
    )
    with timed(log, "cleaning waterfall"):
        waterfall = drop_waterfall(
            con, "traffic_flagged", raw_total, pre_rules=[dup_entry], rules=tc.DROP_RULES
        )
    for row in waterfall:
        log.info("  %-26s %12s  (%.4f%% of raw)", row["rule"],
                 f"{row['rows_removed']:,}", row["pct_of_raw"])
    report["cleaning_waterfall"] = waterfall
    report["missing_days"] = gaps

    reconciled = {row["rule"]: row["rows_removed"] for row in waterfall}
    if reconciled["TOTAL REMOVED"] + reconciled["KEPT"] != raw_total:
        log.error("waterfall does not reconcile against %s raw rows", f"{raw_total:,}")
        return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with timed(log, f"write sorted traffic parquet -> {out_path.name}"):
        con.execute(
            f"""
            COPY (
              SELECT {tc.SELECT_LIST} FROM traffic_flagged
              WHERE drop_reason IS NULL
              ORDER BY data_as_of
            ) TO '{out_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
            """
        )

    with timed(log, "verify chronological sort"):
        verification = verify_sorted(out_path, column="data_as_of")
    report["sort_verification"] = verification
    for key, value in verification.items():
        log.info("  %-26s %s", key, value)
    if not verification["is_sorted"]:
        log.error("SORT VERIFICATION FAILED")
        return 2

    title = f"GeoPulse - NYC DOT traffic speed data quality report ({suffix})"
    save_report(
        report,
        render_markdown(report, waterfall, gaps, title),
        report_dir / f"traffic_quality_report_{suffix}.json",
        report_dir / f"traffic_quality_report_{suffix}.md",
    )
    log.info("report: %s", report_dir / f"traffic_quality_report_{suffix}.md")
    log.info("clean output: %s (%.1f MB, %s rows)", out_path,
             verification["file_size_mb"], f"{verification['rows']:,}")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
