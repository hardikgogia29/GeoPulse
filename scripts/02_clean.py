
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import (  # noqa: E402
    build_deduped_view,
    drop_waterfall,
    flagged_view_sql,
    months_present,
    verify_sorted,
    write_sorted_clean,
    write_sorted_clean_by_month,
)
from src.data.quality import profile_raw, render_markdown, save_report  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true",
                        help="restrict to the configured 7-day dev window")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", default=None,
                        help="DuckDB spill directory and scratch space for month parts")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the quality report against an existing clean file, "
                             "skipping the (expensive) re-sort")
    parser.add_argument("--keep-parts", action="store_true",
                        help="keep the per-month intermediate parquet parts")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("clean", cfg)

    raw_root = resolve_path(cfg, "paths.raw", "trips")
    if not any(raw_root.glob("month=*/*.parquet")):
        log.error("no ingested parquet under %s - run scripts/01_ingest.py first", raw_root)
        return 1

    if args.dev_sample:
        out_path = resolve_path(cfg, "paths.dev_sample", mkdir=True) / "trips_clean.parquet"
        suffix = "dev_sample"
    else:
        out_path = resolve_path(cfg, "paths.interim", mkdir=True) / "trips_clean.parquet"
        suffix = "full"

    report_dir = resolve_path(cfg, "paths.reports", mkdir=True)
    scratch = Path(args.temp_dir) if args.temp_dir else out_path.parent / "_scratch"
    tz = cfg.dotted("time.timezone")

    con = duckdb.connect()
    con.execute(f"SET TimeZone='{tz}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET preserve_insertion_order=true")
    scratch.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{scratch.as_posix()}'")
    con.execute("SET enable_progress_bar=false")

    glob_path = (raw_root / "**" / "*.parquet").as_posix()
    con.execute(
        f"CREATE VIEW raw_all AS SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true)"
    )
    if args.dev_sample:
        start, end = cfg.dotted("dev_sample.start"), cfg.dotted("dev_sample.end")
        # prune hive partitions first so the dev sample reads one month, not 79M rows
        months = sorted({start[:4] + start[5:7], end[:4] + end[5:7]})
        month_list = ", ".join(f"'{m}'" for m in months)
        con.execute(
            f"CREATE VIEW raw AS SELECT * FROM raw_all "
            f"WHERE month IN ({month_list}) "
            f"AND started_at::DATE BETWEEN DATE '{start}' AND DATE '{end}'"
        )
        log.info("dev-sample mode: %s .. %s (partitions %s)", start, end, month_list)
    else:
        con.execute("CREATE VIEW raw AS SELECT * FROM raw_all")

    with timed(log, "profile raw data"):
        report = profile_raw(con, "raw", cfg)
    raw_total = report["total_rows"]
    log.info("raw rows: %s | distinct ride_ids: %s | duplicate extra rows: %s",
             f"{raw_total:,}", f"{report['distinct_ride_ids']:,}",
             f"{report['duplicate_ride_id_extra_rows']:,}")

    with timed(log, "de-duplicate ride ids"):
        dup_entry = build_deduped_view(con, "raw", "deduped")
    log.info("duplicate ride_id rows removed: %s", f"{dup_entry['rows_removed']:,}")

    con.execute(f"CREATE VIEW flagged AS {flagged_view_sql(cfg, 'deduped')}")
    with timed(log, "cleaning waterfall"):
        waterfall = drop_waterfall(con, "flagged", raw_total, pre_rules=[dup_entry])
    for row in waterfall:
        log.info("  %-24s %12s  (%.4f%% of raw)", row["rule"],
                 f"{row['rows_removed']:,}", row["pct_of_raw"])
    report["cleaning_waterfall"] = waterfall

    reconciled = {row["rule"]: row["rows_removed"] for row in waterfall}
    if reconciled["TOTAL REMOVED"] + reconciled["KEPT"] != raw_total:
        log.error("waterfall does not reconcile against %s raw rows - refusing to continue",
                  f"{raw_total:,}")
        return 3

    if args.report_only:
        if not out_path.exists() or out_path.stat().st_size == 0:
            log.error("--report-only needs an existing %s", out_path)
            return 1
        log.info("--report-only: reusing existing %s", out_path.name)
    elif args.dev_sample:
        with timed(log, f"write sorted clean parquet -> {out_path.name}"):
            write_sorted_clean(con, "flagged", out_path)
    else:
        months = months_present(con, "raw")
        log.info("sorting %d month partitions, then concatenating in month order", len(months))

        def progress(month: str, rows: int) -> None:
            log.info("  %s  %12s rows", month, f"{rows:,}")

        part_dir = scratch / "month_parts"
        with timed(log, f"write sorted clean parquet -> {out_path.name}"):
            parts = write_sorted_clean_by_month(
                con, "flagged", out_path, months, part_dir, progress=progress
            )
        if not args.keep_parts:
            for part in parts:
                part.unlink(missing_ok=True)
            part_dir.rmdir()

    with timed(log, "verify chronological sort"):
        verification = verify_sorted(out_path)
    report["sort_verification"] = verification
    for key, value in verification.items():
        log.info("  %-26s %s", key, value)
    if not verification["is_sorted"]:
        log.error("SORT VERIFICATION FAILED - output is not chronologically ordered")
        return 2
    if not args.report_only and verification["rows"] != reconciled["KEPT"]:
        log.error("written rows (%s) != waterfall KEPT (%s)",
                  f"{verification['rows']:,}", f"{reconciled['KEPT']:,}")
        return 4
    log.info("sort verified: 0 row-group violations, 0 pairwise inversions; "
             "row count matches the waterfall")

    title = f"GeoPulse - Citi Bike data quality report ({suffix})"
    markdown = render_markdown(report, waterfall, title)
    json_path = report_dir / f"data_quality_report_{suffix}.json"
    md_path = report_dir / f"data_quality_report_{suffix}.md"
    save_report(report, markdown, json_path, md_path)
    log.info("report: %s", md_path)
    log.info("clean output: %s (%.1f MB, %s rows)", out_path,
             verification["file_size_mb"], f"{verification['rows']:,}")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
