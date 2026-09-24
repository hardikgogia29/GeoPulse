
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import verify_sorted  # noqa: E402
from src.data.timezone import find_dst_transitions  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class Check:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def add(self, item: str, passed: bool, evidence: str) -> None:
        self.results.append((item, bool(passed), evidence))

    def run(self, item: str, fn) -> None:
        try:
            passed, evidence = fn()
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
            passed, evidence = False, f"{type(exc).__name__}: {exc}"
        self.add(item, passed, evidence)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)


def main() -> int:
    cfg = load_config()
    log = get_logger("phase1-check", cfg)
    check = Check()

    interim = resolve_path(cfg, "paths.interim")
    external = resolve_path(cfg, "paths.external")
    spatial = resolve_path(cfg, "paths.spatial")
    dev = resolve_path(cfg, "paths.dev_sample")
    reports = resolve_path(cfg, "paths.reports")
    raw = resolve_path(cfg, "paths.raw")

    # 1. repo structure + configs
    def structure():
        needed = [
            "configs/base.yaml", "configs/h3.yaml", "configs/s2.yaml",
            "configs/lightgbm.yaml", "configs/tft.yaml", "configs/stgnn.yaml",
            "src/data", "src/spatial", "src/features", "src/models",
            "src/evaluation", "src/inventory", "src/rebalancing", "src/utils",
            "scripts", "tests", "notebooks/eda.ipynb",
            "outputs/models", "outputs/metrics", "outputs/figures", "outputs/experiments",
            "requirements.txt", "README.md", "Makefile",
        ]
        missing = [p for p in needed if not (REPO / p).exists()]
        return not missing, ("all present" if not missing else f"missing: {missing}")

    check.run("Repo structure + configs exist", structure)

    def no_hardcoding():
        """Spot-check that the dates/thresholds in configs are not also literals in src/."""
        literals = [cfg.dotted("time.start_date"), cfg.dotted("time.end_date"),
                    cfg.dotted("split.train.end"), cfg.dotted("dev_sample.start")]
        hits = []
        for path in (REPO / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for literal in literals:
                if literal in text:
                    hits.append(f"{path.relative_to(REPO)}:{literal}")
        return not hits, ("no config values hardcoded in src/" if not hits else str(hits))

    check.run("Nothing hardcoded that should be configurable", no_hardcoding)

    # 2. data quality reports
    def quality_reports():
        wanted = [
            reports / "data_quality_report_full.md",
            reports / "data_quality_report_full.json",
            reports / "traffic_quality_report_full.md",
        ]
        missing = [p.name for p in wanted if not p.exists()]
        if missing:
            return False, f"missing: {missing}"
        payload = json.loads((reports / "data_quality_report_full.json").read_text())
        return True, (f"trips profiled {payload['total_rows']:,} raw rows; "
                      f"reports in {reports.relative_to(REPO)}")

    check.run("Data quality report generated and saved", quality_reports)

    # 3. cleaning waterfall with per-rule counts, reconciling
    def waterfall():
        payload = json.loads((reports / "data_quality_report_full.json").read_text())
        rows = {r["rule"]: r["rows_removed"] for r in payload["cleaning_waterfall"]}
        total = payload["total_rows"]
        reconciles = rows["TOTAL REMOVED"] + rows["KEPT"] == total
        named = [r for r in payload["cleaning_waterfall"]
                 if r["rule"] not in ("TOTAL REMOVED", "KEPT")]
        return reconciles, (f"{len(named)} named rules; removed {rows['TOTAL REMOVED']:,} + "
                            f"kept {rows['KEPT']:,} == {total:,} raw rows")

    check.run("Cleaning logs counts per rule and reconciles (no silent drops)", waterfall)

    # 4. DST
    def dst():
        transitions = find_dst_transitions(
            cfg.dotted("time.timezone"),
            datetime.fromisoformat(cfg.dotted("time.start_date")),
            datetime.fromisoformat(cfg.dotted("time.end_date")),
        )
        found = sorted(f"{t.kind}@{t.local_window_start:%Y-%m-%d}" for t in transitions)
        schema = pl.scan_parquet(interim / "trips_clean.parquet").collect_schema()
        utc = schema["started_at"].time_zone == "UTC" and schema["ended_at"].time_zone == "UTC"
        return len(transitions) == 4 and utc, f"{len(found)} transitions {found}; stored tz=UTC:{utc}"

    check.run("Timestamps localized, all four DST transitions verified", dst)

    # 5. station registry
    def registry():
        path = spatial / "station_registry.parquet"
        if not path.exists():
            return False, "missing station_registry.parquet"
        reg = pl.read_parquet(path)
        unique = reg["station_id"].n_unique() == reg.height
        has_flag = "coord_anomaly" in reg.columns
        anomalies = int(reg["coord_anomaly"].sum()) if has_flag else -1
        renamed = int((reg["n_distinct_names"] > 1).sum())
        return unique and has_flag, (f"{reg.height:,} stations, ids unique, "
                                     f"{anomalies} coordinate anomalies flagged, "
                                     f"{renamed} renamed over time")

    check.run("Station registry built, coordinate anomalies flagged", registry)

    # 6. dev sample
    def dev_sample():
        needed = ["trips_clean.parquet", "weather_hourly.parquet", "events.parquet",
                  "station_registry.parquet", "manifest.json"]
        missing = [n for n in needed if not (dev / n).exists()]
        if missing:
            return False, f"missing: {missing}"
        manifest = json.loads((dev / "manifest.json").read_text())
        start = datetime.fromisoformat(manifest["window_local"]["start"])
        end = datetime.fromisoformat(manifest["window_local"]["end"])
        days = (end - start).days
        return days == 7, (f"{days}-day window {start.date()}..{(end).date()}, "
                           f"{manifest.get('trips_rows', 0):,} trips, "
                           f"{manifest.get('traffic_rows', 0):,} traffic readings")

    check.run("7-day dev sample exists", dev_sample)

    # sortedness of both cleaned tables
    def sorted_trips():
        result = verify_sorted(interim / "trips_clean.parquet")
        return result["is_sorted"], (f"{result['rows']:,} rows, {result['row_groups']} row groups, "
                                     f"{result['rowgroup_stat_violations']} stat violations, "
                                     f"{result['pairwise_inversions']} inversions")

    check.run("Cleaned trips are globally sorted by started_at (verified)", sorted_trips)

    def sorted_traffic():
        path = interim / "traffic_clean.parquet"
        if not path.exists():
            return False, "missing traffic_clean.parquet"
        result = verify_sorted(path, column="data_as_of")
        return result["is_sorted"], (f"{result['rows']:,} rows, "
                                     f"{result['pairwise_inversions']} inversions")

    check.run("Cleaned traffic is sorted by data_as_of (verified)", sorted_traffic)

    # external sources
    def sources():
        bits = []
        weather = pl.read_parquet(external / "weather_hourly.parquet")
        bits.append(f"weather {weather.height:,} h")
        events = pl.read_parquet(external / "events.parquet")
        bits.append(f"events {events.height:,}")
        links = pl.read_parquet(spatial / "traffic_links.parquet")
        bits.append(f"traffic links {links.height}")
        raw_traffic = len(list((raw / "traffic").glob("month=*/*.parquet")))
        bits.append(f"traffic parts {raw_traffic}")
        return raw_traffic > 0, ", ".join(bits)

    check.run("Weather, events and traffic ingested for the same period", sources)

    # tests
    def tests():
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
            cwd=REPO, capture_output=True, text=True,
        )
        tail = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        return proc.returncode == 0, (tail[-1] if tail else "no output")

    check.run("Test suite passes", tests)

    width = max(len(item) for item, _, _ in check.results)
    log.info("")
    log.info("PHASE 1 - DEFINITION OF DONE")
    log.info("=" * (width + 60))
    for item, passed, evidence in check.results:
        log.info("[%s] %-*s  %s", "PASS" if passed else "FAIL", width, item, evidence)
    log.info("=" * (width + 60))
    log.info("%d/%d checks passed", sum(p for _, p, _ in check.results), len(check.results))
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
