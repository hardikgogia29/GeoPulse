
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import dst_transitions_for, ingest_file, write_partitioned  # noqa: E402
from src.utils.config import load_config, resolve_path, source_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only ingest the first N files")
    parser.add_argument("--force", action="store_true", help="re-ingest already-written files")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("ingest", cfg)

    src_dir = source_path(cfg, "citibike_trip_dir")
    pattern = cfg.dotted("sources.citibike_trip_glob")
    files = sorted(src_dir.glob(pattern))
    if args.limit:
        files = files[: args.limit]
    if not files:
        log.error("no source CSVs matched %s in %s", pattern, src_dir)
        return 1

    out_root = resolve_path(cfg, "paths.raw", "trips", mkdir=True)
    report_dir = resolve_path(cfg, "paths.reports", mkdir=True)
    tz_name = cfg.dotted("time.timezone")
    transitions = dst_transitions_for(cfg)
    log.info("found %d DST transitions in range: %s", len(transitions),
             ", ".join(f"{t.kind}@{t.local_window_start:%Y-%m-%d}" for t in transitions))

    manifest_path = report_dir / "ingest_manifest.json"
    manifest: dict = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    log.info("ingesting %d files -> %s", len(files), out_root)
    total_rows = 0
    started = time.perf_counter()
    for i, path in enumerate(files, 1):
        if not args.force and path.name in manifest:
            log.info("[%d/%d] %s (cached, skipping)", i, len(files), path.name)
            total_rows += manifest[path.name]["rows_out"]
            continue
        t0 = time.perf_counter()
        df, stats = ingest_file(path, tz_name, transitions)
        stats["parts"] = write_partitioned(df, out_root, path.stem)
        stats["seconds"] = round(time.perf_counter() - t0, 1)
        manifest[path.name] = stats
        total_rows += stats["rows_out"]
        log.info(
            "[%d/%d] %s rows=%s unparsed=%s months=%s %.1fs",
            i, len(files), path.name, f"{stats['rows_out']:,}",
            stats["unparsed_started_at"] + stats["unparsed_ended_at"],
            len(stats["parts"]), stats["seconds"],
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info(
        "ingest complete: %s rows across %d files in %.1f min -> %s",
        f"{total_rows:,}", len(files), (time.perf_counter() - started) / 60, out_root,
    )
    log.info("manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
