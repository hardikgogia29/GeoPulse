
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import all_metrics, hotspot_f1  # noqa: E402
from src.features.advanced import FAMILY_ORDER, cumulative_feature_sets  # noqa: E402
from src.models.splits import load_splits, validate_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402


def load(path: Path, partitioned: bool, cfg, split, columns: list[str],
         frac: float | None, seed: int) -> pl.DataFrame:
    """Read one split, sampling **whole timestamps** rather than individual rows.

    Two reasons this is not just a memory fix:

    * Sampling after `collect()` materialises the entire split first - 86M rows x 175
      columns here, which is what ran the machine out of memory. Hashing `ts` inside
      the scan pushes the filter down so only the kept rows are ever read.
    * Keeping whole timestamps preserves the full region cross-section at each
      retained instant, which Hotspot-F1 needs: it ranks regions against each other
      within a timestamp, so a randomly thinned cross-section would silently change
      what the metric measures.
    """
    scan = pl.scan_parquet(str(path / "**" / "*.parquet") if partitioned else str(path))
    tz = cfg.dotted("time.timezone")
    local_date = pl.col("ts").dt.convert_time_zone(tz).dt.date()
    scan = scan.filter((local_date >= split.start) & (local_date <= split.end))
    if cfg.dotted("train.exclude_dst_unreliable"):
        scan = scan.filter(~pl.col("dst_unreliable"))
    if frac is not None and frac < 1.0:
        keep = int(round(frac * 10_000))
        scan = scan.filter((pl.col("ts").hash(seed) % 10_000) < keep)
    return scan.select(sorted(set(columns))).collect(engine="streaming")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.04)
    parser.add_argument("--valid-frac", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--targets", default="pickup_h1,dropoff_h1")
    args = parser.parse_args()

    cfg = load_config("h3", "lightgbm")
    if args.resolution is not None:
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], "resolution": args.resolution}})
    log = get_logger("ablation", cfg)
    tag = f"h3{cfg.dotted('spatial.resolution')}"

    if validate_splits(cfg):
        log.error("split problems: %s", validate_splits(cfg))
        return 1
    splits = load_splits(cfg)

    if args.dev_sample:
        features_path = resolve_path(cfg, "paths.dev_sample") / f"features4_{tag}.parquet"
        partitioned = False
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        from src.models.splits import Split

        bounds = pl.scan_parquet(str(features_path)).select(
            [pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi")]
        ).collect().to_dicts()[0]
        zone = ZoneInfo(cfg.dotted("time.timezone"))
        lo, hi = bounds["lo"].astimezone(zone).date(), bounds["hi"].astimezone(zone).date()
        cut = hi - timedelta(days=2)
        splits = {"train": Split("train", lo, cut),
                  "validate": Split("validate", cut + timedelta(days=1), hi)}
        log.warning("dev-sample mode: chronological holdout carved from the dev window "
                    "(%s..%s / %s..%s) - a smoke test, not a result", lo, cut,
                    cut + timedelta(days=1), hi)
    else:
        features_path = resolve_path(cfg, "paths.processed") / f"features4_{tag}"
        partitioned = True
    if not features_path.exists():
        log.error("missing %s - run scripts/17_build_features_full.py", features_path)
        return 1

    sets = cumulative_feature_sets(cfg)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    full_set = sets[FAMILY_ORDER[-1][0]]
    needed = ["region_id", "ts", "dst_unreliable", *full_set, *targets]

    seed = cfg.dotted("train.sample_seed")
    with timed(log, "load TRAIN"):
        train = load(features_path, partitioned, cfg, splits["train"], needed,
                     None if args.dev_sample else args.train_frac, seed)
    with timed(log, "load VALIDATE"):
        valid = load(features_path, partitioned, cfg, splits["validate"], needed,
                     None if args.dev_sample else args.valid_frac, seed)
    train = train.drop_nulls(subset=targets)
    valid = valid.drop_nulls(subset=targets)
    log.info("ablation rows: train %s | validate %s (identical at every step)",
             f"{train.height:,}", f"{valid.height:,}")

    regions = sorted(set(train["region_id"].unique()) | set(valid["region_id"].unique()))
    mapping = {r: i for i, r in enumerate(regions)}
    train = train.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))
    valid = valid.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))

    y_train = {t: train[t].to_numpy().astype("float64") for t in targets}
    y_valid = {t: valid[t].to_numpy().astype("float64") for t in targets}
    valid_ts = valid["ts"].to_numpy()

    params = dict(cfg.dotted("baseline"))
    params.pop("early_stopping_rounds", None)
    params.pop("n_estimators", None)
    params["seed"] = cfg.dotted("project.random_seed")
    clip_min = cfg.dotted("predict.clip_min")

    results: list[dict] = []
    for key, label in FAMILY_ORDER:
        columns = ["region_idx"] + [c for c in sets[key] if c != "region_id"]
        x_train = train.select(columns).to_numpy().astype("float32", copy=False)
        x_valid = valid.select(columns).to_numpy().astype("float32", copy=False)
        for target in targets:
            t0 = time.perf_counter()
            train_set = lgb.Dataset(x_train, label=y_train[target],
                                    categorical_feature=[0], free_raw_data=False)
            valid_set = lgb.Dataset(x_valid, label=y_valid[target],
                                    reference=train_set, free_raw_data=False)
            booster = lgb.train(
                params, train_set, num_boost_round=args.rounds,
                valid_sets=[valid_set], valid_names=["validate"],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            prediction = np.clip(booster.predict(x_valid), clip_min, None)
            metrics = all_metrics(y_valid[target], prediction)
            results.append({
                "step": key, "family": label, "n_features": len(columns), "target": target,
                **metrics, **hotspot_f1(y_valid[target], prediction, valid_ts),
                "best_iteration": booster.best_iteration,
                "train_seconds": round(time.perf_counter() - t0, 1),
            })
            log.info("%s %-18s %-11s features=%3d MAE=%.4f WAPE=%.4f HotF1=%.4f (%.0fs)",
                     key, label, target, len(columns), metrics["mae"], metrics["wape"],
                     results[-1]["hotspot_f1"], time.perf_counter() - t0)
            del train_set, valid_set, booster
            gc.collect()
        del x_train, x_valid
        gc.collect()

    frame = pl.DataFrame(results)
    metrics_dir = resolve_path(cfg, "paths.metrics", mkdir=True)
    suffix = "_dev" if args.dev_sample else ""
    frame.write_parquet(metrics_dir / f"ablation_{tag}{suffix}.parquet")
    (metrics_dir / f"ablation_{tag}{suffix}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ---- the table the phase exists to produce
    log.info("")
    log.info("ABLATION  (%s rows train / %s validate, %d rounds, identical at every step)",
             f"{train.height:,}", f"{valid.height:,}", args.rounds)
    log.info("%-2s %-18s %5s %-11s %8s %8s %8s %9s", "", "family", "feat", "target",
             "MAE", "WAPE", "HotF1", "dMAE vs prev")
    verdicts: list[dict] = []
    for index, (key, label) in enumerate(FAMILY_ORDER):
        for target in targets:
            row = next(r for r in results if r["step"] == key and r["target"] == target)
            if index == 0:
                delta = None
            else:
                prev_key = FAMILY_ORDER[index - 1][0]
                prev = next(r for r in results if r["step"] == prev_key and r["target"] == target)
                delta = 100 * (prev["mae"] - row["mae"]) / prev["mae"]
                verdicts.append({"step": key, "family": label, "target": target,
                                 "delta_pct": round(delta, 3)})
            log.info("%-2s %-18s %5d %-11s %8.4f %8.4f %8.4f %9s", key, label,
                     row["n_features"], target, row["mae"], row["wape"], row["hotspot_f1"],
                     "-" if delta is None else f"{delta:+.2f}%")

    log.info("")
    log.info("Marginal value of each family (mean MAE change vs the previous step):")
    for key, label in FAMILY_ORDER[1:]:
        deltas = [v["delta_pct"] for v in verdicts if v["step"] == key]
        mean = sum(deltas) / len(deltas)
        verdict = "HELPS" if mean > 0.05 else ("no effect" if mean > -0.05 else "HURTS")
        log.info("  %s %-18s %+6.2f%%  %s", key, label, mean, verdict)

    (metrics_dir / f"ablation_verdicts_{tag}{suffix}.json").write_text(
        json.dumps(verdicts, indent=2), encoding="utf-8")
    log.info("metrics -> %s", metrics_dir / f"ablation_{tag}{suffix}.parquet")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
