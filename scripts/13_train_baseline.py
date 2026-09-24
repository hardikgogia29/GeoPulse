
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import all_metrics, hotspot_f1  # noqa: E402
from src.features.basic import feature_columns  # noqa: E402
from src.models import baseline  # noqa: E402
from src.models.splits import load_splits, validate_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402

TARGET_KINDS = {"pickup": "pickups", "dropoff": "dropoffs"}


def load_split(path: Path, partitioned: bool, cfg, split, columns: list[str],
               sample_frac: float | None, seed: int, exclude_dst: bool) -> pl.DataFrame:
    """Read one chronological split, optionally subsampling rows."""
    scan = pl.scan_parquet(
        str(path / "**" / "*.parquet") if partitioned else str(path)
    )
    tz = cfg.dotted("time.timezone")
    local_date = pl.col("ts").dt.convert_time_zone(tz).dt.date()
    scan = scan.filter((local_date >= split.start) & (local_date <= split.end))
    if exclude_dst:
        scan = scan.filter(~pl.col("dst_unreliable"))
    scan = scan.select(sorted(set(columns)))
    frame = scan.collect(engine="streaming")
    if sample_frac is not None and sample_frac < 1.0:
        frame = frame.sample(fraction=sample_frac, seed=seed, shuffle=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--spatial", default="h3", help="config overlay: h3 or s2")
    parser.add_argument("--resolution", type=int, default=None,
                        help="override the spatial resolution / S2 level")
    parser.add_argument("--train-sample-frac", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.spatial, "lightgbm")
    if args.resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: args.resolution,
                                          "resolution": args.resolution}})
    log = get_logger("baseline", cfg)
    from src.spatial.h3_indexer import make_indexer as _mk
    _idx = _mk(cfg)
    tag = f"{_idx.name}{_idx.resolution}"

    problems = validate_splits(cfg)
    if problems:
        for problem in problems:
            log.error("split problem: %s", problem)
        return 1
    splits = load_splits(cfg)

    if args.dev_sample:
        features_path = resolve_path(cfg, "paths.dev_sample") / f"features_{tag}.parquet"
        partitioned = False
        # the dev window sits inside TRAIN, so carve a chronological holdout from it
        available = pl.scan_parquet(str(features_path)).select(
            [pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi")]
        ).collect().to_dicts()[0]
        tz = cfg.dotted("time.timezone")
        zone = ZoneInfo(tz)
        lo = available["lo"].astimezone(zone).date()
        hi = available["hi"].astimezone(zone).date()
        from src.models.splits import Split

        cut = hi - timedelta(days=2)
        splits = {
            "train": Split("train", lo, cut),
            "validate": Split("validate", cut + timedelta(days=1), hi),
        }
        log.warning("dev-sample mode: the 7-day window lies inside TRAIN, so a "
                    "chronological holdout is carved from it (train %s..%s, "
                    "validate %s..%s). Dev numbers are a smoke test, not results.",
                    lo, cut, cut + timedelta(days=1), hi)
    else:
        features_path = resolve_path(cfg, "paths.processed") / f"features_{tag}"
        partitioned = True
    if not features_path.exists():
        log.error("missing %s - run scripts/12_build_features.py", features_path)
        return 1

    model_features = feature_columns(cfg)
    horizons = cfg.dotted("time.horizons")
    sample_frac = (
        args.train_sample_frac
        if args.train_sample_frac is not None
        else (1.0 if args.dev_sample else cfg.dotted("train.train_sample_frac"))
    )
    seed = cfg.dotted("train.sample_seed")
    exclude_dst = cfg.dotted("train.exclude_dst_unreliable")

    baseline_cols = [
        baseline.lag_column_for(source, season, h)
        for source in TARGET_KINDS.values()
        for season in baseline.seasonal_lag_steps(cfg).values()
        for h in horizons
    ]
    target_cols = [f"{kind}_h{h}" for kind in TARGET_KINDS for h in horizons]
    # TRAIN does not need the Seasonal Naive lag columns - the baseline is only
    # scored on VALIDATION - and carrying 16 extra columns over 21M rows is GBs.
    train_cols = ["region_id", "dst_unreliable", *model_features, *target_cols]
    valid_cols = [*train_cols, "ts", *baseline_cols]

    model_inputs = ["region_idx"] + [c for c in model_features if c != "region_id"]
    regions = pl.scan_parquet(
        str(features_path / "**" / "*.parquet") if partitioned else str(features_path)
    ).select(pl.col("region_id").unique()).collect()["region_id"].sort().to_list()
    mapping = {region: index for index, region in enumerate(regions)}

    def design_matrix(frame: pl.DataFrame) -> np.ndarray:
        return (
            frame.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))
            .select(model_inputs)
            .to_numpy()
            .astype("float32", copy=False)
        )

    with timed(log, "load TRAIN"):
        train = load_split(features_path, partitioned, cfg, splits["train"], train_cols,
                           sample_frac, seed, exclude_dst)
        # Drop rows whose targets run past the end of the series ONCE, up front.
        # Otherwise every target needs a boolean-indexed COPY of the design matrix,
        # which is another 3 GB per model. At most regions x max(horizon) rows go.
        before = train.height
        train = train.drop_nulls(subset=target_cols)
        y_train = {t: train[t].to_numpy().astype("float64") for t in target_cols}
        x_train = design_matrix(train)
        del train
        gc.collect()
    log.info("train rows %s (sample_frac=%s, dropped %s with horizon overrun)",
             f"{x_train.shape[0]:,}", sample_frac, f"{before - x_train.shape[0]:,}")

    with timed(log, "load VALIDATE"):
        valid = load_split(features_path, partitioned, cfg, splits["validate"], valid_cols,
                           None, seed, exclude_dst)
        before_valid = valid.height
        valid = valid.drop_nulls(subset=target_cols)
        y_valid = {t: valid[t].to_numpy().astype("float64") for t in target_cols}
        valid_ts = valid["ts"].to_numpy()
        baseline_values = {c: valid[c].to_numpy().astype("float64") for c in set(baseline_cols)}
        x_valid = design_matrix(valid)
        del valid
        gc.collect()
    log.info("validate rows %s (full, dropped %s with horizon overrun)",
             f"{x_valid.shape[0]:,}", f"{before_valid - x_valid.shape[0]:,}")
    if x_train.shape[0] == 0 or x_valid.shape[0] == 0:
        log.error("empty split - nothing to train on")
        return 2
    log.info("design matrix: train %s x %d (%.2f GB) | valid %s x %d (%.2f GB)",
             f"{x_train.shape[0]:,}", x_train.shape[1], x_train.nbytes / 1e9,
             f"{x_valid.shape[0]:,}", x_valid.shape[1], x_valid.nbytes / 1e9)

    params = dict(cfg.dotted("baseline"))
    early_stopping = params.pop("early_stopping_rounds", 50)
    n_estimators = args.n_estimators or params.pop("n_estimators", 500)
    params.pop("n_estimators", None)
    params["seed"] = cfg.dotted("project.random_seed")
    clip_min = cfg.dotted("predict.clip_min")

    results: list[dict] = []
    models_dir = resolve_path(cfg, "paths.models", mkdir=True)

    for kind, source in TARGET_KINDS.items():
        for h in horizons:
            target = f"{kind}_h{h}"
            y_tr = y_train[target]
            y_va = y_valid[target]

            # --- Seasonal Naive, both variants, on the SAME validation rows
            for variant, season in baseline.seasonal_lag_steps(cfg).items():
                column = baseline.lag_column_for(source, season, h)
                prediction = np.clip(baseline_values[column], 0, None)
                metrics = all_metrics(y_va, prediction)
                metrics["pred_min"] = float(prediction.min())
                metrics["pred_max"] = float(prediction.max())
                results.append({"model": f"seasonal_naive_{variant}", "target": target,
                                "horizon": h, "split": "validate", **metrics,
                                **hotspot_f1(y_va, prediction, valid_ts)})

            # --- LightGBM. No boolean indexing here: horizon-overrun rows were
            # dropped up front, so the design matrix is reused as-is across all
            # eight models instead of being copied eight times.
            t0 = time.perf_counter()
            train_set = lgb.Dataset(x_train, label=y_tr,
                                    categorical_feature=[0], free_raw_data=False)
            valid_set = lgb.Dataset(x_valid, label=y_va,
                                    reference=train_set, free_raw_data=False)
            booster = lgb.train(
                params, train_set, num_boost_round=n_estimators,
                valid_sets=[valid_set], valid_names=["validate"],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                           lgb.log_evaluation(0)],
            )
            prediction = np.clip(booster.predict(x_valid), clip_min, None)
            metrics = all_metrics(y_va, prediction)
            metrics["pred_min"] = float(prediction.min())
            metrics["pred_max"] = float(prediction.max())
            results.append({"model": "lightgbm", "target": target, "horizon": h,
                            "split": "validate", **metrics,
                            **hotspot_f1(y_va, prediction, valid_ts),
                            "best_iteration": booster.best_iteration,
                            "train_seconds": round(time.perf_counter() - t0, 1)})
            booster.save_model(str(models_dir / f"lgbm_{tag}_{target}.txt"),
                               num_iteration=booster.best_iteration)
            log.info("%-12s %-12s MAE=%.4f (naive best %.4f) iters=%d %.0fs",
                     "lightgbm", target, metrics["mae"],
                     min(r["mae"] for r in results
                         if r["target"] == target and r["model"].startswith("seasonal")),
                     booster.best_iteration, time.perf_counter() - t0)
            del train_set, valid_set, booster
            gc.collect()

    frame = pl.DataFrame(results)
    metrics_dir = resolve_path(cfg, "paths.metrics", mkdir=True)
    suffix = "_dev" if args.dev_sample else ""
    frame.write_parquet(metrics_dir / f"phase3_baseline_{tag}{suffix}.parquet")
    (metrics_dir / f"phase3_baseline_{tag}{suffix}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    # --- the gate
    log.info("")
    log.info("%-14s %-12s %8s %8s %8s %8s", "MODEL", "TARGET", "MAE", "RMSE", "WAPE", "HotF1")
    naive_variants = list(baseline.seasonal_lag_steps(cfg))
    best_variant_mae = {v: float(np.mean([r["mae"] for r in results
                                          if r["model"] == f"seasonal_naive_{v}"]))
                        for v in naive_variants}
    best_variant = min(best_variant_mae, key=best_variant_mae.get)
    log.info("seasonal naive variants (mean MAE over all targets): %s -> best: %s",
             {k: round(v, 4) for k, v in best_variant_mae.items()}, best_variant)

    wins, losses = [], []
    for kind in TARGET_KINDS:
        for h in horizons:
            target = f"{kind}_h{h}"
            rows = {r["model"]: r for r in results if r["target"] == target}
            naive = rows[f"seasonal_naive_{best_variant}"]
            model = rows["lightgbm"]
            for name, row in (("naive_" + best_variant, naive), ("lightgbm", model)):
                log.info("%-14s %-12s %8.4f %8.4f %8.4f %8.4f", name, target,
                         row["mae"], row["rmse"], row["wape"], row["hotspot_f1"])
            (wins if model["mae"] < naive["mae"] else losses).append(
                (target, model["mae"], naive["mae"]))

    log.info("")
    log.info("GATE: LightGBM beats Seasonal Naive (%s) on validation MAE for %d/%d targets",
             best_variant, len(wins), len(wins) + len(losses))
    for target, model_mae, naive_mae in wins:
        log.info("  WIN  %-12s %.4f < %.4f  (%.1f%% better)", target, model_mae, naive_mae,
                 100 * (naive_mae - model_mae) / naive_mae)
    for target, model_mae, naive_mae in losses:
        log.error("  LOSS %-12s %.4f >= %.4f", target, model_mae, naive_mae)

    summary = {"tag": tag, "best_naive_variant": best_variant,
               "naive_variant_mean_mae": best_variant_mae,
               "targets_won": len(wins), "targets_total": len(wins) + len(losses),
               "train_rows": int(x_train.shape[0]), "validate_rows": int(x_valid.shape[0]),
               "train_sample_frac": sample_frac, "gate_passed": not losses}
    (metrics_dir / f"phase3_gate_{tag}{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log.info("metrics -> %s", metrics_dir / f"phase3_baseline_{tag}{suffix}.parquet")
    return 0 if not losses else 4


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
