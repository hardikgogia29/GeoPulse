
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
from src.features.advanced import cumulative_feature_sets  # noqa: E402
from src.models.splits import load_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402

TARGET_KINDS = {"pickup": "pickups", "dropoff": "dropoffs"}


def load(path: Path, cfg, split, columns, frac, seed):
    scan = pl.scan_parquet(str(path / "**" / "*.parquet"))
    tz = cfg.dotted("time.timezone")
    local_date = pl.col("ts").dt.convert_time_zone(tz).dt.date()
    scan = scan.filter((local_date >= split.start) & (local_date <= split.end))
    if cfg.dotted("train.exclude_dst_unreliable"):
        scan = scan.filter(~pl.col("dst_unreliable"))
    if frac is not None and frac < 1.0:
        # sample whole timestamps so each retained instant keeps its full region
        # cross-section, which Hotspot-F1 ranks within
        scan = scan.filter((pl.col("ts").hash(seed) % 10_000) < int(round(frac * 10_000)))
    return scan.select(sorted(set(columns))).collect(engine="streaming")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", default="h3")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--train-frac", type=float, default=0.20)
    parser.add_argument("--valid-frac", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--objective", default=None,
                        help="skip the objective bake-off and use this one")
    parser.add_argument("--params-json", default=None,
                        help="JSON of tuned params, to reuse a previous Optuna result")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse already-fitted target models; they are still scored")
    args = parser.parse_args()

    cfg = load_config(args.spatial, "lightgbm")
    if args.resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: args.resolution,
                                         "resolution": args.resolution}})
    log = get_logger("final", cfg)
    from src.spatial.h3_indexer import make_indexer

    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"

    features_path = resolve_path(cfg, "paths.processed") / f"features4_{tag}"
    if not features_path.exists():
        log.error("missing %s - run scripts/17_build_features_full.py", features_path)
        return 1

    families = cfg.dotted("advanced_features.final_families")
    feature_set = cumulative_feature_sets(cfg)[families[-1]]
    log.info("final feature set from the ablation: families %s -> %d features",
             "+".join(families), len(feature_set))

    splits = load_splits(cfg)
    horizons = cfg.dotted("time.horizons")
    targets = [f"{kind}_h{h}" for kind in TARGET_KINDS for h in horizons]
    needed = ["region_id", "ts", "dst_unreliable", *feature_set, *targets]
    seed = cfg.dotted("train.sample_seed")

    with timed(log, "load TRAIN"):
        train = load(features_path, cfg, splits["train"], needed, args.train_frac, seed)
        train = train.drop_nulls(subset=targets)
    with timed(log, "load VALIDATE"):
        valid = load(features_path, cfg, splits["validate"], needed, args.valid_frac, seed)
        valid = valid.drop_nulls(subset=targets)
    log.info("train %s rows | validate %s rows", f"{train.height:,}", f"{valid.height:,}")

    regions = sorted(set(train["region_id"].unique()) | set(valid["region_id"].unique()))
    mapping = {r: i for i, r in enumerate(regions)}
    inputs = ["region_idx"] + [c for c in feature_set if c != "region_id"]
    x_train = (train.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))
               .select(inputs).to_numpy().astype("float32", copy=False))
    x_valid = (valid.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))
               .select(inputs).to_numpy().astype("float32", copy=False))
    y_train = {t: train[t].to_numpy().astype("float64") for t in targets}
    y_valid = {t: valid[t].to_numpy().astype("float64") for t in targets}
    valid_ts = valid["ts"].to_numpy()
    del train, valid
    gc.collect()
    log.info("design matrix train %s x %d (%.2f GB)", f"{x_train.shape[0]:,}",
             x_train.shape[1], x_train.nbytes / 1e9)

    base = dict(cfg.dotted("baseline"))
    base.pop("early_stopping_rounds", None)
    base.pop("n_estimators", None)
    base["seed"] = cfg.dotted("project.random_seed")

    # ---- objective selection on validation, not by assumption
    probe = "pickup_h1"
    objective_scores = {}
    for objective in (() if args.objective else ("poisson", "regression_l1")):
        params = {**base, "objective": objective,
                  "metric": "mae" if objective == "regression_l1" else "mae"}
        booster = lgb.train(
            params, lgb.Dataset(x_train, label=y_train[probe], categorical_feature=[0],
                                free_raw_data=False),
            num_boost_round=300,
            valid_sets=[lgb.Dataset(x_valid, label=y_valid[probe], free_raw_data=False)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        mae = float(np.abs(np.clip(booster.predict(x_valid), 0, None) - y_valid[probe]).mean())
        objective_scores[objective] = mae
        log.info("objective %-14s validation MAE %.4f (%d iters)", objective, mae,
                 booster.best_iteration)
        del booster
        gc.collect()
    best_objective = args.objective or min(objective_scores, key=objective_scores.get)
    base["objective"] = best_objective
    log.info("keeping objective: %s%s", best_objective,
             " (supplied, bake-off skipped)" if args.objective else "")

    tuned = {}
    if args.params_json:
        tuned = json.loads(Path(args.params_json).read_text(encoding="utf-8")
                           if Path(args.params_json).exists() else args.params_json)
        base.update(tuned)
        log.info("reusing supplied tuned params (no search): %s", tuned)
    elif args.tune:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective_fn(trial):
            space = cfg.dotted("tuning.search_space")
            params = {**base,
                      "num_leaves": trial.suggest_int("num_leaves", *[space["num_leaves"][k] for k in ("low", "high")]),
                      "max_depth": trial.suggest_int("max_depth", *[space["max_depth"][k] for k in ("low", "high")]),
                      "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"]["low"], space["learning_rate"]["high"], log=True),
                      "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *[space["min_data_in_leaf"][k] for k in ("low", "high")]),
                      "feature_fraction": trial.suggest_float("feature_fraction", space["feature_fraction"]["low"], space["feature_fraction"]["high"]),
                      "bagging_fraction": trial.suggest_float("bagging_fraction", space["bagging_fraction"]["low"], space["bagging_fraction"]["high"]),
                      "lambda_l1": trial.suggest_float("lambda_l1", space["lambda_l1"]["low"], space["lambda_l1"]["high"]),
                      "lambda_l2": trial.suggest_float("lambda_l2", space["lambda_l2"]["low"], space["lambda_l2"]["high"])}
            booster = lgb.train(
                params, lgb.Dataset(x_train, label=y_train[probe], categorical_feature=[0],
                                    free_raw_data=False),
                num_boost_round=400,
                valid_sets=[lgb.Dataset(x_valid, label=y_valid[probe], free_raw_data=False)],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
            score = float(np.abs(np.clip(booster.predict(x_valid), 0, None) - y_valid[probe]).mean())
            del booster
            gc.collect()
            return score

        with timed(log, f"optuna search ({args.trials} trials on {probe})"):
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective_fn, n_trials=args.trials)
        tuned = study.best_params
        log.info("best trial MAE %.4f: %s", study.best_value, tuned)
        base.update(tuned)

    models_dir = resolve_path(cfg, "paths.models", mkdir=True)
    results = []
    for target in targets:
        t0 = time.perf_counter()
        model_path = models_dir / f"lgbm_final_{tag}_{target}.txt"
        reused = args.skip_existing and model_path.exists()
        if reused:
            # fitted by an earlier run - load and score it so the metrics file is
            # complete, rather than silently leaving a hole or refitting for 4 minutes
            booster = lgb.Booster(model_file=str(model_path))
        else:
            booster = lgb.train(
                base, lgb.Dataset(x_train, label=y_train[target], categorical_feature=[0],
                                  free_raw_data=False),
                num_boost_round=args.rounds,
                valid_sets=[lgb.Dataset(x_valid, label=y_valid[target], free_raw_data=False)],
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        prediction = np.clip(booster.predict(x_valid), 0, None)
        metrics = all_metrics(y_valid[target], prediction)
        results.append({"model": "lightgbm_final", "target": target, "split": "validate",
                        **metrics, **hotspot_f1(y_valid[target], prediction, valid_ts),
                        "best_iteration": (booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()),
                        "reused": reused,
                        "train_seconds": round(time.perf_counter() - t0, 1)})
        if not reused:
            booster.save_model(str(model_path), num_iteration=booster.best_iteration)
        importance = booster.feature_importance("gain")
        top = sorted(zip(inputs, importance), key=lambda kv: -kv[1])[:15]
        log.info("%-12s MAE=%.4f iters=%-4d %.0fs%s | top: %s", target, metrics["mae"],
                 (booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()),
                 results[-1]["train_seconds"], " (reused)" if reused else "",
                 ", ".join(name for name, _ in top[:5]))
        if target == "pickup_h1":
            (resolve_path(cfg, "paths.metrics", mkdir=True) /
             f"feature_importance_{tag}.json").write_text(
                json.dumps([{"feature": n, "gain": float(g)} for n, g in
                            sorted(zip(inputs, importance), key=lambda kv: -kv[1])],
                           indent=2), encoding="utf-8")
        del booster
        gc.collect()

    metrics_dir = resolve_path(cfg, "paths.metrics", mkdir=True)
    pl.DataFrame(results).write_parquet(metrics_dir / f"final_lightgbm_{tag}.parquet")
    (metrics_dir / f"final_lightgbm_{tag}.json").write_text(json.dumps({
        "tag": tag, "families": families, "n_features": len(inputs),
        "objective_scores": objective_scores, "objective": best_objective,
        "tuned_params": tuned, "train_rows": int(x_train.shape[0]),
        "valid_rows": int(x_valid.shape[0]), "results": results,
    }, indent=2), encoding="utf-8")
    log.info("mean validation MAE %.4f -> %s",
             float(np.mean([r["mae"] for r in results])),
             metrics_dir / f"final_lightgbm_{tag}.parquet")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
