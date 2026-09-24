
from __future__ import annotations

import argparse
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
from src.features.basic import feature_columns as basic_feature_columns  # noqa: E402
from src.models import baseline  # noqa: E402
from src.models.splits import load_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, timed  # noqa: E402

TARGET_KINDS = {"pickup": "pickups", "dropoff": "dropoffs"}


def load_test(path: Path, cfg, columns: list[str], frac: float, seed: int) -> pl.DataFrame:
    split = load_splits(cfg)["test"]
    tz = cfg.dotted("time.timezone")
    scan = pl.scan_parquet(str(path / "**" / "*.parquet"))
    local_date = pl.col("ts").dt.convert_time_zone(tz).dt.date()
    scan = scan.filter((local_date >= split.start) & (local_date <= split.end))
    scan = scan.filter(~pl.col("dst_unreliable"))
    if frac < 1.0:
        scan = scan.filter((pl.col("ts").hash(seed) % 10_000) < int(round(frac * 10_000)))
    return scan.select(sorted(set(columns))).collect(engine="streaming")


def breakdowns(frame: pl.DataFrame, actual: np.ndarray, prediction: np.ndarray,
               cfg) -> list[dict]:
    """Where the model fails, not just how much on average."""
    tz = cfg.dotted("time.timezone")
    # only `ts` is genuinely required; `is_raining` rides along when the table has it
    carried = ["ts"] + (["is_raining"] if "is_raining" in frame.columns else [])
    work = frame.select(carried).with_columns([
        pl.Series("abs_error", np.abs(actual - prediction)),
        pl.Series("actual", actual),
        pl.col("ts").dt.convert_time_zone(tz).dt.hour().alias("hour"),
        (pl.col("ts").dt.convert_time_zone(tz).dt.weekday() >= 6).alias("is_weekend"),
    ])
    rows = []
    groups = [("hour_of_day", "hour"), ("weekend", "is_weekend")]
    if "is_raining" in work.columns:
        groups.append(("rain", "is_raining"))
    for name, expr in groups:
        grouped = work.group_by(expr).agg([
            pl.col("abs_error").mean().alias("mae"),
            pl.col("actual").mean().alias("actual_mean"),
            pl.len().alias("n"),
        ]).sort(expr)
        for row in grouped.to_dicts():
            rows.append({"breakdown": name, "bucket": str(row[expr]),
                         "mae": row["mae"], "actual_mean": row["actual_mean"],
                         "n": row["n"]})
    # demand terciles by the region's own average level
    terciles = work.with_columns(
        pl.col("actual").qcut(3, labels=["low", "mid", "high"], allow_duplicates=True)
        .alias("demand_band")
    ).group_by("demand_band").agg([
        pl.col("abs_error").mean().alias("mae"),
        pl.col("actual").mean().alias("actual_mean"),
        pl.len().alias("n"),
    ])
    for row in terciles.to_dicts():
        rows.append({"breakdown": "demand_band", "bucket": str(row["demand_band"]),
                     "mae": row["mae"], "actual_mean": row["actual_mean"], "n": row["n"]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", default="h3")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--test-frac", type=float, default=1.0)
    parser.add_argument("--top-errors", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.spatial, "lightgbm")
    if args.resolution is not None:
        key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: args.resolution,
                                         "resolution": args.resolution}})
    log = get_logger("phase7", cfg)
    from src.spatial.h3_indexer import make_indexer

    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"
    horizons = cfg.dotted("time.horizons")
    reported = cfg.dotted("time.reported_horizons")
    seed = cfg.dotted("train.sample_seed")
    models_dir = resolve_path(cfg, "paths.models")
    metrics_dir = resolve_path(cfg, "paths.metrics", mkdir=True)

    log.warning("OPENING THE TEST SPLIT (Nov-Dec 2024). This is the only script that "
                "reads it; every earlier decision was made on validation.")

    families = cfg.dotted("advanced_features.final_families")
    final_features = cumulative_feature_sets(cfg)[families[-1]]
    basic_features = basic_feature_columns(cfg)
    targets = [f"{kind}_h{h}" for kind in TARGET_KINDS for h in horizons]
    naive_cols = [
        baseline.lag_column_for(source, season, h)
        for source in TARGET_KINDS.values()
        for season in baseline.seasonal_lag_steps(cfg).values() for h in horizons
    ]

    results, breakdown_rows, error_rows = [], [], []

    # ---------------- feature-table models (Seasonal Naive, both LightGBMs)
    features4 = resolve_path(cfg, "paths.processed") / f"features4_{tag}"
    features_basic = resolve_path(cfg, "paths.processed") / f"features_{tag}"

    if features4.exists():
        # `pickups` and `is_raining` are not model inputs - they are carried purely
        # so the error breakdown can split by demand level and by weather
        needed = ["region_id", "ts", "dst_unreliable", *final_features, *targets,
                  "pickups", "is_raining"]
        with timed(log, "load TEST (final feature set)"):
            test = load_test(features4, cfg, needed, args.test_frac, seed)
            test = test.drop_nulls(subset=targets)
        log.info("TEST rows: %s", f"{test.height:,}")
        regions = sorted(test["region_id"].unique().to_list())
        mapping = {r: i for i, r in enumerate(regions)}
        inputs = ["region_idx"] + [c for c in final_features if c != "region_id"]
        x_test = (test.with_columns(
            pl.col("region_id").replace_strict(mapping).alias("region_idx"))
            .select(inputs).to_numpy().astype("float32", copy=False))
        test_ts = test["ts"].to_numpy()

        for target in targets:
            actual = test[target].to_numpy().astype("float64")
            path = models_dir / f"lgbm_final_{tag}_{target}.txt"
            if not path.exists():
                continue
            booster = lgb.Booster(model_file=str(path))
            t0 = time.perf_counter()
            prediction = np.clip(booster.predict(x_test), 0, None)
            infer = time.perf_counter() - t0
            results.append({"model": "lightgbm_final", "target": target,
                            "horizon": int(target.split("_h")[1]), "split": "test",
                            **all_metrics(actual, prediction),
                            **hotspot_f1(actual, prediction, test_ts),
                            "infer_seconds": round(infer, 2),
                            "artifact_mb": round(path.stat().st_size / 1e6, 2)})
            if target == "pickup_h1":
                breakdown_rows += [{**r, "model": "lightgbm_final", "target": target}
                                   for r in breakdowns(test, actual, prediction, cfg)]
                order = np.argsort(-np.abs(actual - prediction))[:args.top_errors]
                tz = cfg.dotted("time.timezone")
                local = test["ts"].dt.convert_time_zone(tz)
                for i in order:
                    error_rows.append({
                        "model": "lightgbm_final", "target": target,
                        "region_id": test["region_id"][int(i)],
                        "ts_local": str(local[int(i)]),
                        "hour": int(local.dt.hour()[int(i)]),
                        "weekday": int(local.dt.weekday()[int(i)]),
                        "actual": float(actual[i]), "predicted": float(prediction[i]),
                        "abs_error": float(abs(actual[i] - prediction[i])),
                    })
            log.info("lightgbm_final %-12s MAE=%.4f WAPE=%.4f HotF1=%.4f",
                     target, results[-1]["mae"], results[-1]["wape"],
                     results[-1]["hotspot_f1"])
            del booster

        # Seasonal Naive on the same TEST rows
        naive_needed = ["region_id", "ts", "dst_unreliable", *targets, *naive_cols]
        naive_test = load_test(features_basic, cfg, naive_needed, args.test_frac, seed) \
            if features_basic.exists() else None
        if naive_test is not None:
            naive_test = naive_test.drop_nulls(subset=targets)
            naive_ts = naive_test["ts"].to_numpy()
            for kind, source in TARGET_KINDS.items():
                for h in horizons:
                    target = f"{kind}_h{h}"
                    actual = naive_test[target].to_numpy().astype("float64")
                    for variant, season in baseline.seasonal_lag_steps(cfg).items():
                        column = baseline.lag_column_for(source, season, h)
                        prediction = np.clip(
                            naive_test[column].to_numpy().astype("float64"), 0, None)
                        results.append({"model": f"seasonal_naive_{variant}",
                                        "target": target, "horizon": h, "split": "test",
                                        **all_metrics(actual, prediction),
                                        **hotspot_f1(actual, prediction, naive_ts),
                                        "infer_seconds": 0.0, "artifact_mb": 0.0})

        # Phase 3 basic LightGBM, for the "did the extra features pay on TEST too" line
        if features_basic.exists():
            basic_needed = ["region_id", "ts", "dst_unreliable", *basic_features, *targets]
            basic_test = load_test(features_basic, cfg, basic_needed, args.test_frac, seed)
            basic_test = basic_test.drop_nulls(subset=targets)
            b_regions = sorted(basic_test["region_id"].unique().to_list())
            b_map = {r: i for i, r in enumerate(b_regions)}
            b_inputs = ["region_idx"] + [c for c in basic_features if c != "region_id"]
            x_basic = (basic_test.with_columns(
                pl.col("region_id").replace_strict(b_map).alias("region_idx"))
                .select(b_inputs).to_numpy().astype("float32", copy=False))
            b_ts = basic_test["ts"].to_numpy()
            for target in targets:
                path = models_dir / f"lgbm_{tag}_{target}.txt"
                if not path.exists():
                    continue
                booster = lgb.Booster(model_file=str(path))
                actual = basic_test[target].to_numpy().astype("float64")
                t0 = time.perf_counter()
                prediction = np.clip(booster.predict(x_basic), 0, None)
                results.append({"model": "lightgbm_basic", "target": target,
                                "horizon": int(target.split("_h")[1]), "split": "test",
                                **all_metrics(actual, prediction),
                                **hotspot_f1(actual, prediction, b_ts),
                                "infer_seconds": round(time.perf_counter() - t0, 2),
                                "artifact_mb": round(path.stat().st_size / 1e6, 2)})
                del booster
            del x_basic, basic_test

    # ---------------- deep models, scored from the tensor bundle
    bundle_dir = resolve_path(cfg, "paths.processed") / f"deep_bundle_{tag}"
    if bundle_dir.exists():
        import torch

        from src.models.deep import Bundle, CompactTFT, STGNN, valid_window_starts

        bundle = Bundle.load(bundle_dir)
        scaled, weather = bundle.scaled_demand(), bundle.scaled_weather()
        static = bundle.scaled_static()
        edges = torch.from_numpy(bundle.edges.astype(np.int64))
        seasonal_lags = [96, 672]
        for name in ("stgnn", "tft"):
            path = models_dir / f"{name}_{tag}.pt"
            if not path.exists():
                log.info("no %s checkpoint - skipping", name)
                continue
            window = 24 if name == "stgnn" else 96
            anchors = valid_window_starts(bundle, "test", window,
                                          max(seasonal_lags) + window)
            if anchors.size == 0:
                continue
            anchors = anchors[::max(1, len(anchors) // 600)][:600]
            if name == "stgnn":
                n_dyn = 2 + 2 * len(seasonal_lags) + bundle.calendar.shape[1] + weather.shape[1]
                model = STGNN(n_dyn, static.shape[1], len(horizons))
            else:
                model = CompactTFT(2 + weather.shape[1], bundle.calendar.shape[1],
                                   static.shape[1], len(horizons))
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            offsets = np.arange(-window + 1, 1)
            preds, actuals, group = [], [], []
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            with torch.no_grad():
                for i in range(0, len(anchors), 8 if name == "stgnn" else 256):
                    chunk = anchors[i:i + (8 if name == "stgnn" else 256)]
                    idx = chunk[:, None] + offsets[None, :]
                    if name == "stgnn":
                        demand = scaled[idx]
                        n = demand.shape[2]
                        cal = bundle.calendar[idx][:, :, None, :].repeat(n, axis=2)
                        wx = weather[idx][:, :, None, :].repeat(n, axis=2)
                        seas = [scaled[idx - lag] for lag in seasonal_lags]
                        dyn = np.concatenate([demand, *seas, cal, wx], axis=-1)
                        out = model(torch.from_numpy(dyn).float(),
                                    torch.from_numpy(static).float(), edges).numpy()
                        tgt = np.stack([bundle.demand[chunk + h] for h in horizons], axis=2)
                        preds.append(out.reshape(-1, len(horizons), 2))
                        actuals.append(tgt.reshape(-1, len(horizons), 2))
                        group.append(np.repeat(chunk, n))
                    else:
                        reg = rng.integers(0, bundle.n_regions, size=len(chunk))
                        obs = np.concatenate([scaled[idx, reg[:, None]], weather[idx]], axis=-1)
                        out = model(torch.from_numpy(obs).float(),
                                    torch.from_numpy(bundle.calendar[idx]).float(),
                                    torch.from_numpy(static[reg]).float()).numpy()
                        tgt = np.stack([bundle.demand[chunk + h, reg] for h in horizons], axis=1)
                        preds.append(out)
                        actuals.append(tgt)
                        group.append(chunk)
            infer = time.perf_counter() - t0
            prediction = np.concatenate(preds)
            actual = np.concatenate(actuals).astype("float64")
            groups = np.concatenate(group)
            for i, h in enumerate(horizons):
                for j, kind in enumerate(TARGET_KINDS):
                    entry = {"model": name, "target": f"{kind}_h{h}", "horizon": h,
                             "split": "test",
                             **all_metrics(actual[:, i, j], prediction[:, i, j]),
                             "infer_seconds": round(infer, 2),
                             "artifact_mb": round(path.stat().st_size / 1e6, 2)}
                    if name == "stgnn":
                        entry.update(hotspot_f1(actual[:, i, j], prediction[:, i, j], groups))
                    results.append(entry)
            log.info("%s scored on %s test anchors", name, f"{len(anchors):,}")

    if not results:
        log.error("no models found to evaluate")
        return 1

    frame = pl.DataFrame(results, infer_schema_length=None)
    frame.write_parquet(metrics_dir / f"phase7_test_{tag}.parquet")
    (metrics_dir / f"phase7_test_{tag}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    if breakdown_rows:
        pl.DataFrame(breakdown_rows).write_parquet(
            metrics_dir / f"phase7_breakdowns_{tag}.parquet")
    if error_rows:
        pl.DataFrame(error_rows).write_parquet(
            metrics_dir / f"phase7_top_errors_{tag}.parquet")

    # ---------------- the comparison matrix
    log.info("")
    log.info("MODEL COMPARISON ON TEST (%s)", tag)
    log.info("%-22s %-11s %8s %8s %8s %9s", "model", "target", "MAE", "RMSE", "WAPE", "HotF1")
    for h in reported:
        for kind in TARGET_KINDS:
            target = f"{kind}_h{h}"
            for row in sorted([r for r in results if r["target"] == target],
                              key=lambda r: r["mae"]):
                log.info("%-22s %-11s %8.4f %8.4f %8.4f %9s", row["model"], target,
                         row["mae"], row["rmse"], row["wape"],
                         f"{row['hotspot_f1']:.4f}" if "hotspot_f1" in row else "-")
        log.info("")

    log.info("metrics -> %s", metrics_dir / f"phase7_test_{tag}.parquet")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
