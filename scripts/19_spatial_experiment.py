
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import all_metrics, hotspot_f1  # noqa: E402
from src.features.basic import feature_columns  # noqa: E402
from src.models import baseline  # noqa: E402
from src.models.splits import load_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TARGETS = ["pickup_h1", "dropoff_h1"]


def run(script: str, *args: str) -> float:
    """Run a pipeline stage as a subprocess; return wall-clock seconds."""
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", str(REPO / "scripts" / script), *args],
        cwd=REPO, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
    return time.perf_counter() - started


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def train_and_score(cfg, tag: str, log, train_frac: float, valid_frac: float,
                    rounds: int) -> dict:
    """One identical LightGBM per h1 target. Same budget for every spatial config."""
    features_path = resolve_path(cfg, "paths.processed") / f"features_{tag}"
    splits = load_splits(cfg)
    model_features = feature_columns(cfg)
    naive_cols = [
        baseline.lag_column_for("pickups" if t.startswith("pickup") else "dropoffs",
                                baseline.seasonal_lag_steps(cfg)["same_week"],
                                int(t.partition("_h")[2]))
        for t in TARGETS
    ]
    tz = cfg.dotted("time.timezone")
    seed = cfg.dotted("train.sample_seed")

    def load(split, frac):
        scan = pl.scan_parquet(str(features_path / "**" / "*.parquet"))
        local_date = pl.col("ts").dt.convert_time_zone(tz).dt.date()
        scan = scan.filter((local_date >= split.start) & (local_date <= split.end))
        scan = scan.filter(~pl.col("dst_unreliable"))
        frame = scan.select(sorted(set(["ts", *model_features, *TARGETS, *naive_cols]))).collect(
            engine="streaming")
        if frac < 1.0:
            frame = frame.sample(fraction=frac, seed=seed, shuffle=False)
        return frame.drop_nulls(subset=TARGETS)

    t0 = time.perf_counter()
    train = load(splits["train"], train_frac)
    valid = load(splits["validate"], valid_frac)
    load_seconds = time.perf_counter() - t0

    regions = sorted(set(train["region_id"].unique()) | set(valid["region_id"].unique()))
    mapping = {r: i for i, r in enumerate(regions)}
    inputs = ["region_idx"] + [c for c in model_features if c != "region_id"]
    x_train = (train.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))
               .select(inputs).to_numpy().astype("float32", copy=False))
    x_valid = (valid.with_columns(pl.col("region_id").replace_strict(mapping).alias("region_idx"))
               .select(inputs).to_numpy().astype("float32", copy=False))
    y_train = {t: train[t].to_numpy().astype("float64") for t in TARGETS}
    y_valid = {t: valid[t].to_numpy().astype("float64") for t in TARGETS}
    valid_ts = valid["ts"].to_numpy()
    naive_values = {c: valid[c].to_numpy().astype("float64") for c in set(naive_cols)}
    del train, valid
    gc.collect()

    params = dict(cfg.dotted("baseline"))
    params.pop("early_stopping_rounds", None)
    params.pop("n_estimators", None)
    params["seed"] = cfg.dotted("project.random_seed")

    out: dict = {"train_rows": int(x_train.shape[0]), "valid_rows": int(x_valid.shape[0]),
                 "load_seconds": round(load_seconds, 1), "n_features": len(inputs)}
    train_seconds = infer_seconds = 0.0
    for target in TARGETS:
        t0 = time.perf_counter()
        train_set = lgb.Dataset(x_train, label=y_train[target],
                                categorical_feature=[0], free_raw_data=False)
        valid_set = lgb.Dataset(x_valid, label=y_valid[target], reference=train_set,
                                free_raw_data=False)
        booster = lgb.train(params, train_set, num_boost_round=rounds,
                            valid_sets=[valid_set], valid_names=["validate"],
                            callbacks=[lgb.early_stopping(30, verbose=False),
                                       lgb.log_evaluation(0)])
        train_seconds += time.perf_counter() - t0
        t1 = time.perf_counter()
        prediction = np.clip(booster.predict(x_valid), 0, None)
        infer_seconds += time.perf_counter() - t1
        metrics = all_metrics(y_valid[target], prediction)
        out[target] = {**metrics, **hotspot_f1(y_valid[target], prediction, valid_ts),
                       "best_iteration": booster.best_iteration}
        log.info("    %-11s MAE=%.4f RMSE=%.4f WAPE=%.4f HotF1=%.4f", target,
                 metrics["mae"], metrics["rmse"], metrics["wape"],
                 out[target]["hotspot_f1"])
        del train_set, valid_set, booster
        gc.collect()
    out["train_seconds"] = round(train_seconds, 1)
    out["infer_seconds"] = round(infer_seconds, 2)
    out["mean_mae"] = float(np.mean([out[t]["mae"] for t in TARGETS]))
    out["mean_wape"] = float(np.mean([out[t]["wape"] for t in TARGETS]))
    out["mean_hotspot_f1"] = float(np.mean([out[t]["hotspot_f1"] for t in TARGETS]))

    # Seasonal Naive at THIS resolution, on the same validation rows.
    #
    # This is what makes the resolutions comparable at all. No raw accuracy metric is
    # scale-invariant here: MAE falls automatically as cells shrink (smaller counts),
    # and WAPE falls automatically as cells grow (aggregation smooths relative error).
    # Taken to the limit each would crown a degenerate grid. A skill score - how much
    # better the model is than the trivial forecast *at the same resolution* - divides
    # the scale effect out, because both numerator and denominator live on that grid.
    naive_mae, naive_wape = [], []
    for target in TARGETS:
        kind, _, horizon = target.partition("_h")
        source = "pickups" if kind == "pickup" else "dropoffs"
        season = baseline.seasonal_lag_steps(cfg)["same_week"]
        column = baseline.lag_column_for(source, season, int(horizon))
        prediction = np.clip(naive_values[column], 0, None)
        metrics = all_metrics(y_valid[target], prediction)
        out[f"naive_{target}"] = metrics
        naive_mae.append(metrics["mae"])
        naive_wape.append(metrics["wape"])
    out["naive_mean_mae"] = float(np.mean(naive_mae))
    out["naive_mean_wape"] = float(np.mean(naive_wape))
    out["skill_vs_naive"] = float(1.0 - out["mean_mae"] / out["naive_mean_mae"])
    del x_train, x_valid
    gc.collect()
    return out


def build_config(cfg, log, overlay: str, resolution: int, tag: str,
                 rebuild: bool) -> dict:
    """Panel + basic features for one spatial config, with timings and sizes."""
    processed = resolve_path(cfg, "paths.processed")
    panel_path = processed / f"panel_{tag}"
    features_path = processed / f"features_{tag}"
    timings = {"panel_seconds": 0.0, "features_seconds": 0.0}

    if rebuild or not panel_path.exists():
        log.info("  building panel %s", tag)
        timings["panel_seconds"] = round(
            run("10_build_panel.py", "--spatial", overlay, "--resolution", str(resolution),
                "--memory-limit", "9GB"), 1)
    if rebuild or not features_path.exists():
        log.info("  building features %s", tag)
        timings["features_seconds"] = round(
            run("12_build_features.py", "--spatial", overlay, "--resolution", str(resolution),
                "--memory-limit", "9GB", "--batch-rows", "12000000"), 1)

    meta = pl.read_parquet(resolve_path(cfg, "paths.spatial") / f"regions_{tag}.parquet")
    panel_stats = pl.scan_parquet(str(panel_path / "**" / "*.parquet")).select([
        pl.len().alias("panel_rows"),
        (pl.col("pickups") == 0).mean().alias("zero_pickup_share"),
        pl.col("pickups").sum().alias("total_pickups"),
    ]).collect(engine="streaming").to_dicts()[0]

    return {
        "tag": tag, "system": overlay, "resolution": resolution,
        "active_regions": meta.height,
        "median_area_km2": float(meta["area_km2"].median()),
        "median_trips_per_region": float(meta["total_trips"].median()),
        "mean_trips_per_region": float(meta["total_trips"].mean()),
        "panel_rows": int(panel_stats["panel_rows"]),
        "zero_demand_pct": round(100 * float(panel_stats["zero_pickup_share"]), 2),
        "panel_mb": round(dir_size_mb(panel_path), 1),
        "features_mb": round(dir_size_mb(features_path), 1),
        **timings,
    }


def match_s2_level(cfg, h3_stats: dict, log) -> dict:
    """Pick the S2 level closest to the winning H3 resolution in granularity.

    Distance is a weighted sum of |log(ratio)| on median cell area and on active-cell
    count. Log-ratio keeps the choice scale-independent: a level twice as coarse and
    one twice as fine are equally far away, which a raw difference would not capture.
    """
    from src.spatial.s2_indexer import S2Indexer

    candidates = cfg.dotted("spatial.candidate_levels", [12, 13, 14, 15, 16])
    area_weight = cfg.dotted("matching.area_weight", 0.5)
    count_weight = cfg.dotted("matching.active_cell_count_weight", 0.5)

    trips = pl.read_parquet(
        resolve_path(cfg, "paths.interim") / "trips_clean.parquet",
        columns=["start_lat", "start_lng"],
    ).unique()
    rows = []
    for level in candidates:
        indexer = S2Indexer(level)
        cells = {indexer.index(lat, lng) for lat, lng in trips.iter_rows()}
        areas = [indexer.area_km2(c) for c in list(cells)[:2000]]
        median_area = float(np.median(areas))
        area_ratio = abs(math.log(median_area / h3_stats["median_area_km2"]))
        count_ratio = abs(math.log(len(cells) / h3_stats["touched_regions"]))
        score = area_weight * area_ratio + count_weight * count_ratio
        rows.append({"level": level, "touched_cells": len(cells),
                     "median_area_km2": round(median_area, 6),
                     "log_area_distance": round(area_ratio, 4),
                     "log_count_distance": round(count_ratio, 4),
                     "score": round(score, 4)})
        log.info("  S2 L%-2d cells=%-6d area=%.5f km2  logdist area=%.3f count=%.3f  score=%.4f",
                 level, len(cells), median_area, area_ratio, count_ratio, score)
    best = min(rows, key=lambda r: r["score"])
    log.info("  -> matched S2 level: %d (score %.4f)", best["level"], best["score"])
    return {"candidates": rows, "chosen_level": best["level"],
            "weights": {"area": area_weight, "active_cell_count": count_weight}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--resolutions", default="8,9,10")
    parser.add_argument("--s2-level", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.06)
    parser.add_argument("--valid-frac", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cfg = load_config("h3", "lightgbm")
    log = get_logger("phase5", cfg)
    experiments = resolve_path(cfg, "paths.experiments", mkdir=True)
    results_path = experiments / "phase5_spatial.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    if args.part in ("A", "all"):
        log.info("PART A - H3 resolution study (model and features held constant)")
        for resolution in [int(r) for r in args.resolutions.split(",")]:
            tag = f"h3{resolution}"
            log.info("  --- %s ---", tag)
            entry = build_config(cfg, log, "h3", resolution, tag, args.rebuild)
            run_cfg = Config({**load_config("h3", "lightgbm"),
                              "spatial": {**cfg["spatial"], "resolution": resolution}})
            entry.update(train_and_score(run_cfg, tag, log, args.train_frac,
                                         args.valid_frac, args.rounds))
            results.setdefault("h3", {})[str(resolution)] = entry
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            log.info("  %s: %d regions, %.2f%% zero-demand, mean MAE %.4f",
                     tag, entry["active_regions"], entry["zero_demand_pct"],
                     entry["mean_mae"])

        table = [results["h3"][k] for k in sorted(results["h3"], key=int)]
        for row in table:
            row["mean_wape"] = float(np.mean([row[t]["wape"] for t in TARGETS]))
        # MAE is NOT comparable across resolutions: coarser cells hold more demand, so
        # their absolute errors are mechanically larger and picking by MAE would always
        # crown the finest grid regardless of skill. WAPE is scale-free (error relative
        # to demand) and Hotspot-F1 is rank-based, so both survive the change of scale.
        best = max(table, key=lambda r: r["skill_vs_naive"])
        results["best_h3_resolution"] = best["resolution"]
        results["selection_criterion"] = (
            "skill vs Seasonal Naive at the same resolution (1 - model_MAE/naive_MAE). "
            "Raw MAE falls automatically as cells shrink and raw WAPE falls "
            "automatically as cells grow, so neither is comparable across "
            "resolutions; a skill score divides the scale effect out."
        )
        log.info("")
        log.info("%-6s %8s %10s %7s %9s %8s %9s %8s %8s", "res", "regions", "area_km2",
                 "zero%", "MAE(*)", "WAPE(*)", "naiveMAE", "SKILL", "HotF1")
        for row in table:
            log.info("%-6s %8d %10.5f %7.2f %9.4f %8.4f %9.4f %8.4f %8.4f",
                     row["tag"], row["active_regions"], row["median_area_km2"],
                     row["zero_demand_pct"], row["mean_mae"], row["mean_wape"],
                     row["naive_mean_mae"], row["skill_vs_naive"], row["mean_hotspot_f1"])
        log.info("  (*) raw MAE and WAPE are NOT comparable across resolutions - MAE "
                 "shrinks with cell size, WAPE grows with it. SKILL is.")
        log.info("  best H3 resolution by skill vs Seasonal Naive: %s "
                 "(skill %.4f, %.1f%% zero-demand, %d regions)",
                 best["tag"], best["skill_vs_naive"], best["zero_demand_pct"],
                 best["active_regions"])
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.part in ("B", "all"):
        log.info("PART B - S2 level matching")
        best_res = results.get("best_h3_resolution")
        if best_res is None:
            log.error("run part A first")
            return 1
        h3_entry = results["h3"][str(best_res)]
        activity = pl.read_parquet(
            resolve_path(cfg, "paths.spatial") / f"regions_h3{best_res}.parquet")
        h3_stats = {"median_area_km2": h3_entry["median_area_km2"],
                    "touched_regions": activity.height}
        cfg_s2 = load_config("s2")
        results["s2_matching"] = match_s2_level(cfg_s2, h3_stats, log)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.part in ("C", "all"):
        level = args.s2_level or results.get("s2_matching", {}).get("chosen_level")
        if level is None:
            log.error("no S2 level - run part B or pass --s2-level")
            return 1
        log.info("PART C - H3 vs S2 at matched granularity (S2 level %d)", level)
        tag = f"s2{level}"
        entry = build_config(cfg, log, "s2", level, tag, args.rebuild)
        run_cfg = Config({**load_config("s2", "lightgbm"),
                          "spatial": {**load_config("s2")["spatial"],
                                      "level": level, "resolution": level}})
        entry.update(train_and_score(run_cfg, tag, log, args.train_frac,
                                     args.valid_frac, args.rounds))
        results["s2"] = {str(level): entry}
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        best_res = results.get("best_h3_resolution")
        h3_entry = results["h3"][str(best_res)]
        log.info("")
        log.info("%-8s %8s %10s %8s %8s %8s %8s %8s %9s", "system", "regions",
                 "area_km2", "zero%", "MAE(*)", "WAPE", "HotF1", "train_s", "size_MB")
        for row in (h3_entry, entry):
            log.info("%-8s %8d %10.5f %8.2f %8.4f %8.4f %8.4f %8.0f %9.0f",
                     row["tag"], row["active_regions"], row["median_area_km2"],
                     row["zero_demand_pct"], row["mean_mae"], row["mean_wape"],
                     row["mean_hotspot_f1"], row["train_seconds"], row["features_mb"])
        for row in (h3_entry, entry):
            row.setdefault("mean_wape",
                           float(np.mean([row[t]["wape"] for t in TARGETS])))
        # matched granularity still is not identical granularity, so compare on the
        # scale-free metric here too
        delta = 100 * (entry["mean_wape"] - h3_entry["mean_wape"]) / h3_entry["mean_wape"]
        winner = "H3" if h3_entry["mean_wape"] <= entry["mean_wape"] else "S2"
        results["h3_vs_s2"] = {"winner": winner, "s2_wape_vs_h3_pct": round(delta, 3),
                               "h3_wape": h3_entry["mean_wape"],
                               "s2_wape": entry["mean_wape"]}
        log.info("  %s wins on WAPE (S2 is %+.2f%% vs H3)", winner, delta)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.info("results -> %s", results_path)
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
