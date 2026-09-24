
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.basic import feature_columns, target_columns  # noqa: E402
from src.models.baseline import seasonal_lag_steps  # noqa: E402
from src.models.splits import load_splits, validate_splits  # noqa: E402
from src.utils.config import Config, load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class Check:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def run(self, item: str, fn) -> None:
        try:
            passed, evidence = fn()
        except Exception as exc:  # noqa: BLE001
            passed, evidence = False, f"{type(exc).__name__}: {exc}"
        self.results.append((item, bool(passed), evidence))

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config("h3", "lightgbm")
    if args.resolution is not None:
        cfg = Config({**cfg, "spatial": {**cfg["spatial"], "resolution": args.resolution}})
    log = get_logger("phase3-check", cfg)
    tag = f"h3{cfg.dotted('spatial.resolution')}"
    metrics_dir = resolve_path(cfg, "paths.metrics")
    models_dir = resolve_path(cfg, "paths.models")
    check = Check()

    full_metrics = metrics_dir / f"phase3_baseline_{tag}.json"
    dev_metrics = metrics_dir / f"phase3_baseline_{tag}_dev.json"
    gate_file = metrics_dir / f"phase3_gate_{tag}.json"
    if not full_metrics.exists():
        log.error("missing %s - run scripts/13_train_baseline.py", full_metrics)
        return 1
    results = json.loads(full_metrics.read_text())
    gate = json.loads(gate_file.read_text()) if gate_file.exists() else {}

    horizons = cfg.dotted("time.horizons")
    targets = [f"{kind}_h{h}" for kind in ("pickup", "dropoff") for h in horizons]

    def naive_reported():
        variants = set(seasonal_lag_steps(cfg))
        present = {r["model"].replace("seasonal_naive_", "")
                   for r in results if r["model"].startswith("seasonal_naive_")}
        covered = {
            v: sorted({r["target"] for r in results if r["model"] == f"seasonal_naive_{v}"})
            for v in variants
        }
        complete = all(set(covered[v]) == set(targets) for v in variants)
        means = gate.get("naive_variant_mean_mae", {})
        return (present == variants and complete), (
            f"both variants scored on all {len(targets)} targets; mean MAE "
            f"{ {k: round(v, 4) for k, v in means.items()} }, best = "
            f"{gate.get('best_naive_variant')}"
        )

    check.run("Seasonal Naive computed for both variants, best flagged", naive_reported)

    def eight_models():
        rows = [r for r in results if r["model"] == "lightgbm"]
        saved = sorted(models_dir.glob(f"lgbm_{tag}_*.txt"))
        return (len(rows) == 8 and len(saved) == 8), (
            f"{len(rows)} LightGBM results, {len(saved)} model files saved "
            f"({', '.join(p.stem.replace(f'lgbm_{tag}_', '') for p in saved)})"
        )

    check.run("8 LightGBM models trained (pickup/dropoff x h1-h4)", eight_models)

    def clipped():
        negatives = [
            (r["model"], r["target"], r["pred_min"])
            for r in results if r.get("pred_min", 0) < 0
        ]
        return not negatives, (
            "no negative predictions from any model"
            if not negatives else f"negative predictions: {negatives}"
        )

    check.run("Predictions clipped >= 0", clipped)

    def beats_naive():
        best = gate.get("best_naive_variant")
        wins, losses = [], []
        for target in targets:
            rows = {r["model"]: r for r in results if r["target"] == target}
            naive = rows.get(f"seasonal_naive_{best}")
            model = rows.get("lightgbm")
            if naive is None or model is None:
                losses.append(f"{target}: missing result")
                continue
            (wins if model["mae"] < naive["mae"] else losses).append(target)
        improvements = [
            100 * (
                next(r for r in results if r["target"] == t and r["model"] == f"seasonal_naive_{best}")["mae"]
                - next(r for r in results if r["target"] == t and r["model"] == "lightgbm")["mae"]
            ) / next(r for r in results if r["target"] == t and r["model"] == f"seasonal_naive_{best}")["mae"]
            for t in targets
        ]
        return not losses, (
            f"LightGBM beats seasonal_naive_{best} on {len(wins)}/{len(targets)} targets; "
            f"MAE improvement {min(improvements):.1f}%..{max(improvements):.1f}%"
            if not losses else f"LOSSES: {losses}"
        )

    check.run("LightGBM beats Seasonal Naive on validation MAE", beats_naive)

    def splits_sound():
        problems = validate_splits(cfg)
        splits = load_splits(cfg)
        return not problems, " | ".join(
            f"{s.name} {s.start}..{s.end} ({s.days}d)" for s in splits.values()
        )

    check.run("Chronological splits are contiguous and non-overlapping", splits_sound)

    def test_untouched():
        """No Phase 3 artifact may contain a metric computed on the TEST split."""
        offenders = [r for r in results if r.get("split") == "test"]
        return not offenders, (
            "no TEST-split metrics recorded - test opens once, in Phase 7"
            if not offenders else f"{len(offenders)} rows scored on TEST"
        )

    check.run("TEST split untouched", test_untouched)

    def leakage_tests():
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_features.py", "-q", "--no-header"],
            cwd=REPO, capture_output=True, text=True,
        )
        tail = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        return proc.returncode == 0, (tail[-1] if tail else "no output")

    check.run("Leakage tests pass (every lag re-derived by timestamp join)", leakage_tests)

    def all_tests():
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
            cwd=REPO, capture_output=True, text=True,
        )
        tail = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        return proc.returncode == 0, (tail[-1] if tail else "no output")

    check.run("Full data-validation test suite passes", all_tests)

    def dev_and_full():
        if not dev_metrics.exists():
            return False, "dev-sample run missing - verify on 7 days before two years"
        dev = json.loads(dev_metrics.read_text())
        dev_models = len({r["target"] for r in dev if r["model"] == "lightgbm"})
        return dev_models == 8, (
            f"dev run scored {dev_models}/8 targets; full run trained on "
            f"{gate.get('train_rows', 0):,} rows, validated on "
            f"{gate.get('validate_rows', 0):,}"
        )

    check.run("Verified on dev sample AND the full two-year data", dev_and_full)

    def features_sane():
        inputs = feature_columns(cfg)
        overlap = set(inputs) & set(target_columns(cfg))
        return not overlap, (
            f"{len(inputs)} model inputs, no overlap with the {len(target_columns(cfg))} targets"
        )

    check.run("Model inputs contain no target columns", features_sane)

    width = max(len(item) for item, _, _ in check.results)
    log.info("")
    log.info("PHASE 3 - DEFINITION OF DONE  (%s)   [STOP-AND-VERIFY GATE]", tag)
    log.info("=" * (width + 70))
    for item, passed, evidence in check.results:
        log.info("[%s] %-*s  %s", "PASS" if passed else "FAIL", width, item, evidence)
    log.info("=" * (width + 70))
    log.info("%d/%d checks passed - gate %s",
             sum(p for _, p, _ in check.results), len(check.results),
             "PASSED" if check.ok else "FAILED")

    if check.ok:
        table = pl.DataFrame(results).filter(pl.col("model") == "lightgbm").select(
            ["target", "mae", "rmse", "wape", "hotspot_f1", "best_iteration"]
        )
        log.info("")
        log.info("LightGBM validation metrics:\n%s", table)
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
