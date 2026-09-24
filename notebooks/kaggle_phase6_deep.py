"""GeoPulse Phase 6 - TFT + ST-GNN on a Kaggle GPU.

Paste this into a Kaggle notebook cell (or run as a script) after attaching the
exported bundle as a dataset. It carries its own copies of the model definitions so
the notebook is self-contained - no repo checkout needed on Kaggle.

SETUP
-----
1. Everything needed is already assembled locally in `kaggle_upload/`:
       deep_bundle_h38/   (Phase 5 winner: H3-8)
       deep_bundle_s213/  (matched S2 level 13)
       deep_bundle_h39/   (optional - the resolution Phases 3-4 were built at)
       deep.py            (the model definitions, so Kaggle and local cannot diverge)
2. Upload `kaggle_upload/` as a Kaggle Dataset named `geopulse-deep-bundles`
3. Notebook settings: Accelerator = GPU T4 x2 or P100, Internet = off.
4. Paste this file into a cell and run. It trains 4 configs:
   {h38, s213} x {ST-GNN, TFT} - the deep half of the Phase 6 model matrix.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------- configuration
# Phase 5 chose H3-8 (best skill vs Seasonal Naive) and matched it to S2 level 13.
# Running both gives the {best-H3, matched-S2} x {TFT, ST-GNN} half of the Phase 6
# model matrix; the LightGBM half is trained locally.
KAGGLE_INPUT = Path("/kaggle/input")


def _find(pattern: str, is_dir: bool):
    roots = [KAGGLE_INPUT] if KAGGLE_INPUT.exists() else [Path(".")]
    hits = []
    for root in roots:
        for candidate in root.rglob(pattern):
            if candidate.is_dir() == is_dir:
                hits.append(candidate)
    return sorted(hits)


print("--- what is actually attached ---")
for path in sorted(KAGGLE_INPUT.glob("*")) if KAGGLE_INPUT.exists() else []:
    print(" ", path)
    for child in sorted(path.glob("*"))[:12]:
        print("   ", child.name)

BUNDLE_DIRS = {d.name.replace("deep_bundle_", ""): d
               for d in _find("deep_bundle_*", is_dir=True)}
print("bundles found:", {k: str(v) for k, v in BUNDLE_DIRS.items()})
BUNDLE_TAGS = [t for t in ("h38", "s213") if t in BUNDLE_DIRS] or list(BUNDLE_DIRS)
OUT_DIR = Path("/kaggle/working")
EPOCHS = 8
STGNN_BATCH = 8            # one sample = one timestamp = every region at once
TFT_BATCH = 512
BATCHES_PER_EPOCH = 500
SEASONAL_LAGS = [96, 672]  # same-time-yesterday, same-time-last-week
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_deep_candidates = _find("deep.py", is_dir=False)
print("deep.py candidates:", [str(c) for c in _deep_candidates])
if not _deep_candidates:
    raise SystemExit(
        "deep.py was not found anywhere under /kaggle/input. It must ship with the "
        "bundle so the model definitions match the repo exactly - re-implementing "
        "them here would let the Kaggle run and the local run silently diverge. "
        "Check the listing printed above: if the dataset shows the files but not "
        "deep.py, re-upload with 'deep.py' included (Kaggle can drop loose .py files "
        "if the dataset was created from a zip)."
    )
DEEP_SRC = _deep_candidates[0]
print(f"using model definitions from: {DEEP_SRC}")
exec(DEEP_SRC.read_text())  # noqa: S102 - trusted, self-authored source


def run(model_name: str, bundle_dir: Path) -> dict:
    bundle = Bundle.load(bundle_dir)  # noqa: F821 - from deep.py
    horizons = bundle.meta["horizons"]
    scaled = bundle.scaled_demand()
    weather = bundle.scaled_weather()
    static_np = bundle.scaled_static()
    static_t = torch.from_numpy(static_np).float().to(DEVICE)
    edges = torch.from_numpy(bundle.edges.astype(np.int64)).to(DEVICE)

    window = 24 if model_name == "stgnn" else 96
    batch_size = STGNN_BATCH if model_name == "stgnn" else TFT_BATCH
    seasonal_max = max(SEASONAL_LAGS) + window
    train_anchors = valid_window_starts(bundle, "train", window, seasonal_max)  # noqa: F821
    valid_anchors = valid_window_starts(bundle, "validate", window, seasonal_max)  # noqa: F821
    print(f"{model_name}: {len(train_anchors):,} train / {len(valid_anchors):,} valid anchors")

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    if model_name == "stgnn":
        n_dynamic = 2 + 2 * len(SEASONAL_LAGS) + bundle.calendar.shape[1] + weather.shape[1]
        model = STGNN(n_dynamic, static_np.shape[1], len(horizons)).to(DEVICE)  # noqa: F821
    else:
        model = CompactTFT(2 + weather.shape[1], bundle.calendar.shape[1],  # noqa: F821
                           static_np.shape[1], len(horizons)).to(DEVICE)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    def make_batch(anchors, train: bool):
        offsets = np.arange(-window + 1, 1)
        idx = anchors[:, None] + offsets[None, :]
        if model_name == "stgnn":
            demand = scaled[idx]
            n = demand.shape[2]
            calendar = bundle.calendar[idx][:, :, None, :].repeat(n, axis=2)
            wx = weather[idx][:, :, None, :].repeat(n, axis=2)
            seasonal = [scaled[idx - lag] for lag in SEASONAL_LAGS]
            dynamic = np.concatenate([demand, *seasonal, calendar, wx], axis=-1)
            targets = np.stack([bundle.demand[anchors + h] for h in horizons], axis=2)
            return ((torch.from_numpy(dynamic).float().to(DEVICE),),
                    torch.from_numpy(targets.astype(np.float32)).to(DEVICE))
        regions = rng.integers(0, bundle.n_regions, size=len(anchors))
        observed = np.concatenate([scaled[idx, regions[:, None]], weather[idx]], axis=-1)
        known = bundle.calendar[idx]
        targets = np.stack([bundle.demand[anchors + h, regions] for h in horizons], axis=1)
        return ((torch.from_numpy(observed).float().to(DEVICE),
                 torch.from_numpy(known).float().to(DEVICE),
                 torch.from_numpy(static_np[regions]).float().to(DEVICE)),
                torch.from_numpy(targets.astype(np.float32)).to(DEVICE))

    def forward(inputs):
        return (model(inputs[0], static_t, edges) if model_name == "stgnn"
                else model(*inputs))

    best, history = float("inf"), []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        started, losses = time.perf_counter(), []
        for _ in range(BATCHES_PER_EPOCH):
            anchors = rng.choice(train_anchors, size=batch_size, replace=False)
            inputs, targets = make_batch(anchors, True)
            optimiser.zero_grad()
            loss = poisson_nll(forward(inputs), targets)  # noqa: F821
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))

        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            # Spread evaluation across the WHOLE validation period. The split starts
            # at midnight on 1 September, so a contiguous prefix would score the model
            # almost entirely on empty overnight bins.
            stride = max(1, len(valid_anchors) // (120 * batch_size))
            eval_anchors = valid_anchors[::stride][:120 * batch_size]
            for i in range(0, len(eval_anchors), batch_size):
                anchors = eval_anchors[i:i + batch_size]
                if len(anchors) == 0:
                    break
                inputs, targets = make_batch(anchors, False)
                preds.append(forward(inputs).cpu().numpy().reshape(-1, len(horizons), 2))
                actuals.append(targets.cpu().numpy().reshape(-1, len(horizons), 2))
        prediction, actual = np.concatenate(preds), np.concatenate(actuals)
        mae = float(np.abs(prediction - actual).mean())
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "valid_mae": mae, "seconds": round(time.perf_counter() - started, 1)})
        print(f"  epoch {epoch}/{EPOCHS} loss={history[-1]['loss']:.4f} "
              f"MAE={mae:.4f} ({history[-1]['seconds']:.0f}s)")
        if mae < best:
            best = mae
            torch.save(model.state_dict(), OUT_DIR / f"{model_name}_{bundle.meta['tag']}.pt")

    result = {"model": model_name, "tag": bundle.meta["tag"], "device": DEVICE,
              "best_valid_mae": best, "history": history}
    (OUT_DIR / f"deep_metrics_{model_name}_{bundle.meta['tag']}.json").write_text(
        json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    if DEVICE == "cuda":
        print(torch.cuda.get_device_name(0),
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    summary = []
    for tag in BUNDLE_TAGS:
        bundle_dir = BUNDLE_DIRS.get(tag)
        if bundle_dir is None or not bundle_dir.exists():
            print(f"skipping {tag}: no deep_bundle_{tag} directory found")
            continue
        for name in ("stgnn", "tft"):
            print(f"\n=== {name} @ {tag} ===")
            result = run(name, bundle_dir)
            summary.append(result)
            print(f"  best valid MAE: {result['best_valid_mae']:.4f}")

    (OUT_DIR / "deep_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for row in summary:
        print(f"  {row['model']:6s} {row['tag']:6s} valid MAE {row['best_valid_mae']:.4f}")
    print(f"\nDownload from {OUT_DIR}: *.pt and deep_metrics_*.json / deep_summary.json")
