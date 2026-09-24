
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import all_metrics  # noqa: E402
from src.models.deep import (  # noqa: E402
    Bundle, CompactTFT, STGNN, poisson_nll, valid_window_starts,
)
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

SEASONAL_LAGS = [96, 672]


def stgnn_batch(bundle: Bundle, scaled: np.ndarray, weather: np.ndarray,
                anchors: np.ndarray, window: int, device: str):
    """[B, T, N, F] dynamic tensor + [B, N, H, 2] targets."""
    horizons = bundle.meta["horizons"]
    offsets = np.arange(-window + 1, 1)
    idx = anchors[:, None] + offsets[None, :]                     # [B, T]
    demand = scaled[idx]                                          # [B, T, N, 2]
    b, t, n, _ = demand.shape
    calendar = bundle.calendar[idx][:, :, None, :].repeat(n, axis=2)
    wx = weather[idx][:, :, None, :].repeat(n, axis=2)
    seasonal = [scaled[idx - lag] for lag in SEASONAL_LAGS]       # each [B, T, N, 2]
    dynamic = np.concatenate([demand, *seasonal, calendar, wx], axis=-1)
    targets = np.stack(
        [bundle.demand[anchors + h] for h in horizons], axis=2
    ).astype(np.float32)                                          # [B, N, H, 2]
    return (torch.from_numpy(dynamic).float().to(device),
            torch.from_numpy(targets).to(device))


def tft_batch(bundle: Bundle, scaled: np.ndarray, weather: np.ndarray,
              static: np.ndarray, anchors: np.ndarray, regions: np.ndarray,
              window: int, device: str):
    horizons = bundle.meta["horizons"]
    offsets = np.arange(-window + 1, 1)
    idx = anchors[:, None] + offsets[None, :]                     # [B, T]
    demand = scaled[idx, regions[:, None]]                        # [B, T, 2]
    wx = weather[idx]                                             # [B, T, W]
    observed = np.concatenate([demand, wx], axis=-1)
    known = bundle.calendar[idx]                                  # [B, T, C]
    targets = np.stack(
        [bundle.demand[anchors + h, regions] for h in horizons], axis=1
    ).astype(np.float32)                                          # [B, H, 2]
    return (torch.from_numpy(observed).float().to(device),
            torch.from_numpy(known).float().to(device),
            torch.from_numpy(static[regions]).float().to(device),
            torch.from_numpy(targets).to(device))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["stgnn", "tft"], required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--tag", default="h39")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--batches-per-epoch", type=int, default=400)
    parser.add_argument("--valid-batches", type=int, default=120)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config("h3", "stgnn" if args.model == "stgnn" else "tft")
    log = get_logger(args.model, cfg)
    bundle_path = Path(args.bundle) if args.bundle else (
        resolve_path(cfg, "paths.processed") / f"deep_bundle_{args.tag}")
    if not bundle_path.exists():
        log.error("missing bundle at %s - run scripts/20_export_deep_bundle.py", bundle_path)
        return 1

    bundle = Bundle.load(bundle_path)
    horizons = bundle.meta["horizons"]
    device = args.device
    log.info("bundle %s: %s steps x %s regions | device=%s", bundle.meta["tag"],
             f"{bundle.n_steps:,}", f"{bundle.n_regions:,}", device)
    if device == "cuda":
        log.info("gpu: %s (%.1f GB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    scaled = bundle.scaled_demand()
    weather = bundle.scaled_weather()
    static = bundle.scaled_static()
    edges = torch.from_numpy(bundle.edges.astype(np.int64)).to(device)

    window = args.window or (cfg.dotted("temporal.window_steps") if args.model == "stgnn"
                             else cfg.dotted("model.encoder_length"))
    batch_size = args.batch_size or (8 if args.model == "stgnn" else 256)
    lr = args.lr or cfg.dotted("train.learning_rate")
    seasonal_max = max(SEASONAL_LAGS) + window

    train_anchors = valid_window_starts(bundle, "train", window, seasonal_max)
    valid_anchors = valid_window_starts(bundle, "validate", window, seasonal_max)
    log.info("anchors: train %s | validate %s (window=%d)",
             f"{len(train_anchors):,}", f"{len(valid_anchors):,}", window)
    if len(train_anchors) == 0 or len(valid_anchors) == 0:
        log.error("no usable windows")
        return 2

    rng = np.random.default_rng(cfg.dotted("project.random_seed"))
    torch.manual_seed(cfg.dotted("project.random_seed"))

    if args.model == "stgnn":
        n_dynamic = 2 + 2 * len(SEASONAL_LAGS) + bundle.calendar.shape[1] + weather.shape[1]
        model = STGNN(n_dynamic, static.shape[1], len(horizons),
                      hidden=cfg.dotted("model.gat_hidden", 32),
                      gru_hidden=cfg.dotted("model.gru_hidden", 64),
                      heads=cfg.dotted("model.gat_heads", 4)).to(device)
    else:
        model = CompactTFT(n_observed=2 + weather.shape[1],
                           n_known=bundle.calendar.shape[1],
                           n_static=static.shape[1], n_horizons=len(horizons),
                           hidden=cfg.dotted("model.hidden_size", 64),
                           heads=cfg.dotted("model.attention_head_size", 4)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("%s: %s parameters", args.model, f"{n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best_mae, best_state, history = float("inf"), None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses = []
        for _ in range(args.batches_per_epoch):
            anchors = rng.choice(train_anchors, size=batch_size, replace=False)
            optimiser.zero_grad()
            if args.model == "stgnn":
                dynamic, targets = stgnn_batch(bundle, scaled, weather, anchors,
                                               window, device)
                prediction = model(dynamic, torch.from_numpy(static).float().to(device),
                                   edges)
            else:
                regions = rng.integers(0, bundle.n_regions, size=batch_size)
                observed, known, stat, targets = tft_batch(
                    bundle, scaled, weather, static, anchors, regions, window, device)
                prediction = model(observed, known, stat)
            loss = poisson_nll(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))

        model.eval()
        preds, actuals = [], []
        # Evaluate on anchors spread ACROSS the whole validation period, not the first
        # N. The validation split starts at midnight on 1 September, so taking a
        # contiguous prefix scores the model almost entirely on empty overnight bins
        # and reports an MAE that has nothing to do with its real accuracy.
        stride = max(1, len(valid_anchors) // (args.valid_batches * batch_size))
        eval_anchors = valid_anchors[::stride][:args.valid_batches * batch_size]
        eval_rng = np.random.default_rng(cfg.dotted("project.random_seed"))
        with torch.no_grad():
            for i in range(0, len(eval_anchors), batch_size):
                anchors = eval_anchors[i:i + batch_size]
                if len(anchors) == 0:
                    break
                if args.model == "stgnn":
                    dynamic, targets = stgnn_batch(bundle, scaled, weather, anchors,
                                                   window, device)
                    prediction = model(dynamic,
                                       torch.from_numpy(static).float().to(device), edges)
                else:
                    # deterministic region choice so epochs are comparable
                    regions = eval_rng.integers(0, bundle.n_regions, size=len(anchors))
                    observed, known, stat, targets = tft_batch(
                        bundle, scaled, weather, static, anchors, regions, window, device)
                    prediction = model(observed, known, stat)
                preds.append(prediction.cpu().numpy().reshape(-1, len(horizons), 2))
                actuals.append(targets.cpu().numpy().reshape(-1, len(horizons), 2))
        prediction = np.concatenate(preds)
        actual = np.concatenate(actuals)
        per_h = {
            f"h{h}": {
                "pickup_mae": float(np.abs(prediction[:, i, 0] - actual[:, i, 0]).mean()),
                "dropoff_mae": float(np.abs(prediction[:, i, 1] - actual[:, i, 1]).mean()),
            }
            for i, h in enumerate(horizons)
        }
        mean_mae = float(np.mean([v for d in per_h.values() for v in d.values()]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "valid_mean_mae": mean_mae, "per_horizon": per_h,
                        "seconds": round(time.perf_counter() - epoch_start, 1)})
        log.info("epoch %d/%d loss=%.4f valid MAE=%.4f (h1 pickup %.4f) %.0fs",
                 epoch, args.epochs, history[-1]["train_loss"], mean_mae,
                 per_h[f"h{horizons[0]}"]["pickup_mae"], history[-1]["seconds"])
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_dir = Path(args.out) if args.out else resolve_path(cfg, "paths.models", mkdir=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"{args.model}_{bundle.meta['tag']}.pt")
    metrics = {
        "model": args.model, "tag": bundle.meta["tag"], "device": device,
        "parameters": n_params, "window": window, "batch_size": batch_size,
        "epochs": args.epochs, "best_valid_mean_mae": best_mae, "history": history,
    }
    metrics_dir = resolve_path(cfg, "paths.metrics", mkdir=True)
    (metrics_dir / f"deep_{args.model}_{bundle.meta['tag']}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("best valid mean MAE %.4f -> %s", best_mae,
             out_dir / f"{args.model}_{bundle.meta['tag']}.pt")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    raise SystemExit(code)
