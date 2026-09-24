
from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.features.advanced import cumulative_feature_sets
from src.models import baseline
from src.utils.config import Config, load_config, resolve_path

TARGET_KINDS = {"pickup": "pickups", "dropoff": "dropoffs"}
SEASONAL_LAGS = [96, 672]      # same-time-yesterday, same-time-last-week
DEEP_WINDOW = {"stgnn": 24, "tft": 96}

MODEL_LABELS = {
    "lightgbm_final": "LightGBM",
    "stgnn": "ST-GNN",
    "tft": "TFT",
    "seasonal_naive_same_day": "Seasonal Naive (yesterday)",
    "seasonal_naive_same_week": "Seasonal Naive (last week)",
}


class ForecastError(RuntimeError):
    """A request that cannot be served truthfully rather than approximately."""


@dataclass(frozen=True)
class GridSpec:
    spatial: str
    resolution: int

    @property
    def tag(self) -> str:
        return f"{'s2' if self.spatial == 's2' else 'h3'}{self.resolution}"


def build_config(spatial: str, resolution: int) -> Config:
    cfg = load_config(spatial, "lightgbm")
    key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
    return Config({**cfg, "spatial": {**cfg["spatial"], key: resolution,
                                      "resolution": resolution}})


@functools.lru_cache(maxsize=4)
def _engine(spatial: str, resolution: int) -> "ForecastEngine":
    return ForecastEngine(spatial, resolution)


def get_engine(spatial: str = "h3", resolution: int = 8) -> "ForecastEngine":
    """Process-wide cached engine - building one loads models and a tensor bundle."""
    return _engine(spatial, resolution)


class ForecastEngine:
    def __init__(self, spatial: str = "h3", resolution: int = 8) -> None:
        self.cfg = build_config(spatial, resolution)
        self.grid = GridSpec(spatial, resolution)
        self.tag = self.grid.tag
        self.interval = self.cfg.dotted("time.interval_minutes")
        self.tz = self.cfg.dotted("time.timezone")
        self.horizons = self.cfg.dotted("time.horizons")
        self.processed = resolve_path(self.cfg, "paths.processed")
        self.models_dir = resolve_path(self.cfg, "paths.models")
        self.features4 = self.processed / f"features4_{self.tag}"
        self.features_basic = self.processed / f"features_{self.tag}"
        self._boosters: dict[tuple[str, int], object] = {}
        self._bundle = None
        self._deep: dict[str, object] = {}
        self._region_index: dict[str, int] | None = None
        self._feature_set = cumulative_feature_sets(self.cfg)[
            self.cfg.dotted("advanced_features.final_families")[-1]]

    # ------------------------------------------------------------------ metadata
    @functools.cached_property
    def region_ids(self) -> list[str]:
        return sorted(pl.read_parquet(
            resolve_path(self.cfg, "paths.spatial") / f"regions_{self.tag}.parquet"
        )["region_id"].to_list())

    @functools.cached_property
    def bounds(self) -> dict:
        """Time range the models can actually be asked about."""
        meta = self._load_bundle().meta
        return {"first_ts": meta["first_ts"], "last_ts": meta["last_ts"],
                "splits": meta["split_index"]}

    def available_models(self) -> list[str]:
        names = []
        if (self.models_dir / f"lgbm_final_{self.tag}_pickup_h1.txt").exists():
            names.append("lightgbm_final")
        for deep in ("stgnn", "tft"):
            if (self.models_dir / f"{deep}_{self.tag}.pt").exists():
                names.append(deep)
        names += [f"seasonal_naive_{v}"
                  for v in baseline.seasonal_lag_steps(self.cfg)]
        return names

    # -------------------------------------------------------------- time helpers
    def snap(self, when: datetime) -> datetime:
        """Round down to the bin grid. The UI passes human times; models need bins."""
        step = timedelta(minutes=self.interval)
        epoch = datetime(1970, 1, 1, tzinfo=when.tzinfo)
        return epoch + ((when - epoch) // step) * step

    def _bundle_index(self, ts: datetime) -> int:
        bundle = self._load_bundle()
        first = datetime.fromisoformat(bundle.meta["first_ts"])
        offset = (ts - first).total_seconds() / (60 * self.interval)
        if offset != int(offset):
            raise ForecastError(f"{ts} is not on the {self.interval}-minute grid")
        return int(offset)

    # ------------------------------------------------------------------- loaders
    def _load_bundle(self):
        if self._bundle is None:
            from src.models.deep import Bundle

            path = self.processed / f"deep_bundle_{self.tag}"
            if not path.exists():
                raise ForecastError(f"no tensor bundle at {path}")
            self._bundle = Bundle.load(path)
            self._scaled = self._bundle.scaled_demand()
            self._weather = self._bundle.scaled_weather()
            self._static = self._bundle.scaled_static()
        return self._bundle

    def _load_booster(self, kind: str, horizon: int):
        key = (kind, horizon)
        if key not in self._boosters:
            import lightgbm as lgb

            path = self.models_dir / f"lgbm_final_{self.tag}_{kind}_h{horizon}.txt"
            if not path.exists():
                raise ForecastError(f"missing model artifact {path}")
            self._boosters[key] = lgb.Booster(model_file=str(path))
        return self._boosters[key]

    def _load_deep(self, name: str):
        if name not in self._deep:
            import torch

            from src.models.deep import CompactTFT, STGNN

            bundle = self._load_bundle()
            n_h = len(self.horizons)
            if name == "stgnn":
                n_dyn = (2 + 2 * len(SEASONAL_LAGS) + bundle.calendar.shape[1]
                         + self._weather.shape[1])
                model = STGNN(n_dyn, self._static.shape[1], n_h)
            else:
                model = CompactTFT(2 + self._weather.shape[1],
                                   bundle.calendar.shape[1],
                                   self._static.shape[1], n_h)
            path = self.models_dir / f"{name}_{self.tag}.pt"
            if not path.exists():
                raise ForecastError(f"missing checkpoint {path}")
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            self._deep[name] = model
        return self._deep[name]

    @property
    def _region_idx_map(self) -> dict[str, int]:
        """The dense rank training used - over the feature table's full region list."""
        if self._region_index is None:
            regions = sorted(pl.scan_parquet(str(self.features4 / "**" / "*.parquet"))
                             .select("region_id").unique()
                             .collect(engine="streaming")["region_id"].to_list())
            self._region_index = {r: i for i, r in enumerate(regions)}
        return self._region_index

    # ----------------------------------------------------------------- forecasts
    def forecast(self, model_name: str, forecast_time: datetime,
                 horizon: int = 1) -> pl.DataFrame:
        """Region-level forecast for one instant.

        Returns one row per active region with `predicted_pickups`,
        `predicted_dropoffs` for `horizon`, plus cumulative totals over 1..horizon
        (what the inventory projection needs).
        """
        if horizon not in self.horizons:
            raise ForecastError(
                f"horizon {horizon} not trained; have {self.horizons}")
        if forecast_time.tzinfo is None:
            raise ForecastError("forecast_time must be timezone-aware")
        ts = forecast_time - timedelta(minutes=self.interval)
        steps = [h for h in self.horizons if h <= horizon]

        if model_name == "lightgbm_final":
            frame = self._lightgbm(ts, steps)
        elif model_name in DEEP_WINDOW:
            frame = self._deep_forecast(model_name, ts, steps)
        elif model_name.startswith("seasonal_naive_"):
            frame = self._naive(model_name, ts, steps)
        else:
            raise ForecastError(
                f"unknown model {model_name!r}; have {self.available_models()}")

        return frame.with_columns([
            pl.lit(model_name).alias("model"),
            pl.lit(MODEL_LABELS.get(model_name, model_name)).alias("model_label"),
            pl.lit(self.tag).alias("grid"),
            pl.lit(horizon).alias("horizon"),
        ]).sort("region_id")

    def _lightgbm(self, ts: datetime, steps: list[int]) -> pl.DataFrame:
        frame = (pl.scan_parquet(str(self.features4 / "**" / "*.parquet"))
                 .filter(pl.col("ts") == ts)
                 .select(sorted({"region_id", *self._feature_set}))
                 .collect(engine="streaming").sort("region_id"))
        if frame.height == 0:
            raise ForecastError(f"no feature row at ts={ts}")
        inputs = ["region_idx"] + [c for c in self._feature_set if c != "region_id"]
        design = (frame.with_columns(
            pl.col("region_id").replace_strict(self._region_idx_map)
            .alias("region_idx")).select(inputs)
            .to_numpy().astype("float32", copy=False))
        out = {"region_id": frame["region_id"].to_list()}
        for kind in TARGET_KINDS:
            total = np.zeros(design.shape[0])
            for h in steps:
                pred = np.clip(self._load_booster(kind, h).predict(design), 0, None)
                total = total + pred
                if h == steps[-1]:
                    out[f"predicted_{TARGET_KINDS[kind]}"] = pred
            out[f"cumulative_{TARGET_KINDS[kind]}"] = total
        return pl.DataFrame(out)

    def _naive(self, model_name: str, ts: datetime, steps: list[int]) -> pl.DataFrame:
        variant = model_name[len("seasonal_naive_"):]
        seasons = baseline.seasonal_lag_steps(self.cfg)
        if variant not in seasons:
            raise ForecastError(f"unknown variant {variant!r}; have {list(seasons)}")
        season = seasons[variant]
        columns = {kind: [baseline.lag_column_for(src, season, h) for h in steps]
                   for kind, src in TARGET_KINDS.items()}
        frame = (pl.scan_parquet(str(self.features_basic / "**" / "*.parquet"))
                 .filter(pl.col("ts") == ts)
                 .select(sorted({"region_id", *columns["pickup"], *columns["dropoff"]}))
                 .collect(engine="streaming").sort("region_id"))
        if frame.height == 0:
            raise ForecastError(f"no baseline row at ts={ts}")
        out = {"region_id": frame["region_id"].to_list()}
        for kind, names in columns.items():
            vectors = [np.clip(frame[c].fill_null(0).to_numpy().astype("float64"),
                               0, None) for c in names]
            out[f"predicted_{TARGET_KINDS[kind]}"] = vectors[-1]
            out[f"cumulative_{TARGET_KINDS[kind]}"] = np.sum(vectors, axis=0)
        return pl.DataFrame(out)

    def _deep_forecast(self, name: str, ts: datetime,
                       steps: list[int]) -> pl.DataFrame:
        import torch

        bundle = self._load_bundle()
        model = self._load_deep(name)
        window = DEEP_WINDOW[name]
        anchor = self._bundle_index(ts)
        need = max(SEASONAL_LAGS) + window
        if anchor < need:
            raise ForecastError(
                f"{name} needs {need} steps of history before {ts}; the bundle starts "
                f"at {bundle.meta['first_ts']}")
        if anchor >= bundle.n_steps:
            raise ForecastError(f"{ts} is past the end of the data")

        idx = np.arange(anchor - window + 1, anchor + 1)[None, :]
        n = bundle.n_regions
        with torch.no_grad():
            if name == "stgnn":
                demand = self._scaled[idx]
                cal = bundle.calendar[idx][:, :, None, :].repeat(n, axis=2)
                wx = self._weather[idx][:, :, None, :].repeat(n, axis=2)
                seasonal = [self._scaled[idx - lag] for lag in SEASONAL_LAGS]
                dynamic = np.concatenate([demand, *seasonal, cal, wx], axis=-1)
                out = model(torch.from_numpy(dynamic).float(),
                            torch.from_numpy(self._static).float(),
                            torch.from_numpy(bundle.edges.astype(np.int64))
                            ).numpy()[0]           # [N, H, 2]
            else:
                regions = np.arange(n)
                rows = np.repeat(idx, n, axis=0)   # [N, W]
                observed = np.concatenate(
                    [self._scaled[rows, regions[:, None]],
                     self._weather[rows]], axis=-1)
                out = model(torch.from_numpy(observed).float(),
                            torch.from_numpy(bundle.calendar[rows]).float(),
                            torch.from_numpy(self._static[regions]).float()
                            ).numpy()              # [N, H, 2]

        out = np.clip(out, 0, None)
        keep = [self.horizons.index(h) for h in steps]
        last = self.horizons.index(steps[-1])
        return pl.DataFrame({
            "region_id": bundle.meta["region_ids"],
            "predicted_pickups": out[:, last, 0],
            "predicted_dropoffs": out[:, last, 1],
            "cumulative_pickups": out[:, keep, 0].sum(axis=1),
            "cumulative_dropoffs": out[:, keep, 1].sum(axis=1),
        })

    # -------------------------------------------------------------------- actuals
    def actuals(self, forecast_time: datetime, horizon: int = 1) -> pl.DataFrame:
        """What really happened - available because the whole window is historical."""
        ts = forecast_time - timedelta(minutes=self.interval)
        target = ts + timedelta(minutes=self.interval * horizon)
        frame = (pl.scan_parquet(str(self.processed / f"panel_{self.tag}"
                                     / "**" / "*.parquet"))
                 .filter(pl.col("ts") == target)
                 .select(["region_id", "pickups", "dropoffs"])
                 .collect(engine="streaming").sort("region_id"))
        return frame.rename({"pickups": "actual_pickups",
                             "dropoffs": "actual_dropoffs"})
