
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class Config(dict):
    """dict with attribute access and dotted lookup, so cfg.get_path('paths.raw') works."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute errors should be loud
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def dotted(self, key: str, default: Any = "__raise__") -> Any:
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__raise__":
                    raise KeyError(f"missing config key: {key}")
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=None)
def load_config(*overlays: str) -> Config:
    """Load configs/base.yaml, then merge the named overlays in order.

    >>> cfg = load_config("h3", "lightgbm")
    """
    merged: dict = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    for name in overlays:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"config overlay not found: {path}")
        merged = _deep_merge(merged, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return Config(merged)


def resolve_path(cfg: Config, dotted_key: str, *parts: str, mkdir: bool = False) -> Path:
    """Resolve a `paths.*` config entry to an absolute Path under the repo root.

    Absolute values in the config (e.g. a data_root outside the repo) are respected.
    """
    raw = cfg.dotted(dotted_key)
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if parts:
        path = path.joinpath(*parts)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def source_path(cfg: Config, key: str) -> Path:
    """Resolve a `sources.*` entry (these are absolute machine paths)."""
    return Path(os.path.expandvars(str(cfg.dotted(f"sources.{key}"))))
