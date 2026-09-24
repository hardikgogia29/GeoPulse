
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = (cfg or {}).get("logging", {}).get("level", "INFO")
        fmt = (cfg or {}).get("logging", {}).get(
            "format", "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str):
    start = time.perf_counter()
    logger.info("START %s", label)
    yield
    logger.info("DONE  %s (%.1fs)", label, time.perf_counter() - start)


@dataclass
class StatsRecorder:
    """Accumulates named counters/records and dumps them to JSON.

    Used so that every row we drop is attributable to a named, counted rule -
    Phase 1 forbids silent drops.
    """

    name: str
    counters: dict[str, int] = field(default_factory=dict)
    records: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, value: int) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(value)

    def set(self, key: str, value: Any) -> None:
        self.records[key] = value

    def merge(self, other: "StatsRecorder") -> None:
        for key, value in other.counters.items():
            self.add(key, value)
        self.records.update(other.records)

    def to_dict(self) -> dict:
        return {"name": self.name, "counters": self.counters, "records": self.records}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path
