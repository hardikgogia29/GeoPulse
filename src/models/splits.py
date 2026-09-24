
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Split:
    name: str
    start: date
    end: date  # inclusive

    def sql(self, cfg, ts_col: str = "ts") -> str:
        tz = cfg.dotted("time.timezone")
        local_date = f"({ts_col} AT TIME ZONE '{tz}')::DATE"
        return f"{local_date} >= DATE '{self.start}' AND {local_date} <= DATE '{self.end}'"

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def load_splits(cfg) -> dict[str, Split]:
    return {
        name: Split(
            name,
            date.fromisoformat(cfg.dotted(f"split.{name}.start")),
            date.fromisoformat(cfg.dotted(f"split.{name}.end")),
        )
        for name in ("train", "validate", "test")
    }


def validate_splits(cfg) -> list[str]:
    """Return a list of problems; empty means the splits are sound.

    Checked here rather than only in tests so any script that trains can assert it
    cheaply before spending an hour on a leaky experiment.
    """
    splits = load_splits(cfg)
    train, validate, test = splits["train"], splits["validate"], splits["test"]
    problems: list[str] = []

    for split in splits.values():
        if split.start > split.end:
            problems.append(f"{split.name}: start {split.start} after end {split.end}")

    if train.end >= validate.start:
        problems.append(f"train ends {train.end} but validate starts {validate.start}")
    if validate.end >= test.start:
        problems.append(f"validate ends {validate.end} but test starts {test.start}")

    if validate.start - train.end != timedelta(days=1):
        problems.append(f"gap between train and validate: {train.end} -> {validate.start}")
    if test.start - validate.end != timedelta(days=1):
        problems.append(f"gap between validate and test: {validate.end} -> {test.start}")

    window_start = date.fromisoformat(cfg.dotted("time.start_date"))
    window_end = date.fromisoformat(cfg.dotted("time.end_date"))
    if train.start < window_start:
        problems.append(f"train starts {train.start} before the data window {window_start}")
    if test.end > window_end:
        problems.append(f"test ends {test.end} after the data window {window_end}")
    return problems
