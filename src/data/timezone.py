
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

AMBIGUOUS_POLICY = "earliest"  # the first (DST) occurrence of a repeated local hour


@dataclass(frozen=True)
class DstTransition:
    kind: str  # "spring_forward" | "fall_back"
    utc_instant: datetime
    local_window_start: datetime  # naive local
    local_window_end: datetime  # naive local, exclusive


def find_dst_transitions(tz_name: str, start: datetime, end: datetime) -> list[DstTransition]:
    """Locate DST transitions by probing the UTC offset hour by hour.

    Cheap enough to run once per pipeline (a few thousand iterations per year) and
    it reads the real tz database rather than hardcoding transition dates.
    """
    tz = ZoneInfo(tz_name)
    transitions: list[DstTransition] = []
    cursor = start.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end_utc = end.replace(tzinfo=timezone.utc)
    prev_offset = cursor.astimezone(tz).utcoffset()
    while cursor <= end_utc:
        cursor += timedelta(hours=1)
        offset = cursor.astimezone(tz).utcoffset()
        if offset != prev_offset:
            delta = offset - prev_offset
            local_after = cursor.astimezone(tz).replace(tzinfo=None)
            if delta > timedelta(0):  # clocks jumped forward -> a local hour vanished
                transitions.append(
                    DstTransition(
                        kind="spring_forward",
                        utc_instant=cursor,
                        local_window_start=local_after - delta,
                        local_window_end=local_after,
                    )
                )
            else:  # clocks went back -> a local hour repeats
                transitions.append(
                    DstTransition(
                        kind="fall_back",
                        utc_instant=cursor,
                        local_window_start=local_after,
                        local_window_end=local_after - delta,
                    )
                )
            prev_offset = offset
    return transitions


def count_dst_affected(
    df: pl.DataFrame, columns: list[str], transitions: list[DstTransition]
) -> dict[str, int]:
    """Count naive-local timestamps falling in nonexistent / ambiguous windows."""
    counts: dict[str, int] = {}
    for col in columns:
        for transition in transitions:
            key = f"{col}__{transition.kind}__{transition.local_window_start:%Y%m%d}"
            counts[key] = int(
                df.select(
                    (
                        (pl.col(col) >= transition.local_window_start)
                        & (pl.col(col) < transition.local_window_end)
                    ).sum()
                ).item()
            )
    return {k: v for k, v in counts.items() if v > 0}


def localize_to_utc(expr: pl.Expr, tz_name: str) -> pl.Expr:
    """Naive local -> tz-aware UTC.

    Ambiguous (fall-back) times take the first/DST occurrence. Nonexistent
    (spring-forward) times are shifted forward by the size of the gap, which for
    US DST is always one hour.
    """
    localized = expr.dt.replace_time_zone(tz_name, ambiguous=AMBIGUOUS_POLICY, non_existent="null")
    shifted = (expr + pl.duration(hours=1)).dt.replace_time_zone(
        tz_name, ambiguous=AMBIGUOUS_POLICY, non_existent="null"
    )
    return pl.coalesce(localized, shifted).dt.convert_time_zone("UTC")


def to_local(expr: pl.Expr, tz_name: str) -> pl.Expr:
    """tz-aware UTC -> tz-aware local, for calendar feature derivation."""
    return expr.dt.convert_time_zone(tz_name)


def dst_unreliable_utc_hours(transitions: list[DstTransition]) -> list[dict]:
    """UTC hours whose ride counts are distorted by DST, and how.

    A naive local timestamp carries no information about *which* pass through a
    repeated hour it belongs to, so the reconstruction cannot be exact. Under the
    `ambiguous="earliest"` policy the distortion is deterministic and enumerable:

    * ``over_filled`` - the first (DST) pass absorbs both hours' rides.
    * ``empty``       - the second (standard-time) pass receives none.
    * ``shifted_in``  - spring-forward rows that could not exist are pushed here.

    Phase 2 onward should flag these hours rather than treat them as observed
    demand. Two hours per year are affected out of 8,760.
    """
    hours: list[dict] = []
    for transition in transitions:
        instant = transition.utc_instant.replace(minute=0, second=0, microsecond=0)
        if transition.kind == "fall_back":
            hours.append(
                {
                    "utc_hour_start": instant - timedelta(hours=1),
                    "effect": "over_filled",
                    "transition": transition.kind,
                    "local_date": transition.local_window_start.date().isoformat(),
                }
            )
            hours.append(
                {
                    "utc_hour_start": instant,
                    "effect": "empty",
                    "transition": transition.kind,
                    "local_date": transition.local_window_start.date().isoformat(),
                }
            )
        else:
            hours.append(
                {
                    "utc_hour_start": instant,
                    "effect": "shifted_in",
                    "transition": transition.kind,
                    "local_date": transition.local_window_start.date().isoformat(),
                }
            )
    return sorted(hours, key=lambda h: h["utc_hour_start"])
