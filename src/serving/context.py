
from __future__ import annotations

import functools
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.utils.config import resolve_path
from src.utils.geo import haversine_km

#: features the explanation reads, all of them genuine model inputs
DRIVER_COLUMNS = [
    "pickups_lag_0", "dropoffs_lag_0",
    "pickups_ewm_4", "pickups_roll_mean_4", "pickups_roll_mean_96",
    "pickups_slot_expanding_mean", "dropoffs_slot_expanding_mean",
    "pickups_seasonal_last_week_h1", "pickups_seasonal_yesterday_h1",
    "is_weekend", "is_holiday", "is_morning_rush", "is_evening_rush",
    "hour", "day_of_week", "slot_of_day",
]
#: shown as context only - measured at roughly zero effect by the ablation
CONTEXT_COLUMNS = ["temperature_2m", "precipitation", "is_raining", "is_snowing",
                   "high_wind"]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def season_of(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
            5: "spring", 6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"}[month]


@functools.lru_cache(maxsize=1)
def _geocoded_events() -> pl.DataFrame:
    """Event permits joined to the coordinates Phase 4 resolved for them."""
    from src.utils.config import load_config

    cfg = load_config("h3", "lightgbm")
    external = resolve_path(cfg, "paths.external")
    events = pl.read_parquet(external / "events.parquet").select(
        ["event_id", "event_name", "event_type", "event_borough",
         "event_location", "event_start", "event_end"])
    geo = (pl.read_parquet(external / "event_geocodes.parquet")
           .filter(pl.col("status") == "ok")
           .select(["event_location", "lat", "lng", "label", "confidence"]))
    return events.join(geo, on="event_location", how="inner")


def nearby_events(lat: float, lng: float, when: datetime, *,
                  radius_km: float = 1.0, limit: int = 5) -> list[dict]:
    """Permitted events overlapping `when` within `radius_km`.

    Returned as **context, not cause** - see the module docstring.
    """
    frame = _geocoded_events()
    active = frame.filter((pl.col("event_start") <= when)
                          & (pl.col("event_end") >= when))
    if active.height == 0:
        return []
    distances = haversine_km(
        np.full(active.height, lat), np.full(active.height, lng),
        active["lat"].to_numpy(), active["lng"].to_numpy())
    active = active.with_columns(pl.Series("distance_km", distances)).filter(
        pl.col("distance_km") <= radius_km).sort("distance_km").head(limit)
    return [{
        "name": r["event_name"], "type": r["event_type"],
        "venue": r["label"], "borough": r["event_borough"],
        "distance_km": round(r["distance_km"], 2),
        "starts": str(r["event_start"]), "ends": str(r["event_end"]),
        "influences_forecast": False,
        "note": "shown as context; the ablation measured events at -0.01% MAE",
    } for r in active.to_dicts()]


def explain(engine, region_id: str, forecast_time: datetime,
            predicted_pickups: float) -> dict:
    """The real drivers behind one region's forecast, plus non-causal context.

    Everything under `drivers` is a feature the model actually consumes. Everything
    under `context` is not, and is labelled.
    """
    ts = forecast_time - timedelta(minutes=engine.interval)
    wanted = sorted(set(DRIVER_COLUMNS + CONTEXT_COLUMNS))
    frame = (pl.scan_parquet(str(engine.features4 / "**" / "*.parquet"))
             .filter((pl.col("ts") == ts) & (pl.col("region_id") == region_id))
             .select([c for c in wanted])
             .collect(engine="streaming"))
    if frame.height == 0:
        return {"drivers": [], "context": {}, "available": False}
    row = frame.to_dicts()[0]

    local = forecast_time.astimezone(
        __import__("zoneinfo").ZoneInfo(engine.tz))
    typical = float(row.get("pickups_slot_expanding_mean") or 0.0)
    recent = float(row.get("pickups_ewm_4") or 0.0)
    last_week = float(row.get("pickups_seasonal_last_week_h1") or 0.0)

    drivers = []
    if typical > 0:
        ratio = recent / typical
        direction = "above" if ratio >= 1 else "below"
        drivers.append({
            "factor": "recent momentum",
            "detail": (f"the last hour here averaged {recent:.1f} pickups per 15 min, "
                       f"{abs(ratio - 1) * 100:.0f}% {direction} this cell's usual "
                       f"{WEEKDAYS[local.weekday()]} {local:%H:%M} level "
                       f"({typical:.1f})"),
            "weight": "highest-gain feature family in the model",
        })
    else:
        drivers.append({
            "factor": "recent momentum",
            "detail": f"the last hour averaged {recent:.1f} pickups per 15 min",
            "weight": "highest-gain feature family in the model",
        })
    drivers.append({
        "factor": "weekly seasonality",
        "detail": f"the same slot last week saw {last_week:.0f} pickups",
        "weight": "calendar/seasonal features added +6.6% MAE in the ablation",
    })

    labels = []
    if row.get("is_morning_rush"):
        labels.append("morning rush")
    if row.get("is_evening_rush"):
        labels.append("evening rush")
    if row.get("is_weekend"):
        labels.append("weekend")
    if row.get("is_holiday"):
        labels.append("public holiday")
    drivers.append({
        "factor": "time of week",
        "detail": (f"{WEEKDAYS[local.weekday()]} {local:%H:%M}"
                   + (f" - {', '.join(labels)}" if labels else "")
                   + f", {season_of(local.month)}"),
        "weight": "calendar features, the second-largest gain in the model",
    })

    weather_bits = []
    if row.get("temperature_2m") is not None:
        weather_bits.append(f"{row['temperature_2m']:.0f}C")
    if row.get("is_raining"):
        weather_bits.append("raining")
    if row.get("is_snowing"):
        weather_bits.append("snowing")
    if row.get("high_wind"):
        weather_bits.append("high wind")

    return {
        "available": True,
        "drivers": drivers,
        "context": {
            "weather": ", ".join(weather_bits) or "no notable weather",
            "weather_influences_forecast": False,
            "note": ("weather, events and traffic were all tested and rejected by the "
                     "Phase 4 ablation; they are shown for interest, not because the "
                     "model uses them"),
        },
        "predicted_pickups": predicted_pickups,
    }


def availability_verdict(pickups: float, dropoffs: float,
                         capacity: float | None = None) -> dict:
    """Turn a flow forecast into an honest availability signal.

    We forecast **trips**, not dock occupancy - no historical dock counts exist for
    2023-24 (see docs/DATA_SOURCES.md). So this reports *pressure*: how hard the
    location is being drained relative to what is arriving. It deliberately does not
    print a bike count, which would imply a measurement we never made.
    """
    net = dropoffs - pickups
    total = pickups + dropoffs
    if total < 1.0:
        return {"verdict": "quiet", "label": "Very low activity",
                "net_flow": net, "confidence": "low",
                "detail": "barely any traffic here in this window - "
                          "bikes are unlikely to move much either way"}
    pressure = net / max(total, 1e-9)
    if pressure <= -0.25:
        verdict, label = "draining", "Bikes leaving fast"
        detail = "outflow clearly exceeds inflow - arrive early or expect a walk"
    elif pressure <= -0.05:
        verdict, label = "tightening", "Slightly more departures than arrivals"
        detail = "mild net outflow - usually still findable"
    elif pressure < 0.05:
        verdict, label = "balanced", "Arrivals and departures balanced"
        detail = "inflow and outflow roughly cancel"
    elif pressure < 0.25:
        verdict, label = "filling", "More arrivals than departures"
        detail = "bikes accumulating here"
    else:
        verdict, label = "crowded", "Docks filling up"
        detail = "strong net inflow - returning a bike may be the harder problem"
    return {
        "verdict": verdict, "label": label, "detail": detail,
        "net_flow": net, "pressure": pressure,
        "confidence": "high" if total >= 5 else "medium",
        "measures": "predicted trip flow, not dock occupancy",
        "caveat": ("Citi Bike publishes no historical dock counts for 2023-24, so "
                   "this is a flow-pressure signal rather than a bike count"),
    }
