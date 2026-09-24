
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from src.agent import gazetteer
from src.serving.context import (
    availability_verdict, explain as explain_region, nearby_events, season_of,
)
from src.serving.engine import MODEL_LABELS, ForecastError, get_engine
from src.serving.routing import route_to_best_station
from src.serving.stations import nearest_stations, split_to_station

#: the window the models were trained for; anything outside it is refused
DATA_START = datetime(2023, 1, 8, tzinfo=timezone.utc)   # +1wk for the 672-step lag
DATA_END = datetime(2024, 12, 31, tzinfo=timezone.utc)
DEFAULT_DEMO_TIME = datetime(2024, 11, 6, 22, 0, tzinfo=timezone.utc)  # 5pm Wed


def _tz(engine) -> ZoneInfo:
    return ZoneInfo(engine.tz)


def resolve_time(when: str | None = None, *, spatial: str = "h3",
                 resolution: int = 8) -> dict:
    """Turn a user's time into a bin the models can actually answer for.

    The models cover Jan 2023 - Dec 2024. "Now" is outside that, so a request for
    the current time is mapped to the **same weekday and clock time in the matching
    week of 2024** and labelled `typical_conditions`. That is a real forecast of a
    comparable moment - not a live one - and the label must survive to the UI.
    """
    engine = get_engine(spatial, resolution)
    tz = _tz(engine)
    mode = "explicit"

    if when in (None, "", "now"):
        now = datetime.now(tz)
        # same weekday + clock time, projected back into the data window
        target = now.replace(year=2024)
        shift = (now.weekday() - target.weekday()) % 7
        target = target + timedelta(days=shift)
        if not (DATA_START <= target.astimezone(timezone.utc) <= DATA_END):
            target = DEFAULT_DEMO_TIME.astimezone(tz)
        mode = "typical_conditions"
        resolved = target
    else:
        try:
            resolved = datetime.fromisoformat(when)
        except ValueError:
            return {"ok": False,
                    "error": f"could not read the time {when!r}; use ISO format "
                             f"like 2024-11-06T17:00 or the word 'now'"}
        if resolved.tzinfo is None:
            resolved = resolved.replace(tzinfo=tz)

    utc = engine.snap(resolved.astimezone(timezone.utc))
    if not (DATA_START <= utc <= DATA_END):
        return {"ok": False,
                "error": (f"{utc:%Y-%m-%d %H:%M} UTC is outside the modelled window "
                          f"({DATA_START:%Y-%m-%d} to {DATA_END:%Y-%m-%d}). The models "
                          f"were trained on 2023-2024 Citi Bike data and there is no "
                          f"live feed wired in."),
                "window": {"start": DATA_START.isoformat(), "end": DATA_END.isoformat()}}

    local = utc.astimezone(tz)
    return {
        "ok": True,
        "forecast_time_utc": utc.isoformat(),
        "local": local.isoformat(),
        "local_pretty": f"{local:%A %d %B %Y, %H:%M}",
        "weekday": f"{local:%A}",
        "season": season_of(local.month),
        "mode": mode,
        "is_live": False,
        "note": ("mapped to the equivalent 2024 weekday and time - the models cover "
                 "2023-2024 only, so this is typical conditions, not live"
                 if mode == "typical_conditions" else
                 "a specific historical instant, so the real outcome is known too"),
    }


def find_place(query: str) -> dict:
    """Locate a place from our own station and landmark data. No external API."""
    matches = gazetteer.search(query, limit=5)
    if not matches:
        return {"ok": False, "query": query, "matches": [],
                "error": "no match in our gazetteer (2,459 stations + 19,011 NYC "
                         "landmarks). Try a nearby intersection or park."}
    best = matches[0]
    return {
        "ok": True, "query": query,
        "best": best.as_dict(),
        "matches": [m.as_dict() for m in matches],
        "confident": best.is_confident,
        "note": None if best.is_confident else
        ("no confident match - the best guess scored low, so confirm with the user "
         "before using it rather than assuming. Our gazetteer has real holes: it is "
         "built from Citi Bike station names, so areas without stations are missing."),
    }


def station_forecast(lat: float, lng: float, when: str | None = None, *,
                     horizon: int = 4, model: str = "lightgbm_final",
                     spatial: str = "h3", resolution: int = 8,
                     k: int = 4) -> dict:
    """Forecast for the nearest stations to a point, with an availability verdict."""
    engine = get_engine(spatial, resolution)
    stamp = resolve_time(when, spatial=spatial, resolution=resolution)
    if not stamp["ok"]:
        return stamp
    forecast_time = datetime.fromisoformat(stamp["forecast_time_utc"])

    try:
        regions = engine.forecast(model, forecast_time, horizon)
    except ForecastError as exc:
        return {"ok": False, "error": str(exc)}

    lookup = {r["region_id"]: r for r in regions.to_dicts()}
    stations = nearest_stations(lat, lng, k=k, spatial=spatial, resolution=resolution)
    minutes = horizon * engine.interval

    results = []
    for station in stations:
        region = lookup.get(station.region_id)
        if region is None:
            continue
        split = split_to_station(region["cumulative_pickups"],
                                 region["cumulative_dropoffs"], station)
        verdict = availability_verdict(split["predicted_pickups"],
                                       split["predicted_dropoffs"])
        results.append({
            "station": station.as_dict(),
            "window_minutes": minutes,
            "predicted_pickups": round(split["predicted_pickups"], 1),
            "predicted_dropoffs": round(split["predicted_dropoffs"], 1),
            "region_pickups": round(split["region_pickups"], 1),
            "region_dropoffs": round(split["region_dropoffs"], 1),
            "availability": verdict,
            "derivation": split["method"],
            "approximate": True,
        })

    return {
        "ok": True,
        "time": stamp,
        "model": model, "model_label": MODEL_LABELS.get(model, model),
        "grid": engine.tag,
        "horizon": horizon, "window_minutes": minutes,
        "stations": results,
        "caveats": [
            "station numbers are a region forecast split by each station's historical "
            "share of its cell's trips - not a per-station model",
            "availability is predicted trip flow, not dock occupancy; no historical "
            "dock counts exist for 2023-24",
        ],
    }


def route_to_bike(lat: float, lng: float, when: str | None = None, *,
                  horizon: int = 4, model: str = "lightgbm_final",
                  spatial: str = "h3", resolution: int = 8,
                  prefer_available: bool = True) -> dict:
    """Walk the user to a dock worth walking to, and return the path.

    Not simply the nearest dock: A* picks the target by walking distance *plus* a
    penalty from the demand forecast, so a slightly longer walk to a dock that is
    filling can beat a closer one that is emptying.
    """
    forecast = station_forecast(lat, lng, when, horizon=horizon, model=model,
                                spatial=spatial, resolution=resolution, k=8)
    if not forecast.get("ok"):
        return forecast
    route = route_to_best_station(lat, lng, forecast["stations"],
                                  spatial=spatial, resolution=resolution,
                                  prefer_available=prefer_available)
    if not route.get("ok"):
        return route
    route["time"] = forecast["time"]
    route["model"] = forecast["model"]
    route["model_label"] = forecast["model_label"]
    route["grid"] = forecast["grid"]
    route["window_minutes"] = forecast["window_minutes"]
    return route


def explain_forecast(lat: float, lng: float, when: str | None = None, *,
                     horizon: int = 4, model: str = "lightgbm_final",
                     spatial: str = "h3", resolution: int = 8) -> dict:
    """Why the forecast says what it does, separating drivers from context."""
    engine = get_engine(spatial, resolution)
    stamp = resolve_time(when, spatial=spatial, resolution=resolution)
    if not stamp["ok"]:
        return stamp
    forecast_time = datetime.fromisoformat(stamp["forecast_time_utc"])
    stations = nearest_stations(lat, lng, k=1, spatial=spatial, resolution=resolution)
    if not stations:
        return {"ok": False, "error": "no station near that point"}
    region_id = stations[0].region_id
    try:
        regions = engine.forecast(model, forecast_time, horizon)
    except ForecastError as exc:
        return {"ok": False, "error": str(exc)}
    row = regions.filter(pl.col("region_id") == region_id).to_dicts()
    if not row:
        return {"ok": False, "error": f"region {region_id} not forecast"}

    detail = explain_region(engine, region_id, forecast_time,
                            row[0]["predicted_pickups"])
    detail["events_nearby"] = nearby_events(lat, lng, forecast_time)
    detail["ok"] = True
    detail["time"] = stamp
    detail["region_id"] = region_id
    detail["nearest_station"] = stations[0].as_dict()
    detail["honesty_rule"] = (
        "`drivers` are features the model consumes. `events_nearby` and `context` "
        "are NOT model inputs - the Phase 4 ablation measured events at -0.01% and "
        "weather at -0.05% MAE and both were excluded. Never present them as causes."
    )
    return detail


def compare_models(lat: float, lng: float, when: str | None = None, *,
                   horizon: int = 4, spatial: str = "h3",
                   resolution: int = 8) -> dict:
    """Run every available model on the same instant and cell, side by side."""
    engine = get_engine(spatial, resolution)
    stamp = resolve_time(when, spatial=spatial, resolution=resolution)
    if not stamp["ok"]:
        return stamp
    forecast_time = datetime.fromisoformat(stamp["forecast_time_utc"])
    stations = nearest_stations(lat, lng, k=1, spatial=spatial, resolution=resolution)
    if not stations:
        return {"ok": False, "error": "no station near that point"}
    station = stations[0]

    actual = engine.actuals(forecast_time, horizon).filter(
        pl.col("region_id") == station.region_id).to_dicts()
    # Everything below is the SINGLE bin at `horizon`, never the 1..h cumulative,
    # because that is the only quantity `actuals()` returns - comparing a cumulative
    # forecast against a one-bin actual would understate every model by ~4x.
    rows = []
    for name in engine.available_models():
        try:
            frame = engine.forecast(name, forecast_time, horizon)
        except ForecastError:
            continue
        match = frame.filter(pl.col("region_id") == station.region_id).to_dicts()
        if not match:
            continue
        split = split_to_station(match[0]["predicted_pickups"],
                                 match[0]["predicted_dropoffs"], station)
        rows.append({
            "model": name, "label": MODEL_LABELS.get(name, name),
            "region_pickups": round(match[0]["predicted_pickups"], 1),
            "region_dropoffs": round(match[0]["predicted_dropoffs"], 1),
            "station_pickups": round(split["predicted_pickups"], 1),
            "station_dropoffs": round(split["predicted_dropoffs"], 1),
            "error_vs_actual": (round(match[0]["predicted_pickups"]
                                      - actual[0]["actual_pickups"], 1)
                                if actual else None),
        })
    actual_region = round(actual[0]["actual_pickups"], 1) if actual else None
    return {
        "ok": True, "time": stamp, "grid": engine.tag, "horizon": horizon,
        "bin_minutes": engine.interval,
        "station": station.as_dict(),
        "models": rows,
        "actual_region_pickups": actual_region,
        "actual_station_pickups": (round(actual_region * station.pickup_share, 1)
                                   if actual_region is not None else None),
        "note": ("All figures are the single 15-minute bin at this horizon, so the "
                 "`region_*` predictions are directly comparable to "
                 "`actual_region_pickups` - what really happened, available because "
                 "the window is historical. `station_*` values are that region "
                 "figure times this station's historical share. TEST results rank "
                 "these LightGBM > ST-GNN > TFT on MAE."),
    }


def city_overview(when: str | None = None, *, horizon: int = 4,
                  model: str = "lightgbm_final", spatial: str = "h3",
                  resolution: int = 8, top: int = 12) -> dict:
    """Every region's forecast - what the map draws - plus the busiest cells."""
    engine = get_engine(spatial, resolution)
    stamp = resolve_time(when, spatial=spatial, resolution=resolution)
    if not stamp["ok"]:
        return stamp
    forecast_time = datetime.fromisoformat(stamp["forecast_time_utc"])
    try:
        frame = engine.forecast(model, forecast_time, horizon)
    except ForecastError as exc:
        return {"ok": False, "error": str(exc)}
    frame = frame.with_columns(
        (pl.col("cumulative_dropoffs") - pl.col("cumulative_pickups")).alias("net_flow"))
    hottest = frame.sort("cumulative_pickups", descending=True).head(top)
    return {
        "ok": True, "time": stamp, "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "grid": engine.tag, "horizon": horizon,
        "window_minutes": horizon * engine.interval,
        "regions": frame.select(["region_id", "cumulative_pickups",
                                 "cumulative_dropoffs", "net_flow"]).to_dicts(),
        "total_predicted_pickups": round(frame["cumulative_pickups"].sum(), 1),
        "busiest": hottest.select(["region_id", "cumulative_pickups",
                                   "net_flow"]).to_dicts(),
    }


#: name -> callable, for the agent loop to dispatch on
TOOLS = {
    "find_place": find_place,
    "resolve_time": resolve_time,
    "station_forecast": station_forecast,
    "explain_forecast": explain_forecast,
    "compare_models": compare_models,
    "city_overview": city_overview,
    "route_to_bike": route_to_bike,
}
