
from __future__ import annotations

import functools
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.agent import tools as toolkit  # noqa: E402
from src.agent.assistant import Assistant  # noqa: E402
from src.serving.engine import MODEL_LABELS, ForecastError, get_engine  # noqa: E402
from src.serving.stations import station_table  # noqa: E402

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="GeoPulse", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

GRIDS = [
    {"id": "h3-8", "spatial": "h3", "resolution": 8, "label": "H3 · res 8",
     "detail": "321 hexagons, ~0.74 km² · the Phase 5 winner"},
    {"id": "s2-13", "spatial": "s2", "resolution": 13, "label": "S2 · level 13",
     "detail": "231 quadrilaterals, ~1.09 km² · matched to H3-8"},
]


def basemaps() -> list[dict]:
    """Vector basemaps, keyless first.

    Vector rather than raster matters here for a specific reason: with a vector
    style the choropleth can be inserted *beneath* the label layers, so street and
    place names stay readable on top of it. Raster tiles are a single flat image -
    anything we draw necessarily covers the labels.

    The keyless providers are genuinely free and need no signup. The keyed ones only
    appear if the corresponding environment variable is set, so nothing here breaks
    on a fresh clone.
    """
    styles = [
        {"id": "liberty", "label": "Streets",
         "url": "https://tiles.openfreemap.org/styles/liberty",
         "theme": "light", "detail": "OpenFreeMap · full detail, no key"},
        {"id": "positron", "label": "Clean",
         "url": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
         "theme": "light", "detail": "CARTO Positron · muted, data-first"},
        {"id": "voyager", "label": "Colour",
         "url": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
         "theme": "light", "detail": "CARTO Voyager"},
        {"id": "dark-matter", "label": "Dark",
         "url": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
         "theme": "dark", "detail": "CARTO Dark Matter"},
    ]
    maptiler = os.environ.get("MAPTILER_KEY")
    if maptiler:
        styles += [
            {"id": "maptiler-streets", "label": "MapTiler",
             "url": f"https://api.maptiler.com/maps/streets-v2/style.json?key={maptiler}",
             "theme": "light", "detail": "MapTiler Streets v2"},
            {"id": "maptiler-dataviz", "label": "Dataviz",
             "url": f"https://api.maptiler.com/maps/dataviz/style.json?key={maptiler}",
             "theme": "light", "detail": "MapTiler Dataviz · built for overlays"},
        ]
    stadia = os.environ.get("STADIA_KEY")
    if stadia:
        styles.append(
            {"id": "stadia-smooth", "label": "Alidade",
             "url": f"https://tiles.stadiamaps.com/styles/alidade_smooth.json?api_key={stadia}",
             "theme": "light", "detail": "Stadia Alidade Smooth"})
    return styles


@functools.lru_cache(maxsize=1)
def _assistant() -> Assistant:
    return Assistant()


def _grid(grid_id: str) -> tuple[str, int]:
    for grid in GRIDS:
        if grid["id"] == grid_id:
            return grid["spatial"], grid["resolution"]
    return "h3", 8


class ChatRequest(BaseModel):
    message: str
    grid: str = "h3-8"
    model: str = "lightgbm_final"
    horizon: int = 4
    when: str | None = None
    history: list | None = None
    last_place: dict | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    """Everything the frontend needs to render its controls, from real artifacts."""
    engine = get_engine("h3", 8)
    brain = _assistant()
    zone = ZoneInfo(engine.tz)

    def local(moment: datetime) -> str:
        """Naive NYC-local ISO. The picker is a naive field and the backend reads a
        naive value as NYC local, so handing it a UTC string would show a time five
        hours off and shift the forecast the moment the user touched it."""
        return moment.astimezone(zone).replace(tzinfo=None).isoformat()

    return {
        "grids": GRIDS,
        "basemaps": basemaps(),
        "models": [{"id": name, "label": MODEL_LABELS.get(name, name)}
                   for name in engine.available_models()],
        "horizons": [{"value": h, "label": f"{h * engine.interval} min"}
                     for h in engine.horizons],
        "window": {"start": toolkit.DATA_START.isoformat(),
                   "end": toolkit.DATA_END.isoformat(),
                   "default": toolkit.DEFAULT_DEMO_TIME.isoformat(),
                   "start_local": local(toolkit.DATA_START),
                   "end_local": local(toolkit.DATA_END),
                   "default_local": local(toolkit.DEFAULT_DEMO_TIME),
                   "timezone": engine.tz},
        "assistant_mode": brain.mode,
        "assistant_model": brain.model or None,
        "notice": ("Models cover Jan 2023 – Dec 2024. Nothing here is live: a request "
                   "for “now” is mapped to the equivalent 2024 weekday and time."),
    }


@app.get("/api/geojson")
def geojson(grid: str = "h3-8") -> dict:
    """Cell polygons for the map, straight from the spatial indexer."""
    spatial, resolution = _grid(grid)
    engine = get_engine(spatial, resolution)
    from src.spatial.h3_indexer import make_indexer

    indexer = make_indexer(engine.cfg)
    features = []
    for region_id in engine.region_ids:
        ring = [[lng, lat] for lat, lng in indexer.boundary(region_id)]
        if not ring:
            continue
        ring.append(ring[0])                    # GeoJSON polygons must close
        features.append({
            "type": "Feature",
            "properties": {"region_id": region_id},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/overview")
def overview(when: str | None = None, grid: str = "h3-8",
             model: str = "lightgbm_final", horizon: int = 4) -> JSONResponse:
    spatial, resolution = _grid(grid)
    data = toolkit.city_overview(when=when, horizon=horizon, model=model,
                                 spatial=spatial, resolution=resolution)
    return JSONResponse(data, status_code=200 if data.get("ok") else 400)


@app.get("/api/stations")
def stations(grid: str = "h3-8") -> dict:
    """Every station with its region, so the map can dot them and attribute demand."""
    spatial, resolution = _grid(grid)
    frame = station_table("v1", spatial, resolution)
    return {"stations": frame.select(
        ["station_id", "station_name", "lat", "lng", "region_id",
         "pickup_share", "dropoff_share", "share_confidence"]).to_dicts()}


@app.get("/api/place")
def place(q: str) -> JSONResponse:
    data = toolkit.find_place(q)
    return JSONResponse(data, status_code=200 if data.get("ok") else 404)


@app.get("/api/station_forecast")
def station_forecast(lat: float, lng: float, when: str | None = None,
                     grid: str = "h3-8", model: str = "lightgbm_final",
                     horizon: int = 4, k: int = 4) -> JSONResponse:
    spatial, resolution = _grid(grid)
    data = toolkit.station_forecast(lat, lng, when, horizon=horizon, model=model,
                                    spatial=spatial, resolution=resolution, k=k)
    return JSONResponse(data, status_code=200 if data.get("ok") else 400)


@app.get("/api/route")
def route(lat: float, lng: float, when: str | None = None, grid: str = "h3-8",
          model: str = "lightgbm_final", horizon: int = 4,
          prefer_available: bool = True) -> JSONResponse:
    """A* walking route to the dock worth walking to, not merely the closest."""
    spatial, resolution = _grid(grid)
    data = toolkit.route_to_bike(lat, lng, when, horizon=horizon, model=model,
                                 spatial=spatial, resolution=resolution,
                                 prefer_available=prefer_available)
    return JSONResponse(data, status_code=200 if data.get("ok") else 400)


@app.get("/api/explain")
def explain(lat: float, lng: float, when: str | None = None, grid: str = "h3-8",
            model: str = "lightgbm_final", horizon: int = 4) -> JSONResponse:
    spatial, resolution = _grid(grid)
    data = toolkit.explain_forecast(lat, lng, when, horizon=horizon, model=model,
                                    spatial=spatial, resolution=resolution)
    return JSONResponse(data, status_code=200 if data.get("ok") else 400)


@app.get("/api/compare")
def compare(lat: float, lng: float, when: str | None = None, grid: str = "h3-8",
            horizon: int = 4) -> JSONResponse:
    spatial, resolution = _grid(grid)
    data = toolkit.compare_models(lat, lng, when, horizon=horizon,
                                  spatial=spatial, resolution=resolution)
    return JSONResponse(data, status_code=200 if data.get("ok") else 400)


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    spatial, resolution = _grid(request.grid)
    session = {
        "spatial": spatial, "resolution": resolution, "model": request.model,
        "horizon": request.horizon, "when": request.when,
        "history": request.history or [], "last_place": request.last_place,
    }
    reply = _assistant().reply(request.message, session=session)
    return {
        "text": reply.get("text", ""),
        "mode": reply.get("mode"),
        "degraded": reply.get("degraded"),
        "place": reply.get("place") or request.last_place,
        "history": reply.get("history") or [],
        "tool_calls": [{"tool": t["tool"], "input": t["input"]}
                       for t in reply.get("tool_calls", [])],
        # the raw tool payloads drive the result cards, so the UI shows the same
        # numbers the assistant was given rather than re-deriving them from prose
        "data": [t["result"] for t in reply.get("tool_calls", [])],
    }


@app.get("/api/health")
def health() -> dict:
    try:
        engine = get_engine("h3", 8)
        return {"ok": True, "models": engine.available_models(),
                "regions": len(engine.region_ids),
                "assistant": _assistant().mode}
    except ForecastError as exc:
        return {"ok": False, "error": str(exc)}
