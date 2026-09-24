
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from src.serving.routing import DETOUR_FACTOR, Graph, astar
from src.utils.geo import haversine_km

OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "GeoPulse/1.0 (NYC Citi Bike research project)"
#: padding around the origin-destination box, so the route can leave the direct
#: corridor when the street layout requires it
PAD_KM = 0.45
#: bbox coordinates are rounded to this many decimals for the cache key (~1 km)
CACHE_PRECISION = 2
CACHE_TTL_SECONDS = 60 * 60 * 24 * 30

#: excluded because you cannot (or may not) walk on them
EXCLUDED_HIGHWAYS = ("motorway", "motorway_link", "trunk", "trunk_link",
                     "construction", "proposed", "raceway", "bus_guideway")


class StreetError(RuntimeError):
    """Raised when a street network cannot be obtained or used."""


def _cache_dir() -> Path:
    from src.utils.config import load_config, resolve_path

    path = resolve_path(load_config("h3", "lightgbm"), "paths.external") / "streets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bbox_for(points: list[tuple[float, float]], pad_km: float = PAD_KM):
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    dlat = pad_km / 111.32
    # longitude degrees shrink with latitude, so pad in the local scale
    dlng = pad_km / (111.32 * max(math.cos(math.radians(sum(lats) / len(lats))), 0.1))
    return (min(lats) - dlat, min(lngs) - dlng,
            max(lats) + dlat, max(lngs) + dlng)


def _overpass_query(south: float, west: float, north: float, east: float) -> str:
    excluded = "|".join(EXCLUDED_HIGHWAYS)
    return (
        f'[out:json][timeout:60];'
        f'way["highway"]["highway"!~"^({excluded})$"]'
        f'["foot"!="no"]["access"!~"^(private|no)$"]'
        f'({south:.5f},{west:.5f},{north:.5f},{east:.5f});'
        f'(._;>;);out skel qt;'
    )


def fetch_network(south: float, west: float, north: float, east: float,
                  *, timeout: int = 75) -> dict:
    """Raw Overpass elements for one box, memoised on disk."""
    key = "_".join(f"{v:.{CACHE_PRECISION}f}" for v in (south, west, north, east))
    cached = _cache_dir() / f"walk_{key}.json"
    if cached.exists() and (time.time() - cached.stat().st_mtime) < CACHE_TTL_SECONDS:
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached.unlink(missing_ok=True)     # a truncated write, refetch

    body = urllib.parse.urlencode(
        {"data": _overpass_query(south, west, north, east)}).encode()
    request = urllib.request.Request(OVERPASS, data=body,
                                     headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise StreetError(f"could not reach Overpass: {exc}") from exc

    cached.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def build_graph(payload: dict) -> Graph:
    """Turn Overpass elements into the same Graph shape `routing.astar` consumes."""
    coords: dict[int, tuple[float, float]] = {}
    for element in payload.get("elements", ()):
        if element.get("type") == "node":
            coords[element["id"]] = (element["lat"], element["lon"])

    used: dict[int, int] = {}          # osm id -> dense index
    lat_list: list[float] = []
    lng_list: list[float] = []
    ids: list[str] = []

    def index_of(osm_id: int) -> int | None:
        if osm_id in used:
            return used[osm_id]
        point = coords.get(osm_id)
        if point is None:
            return None
        used[osm_id] = len(ids)
        ids.append(str(osm_id))
        lat_list.append(point[0])
        lng_list.append(point[1])
        return used[osm_id]

    pairs: list[tuple[int, int]] = []
    for element in payload.get("elements", ()):
        if element.get("type") != "way":
            continue
        refs = element.get("nodes", ())
        for a_id, b_id in zip(refs, refs[1:]):
            a, b = index_of(a_id), index_of(b_id)
            if a is not None and b is not None and a != b:
                pairs.append((a, b))

    if not pairs:
        raise StreetError("no walkable streets found in that area")

    lat = np.asarray(lat_list, dtype=float)
    lng = np.asarray(lng_list, dtype=float)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in ids]
    seen: set[tuple[int, int]] = set()
    for a, b in pairs:
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        metres = float(haversine_km(np.array([lat[a]]), np.array([lng[a]]),
                                    np.array([lat[b]]), np.array([lng[b]]))[0]) * 1000
        adjacency[a].append((b, metres))
        adjacency[b].append((a, metres))

    # street segment lengths are the real ground distance, so no detour factor here -
    # that correction exists only to approximate streets from straight lines
    return Graph(ids=ids, names=ids, lat=lat, lng=lng, adjacency=adjacency)


def snap(graph: Graph, lat: float, lng: float) -> int:
    """Nearest street node to an arbitrary point."""
    distances = haversine_km(np.full(graph.n_nodes, lat), np.full(graph.n_nodes, lng),
                             graph.lat, graph.lng)
    return int(np.argmin(distances))


def walk_route(origin: tuple[float, float],
               destination: tuple[float, float]) -> dict:
    """Street-following path between two points.

    Returns the coordinate list, the true walked distance, and how far each end had
    to be snapped to the network - a large snap means the point sits away from any
    mapped street and the first or last leg is a straight line across open ground.
    """
    south, west, north, east = _bbox_for([origin, destination])
    graph = build_graph(fetch_network(south, west, north, east))

    start, goal = snap(graph, *origin), snap(graph, *destination)
    if start == goal:
        raise StreetError("origin and destination snap to the same street node")
    found = astar(graph, start, goal)
    if found is None:
        raise StreetError("no walkable path between those points in this area")
    path, metres = found

    def gap(point: tuple[float, float], node: int) -> float:
        return float(haversine_km(np.array([point[0]]), np.array([point[1]]),
                                  np.array([graph.lat[node]]),
                                  np.array([graph.lng[node]]))[0]) * 1000

    start_gap, goal_gap = gap(origin, start), gap(destination, goal)
    coordinates = ([[origin[1], origin[0]]]
                   + [[float(graph.lng[i]), float(graph.lat[i])] for i in path]
                   + [[destination[1], destination[0]]])

    return {
        "coordinates": coordinates,
        "metres": metres + start_gap + goal_gap,
        "street_metres": metres,
        "snap_metres": {"origin": round(start_gap), "destination": round(goal_gap)},
        "nodes": len(path),
        "graph": {"nodes": graph.n_nodes, "edges": graph.n_edges},
        "source": "OpenStreetMap via Overpass",
        "follows_streets": True,
    }
