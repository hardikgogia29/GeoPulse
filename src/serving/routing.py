
from __future__ import annotations

import functools
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from src.serving.stations import Station, station_table
from src.utils.geo import haversine_km

#: straight-line distance under-states walking on a street grid; 1.30 is the usual
#: correction for gridded cities and keeps the A* heuristic admissible
DETOUR_FACTOR = 1.30
#: two stations further apart than this are not treated as walkable neighbours
MAX_EDGE_KM = 0.75
#: how many neighbours each station links to
NEIGHBOURS = 6
#: how many stations the starting point connects into
ORIGIN_LINKS = 6
#: a fully draining dock costs this many extra "virtual metres" when choosing a target
PRESSURE_PENALTY_M = 550.0
#: past this, calling it a "walk" is silly - large parts of Brooklyn and Queens have
#: no Citi Bike coverage at all, and the nearest dock can be kilometres off. The route
#: is still returned, flagged, so the UI can say "too far" instead of implying a stroll
MAX_REASONABLE_WALK_M = 2000.0


@dataclass
class Graph:
    ids: list[str]
    names: list[str]
    lat: np.ndarray
    lng: np.ndarray
    adjacency: list[list[tuple[int, float]]]   # node -> [(neighbour, metres)]
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {sid: i for i, sid in enumerate(self.ids)}

    @property
    def n_nodes(self) -> int:
        return len(self.ids)

    @property
    def n_edges(self) -> int:
        return sum(len(a) for a in self.adjacency) // 2


@functools.lru_cache(maxsize=4)
def build_graph(spatial: str = "h3", resolution: int = 8) -> Graph:
    """k-nearest-neighbour walking graph over every station, built once and cached."""
    frame = station_table("v1", spatial, resolution)
    lat = frame["lat"].to_numpy()
    lng = frame["lng"].to_numpy()
    n = len(lat)

    # 2,459 stations -> a 2,459^2 distance matrix is ~48 MB of float64, which is fine
    # and far simpler than a spatial index for a graph built once per process
    distances = haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    np.fill_diagonal(distances, np.inf)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    order = np.argsort(distances, axis=1)[:, :NEIGHBOURS]
    seen: set[tuple[int, int]] = set()
    for a in range(n):
        for b in order[a]:
            b = int(b)
            km = float(distances[a, b])
            if km > MAX_EDGE_KM:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            metres = km * 1000 * DETOUR_FACTOR
            adjacency[a].append((b, metres))
            adjacency[b].append((a, metres))

    return Graph(ids=frame["station_id"].to_list(),
                 names=frame["station_name"].to_list(),
                 lat=lat, lng=lng, adjacency=adjacency)


def _heuristic(graph: Graph, node: int, goal: int) -> float:
    """Straight-line metres, scaled by the same detour factor the edges use.

    Admissible: every real path between two points is at least the straight line, and
    both sides carry the same factor, so this never over-estimates and A* stays
    optimal.
    """
    km = haversine_km(np.array([graph.lat[node]]), np.array([graph.lng[node]]),
                      np.array([graph.lat[goal]]), np.array([graph.lng[goal]]))[0]
    return float(km) * 1000 * DETOUR_FACTOR


def astar(graph: Graph, start: int, goal: int) -> tuple[list[int], float] | None:
    """Shortest walking path between two station nodes. None when disconnected."""
    if start == goal:
        return [start], 0.0
    open_heap: list[tuple[float, int]] = [(0.0, start)]
    came_from: dict[int, int] = {}
    best = {start: 0.0}
    closed: set[int] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1], best[current]
        if current in closed:
            continue
        closed.add(current)
        for neighbour, weight in graph.adjacency[current]:
            if neighbour in closed:
                continue
            candidate = best[current] + weight
            if candidate < best.get(neighbour, math.inf):
                best[neighbour] = candidate
                came_from[neighbour] = current
                heapq.heappush(
                    open_heap,
                    (candidate + _heuristic(graph, neighbour, goal), neighbour))
    return None


def _pressure_penalty(pickups: float, dropoffs: float) -> float:
    """Virtual metres added for a dock that is emptying.

    Turns "how far" and "will anything be there" into one comparable number, so the
    router can prefer a slightly longer walk to a dock that is actually filling.
    """
    total = pickups + dropoffs
    if total < 1.0:
        return PRESSURE_PENALTY_M * 0.5      # dead dock: not obviously a good bet
    pressure = (dropoffs - pickups) / total  # -1 draining .. +1 filling
    return float(np.clip(-pressure, 0.0, 1.0) * PRESSURE_PENALTY_M)


def route_to_best_station(lat: float, lng: float, candidates: list[dict], *,
                          spatial: str = "h3", resolution: int = 8,
                          prefer_available: bool = True) -> dict:
    """Choose a dock worth walking to, then A* a path to it.

    `candidates` are `station_forecast` rows - each carries the station and its
    predicted pickups/dropoffs, so the choice can weigh availability against
    distance instead of blindly taking the closest.
    """
    if not candidates:
        return {"ok": False, "error": "no stations near that point"}
    graph = build_graph(spatial, resolution)

    scored = []
    for row in candidates:
        station = row["station"]
        node = graph.index.get(station["station_id"])
        if node is None:
            continue
        # Measure from the graph's own coordinates rather than the caller's
        # `distance_km`. In production both come from the station registry and
        # agree, but the drawn legs are derived from the graph, so scoring on a
        # different number would let the chosen target and the reported walk
        # silently disagree.
        walk_km = float(haversine_km(np.array([lat]), np.array([lng]),
                                     np.array([graph.lat[node]]),
                                     np.array([graph.lng[node]]))[0])
        walk_m = walk_km * 1000 * DETOUR_FACTOR
        penalty = (_pressure_penalty(row["predicted_pickups"],
                                     row["predicted_dropoffs"])
                   if prefer_available else 0.0)
        scored.append({"row": row, "node": node, "walk_m": walk_m,
                       "penalty_m": penalty, "score": walk_m + penalty})
    if not scored:
        return {"ok": False, "error": "no candidate station is in the walking graph"}

    scored.sort(key=lambda s: s["score"])
    chosen = scored[0]
    nearest = min(scored, key=lambda s: s["walk_m"])

    # The origin is an arbitrary point, not a station, so it is not in the graph.
    # If the chosen dock is itself within a walkable hop, the honest route is simply
    # to walk to it - threading through an intermediate station would both look wrong
    # and overstate the distance. A* earns its place only when the target is beyond
    # one hop, where the graph supplies the intermediate docks.
    direct_km = chosen["walk_m"] / (1000 * DETOUR_FACTOR)
    if direct_km <= MAX_EDGE_KM:
        path_nodes, path_m, hops = [chosen["node"]], 0.0, 0
    else:
        entry = nearest["node"]
        found = astar(graph, entry, chosen["node"])
        path_nodes, path_m = found if found else ([entry], 0.0)
        hops = max(len(path_nodes) - 1, 0)
    reached = path_nodes[-1] == chosen["node"]

    coordinates = [[lng, lat]] + [[float(graph.lng[i]), float(graph.lat[i])]
                                  for i in path_nodes]

    # Upgrade the drawn line to a real pavement-following path where we can. The
    # station graph decides *which* dock; OpenStreetMap decides *how you walk there*.
    # Imported here rather than at module scope because streets.py reuses this
    # module's Graph and astar.
    street = None
    try:
        from src.serving.streets import StreetError, walk_route

        target_node = path_nodes[-1]
        street = walk_route((lat, lng),
                            (float(graph.lat[target_node]),
                             float(graph.lng[target_node])))
    except Exception:                                   # noqa: BLE001
        # no network, an Overpass rate limit, or nothing mapped nearby - fall back
        # to the straight-line station path rather than failing the request
        street = None
    legs = []
    previous = (lat, lng)
    for node in path_nodes:
        here = (float(graph.lat[node]), float(graph.lng[node]))
        km = haversine_km(np.array([previous[0]]), np.array([previous[1]]),
                          np.array([here[0]]), np.array([here[1]]))[0]
        legs.append({"to": graph.names[node],
                     "metres": round(float(km) * 1000 * DETOUR_FACTOR)})
        previous = here

    # legs[0] is the walk from the origin into the graph; path_m covers the hops
    # after that. When the target was one direct hop, path_m is zero by construction.
    # legs[0] is the walk from the origin into the graph; path_m covers the hops
    entry_m = legs[0]["metres"] if legs else 0.0
    total_m = entry_m + path_m
    # a street path is the truth when we have one: it measures real pavement rather
    # than a straight line scaled by a detour guess
    if street is not None:
        total_m = street["metres"]
        coordinates = street["coordinates"]
    too_far = total_m > MAX_REASONABLE_WALK_M
    return {
        "ok": True,
        "too_far": too_far,
        "walkable": not too_far,
        "target": chosen["row"],
        "reached_target": reached,
        "graph_hops": hops,
        "chose_nearest": chosen["node"] == nearest["node"],
        "walk_metres": round(total_m),
        "walk_minutes": round(total_m / 80.0, 1),          # ~4.8 km/h
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "legs": legs,
        "alternatives": [{
            "station": s["row"]["station"]["station_name"],
            "walk_metres": round(s["walk_m"]),
            "availability_penalty_metres": round(s["penalty_m"]),
            "score": round(s["score"]),
        } for s in scored[:4]],
        "graph": {"nodes": graph.n_nodes, "edges": graph.n_edges,
                  "neighbours_per_node": NEIGHBOURS,
                  "max_edge_km": MAX_EDGE_KM},
        "follows_streets": street is not None,
        "street_detail": street and {
            "nodes": street["nodes"], "graph": street["graph"],
            "snap_metres": street["snap_metres"], "source": street["source"]},
        "method": ("A* over the OpenStreetMap walking network"
                   if street is not None
                   else "A* over the station-proximity graph"),
        "reachability": (
            f"The nearest dock is {total_m / 1000:.1f} km away - beyond walking "
            f"distance. Citi Bike does not cover this part of the city."
            if too_far else "within walking distance"),
        "caveat": (
            "The dock is chosen on a station graph weighted by distance and forecast "
            "availability; the walking line is A* over the OpenStreetMap pedestrian "
            "network, so it follows real streets and the distance is measured, not "
            "estimated."
            if street is not None else
            "Street data was unavailable, so this falls back to a straight line "
            f"between docks with a {DETOUR_FACTOR}x detour factor applied to the "
            f"distance. The line cuts corners; the length is still realistic."),
    }
