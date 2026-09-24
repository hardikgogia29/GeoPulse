"""Tests for the A* walking router.

The graph properties are asserted rather than assumed, because the whole claim of
this module is "real A* over a real graph" - if the graph is malformed or the
heuristic is inadmissible, the paths are just plausible-looking noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.serving.routing import (
    DETOUR_FACTOR, MAX_EDGE_KM, MAX_REASONABLE_WALK_M, astar, build_graph,
    route_to_best_station, _heuristic, _pressure_penalty,
)
from src.utils.geo import haversine_km


@pytest.fixture(scope="module")
def graph():
    return build_graph("h3", 8)


def test_graph_is_built_and_symmetric(graph):
    assert graph.n_nodes > 2000
    assert graph.n_edges > 1000
    # every edge must exist in both directions or A* can find one-way paths that
    # are not walkable in reverse
    for a, neighbours in enumerate(graph.adjacency):
        for b, weight in neighbours:
            back = dict(graph.adjacency[b])
            assert a in back, f"edge {a}->{b} is not mirrored"
            assert back[a] == pytest.approx(weight)


def test_no_edge_exceeds_the_walking_cap(graph):
    """The cap is what stops the graph inventing links across rivers."""
    for a, neighbours in enumerate(graph.adjacency):
        for b, metres in neighbours:
            km = metres / (1000 * DETOUR_FACTOR)
            assert km <= MAX_EDGE_KM + 1e-9


def test_heuristic_is_admissible(graph):
    """A* is only optimal if the heuristic never over-estimates the true cost.

    Checked against the real edge weights: for any edge, h(a) must not exceed
    cost(a,b) + h(b) - the triangle inequality that makes it consistent, which
    implies admissible.
    """
    rng = np.random.default_rng(0)
    goal = int(rng.integers(0, graph.n_nodes))
    for a in rng.integers(0, graph.n_nodes, size=200):
        a = int(a)
        for b, weight in graph.adjacency[a]:
            assert _heuristic(graph, a, goal) <= weight + _heuristic(graph, b, goal) + 1e-6


def test_astar_finds_a_path_and_reports_its_true_length(graph):
    start = graph.index[graph.ids[0]]
    goal = graph.index[graph.ids[500]]
    found = astar(graph, start, goal)
    if found is None:
        pytest.skip("those two stations are in different components")
    path, metres = found
    assert path[0] == start and path[-1] == goal
    # the reported distance must equal the sum of the edges actually traversed
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += dict(graph.adjacency[a])[b]
    assert total == pytest.approx(metres, rel=1e-9)


def test_astar_is_at_least_as_short_as_the_straight_line(graph):
    """A path through a graph can never beat flying directly."""
    start = graph.index[graph.ids[10]]
    goal = graph.index[graph.ids[400]]
    found = astar(graph, start, goal)
    if found is None:
        pytest.skip("disconnected")
    _, metres = found
    straight = haversine_km(
        np.array([graph.lat[start]]), np.array([graph.lng[start]]),
        np.array([graph.lat[goal]]), np.array([graph.lng[goal]]))[0] * 1000
    assert metres >= straight - 1e-6


def test_astar_returns_none_when_disconnected(graph):
    isolated = [i for i, a in enumerate(graph.adjacency) if not a]
    if not isolated:
        pytest.skip("no isolated station in this build")
    other = next(i for i, a in enumerate(graph.adjacency) if a)
    assert astar(graph, isolated[0], other) is None


def test_same_node_is_a_zero_length_path(graph):
    path, metres = astar(graph, 5, 5)
    assert path == [5] and metres == 0.0


# ------------------------------------------------------------------ target choice
def test_pressure_penalty_direction():
    """Draining docks are penalised; filling ones are not."""
    draining = _pressure_penalty(pickups=40, dropoffs=5)
    filling = _pressure_penalty(pickups=5, dropoffs=40)
    balanced = _pressure_penalty(pickups=20, dropoffs=20)
    assert draining > balanced >= filling
    assert filling == 0.0


def test_a_dead_dock_is_not_treated_as_a_good_bet():
    """Zero predicted flow means no evidence, not 'plenty available'."""
    assert _pressure_penalty(0.0, 0.0) > 0.0


ORIGIN = (40.75, -73.98)


def _candidate(name, distance_km, pickups, dropoffs, station_id):
    """A station_forecast row whose coordinates actually match its stated distance.

    The router scores on `distance_km` but re-derives the drawn legs from the
    coordinates, so a fixture that disagrees with itself tests nothing real.
    """
    lat = ORIGIN[0] + distance_km / 111.32          # degrees latitude per km
    return {
        "station": {"station_id": station_id, "station_name": name,
                    "lat": lat, "lng": ORIGIN[1], "region_id": "r",
                    "pickup_share": 0.1, "dropoff_share": 0.1,
                    "lifetime_trips": 9999, "share_confidence": "high",
                    "distance_km": distance_km},
        "predicted_pickups": pickups, "predicted_dropoffs": dropoffs,
    }


def _pair(graph):
    """A real station, its closest real neighbour, and an origin beside the first.

    The two must be close enough that the availability penalty can plausibly
    outweigh the extra walk - a pair a kilometre apart would make the test pass or
    fail on geography rather than on the router's logic.
    """
    for a in range(graph.n_nodes):
        if not graph.adjacency[a]:
            continue
        b, metres = min(graph.adjacency[a], key=lambda kv: kv[1])
        if metres <= 350:
            origin = (float(graph.lat[a]) + 0.0003, float(graph.lng[a]))  # ~33 m
            return a, b, origin
    pytest.skip("no station pair close enough to make the trade-off meaningful")


def test_router_can_prefer_a_further_dock_that_is_filling(graph):
    """The whole reason a forecast sits behind the router."""
    a, b, origin = _pair(graph)
    near_draining = _candidate("near", 0.03, 40, 4, graph.ids[a])
    far_filling = _candidate("far", 0.30, 4, 40, graph.ids[b])
    out = route_to_best_station(*origin, [near_draining, far_filling],
                                prefer_available=True)
    assert out["ok"]
    assert out["target"]["station"]["station_id"] == graph.ids[b]
    assert out["chose_nearest"] is False


def test_prefer_available_false_takes_the_nearest(graph):
    """With the forecast switched off it degrades to plain nearest-dock."""
    a, b, origin = _pair(graph)
    near_draining = _candidate("near", 0.03, 40, 4, graph.ids[a])
    far_filling = _candidate("far", 0.30, 4, 40, graph.ids[b])
    out = route_to_best_station(*origin, [near_draining, far_filling],
                                prefer_available=False)
    assert out["target"]["station"]["station_id"] == graph.ids[a]
    assert out["chose_nearest"] is True


def test_unreachable_distances_are_flagged_not_presented_as_a_walk(graph):
    """Large parts of Brooklyn and Queens have no docks at all; the nearest can be
    kilometres away, and calling that a walk would be absurd."""
    # a real station, and an origin genuinely far from it - the router measures from
    # the graph's coordinates, so the fixture cannot fake the distance
    node = graph.index[graph.ids[0]]
    origin = (float(graph.lat[node]) - 0.09, float(graph.lng[node]))   # ~10 km south
    far = _candidate("far away", 9.0, 5, 5, graph.ids[0])
    out = route_to_best_station(*origin, [far])
    assert out["ok"]
    assert out["walkable"] is False
    assert out["walk_metres"] > MAX_REASONABLE_WALK_M
    assert "beyond walking distance" in out["reachability"]


def test_geometry_starts_at_the_origin(graph):
    row = _candidate("dock", 0.2, 10, 10, graph.ids[0])
    out = route_to_best_station(40.7500, -73.9800, [row])
    first = out["geometry"]["coordinates"][0]
    assert first == [-73.9800, 40.7500]  # noqa: PLR2004 - the origin, unchanged
    assert len(out["geometry"]["coordinates"]) >= 2


def test_no_candidates_is_an_error_not_an_empty_route():
    out = route_to_best_station(*ORIGIN, [])
    assert out["ok"] is False
