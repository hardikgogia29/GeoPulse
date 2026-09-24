"""Tests for the OpenStreetMap walking graph.

Built from a synthetic Overpass payload so these run offline - the network path is
exercised by the app, but the parsing and graph construction is pure logic and
should not need Overpass to be up.
"""

from __future__ import annotations

import pytest

from src.serving.routing import astar
from src.serving.streets import StreetError, build_graph, snap


def _payload():
    """A 3x1 ladder of street nodes: 1-2-3 along a road, plus a detached node."""
    return {"elements": [
        {"type": "node", "id": 1, "lat": 40.7500, "lon": -73.9900},
        {"type": "node", "id": 2, "lat": 40.7500, "lon": -73.9890},
        {"type": "node", "id": 3, "lat": 40.7500, "lon": -73.9880},
        {"type": "node", "id": 9, "lat": 40.7600, "lon": -73.9700},
        {"type": "way", "id": 100, "nodes": [1, 2, 3]},
    ]}


def test_ways_become_consecutive_edges():
    graph = build_graph(_payload())
    # only the three nodes referenced by a way are kept; the loose node is dropped
    assert graph.n_nodes == 3
    assert graph.n_edges == 2
    assert {b for b, _ in graph.adjacency[graph.index["2"]]} == {
        graph.index["1"], graph.index["3"]}


def test_edges_are_bidirectional():
    graph = build_graph(_payload())
    for a, neighbours in enumerate(graph.adjacency):
        for b, weight in neighbours:
            assert a in dict(graph.adjacency[b])
            assert dict(graph.adjacency[b])[a] == pytest.approx(weight)


def test_edge_weights_are_true_ground_distance_not_scaled():
    """Street segments are real geometry, so no detour factor may be applied."""
    graph = build_graph(_payload())
    weight = dict(graph.adjacency[graph.index["1"]])[graph.index["2"]]
    # 0.001 degrees of longitude at this latitude is ~84 m
    assert 75 < weight < 95


def test_astar_runs_over_a_street_graph():
    graph = build_graph(_payload())
    path, metres = astar(graph, graph.index["1"], graph.index["3"])
    assert [graph.ids[i] for i in path] == ["1", "2", "3"]
    assert metres == pytest.approx(
        sum(dict(graph.adjacency[a])[b] for a, b in zip(path, path[1:])))


def test_a_payload_with_no_ways_is_an_error():
    with pytest.raises(StreetError, match="no walkable streets"):
        build_graph({"elements": [
            {"type": "node", "id": 1, "lat": 40.75, "lon": -73.99}]})


def test_way_referencing_a_missing_node_is_skipped_not_fatal():
    """Overpass can return a way whose nodes fall outside the requested box."""
    payload = _payload()
    payload["elements"].append({"type": "way", "id": 200, "nodes": [3, 12345]})
    graph = build_graph(payload)
    assert graph.n_nodes == 3          # the dangling reference contributed nothing


def test_snap_finds_the_closest_node():
    graph = build_graph(_payload())
    assert graph.ids[snap(graph, 40.7500, -73.98805)] == "3"
    assert graph.ids[snap(graph, 40.7501, -73.99010)] == "1"
