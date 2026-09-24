"""The spatial indexer contract, and that SQL and Python indexing agree exactly.

Bulk indexing happens in DuckDB for speed; the Python API is used for neighbours and
geometry. If those two ever disagree the panel and the neighbour features would be
built on different cells, so the agreement is asserted rather than assumed.
"""

import duckdb
import pytest

from src.spatial.base import SpatialIndexer
from src.spatial.h3_indexer import H3Indexer, load_h3_extension

# real NYC points spread across the service area
POINTS = [
    (40.7831, -73.9712),   # Central Park
    (40.71146364, -74.00552427),  # Spruce St & Nassau St
    (40.724861, -73.992131),      # E 1 St & Bowery
    (40.6334, -74.0166),          # Bay Ridge
    (40.8900, -73.9000),          # Bronx
    (40.7420, -74.0350),          # Hoboken / Jersey City side
]


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    load_h3_extension(connection)
    yield connection
    connection.close()


@pytest.mark.parametrize("resolution", [8, 9, 10])
def test_sql_and_python_indexing_agree(con, resolution):
    indexer = H3Indexer(resolution)
    values = ", ".join(f"({lat}, {lng})" for lat, lng in POINTS)
    expr = indexer.sql_index_expr("lat", "lng")
    rows = con.execute(
        f"SELECT lat, lng, {expr} AS region FROM (VALUES {values}) AS t(lat, lng)"
    ).fetchall()
    for lat, lng, sql_region in rows:
        assert sql_region == indexer.index(lat, lng), f"({lat}, {lng}) at res {resolution}"


def test_h3_indexer_satisfies_the_protocol():
    assert isinstance(H3Indexer(9), SpatialIndexer)


@pytest.mark.parametrize("resolution", [8, 9, 10])
def test_neighbors_are_the_first_ring(resolution):
    indexer = H3Indexer(resolution)
    region = indexer.index(*POINTS[0])
    ring = indexer.neighbors(region)
    # a hexagon has 6 neighbours; the 122 pentagons worldwide have 5, and none of
    # them are near NYC
    assert len(ring) == 6
    assert region not in ring
    assert len(set(ring)) == len(ring)
    # neighbourhood is symmetric
    for neighbour in ring:
        assert region in indexer.neighbors(neighbour)


def test_centroid_round_trips_to_the_same_cell():
    for resolution in (8, 9, 10):
        indexer = H3Indexer(resolution)
        for lat, lng in POINTS:
            region = indexer.index(lat, lng)
            assert indexer.index(*indexer.centroid(region)) == region


def test_area_shrinks_with_resolution():
    region_areas = []
    for resolution in (8, 9, 10):
        indexer = H3Indexer(resolution)
        region_areas.append(indexer.area_km2(indexer.index(*POINTS[0])))
    assert region_areas[0] > region_areas[1] > region_areas[2]
    # each H3 step is roughly a 7x area reduction
    assert 5 < region_areas[0] / region_areas[1] < 9
    assert 5 < region_areas[1] / region_areas[2] < 9


def test_boundary_is_a_closed_polygon():
    indexer = H3Indexer(9)
    outline = indexer.boundary(indexer.index(*POINTS[0]))
    assert len(outline) == 6
    assert all(len(point) == 2 for point in outline)
    lats = [p[0] for p in outline]
    lngs = [p[1] for p in outline]
    centroid = indexer.centroid(indexer.index(*POINTS[0]))
    assert min(lats) < centroid[0] < max(lats)
    assert min(lngs) < centroid[1] < max(lngs)


def test_distinct_points_in_different_cells_stay_distinct():
    indexer = H3Indexer(10)
    regions = {indexer.index(lat, lng) for lat, lng in POINTS}
    assert len(regions) == len(POINTS)
