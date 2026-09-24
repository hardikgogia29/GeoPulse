
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpatialIndexer(Protocol):
    name: str
    resolution: int

    def index(self, lat: float, lng: float) -> str:
        """Point -> region id."""

    def neighbors(self, region_id: str, k: int = 1) -> list[str]:
        """The k-ring around a region, excluding the region itself."""

    def centroid(self, region_id: str) -> tuple[float, float]:
        """(lat, lng) of the region centre."""

    def area_km2(self, region_id: str) -> float:
        """Region area in square kilometres."""

    def boundary(self, region_id: str) -> list[tuple[float, float]]:
        """Region outline as [(lat, lng), ...]."""

    def sql_index_expr(self, lat_col: str, lng_col: str) -> str:
        """A SQL expression assigning a region id, for in-database indexing.

        79M trips x 2 endpoints is far too many round trips for a Python loop, so
        the bulk assignment happens in DuckDB. Implementations must guarantee this
        agrees with :meth:`index` exactly - `tests/test_spatial.py` asserts it.
        """
