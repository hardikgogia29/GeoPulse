
from __future__ import annotations

import duckdb
import h3


class H3Indexer:
    name = "h3"

    def __init__(self, resolution: int) -> None:
        self.resolution = int(resolution)

    def index(self, lat: float, lng: float) -> str:
        return h3.latlng_to_cell(lat, lng, self.resolution)

    def neighbors(self, region_id: str, k: int = 1) -> list[str]:
        # h3-py v4 returns a list, not a set
        return sorted(cell for cell in h3.grid_disk(region_id, k) if cell != region_id)

    def centroid(self, region_id: str) -> tuple[float, float]:
        return tuple(h3.cell_to_latlng(region_id))

    def area_km2(self, region_id: str) -> float:
        return float(h3.cell_area(region_id, unit="km^2"))

    def boundary(self, region_id: str) -> list[tuple[float, float]]:
        return [tuple(point) for point in h3.cell_to_boundary(region_id)]

    def sql_index_expr(self, lat_col: str, lng_col: str) -> str:
        return f"h3_latlng_to_cell_string({lat_col}, {lng_col}, {self.resolution})"

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"H3Indexer(resolution={self.resolution})"


def load_h3_extension(con: duckdb.DuckDBPyConnection) -> None:
    """Make `h3_*` SQL functions available on this connection."""
    try:
        con.execute("LOAD h3")
    except duckdb.Error:
        con.execute("INSTALL h3 FROM community")
        con.execute("LOAD h3")


def make_indexer(cfg) -> H3Indexer:
    """Build the indexer named by the loaded spatial config overlay."""
    system = cfg.dotted("spatial.system")
    if system == "h3":
        return H3Indexer(cfg.dotted("spatial.resolution"))
    if system == "s2":  # implemented in Phase 5
        from src.spatial.s2_indexer import S2Indexer

        return S2Indexer(cfg.dotted("spatial.level"))
    raise ValueError(f"unknown spatial system: {system}")
