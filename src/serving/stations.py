
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.utils.config import resolve_path
from src.utils.geo import haversine_km

#: below this many lifetime trips a station's share is too noisy to trust
MIN_TRIPS_FOR_CONFIDENT_SHARE = 2_000


@dataclass(frozen=True)
class Station:
    station_id: str
    station_name: str
    lat: float
    lng: float
    region_id: str
    pickup_share: float
    dropoff_share: float
    lifetime_trips: int
    share_confidence: str  # "high" | "low"
    distance_km: float = 0.0

    def as_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "lat": self.lat,
            "lng": self.lng,
            "region_id": self.region_id,
            "pickup_share": round(self.pickup_share, 4),
            "dropoff_share": round(self.dropoff_share, 4),
            "lifetime_trips": self.lifetime_trips,
            "share_confidence": self.share_confidence,
            "distance_km": round(self.distance_km, 3),
        }


@functools.lru_cache(maxsize=8)
def station_table(cfg_key: str, spatial: str, resolution: int) -> pl.DataFrame:
    """Stations indexed to regions, with their share of each region's trips.

    Cached per (spatial, resolution): indexing 2,459 points is fast, but the app
    calls this on every request and there is no reason to redo it.
    """
    from src.utils.config import Config, load_config

    cfg = load_config(spatial, "lightgbm")
    key = "level" if cfg["spatial"].get("system") == "s2" else "resolution"
    cfg = Config({**cfg, "spatial": {**cfg["spatial"], key: resolution,
                                     "resolution": resolution}})
    from src.spatial.h3_indexer import make_indexer

    indexer = make_indexer(cfg)
    tag = f"{indexer.name}{indexer.resolution}"

    registry = pl.read_parquet(
        resolve_path(cfg, "paths.spatial") / "station_registry.parquet"
    ).select(["station_id", "station_name", "lat", "lng", "pickups", "dropoffs"])
    # the registry stores counts as Decimal(scale=0); dividing those keeps scale 0,
    # so every share would truncate to exactly 0. Cast before any arithmetic.
    registry = registry.with_columns([
        pl.col("pickups").cast(pl.Float64), pl.col("dropoffs").cast(pl.Float64)])

    # only regions the model actually forecasts; a station in a dropped region has
    # no forecast to be given a share of
    active = set(pl.read_parquet(
        resolve_path(cfg, "paths.spatial") / f"regions_{tag}.parquet"
    )["region_id"].to_list())

    region_ids = [indexer.index(lat, lng) for lat, lng
                  in zip(registry["lat"].to_list(), registry["lng"].to_list())]
    frame = registry.with_columns(pl.Series("region_id", region_ids)).filter(
        pl.col("region_id").is_in(active))

    totals = frame.group_by("region_id").agg([
        pl.col("pickups").sum().alias("region_pickups"),
        pl.col("dropoffs").sum().alias("region_dropoffs"),
    ])
    return (frame.join(totals, on="region_id")
            .with_columns([
                # guard the divide: a region whose stations all recorded zero of one
                # direction falls back to an equal split rather than a NaN share
                pl.when(pl.col("region_pickups") > 0)
                  .then(pl.col("pickups") / pl.col("region_pickups"))
                  .otherwise(1.0 / pl.len().over("region_id"))
                  .alias("pickup_share"),
                pl.when(pl.col("region_dropoffs") > 0)
                  .then(pl.col("dropoffs") / pl.col("region_dropoffs"))
                  .otherwise(1.0 / pl.len().over("region_id"))
                  .alias("dropoff_share"),
                (pl.col("pickups") + pl.col("dropoffs")).alias("lifetime_trips"),
            ])
            .with_columns(
                pl.when(pl.col("lifetime_trips") >= MIN_TRIPS_FOR_CONFIDENT_SHARE)
                  .then(pl.lit("high")).otherwise(pl.lit("low"))
                  .alias("share_confidence")
            ))


def _to_station(row: dict, distance_km: float = 0.0) -> Station:
    return Station(
        station_id=row["station_id"], station_name=row["station_name"],
        lat=row["lat"], lng=row["lng"], region_id=row["region_id"],
        pickup_share=row["pickup_share"], dropoff_share=row["dropoff_share"],
        lifetime_trips=int(row["lifetime_trips"]),
        share_confidence=row["share_confidence"], distance_km=distance_km,
    )


def nearest_stations(lat: float, lng: float, k: int = 5, *,
                     spatial: str = "h3", resolution: int = 8) -> list[Station]:
    """The k closest stations to a point, nearest first, with great-circle distance."""
    frame = station_table("v1", spatial, resolution)
    distances = haversine_km(
        np.full(frame.height, lat), np.full(frame.height, lng),
        frame["lat"].to_numpy(), frame["lng"].to_numpy())
    order = np.argsort(distances)[:k]
    rows = frame.to_dicts()
    return [_to_station(rows[int(i)], float(distances[int(i)])) for i in order]


def stations_in_region(region_id: str, *, spatial: str = "h3",
                       resolution: int = 8) -> list[Station]:
    """Every station the model's region contains, busiest first."""
    frame = station_table("v1", spatial, resolution).filter(
        pl.col("region_id") == region_id).sort("lifetime_trips", descending=True)
    return [_to_station(row) for row in frame.to_dicts()]


def split_to_station(region_pickups: float, region_dropoffs: float,
                     station: Station) -> dict:
    """Apply a station's historical share to its region's forecast.

    Returns the station-level numbers **plus the region numbers they came from**, so
    a caller can always show the modelled quantity next to the derived one.
    """
    return {
        "predicted_pickups": region_pickups * station.pickup_share,
        "predicted_dropoffs": region_dropoffs * station.dropoff_share,
        "region_pickups": region_pickups,
        "region_dropoffs": region_dropoffs,
        "pickup_share": station.pickup_share,
        "dropoff_share": station.dropoff_share,
        "share_confidence": station.share_confidence,
        "method": "region_forecast_x_historical_station_share",
        "approximate": True,
    }
