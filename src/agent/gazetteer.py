
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import polars as pl

from src.utils.config import load_config, resolve_path

#: common ways people write NYC street names, mapped to the registry's spelling
ALIASES = {
    "ave": "av", "avenue": "av", "street": "st", "square": "sq",
    "park": "park", "west": "w", "east": "e", "north": "n", "south": "s",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    out = []
    for token in tokens:
        token = re.sub(r"(\d+)(st|nd|rd|th)$", r"\1", token)  # 21st -> 21
        out.append(ALIASES.get(token, token))
    return [t for t in out if t not in {"and", "the", "at", "near", "of", "ny", "nyc",
                                        "new", "york", "usa", "i", "am", "im"}]


#: below this, a match is a coincidence of common words ("st", "park") rather than a
#: real location. Our gazetteer has genuine holes - there is no "Wall St" station,
#: for instance - and quietly returning the nearest weak match would put the rider
#: in the wrong borough. Callers must surface `is_confident` and ask, not guess.
CONFIDENCE_FLOOR = 0.50


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lng: float
    source: str          # "station" | "landmark"
    score: float
    detail: str = ""

    @property
    def is_confident(self) -> bool:
        return self.score >= CONFIDENCE_FLOOR

    def as_dict(self) -> dict:
        return {"name": self.name, "lat": self.lat, "lng": self.lng,
                "source": self.source, "score": round(self.score, 3),
                "detail": self.detail, "is_confident": self.is_confident}


@functools.lru_cache(maxsize=1)
def _entries() -> list[tuple[str, float, float, str, str, frozenset]]:
    cfg = load_config("h3", "lightgbm")
    rows: list[tuple[str, float, float, str, str, frozenset]] = []

    registry = pl.read_parquet(
        resolve_path(cfg, "paths.spatial") / "station_registry.parquet"
    ).select(["station_name", "lat", "lng", "pickups", "dropoffs"])
    for r in registry.to_dicts():
        rows.append((r["station_name"], float(r["lat"]), float(r["lng"]),
                     "station", "Citi Bike station",
                     frozenset(normalise(r["station_name"]))))

    geo = (pl.read_parquet(resolve_path(cfg, "paths.external") / "event_geocodes.parquet")
           .filter(pl.col("status") == "ok")
           .select(["event_location", "label", "lat", "lng", "matched_borough"])
           .unique(subset=["label"]))
    for r in geo.to_dicts():
        label = r["label"] or r["event_location"]
        rows.append((label, float(r["lat"]), float(r["lng"]), "landmark",
                     r["matched_borough"] or "NYC landmark",
                     frozenset(normalise(label)) | frozenset(
                         normalise(r["event_location"] or ""))))
    return rows


def search(query: str, limit: int = 5) -> list[Place]:
    """Best matches for a free-text location, best first."""
    wanted = set(normalise(query))
    if not wanted:
        return []
    lowered = query.lower().strip()
    scored: list[tuple[float, Place]] = []
    for name, lat, lng, source, detail, tokens in _entries():
        if not tokens:
            continue
        overlap = len(wanted & tokens)
        if not overlap:
            continue
        # Jaccard keeps a long landmark name from beating an exact short match
        score = overlap / len(wanted | tokens)
        if lowered in name.lower():
            score += 0.35
        # The "all your words are present" bonus only means something for a real
        # phrase. For a single common word it fires on coincidence - "why is that the
        # case?" leaves "case", which is contained in "Case St & 94 St" - and would
        # push a stray word above the confidence floor.
        if len(wanted) >= 2 and wanted <= tokens:
            score += 0.25
        # a tie between a station and a landmark goes to the station: riders are
        # asking about docks, and station names are the more precise coordinate
        if source == "station":
            score += 0.02
        scored.append((score, Place(name, lat, lng, source, score, detail)))
    scored.sort(key=lambda kv: -kv[0])
    seen, out = set(), []
    for _, place in scored:
        key = (round(place.lat, 4), round(place.lng, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(place)
        if len(out) >= limit:
            break
    return out
