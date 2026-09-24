
from __future__ import annotations

import os
import time
from datetime import date, timedelta

import polars as pl
import requests

OBSERVATION_SCHEMA: dict[str, type[pl.DataType]] = {
    "id": pl.String,
    "link_id": pl.String,
    "speed": pl.Float64,
    "travel_time": pl.Float64,
    "status": pl.String,
    "data_as_of": pl.String,
}

TS_FORMATS = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S"]


class SocrataClient:
    """Minimal Socrata SODA 2.1 client with retry/backoff and optional app token."""

    def __init__(self, cfg, log):
        self.domain = cfg.dotted("traffic.domain")
        self.dataset = cfg.dotted("traffic.dataset_id")
        self.timeout = cfg.dotted("traffic.request_timeout_seconds")
        self.max_retries = cfg.dotted("traffic.max_retries")
        self.backoff = cfg.dotted("traffic.retry_backoff_seconds")
        self.log = log
        self.session = requests.Session()
        token = os.environ.get(cfg.dotted("traffic.app_token_env") or "", "")
        if token:
            self.session.headers["X-App-Token"] = token
            log.info("using Socrata app token from $%s", cfg.dotted("traffic.app_token_env"))
        else:
            log.info(
                "no Socrata app token found in $%s - anonymous requests work but are "
                "throttled harder", cfg.dotted("traffic.app_token_env")
            )

    def url(self, fmt: str = "csv") -> str:
        return f"https://{self.domain}/resource/{self.dataset}.{fmt}"

    def get(self, params: dict, fmt: str = "csv") -> requests.Response:
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(self.url(fmt), params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001 - retry on any transport failure
                last = exc
                wait = self.backoff * attempt
                self.log.warning("request failed (%s), retry %d/%d in %ds",
                                 type(exc).__name__, attempt, self.max_retries, wait)
                time.sleep(wait)
        raise RuntimeError(f"Socrata request failed after {self.max_retries} attempts: {last}")


def parse_timestamp(column: str) -> pl.Expr:
    return pl.coalesce(
        [pl.col(column).str.to_datetime(format=fmt, strict=False, time_unit="us") for fmt in TS_FORMATS]
    ).alias(column)


def day_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """Inclusive-start / exclusive-end windows covering [start, end]."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    last = end + timedelta(days=1)
    while cursor < last:
        stop = min(cursor + timedelta(days=chunk_days), last)
        chunks.append((cursor, stop))
        cursor = stop
    return chunks


def fetch_observations(client: SocrataClient, cfg, lo: date, hi: date) -> pl.DataFrame:
    """One time-window page of observations, typed but not yet cleaned."""
    columns = cfg.dotted("traffic.observation_columns")
    params = {
        "$select": ", ".join(columns),
        "$where": f"data_as_of >= '{lo.isoformat()}T00:00:00' "
                  f"and data_as_of < '{hi.isoformat()}T00:00:00'",
        "$order": "data_as_of",
        "$limit": cfg.dotted("traffic.page_limit"),
    }
    response = client.get(params, fmt="csv")
    if not response.text.strip():
        return pl.DataFrame(schema=OBSERVATION_SCHEMA)
    return pl.read_csv(
        response.content,
        schema_overrides=OBSERVATION_SCHEMA,
        infer_schema_length=0,
        null_values=["", "NULL", "null", "NA"],
    )


def fetch_link_registry(client: SocrataClient, cfg, sample_days: list[date], log) -> pl.DataFrame:
    """One row per (link, geometry), sampled across the window.

    A single `$group by link_points` over 23.8M rows makes the server drop the
    connection, so instead we sample a short window on several days spread across
    the period and take distinct links. A link whose geometry changed mid-period
    therefore still shows up, with `seen_on` recording where.
    """
    columns = cfg.dotted("traffic.registry_columns")
    frames: list[pl.DataFrame] = []
    for day in sample_days:
        params = {
            "$select": ", ".join(columns),
            "$where": f"data_as_of >= '{day.isoformat()}T12:00:00' "
                      f"and data_as_of < '{day.isoformat()}T12:30:00'",
            "$limit": 50000,
        }
        response = client.get(params, fmt="csv")
        if not response.text.strip():
            log.warning("  link registry sample %s: empty", day)
            continue
        frame = pl.read_csv(response.content, infer_schema_length=0).with_columns(
            pl.lit(day.isoformat()).alias("seen_on")
        )
        frames.append(frame)
        log.info("  link registry sample %s: %d rows", day, frame.height)
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="vertical_relaxed")
    return (
        combined.group_by(["link_id", "link_points"])
        .agg(
            [
                pl.col("borough").drop_nulls().first().alias("borough"),
                pl.col("link_name").drop_nulls().first().alias("link_name"),
                pl.col("owner").drop_nulls().first().alias("owner"),
                pl.col("encoded_poly_line").drop_nulls().first().alias("encoded_poly_line"),
                pl.col("seen_on").min().alias("first_sampled_on"),
                pl.col("seen_on").max().alias("last_sampled_on"),
                pl.len().alias("sample_rows"),
            ]
        )
        .sort(["link_id", "first_sampled_on"])
    )


def add_link_geometry_summary(registry: pl.DataFrame) -> pl.DataFrame:
    """Derive per-link endpoints, midpoint and vertex count from `link_points`.

    Phase 4 maps links to spatial cells; which method it uses (midpoint vs. full
    geometry overlap) is a Phase 4 decision, so this only prepares the inputs.
    """
    points = (
        registry["link_points"]
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.split(" ")
    )
    lats: list[float | None] = []
    lngs: list[float | None] = []
    mid_lats: list[float | None] = []
    mid_lngs: list[float | None] = []
    end_lats: list[float | None] = []
    end_lngs: list[float | None] = []
    counts: list[int] = []
    for raw in points.to_list():
        coords = []
        for token in raw or []:
            parts = token.split(",")
            if len(parts) != 2:
                continue
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        counts.append(len(coords))
        if not coords:
            lats.append(None); lngs.append(None)
            mid_lats.append(None); mid_lngs.append(None)
            end_lats.append(None); end_lngs.append(None)
            continue
        lats.append(coords[0][0]); lngs.append(coords[0][1])
        end_lats.append(coords[-1][0]); end_lngs.append(coords[-1][1])
        middle = coords[len(coords) // 2]
        mid_lats.append(middle[0]); mid_lngs.append(middle[1])
    return registry.with_columns(
        [
            pl.Series("start_lat", lats),
            pl.Series("start_lng", lngs),
            pl.Series("end_lat", end_lats),
            pl.Series("end_lng", end_lngs),
            pl.Series("mid_lat", mid_lats),
            pl.Series("mid_lng", mid_lngs),
            pl.Series("n_vertices", counts),
        ]
    )
