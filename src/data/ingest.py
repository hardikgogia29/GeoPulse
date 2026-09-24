
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from src.data.timezone import count_dst_affected, find_dst_transitions, localize_to_utc

# station ids look like "5636.13" - they are LABELS, not numbers. Reading them as
# floats silently turns "5137.10" into 5137.1 and merges distinct stations.
RAW_SCHEMA: dict[str, type[pl.DataType]] = {
    "ride_id": pl.String,
    "rideable_type": pl.String,
    "started_at": pl.String,
    "ended_at": pl.String,
    "start_station_name": pl.String,
    "start_station_id": pl.String,
    "end_station_name": pl.String,
    "end_station_id": pl.String,
    "start_lat": pl.Float64,
    "start_lng": pl.Float64,
    "end_lat": pl.Float64,
    "end_lng": pl.Float64,
    "member_casual": pl.String,
}

_TS_FORMATS = ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"]


def parse_timestamp(column: str) -> pl.Expr:
    """Parse a timestamp column, tolerating the format variants Citi Bike has used."""
    attempts = [
        pl.col(column).str.to_datetime(format=fmt, strict=False, time_unit="us")
        for fmt in _TS_FORMATS
    ]
    return pl.coalesce(attempts).alias(column)


def read_raw_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        schema_overrides=RAW_SCHEMA,
        try_parse_dates=False,
        infer_schema_length=0,
        null_values=["", "NULL", "null", "NA"],
    )


def ingest_file(path: Path, tz_name: str, transitions: list) -> tuple[pl.DataFrame, dict]:
    """Type + localize one raw CSV. Returns (dataframe, per-file stats)."""
    df = read_raw_csv(path)
    stats: dict = {"source_file": path.name, "rows_read": df.height}

    df = df.with_columns([parse_timestamp("started_at"), parse_timestamp("ended_at")])
    stats["unparsed_started_at"] = int(df["started_at"].null_count())
    stats["unparsed_ended_at"] = int(df["ended_at"].null_count())

    parsed = df.drop_nulls(subset=["started_at", "ended_at"])
    stats["dst_affected"] = count_dst_affected(parsed, ["started_at", "ended_at"], transitions)

    df = df.with_columns(
        [
            localize_to_utc(pl.col("started_at"), tz_name).alias("started_at"),
            localize_to_utc(pl.col("ended_at"), tz_name).alias("ended_at"),
        ]
    ).with_columns(
        [
            (pl.col("ended_at") - pl.col("started_at")).dt.total_seconds().alias("ride_duration"),
            pl.col("started_at").dt.convert_time_zone(tz_name).dt.strftime("%Y%m").alias("month"),
        ]
    )

    stats["rows_out"] = df.height
    stats["null_month"] = int(df["month"].null_count())
    return df, stats


def write_partitioned(df: pl.DataFrame, out_root: Path, source_stem: str) -> list[str]:
    """Write one parquet part per month under hive-style `month=YYYYMM/` folders."""
    written: list[str] = []
    for (month,), part in df.partition_by("month", as_dict=True).items():
        if month is None:
            month = "unknown"
        target_dir = out_root / f"month={month}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"part-{source_stem}.parquet"
        part.drop("month").write_parquet(target, compression="zstd", compression_level=3)
        written.append(str(target.relative_to(out_root)))
    return written


def dst_transitions_for(cfg) -> list:
    return find_dst_transitions(
        cfg.dotted("time.timezone"),
        datetime.fromisoformat(cfg.dotted("time.start_date")),
        datetime.fromisoformat(cfg.dotted("time.end_date")),
    )
