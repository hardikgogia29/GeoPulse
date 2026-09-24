"""DST correctness - Phase 1 DoD requires all four 2023-2024 transitions verified."""

from datetime import datetime

import polars as pl
import pytest

from src.data.timezone import count_dst_affected, find_dst_transitions, localize_to_utc

TZ = "America/New_York"
WINDOW = (datetime(2023, 1, 1), datetime(2024, 12, 31))


@pytest.fixture(scope="module")
def transitions():
    return find_dst_transitions(TZ, *WINDOW)


def test_all_four_transitions_found(transitions):
    found = {(t.kind, t.local_window_start.date().isoformat()) for t in transitions}
    assert found == {
        ("spring_forward", "2023-03-12"),
        ("fall_back", "2023-11-05"),
        ("spring_forward", "2024-03-10"),
        ("fall_back", "2024-11-03"),
    }


def test_transition_windows_are_one_hour(transitions):
    for transition in transitions:
        delta = transition.local_window_end - transition.local_window_start
        assert delta.total_seconds() == 3600, transition


@pytest.mark.parametrize(
    "naive,expected_utc",
    [
        # standard time (EST, UTC-5) and daylight time (EDT, UTC-4)
        (datetime(2024, 1, 15, 12, 0), "2024-01-15 17:00:00+00:00"),
        (datetime(2024, 7, 4, 12, 0), "2024-07-04 16:00:00+00:00"),
        # the hour before / after spring forward
        (datetime(2023, 3, 12, 1, 30), "2023-03-12 06:30:00+00:00"),
        (datetime(2023, 3, 12, 3, 30), "2023-03-12 07:30:00+00:00"),
        # nonexistent local time -> shifted forward one hour
        (datetime(2023, 3, 12, 2, 30), "2023-03-12 07:30:00+00:00"),
        (datetime(2024, 3, 10, 2, 15), "2024-03-10 07:15:00+00:00"),
        # ambiguous local time -> first (EDT) occurrence
        (datetime(2023, 11, 5, 1, 30), "2023-11-05 05:30:00+00:00"),
        (datetime(2024, 11, 3, 1, 45), "2024-11-03 05:45:00+00:00"),
    ],
)
def test_localize_to_utc(naive, expected_utc):
    got = (
        pl.DataFrame({"t": [naive]})
        .select(localize_to_utc(pl.col("t"), TZ).alias("utc"))
        .item()
    )
    assert str(got) == expected_utc


def test_no_nulls_across_a_full_transition_day():
    """Every minute of both transition days must localize to a real instant."""
    for day in ("2023-03-12", "2023-11-05", "2024-03-10", "2024-11-03"):
        minutes = pl.datetime_range(
            datetime.fromisoformat(f"{day}T00:00:00"),
            datetime.fromisoformat(f"{day}T23:59:00"),
            interval="1m",
            eager=True,
        )
        out = pl.DataFrame({"t": minutes}).select(localize_to_utc(pl.col("t"), TZ).alias("utc"))
        assert out["utc"].null_count() == 0, day


def test_fall_back_mapping_is_strictly_increasing_and_injective():
    """Distinct local times must map to distinct, increasing UTC instants.

    Note the documented limitation this pins down: with `ambiguous="earliest"` the
    *second* pass through the repeated local hour (06:00-07:00 UTC) is unreachable
    from a naive timestamp, so that UTC hour is left empty and the first pass is
    over-filled. See `dst_unreliable_utc_hours` - Phase 2 flags those hours rather
    than pretending the reconstruction is exact.
    """
    naive = [datetime(2024, 11, 3, h, 30) for h in (0, 1, 2, 3)]
    got = (
        pl.DataFrame({"t": naive})
        .select(localize_to_utc(pl.col("t"), TZ).alias("utc"))["utc"]
        .to_list()
    )
    assert got == sorted(got)
    assert len(set(got)) == len(got)
    assert [str(v) for v in got] == [
        "2024-11-03 04:30:00+00:00",
        "2024-11-03 05:30:00+00:00",
        "2024-11-03 07:30:00+00:00",
        "2024-11-03 08:30:00+00:00",
    ]


def test_dst_unreliable_hours_are_enumerated(transitions):
    from src.data.timezone import dst_unreliable_utc_hours

    hours = dst_unreliable_utc_hours(transitions)
    kinds = {(h["utc_hour_start"].isoformat(), h["effect"]) for h in hours}
    assert ("2023-11-05T05:00:00+00:00", "over_filled") in kinds
    assert ("2023-11-05T06:00:00+00:00", "empty") in kinds
    assert ("2024-11-03T05:00:00+00:00", "over_filled") in kinds
    assert ("2024-11-03T06:00:00+00:00", "empty") in kinds
    assert all(h["effect"] in {"over_filled", "empty", "shifted_in"} for h in hours)


def test_count_dst_affected(transitions):
    df = pl.DataFrame(
        {"t": [datetime(2023, 3, 12, 2, 30), datetime(2024, 11, 3, 1, 45), datetime(2024, 7, 4, 12, 0)]}
    )
    counts = count_dst_affected(df, ["t"], transitions)
    assert counts == {"t__spring_forward__20230312": 1, "t__fall_back__20241103": 1}
