# GeoPulse - Citi Bike data quality report (full)

Raw rows profiled: **79,410,195**

## Nulls per raw column

| ride_id | rideable_type | started_at | ended_at | start_station_name | start_station_id | end_station_name | end_station_id | start_lat | start_lng | end_lat | end_lng | member_casual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 48,033 | 48,033 | 214,452 | 222,480 | 48,033 | 48,033 | 221,610 | 221,610 | 0 |

## Ride id uniqueness

| distinct_ride_ids | duplicate_extra_rows |
|---|---|
| 79,410,195 | 0 |

## Timestamp validity

| null_started_at | null_ended_at | end_le_start | end_eq_start |
|---|---|---|---|
| 0 | 0 | 643 | 0 |

## Ride duration (seconds, raw)

| min | p01 | p25 | median | p75 | p99 | p999 | max | mean | under_60s | over_24h |
|---|---|---|---|---|---|---|---|---|---|---|
| -3,478 | 81.00 | 317.00 | 552.00 | 966.00 | 4,054.00 | 28,349.42 | 10,038,455 | 862.47 | 825 | 37,708 |

## Coordinate ranges

| start_lat_min | start_lat_max | start_lng_min | start_lng_max | end_lat_min | end_lat_max | end_lng_min | end_lng_max |
|---|---|---|---|---|---|---|---|
| 40.63 | 40.89 | -74.09 | -73.85 | 0.00 | 40.89 | -74.09 | 0.00 |


Rows with any coordinate outside the configured service bbox: **187**
Rows with `started_at` outside the project window: **276**

## Station id completeness

| null_start_station_id | null_end_station_id | null_either_station_id | distinct_start_station_ids | distinct_end_station_ids |
|---|---|---|---|---|
| 48,033 | 222,480 | 242,748 | 2,453 | 2,459 |

## DST transition hours (UTC)

Citi Bike publishes naive local wall-clock time, which cannot say which pass through the repeated fall-back hour a ride belongs to. Under the `ambiguous="earliest"` policy the distortion is deterministic: the first (EDT) pass absorbs both hours' rides (`over_filled`), the second (EST) pass is unreachable and therefore empty by construction, and spring-forward timestamps that cannot exist are shifted one hour forward (`shifted_in`). Four hours across 2023-2024 are affected; Phase 2 must flag these bins rather than model them as observed demand.

| utc_hour | rides | expected_effect |
|---|---|---|
| 2023-03-12 04:00:00 | 1,389 | - |
| 2023-03-12 05:00:00 | 1,048 | - |
| 2023-03-12 06:00:00 | 781 | - |
| 2023-03-12 07:00:00 | 617 | shifted_in |
| 2023-03-12 08:00:00 | 406 | - |
| 2023-03-12 09:00:00 | 236 | - |
| 2023-11-05 03:00:00 | 3,132 | - |
| 2023-11-05 04:00:00 | 2,437 | - |
| 2023-11-05 05:00:00 | 2,921 | over_filled |
| 2023-11-05 06:00:00 | 0 | empty |
| 2023-11-05 07:00:00 | 755 | - |
| 2023-11-05 08:00:00 | 521 | - |
| 2024-03-10 04:00:00 | 509 | - |
| 2024-03-10 05:00:00 | 666 | - |
| 2024-03-10 06:00:00 | 695 | - |
| 2024-03-10 07:00:00 | 582 | shifted_in |
| 2024-03-10 08:00:00 | 404 | - |
| 2024-03-10 09:00:00 | 288 | - |
| 2024-11-03 03:00:00 | 3,711 | - |
| 2024-11-03 04:00:00 | 3,010 | - |
| 2024-11-03 05:00:00 | 3,534 | over_filled |
| 2024-11-03 06:00:00 | 0 | empty |
| 2024-11-03 07:00:00 | 948 | - |
| 2024-11-03 08:00:00 | 580 | - |

## Rides per day

| n_days | min | median | mean | max | first_day | last_day |
|---|---|---|---|---|---|---|
| 735 | 1 | 110,799.00 | 108,041.08 | 194,315 | 2022-12-14 | 2024-12-31 |

## Rides per month

| month | rides |
|---|---|
| 2022-12 | 276 |
| 2023-01 | 1,795,329 |
| 2023-02 | 1,696,101 |
| 2023-03 | 2,119,314 |
| 2023-04 | 2,749,360 |
| 2023-05 | 3,453,576 |
| 2023-06 | 3,451,869 |
| 2023-07 | 3,659,372 |
| 2023-08 | 3,964,206 |
| 2023-09 | 3,471,658 |
| 2023-10 | 3,724,615 |
| 2023-11 | 2,816,850 |
| 2023-12 | 2,204,870 |
| 2024-01 | 1,887,908 |
| 2024-02 | 2,121,506 |
| 2024-03 | 2,663,416 |
| 2024-04 | 3,216,957 |
| 2024-05 | 4,135,049 |
| 2024-06 | 4,782,935 |
| 2024-07 | 4,722,952 |
| 2024-08 | 4,603,942 |
| 2024-09 | 4,997,288 |
| 2024-10 | 5,150,717 |
| 2024-11 | 3,709,229 |
| 2024-12 | 2,310,900 |

## Rideable type

| rideable_type | n |
|---|---|
| electric_bike | 46,901,076 |
| classic_bike | 32,509,119 |

## Membership

| member_casual | n |
|---|---|
| member | 64,260,069 |
| casual | 15,150,126 |

## Cleaning waterfall

Each row is attributed to the **first** rule it violates, so the counts sum exactly to the rows removed. No row is dropped without a named rule.

| rule | rows_removed | pct_of_raw |
|---|---|---|
| duplicate_ride_id | 0 | 0.00 |
| null_ride_id | 0 | 0.00 |
| null_timestamp | 0 | 0.00 |
| outside_project_window | 276 | 0.0003 |
| end_before_start | 643 | 0.0008 |
| duration_too_short | 182 | 0.0002 |
| duration_too_long | 37,627 | 0.05 |
| missing_coords | 206,222 | 0.26 |
| coords_out_of_bbox | 178 | 0.0002 |
| TOTAL REMOVED | 245,128 | 0.31 |
| KEPT | 79,165,067 | 99.69 |

