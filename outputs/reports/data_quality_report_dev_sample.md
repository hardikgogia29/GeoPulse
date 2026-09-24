# GeoPulse - Citi Bike data quality report (dev_sample)

Raw rows profiled: **1,127,270**

## Nulls per raw column

| ride_id | rideable_type | started_at | ended_at | start_station_name | start_station_id | end_station_name | end_station_id | start_lat | start_lng | end_lat | end_lng | member_casual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 377 | 377 | 3,001 | 3,305 | 377 | 377 | 3,296 | 3,296 | 0 |

## Ride id uniqueness

| distinct_ride_ids | duplicate_extra_rows |
|---|---|
| 1,127,270 | 0 |

## Timestamp validity

| null_started_at | null_ended_at | end_le_start | end_eq_start |
|---|---|---|---|
| 0 | 0 | 0 | 0 |

## Ride duration (seconds, raw)

| min | p01 | p25 | median | p75 | p99 | p999 | max | mean | under_60s | over_24h |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 84.00 | 340.00 | 600.00 | 1,051.00 | 4,438.00 | 26,935.08 | 89,996 | 910.74 | 12 | 313 |

## Coordinate ranges

| start_lat_min | start_lat_max | start_lng_min | start_lng_max | end_lat_min | end_lat_max | end_lng_min | end_lng_max |
|---|---|---|---|---|---|---|---|
| 40.63 | 40.89 | -74.07 | -73.85 | 0.00 | 40.89 | -74.07 | 0.00 |


Rows with any coordinate outside the configured service bbox: **1**
Rows with `started_at` outside the project window: **0**

## Station id completeness

| null_start_station_id | null_end_station_id | null_either_station_id | distinct_start_station_ids | distinct_end_station_ids |
|---|---|---|---|---|
| 377 | 3,305 | 3,527 | 2,241 | 2,190 |

## Rides per day

| n_days | min | median | mean | max | first_day | last_day |
|---|---|---|---|---|---|---|
| 7 | 144,900 | 167,253.00 | 161,038.57 | 172,804 | 2024-06-03 | 2024-06-09 |

## Rides per month

| month | rides |
|---|---|
| 2024-06 | 1,127,270 |

## Rideable type

| rideable_type | n |
|---|---|
| electric_bike | 712,097 |
| classic_bike | 415,173 |

## Membership

| member_casual | n |
|---|---|
| member | 871,537 |
| casual | 255,733 |

## Cleaning waterfall

Each row is attributed to the **first** rule it violates, so the counts sum exactly to the rows removed. No row is dropped without a named rule.

| rule | rows_removed | pct_of_raw |
|---|---|---|
| duplicate_ride_id | 0 | 0.00 |
| null_ride_id | 0 | 0.00 |
| null_timestamp | 0 | 0.00 |
| outside_project_window | 0 | 0.00 |
| end_before_start | 0 | 0.00 |
| duration_too_short | 12 | 0.00 |
| duration_too_long | 313 | 0.03 |
| missing_coords | 3,214 | 0.29 |
| coords_out_of_bbox | 1 | 0.00 |
| TOTAL REMOVED | 3,540 | 0.31 |
| KEPT | 1,123,730 | 99.69 |

