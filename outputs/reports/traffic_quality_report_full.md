# GeoPulse - NYC DOT traffic speed data quality report (full)

Raw traffic observations profiled: **23,758,184** across **138** road links

## Completeness

| null_timestamp | null_link_id | null_speed | null_travel_time | first_reading | last_reading |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 2023-01-01 00:03:03-05:00 | 2024-12-31 23:58:09-05:00 |

## Feed status codes

`0` is a valid reading. Anything else means the sensor reported nothing; those rows carry `speed = 0` and are removed rather than averaged in.

| status | rows | mean_speed |
|---|---|---|
| 0 | 17,904,379 | 38.07 |
| -101 | 5,853,805 | 4.58 |

## Speed distribution (valid readings only, mph)

| min | p25 | median | p75 | p99 | max | mean |
|---|---|---|---|---|---|---|
| 0.62 | 24.23 | 42.87 | 51.57 | 62.75 | 103.14 | 38.07 |

## Cleaning waterfall

| rule | rows_removed | pct_of_raw |
|---|---|---|
| duplicate_link_timestamp | 0 | 0.00 |
| null_timestamp | 0 | 0.00 |
| null_link_id | 0 | 0.00 |
| outside_project_window | 0 | 0.00 |
| invalid_status | 5,853,805 | 24.64 |
| speed_out_of_range | 13 | 0.0001 |
| travel_time_nonpositive | 0 | 0.00 |
| TOTAL REMOVED | 5,853,818 | 24.64 |
| KEPT | 17,904,366 | 75.36 |

## Feed outages

Calendar days in the project window with **no** traffic readings: **45** of 731.

These are real gaps in the published feed, not a fetch failure. Phase 4 must treat them as missing data (with a `traffic_missing` flag), never as free-flowing traffic.

```
2023-01-14
2023-01-15
2023-01-16
2023-01-17
2023-01-18
2023-01-19
2023-01-20
2023-05-21
2023-05-22
2023-05-23
2023-05-24
2023-05-25
2023-05-26
2023-05-27
2023-05-28
2023-05-29
2023-05-30
2023-05-31
2023-06-01
2023-06-02
2023-06-03
2023-06-04
2023-06-05
2023-06-06
2023-06-07
2023-07-13
2023-07-14
2023-07-23
2024-01-30
2024-01-31
2024-02-01
2024-02-02
2024-02-03
2024-02-04
2024-02-05
2024-02-06
2024-02-07
2024-02-08
2024-02-09
2024-02-10
2024-02-11
2024-08-25
2024-12-25
2024-12-26
2024-12-27
```

