# GeoPulse - NYC DOT traffic speed data quality report (dev_sample)

Raw traffic observations profiled: **242,341** across **121** road links

## Completeness

| null_timestamp | null_link_id | null_speed | null_travel_time | first_reading | last_reading |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 2024-06-03 00:03:03-04:00 | 2024-06-09 23:59:09-04:00 |

## Feed status codes

`0` is a valid reading. Anything else means the sensor reported nothing; those rows carry `speed = 0` and are removed rather than averaged in.

| status | rows | mean_speed |
|---|---|---|
| 0 | 187,575 | 36.92 |
| -101 | 54,766 | 4.77 |

## Speed distribution (valid readings only, mph)

| min | p25 | median | p75 | p99 | max | mean |
|---|---|---|---|---|---|---|
| 0.62 | 21.74 | 41.01 | 51.57 | 62.75 | 90.72 | 36.92 |

## Cleaning waterfall

| rule | rows_removed | pct_of_raw |
|---|---|---|
| duplicate_link_timestamp | 0 | 0.00 |
| null_timestamp | 0 | 0.00 |
| null_link_id | 0 | 0.00 |
| outside_project_window | 0 | 0.00 |
| invalid_status | 54,766 | 22.60 |
| speed_out_of_range | 0 | 0.00 |
| travel_time_nonpositive | 0 | 0.00 |
| TOTAL REMOVED | 54,766 | 22.60 |
| KEPT | 187,575 | 77.40 |

## Feed outages

Calendar days in the project window with **no** traffic readings: **0** of 7.

These are real gaps in the published feed, not a fetch failure. Phase 4 must treat them as missing data (with a `traffic_missing` flag), never as free-flowing traffic.

