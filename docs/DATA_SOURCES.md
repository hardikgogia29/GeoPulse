# Data Sources

**Date range for all sources below: Jan 1, 2023 - Dec 31, 2024** (matches the Citi
Bike trip data actually on hand).

## Required for v1

### Citi Bike trip history
- **Source:** https://citibikenyc.com/system-data (monthly CSV/zip files)
- **Period:** Jan 2023 - Dec 2024 (acquired: 90 CSVs, ~15 GB, 79,410,195 rows)
- **Fields used:** ride_id, rideable_type, started_at, ended_at, start/end
  station_id + name + lat/lng, member_casual
- **Station ids are strings, not numbers.** They look like `5137.10`; parsing them as
  floats collapses that onto `5137.1` and merges distinct stations.
- **Note:** the station roster changes over this period (the system expanded through
  2023-2024) - use *actual trip activity* to define active stations/regions per
  Phase 1-2, not a static station list. Confirmed empirically: 255 stations appear in
  2023-2024 trips but are absent from the roster snapshot.

### Weather
- **Source:** Open-Meteo Historical API (https://archive-api.open-meteo.com/v1/archive,
  free, no key)
- **Reference point:** Central Park (lat 40.7831, lon -73.9712)
- **Fields:** temperature, apparent temperature, humidity, precipitation, rain,
  snowfall, snow depth, wind speed/gusts/direction, cloud cover, weather code - hourly
- **`visibility` is not available** from this archive (returns 100% null); removed
  from the requested variable list rather than kept as a dead column.
- Join rule: `weather_timestamp <= forecast_time`, latest available reading, no
  future readings. See Phase 4 for missing-value handling.
- **Status:** `scripts/03_fetch_weather.py` -> `data/external/weather_hourly.parquet`,
  17,544 rows = 731 days x 24 h, no gaps.

### Events / festivals
- **Source:** NYC Open Data - "NYC Permitted Event Information - Historical"
- **Status:** `scripts/04_prepare_events.py` -> `data/external/events.parquet`,
  557,762 rows / 75,741 distinct events overlapping the window.
- **Geocoding gap:** the export has NO lat/lng - only a free-text street description
  (`BROADWAY between WEST 43 STREET and WEST 44 STREET`), plus borough, community
  board and police precinct. **19,012 distinct location strings** need geocoding
  before events can be mapped to cells; the inventory is saved to
  `data/external/event_locations.parquet`. This is a Phase 4 task.
- **Known gap:** covers *permitted* public events (parades, street fairs, block
  parties) - it does not reliably capture ticketed arena events (MSG, Barclays,
  stadiums). Documented limitation, not a bug.

### Traffic
- **Source:** NYC Open Data - "Real-Time Traffic Speed Data", Socrata dataset
  **`i4gi-tjb9`** (https://dev.socrata.com/foundry/data.cityofnewyork.us/i4gi-tjb9)
- **Status:** `scripts/07_fetch_traffic.py` -> `data/raw/traffic/month=YYYYMM/`,
  then `scripts/08_clean_traffic.py` -> `data/interim/traffic_clean.parquet`.
  ~23.8M observations across ~128 road links, roughly one reading per link every
  five minutes.
- **Do not use the Socrata UI CSV export.** The manual download
  (`DOT_traffic_speeds_after_2018-07-01_*.csv`) was row-capped: 194,130 rows covering
  **2018-07-26 to 2018-07-30 only**, zero overlap with the project window. The API
  pull with an explicit `$where` on `data_as_of` is the supported path.
- **Fetch strategy.** Paginate by **time window**, not `$offset` - `data_as_of` is
  indexed, whereas deep offsets degrade badly over 23.8M rows and cannot be resumed.
  The fetch requests only the columns that vary per observation
  (`link_id, speed, travel_time, status, data_as_of`); `link_points` and
  `encoded_poly_line` are **static per link** and are pulled once into
  `data/spatial/traffic_links.parquet`. Repeating that geometry on every row is what
  made the UI export unusably large.
- **`id` is not a row id.** It is the link's own small integer identifier - there are
  as many distinct `id` values as links. The natural key is `(link_id, data_as_of)`.
- **`status` is the feed's own validity marker.** `0` is a real reading; `-101`
  (~22% of rows) means the sensor reported nothing, and those rows carry `speed = 0`.
  Keeping them drags every average toward zero, so they are dropped by the named
  `invalid_status` rule.
- **`data_as_of` is a floating timestamp** - naive local New York wall-clock, like the
  Citi Bike timestamps - so it goes through the same DST-aware localization.
- **Optional app token:** set `$SOCRATA_APP_TOKEN`. Anonymous requests work but are
  throttled harder.

#### Traffic coverage is the real limitation
The feed covers arterials, highways and bridges - not the local streets Citi Bike
mostly runs on. Measured against the station registry
(`data/spatial/traffic_link_station_proximity.parquet`):

| links within X of any Citi Bike station | count | share |
|---|---|---|
| 250 m | 27 / 128 | 21% |
| 500 m | 46 / 128 | 36% |
| 1 km  | 57 / 128 | 45% |
| 5 km  | 80 / 128 | 62% |

| borough | links | median km to nearest station | within 500 m |
|---|---|---|---|
| Queens | 39 | 5.85 | 9 |
| Staten Island | 27 | 11.68 | 0 |
| Manhattan | 26 | 0.24 | 25 |
| Bronx | 24 | 1.23 | 6 |
| Brooklyn | 12 | 0.60 | 6 |

So traffic features will be dense in Manhattan and sparse-to-absent elsewhere, and
all 27 Staten Island links are irrelevant (Citi Bike does not serve Staten Island).
The feed also has **real multi-day outages** (e.g. 2023-01-14..20, 2023-05-21..06-08)
which must be treated as missing, never as free-flowing traffic. Let the Phase 4
ablation decide whether family F earns its complexity - this table is why a null
result would be a legitimate finding rather than a bug.

### Station roster (supplementary)
- `citibike_stations_data.csv`: id, name, latitude, longitude for ~2.2k stations.
- **No capacity/dock count column** - so Phase 8 uses statistical capacity estimation.
- Used only as a cross-check on the trip-derived station registry, never as the
  definition of "active station". Cross-check result: observed median coordinates
  agree with the roster to under 10 cm; 2,204 of 2,459 observed stations appear in the
  roster, and only 2 roster stations never appear in 2023-2024 trips.

## Deferred - not required for v1, add later if pursued

### Station capacity / dock counts
- Phase 8 estimates capacity statistically from trip-flow patterns (q95 of daily
  cumulative net-flow range) because no ground-truth capacity data is on hand.
- Citi Bike's live GBFS feed (`station_information.json`) has *current* capacity
  only - not historical. A historical GBFS archive would be needed for accurate
  2023-2024 capacity.
- Until then the statistical estimate is the real v1 approach, not a placeholder.

### Road quality / pavement condition
- NYC DOT pavement rating + resurfacing records. Low priority.

### Subway/transit proximity
- MTA subway entrance locations. Plausible demand driver, optional.

### Venue-specific event schedules
- Would fill the "ticketed arena events" gap. No single clean public dataset.

### School calendar
- NYC DOE academic calendar. Minor effect. Skip unless ablation shows a gap.

## Not a data pull - handle in code
- **Public holidays:** Python `holidays` library.
- **Sunrise/sunset / daylight:** `astral` or a formula; not worth sourcing externally.
