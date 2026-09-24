# What `forecast_time` means, and why it decides the leakage rule

Every leakage question in this project reduces to one definition, so it is written
down here once rather than re-litigated per feature.

## The convention

A panel row is keyed by `ts`, the **start** of a 15-minute bin. The forecast is made
at the **end** of that bin — the instant `ts + 15min`, which is also the start of bin
`ts+1`.

```
        bin t-1        bin t          bin t+1        bin t+2
    |--------------|--------------|--------------|--------------|
                                  ^
                            forecast_time
                        (everything left of here
                          is fully observed)

    pickup_h1[t] = pickups[t+1]   <- the next 15 minutes
    pickup_h2[t] = pickups[t+2]   <- 15-30 minutes ahead
    pickup_h4[t] = pickups[t+4]   <- 45-60 minutes ahead
```

## Two consequences

**1. The horizon labels are honest.** `h1` really is a 15-minutes-ahead forecast. Under
the other reading — forecasting *at* `ts`, while bin `t` is still filling — `h1` would
actually be 15-to-30 minutes ahead, and the project's "15/30/45/60-minute horizons"
claim would be wrong by one bin everywhere.

**2. The current bin is observable, so `pickups[t]` is a legal feature.** It is the
last *completed* bin at `forecast_time`. Phase 3 used only `lag_1..lag_672` and left
this out, which was conservative but not free: it discarded the single most predictive
input. Phase 4 adds it as the lag-0 term of feature family A.

This is the operational reading too. In Phase 8 an operator stands at a moment,
knows everything that has already happened, and asks what the next hour looks like.

## What is still forbidden

* Any feature reading a bin at or after `ts + 15min` (except the targets themselves).
* Weather observations timestamped after `forecast_time` — the join rule stays
  `weather_timestamp <= forecast_time`.
* Events not yet started at `forecast_time`, other than through their *scheduled*
  start/end times, which are known in advance by definition.
* Statistics (medians, encodings, scalers) fitted on validation or test rows.
* Expanding seasonal expectations that include the current or any future row.

## Where this is enforced

`tests/test_features.py` re-derives every lag by an independent timestamp join, and
`tests/test_advanced_features.py` does the same for rolling, seasonal and external
joins. The lag-0 feature is asserted to equal the row's own observed demand and
nothing else.
