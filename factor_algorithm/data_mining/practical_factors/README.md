# tiger_factors.factor_algorithm.data_mining.practical_factors

This folder contains the "practical" factor variants cleaned from the
screenshots and rewritten into directly testable Python.

Current factors include the original practical formulas plus two named
multi-factor building blocks:

- `factor_001_volume_flow_sine_skew`
- `factor_086_low_correlation_252d`
- `factor_087_risk_adjusted_momentum_6m_12m`

Original `factor_001_volume_flow_sine_skew` formula:

```text
-1 * (SIN(TS_MEAN(AF_CLOSE/TS_DELAY(AF_CLOSE,1)-1,5))
      * CS_SKEW(AF_VWAP)
      * TS_MEDIAN(MAIN_IN_FLOW_DAYS_10D_V2,20)
      + LOG(1+TS_MAX_STD(AF_HIGH-AF_LOW,60,3))
      * (TS_MAX_MEAN(VOLUME,20,5)/TS_MEAN(VOLUME,20)))
```

If `MAIN_IN_FLOW_DAYS_10D_V2` is not available, the engine falls back to a
simple proxy based on positive `close - open` days over the past 10 sessions.

## 10-year evaluation script

Run the integrated 10-year evaluation with:

```bash
PYTHONPATH=. ./.venv/bin/python -m tiger_factors.examples.practical_factor_10y_eval \
  --db-path /Volumes/Quant_Disk/data/yahoo_us_stock.db \
  --universe-csv /Volumes/Quant_Disk/data/sp500_ticker_start_end.csv \
  --start 2014-01-01 \
  --end 2024-12-31
```
