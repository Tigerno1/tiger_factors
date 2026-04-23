# tiger_factors.factor_algorithm.data_mining

`data_mining` is the cleaned alpha-candidate folder for the formulas shown in the screenshots.

## What is inside

- a `DataMiningEngine` that can evaluate the cleaned formulas on a long OHLCV panel
- module-level factor functions such as `factor_013_volume(...)`
- a small registry with the cleaned formulas and their source expressions
- a `practical_factors` subfolder for directly testable formulas that are a bit
  closer to production-style signals

## Cleaned factor set

The original noisy formulas were simplified to:

- `factor_013_volume`: raw volume
- `factor_006_price_volume_var40`: `ts_var(close / ts_wma(volume, 5), 40)`
- `factor_010_vwap_corr_inverse_high_lag10`: `Ref(ts_corr(vwap, Inv(high), 30), 10)`
- `factor_011_vwap_over_low_min10`: `2 * vwap / ts_min(low, 10)`
- `factor_012_volume_sum5`: `ts_sum(volume, 5)`
- `factor_007_volume_ema40`: `ts_ema(volume, 40)`
- `factor_002_intraday_strength`: `(close - open) / open`
- `factor_008_close_low_var10`: `ts_var(close * low, 10)`
- `factor_021_log1p_inverse_turnover_pressure`: `log1p(10 / (volume * (high + 2)))`
- `factor_025_open_high_cov5_wma5`: `ts_wma(ts_cov(open, high, 5), 5)`
- `factor_031_nested_cov_open`: nested covariance style factor
- `factor_032_low_pctchange_momentum`: `ts_pctchange(ts_min(ts_wma(low, 10), 50), 20)`
- `factor_024_close_over_high`: `close / high`
- `factor_029_vwap_minus_high`: `vwap - high`
- `factor_040_volume_high_corr_strength`: `ts_corr(volume, high, 20) * ts_sum(volume, 30)`

See `data_mining/practical_factors/README.md` for the practical-factor variant
currently added to the folder.

## Testing

Run the package tests:

```bash
./.venv/bin/python -m pytest tiger_factors/tests/test_data_mining.py -q
```

Or run the broader factor workflow tests:

```bash
./.venv/bin/python -m pytest tiger_factors/tests tiger_research/tests -q
```
