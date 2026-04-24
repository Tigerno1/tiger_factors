# Multifactors Evaluation Examples

This folder contains minimal examples for screening factors in `tiger_factors.multifactor_evaluation`.

Scripts included here:

- `run_multifactors_screening.py`
  - registry screening example
- `multifactors_correlation_demo.py`
  - correlation matrix plus high/low correlation filtering
- `multifactors_factor_research.py`
  - full Alpha101 multifactors research pipeline
- `alpha101_multifactor_20_factor_demo.py`
  - end-to-end Alpha101 multifactor workflow with 20 factors and all tear sheets
- `multifactors_screening_demo.py`
  - factor diagnostics plus multifactors screening demo
- `multifactors_lowcorr_combo_backtest.py`
  - low-correlation combo backtest with adaptive per-factor thresholds
- `multifactors_lowcorr6_combo_standardized_ma10.py`
  - six-factor standardized combo with 10-day moving average smoothing
- `multifactors_lowcorr_combo_backtest_standardized.py`
  - low-correlation combo backtest with standardized per-factor thresholds
- `multifactors_resonance_strategy.py`
  - resonance strategy with agreement threshold and lagged execution

## 1. Screen a registry from strategy folders

Expected layout:

```text
strategies_root/
  strategy_a/
    summary/
      evaluation.parquet
  strategy_b/
    summary/
      evaluation.parquet
```

Run:

```bash
python -m tiger_factors.multifactor_evaluation.examples.run_multifactors_screening \
  --registry-root /Volumes/Quant_Disk/evaluation/multifactors_strategies \
  --min-ic-mean 0.005 \
  --min-rank-ic-mean 0.005 \
  --min-sharpe 0.30 \
  --max-turnover 0.70
```

This will:

- Load every `summary/evaluation.parquet` under the root folder.
- Merge them into a registry.
- Apply the unified multifactors filter.
- Print the selected rows.
- Save the screened registry to `--output-path` unless `--no-write` is used.

## 2. Build a multifactors correlation matrix

```bash
python -m tiger_factors.multifactor_evaluation.examples.multifactors_correlation_demo
```

This loads the persisted Alpha101 summary registry from
`/Volumes/Quant_Disk/evaluation/summary/summary_registry.parquet`, screens the
usable factors, computes the factor correlation matrix, and writes:

- `multifactors_factor_correlation_matrix.parquet`
- `multifactors_factor_correlation_pairs.parquet`
- `multifactors_high_corr_pairs.parquet`
- `multifactors_factor_correlation_summary.parquet`
- `multifactors_high_corr_factors.parquet`
- `multifactors_low_corr_factors.parquet`
- `multifactors_cluster_representatives.parquet`

## 3. Use the screening API directly in Python

```python
import pandas as pd
from tiger_factors.factor_screener import FactorMetricFilterConfig, screen_factor_registry

registry = pd.read_parquet("/path/to/multifactors_registry.parquet")

screened = screen_factor_registry(
    registry,
    config=FactorMetricFilterConfig(
        min_ic_mean=0.01,
        min_rank_ic_mean=0.01,
        min_sharpe=0.40,
        max_turnover=0.50,
    ),
)

print(screened[["strategy_name", "factor_name", "ic_mean", "fitness"]].head())
```

## 5. Run a full Alpha101 multifactor example

```bash
python -m tiger_factors.multifactor_evaluation.examples.alpha101_multifactor_20_factor_demo \
  --persist-outputs
```

This example:

- computes `alpha_001` through `alpha_020`
- screens and blends the factors
- runs the factor backtest
- generates `summary`, `positions`, `trades`, `portfolio`, and the combined `report.html`

## 4. Build a registry from a list of strategy summaries

```python
from pathlib import Path
from tiger_factors.factor_screener import build_factor_registry

registry = build_factor_registry(
    {
        "alpha_a": Path("/Volumes/Quant_Disk/evaluation/alpha_a/summary/summary.parquet"),
        "alpha_b": Path("/Volumes/Quant_Disk/evaluation/alpha_b/summary/summary.parquet"),
    }
)
```

## Notes

- The screening layer expects a summary table with the core factor metrics.
- The most important fields are `fitness`, `ic_mean`, `rank_ic_mean`, `sharpe`, and `turnover`.
- If you already have a per-strategy `summary/evaluation.parquet`, you can use the registry path and do not need to re-run factor evaluation.
