# Factor Preprocessing

`tiger_factors.factor_preprocessing` is the factor-level preprocessing layer.
It collects the cleaning, scaling, neutralization, and discretization steps
that typically happen before factor combination or portfolio construction.

## Module Map

- `missing`
  - `coerce_factor_panel`
  - `coerce_target_panel`
  - `fill_missing_factor_panel`
- `outliers`
  - `detect_outliers_factor_panel`
  - `replace_outliers_with_nan`
  - `winsorize_factor_panel`
  - `detect_anomalies_isolation_forest`
- `scaling`
  - `demean`
  - `zscore`
  - `robust_zscore`
  - `rank_pct`
  - `rank_centered`
  - `minmax_scale`
  - `l1_normalize`
  - `l2_normalize`
  - `normalize_cross_section`
  - `winsorize_cross_section`
  - `winsorize_mad`
  - `winsorize_quantile`
  - `preprocess_cross_section`
  - `scale_factor_panel`
- `neutralization`
  - `neutralize_cross_section`
  - `neutralize_factor_panel`
  - `cs_rank`
  - `cs_zscore`
  - `cs_winsorize`
  - `cs_winsorize_mad`
  - `cs_minmax_pos`
  - `cs_minmax_neg`
  - `cs_neutralize`
- `binning`
  - `bin_factor_panel`
  - `woe_encode_binned`
  - `target_encode_binned`
  - `onehot_encode_binned`
- `pipeline`
  - `preprocess_factor_panel`
  - `FactorPreprocessor`

## Usage

Prefer the submodule that matches the task:

```python
from tiger_factors.factor_preprocessing.missing import fill_missing_factor_panel
from tiger_factors.factor_preprocessing.outliers import winsorize_factor_panel
from tiger_factors.factor_preprocessing.scaling import zscore
from tiger_factors.factor_preprocessing.neutralization import neutralize_factor_panel
from tiger_factors.factor_preprocessing.binning import bin_factor_panel
from tiger_factors.factor_preprocessing.pipeline import preprocess_factor_panel
```

The package root keeps compatibility re-exports for older code, but the
submodules are the preferred API for new code.
