# Tiger Multifactor Evaluation

`tiger_factors.multifactor_evaluation` is the Tiger-native layer for
factor screening, factor blending, cross-sectional backtesting, and portfolio
comparison.

It sits above `tiger_factors.factor_frame` and above the report layers used for
positions, trades, and portfolio tear sheets:

- `factor_frame` builds the long research tables and model scores
- `multifactor_evaluation` screens, blends, allocates, and backtests factor
  panels
- `multifactor_evaluation.reporting` renders local positions / trades / portfolio reports

## CSM integration

The cross-sectional selection model in
[`tiger_factors.factor_frame.csm`](/Users/yuanhuzhang/Tiger/tiger_quant/tiger_factors/factor_frame/csm.py)
can be used through this package in three ways:

- `run_csm_backtest(...)`
  - fit the model
  - convert scores into a wide factor panel
  - run `run_factor_backtest(...)`
- `run_csm_selection_backtest(...)`
  - fit the model
  - convert the selected basket into a wide selection panel
  - run `run_factor_backtest(...)`
- `run_csm_factor_frame_selection_backtest(...)`
  - infer numeric factor columns from a Tiger-style long research frame when
    you do not want to pass them manually
  - fit the model, build a selection panel, and run `run_factor_backtest(...)`
- `run_factor_backtest(...)`
  - accept any wide date x code panel, including CSM score panels

## Typical flow

```python
from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation import run_csm_selection_backtest

feature_columns = infer_csm_feature_columns(training_frame)
model = build_csm_model(feature_columns)
model.fit(training_frame)
result = run_csm_selection_backtest(
    training_frame,
    close_panel,
    top_n=10,
    bottom_n=10,
    long_only=False,
)
```

The returned backtest can be passed directly to
`tiger_factors.multifactor_evaluation.reporting.portfolio.run_portfolio_from_backtest(...)`
for local report generation.

For multiple-testing correction, family-level validation, and factor zoo
screening utilities, see `tiger_factors.multifactor_evaluation.validation`.
For an empirical-Bayes / local-FDR variant that treats the family as a
two-group mixture and returns posterior signal probabilities, see
`tiger_factors.multifactor_evaluation.bayes_validation`.
For a hierarchical Bayes variant that borrows evidence from factor clusters
and returns cluster-aware posterior probabilities, see
`tiger_factors.multifactor_evaluation.bayes_validation`.
The Bayesian helpers also include:

- `bayesian_fdr(...)`
- `bayesian_fwer(...)`
- `alpha_hacking_bayesian_update(...)`
- `dynamic_bayesian_alpha(...)`
- `rolling_bayesian_alpha(...)`
For a minimal classical-vs-Bayesian multiple-testing walkthrough, see
`tiger_factors.examples.bayes_multiple_testing_demo`.
For a hierarchical Bayes walkthrough that borrows evidence from factor
clusters, see `tiger_factors.examples.hierarchical_bayes_multiple_testing_demo`.
For an alpha-hacking / out-of-sample posterior walkthrough, see
`tiger_factors.examples.alpha_hacking_posterior_demo`.
For a rolling posterior-alpha walkthrough, see
`tiger_factors.examples.posterior_alpha_timeseries_demo`.
For the classic `BM` + `F-Score` value-quality screen and its `HH` vs `LL`
long-short backtest, see `tiger_factors.examples.value_quality_long_short_demo`,
`tiger_factors.examples.value_quality_real_data_demo`,
`tiger_factors.examples.value_quality_factor_store_demo`,
`run_value_quality_long_short_backtest(...)`, and
`run_value_quality_combo_from_columns(...)`.
The factor-store demo uses `TigerFactorLibrary.load_factor_frame(...)` to
assemble multiple stored factor panels into a single long research frame
before backtesting.
The direct-column helper is also exported from this package root so you can
import it alongside the CSM and Bayesian utilities.
For a fully generic factor-store backtest that loads the stored panels,
fetches the matching close panel, and runs the composite backtest in one
call, use `multi_factor_backtest_from_store(...)`.

### Recommended handoff

If you already start from a `factor_frame` long table, the shortest path is:

1. `build_csm_training_frame(...)` in `tiger_factors.factor_frame.csm`
2. `build_csm_model(...)` + `fit(...)`
3. `selection_panel(...)`
4. `run_csm_selection_backtest(...)` or `run_csm_factor_frame_selection_backtest(...)`
5. `run_portfolio_from_backtest(...)`

Minimal CSM backtest:

```python
from tiger_factors.multifactor_evaluation import run_csm_factor_frame_selection_backtest

result = run_csm_factor_frame_selection_backtest(
    factor_frame_with_forward_returns,
    close_panel,
    ("momentum", "value", "quality"),
    top_n=10,
    bottom_n=10,
    long_only=False,
)
```

### Rolling posterior alpha

`rolling_bayesian_alpha(...)` in
[`tiger_factors.multifactor_evaluation.bayes_validation`](/Users/yuanhuzhang/Tiger/tiger_quant/tiger_factors/multifactor_evaluation/bayes_validation.py)
fits a trailing-window empirical-Bayes shrinkage model to an alpha time
series and returns:

- rolling OLS alpha
- rolling posterior alpha
- Bayesian FDR / FWER over the discovered windows

This is the light-weight time-evolution companion to the family-level
Bayesian multiple-testing helpers.
