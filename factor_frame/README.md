# Tiger Factor Frame

`tiger_factors.factor_frame` is the Tiger-native research engine for
vectorized factor construction.

It is intentionally one layer below evaluation:

- feed materialized DataFrames directly
- align price, financial, valuation, macro, and news feeds
- broadcast date-level feeds across the equity universe
- run factor strategies over a unified research context
- return a long `factor_frame` that can be passed to `FactorEvaluationEngine`

- Lagging and calendars:

- `bday_lag=True` uses business-day lagging for feeds
- `bday_lag=False` uses calendar-day lagging
- `FactorFrameEngine(freq="1d")` or intraday values such as `freq="1h"`,
  `freq="1min"`, `freq="15min"`, `freq="20min"`, `freq="30min"`, or
  `freq="2h"` select the primary research data frequency
  and drive daily vs intraday alignment
- `calendar=None` leaves the engine without an exchange calendar; set it
  explicitly, such as `"XNYS"`, to use exchange-session logic
- `label_side="right"` is the default intraday label convention, meaning the
  timestamp marks the end of the bar; use `"left"` for start-labeled bars or
  `"auto"` to infer from the source when possible
- `FactorResearchEngine` accepts the same `freq`, `as_ex`, `calendar`, and
  `bday_lag` arguments through the same research façade
- `FactorFrameEngine(start=..., end=...)` trims every date-aligned feed and
  the combined research context to one shared time window
- each feed declares its own `align_mode`
  - `code_date` for price / financial / valuation / news style tables
  - `date` for macro-style feeds
  - `code` for code-only lookup tables such as company or industry maps
- `feed_financial(...)` and `feed_valuation(...)` default to `lag_sessions=1`
- `feed_financial(...)` and `feed_valuation(...)` default to `fill_method="ffill"`
- `feed_price(...)` defaults to `lag_sessions=0`
- `feed_macro(...)` and `feed_news(...)` default to `lag_sessions=0`
- each feed can also declare a `fill_method` such as `ffill`, `bfill`, or
  `both` to fill its joined columns after alignment
- fill is time-wise per column; one field never borrows values from another
- `build_context()` returns the normalized feeds plus the alignment config
- `add_screen(...)` adds a boolean universe gate
- `add_classifier(...)` adds a categorical lookup term
- screens are applied before factor computation only; factor callables see the
  screened feeds and screened combined context, and there is no post-factor
  screen pass
- any post-factor ranking or thresholding belongs in evaluation or strategy,
  not in the factor-frame screen layer
- `availability_column` can be used to align report-date data on the first
  available trading date before the session lag is applied
- `align_mode="outer"` keeps the widest universe, while `"intersection"`
  keeps only dates/codes shared by all code-level feeds
- feed alignment and feed lag are separate:
  - `align_mode` decides how rows are matched
  - `lag_sessions` decides when a feed becomes visible
- `add_factor(...)` is a convenience alias for `add_strategy(...)`
- `FactorFrameTemplate` / `factor_template(...)` provide parameterized factor
  constructors for reusable template-style factor definitions
- `FactorDefinition` is the structured recipe layer for complex factors with
  explicit preparation, computation, and post-processing steps
- `FactorDefinitionRegistry` lets you register named definitions once and
  enable them later by name

Short registry example:

```python
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import FactorFrameEngine
from tiger_factors.factor_frame import IndustryNeutralMomentumDefinition

registry = FactorDefinitionRegistry()
registry.register_many(
    IndustryNeutralMomentumDefinition(
        name="industry_neutral_momentum",
        window=20,
    ),
)

engine = FactorFrameEngine(definition_registry=registry)
engine.add_definition("industry_neutral_momentum")
```

If you typo the name, `registry.get(...)` tells you which definitions are
available and suggests close matches.
- `FactorResearchEngine` is a thin façade over `FactorFrameEngine` for
  chaining feeds, templates, and `run()` in one place
- `CSMModel` in `tiger_factors.factor_frame.csm` provides a research-layer
  cross-sectional stock selection model: fit on long panels, score each
  cross-section, rank names, and emit top/bottom selections that can be
  handed to evaluation or strategy layers
- `build_csm_training_frame(...)` prepares a canonical long research table
  for CSM training when the upstream output already follows Tiger's
  ``date_`` / ``code`` / feature / forward-return layout
- `infer_csm_feature_columns(...)` can auto-detect numeric feature columns
  from a long research frame when you do not want to spell them out by hand
- CSM has three practical handoff paths:
  - `build_csm_model(...)` + `fit(...)` + `predict(...)` for pure research
  - `score_panel(...)` or `selection_panel(...)` for multifactor backtests
  - `build_csm_training_frame(...)` + `csm_factor_frame_research_demo.py`
    for a factor-frame-first workflow
- Recommended CSM workflow:
  1. build or fetch a long `factor_frame`
  2. call `build_csm_training_frame(...)` if you already have the
     Tiger-style `date_` / `code` / feature / label layout
  3. fit `CSMModel`
  4. use `score_panel(...)` for a wide score factor or `selection_panel(...)`
     for a basket-style signal
  5. pass the wide panel to `tiger_factors.multifactor_evaluation`
  6. send the resulting backtest to
     `tiger_factors.multifactor_evaluation.reporting.portfolio`
     for positions / trades / portfolio reports

CSM quickstart:

```python
from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import build_csm_training_frame
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation import run_csm_factor_frame_selection_backtest

training_frame = build_csm_training_frame(
    factor_frame_with_forward_returns,
    ("momentum", "value", "quality"),
)
feature_columns = infer_csm_feature_columns(training_frame)
model = build_csm_model(feature_columns, fit_method="ranknet")
model.fit(training_frame)
result = run_csm_factor_frame_selection_backtest(
    training_frame,
    close_panel,
    top_n=10,
    bottom_n=10,
    long_only=False,
)
```
- factor components in `tiger_factors.factor_frame.factors` provide small
  reusable operators such as `price`, `financial`, `cs_rank`, `cs_zscore`,
  and `ts_momentum`
- classifiers are built before screens, so `ctx.classifier(...)` is available
  when a screen needs sector or industry labels

Core API:

```python
from tiger_factors.factor_frame import FactorFrameEngine

engine = FactorFrameEngine()
engine.feed_price(price_df, align_mode="code_date")
engine.feed_financial(financial_df, align_mode="code_date", lag_sessions=1, fill_method="ffill")
engine.feed_valuation(valuation_df, align_mode="code_date")
engine.feed_macro(macro_df, align_mode="date")
engine.feed("companies", companies_df, align_mode="code")
engine.add_strategy("momentum", momentum_strategy)
engine.add_strategy("quality", quality_strategy)

result = engine.run()
factor_frame = result.factor_frame
combined_frame = result.combined_frame
```

Windowed research works the same way:

```python
engine = FactorFrameEngine(start="2024-01-01", end="2024-06-30")
```

Strategy functions receive a `FactorFrameContext`:

- `ctx.feed_frame(name)` returns the standardized feed table
- `ctx.feed_wide(name, value_column=...)` returns a wide date x code matrix
- `ctx.feed_series(name, value_column=...)` returns a date series
- `ctx.screen(name)` returns the normalized screen frame
- `ctx.classifier(name)` returns the normalized classifier frame
- `ctx.combined_frame` exposes the fully merged research table

`factor_frame` output uses Tiger's canonical long format:

- `date_`
- `code`
- one column per factor strategy

That output can be passed straight into `tiger_factors.factor_evaluation`.

Fluent factor DSL helpers live in `tiger_factors.factor_frame.factors`:

```python
from tiger_factors.factor_frame import factor, price, financial, cs_rank

momentum = factor("momentum", price(value_column="close").pct_change(20).cs_rank())
quality = factor("quality", financial(value_column="net_income").rolling_mean(4))
value = factor("value", (-price(value_column="close")).rank(axis=1))
smoothed = factor("smoothed", price(value_column="close").lag(1).winsorize().zscore())
```

These helpers return reusable factor expressions that can be handed directly
to `engine.add_factor(...)`.

Reusable factor templates are also supported:

```python
from tiger_factors.factor_frame import factor_template, price

momentum_template = factor_template(
    "momentum",
    lambda window=20: price(value_column="close").pct_change(window).cs_rank(),
    defaults={"window": 20},
)

engine.add_factors(
    momentum_template(factor_name="momentum_20", window=20),
    momentum_template(factor_name="momentum_60", window=60),
)
```

For a group-aware example, see `tiger_factors.examples.factor_frame_group_demo`.
For a grouped-factor engine walkthrough that builds families, validates them,
and runs a portfolio backtest, see
`tiger_factors.examples.grouped_factor_engine_demo`.
For a registry-backed grouped-factor walkthrough that uses real structured
definitions and SimFin feeds, see
`tiger_factors.examples.grouped_factor_registry_demo`.
For a sector-classifier / tech-screen / momentum example, see
`tiger_factors.examples.factor_frame_sector_screen_demo`.
For a Tiger API fetch-backed version of the same flow, see
`tiger_factors.examples.factor_frame_sector_screen_fetch_demo`.
For a practical real-data walkthrough that shows common screening patterns,
builds factors only after the screen, and then evaluates each factor
individually, see `tiger_factors.examples.factor_frame_screen_evaluation_demo`.
For a structured `FactorDefinition` example that combines sector-neutral
momentum, residual momentum, event-driven alpha, and a weighted composite,
see `tiger_factors.examples.factor_frame_definition_demo`.
For a shorter registry-first example that registers multiple definitions and
enables them later by name, see
`tiger_factors.examples.factor_frame_definition_registry_demo`.
For a full research handoff into evaluation, see
`tiger_factors.examples.factor_frame_group_research`.
For a library-backed example that fetches price / financial / company /
industry data before feeding the engine, see
`tiger_factors.examples.library_factor_frame_demo`.
For a CSM-first research-to-backtest walkthrough that starts from a
FactorResearchEngine output, see
`tiger_factors.examples.csm_factor_frame_research_demo`.
For a minimal CSM fit / score / selection example, see
`tiger_factors.examples.csm_model_demo`.
For a direct CSM -> selection backtest example, see
`tiger_factors.examples.csm_model_backtest_demo`.
For a CSM -> selection -> local portfolio report example, see
`tiger_factors.examples.csm_model_portfolio_demo`.
For a grouped-factor research engine that combines registered definitions into
family scores, see `tiger_factors.factor_frame.group_engine`.

The recommended end-to-end research recipe is documented in
[`RESEARCH.md`](/Users/yuanhuzhang/Tiger/tiger_quant/tiger_factors/factor_frame/RESEARCH.md).

Common chainable operators include:

- `abs(...)`
- `log(...)`
- `exp(...)`
- `sqrt(...)`
- `sign(...)`
- `lag(...)`
- `diff(...)`
- `cumsum(...)`
- `cumprod(...)`
- `rank_desc(...)`
- `winsorize(...)`
- `zscore(...)`
- `demean(...)`
- `neutralize(...)`
- `group_neutralize(...)`
- `group_demean(...)`
- `group_rank(...)`
- `group_zscore(...)`
- `group_scale(...)`
- `mask(...)`
- `clip_lower(...)`
- `clip_upper(...)`
- `isna(...)`
- `notna(...)`
- `replace(...)`
- `rolling_min(...)`
- `rolling_max(...)`
- `rolling_corr(...)`
- `rolling_cov(...)`
- `rolling_rank(...)`
- `rolling_skew(...)`
- `rolling_kurt(...)`
- `rolling_median(...)`
- `rolling_abs(...)`
- `ts_abs(...)`
- `rolling_sign(...)`
- `ts_sign(...)`
- `rolling_wma(...)`
- `ts_wma(...)`
- `rolling_ema(...)`
- `ts_ema(...)`
- `rolling_delay(...)`
- `ts_delay(...)`
- `rolling_delta(...)`
- `ts_delta(...)`
- `rolling_pct_change(...)`
- `ts_pct_change(...)`
- `rolling_prod(...)`
- `ts_prod(...)`
- `rolling_mean(...)`
- `rolling_std(...)`
- `rolling_var(...)`
- `rolling_sum(...)`
- `mean(...)`
- `sum(...)`
- `std(...)`
- `var(...)`
- `min(...)`
- `max(...)`
- `corr(...)`
- `cov(...)`
- `ifelse(...)`
- `ts_zscore(...)`
- `ts_median(...)`
- `ts_corr(...)`
- `ts_beta(...)`
- `ts_var(...)`
- `ts_skew(...)`
- `ts_kurt(...)`
- `ewm_mean(...)`
- `ewm_std(...)`
- `rolling_sharpe(...)`
- `rolling_information_ratio(...)`
- `minmax_scale(...)`
- `cs_scale(...)`
- `l1_normalize(...)`
- `l2_normalize(...)`
- `fillna(...)`
- `where(...)`
- `top_n(...)`
- `bottom_n(...)`
- `pct_change(...)`
- `rank(...)`
- `rolling_mean(...)`
- `rolling_std(...)`
- `rolling_sum(...)`
- `cs_rank(...)`
- `cs_zscore(...)`
- `cs_demean(...)`
