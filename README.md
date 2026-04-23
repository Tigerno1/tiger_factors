# Tiger Factors

`tiger_factors` is a dedicated factor-building package for the Tiger Quant
workspace. It is designed to:

- fetch data from `tiger_api`
- align trading dates and fundamental report dates
- compute single factors in a consistent long-table format
- save each factor as its own parquet dataset plus metadata json

## Quick Start

Activate the shared Python 3.12 environment first:

```bash
source ../.venv/bin/activate
```

Build one factor:

```bash
python -m tiger_factors --factor momentum_12m_1m --codes AAPL MSFT NVDA --start 2018-01-01 --end 2024-12-31
```

Outputs are written by default to:

```text
../src/output/factors
```

If you already have `BM` and `FSCORE` saved in the Tiger factor store and
want to run the classic value-quality screen end to end, use:

```bash
python -m tiger_factors.examples.value_quality_factor_store_demo \
  --store-root /Volumes/Quant_Disk \
  --factor-provider tiger \
  --bm-factor BM \
  --fscore-factor FSCORE \
  --price-provider yahoo
```

That path loads the stored factor panels, joins them into a research frame,
fetches the matching close panel, runs the `HH` versus `LL` backtest, and
produces both a portfolio tear sheet and a standard equity-curve report.

For a more generic factor-store assembly example that loads any basket of
stored factors into one Tiger-style research frame, use:

```bash
python -m tiger_factors.examples.factor_store_multi_factor_demo \
  --store-root /Volumes/Quant_Disk \
  --factor-provider tiger \
  --factor-names BM FSCORE BMFSCORE \
  --save-csv
```

That demo loads each stored factor as a panel, merges them into one long
research frame, prints coverage statistics, and can optionally write the
assembled frame to CSV for downstream evaluation or backtesting.

For a companion demo that takes the same stored factors, builds an equal-
weight composite, runs the generic multifactor backtest, and emits a
portfolio tear sheet plus an equity-curve chart, use:

```bash
python -m tiger_factors.examples.factor_store_multi_factor_backtest_demo \
  --store-root /Volumes/Quant_Disk \
  --factor-provider tiger \
  --factor-names BM FSCORE BMFSCORE
```

To override the equal-weight default, pass a JSON object as either a literal
string or a file path:

```bash
python -m tiger_factors.examples.factor_store_multi_factor_backtest_demo \
  --store-root /Volumes/Quant_Disk \
  --factor-provider tiger \
  --factor-names BM FSCORE BMFSCORE \
  --weights-json '{"BM": 0.6, "FSCORE": 0.3, "BMFSCORE": 0.1}'
```

## Factor Diagnostics and Multifactors Screening

If a saved factor does not produce a report, the most common causes are:

- the report job was never run for that factor, so only the raw factor parquet exists
- the factor is too sparse, constant, or almost binary
- cleaning fails because `max_loss` is exceeded or quantile/binning is not possible

You can inspect those cases and run a lightweight multifactor screen with:

```bash
python -m tiger_factors.multifactor_evaluation.examples.multifactors_screening_demo \
  --factor-names alpha_007 alpha_021 alpha_046 alpha_049 \
  --top-n 10
```

For the new one-factor-per-file financial/valuation bundles, run the bundled
screening example:

```bash
python -m tiger_factors.examples.single_factor_screening
```

For a full sweep over all `alpha_*` factors, add `--all-factors`.

The demo uses `tiger_factors.factor_evaluation.evaluate_factor_panel()` for single-factor diagnostics and
`tiger_factors.multifactor_evaluation.screen_factor_metrics()` for the shared screening
rules. It does not persist anything unless `--persist-outputs` is set.

For a compact end-to-end demo that starts from raw prices, builds three toy
factors, evaluates them individually, screens them as a small multi-factor
set, and finishes with a backtest, run:

```bash
python -m tiger_factors.examples.three_factor_research_demo
```

The script keeps the factor construction intentionally simple so it is easy to
read and modify:

- 20-day momentum
- 5-day mean reversion
- 20-day low-volatility

It writes single-factor tear sheets and the combined-factor backtest report
into `out/three_factor_research_demo/` by default.

For a single-factor research entrypoint that wraps factor preparation,
evaluation, tear sheets, and horizon analysis, run:

```bash
python -m tiger_factors.research \
  --factor-path /Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_001.parquet \
  --price-path /Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet \
  --factor-column alpha_001 \
  --output-dir /Volumes/Quant_Disk/evaluation/alpha_001_research
```

This entrypoint is the default “single factor research” façade:

- it normalizes factor and price inputs
- evaluates IC / rank IC / Sharpe / turnover / fitness
- writes the standard Tiger tear sheets
- optionally writes the native tear sheet and horizon summary

If you want to go one step higher and build the factor directly from a Tiger
pipeline, use:

```bash
python -m tiger_factors.examples.pipeline_single_factor_research
```

That path is the closest thing to a Zipline-style research flow:

- define the factor in `tiger_factors.factor_maker.pipeline`
- run it through `PipelineEngine`
- hand the resulting factor column to the Tiger research façade
- evaluate, tear-sheet, and horizon-analyze in one place

## Algorithm Layer

If you want the reusable factor-generation algorithms themselves, start with
`tiger_factors.factor_algorithm`:

- `alpha101`
- `gtja191`
- `sunday100plus`
- `experimental`
- `data_mining`
- `financial_factors`
- `qlib_factors`
- `valuation_factors`
- `factor_timing`
- `traditional_factors`

The `traditional_factors` subpackage vendors the public OpenAssetPricing /
Chen-Zimmermann code and exposes it as normal Python callables and a local
portfolio engine:

- [`tiger_factors.factor_algorithm.traditional_factors.available_factors`](./factor_algorithm/traditional_factors/factor_functions.py)
- [`tiger_factors.factor_algorithm.traditional_factors.run_original_factor`](./factor_algorithm/traditional_factors/factor_functions.py)
- [`tiger_factors.factor_algorithm.traditional_factors.TraditionalFactorPipelineEngine`](./factor_algorithm/traditional_factors/pipeline.py)
- [`tiger_factors.factor_algorithm.traditional_factors.TraditionalPortfolioEngine`](./factor_algorithm/traditional_factors/portfolio.py)
- [`tiger_factors.factor_algorithm.traditional_factors.traditional_factor_group_index`](./factor_algorithm/traditional_factors/index.py)

For a package-level overview, see:

- [`tiger_factors/factor_algorithm/README.md`](./factor_algorithm/README.md)

## Factor Frame

If you want a more direct research engine that accepts materialized data
frames, aligns them, and then runs vectorized factor strategies, use
`tiger_factors.factor_frame`.

Example:

```python
from tiger_factors.factor_frame import FactorFrameEngine

engine = FactorFrameEngine()
engine.feed_price(price_df, align_mode="code_date")
engine.feed_financial(financial_df, align_mode="code_date")
engine.feed_valuation(valuation_df, align_mode="code_date")
engine.feed_macro(macro_df, align_mode="date")
engine.feed("companies", companies_df, align_mode="code")
engine.add_strategy("momentum", momentum_factor)
engine.add_strategy("quality", quality_factor)
result = engine.run()
factor_frame = result.factor_frame
```

You can also bound the research window once at the engine level:

```python
engine = FactorFrameEngine(start="2024-01-01", end="2024-06-30")
```

The new factor frame sits below evaluation:

- feeds are materialized DataFrames or Series
- the engine handles alignment, broadcasting, and merging
- each feed declares one of three alignment modes: `code_date`, `date`, or `code`
- `calendar=True/False` controls whether lagging uses business days or calendar days
- `financial` and `valuation` feeds default to a one-session lag
- `build_context()` exposes the normalized feeds and the research config
- `availability_column` enables point-in-time alignment for announcement-date data
- `align_mode="outer"|"intersection"` controls how the shared research table is built
- feed alignment and feed lag are separate concepts
- `add_factor(...)` is the research-friendly alias for `add_strategy(...)`
- `FactorFrameTemplate` and `factor_template(...)` let one template build
  multiple parameterized factor variants
- `FactorDefinition` is the structured recipe layer for more complex factors
  that need explicit preparation, computation, or post-processing steps
- `FactorDefinitionRegistry` lets you pre-register named definitions and
  enable them later by name
- For a shorter registry-first example that registers several definitions and
  enables them later by name, see
  `tiger_factors.examples.factor_frame_definition_registry_demo`.
- `FactorResearchEngine` is a thin façade when you want a single research
  object that chains feeds, templates, and `run()`
- reusable factor components live in `tiger_factors.factor_frame.factors`
- the factor DSL is chainable, so you can write expressions like
  `price(value_column="close").pct_change(20).cs_rank()`
- common post-processing helpers include `lag`, `winsorize`, `zscore`,
  `diff`, `cumsum`, `cumprod`, `rank_desc`, `demean`, `neutralize`,
  `group_neutralize`, `group_demean`, `group_rank`, `group_zscore`, `group_scale`,
  `abs`, `log`, `exp`, `sqrt`, `sign`,
  `mask`, `clip_lower`, `clip_upper`, `isna`, `notna`, `replace`,
  `rolling_min`, `rolling_max`, `rolling_corr`, `rolling_cov`, `rolling_rank`,
  `rolling_skew`, `rolling_kurt`, `rolling_median`,
  `rolling_abs`, `rolling_sign`, `rolling_wma`, `rolling_ema`,
  `rolling_mean`, `rolling_std`, `rolling_var`, `rolling_sum`,
  `mean`, `sum`, `std`, `var`, `min`, `max`, `corr`, `cov`, `ifelse`,
  `ts_zscore`, `ts_median`, `ts_abs`, `ts_sign`, `ts_wma`, `ts_ema`,
  `rolling_delay`, `ts_delay`, `rolling_delta`, `ts_delta`,
  `rolling_pct_change`, `ts_pct_change`, `rolling_prod`, `ts_prod`,
  `ts_corr`, `ts_beta`, `ts_var`, `ts_skew`, `ts_kurt`,
  `ewm_mean`, `ewm_std`, `rolling_sharpe`, `rolling_information_ratio`,
  `minmax_scale`, `cs_scale`,
  `l1_normalize`, `l2_normalize`, `fillna`, `where`, `top_n`, and `bottom_n`
- strategies are pure pandas / NumPy functions
- the returned `factor_frame` can be sent to `FactorEvaluationEngine`

For a runnable demo, see:

```bash
python -m tiger_factors.examples.factor_frame_demo
```

For a group-aware demo with sector neutralization and group ranking, see:

```bash
python -m tiger_factors.examples.factor_frame_group_demo
```

For a focused classifier/screen/factor example, see:

```bash
python -m tiger_factors.examples.factor_frame_sector_screen_demo
```

That example uses screen as a pre-factor universe gate; any post-factor
ranking or thresholding belongs in evaluation or strategy.

For a practical real-data walkthrough that shows common screening patterns,
builds factors only after the screen, and then evaluates each factor
individually, see:

```bash
python -m tiger_factors.examples.factor_frame_screen_evaluation_demo
```

For the Tiger API fetch-backed version of that flow, see:

```bash
python -m tiger_factors.examples.factor_frame_sector_screen_fetch_demo
```

For a full group-aware research flow that hands the factor frame into
evaluation, see:

```bash
python -m tiger_factors.examples.factor_frame_group_research
```

For a library-backed demo that fetches price, financial, companies, and
industry data first, then feeds them into the factor frame engine, see:

```bash
python -m tiger_factors.examples.library_factor_frame_demo
```

For the factor-frame research recipe, see:

- [`tiger_factors/factor_frame/RESEARCH.md`](/Users/yuanhuzhang/Tiger/tiger_quant/tiger_factors/factor_frame/RESEARCH.md)

## Migrated Utility Modules

Legacy factor helpers that used to live under `vnpy_strategies` now live here:

- `tiger_factors.utils.group_operators`
- `tiger_factors.utils.cross_sectional`
- `tiger_factors.utils.time_series`
- `tiger_factors.utils.combine`

## Pipeline-Style API

`tiger_factors.factor_maker.pipeline` provides a lightweight, Zipline-Pipeline-like
abstraction that works directly with `tiger_api` data.

Example:

```python
from tiger_factors.factor_maker.pipeline import Pipeline, PipelineEngine, Returns, USEquityPricing

momentum = Returns(window_length=252)

pipe = Pipeline(
    columns={
        "momentum": momentum,
        "close_rank": USEquityPricing.column("close").rank(ascending=False),
    },
    screen=momentum.top(100),
)

engine = PipelineEngine()
result = engine.run_pipeline(
    pipe,
    codes=["AAPL", "MSFT", "NVDA"],
    start="2020-01-01",
    end="2024-12-31",
)
```

Tiger API example with both price and fundamentals routed by provider:

```python
from tiger_factors.factor_maker.pipeline import Pipeline, PipelineEngine, Returns, RollingStdDev
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_maker.pipeline import SimFinBalanceSheet, SimFinIncomeStatement, USEquityPricing

library = TigerFactorLibrary(region="us", sec_type="stock", price_provider="yahoo")

codes = library.resolve_universe_codes(
    provider="simfin",
    dataset="companies",
    code_column="symbol",
    limit=500,
)

engine = PipelineEngine(
    library=library,
    region="us",
    sec_type="stock",
    price_provider="yahoo",
    calendar_source="auto",
    provider_overrides={
        "source_type:price": "yahoo",
        "dataset:income_statement": "simfin",
        "dataset:balance_sheet": "simfin",
    },
    fundamental_use_point_in_time=True,
    fundamental_lag_sessions=1,
)

momentum_20 = Returns(window_length=20)
volatility_20 = RollingStdDev(inputs=[USEquityPricing.column("close")], window_length=20)

pipe = Pipeline(
    columns={
        "close": USEquityPricing.column("close"),
        "volume": USEquityPricing.column("volume"),
        "momentum_20": momentum_20,
        "volatility_20": volatility_20,
        "roe_like": SimFinIncomeStatement.column("net_income") / SimFinBalanceSheet.column("total_equity"),
    },
    screen=(
        USEquityPricing.column("close").notnull()
        & USEquityPricing.column("volume").notnull()
        & USEquityPricing.column("volume").top(500)
    ),
)

result = engine.run_pipeline(
    pipe,
    codes=codes,
    start="2024-01-01",
    end="2024-03-31",
)
```

There is also a runnable script at `tiger_factors/examples/pipeline_tigerapi_example.py`.

## Calculation Engine

`tiger_factors.utils.calculation` now stays extremely small. It advances through
trading days and, when you give it an intraday interval, walks from session open
to session close.
For day-driven runs, use `on_pre_open` and `on_post_close`.

```python
from tiger_factors.utils.calculation import CallbackCalculationStrategy, FactorStrategyTemplate, FactorTimeEngine, Interval

engine = FactorTimeEngine(calendar="NYSE", start="2024-01-01")
strategy = CallbackCalculationStrategy(
    on_day=lambda step: {
        "day": step.trading_day.isoformat(),
        "time": step.at.isoformat(),
    },
)
results = engine.run(
    strategy,
    trading_days=["2024-01-02", "2024-01-03", "2024-01-04"],
)

engine.set_interval(min=30)
engine.set_interval(day=1)
engine.set_interval(interval=Interval(day=2))
```

If you want to run logic without subclassing, use the callback wrapper:

```python
from tiger_factors.utils.calculation import CallbackCalculationStrategy

strategy = CallbackCalculationStrategy(
    on_day=lambda step: {
        "day": step.trading_day.isoformat(),
        "step_index": step.step_index,
    },
    on_pre_open=lambda step: print("pre-open", step.trading_day),
    on_post_close=lambda step: print("post-close", step.trading_day),
)
results = engine.run(strategy, trading_days=["2024-01-02"])
```

The engine does not preselect a universe or fetch prices for you. It only
advances time and hands each step to your strategy.

If you want a light wrapper around the time engine, use `FactorMainEngine`:

```python
from tiger_factors.utils.calculation import FactorMainEngine, FactorTimeEngine

main_engine = FactorMainEngine(
    time_engine=FactorTimeEngine(calendar="NYSE", start="2024-01-01"),
)
results = main_engine.run(strategy, trading_days=["2024-01-02"])
```

## Vectorization

`tiger_factors.factor_maker.vectorization` builds a trading-calendar master table first and
then left-joins one or more datasets onto it. The output keeps the original
observation time in `date_` and the shifted effective time in `eff_at`
(`lag=1` by default).

The transformer also exposes staged merge helpers so you can keep different
join shapes separate:

- `merge_code_list(...)` for code-only lookup tables.
- `merge_date_list(...)` for date-only series.
- `merge_code_date_list(...)` for panel data attached to the trading calendar.
- `merge_other_list(...)` for custom-key joins such as `industry_id`.

`merge_other_list(...)` returns an intermediate keyed result. It does not
automatically attach itself to the trading calendar, so you can decide how to
reuse it in a later join step.

```python
import pandas as pd
from tiger_factors.factor_maker.vectorization import FactorVectorizationTransformer, VectorDatasetSpec
from tiger_factors.utils.calculation import Interval

engine = FactorVectorizationTransformer(calendar="NYSE", start="2024-01-01", interval=Interval(day=1), lag=1)
result = engine.merge(
    [
        VectorDatasetSpec(
            name="price",
            frame=pd.DataFrame({"date_": ["2024-01-02"], "close": [10.0]}),
            time_column="date_",
        ),
    ],
)
print(result.frame)
```

You can also pass a plain list of DataFrame / Polars tables. The engine uses
the trading calendar as the left table and merges every input table onto it:

```python
result = engine.merge([price_df, signal_df], time_column="date_", forward_fill=[False, True])
```

If a dataset requests forward fill, its non-key columns are filled along the
calendar order after the left join. The calendar rows always keep both
`date_` and `eff_at` so you can join signals to their usable time later
without introducing lookahead.

`forward_fill` accepts either a single boolean or a list with the same length
as `datasets`. In list form, `None` means "keep that dataset's default" and
does not fall back to a silent auto-padding rule.

You can also convert a long panel to a wide matrix directly from the engine:

```python
wide_close = engine.long_to_wide(price_df, time_column="date_", code_column="code", value_columns=["close"])
wide_ohlc = engine.long_to_wide(price_df, time_column="date_", code_column="code", value_columns=["open", "high", "low", "close"])
```

For a single value column, the output is a `date x code` matrix. For multiple
value columns, the output uses `code__field` columns such as `AAA__open` and
`AAA__close`.

The transformer can also normalize raw OHLCV data into adjusted long form.
This path simply maps the table to the reference `date_` / `code` schema and
delegates the whole-table adjustment to `tiger_reference.adj_df`:

```python
adjusted = engine.adjust(raw_price_df, time_column="date_", code_column="code")
```

When `adj_close` is present, `open/high/low/close` are scaled by the split
adjustment factor implied by `close / adj_close`, `volume` is adjusted by the
split factor only, and an optional `dividend` column adds an extra price-only
adjustment component. The result keeps the long panel layout for downstream
factor pipelines.

## Data Mining

`tiger_factors.factor_algorithm.data_mining` contains a cleaned set of alpha-candidate formulas
derived from the data-mining screenshots. It focuses on the short list of
manually curated factors that are worth testing first.

```python
from tiger_factors.factor_algorithm.data_mining import DataMiningEngine, available_factors as data_mining_factor_names

print(data_mining_factor_names())
```

## Financial Factors

`tiger_factors.factor_algorithm.financial_factors` provides a compact SimFin-based financial
factor engine with three thin frequency-specific classes. The companion
`record_financial_factors()` helper writes the quarterly / annual / TTM bundles
to disk:

- `QuarterlyFinancialFactorEngine`
- `AnnualFinancialFactorEngine`
- `TTMFinancialFactorEngine`

The engine loads SimFin balance sheet / income statement / cash-flow data,
derives a broad set of accounting ratios, and also builds a separate
price-aligned valuation panel for PB / PS / PE style metrics. Both panels are
expanded through the same compact transform grid.

```python
from tiger_factors.factor_algorithm.financial_factors import record_financial_factors

runs = record_financial_factors(
    library=library,
    codes=["AAPL", "MSFT"],
    start="2018-01-01",
    end="2024-12-31",
    output_root="/Volumes/Quant_Disk/evaluation/financial_factors",
    price_provider="simfin",
)
print(runs[0]["parquet_path"])
```

There is also a runnable script at `tiger_factors/examples/financial_factor_engine.py`.

## Valuation Factors

`tiger_factors.factor_algorithm.valuation_factors` is the dedicated home for price-aligned
valuation metrics such as PB, PS, PE, EV/EBITDA, and related transforms. The
valuation recorder uses SimFin price data together with SimFin financial data.

```python
from tiger_factors.factor_algorithm.valuation_factors import record_valuation_factors

runs = record_valuation_factors(
    library=library,
    codes=["AAPL", "MSFT"],
    start="2018-01-01",
    end="2024-12-31",
    output_root="/Volumes/Quant_Disk/evaluation/valuation_factors",
    price_provider="simfin",
)
print(runs[0]["parquet_path"])
```

There is also a runnable script at `tiger_factors/examples/valuation_factor_engine.py`.

## Alpha101

`tiger_factors.factor_algorithm.alpha101` now includes a full `Alpha101Engine` with factor ids `1..101`.

If you want a rolling-smoothed Alpha101 workflow that computes the factors in
memory and then runs the shared multifactor screening rules without saving the
single-factor files first, use:

```bash
python -m tiger_factors.examples.alpha101_smoothed_screening
```

Minimal usage:

```python
from tiger_factors.factor_algorithm.alpha101 import Alpha101Engine

engine = Alpha101Engine(price_df)
alpha_001 = engine.compute(1)
alpha_101 = engine.compute(101)
all_factors = engine.compute_all()
print(engine.alpha_description(1))  # 这个 alpha 的中文摘要
```

If you want to compute several alpha ids in parallel and save each factor as
its own parquet file, use the parallel helper:

```python
from tiger_factors.factor_maker.vectorization.indicators import Alpha101IndicatorTransformer

transformer = Alpha101IndicatorTransformer(
    calendar="XNYS",
    start="2024-01-01",
    end="2024-12-31",
)
result = transformer.compute_all_alpha101_parallel(
    alpha_ids=[1, 2, 3],
    codes=["AAPL", "MSFT"],
    start="2024-01-01",
    end="2024-12-31",
    compute_workers=4,
    save_workers=2,
    save_factors=True,
)

print(result.factor_frame.head())
print(result.saved_factor_paths)
```

`compute_all_alpha101_parallel(...)` uses a process pool for factor computation
and a thread pool for parquet writes. Each factor is still stored as its own
parquet dataset under the factor output directory.

Input columns:

- required: `date_` or `date`, `code` or `symbol`, `open`, `high`, `low`, `close`, `volume`
- optional: `vwap`, `return`, `industry`, `market_value`

If optional columns are missing:

- `vwap` falls back to `(open + high + low + close) / 4`
- `return` is computed from close-to-close pct change
- `industry` falls back to `unknown`
- `market_value` falls back to `close * volume`

## Qlib Alpha Factor Sets

`tiger_factors.factor_algorithm.qlib_factors` exposes Qlib's predefined Alpha158 / Alpha360
feature families as Tiger-native factor set adapters.

They are **not single factors**. They are factor libraries / feature sets.

```python
from tiger_factors.factor_algorithm.qlib_factors import Alpha158FactorSet, Alpha360FactorSet

print(len(Alpha158FactorSet().feature_names))  # 158
print(len(Alpha360FactorSet().feature_names))  # 360
```

If you already have a Qlib provider directory, you can also fetch the actual
factor panels through the Tiger wrapper:

```python
from tiger_factors.factor_algorithm.qlib_factors import QlibAlphaFactorSetEngine

engine = QlibAlphaFactorSetEngine(
    provider_uri="/path/to/qlib/provider",
    region="us",
    instruments="all",
    start_time="2020-01-01",
    end_time="2020-12-31",
)
alpha158_panel = engine.fetch_alpha158(selector=slice("2020-01-01", "2020-12-31"))
alpha360_panel = engine.fetch_alpha360(selector=slice("2020-01-01", "2020-12-31"))
```
