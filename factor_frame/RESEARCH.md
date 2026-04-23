# Factor Frame Research Recipe

This note describes the recommended Tiger workflow for research on top of
`tiger_factors.factor_frame`.

## 1. Feed materialized data

The engine expects DataFrames or Series that you already materialized:

- `price`
- `financial`
- `valuation`
- `macro`
- `news`

Feed data directly and let the engine handle:

- alignment
- lagging
- calendar-aware shifting
- point-in-time adjustment via `availability_column`
- per-feed fill via `fill_method` such as `ffill`, `bfill`, or `both`
- fill is always time-wise within the same field; columns do not cross-fill

Typical usage:

```python
engine = FactorFrameEngine(freq="1d", bday_lag=True, use_point_in_time=True)
engine.feed_price(price_df)
engine.feed_financial(financial_df, lag_sessions=1)
engine.feed_valuation(valuation_df, lag_sessions=1)
engine.feed_macro(macro_df, code_column=None)
```

Use `freq="1d"` for day-level primary data and hour/minute-style intraday
frequencies such as `freq="1h"`, `freq="1min"`, `freq="15min"`,
`freq="20min"`, `freq="30min"`, or `freq="2h"` for intraday primary data.
If you also set `calendar`, the engine will use that exchange calendar
explicitly; leaving it `None` disables exchange-session shifting.
For intraday data, `label_side="right"` is the default convention, `left`
means start-labeled bars, and `auto` lets the engine try to infer the source
convention when possible.

If you want a thinner façade that chains the same flow in one object, use
`FactorResearchEngine` from the same package. It wraps `FactorFrameEngine`
and keeps the `feed -> add_template/add_definition -> run` style compact.

## 2. Add research strategies or definitions

Write strategies as pure pandas / NumPy functions that accept a
`FactorFrameContext`. Use `FactorDefinition` when the factor needs an explicit
prepare / compute / postprocess structure.

```python
def momentum(ctx):
    close = ctx.feed_wide("price", "close")
    return close.pct_change(20)

def quality(ctx):
    net_income = ctx.feed_wide("financial", "net_income")
    total_equity = ctx.feed_wide("financial", "total_equity")
    return net_income.div(total_equity.replace(0, pd.NA))
```

Register them with:

```python
engine.add_strategy("momentum", momentum)
engine.add_strategy("quality", quality)
```

If the workflow is factor-centric, `add_factor(...)` is the same entrypoint.

For a structured factor recipe, use a definition object:

```python
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import IndustryNeutralMomentumDefinition

registry = FactorDefinitionRegistry()
industry_neutral_momentum = IndustryNeutralMomentumDefinition(
    name="industry_neutral_momentum",
    window=20,
)
registry.register(industry_neutral_momentum)
engine = FactorFrameEngine(definition_registry=registry)
engine.add_definition("industry_neutral_momentum")
```

If you want to register several definitions at once, use `register_many(...)`:

```python
registry.register_many(
    industry_neutral_momentum,
    residual_momentum,
    event_alpha,
)
```

If you typo the name, `registry.get(...)` will raise a `KeyError` that lists
available definitions and suggests close matches.

### Screens and classifiers

- `add_classifier(...)` registers a categorical lookup or grouping term
- classifiers are built first and can be consumed by screens
- `add_screen(...)` registers a boolean universe gate
- screens are applied before factor computation only
- factor callables see the screened feeds and screened combined context
- there is no post-factor screen pass in this layer
- post-factor ranking, cutoff, or thresholding belongs in evaluation or
  strategy layers
- classifiers are available on the `FactorFrameContext` for group-aware research
- common pre-factor screens are usually:
  - sector / industry universe gates
  - minimum price floors
  - minimum history checks for rolling windows
  - financial completeness checks for sparse fundamentals
  - optional factor-specific gates such as liquidity or listing age
- for a full pre-screen -> factor -> evaluation walkthrough, see
  `tiger_factors.examples.factor_frame_screen_evaluation_demo`

```python
engine.add_screen("liquidity", lambda ctx: ctx.feed_wide("price", "close") > 0)
engine.add_classifier(
    "sector",
    lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"],
)
```

## 3. Use the factor DSL

The `tiger_factors.factor_frame.factors` module provides chainable helpers for
expressing factors with a compact syntax.

```python
from tiger_factors.factor_frame import factor, price, financial

momentum = factor("momentum", price(value_column="close").pct_change(20).cs_rank())
quality = factor("quality", financial(value_column="net_income").rolling_mean(4))
```

## 4. Run the engine

```python
result = engine.run()
factor_frame = result.factor_frame
combined_frame = result.combined_frame
```

`factor_frame` is returned in long format:

- `date_`
- `code`
- one column per strategy

That frame can be passed directly into the Tiger factor-evaluation layer.

## 5. Evaluate separately

Keep evaluation separate from research construction.

```python
from tiger_factors.research import SingleFactorResearch

research = SingleFactorResearch(
    factor_frame=factor_frame[["date_", "code", "momentum"]],
    price_frame=price_long,
    factor_column="momentum",
)
result = research.run()
```

## 6. Group-aware research

If you need sector or industry workflows, the factor frame includes group
helpers:

- `group_neutralize(...)`
- `group_demean(...)`
- `group_rank(...)`
- `group_zscore(...)`
- `group_scale(...)`

The recommended pattern is:

```python
sector = pd.Series({"AAPL": "technology", "MSFT": "technology", "NVDA": "semis"})
factor = group_neutralize(price_df.pct_change(10), sector)
```

If you want a full example, see:

- `tiger_factors.examples.factor_frame_demo`
- `tiger_factors.examples.factor_frame_group_demo`
- `tiger_factors.examples.factor_frame_group_research`

## 7. Practical checklist

- feed materialized data, do not fetch inside the engine
- keep research and evaluation separate
- use `lag_sessions` on each feed when the source has reporting delay
- use `availability_column` for point-in-time alignment
- use `group_*` helpers when a sector or industry label changes the signal
- keep the final `factor_frame` in long format so it can be evaluated
