# `tiger_factors.factor_algorithm.traditional_factors`

This subpackage vendors the public OpenAssetPricing / Chen-Zimmermann code and
wraps it with Tiger-friendly Python APIs.

## What is included

- `factor_functions.py`
  - signal-by-signal wrappers around the upstream Python predictors
  - exposes one callable per vendored signal
  - includes metadata lookup and upstream script execution helpers
- `pipeline.py`
  - a batch runner that executes the vendored signal catalog and stores each
    result through `TigerFactorLibrary.save_factor`
- `portfolio.py`
  - a local Python portfolio engine for the monthly / daily portfolio logic
- `index.py`
  - a factor-family grouping index that clusters the vendored signals into
    broad families such as value, profitability, investment, risk, liquidity,
    momentum, and related groups
- `common_factors.py`
  - Tiger-side aliases and proxies for widely used finance labels such as
    BM, FSCORE, BMFSCORE, PEAD, IVOL, LIQ, SMB, HML, RMW, CMA, BAB, QMJ,
    INFLC, INFLV, and related common factor names from factor zoo papers
  - `BM` is the direct book-to-market proxy from the vendored OpenAssetPricing
    signal catalog
  - `FSCORE` is the direct Piotroski F-score proxy from the vendored
    `PS` signal
  - `BMFSCORE` is a simple rank-mean composite that captures the classic
    "high B/M + high F-score" value-quality screen
  - `run_value_quality_combo()` returns BM, F-score, the composite, and a
    two-by-two high/low bucket view for the classic value-quality screen
  - `run_value_quality_long_short_backtest()` runs the corresponding HH-versus-
    LL long/short backtest on a close-panel
  - `examples/value_quality_long_short_demo.py` shows a small synthetic
    end-to-end example with a local portfolio report
  - `run_value_quality_combo_from_columns()` builds the same screen directly
    from existing `BM` / `FSCORE` columns for real-data workflows
  - `examples/value_quality_real_data_demo.py` shows the file-based real-data
    workflow with a local portfolio report
  - `INFLC` is a macro inflation proxy that can be aligned to stock panels via
    a CPI / PCE-style macro series passed through `datasets={...}`
  - `INFLV` is the corresponding inflation-volatility proxy built from the same
    macro series
  - chart-style aliases such as `SMBs`, `HMLs`, `CMAs`, `RMWs`, `MKTBs`,
    `STREV`, and `MOMB` are normalized to the same canonical Tiger labels
  - `common_factor_display_names()` returns the preferred chart-style label for
    each canonical common-factor name
  - `common_factor_family_summary()` gives a compact one-row-per-family table
    with counts and chart-style labels
  - `common_factor_group_markdown()` renders the full canonical catalog as a
    markdown table
- `openassetpricing/`
  - vendored upstream code, metadata, and signal documentation

## Public entry points

```python
from tiger_factors.factor_algorithm.traditional_factors import available_factors
from tiger_factors.factor_algorithm.traditional_factors import factor_metadata
from tiger_factors.factor_algorithm.traditional_factors import run_original_factor
from tiger_factors.factor_algorithm.traditional_factors import available_common_factors
from tiger_factors.factor_algorithm.traditional_factors import common_factor_group_frame
from tiger_factors.factor_algorithm.traditional_factors import common_factor_group_markdown
from tiger_factors.factor_algorithm.traditional_factors import common_factor_group_names
from tiger_factors.factor_algorithm.traditional_factors import common_factor_family_summary
from tiger_factors.factor_algorithm.traditional_factors import common_factor_family_markdown
from tiger_factors.factor_algorithm.traditional_factors import common_factor_display_names
from tiger_factors.factor_algorithm.traditional_factors import common_factor_spec
from tiger_factors.factor_algorithm.traditional_factors import find_common_factor_group
from tiger_factors.factor_algorithm.traditional_factors import run_common_factor
from tiger_factors.factor_algorithm.traditional_factors import TraditionalFactorPipelineEngine
from tiger_factors.factor_algorithm.traditional_factors import TraditionalPortfolioEngine
from tiger_factors.factor_algorithm.traditional_factors import traditional_factor_group_frame
```

## How it fits the Tiger stack

- `factor_algorithm`
  - holds the reusable factor engines
- `factor_store`
  - stores factor outputs through `FactorSpec`
- `factor_evaluation`
  - evaluates a single saved factor
- `multifactor_evaluation`
  - screens and validates factor families
- `tiger_analysis`
  - generates local tear sheets and portfolio reports

If you only need the upstream factor logic, call `run_original_factor(...)`.
If you want batch execution and storage, use
`TraditionalFactorPipelineEngine`. If you want the portfolio layer, use
`TraditionalPortfolioEngine`. If you want to inspect the broad family
grouping, use `traditional_factor_group_index()` or
`traditional_factor_group_frame()`.
