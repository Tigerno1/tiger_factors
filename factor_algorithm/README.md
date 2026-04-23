# `tiger_factors.factor_algorithm`

This package contains the algorithm layer for factor generation and related
research workflows.

## What lives here

- `alpha101`
  - WorldQuant-style Alpha101 factor engine and descriptions.
- `gtja191`
  - GTJA 191 factor engine.
- `sunday100plus`
  - Sunday100+ factor engine.
- `experimental`
  - Research-only factor algorithms such as market breathing and news entropy.
- `data_mining`
  - Data-mining factor engines and practical factor wrappers.
- `financial_factors`
  - Annual, quarterly, and TTM financial factor engines.
- `qlib_factors`
  - Qlib-style alpha factor sets.
- `valuation_factors`
  - Valuation factor engine.
- `factor_timing`
  - Factor timing pipeline.
- `traditional_factors`
  - Vendored OpenAssetPricing / Chen-Zimmermann style factor code and portfolio logic.
  - Includes a Tiger-side common factor catalog for widely used labels such as
    BM, FSCORE, BMFSCORE, PEAD, IVOL, LIQ, SMB, HML, RMW, CMA, BAB, QMJ, the
    macro inflation proxy INFLC, and the inflation-volatility proxy INFLV.

## Vendored OpenAssetPricing code

The `traditional_factors` subpackage vendors the public Chen-Zimmermann code
base and exposes it as normal Python callables.

Useful entry points:

- [`tiger_factors.factor_algorithm.traditional_factors.available_factors`](./traditional_factors/factor_functions.py)
- [`tiger_factors.factor_algorithm.traditional_factors.factor_metadata`](./traditional_factors/factor_functions.py)
- [`tiger_factors.factor_algorithm.traditional_factors.run_original_factor`](./traditional_factors/factor_functions.py)
- [`tiger_factors.factor_algorithm.traditional_factors.TraditionalFactorPipelineEngine`](./traditional_factors/pipeline.py)
- [`tiger_factors.factor_algorithm.traditional_factors.TraditionalPortfolioEngine`](./traditional_factors/portfolio.py)
- [`tiger_factors.factor_algorithm.traditional_factors.traditional_factor_group_index`](./traditional_factors/index.py)
- [`tiger_factors.factor_algorithm.traditional_factors.traditional_factor_group_frame`](./traditional_factors/index.py)
- [`tiger_factors.factor_algorithm.traditional_factors.common_factor_group_index`](./traditional_factors/common_factors.py)
- [`tiger_factors.factor_algorithm.traditional_factors.common_factor_group_frame`](./traditional_factors/common_factors.py)
- [`tiger_factors.factor_algorithm.traditional_factors.common_factor_group_names`](./traditional_factors/common_factors.py)
- [`tiger_factors.factor_algorithm.traditional_factors.find_common_factor_group`](./traditional_factors/common_factors.py)
- [`tiger_factors.factor_algorithm.traditional_factors.run_common_factor`](./traditional_factors/common_factors.py)

The upstream source and metadata are vendored under:

- [`traditional_factors/openassetpricing`](./traditional_factors/openassetpricing)

## Root exports

The package root lazily re-exports the main engines and helpers so you can do:

```python
from tiger_factors.factor_algorithm import Alpha101Engine, TraditionalPortfolioEngine
```

without importing every algorithm family eagerly.
