# factor_test

Factor-level diagnostics and validation utilities.

## Submodules

- `multiple_testing`: classical and Bayesian multiple-testing controls
- `validation`: bootstrap, permutation, and split-stability helpers
- `convexity`: factor convexity / tail nonlinearity tests
- `stability`: factor decay, recent-IC stability, and effectiveness wrappers
- `market_state`: standalone market regime / state detection utilities
- `regime`: factor performance tests conditioned on market states

## Typical usage

```python
from tiger_factors.factor_test import factor_convexity_test
from tiger_factors.factor_test import factor_recent_ic_test
from tiger_factors.factor_test import factor_decay_test
from tiger_factors.factor_test import fit_bayesian_mixture
from tiger_factors.factor_test import market_state_test
from tiger_factors.factor_test import factor_regime_ic_test
from tiger_factors.factor_test import factor_regime_decay_test
from tiger_factors.factor_test import factor_regime_turning_point_test
from tiger_factors.factor_test import factor_regime_report
```
