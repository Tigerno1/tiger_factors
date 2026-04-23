"""Alpha101 8-factor strategy demo without PCA.

This is a thin alias for the direct multifactor strategy flow:

1. load the balanced 8-factor Alpha101 candidate pool
2. screen the factors
3. remove highly correlated factors
4. assign weights
5. blend the selected factors
6. backtest the blended factor
7. optionally generate local positions, trade, and portfolio reports

The script intentionally does not use PCA. It reuses the direct multifactor
strategy implementation so the naming is explicit while the logic stays in one
place.
"""

from __future__ import annotations

from tiger_factors.examples.alpha101_multifactor_portfolio_demo import main


if __name__ == "__main__":
    main()
