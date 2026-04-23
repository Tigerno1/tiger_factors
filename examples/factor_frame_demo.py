"""FactorFrame demo.

This example shows the research flow the new Tiger factor frame is meant to
support:

1. feed materialized price / financial / valuation / macro data
2. let the engine align and broadcast those feeds
3. compute three vectorized factors
4. inspect the merged research table and the final factor frame
5. neutralize a factor by sector to show grouped research workflows

The example is self-contained and uses synthetic data so it can run without
network access.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_frame import FactorFrameEngine
from tiger_factors.factor_frame import group_neutralize
from tiger_factors.factor_frame import group_rank
from tiger_factors.factor_frame import group_scale
from tiger_factors.factor_frame import group_zscore


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=60)
    codes = ["AAPL", "MSFT", "NVDA", "AMZN"]

    rng = np.random.default_rng(11)
    price = pd.DataFrame(index=dates, columns=codes, dtype=float)
    price.iloc[0] = [100.0, 120.0, 140.0, 160.0]
    for i in range(1, len(dates)):
        shock = rng.normal(0.0007, 0.015, size=len(codes))
        price.iloc[i] = price.iloc[i - 1].to_numpy(dtype=float) * (1.0 + shock)

    financial_rows: list[dict[str, object]] = []
    valuation_rows: list[dict[str, object]] = []
    for code_idx, code in enumerate(codes):
        base_equity = 80.0 + code_idx * 15.0
        base_income = 12.0 + code_idx * 4.0
        base_pe = 18.0 + code_idx * 2.0
        for date in dates[::10]:
            financial_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "net_income": base_income + float(code_idx),
                    "total_equity": base_equity + 0.5 * code_idx,
                }
            )
            valuation_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "pe_ratio": base_pe + 0.25 * code_idx,
                }
            )

    macro = pd.DataFrame(
        {
            "date_": dates,
            "cpi": np.linspace(3.0, 4.0, len(dates)),
        }
    )

    return price, pd.DataFrame(financial_rows), pd.DataFrame(valuation_rows), macro


def momentum_factor(ctx) -> pd.DataFrame:
    close = ctx.feed_wide("price", "close")
    cpi = ctx.feed_series("macro", "cpi").reindex(close.index).ffill()
    factor = close.pct_change(20).sub(cpi.pct_change().reindex(close.index).fillna(0.0), axis=0)
    return factor


def quality_factor(ctx) -> pd.DataFrame:
    net_income = ctx.feed_wide("financial", "net_income")
    total_equity = ctx.feed_wide("financial", "total_equity")
    return net_income.div(total_equity.replace(0, np.nan))


def value_factor(ctx) -> pd.DataFrame:
    pe_ratio = ctx.feed_wide("valuation", "pe_ratio")
    return -pe_ratio


def sector_neutral_momentum_factor(ctx) -> pd.DataFrame:
    close = ctx.feed_wide("price", "close")
    raw = close.pct_change(20)
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
        }
    ).reindex(raw.columns)
    return group_neutralize(raw, sector, method="demean")


def sector_ranked_value_factor(ctx) -> pd.DataFrame:
    pe = ctx.feed_wide("valuation", "pe_ratio")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
        }
    ).reindex(pe.columns)
    return group_rank(-pe, sector)


def sector_zscore_quality_factor(ctx) -> pd.DataFrame:
    net_income = ctx.feed_wide("financial", "net_income")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
        }
    ).reindex(net_income.columns)
    return group_zscore(net_income, sector)


def sector_scaled_value_factor(ctx) -> pd.DataFrame:
    pe = ctx.feed_wide("valuation", "pe_ratio")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
        }
    ).reindex(pe.columns)
    return group_scale(-pe, sector)


def main() -> None:
    price, financial, valuation, macro = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.feed_macro(macro, code_column=None)
    engine.add_strategy("momentum", momentum_factor)
    engine.add_strategy("quality", quality_factor)
    engine.add_strategy("value", value_factor)
    engine.add_strategy("sector_neutral_momentum", sector_neutral_momentum_factor)
    engine.add_strategy("sector_ranked_value", sector_ranked_value_factor)
    engine.add_strategy("sector_zscore_quality", sector_zscore_quality_factor)
    engine.add_strategy("sector_scaled_value", sector_scaled_value_factor)

    result = engine.run()
    print("combined_frame:", result.combined_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
