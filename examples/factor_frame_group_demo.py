"""Grouped FactorFrame demo.

This example focuses on the research patterns that usually matter first in
sector-aware factor work:

1. build a unified research context from materialized price / financial /
   valuation feeds
2. express group-aware transforms such as neutralization, group ranking,
   group z-scoring, and group scaling
3. run those factors through the Tiger factor frame engine
4. inspect the resulting factor frame

The demo uses synthetic data so it can run without network access.
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
from tiger_factors.factor_frame import group_demean
from tiger_factors.factor_frame import group_neutralize
from tiger_factors.factor_frame import group_rank
from tiger_factors.factor_frame import group_scale
from tiger_factors.factor_frame import group_zscore


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=40)
    codes = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "BAC"]
    sectors = {
        "AAPL": "technology",
        "MSFT": "technology",
        "NVDA": "semis",
        "AMZN": "internet",
        "JPM": "financials",
        "BAC": "financials",
    }

    rng = np.random.default_rng(21)
    price = pd.DataFrame(index=dates, columns=codes, dtype=float)
    price.iloc[0] = [100.0, 120.0, 140.0, 160.0, 90.0, 85.0]
    for i in range(1, len(dates)):
        shock = rng.normal(0.0005, 0.014, size=len(codes))
        price.iloc[i] = price.iloc[i - 1].to_numpy(dtype=float) * (1.0 + shock)

    financial_rows: list[dict[str, object]] = []
    valuation_rows: list[dict[str, object]] = []
    for code_idx, code in enumerate(codes):
        for date in dates[::8]:
            financial_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "net_income": 10.0 + 2.0 * code_idx,
                    "total_equity": 50.0 + 4.0 * code_idx,
                }
            )
            valuation_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "pe_ratio": 14.0 + 1.5 * code_idx,
                }
            )

    sector_series = pd.Series(sectors, name="sector")
    return price, pd.DataFrame(financial_rows), pd.DataFrame(valuation_rows), sector_series


def sector_neutral_momentum(ctx) -> pd.DataFrame:
    close = ctx.feed_wide("price", "close")
    raw = close.pct_change(10)
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
            "JPM": "financials",
            "BAC": "financials",
        }
    ).reindex(raw.columns)
    return group_neutralize(raw, sector)


def sector_ranked_value(ctx) -> pd.DataFrame:
    pe = ctx.feed_wide("valuation", "pe_ratio")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
            "JPM": "financials",
            "BAC": "financials",
        }
    ).reindex(pe.columns)
    return group_rank(-pe, sector)


def sector_zscore_quality(ctx) -> pd.DataFrame:
    net_income = ctx.feed_wide("financial", "net_income")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
            "JPM": "financials",
            "BAC": "financials",
        }
    ).reindex(net_income.columns)
    return group_zscore(net_income, sector)


def sector_scaled_value(ctx) -> pd.DataFrame:
    pe = ctx.feed_wide("valuation", "pe_ratio")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
            "JPM": "financials",
            "BAC": "financials",
        }
    ).reindex(pe.columns)
    return group_scale(-pe, sector)


def sector_demeaned_quality(ctx) -> pd.DataFrame:
    net_income = ctx.feed_wide("financial", "net_income")
    sector = pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
            "JPM": "financials",
            "BAC": "financials",
        }
    ).reindex(net_income.columns)
    return group_demean(net_income, sector)


def main() -> None:
    price, financial, valuation, _ = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.add_strategy("sector_neutral_momentum", sector_neutral_momentum)
    engine.add_strategy("sector_ranked_value", sector_ranked_value)
    engine.add_strategy("sector_zscore_quality", sector_zscore_quality)
    engine.add_strategy("sector_scaled_value", sector_scaled_value)
    engine.add_strategy("sector_demeaned_quality", sector_demeaned_quality)

    result = engine.run()
    print("combined_frame:", result.combined_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
