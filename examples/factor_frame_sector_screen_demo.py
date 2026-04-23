"""Sector classifier + tech screen + momentum factor demo.

This example shows the Tiger-native three-layer flow on top of the factor
frame engine:

1. feed materialized price and lookup data
2. build a sector classifier
3. derive a technology-only screen from the classifier
4. compute a simple momentum factor
5. inspect the resulting factor frame and intermediate masks
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_frame import FactorResearchEngine


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "BAC"]
    sectors = {
        "AAPL": "technology",
        "MSFT": "technology",
        "NVDA": "semis",
        "AMZN": "internet",
        "JPM": "financials",
        "BAC": "financials",
    }

    rng = np.random.default_rng(7)
    price = pd.DataFrame(index=dates, columns=codes, dtype=float)
    price.iloc[0] = [100.0, 120.0, 140.0, 160.0, 90.0, 85.0]
    for i in range(1, len(dates)):
        shock = rng.normal(0.0008, 0.012, size=len(codes))
        price.iloc[i] = price.iloc[i - 1].to_numpy(dtype=float) * (1.0 + shock)

    company_lookup = pd.DataFrame(
        {
            "code": list(sectors.keys()),
            "sector": list(sectors.values()),
            "industry": [
                "software",
                "software",
                "semiconductors",
                "internet_retail",
                "banks",
                "banks",
            ],
        }
    )
    return price, company_lookup


def main() -> None:
    price_df, company_lookup = _sample_inputs()

    research = FactorResearchEngine(freq="1d", start="2024-01-01", end="2024-02-15")
    research.feed_price(price_df)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")

    research.add_classifier(
        "sector",
        lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"],
    )
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("technology"),
    )
    research.add_factor(
        "momentum",
        lambda ctx: ctx.feed_wide("price", "close").pct_change(10),
    )

    result = research.run()

    print("combined_frame:", result.combined_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print("screen_frames:", sorted(result.screen_frames))
    print("classifier_frames:", sorted(result.classifier_frames))
    print("screen_mask rows:", 0 if result.screen_mask is None else len(result.screen_mask))
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
