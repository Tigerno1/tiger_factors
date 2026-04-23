"""Grouped FactorFrame research demo.

This example shows the full handoff from factor construction to factor
evaluation without wrapping the flow in helper functions.
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
from tiger_factors.factor_evaluation.evaluation import SingleFactorEvaluation


dates = pd.bdate_range("2024-01-01", periods=40)
codes = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "BAC"]
group_labels = pd.Series(
    {
        "AAPL": "technology",
        "MSFT": "technology",
        "NVDA": "semis",
        "AMZN": "internet",
        "JPM": "financials",
        "BAC": "financials",
    },
    name="group",
)

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

financial = pd.DataFrame(financial_rows)
valuation = pd.DataFrame(valuation_rows)

output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "tiger_factor_frame_group_research"

engine = FactorFrameEngine()
engine.feed_price(price)
engine.feed_financial(financial)
engine.feed_valuation(valuation)

sector_neutral_momentum = lambda ctx: group_neutralize(  # noqa: E731
    ctx.feed_wide("price", "close").pct_change(10),
    pd.Series(
        {
            "AAPL": "technology",
            "MSFT": "technology",
            "NVDA": "semis",
            "AMZN": "internet",
            "JPM": "financials",
            "BAC": "financials",
        }
    ).reindex(ctx.feed_wide("price", "close").columns),
)

engine.add_strategy("sector_neutral_momentum", sector_neutral_momentum)

result = engine.run()
factor_frame = result.factor_frame[["date_", "code", "sector_neutral_momentum"]].dropna().copy()

research = SingleFactorEvaluation(
    factor_frame=factor_frame,
    price_frame=(
        price.rename_axis("date_")
        .reset_index()
        .melt(id_vars="date_", var_name="code", value_name="close")
        .dropna(subset=["close"])
        .sort_values(["date_", "code"])
        .reset_index(drop=True)
    ),
    factor_column="sector_neutral_momentum",
    group_labels=group_labels,
    group_labels_cache=False,
    group_neutral=True,
    by_group=True,
)
research_result = research.run(include_native_report=False)

print("factor_frame:", result.factor_frame.shape)
print("evaluation_output_dir:", research_result.output_dir)
print("report_dir:", research_result.report.output_dir if research_result.report is not None else None)
print("report_figures:", sorted(research_result.report.figure_paths.keys()) if research_result.report is not None else [])
