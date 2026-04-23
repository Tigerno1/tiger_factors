"""FactorFrame -> CSM research demo.

This example uses Tiger's factor-frame research engine to build a small
synthetic factor_frame, then hands the long output to CSM for ranking,
selection, and backtesting.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation import run_csm_factor_frame_selection_backtest


def _build_research_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=48)
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rng = np.random.default_rng(41)

    price = pd.DataFrame(index=dates, columns=codes, dtype=float)
    price.iloc[0] = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    for i in range(1, len(dates)):
        shock = rng.normal(0.0005, 0.012, size=len(codes))
        price.iloc[i] = price.iloc[i - 1].to_numpy(dtype=float) * (1.0 + shock)

    financial_rows: list[dict[str, object]] = []
    valuation_rows: list[dict[str, object]] = []
    for code_idx, code in enumerate(codes):
        for date in dates[::8]:
            financial_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "net_income": 8.0 + code_idx * 2.0,
                    "total_equity": 40.0 + code_idx * 5.0,
                }
            )
            valuation_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "pe_ratio": 15.0 + code_idx * 1.5,
                }
            )

    macro = pd.DataFrame({"date_": dates, "cpi": np.linspace(3.0, 4.2, len(dates))})
    return price, pd.DataFrame(financial_rows), pd.DataFrame(valuation_rows), macro


def _build_research_engine(
    price: pd.DataFrame,
    financial: pd.DataFrame,
    valuation: pd.DataFrame,
    macro: pd.DataFrame,
) -> FactorResearchEngine:
    engine = FactorResearchEngine(start="2024-01-01", end="2024-03-31", freq="1d")
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.feed_macro(macro, code_column=None)
    engine.add_factor("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(5, fill_method=None))
    engine.add_factor(
        "quality",
        lambda ctx: ctx.feed_wide("financial", "net_income").div(
            ctx.feed_wide("financial", "total_equity").replace(0, np.nan)
        ),
    )
    engine.add_factor("value", lambda ctx: -ctx.feed_wide("valuation", "pe_ratio"))
    return engine


def main() -> None:
    price, financial, valuation, macro = _build_research_inputs()
    research = _build_research_engine(price, financial, valuation, macro)
    result = research.run()

    factor_frame = result.factor_frame
    forward_returns = price.pct_change(5, fill_method=None).shift(-5)
    label_frame = (
        forward_returns.rename_axis(index="date_")
        .reset_index()
        .melt(id_vars="date_", var_name="code", value_name="forward_return")
        .dropna(subset=["forward_return"])
    )
    training_frame = factor_frame.merge(label_frame, on=["date_", "code"], how="inner")
    feature_columns = infer_csm_feature_columns(training_frame)

    model = build_csm_model(
        feature_columns,
        fit_method="listnet",
        feature_transform="zscore",
        min_group_size=3,
        normalize_score_by_date=True,
        learning_rate=0.1,
        max_iter=100,
    )
    model.fit(training_frame)

    score_panel = model.score_panel(training_frame)
    selection_backtest = run_csm_factor_frame_selection_backtest(
        factor_frame.merge(label_frame, on=["date_", "code"], how="inner"),
        (1.0 + forward_returns.fillna(0.0)).cumprod() * 100.0,
        fit_method="listnet",
        feature_transform="zscore",
        min_group_size=3,
        normalize_score_by_date=True,
        learning_rate=0.1,
        max_iter=100,
        top_n=2,
        bottom_n=2,
        long_only=False,
    )

    print("factor_frame:", factor_frame.shape)
    print("training_frame:", training_frame.shape)
    print("feature weights:")
    print(model.weights_)
    print("\nscore panel shape:", score_panel.shape)
    print("\nselection backtest stats:")
    print(pd.DataFrame(selection_backtest.backtest_stats).T.to_string())


if __name__ == "__main__":
    main()
