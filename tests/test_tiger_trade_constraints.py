from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_portfolio import TigerTradeConstraintConfig
from tiger_factors.factor_portfolio import TigerTradeConstraintData
from tiger_factors.factor_portfolio import apply_trade_constraints_to_scores
from tiger_factors.factor_portfolio import apply_trade_constraints_to_weights
from tiger_factors.factor_portfolio import build_tradeable_universe_mask
from tiger_factors.factor_portfolio import summarize_trade_constraints


def _constraint_data() -> TigerTradeConstraintData:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame(
        {
            "AAA": [10.0, 10.5, 11.0],
            "BBB": [4.0, 4.1, 4.2],
            "CCC": [20.0, np.nan, 21.0],
            "DDD": [30.0, 30.0, 30.0],
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "AAA": [2_000_000.0, 2_100_000.0, 2_200_000.0],
            "BBB": [2_000_000.0, 2_100_000.0, 2_200_000.0],
            "CCC": [2_000_000.0, 2_100_000.0, 2_200_000.0],
            "DDD": [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    market_cap = pd.DataFrame(
        {
            "AAA": [2_000_000_000.0] * 3,
            "BBB": [2_000_000_000.0] * 3,
            "CCC": [2_000_000_000.0] * 3,
            "DDD": [2_000_000_000.0] * 3,
        },
        index=dates,
    )
    halted = pd.DataFrame(
        {
            "AAA": [False, False, False],
            "BBB": [False, False, False],
            "CCC": [False, False, False],
            "DDD": [False, False, True],
        },
        index=dates,
    )
    return TigerTradeConstraintData(
        close=close,
        volume=volume,
        market_cap=market_cap,
        shortable={"AAA": True, "BBB": True, "CCC": False, "DDD": True},
        halted=halted,
    )


def test_tradeable_universe_filters_liquidity_price_halts_and_shortability() -> None:
    config = TigerTradeConstraintConfig(
        min_price=5.0,
        min_dollar_volume=1_000_000.0,
        dollar_volume_window=1,
        min_market_cap=100_000_000.0,
        min_eligible_assets=1,
        max_single_name_weight=None,
        max_industry_weight=None,
    )
    data = _constraint_data()

    long_mask = build_tradeable_universe_mask(data, config, side="long")
    short_mask = build_tradeable_universe_mask(data, config, side="short")

    assert bool(long_mask.loc["2024-01-03", "AAA"]) is True
    assert bool(long_mask.loc["2024-01-03", "BBB"]) is False
    assert bool(long_mask.loc["2024-01-02", "CCC"]) is False
    assert bool(long_mask.loc["2024-01-03", "DDD"]) is False
    assert bool(short_mask.loc["2024-01-03", "CCC"]) is False


def test_apply_trade_constraints_to_scores_masks_untradeable_assets() -> None:
    config = TigerTradeConstraintConfig(
        min_price=5.0,
        min_dollar_volume=1_000_000.0,
        dollar_volume_window=1,
        min_eligible_assets=1,
        max_single_name_weight=None,
        max_industry_weight=None,
    )
    scores = pd.DataFrame(
        {"AAA": [1.0], "BBB": [10.0], "CCC": [2.0], "DDD": [3.0]},
        index=pd.DatetimeIndex(["2024-01-03"]),
    )

    result = apply_trade_constraints_to_scores(scores, _constraint_data(), config, side="long")

    assert result.values.loc["2024-01-03", "AAA"] == 1.0
    assert np.isnan(result.values.loc["2024-01-03", "BBB"])
    assert np.isnan(result.values.loc["2024-01-03", "DDD"])
    assert result.summary.loc["2024-01-03", "eligible_count"] == 2


def test_apply_trade_constraints_to_weights_caps_single_name_and_industry() -> None:
    dates = pd.DatetimeIndex(["2024-01-31"])
    close = pd.DataFrame({"AAA": [10.0], "BBB": [20.0], "CCC": [30.0], "DDD": [40.0]}, index=dates)
    volume = pd.DataFrame({code: [2_000_000.0] for code in close.columns}, index=dates)
    data = TigerTradeConstraintData(
        close=close,
        volume=volume,
        industry={"AAA": "tech", "BBB": "tech", "CCC": "health", "DDD": "health"},
    )
    config = TigerTradeConstraintConfig(
        min_price=1.0,
        min_dollar_volume=1_000_000.0,
        dollar_volume_window=1,
        max_single_name_weight=0.40,
        max_industry_weight=0.60,
        min_eligible_assets=1,
    )
    weights = pd.DataFrame({"AAA": [0.70], "BBB": [0.20], "CCC": [0.10], "DDD": [0.0]}, index=dates)

    constrained = apply_trade_constraints_to_weights(weights, data, config)
    row = constrained.iloc[0]

    assert row.abs().max() <= 0.40 + 1e-12
    assert row.loc[["AAA", "BBB"]].abs().sum() <= 0.60 + 1e-12
    assert row.abs().sum() <= 1.0 + 1e-12


def test_summarize_trade_constraints_reports_counts_and_ratios() -> None:
    mask = pd.DataFrame(
        {"AAA": [True, False], "BBB": [True, True]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )

    summary = summarize_trade_constraints(mask)

    assert summary["eligible_count"].tolist() == [2, 1]
    assert summary["eligible_ratio"].tolist() == [1.0, 0.5]
