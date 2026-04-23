from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tiger_factors.factor_frame import FactorFrameEngine
from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import IndustryNeutralMomentumDefinition
from tiger_factors.factor_frame import FactorFrameFactor
from tiger_factors.factor_frame import FactorFrameExpr
from tiger_factors.factor_frame import FactorFrameTemplate
from tiger_factors.factor_frame import abs
from tiger_factors.factor_frame import corr
from tiger_factors.factor_frame import cov
from tiger_factors.factor_frame import clip_lower
from tiger_factors.factor_frame import clip_upper
from tiger_factors.factor_frame import cs_rank
from tiger_factors.factor_frame import cumprod
from tiger_factors.factor_frame import cumsum
from tiger_factors.factor_frame import diff
from tiger_factors.factor_frame import fillna
from tiger_factors.factor_frame import factor
from tiger_factors.factor_frame import factor_template
from tiger_factors.factor_frame import exp
from tiger_factors.factor_frame import ifelse
from tiger_factors.factor_frame import max
from tiger_factors.factor_frame import mean
from tiger_factors.factor_frame import lag
from tiger_factors.factor_frame import isna
from tiger_factors.factor_frame import log
from tiger_factors.factor_frame import l1_normalize
from tiger_factors.factor_frame import l2_normalize
from tiger_factors.factor_frame import min
from tiger_factors.factor_frame import minmax_scale
from tiger_factors.factor_frame import financial
from tiger_factors.factor_frame import group_demean
from tiger_factors.factor_frame import group_neutralize
from tiger_factors.factor_frame import group_rank
from tiger_factors.factor_frame import group_scale
from tiger_factors.factor_frame import group_zscore
from tiger_factors.factor_frame import neutralize
from tiger_factors.factor_frame import mask
from tiger_factors.factor_frame import notna
from tiger_factors.factor_frame import replace
from tiger_factors.factor_frame import price
from tiger_factors.factor_frame import bottom_n
from tiger_factors.factor_frame import pow
from tiger_factors.factor_frame import rank_desc
from tiger_factors.factor_frame import sign
from tiger_factors.factor_frame import std
from tiger_factors.factor_frame import sqrt
from tiger_factors.factor_frame import rolling_corr
from tiger_factors.factor_frame import rolling_cov
from tiger_factors.factor_frame import rolling_abs
from tiger_factors.factor_frame import rolling_delay
from tiger_factors.factor_frame import rolling_delta
from tiger_factors.factor_frame import rolling_pct_change
from tiger_factors.factor_frame import rolling_prod
from tiger_factors.factor_frame import rolling_sign
from tiger_factors.factor_frame import rolling_median
from tiger_factors.factor_frame import rolling_mean
from tiger_factors.factor_frame import rolling_max
from tiger_factors.factor_frame import rolling_min
from tiger_factors.factor_frame import rolling_kurt
from tiger_factors.factor_frame import rolling_skew
from tiger_factors.factor_frame import rolling_rank
from tiger_factors.factor_frame import rolling_std
from tiger_factors.factor_frame import rolling_var
from tiger_factors.factor_frame import rolling_sum
from tiger_factors.factor_frame import rolling_wma
from tiger_factors.factor_frame import rolling_ema
from tiger_factors.factor_frame import sum
from tiger_factors.factor_frame import ts_zscore
from tiger_factors.factor_frame import ts_abs
from tiger_factors.factor_frame import ts_delay
from tiger_factors.factor_frame import ts_delta
from tiger_factors.factor_frame import ts_pct_change
from tiger_factors.factor_frame import ts_sign
from tiger_factors.factor_frame import ts_median
from tiger_factors.factor_frame import ts_corr
from tiger_factors.factor_frame import ts_beta
from tiger_factors.factor_frame import ts_var
from tiger_factors.factor_frame import ts_skew
from tiger_factors.factor_frame import ts_kurt
from tiger_factors.factor_frame import ts_wma
from tiger_factors.factor_frame import ts_ema
from tiger_factors.factor_frame import ts_prod
from tiger_factors.factor_frame import ewm_mean
from tiger_factors.factor_frame import ewm_std
from tiger_factors.factor_frame import rolling_sharpe
from tiger_factors.factor_frame import rolling_information_ratio
from tiger_factors.factor_frame import cs_scale
from tiger_factors.factor_frame import var
from tiger_factors.factor_frame import winsorize
from tiger_factors.factor_frame import top_n
from tiger_factors.factor_frame import where
from tiger_factors.factor_frame import zscore
from tiger_factors.factor_frame import ts_momentum


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=12)
    codes = ["AAPL", "MSFT", "NVDA"]

    price = pd.DataFrame(index=dates, columns=codes, dtype=float)
    price.iloc[0] = [100.0, 120.0, 140.0]
    for i in range(1, len(dates)):
        price.iloc[i] = price.iloc[i - 1].to_numpy(dtype=float) * np.array([1.01, 0.99, 1.02])

    financial_rows: list[dict[str, object]] = []
    valuation_rows: list[dict[str, object]] = []
    for code_idx, code in enumerate(codes):
        for date in dates[::4]:
            financial_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "net_income": 10.0 + code_idx,
                    "total_equity": 50.0 + 5.0 * code_idx,
                }
            )
            valuation_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "pe_ratio": 15.0 + code_idx,
                }
            )

    macro = pd.DataFrame({"date_": dates, "cpi": np.linspace(3.0, 4.0, len(dates))})
    return price, pd.DataFrame(financial_rows), pd.DataFrame(valuation_rows), macro


def test_factor_frame_engine_builds_combined_and_factor_frames():
    price, financial, valuation, macro = _sample_inputs()

    def momentum(ctx):
        close = ctx.feed_wide("price", "close")
        return close.pct_change(2)

    def quality(ctx):
        net_income = ctx.feed_wide("financial", "net_income")
        total_equity = ctx.feed_wide("financial", "total_equity")
        return net_income / total_equity

    def value(ctx):
        pe_ratio = ctx.feed_wide("valuation", "pe_ratio")
        return -pe_ratio

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.feed_macro(macro, code_column=None)
    engine.add_strategy("momentum", momentum)
    engine.add_strategy("quality", quality)
    engine.add_strategy("value", value)

    result = engine.run()

    assert {"date_", "code", "momentum", "quality", "value"}.issubset(result.factor_frame.columns)
    assert {"date_", "code", "price__close", "financial__net_income", "valuation__pe_ratio", "macro__cpi"}.issubset(
        result.combined_frame.columns
    )
    assert result.factor_frame["code"].nunique() == 3
    assert result.factor_frame["date_"].nunique() > 0
    assert result.output_dir is None


def test_factor_frame_engine_can_save_results(tmp_path):
    price, financial, valuation, macro = _sample_inputs()

    def momentum(ctx):
        return ctx.feed_wide("price", "close").pct_change(2)

    engine = FactorFrameEngine(output_root_dir=tmp_path, save=True)
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.feed_macro(macro, code_column=None)
    engine.add_strategy("momentum", momentum)

    result = engine.run()

    assert result.output_dir == tmp_path
    assert (tmp_path / "factor_frame.parquet").exists()
    assert (tmp_path / "combined_frame.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    assert "factor_frame" in result.saved_paths


def test_factor_frame_engine_lags_financial_and_valuation_by_one_session():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0, 103.0]}, index=dates)
    financial = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "net_income": 10.0, "total_equity": 50.0},
        ]
    )
    valuation = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "pe_ratio": 20.0},
        ]
    )
    macro = pd.DataFrame({"date_": dates, "cpi": [3.0, 3.1, 3.2, 3.3]})

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.feed_macro(macro, code_column=None)
    engine.add_strategy("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    result = engine.run()
    frame = result.combined_frame[result.combined_frame["code"] == "AAPL"].set_index("date_")

    assert pd.isna(frame.loc[dates[0], "financial__net_income"])
    assert frame.loc[dates[1], "financial__net_income"] == 10.0
    assert frame.loc[dates[2], "financial__net_income"] == 10.0
    assert pd.isna(frame.loc[dates[0], "valuation__pe_ratio"])
    assert frame.loc[dates[1], "valuation__pe_ratio"] == 20.0
    assert frame.loc[dates[2], "valuation__pe_ratio"] == 20.0


def test_factor_frame_engine_feed_fill_method_ffill_applies_to_price():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price = pd.DataFrame(
        {
            "AAPL": [100.0, np.nan, 102.0],
            "MSFT": [200.0, 201.0, 202.0],
        },
        index=dates,
    )

    engine = FactorFrameEngine()
    engine.feed_price(price, fill_method="ffill")
    engine.add_factor("price_wide", lambda ctx: ctx.feed_wide("price", "close"))

    result = engine.run()
    frame = result.factor_frame[result.factor_frame["code"] == "AAPL"].set_index("date_")

    assert frame.loc[dates[0], "price_wide"] == 100.0
    assert frame.loc[dates[1], "price_wide"] == 100.0
    assert frame.loc[dates[2], "price_wide"] == 102.0


def test_factor_frame_engine_fill_method_does_not_cross_fill_columns():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "open": 10.0, "close": 11.0},
            {"date_": dates[1], "code": "AAPL", "open": np.nan, "close": 12.0},
            {"date_": dates[2], "code": "AAPL", "open": 13.0, "close": np.nan},
        ]
    )

    engine = FactorFrameEngine()
    engine.feed("price", price, align_mode="code_date", fill_method="ffill")
    result = engine.run()
    frame = result.combined_frame[result.combined_frame["code"] == "AAPL"].set_index("date_")

    assert frame.loc[dates[1], "price__open"] == 10.0
    assert frame.loc[dates[1], "price__close"] == 12.0
    assert frame.loc[dates[2], "price__open"] == 13.0
    assert frame.loc[dates[2], "price__close"] == 12.0


def test_factor_frame_engine_feed_fill_method_ffill_applies_to_financial_with_lag():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=dates)
    financial = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "net_income": 10.0, "total_equity": 50.0},
        ]
    )

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed_financial(financial, lag_sessions=1)
    engine.add_factor("financial_wide", lambda ctx: ctx.feed_wide("financial", "net_income"))

    result = engine.run()
    frame = result.combined_frame[result.combined_frame["code"] == "AAPL"].set_index("date_")

    assert pd.isna(frame.loc[dates[0], "financial__net_income"])
    assert frame.loc[dates[1], "financial__net_income"] == 10.0
    assert frame.loc[dates[2], "financial__net_income"] == 10.0


def test_factor_frame_engine_feed_asof_aligns_to_latest_prior_value():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
    financial = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "net_income": 10.0, "total_equity": 50.0},
            {"date_": dates[3], "code": "AAPL", "net_income": 20.0, "total_equity": 60.0},
        ]
    )

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed("financial", financial, align_mode="asof")
    engine.add_factor("financial_wide", lambda ctx: ctx.feed_wide("financial", "net_income"))

    result = engine.run()
    frame = result.combined_frame[result.combined_frame["code"] == "AAPL"].set_index("date_")

    assert frame.loc[dates[0], "financial__net_income"] == 10.0
    assert frame.loc[dates[1], "financial__net_income"] == 10.0
    assert frame.loc[dates[2], "financial__net_income"] == 10.0
    assert frame.loc[dates[3], "financial__net_income"] == 20.0
    assert frame.loc[dates[4], "financial__net_income"] == 20.0


def test_factor_frame_engine_can_use_calendar_false_for_calendar_day_lag():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    price = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)
    financial = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "net_income": 10.0, "total_equity": 50.0},
        ]
    )

    engine = FactorFrameEngine(bday_lag=False)
    engine.feed_price(price)
    engine.feed_financial(financial, lag_sessions=1)
    engine.add_strategy("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    result = engine.run()
    frame = result.combined_frame[result.combined_frame["code"] == "AAPL"].set_index("date_")

    assert pd.isna(frame.loc[pd.Timestamp("2024-01-05"), "financial__net_income"])
    assert frame.loc[pd.Timestamp("2024-01-06"), "financial__net_income"] == 10.0


def test_factor_frame_engine_build_context_uses_availability_column():
    dates = pd.bdate_range("2024-01-01", periods=3)
    financial = pd.DataFrame(
        [
            {
                "date_": dates[0],
                "available_at": dates[1],
                "code": "AAPL",
                "net_income": 10.0,
                "total_equity": 50.0,
            }
        ]
    )

    engine = FactorFrameEngine(use_point_in_time=True, availability_column="available_at")
    engine.feed_financial(financial, lag_sessions=0)

    context = engine.build_context()
    feed_frame = context.feed_frame("financial")

    assert context.use_point_in_time is True
    assert context.availability_column == "available_at"
    assert context.feed("financial").align_mode == "code_date"
    assert feed_frame.loc[0, "date_"] == pd.Timestamp("2024-01-02")
    assert feed_frame.loc[0, "source_date_"] == pd.Timestamp("2024-01-01")


def test_factor_frame_engine_supports_code_only_feeds_and_broadcasts_them():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]}, index=dates)
    companies = pd.DataFrame({"code": ["AAPL", "MSFT"], "sector": ["tech", "tech"]})

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.feed("companies", companies, align_mode="code")
    engine.add_factor("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    result = engine.run()
    context = engine.build_context()
    companies_feed = context.feed("companies")
    frame = result.combined_frame[result.combined_frame["code"] == "AAPL"].set_index("date_")

    assert companies_feed.align_mode == "code"
    assert "companies__sector" in result.combined_frame.columns
    assert frame.loc[dates[0], "companies__sector"] == "tech"
    assert frame.loc[dates[-1], "companies__sector"] == "tech"
    assert "momentum" in result.factor_frame.columns


def test_factor_frame_engine_supports_pure_code_only_feeds():
    companies = pd.DataFrame({"code": ["AAPL", "MSFT"], "sector": ["tech", "tech"]})

    engine = FactorFrameEngine()
    engine.feed("companies", companies, align_mode="code")

    result = engine.run()

    assert "date_" not in result.combined_frame.columns
    assert set(result.combined_frame["code"]) == {"AAPL", "MSFT"}
    assert "companies__sector" in result.combined_frame.columns


def test_factor_frame_engine_add_factor_is_alias_for_add_strategy():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=dates)

    engine = FactorFrameEngine()
    engine.feed_price(price)
    engine.add_factor("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    result = engine.run()
    assert "momentum" in result.factor_frame.columns


def test_factor_frame_engine_applies_engine_window_to_all_feeds():
    price, financial, valuation, macro = _sample_inputs()

    engine = FactorFrameEngine(start="2024-01-03", end="2024-01-08")
    engine.feed_price(price)
    engine.feed_financial(financial)
    engine.feed_valuation(valuation)
    engine.feed_macro(macro, code_column=None)
    engine.add_factor("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    context = engine.build_context()
    result = engine.run()

    assert context.start == pd.Timestamp("2024-01-03")
    assert context.end == pd.Timestamp("2024-01-08")
    assert context.feed("price").frame["date_"].min() >= pd.Timestamp("2024-01-03")
    assert context.feed("price").frame["date_"].max() <= pd.Timestamp("2024-01-08")
    assert result.combined_frame["date_"].min() >= pd.Timestamp("2024-01-03")
    assert result.combined_frame["date_"].max() <= pd.Timestamp("2024-01-08")


def test_factor_frame_template_can_build_multiple_parameterized_factors():
    price_frame, financial_frame, _, _ = _sample_inputs()

    momentum_template = factor_template(
        "momentum_template",
        lambda window=1: price(value_column="close").pct_change(window).cs_rank(),
        defaults={"window": 1},
    )
    quality_template = factor_template(
        "quality_template",
        lambda window=2: financial(value_column="net_income").rolling_mean(window),
        defaults={"window": 2},
    )

    engine = FactorFrameEngine(start="2024-01-01", end="2024-01-12")
    engine.feed_price(price_frame)
    engine.feed_financial(financial_frame, lag_sessions=0)
    engine.add_factors(
        momentum_template(factor_name="momentum_2", window=2),
        momentum_template(factor_name="momentum_4", window=4),
        quality_template(factor_name="quality_2", window=2),
    )

    result = engine.run()

    assert isinstance(momentum_template, FactorFrameTemplate)
    assert "momentum_2" in result.factor_frame.columns
    assert "momentum_4" in result.factor_frame.columns
    assert "quality_2" in result.factor_frame.columns


def test_factor_research_engine_facade_supports_templates_and_windowing():
    price_frame, financial_frame, valuation_frame, macro_frame = _sample_inputs()

    momentum_template = factor_template(
        "momentum_template",
        lambda window=1: price(value_column="close").pct_change(window).cs_rank(),
        defaults={"window": 1},
    )
    quality_template = factor_template(
        "quality_template",
        lambda window=2: financial(value_column="net_income").rolling_mean(window),
        defaults={"window": 2},
    )

    research = FactorResearchEngine(start="2024-01-03", end="2024-01-08")
    research.feed_price(price_frame)
    research.feed_financial(financial_frame, lag_sessions=0)
    research.feed_valuation(valuation_frame, lag_sessions=0)
    research.feed_macro(macro_frame, code_column=None)
    research.add_template(momentum_template, factor_name="momentum_2", window=2)
    research.add_template(quality_template, factor_name="quality_2", window=2)
    research.add_strategy("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    result = research.run()
    context = research.build_context()

    assert isinstance(research, FactorResearchEngine)
    assert context.start == pd.Timestamp("2024-01-03")
    assert context.end == pd.Timestamp("2024-01-08")
    assert "momentum_2" in result.factor_frame.columns
    assert "quality_2" in result.factor_frame.columns
    assert "momentum" in result.factor_frame.columns


def test_factor_frame_engine_intersection_align_mode_restricts_common_dates_and_codes():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0, 103.0], "MSFT": [200.0, 201.0, 202.0, 203.0]}, index=dates)
    valuation = pd.DataFrame(
        [
            {"date_": dates[0], "code": "AAPL", "pe_ratio": 20.0},
            {"date_": dates[1], "code": "AAPL", "pe_ratio": 21.0},
            {"date_": dates[1], "code": "MSFT", "pe_ratio": 22.0},
            {"date_": dates[2], "code": "MSFT", "pe_ratio": 23.0},
        ]
    )

    engine = FactorFrameEngine(align_mode="intersection")
    engine.feed_price(price)
    engine.feed_valuation(valuation, lag_sessions=0)
    engine.add_factor("momentum", lambda ctx: ctx.feed_wide("price", "close").pct_change(1))

    context = engine.build_context()
    combined = context.combined_frame

    assert set(combined["code"].dropna().astype(str).unique()) == {"AAPL", "MSFT"}
    assert combined["date_"].dropna().nunique() <= 3


def test_factor_frame_components_can_be_reused_directly():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_frame = pd.DataFrame({"AAPL": [100.0, 102.0, 101.0, 105.0], "MSFT": [200.0, 198.0, 202.0, 203.0]}, index=dates)

    momentum_component = factor(
        "momentum",
        lambda ctx: cs_rank(ts_momentum(price(ctx), window=1)),
    )

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(momentum_component)

    result = engine.run()

    assert isinstance(momentum_component, FactorFrameFactor)
    assert "momentum" in result.factor_frame.columns


def test_factor_frame_dsl_supports_fluent_chaining():
    price_frame, financial_frame, _, _ = _sample_inputs()

    momentum_expr = price(value_column="close").pct_change(1).cs_rank()
    quality_expr = financial(value_column="net_income").rolling_mean(2)

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.feed_financial(financial_frame, lag_sessions=0)
    engine.add_factor(factor("dsl_momentum", momentum_expr))
    engine.add_factor(factor("dsl_quality", quality_expr))

    result = engine.run()

    assert isinstance(momentum_expr, FactorFrameExpr)
    assert isinstance(quality_expr, FactorFrameExpr)
    assert "dsl_momentum" in result.factor_frame.columns
    assert "dsl_quality" in result.factor_frame.columns


def test_factor_frame_dsl_supports_common_combinators():
    price_frame, _, _, _ = _sample_inputs()
    groups = pd.Series({"AAPL": "tech", "MSFT": "tech", "NVDA": "semis"})

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(
        factor(
            "dsl_combo",
            price(value_column="close")
            .lag(1)
            .winsorize(lower=0.05, upper=0.95)
            .zscore()
            .demean(),
        )
    )
    engine.add_factor(
        factor(
            "dsl_neutralized",
            price(value_column="close").cs_rank().neutralize(factor("groups", lambda ctx: groups)),
        )
    )

    result = engine.run()

    assert isinstance(lag(price_frame), pd.DataFrame)
    assert isinstance(winsorize(price_frame), pd.DataFrame)
    assert isinstance(zscore(price_frame), pd.DataFrame)
    assert isinstance(neutralize(price_frame, groups), pd.DataFrame)
    assert isinstance(group_neutralize(price_frame, groups), pd.DataFrame)
    assert isinstance(group_demean(price_frame, groups), pd.DataFrame)
    assert isinstance(group_rank(price_frame, groups), pd.DataFrame)
    assert isinstance(group_zscore(price_frame, groups), pd.DataFrame)
    assert isinstance(group_scale(price_frame, groups), pd.DataFrame)
    assert "dsl_combo" in result.factor_frame.columns
    assert "dsl_neutralized" in result.factor_frame.columns


def test_factor_frame_dsl_supports_rolling_and_normalization_helpers():
    price_frame, _, _, _ = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(
        factor(
            "dsl_rolling_minmax",
            price(value_column="close").rolling_min(2).l1_normalize().minmax_scale(feature_range=(-1.0, 1.0)),
        )
    )
    engine.add_factor(
        factor(
            "dsl_rolling_corr",
            price(value_column="close").rolling_corr(price(value_column="close").shift(1), window=2).l2_normalize(),
        )
    )

    result = engine.run()

    assert isinstance(rolling_min(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_max(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_corr(price_frame, price_frame.shift(1), 2), pd.DataFrame)
    assert isinstance(rolling_mean(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_std(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_sum(price_frame, 2), pd.DataFrame)
    assert isinstance(minmax_scale(price_frame), pd.DataFrame)
    assert isinstance(l1_normalize(price_frame), pd.DataFrame)
    assert isinstance(l2_normalize(price_frame), pd.DataFrame)
    assert "dsl_rolling_minmax" in result.factor_frame.columns
    assert "dsl_rolling_corr" in result.factor_frame.columns


def test_factor_frame_dsl_supports_short_rolling_aliases_and_ifelse():
    price_frame, _, _, _ = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    condition_expr = price(value_column="close") > price(value_column="close").shift(1)
    engine.add_factor(factor("dsl_mean", price(value_column="close").mean(2)))
    engine.add_factor(factor("dsl_sum", price(value_column="close").sum(2)))
    engine.add_factor(factor("dsl_std", price(value_column="close").std(2)))
    engine.add_factor(factor("dsl_var", price(value_column="close").var(2)))
    engine.add_factor(factor("dsl_min", price(value_column="close").min(2)))
    engine.add_factor(factor("dsl_max", price(value_column="close").max(2)))
    engine.add_factor(factor("dsl_corr", price(value_column="close").corr(price(value_column="close").shift(1), 2)))
    engine.add_factor(factor("dsl_cov", price(value_column="close").cov(price(value_column="close").shift(1), 2)))
    engine.add_factor(factor("dsl_rolling_rank", price(value_column="close").rolling_rank(2)))
    engine.add_factor(factor("dsl_rolling_skew", price(value_column="close").rolling_skew(2)))
    engine.add_factor(factor("dsl_rolling_kurt", price(value_column="close").rolling_kurt(2)))
    engine.add_factor(factor("dsl_rolling_median", price(value_column="close").rolling_median(2)))
    engine.add_factor(factor("dsl_rolling_var", price(value_column="close").rolling_var(2)))
    engine.add_factor(factor("dsl_ts_zscore", price(value_column="close").ts_zscore(2)))
    engine.add_factor(factor("dsl_ts_median", price(value_column="close").ts_median(2)))
    engine.add_factor(factor("dsl_ts_corr", price(value_column="close").ts_corr(price(value_column="close").shift(1), 2)))
    engine.add_factor(factor("dsl_ts_beta", price(value_column="close").ts_beta(price(value_column="close").shift(1), 2)))
    engine.add_factor(factor("dsl_ts_var", price(value_column="close").ts_var(2)))
    engine.add_factor(factor("dsl_ts_skew", price(value_column="close").ts_skew(2)))
    engine.add_factor(factor("dsl_ts_kurt", price(value_column="close").ts_kurt(2)))
    engine.add_factor(factor("dsl_ewm_mean", price(value_column="close").ewm_mean(span=2)))
    engine.add_factor(factor("dsl_ewm_std", price(value_column="close").ewm_std(span=2)))
    engine.add_factor(factor("dsl_rolling_sharpe", price(value_column="close").rolling_sharpe(2)))
    engine.add_factor(factor("dsl_rolling_ir", price(value_column="close").rolling_ir(2)))
    engine.add_factor(factor("dsl_rolling_abs", price(value_column="close").rolling_abs()))
    engine.add_factor(factor("dsl_rolling_sign", price(value_column="close").rolling_sign()))
    engine.add_factor(factor("dsl_rolling_wma", price(value_column="close").rolling_wma(2)))
    engine.add_factor(factor("dsl_rolling_ema", price(value_column="close").rolling_ema(2)))
    engine.add_factor(factor("dsl_rolling_delay", price(value_column="close").rolling_delay(1)))
    engine.add_factor(factor("dsl_rolling_delta", price(value_column="close").rolling_delta(1)))
    engine.add_factor(factor("dsl_rolling_pct_change", price(value_column="close").rolling_pct_change(1)))
    engine.add_factor(factor("dsl_rolling_prod", price(value_column="close").rolling_prod(2)))
    engine.add_factor(factor("dsl_cs_scale", price(value_column="close").cs_scale(feature_range=(-1.0, 1.0))))
    engine.add_factor(
        factor(
            "dsl_ifelse",
            condition_expr.ifelse(true_value=price(value_column="close"), false_value=0.0),
        )
    )

    result = engine.run()

    assert isinstance(mean(price_frame, 2), pd.DataFrame)
    assert isinstance(sum(price_frame, 2), pd.DataFrame)
    assert isinstance(std(price_frame, 2), pd.DataFrame)
    assert isinstance(var(price_frame, 2), pd.DataFrame)
    assert isinstance(min(price_frame, 2), pd.DataFrame)
    assert isinstance(max(price_frame, 2), pd.DataFrame)
    assert isinstance(corr(price_frame, price_frame.shift(1), 2), pd.DataFrame)
    assert isinstance(cov(price_frame, price_frame.shift(1), 2), pd.DataFrame)
    assert isinstance(rolling_rank(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_skew(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_kurt(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_median(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_var(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_zscore(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_median(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_corr(price_frame, price_frame.shift(1), 2), pd.DataFrame)
    assert isinstance(ts_beta(price_frame, price_frame.shift(1), 2), pd.DataFrame)
    assert isinstance(ts_var(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_skew(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_kurt(price_frame, 2), pd.DataFrame)
    assert isinstance(ewm_mean(price_frame, span=2), pd.DataFrame)
    assert isinstance(ewm_std(price_frame, span=2), pd.DataFrame)
    assert isinstance(rolling_sharpe(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_information_ratio(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_abs(price_frame), pd.DataFrame)
    assert isinstance(rolling_sign(price_frame), pd.DataFrame)
    assert isinstance(rolling_wma(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_ema(price_frame, 2), pd.DataFrame)
    assert isinstance(rolling_delay(price_frame, 1), pd.DataFrame)
    assert isinstance(rolling_delta(price_frame, 1), pd.DataFrame)
    assert isinstance(rolling_pct_change(price_frame, 1), pd.DataFrame)
    assert isinstance(rolling_prod(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_abs(price_frame), pd.DataFrame)
    assert isinstance(ts_sign(price_frame), pd.DataFrame)
    assert isinstance(ts_wma(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_ema(price_frame, 2), pd.DataFrame)
    assert isinstance(ts_delay(price_frame, 1), pd.DataFrame)
    assert isinstance(ts_delta(price_frame, 1), pd.DataFrame)
    assert isinstance(ts_pct_change(price_frame, 1), pd.DataFrame)
    assert isinstance(ts_prod(price_frame, 2), pd.DataFrame)
    assert isinstance(cs_scale(price_frame, feature_range=(-1.0, 1.0)), pd.DataFrame)
    assert isinstance(ifelse(price_frame > price_frame.shift(1), true_value=price_frame, false_value=0.0), pd.DataFrame)
    assert "dsl_mean" in result.factor_frame.columns
    assert "dsl_sum" in result.factor_frame.columns
    assert "dsl_std" in result.factor_frame.columns
    assert "dsl_var" in result.factor_frame.columns
    assert "dsl_min" in result.factor_frame.columns
    assert "dsl_max" in result.factor_frame.columns
    assert "dsl_corr" in result.factor_frame.columns
    assert "dsl_cov" in result.factor_frame.columns
    assert "dsl_rolling_rank" in result.factor_frame.columns
    assert "dsl_rolling_skew" in result.factor_frame.columns
    assert "dsl_rolling_kurt" in result.factor_frame.columns
    assert "dsl_rolling_median" in result.factor_frame.columns
    assert "dsl_rolling_var" in result.factor_frame.columns
    assert "dsl_ts_zscore" in result.factor_frame.columns
    assert "dsl_ts_median" in result.factor_frame.columns
    assert "dsl_ts_corr" in result.factor_frame.columns
    assert "dsl_ts_beta" in result.factor_frame.columns
    assert "dsl_ts_var" in result.factor_frame.columns
    assert "dsl_ts_skew" in result.factor_frame.columns
    assert "dsl_ts_kurt" in result.factor_frame.columns
    assert "dsl_ewm_mean" in result.factor_frame.columns
    assert "dsl_ewm_std" in result.factor_frame.columns
    assert "dsl_rolling_sharpe" in result.factor_frame.columns
    assert "dsl_rolling_ir" in result.factor_frame.columns
    assert "dsl_rolling_abs" in result.factor_frame.columns
    assert "dsl_rolling_sign" in result.factor_frame.columns
    assert "dsl_rolling_wma" in result.factor_frame.columns
    assert "dsl_rolling_ema" in result.factor_frame.columns
    assert "dsl_rolling_delay" in result.factor_frame.columns
    assert "dsl_rolling_delta" in result.factor_frame.columns
    assert "dsl_rolling_pct_change" in result.factor_frame.columns
    assert "dsl_rolling_prod" in result.factor_frame.columns
    assert "dsl_cs_scale" in result.factor_frame.columns
    assert "dsl_ifelse" in result.factor_frame.columns


def test_factor_frame_dsl_supports_basic_math_helpers():
    price_frame, _, _, _ = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(
        factor(
            "dsl_math",
            price(value_column="close").abs().log().exp().sqrt().sign(),
        )
    )
    engine.add_factor(
        factor(
            "dsl_mask",
            price(value_column="close").mask(price(value_column="close") < price(value_column="close").shift(1)),
        )
    )

    result = engine.run()

    assert isinstance(abs(price_frame), pd.DataFrame)
    assert isinstance(log(price_frame.clip(lower=1)), pd.DataFrame)
    assert isinstance(exp(price_frame), pd.DataFrame)
    assert isinstance(sqrt(price_frame.abs()), pd.DataFrame)
    assert isinstance(sign(price_frame), pd.DataFrame)
    assert isinstance(mask(price_frame, price_frame < price_frame.shift(1)), pd.DataFrame)
    assert "dsl_math" in result.factor_frame.columns
    assert "dsl_mask" in result.factor_frame.columns


def test_factor_frame_dsl_supports_diff_and_cleaning_helpers():
    price_frame, _, _, _ = _sample_inputs()
    dirty = price_frame.copy()
    dirty.iloc[0, 0] = np.nan

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(
        factor(
            "dsl_diff",
            price(value_column="close").shift(1).diff(1).clip_lower(-1.0).clip_upper(1.0),
        )
    )
    engine.add_factor(
        factor(
            "dsl_replace",
            price(value_column="close").replace({100.0: 101.0}).fillna(method="ffill"),
        )
    )
    engine.add_factor(
        factor(
            "dsl_isna",
            price(value_column="close").isna(),
        )
    )
    engine.add_factor(
        factor(
            "dsl_notna",
            price(value_column="close").notna(),
        )
    )

    result = engine.run()

    assert isinstance(clip_lower(price_frame, -1.0), pd.DataFrame)
    assert isinstance(clip_upper(price_frame, 1.0), pd.DataFrame)
    assert isinstance(replace(dirty, 100.0, 0.0), pd.DataFrame)
    assert isinstance(isna(dirty), pd.DataFrame)
    assert isinstance(notna(dirty), pd.DataFrame)
    assert "dsl_diff" in result.factor_frame.columns
    assert "dsl_replace" in result.factor_frame.columns
    assert "dsl_isna" in result.factor_frame.columns
    assert "dsl_notna" in result.factor_frame.columns


def test_factor_frame_dsl_supports_power_and_cumulative_helpers():
    price_frame, _, _, _ = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(
        factor(
            "dsl_power",
            price(value_column="close").pow(2).cumsum().cumprod(),
        )
    )
    engine.add_factor(
        factor(
            "dsl_rank_desc",
            price(value_column="close").rank_desc(axis=1),
        )
    )

    result = engine.run()

    assert isinstance(pow(price_frame, 2), pd.DataFrame)
    assert isinstance(cumsum(price_frame), pd.DataFrame)
    assert isinstance(cumprod(price_frame), pd.DataFrame)
    assert isinstance(rank_desc(price_frame, axis=1), pd.DataFrame)
    assert "dsl_power" in result.factor_frame.columns
    assert "dsl_rank_desc" in result.factor_frame.columns


def test_factor_frame_dsl_supports_where_fillna_and_top_bottom_helpers():
    price_frame, _, _, _ = _sample_inputs()

    engine = FactorFrameEngine()
    engine.feed_price(price_frame)
    engine.add_factor(
        factor(
            "dsl_where",
            price(value_column="close").where(price(value_column="close") > price(value_column="close").shift(1)),
        )
    )
    engine.add_factor(
        factor(
            "dsl_fillna",
            price(value_column="close").shift(1).fillna(method="bfill"),
        )
    )
    engine.add_factor(
        factor("dsl_top", price(value_column="close").top_n(1))
    )
    engine.add_factor(
        factor("dsl_bottom", price(value_column="close").bottom_n(1))
    )

    result = engine.run()

    assert isinstance(fillna(price_frame.shift(1), method="bfill"), pd.DataFrame)
    assert isinstance(where(price_frame, price_frame > price_frame.shift(1)), pd.DataFrame)
    assert isinstance(top_n(price_frame, 1), pd.DataFrame)
    assert isinstance(bottom_n(price_frame, 1), pd.DataFrame)
    assert "dsl_where" in result.factor_frame.columns
    assert "dsl_fillna" in result.factor_frame.columns
    assert "dsl_top" in result.factor_frame.columns
    assert "dsl_bottom" in result.factor_frame.columns


def test_factor_research_engine_supports_screens_and_classifiers():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price_frame = pd.DataFrame({"AAPL": [100.0, 102.0, 103.0], "MSFT": [50.0, 60.0, 70.0]}, index=dates)
    company_lookup = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "sector": ["technology", "energy"],
            "industry": ["software", "oil"],
        }
    )

    research = FactorResearchEngine(start="2024-01-01", end="2024-01-05")
    research.feed_price(price_frame)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.add_classifier("sector", lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"])
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("technology"),
    )
    research.add_strategy("close_level", lambda ctx: ctx.feed_wide("price", "close"))

    result = research.run()

    assert "sector" in result.classifier_frames
    assert "tech_only" in result.screen_frames
    assert result.screen_mask is not None
    assert set(result.factor_frame["code"].dropna().unique()) == {"AAPL"}
    assert result.factor_frame["close_level"].min() > 0


@pytest.mark.parametrize("freq", ["1min", "15min", "20min", "30min", "1h", "2h"])
def test_factor_research_engine_uses_explicit_freq_for_intraday_alignment(freq):
    times = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:45:00",
            "2024-01-02 11:00:00",
        ]
    )
    price_frame = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0, 202.0],
        },
        index=times,
    )

    research = FactorResearchEngine(freq=freq, start=times[0], end=times[-1])
    research.feed_price(price_frame)
    research.add_factor("close_level", lambda ctx: ctx.feed_wide("price", "close"))

    context = research.build_context()
    result = research.run()

    assert context.freq == freq
    assert context.time_kind == "intraday"
    assert list(pd.to_datetime(context.combined_frame["date_"]).drop_duplicates()) == list(times)
    assert list(pd.to_datetime(result.combined_frame["date_"]).drop_duplicates()) == list(times)
    assert list(pd.to_datetime(result.factor_frame["date_"]).drop_duplicates()) == list(times)


def test_factor_research_engine_can_shift_intraday_left_labels_to_right_labels():
    times = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:45:00",
            "2024-01-02 10:00:00",
        ]
    )
    price_frame = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0, 202.0],
        },
        index=times,
    )

    research = FactorResearchEngine(freq="15min", label_side="left", start=times[0], end=times[-1] + pd.Timedelta(minutes=15))
    research.feed_price(price_frame)
    research.add_factor("close_level", lambda ctx: ctx.feed_wide("price", "close"))

    context = research.build_context()
    result = research.run()
    shifted = [ts + pd.Timedelta(minutes=15) for ts in times]

    assert context.label_side == "left"
    assert list(pd.to_datetime(context.combined_frame["date_"]).drop_duplicates()) == shifted
    assert list(pd.to_datetime(result.combined_frame["date_"]).drop_duplicates()) == shifted
    assert list(pd.to_datetime(result.factor_frame["date_"]).drop_duplicates()) == shifted


def test_factor_research_engine_can_auto_detect_intraday_label_side_with_calendar():
    times = pd.to_datetime(
        [
            "2024-01-02 14:30:00+00:00",
            "2024-01-02 14:45:00+00:00",
            "2024-01-02 15:00:00+00:00",
        ]
    )
    price_frame = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0],
            "MSFT": [200.0, 201.0, 202.0],
        },
        index=times,
    )

    research = FactorResearchEngine(
        freq="15min",
        calendar="XNYS",
        label_side="auto",
        start=times[0],
        end=times[-1] + pd.Timedelta(minutes=15),
    )
    research.feed_price(price_frame)
    research.add_factor("close_level", lambda ctx: ctx.feed_wide("price", "close"))

    context = research.build_context()
    result = research.run()
    shifted = [pd.Timestamp(ts).tz_convert(None) + pd.Timedelta(minutes=15) for ts in times]

    assert context.label_side == "auto"
    assert list(pd.to_datetime(context.combined_frame["date_"]).drop_duplicates()) == shifted
    assert list(pd.to_datetime(result.combined_frame["date_"]).drop_duplicates()) == shifted
    assert list(pd.to_datetime(result.factor_frame["date_"]).drop_duplicates()) == shifted


def test_factor_research_engine_screens_before_factor_computation():
    dates = pd.bdate_range("2024-01-01", periods=3)
    price_frame = pd.DataFrame({"AAPL": [100.0, 102.0, 103.0], "MSFT": [50.0, 60.0, 70.0]}, index=dates)
    company_lookup = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "sector": ["technology", "energy"],
            "industry": ["software", "oil"],
        }
    )

    research = FactorResearchEngine(start="2024-01-01", end="2024-01-05")
    research.feed_price(price_frame)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.add_classifier("sector", lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"])
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("technology"),
    )
    research.add_strategy(
        "screened_universe_size",
        lambda ctx: pd.DataFrame(
            np.full(ctx.feed_wide("price", "close").shape, float(ctx.feed_wide("price", "close").shape[1])),
            index=ctx.feed_wide("price", "close").index,
            columns=ctx.feed_wide("price", "close").columns,
        ),
    )

    result = research.run()

    assert set(result.factor_frame["code"].dropna().unique()) == {"AAPL"}
    assert set(result.factor_frame["screened_universe_size"].dropna().unique()) == {1.0}


def test_factor_research_engine_supports_factor_definitions():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_frame = pd.DataFrame({"AAPL": [100.0, 102.0, 104.0, 106.0], "MSFT": [50.0, 49.0, 48.0, 47.0]}, index=dates)
    company_lookup = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "sector": ["Technology", "Energy"],
            "industry": ["Software", "Oil"],
        }
    )

    research = FactorResearchEngine(start="2024-01-01", end="2024-01-08")
    research.feed_price(price_frame)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.add_classifier("sector", lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"])
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("Technology"),
    )
    research.add_definition(
        IndustryNeutralMomentumDefinition(
            name="industry_neutral_momentum",
            window=1,
            price_feed="price",
            price_column="close",
            classifier_name="sector",
            classifier_column="sector",
        )
    )

    result = research.run()

    assert "industry_neutral_momentum" in result.factor_frame.columns
    assert set(result.factor_frame["code"].dropna().unique()) == {"AAPL"}


def test_factor_research_engine_supports_definition_registry_lookup():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_frame = pd.DataFrame({"AAPL": [100.0, 102.0, 104.0, 106.0], "MSFT": [50.0, 49.0, 48.0, 47.0]}, index=dates)
    company_lookup = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "sector": ["Technology", "Energy"],
            "industry": ["Software", "Oil"],
        }
    )

    registry = FactorDefinitionRegistry()
    registry.register(
        IndustryNeutralMomentumDefinition(
            name="industry_neutral_momentum",
            window=1,
            price_feed="price",
            price_column="close",
            classifier_name="sector",
            classifier_column="sector",
        )
    )

    research = FactorResearchEngine(start="2024-01-01", end="2024-01-08", definition_registry=registry)
    research.feed_price(price_frame)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.add_classifier("sector", lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"])
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("Technology"),
    )
    research.add_definition("industry_neutral_momentum")

    result = research.run()

    assert "industry_neutral_momentum" in result.factor_frame.columns
    assert set(result.factor_frame["code"].dropna().unique()) == {"AAPL"}


def test_factor_definition_registry_register_many():
    registry = FactorDefinitionRegistry()
    registry.register_many(
        IndustryNeutralMomentumDefinition(
            name="momentum_one",
            window=1,
            price_feed="price",
            price_column="close",
            classifier_name="sector",
            classifier_column="sector",
        ),
        IndustryNeutralMomentumDefinition(
            name="momentum_two",
            window=2,
            price_feed="price",
            price_column="close",
            classifier_name="sector",
            classifier_column="sector",
        ),
    )

    assert registry.names() == ("momentum_one", "momentum_two")
    assert "momentum_one" in registry
    assert "momentum_two" in registry


def test_factor_definition_registry_unknown_name_suggests_close_match():
    registry = FactorDefinitionRegistry()
    registry.register(
        IndustryNeutralMomentumDefinition(
            name="industry_neutral_momentum",
            window=1,
            price_feed="price",
            price_column="close",
            classifier_name="sector",
            classifier_column="sector",
        )
    )

    try:
        registry.get("industry_neutral_mometum")
    except KeyError as exc:
        message = str(exc)
        assert "Unknown factor definition" in message
        assert "industry_neutral_momentum" in message
        assert "Did you mean" in message
    else:
        raise AssertionError("registry.get() should raise KeyError for an unknown name")
