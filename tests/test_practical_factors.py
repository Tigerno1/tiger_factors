from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors import PracticalFactorEngine, practical_factor_names
from tiger_factors.factor_algorithm.data_mining.practical_factors import (
    factor_001_volume_flow_sine_skew,
    factor_041_change_pct_decaymax_volume_standardized,
    factor_042_round_ema_ma_vol_sharpe_spread,
    factor_043_rank_normalize_log_regression_residual,
    factor_044_ts_rank_close_maxsum_percentage_avdiff,
    factor_045_rank_normalize_log_returns_rocttm,
    factor_046_rank_corr_rank_mean_close_volume_turnrate,
    factor_047_turn_rate_mean_ratio,
    factor_048_close_30d_return,
    factor_049_wma_high_open_spread,
    factor_050_max_drawdown_net_mf_amount_v2,
    factor_051_price_liquidity_preference,
    factor_052_volume_stability_adjustment,
    factor_053_liquidity_quality_filter,
    factor_054_direction_strength_times_volatility,
    factor_055_open_close_momentum_resonance,
    factor_056_short_term_volatility_adjusted_return,
    factor_057_inverse_volatility_covariance,
    factor_058_double_volatility_ratio,
    factor_059_momentum_flow_decay,
    factor_060_flow_diff_decay,
    factor_061_momentum_liquidity_flow,
    factor_062_main_flow_decay_volatility_synergy,
    factor_063_main_slarge_flow_rank_synergy,
    factor_064_main_slarge_flow_decay_spread,
    factor_065_cne5_beta_size_ir_spread,
    factor_066_main_flow_volatility_synergy,
    factor_067_flow_rank_decay,
    factor_068_flow_decay_pctchange,
    factor_069_rank_close_delta_cumulative,
    factor_070_abs_reinstatement_flow_momentum,
    factor_071_main_flow_decay_10d,
    factor_072_price_momentum_flow_volatility_inverse_coupling,
    factor_073_close_rank_decay_vroc,
    factor_074_price_main_flow_volatility_negative,
    factor_075_price_volume_volatility_negative,
    factor_076_close_mean_volume_turnrate_reversal,
    factor_077_liquidity_stability_factor,
    factor_078_signedpower_change_pct_mean,
    factor_079_weighted_price_vwap_volatility,
    factor_080_breakout_reversal_condition,
    factor_081_volume_close_covariance_negative,
    factor_082_reversal_turnrate,
    factor_083_signedpower_turnrate_rank_close,
    factor_084_volume_amount_relative_strength,
    factor_085_regression_residual_volatility,
)
from tiger_factors.utils import panel_ops as po


def _sample_practical_panel(include_flow: bool = True) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    rows = []
    configs = [
        ("AAA", 100.0),
        ("BBB", 60.0),
        ("CCC", 80.0),
        ("DDD", 120.0),
    ]
    for code, base in configs:
        for idx, date in enumerate(dates):
            close = base + 0.3 * idx + np.sin(idx / 5.0)
            open_ = close - 0.4
            high = close + 1.2
            low = close - 1.5
            vwap = (open_ + high + low + close) / 4.0
            row = {
                "date_": date,
                "code": code,
                "AF_OPEN": open_,
                "AF_HIGH": high,
                "AF_LOW": low,
                "AF_CLOSE": close,
                "AF_VWAP": vwap,
                "VOLUME": float(1_000_000 + idx * 2_500 + (abs(hash(code)) % 997)),
                "TURN_RATE": float(0.01 + 0.0004 * idx + (len(code) * 0.0001)),
                "FACTOR_ROCTTM": float(0.02 + 0.0002 * idx + (len(code) * 0.00005)),
                "NET_MF_AMOUNT_V2": float(5000.0 + 12.0 * idx + np.sin(idx / 6.0) * 40.0),
                "AMOUNT": float((1_000_000 + idx * 2_500 + (abs(hash(code)) % 997)) * close),
                "FACTOR_VROC12D": float(0.01 + 0.0002 * idx + 0.0003 * len(code)),
                "REINSTATEMENT_CHG_60D": float(0.015 + 0.00015 * idx + 0.0002 * len(code)),
                "MAIN_IN_FLOW_20D_V2": float(1_000.0 + 25.0 * idx + 10.0 * len(code)),
                "SLARGE_IN_FLOW_V2": float(700.0 + 18.0 * idx + 8.0 * len(code)),
                "FACTOR_VOL60D": float(0.02 + 0.00005 * idx + 0.0001 * len(code)),
                "FACTOR_TVSD20D": float(0.015 + 0.00003 * idx + 0.00008 * len(code)),
                "FACTOR_CNE5_BETA": float(0.8 + 0.001 * idx + 0.01 * len(code)),
                "FACTOR_CNE5_SIZE": float(1.2 + 0.0007 * idx + 0.008 * len(code)),
                "MAIN_IN_FLOW_V2": float(1_100.0 + 24.0 * idx + 11.0 * len(code)),
            }
            if include_flow:
                row["MAIN_IN_FLOW_DAYS_10D_V2"] = float((idx % 10) + (len(code) % 3))
            rows.append(row)
    return pd.DataFrame(rows)


def test_practical_factor_registry_exposes_the_new_factor() -> None:
    names = practical_factor_names()
    assert names == (
        "factor_001_volume_flow_sine_skew",
        "factor_041_change_pct_decaymax_volume_standardized",
        "factor_042_round_ema_ma_vol_sharpe_spread",
        "factor_043_rank_normalize_log_regression_residual",
        "factor_044_ts_rank_close_maxsum_percentage_avdiff",
        "factor_045_rank_normalize_log_returns_rocttm",
        "factor_046_rank_corr_rank_mean_close_volume_turnrate",
        "factor_047_turn_rate_mean_ratio",
        "factor_048_close_30d_return",
        "factor_049_wma_high_open_spread",
        "factor_050_max_drawdown_net_mf_amount_v2",
        "factor_051_price_liquidity_preference",
        "factor_052_volume_stability_adjustment",
        "factor_053_liquidity_quality_filter",
        "factor_054_direction_strength_times_volatility",
        "factor_055_open_close_momentum_resonance",
        "factor_056_short_term_volatility_adjusted_return",
        "factor_057_inverse_volatility_covariance",
        "factor_058_double_volatility_ratio",
        "factor_059_momentum_flow_decay",
        "factor_060_flow_diff_decay",
        "factor_061_momentum_liquidity_flow",
        "factor_062_main_flow_decay_volatility_synergy",
        "factor_063_main_slarge_flow_rank_synergy",
        "factor_064_main_slarge_flow_decay_spread",
        "factor_065_cne5_beta_size_ir_spread",
        "factor_066_main_flow_volatility_synergy",
        "factor_067_flow_rank_decay",
        "factor_068_flow_decay_pctchange",
        "factor_069_rank_close_delta_cumulative",
        "factor_070_abs_reinstatement_flow_momentum",
        "factor_071_main_flow_decay_10d",
        "factor_072_price_momentum_flow_volatility_inverse_coupling",
        "factor_073_close_rank_decay_vroc",
        "factor_074_price_main_flow_volatility_negative",
        "factor_075_price_volume_volatility_negative",
        "factor_076_close_mean_volume_turnrate_reversal",
        "factor_077_liquidity_stability_factor",
        "factor_078_signedpower_change_pct_mean",
        "factor_079_weighted_price_vwap_volatility",
        "factor_080_breakout_reversal_condition",
        "factor_081_volume_close_covariance_negative",
        "factor_082_reversal_turnrate",
        "factor_083_signedpower_turnrate_rank_close",
        "factor_084_volume_amount_relative_strength",
        "factor_085_regression_residual_volatility",
    )


def test_practical_factor_engine_matches_manual_formula() -> None:
    panel = _sample_practical_panel(include_flow=True)
    engine = PracticalFactorEngine(panel)

    computed = engine.compute("factor_001_volume_flow_sine_skew")
    factor_name = "factor_001_volume_flow_sine_skew"

    close_return = engine.data["close"] / engine._delay(engine.data["close"], 1) - 1.0
    manual = -1.0 * (
        np.sin(engine._ts_mean(close_return, 5))
        * engine._cs_skew(engine.data["vwap"])
        * engine._ts_median(engine.data["main_in_flow_days_10d_v2"], 20)
        + np.log1p(engine._ts_max_std(engine.data["high"] - engine.data["low"], 60, 3))
        * (engine._ts_max_mean(engine.data["volume"], 20, 5) / engine._ts_mean(engine.data["volume"], 20))
    )
    expected = engine._finalize(manual, factor_name)

    merged = computed.merge(expected, on=["date_", "code"], how="inner", suffixes=("", "_expected"))
    assert not merged.empty
    assert np.allclose(
        merged[factor_name],
        merged[f"{factor_name}_expected"],
        equal_nan=True,
    )


def test_practical_factor_engine_falls_back_without_flow_column() -> None:
    panel = _sample_practical_panel(include_flow=False)
    result = factor_001_volume_flow_sine_skew(panel)

    assert {"date_", "code", "factor_001_volume_flow_sine_skew"}.issubset(result.columns)
    assert result["factor_001_volume_flow_sine_skew"].notna().any()


def test_practical_factor_decaymax_volume_standardized_matches_manual_formula() -> None:
    panel = _sample_practical_panel(include_flow=True)
    engine = PracticalFactorEngine(panel)

    computed = engine.compute("factor_041_change_pct_decaymax_volume_standardized")
    factor_name = "factor_041_change_pct_decaymax_volume_standardized"

    change_pct = engine.data["close"] / engine._delay(engine.data["close"], 1).replace(0, np.nan) - 1.0
    manual = -1.0 * engine._cs_standardize(
        engine._ts_max(engine._ts_decay_linear(change_pct, 20) * np.log1p(engine.data["volume"].clip(lower=0.0)), 3)
    )
    expected = engine._finalize(manual, factor_name)

    merged = computed.merge(expected, on=["date_", "code"], how="inner", suffixes=("", "_expected"))
    assert not merged.empty
    assert np.allclose(
        merged[factor_name],
        merged[f"{factor_name}_expected"],
        equal_nan=True,
    )


def test_practical_factor_round_ema_ma_vol_sharpe_spread_matches_manual_formula() -> None:
    dates = pd.date_range("2023-01-01", periods=240, freq="D")
    rows = []
    for code, base in [("AAA", 100.0), ("BBB", 60.0), ("CCC", 80.0), ("DDD", 120.0)]:
        for idx, date in enumerate(dates):
            close = base + 0.2 * idx + np.sin(idx / 7.0)
            open_ = close - 0.3
            high = close + 1.1
            low = close - 1.2
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "AF_OPEN": open_,
                    "AF_HIGH": high,
                    "AF_LOW": low,
                    "AF_CLOSE": close,
                    "AF_VWAP": (open_ + high + low + close) / 4.0,
                    "VOLUME": float(1_000_000 + idx * 3_000 + (abs(hash(code)) % 101)),
                }
            )
    panel = pd.DataFrame(rows)
    engine = PracticalFactorEngine(panel)

    computed = engine.compute("factor_042_round_ema_ma_vol_sharpe_spread")
    factor_name = "factor_042_round_ema_ma_vol_sharpe_spread"

    previous_close = engine._delay(engine.data["close"], 1).replace(0, np.nan)
    returns = engine.data["close"] / previous_close - 1.0
    manual = (
        engine._round(engine._ts_ema(engine.data["close"], 10) / engine._ts_mean(engine.data["close"], 5).replace(0, np.nan), 1)
        * (engine._rolling_volatility(returns, 120) - engine._rolling_volatility(returns, 20))
        + (engine._rolling_sharpe(returns, 120) - engine._rolling_sharpe(returns, 20))
    )
    expected = engine._finalize(manual, factor_name)

    merged = computed.merge(expected, on=["date_", "code"], how="inner", suffixes=("", "_expected"))
    assert not merged.empty
    assert np.allclose(
        merged[factor_name],
        merged[f"{factor_name}_expected"],
        equal_nan=True,
    )


def test_practical_factor_rank_normalize_log_regression_residual_matches_manual_formula() -> None:
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rows = []
    for code, base in [("AAA", 100.0), ("BBB", 60.0), ("CCC", 80.0), ("DDD", 120.0)]:
        for idx, date in enumerate(dates):
            close = base + 0.12 * idx + np.cos(idx / 6.0)
            open_ = close - 0.25
            high = close + 0.9
            low = close - 1.0
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "AF_OPEN": open_,
                    "AF_HIGH": high,
                    "AF_LOW": low,
                    "AF_CLOSE": close,
                    "AF_VWAP": (open_ + high + low + close) / 4.0,
                    "TURN_RATE": float(0.01 + 0.001 * idx + (len(code) * 0.0001)),
                    "VOLUME": float(1_000_000 + idx * 1_500 + (abs(hash(code)) % 79)),
                }
            )
    panel = pd.DataFrame(rows)
    engine = PracticalFactorEngine(panel)

    computed = engine.compute("factor_043_rank_normalize_log_regression_residual")
    factor_name = "factor_043_rank_normalize_log_regression_residual"

    inverse_turn_rate = engine._inv(engine.data["turn_rate"])
    residual = engine._cs_regression_residual(engine.data["close"], inverse_turn_rate)
    logged = np.log1p(residual.clip(lower=-0.999999999))
    ranked = logged.groupby(engine.data["date"], sort=False).rank(pct=True)
    manual = -1.0 * ranked.groupby(engine.data["date"], sort=False).transform(
        lambda s: 0.0 if s.dropna().empty else (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > engine.eps else 1.0)
    )
    expected = engine._finalize(manual, factor_name)

    merged = computed.merge(expected, on=["date_", "code"], how="inner", suffixes=("", "_expected"))
    assert not merged.empty
    assert np.allclose(
        merged[factor_name],
        merged[f"{factor_name}_expected"],
        equal_nan=True,
    )


def test_new_practical_factors_match_manual_formulas() -> None:
    panel = _sample_practical_panel(include_flow=True)
    engine = PracticalFactorEngine(panel)

    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    rows = []
    phase_map = {"AAA": 0.0, "BBB": 1.5, "CCC": 3.0, "DDD": 4.5}
    for code, base in [("AAA", 100.0), ("BBB", 60.0), ("CCC", 80.0), ("DDD", 120.0)]:
        phase = phase_map[code]
        for idx, date in enumerate(dates):
            close = base + 0.15 * idx + 3.0 * np.sin(idx / 4.0 + phase)
            open_ = close - (0.2 + 0.05 * np.cos(idx / 5.0 + phase))
            high = close + 1.0 + 0.1 * np.sin(idx / 6.0 + phase)
            low = close - 1.2
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "AF_OPEN": open_,
                    "AF_HIGH": high,
                    "AF_LOW": low,
                    "AF_CLOSE": close,
                    "AF_VWAP": (open_ + high + low + close) / 4.0,
                    "VOLUME": float(900_000 + idx * 3_000 + (phase * 1_000) + 80.0 * np.cos(idx / 3.5 + phase)),
                    "TURN_RATE": float(0.01 + 0.0003 * idx + (len(code) * 0.0001) + 0.0008 * np.sin(idx / 7.0 + phase)),
                    "FACTOR_ROCTTM": float(0.02 + 0.00015 * idx + (len(code) * 0.00005)),
                    "NET_MF_AMOUNT_V2": float(4_500.0 + 14.0 * idx + 120.0 * np.sin(idx / 6.0 + phase)),
                    "MAIN_IN_FLOW_DAYS_10D_V2": float((idx % 10) + (len(code) % 3)),
                }
            )
    panel_046 = pd.DataFrame(rows)
    engine_046 = PracticalFactorEngine(panel_046)

    cases = [
        (
            "factor_044_ts_rank_close_maxsum_percentage_avdiff",
            factor_044_ts_rank_close_maxsum_percentage_avdiff(panel),
            -1.0
            * engine._ts_rank_pct(engine.data["close"], 20)
            * ((engine._ts_max_sum(engine.data["high"], 20, 5) - engine._ts_quantile(engine.data["close"], 20, 50))
               + engine._ts_av_diff(engine.data["close"], 20)),
        ),
        (
            "factor_045_rank_normalize_log_returns_rocttm",
            factor_045_rank_normalize_log_returns_rocttm(panel),
            engine._cs_rank(
                engine._cs_standardize(
                    pd.Series(
                        np.log((engine._ts_returns(engine.data["close"], 15, mode=1) + engine.data["factor_rocttm"]).clip(lower=1e-12)),
                        index=engine.data.index,
                    )
                )
            ),
        ),
        (
            "factor_046_rank_corr_rank_mean_close_volume_turnrate",
            factor_046_rank_corr_rank_mean_close_volume_turnrate(panel_046),
            -1.0
            * engine_046._cs_rank(
                engine_046._ts_corr(
                    engine_046._cs_rank(engine_046._ts_mean(engine_046.data["close"], 15)),
                    engine_046._cs_rank(engine_046._ts_mean(engine_046.data["volume"], 15)),
                    10,
                )
            )
            * engine_046._cs_rank(engine_046._ts_mean(engine_046.data["change_pct"], 15))
            * engine_046._cs_rank(engine_046._ts_mean(engine_046.data["turn_rate"], 15))
            * engine_046._cs_rank(engine_046._ts_mean(engine_046.data["volume"], 15)),
        ),
        (
            "factor_047_turn_rate_mean_ratio",
            factor_047_turn_rate_mean_ratio(panel),
            -1.0 * (engine._ts_mean(engine.data["turn_rate"], 20) / engine._ts_mean(engine.data["turn_rate"], 120).replace(0, np.nan)),
        ),
        (
            "factor_048_close_30d_return",
            factor_048_close_30d_return(panel),
            -1.0 * (engine.data["close"] / engine._delay(engine.data["close"], 30) - 1.0),
        ),
        (
            "factor_049_wma_high_open_spread",
            factor_049_wma_high_open_spread(panel),
            -1.0 * engine._ts_wma(engine.data["high"] - engine.data["open"], 5),
        ),
        (
            "factor_050_max_drawdown_net_mf_amount_v2",
            factor_050_max_drawdown_net_mf_amount_v2(panel),
            -1.0 * engine._rolling_max_drawdown(engine.data["net_mf_amount_v2"], 15),
        ),
        (
            "factor_051_price_liquidity_preference",
            factor_051_price_liquidity_preference(panel),
            -1.0 * engine._cs_rank(engine.data["close"]) * engine._cs_rank(engine.data["volume"]),
        ),
        (
            "factor_052_volume_stability_adjustment",
            factor_052_volume_stability_adjustment(panel),
            engine._ts_pctchange(engine.data["factor_vol60d"], 5)
            - engine._ts_pctchange(engine.data["factor_tvsd20d"], 5),
        ),
        (
            "factor_053_liquidity_quality_filter",
            factor_053_liquidity_quality_filter(panel),
            engine._cs_rank(engine.data["volume"]) * (1.0 - engine._cs_rank(engine._ts_std(engine.data["close"], 10))),
        ),
        (
            "factor_054_direction_strength_times_volatility",
            factor_054_direction_strength_times_volatility(panel),
            -1.0
            * engine._cs_rank(engine._ts_sum(engine.data["close"] - engine.data["open"], 15))
            * engine._cs_rank(engine._ts_std(engine.data["close"], 15)),
        ),
        (
            "factor_055_open_close_momentum_resonance",
            factor_055_open_close_momentum_resonance(panel),
            (engine._ts_mean(engine.data["open"], 10) - engine._ts_mean(engine.data["close"], 10))
            * engine._cs_rank(engine._ts_delta(engine.data["close"], 10)),
        ),
        (
            "factor_056_short_term_volatility_adjusted_return",
            factor_056_short_term_volatility_adjusted_return(panel),
            engine._cs_rank(engine._ts_sum(engine.data["close"] - engine.data["open"], 15))
            * (1.0 - engine._cs_rank(engine._ts_std(engine.data["close"], 15))),
        ),
        (
            "factor_057_inverse_volatility_covariance",
            factor_057_inverse_volatility_covariance(panel),
            -1.0
            * engine._ts_cov(
                engine.data["factor_vroc12d"],
                engine._ts_std(engine._cs_rank(engine.data["close"]), 15),
                20,
            ),
        ),
        (
            "factor_058_double_volatility_ratio",
            factor_058_double_volatility_ratio(panel),
            engine._ts_std(engine.data["reinstatement_chg_60d"], 35)
            / engine._ts_std(engine.data["factor_tvsd20d"], 35).replace(0, np.nan),
        ),
        (
            "factor_059_momentum_flow_decay",
            factor_059_momentum_flow_decay(panel),
            engine._ts_decay_linear(engine.data["factor_vroc12d"], 10)
            + engine._ts_decay_linear(engine.data["main_in_flow_20d_v2"], 10),
        ),
        (
            "factor_060_flow_diff_decay",
            factor_060_flow_diff_decay(panel),
            engine._ts_decay_linear(
                engine._cs_rank(engine.data["main_in_flow_20d_v2"]).abs()
                - engine._cs_rank(engine.data["slarge_in_flow_v2"]).abs(),
                15,
            ),
        ),
        (
            "factor_061_momentum_liquidity_flow",
            factor_061_momentum_liquidity_flow(panel),
            engine._ts_decay_linear(engine.data["factor_vol60d"], 10)
            * engine._ts_pctchange(engine.data["main_in_flow_20d_v2"], 10),
        ),
        (
            "factor_062_main_flow_decay_volatility_synergy",
            factor_062_main_flow_decay_volatility_synergy(panel),
            engine._ts_decay_linear(engine.data["main_in_flow_20d_v2"], 10) * engine.data["factor_vol60d"],
        ),
        (
            "factor_063_main_slarge_flow_rank_synergy",
            factor_063_main_slarge_flow_rank_synergy(panel),
            engine._cs_rank(
                engine._ts_rank_pct(engine.data["main_in_flow_20d_v2"], 30)
                * engine._ts_decay_linear(engine.data["slarge_in_flow_v2"], 15)
            ),
        ),
        (
            "factor_064_main_slarge_flow_decay_spread",
            factor_064_main_slarge_flow_decay_spread(panel),
            engine._ts_decay_linear(engine.data["main_in_flow_20d_v2"], 5)
            - engine._ts_decay_linear(engine.data["slarge_in_flow_v2"], 5),
        ),
        (
            "factor_065_cne5_beta_size_ir_spread",
            factor_065_cne5_beta_size_ir_spread(panel),
            engine._ts_ir(engine.data["factor_cne5_beta"], 20) - engine._ts_ir(engine.data["factor_cne5_size"], 20),
        ),
        (
            "factor_066_main_flow_volatility_synergy",
            factor_066_main_flow_volatility_synergy(panel),
            engine._ts_rank_pct(engine.data["main_in_flow_20d_v2"], 10) * engine.data["factor_vol60d"],
        ),
        (
            "factor_067_flow_rank_decay",
            factor_067_flow_rank_decay(panel),
            engine._ts_decay_linear(
                engine._cs_rank(engine.data["main_in_flow_20d_v2"] - engine.data["slarge_in_flow_v2"]),
                15,
            ),
        ),
        (
            "factor_068_flow_decay_pctchange",
            factor_068_flow_decay_pctchange(panel),
            engine._ts_pctchange(engine._ts_decay_linear(engine.data["main_in_flow_20d_v2"], 6), 3),
        ),
        (
            "factor_069_rank_close_delta_cumulative",
            factor_069_rank_close_delta_cumulative(panel),
            -1.0 * engine._ts_sum(engine._ts_delta(engine._cs_rank(engine.data["close"]), 1), 40),
        ),
        (
            "factor_070_abs_reinstatement_flow_momentum",
            factor_070_abs_reinstatement_flow_momentum(panel),
            engine.data["reinstatement_chg_60d"].abs() * engine._ts_decay_linear(engine.data["main_in_flow_20d_v2"], 15),
        ),
        (
            "factor_071_main_flow_decay_10d",
            factor_071_main_flow_decay_10d(panel),
            engine._ts_decay_linear(engine.data["main_in_flow_20d_v2"], 10),
        ),
        (
            "factor_072_price_momentum_flow_volatility_inverse_coupling",
            factor_072_price_momentum_flow_volatility_inverse_coupling(panel),
            -1.0
            * engine._cs_rank(engine.data["close"] / engine._delay(engine.data["close"], 15))
            * engine._cs_rank(engine._ts_std(engine.data["main_in_flow_v2"], 15)),
        ),
        (
            "factor_073_close_rank_decay_vroc",
            factor_073_close_rank_decay_vroc(panel),
            -1.0
            * engine._ts_pctchange(engine._cs_rank(engine.data["close"]), 10)
            * engine._ts_decay_linear(engine.data["factor_vroc12d"], 60),
        ),
        (
            "factor_074_price_main_flow_volatility_negative",
            factor_074_price_main_flow_volatility_negative(panel),
            -1.0
            * engine._cs_rank(engine._ts_std(engine.data["close"], 10))
            * engine._cs_rank(engine._ts_std(engine.data["main_in_flow_v2"], 10)),
        ),
        (
            "factor_075_price_volume_volatility_negative",
            factor_075_price_volume_volatility_negative(panel),
            -1.0
            * engine._cs_rank(engine._ts_std(engine.data["close"], 10))
            * engine._cs_rank(engine._ts_std(engine.data["volume"], 10)),
        ),
        (
            "factor_076_close_mean_volume_turnrate_reversal",
            factor_076_close_mean_volume_turnrate_reversal(panel),
            -1.0
            * engine._cs_rank(engine.data["close"] / engine._ts_mean(engine.data["close"], 20).replace(0, np.nan))
            * engine._cs_rank(engine._ts_mean(engine.data["volume"], 20))
            * engine._cs_rank(engine._ts_mean(engine.data["turn_rate"], 20)),
        ),
        (
            "factor_077_liquidity_stability_factor",
            factor_077_liquidity_stability_factor(panel),
            -1.0 * engine._ts_mean(engine._cs_rank(engine.data["amount"]), 10),
        ),
        (
            "factor_078_signedpower_change_pct_mean",
            factor_078_signedpower_change_pct_mean(panel),
            -1.0 * engine._ts_mean(po.signed_power(engine.data["change_pct"], 2), 30),
        ),
        (
            "factor_079_weighted_price_vwap_volatility",
            factor_079_weighted_price_vwap_volatility(panel),
            -1.0 * engine._ts_std((engine.data["close"] - engine.data["vwap"]) * engine.data["volume"], 10),
        ),
        (
            "factor_080_breakout_reversal_condition",
            factor_080_breakout_reversal_condition(panel),
            pd.Series(
                np.where(
                    engine._ts_mean(engine.data["close"], 35) < engine.data["close"],
                    -1.0 * engine._ts_delta(engine.data["close"], 10),
                    0.0,
                ),
                index=engine.data.index,
            ),
        ),
        (
            "factor_081_volume_close_covariance_negative",
            factor_081_volume_close_covariance_negative(panel),
            -1.0 * engine._ts_cov(engine._ts_delta(engine.data["volume"], 1), engine._ts_delta(engine.data["close"], 1), 30),
        ),
        (
            "factor_082_reversal_turnrate",
            factor_082_reversal_turnrate(panel),
            -1.0 * (engine.data["close"] / engine._delay(engine.data["close"], 5) * engine.data["turn_rate"]),
        ),
        (
            "factor_083_signedpower_turnrate_rank_close",
            factor_083_signedpower_turnrate_rank_close(panel),
            -1.0
            * po.signed_power(engine._cs_rank(engine._ts_std(engine.data["turn_rate"], 14)), 2)
            * engine._ts_rank_pct(engine.data["close"], 30),
        ),
        (
            "factor_084_volume_amount_relative_strength",
            factor_084_volume_amount_relative_strength(panel),
            engine._cs_rank(engine._ts_sum(engine.data["volume"], 30))
            / engine._cs_rank(engine._ts_sum(engine.data["amount"], 30)).replace(0, np.nan),
        ),
        (
            "factor_085_regression_residual_volatility",
            factor_085_regression_residual_volatility(panel),
            -1.0 * engine._ts_std(engine._cs_regression_residual(engine.data["close"], engine.data["volume"]), 20),
        ),
    ]

    for factor_name, computed, manual in cases:
        expected = engine._finalize(manual, factor_name)
        if computed.empty or expected.empty:
            assert computed.empty and expected.empty, factor_name
            continue
        merged = computed.merge(expected, on=["date_", "code"], how="inner", suffixes=("", "_expected"))
        assert not merged.empty, factor_name
        assert np.allclose(
            merged[factor_name],
            merged[f"{factor_name}_expected"],
            equal_nan=True,
        ), factor_name
