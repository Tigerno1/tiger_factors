from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from tiger_factors.factor_algorithm.data_mining.factors import DataMiningEngine
from tiger_factors.utils import panel_ops as po


@dataclass(frozen=True)
class PracticalFactorSpec:
    name: str
    description: str
    source: str
    builder: Callable[["PracticalFactorEngine"], pd.Series]


class PracticalFactorEngine(DataMiningEngine):
    """Practical factor engine for cleaned, directly testable factor formulas.

    The engine accepts a long OHLCV panel with optional aliases:
    - `AF_OPEN`, `AF_HIGH`, `AF_LOW`, `AF_CLOSE`, `AF_VWAP`, `VOLUME`
    - `MAIN_IN_FLOW_DAYS_10D_V2`

    If the inflow column is missing, a simple proxy based on the number of
    positive close-open days in the past 10 sessions is used.
    """

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        rename_map: dict[str, str] = {}
        if "date_" in frame.columns:
            rename_map["date_"] = "date"
        if "code" in frame.columns:
            rename_map["code"] = "symbol"
        alias_map = {
            "AF_OPEN": "open",
            "AF_HIGH": "high",
            "AF_LOW": "low",
            "AF_CLOSE": "close",
            "AF_VWAP": "vwap",
            "VOLUME": "volume",
            "TURN_RATE": "turn_rate",
            "FACTOR_ROCTTM": "factor_rocttm",
            "ROCTTM": "factor_rocttm",
            "NET_MF_AMOUNT_V2": "net_mf_amount_v2",
            "AMOUNT": "amount",
            "CHANGE_PCT": "change_pct",
            "FACTOR_VROC12D": "factor_vroc12d",
            "REINSTATEMENT_CHG_60D": "reinstatement_chg_60d",
            "MAIN_IN_FLOW_20D_V2": "main_in_flow_20d_v2",
            "SLARGE_IN_FLOW_V2": "slarge_in_flow_v2",
            "FACTOR_VOL60D": "factor_vol60d",
            "FACTOR_TVSD20D": "factor_tvsd20d",
            "FACTOR_CNE5_BETA": "factor_cne5_beta",
            "FACTOR_CNE5_SIZE": "factor_cne5_size",
            "MAIN_IN_FLOW_V2": "main_in_flow_v2",
            "MAIN_IN_FLOW_DAYS_10D_V2": "main_in_flow_days_10d_v2",
        }
        for source, target in alias_map.items():
            if source in frame.columns and target not in frame.columns:
                rename_map[source] = target
        frame = frame.rename(columns=rename_map)

        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"practical factor input is missing columns: {sorted(missing)}")

        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame["symbol"] = frame["symbol"].astype(str)
        frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "turn_rate",
            "factor_rocttm",
            "net_mf_amount_v2",
            "amount",
            "change_pct",
            "factor_vroc12d",
            "reinstatement_chg_60d",
            "main_in_flow_20d_v2",
            "slarge_in_flow_v2",
            "factor_vol60d",
            "factor_tvsd20d",
            "factor_cne5_beta",
            "factor_cne5_size",
            "main_in_flow_v2",
            "main_in_flow_days_10d_v2",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "vwap" not in frame.columns:
            frame["vwap"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
        if "change_pct" not in frame.columns:
            frame["change_pct"] = frame.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)
        if "amount" not in frame.columns:
            frame["amount"] = frame["volume"] * frame["close"]
        if "factor_vroc12d" not in frame.columns:
            frame["factor_vroc12d"] = frame.groupby("symbol", sort=False)["volume"].transform(
                lambda s: s.pct_change(12)
            )
        if "reinstatement_chg_60d" not in frame.columns:
            frame["reinstatement_chg_60d"] = frame.groupby("symbol", sort=False)["close"].transform(
                lambda s: s.pct_change(60)
            )
        if "main_in_flow_20d_v2" not in frame.columns:
            signed_flow = np.sign(frame["close"] - frame["open"]).fillna(0.0) * frame["volume"].fillna(0.0)
            frame["main_in_flow_20d_v2"] = signed_flow.groupby(frame["symbol"], sort=False).transform(
                lambda s: s.rolling(20).sum()
            )
        if "main_in_flow_v2" not in frame.columns and "main_in_flow_20d_v2" in frame.columns:
            frame["main_in_flow_v2"] = frame["main_in_flow_20d_v2"]
        if "main_in_flow_20d_v2" not in frame.columns and "main_in_flow_v2" in frame.columns:
            frame["main_in_flow_20d_v2"] = frame["main_in_flow_v2"]
        if "slarge_in_flow_v2" not in frame.columns:
            slarge_flow = frame["volume"].fillna(0.0) * (frame["high"] - frame["low"]).abs().fillna(0.0)
            frame["slarge_in_flow_v2"] = slarge_flow.groupby(frame["symbol"], sort=False).transform(
                lambda s: s.rolling(20).sum()
            )
        if "factor_vol60d" not in frame.columns:
            frame["factor_vol60d"] = frame.groupby("symbol", sort=False)["close"].transform(
                lambda s: s.rolling(60).std(ddof=0)
            )
        if "factor_tvsd20d" not in frame.columns:
            frame["factor_tvsd20d"] = frame.groupby("symbol", sort=False)["close"].transform(
                lambda s: s.rolling(20).std(ddof=0)
            )

        return frame

    def _ts_median(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_median(self.data, series, window)

    def _ts_mean(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_mean(self.data, series, window)

    def _ts_delta(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_delta(self.data, series, window)

    def _ts_rank_pct(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_rank_pct(self.data, series, window)

    def _ts_quantile(self, series: pd.Series, window: int, percentile: float) -> pd.Series:
        return po.ts_quantile(self.data, series, window, percentile)

    def _ts_std(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_std(self.data, series, window)

    def _ts_max_mean(self, series: pd.Series, outer_window: int, inner_window: int) -> pd.Series:
        rolling_mean = po.ts_mean(self.data, series, inner_window)
        return po.ts_max(self.data, rolling_mean, outer_window)

    def _ts_max_std(self, series: pd.Series, outer_window: int, inner_window: int) -> pd.Series:
        rolling_std = po.ts_std(self.data, series, inner_window)
        return po.ts_max(self.data, rolling_std, outer_window)

    def _ts_max(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_max(self.data, series, window)

    def _ts_max_sum(self, series: pd.Series, outer_window: int, inner_window: int) -> pd.Series:
        summed = self._ts_sum(series, inner_window)
        return po.ts_max(self.data, summed, outer_window)

    def _ts_av_diff(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_av_diff(self.data, series, window)

    def _ts_returns(self, series: pd.Series, periods: int, mode: int = 1) -> pd.Series:
        return po.ts_returns(self.data, series, periods, mode=mode)

    def _ts_decay_linear(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_decay_linear(self.data, series, window)

    def _cs_skew(self, series: pd.Series) -> pd.Series:
        return po.cs_skew(self.data, series, date_col="date")

    def _cs_standardize(self, series: pd.Series) -> pd.Series:
        return po.cs_standardize(self.data, series, date_col="date", eps=self.eps)

    def _cs_rank(self, series: pd.Series) -> pd.Series:
        return po.cs_rank(self.data, series, date_col="date", pct=True)

    def _round(self, series: pd.Series, decimals: int = 1) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.Series(np.round(numeric, decimals), index=series.index)

    def _rolling_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        return po.rolling_volatility(self.data, returns, window)

    def _rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        return po.rolling_sharpe(self.data, returns, window)

    def _ts_ir(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_ir(self.data, series, window)

    def _cs_regression_residual(self, y: pd.Series, x: pd.Series) -> pd.Series:
        return po.cs_regression_residual(self.data, y, x, date_col="date")

    def _rolling_max_drawdown(self, series: pd.Series, window: int) -> pd.Series:
        return po.rolling_max_drawdown(self.data, series, window)

    def _resolve_main_in_flow(self) -> pd.Series:
        if "main_in_flow_days_10d_v2" in self.data.columns:
            return self.data["main_in_flow_days_10d_v2"]

        positive_days = (self.data["close"] > self.data["open"]).astype(float)
        return self._group_apply(positive_days, lambda s: s.rolling(10).sum())

    def factor_001_volume_flow_sine_skew(self) -> pd.Series:
        previous_close = self._delay(self.data["close"], 1).replace(0, np.nan)
        close_return = self.data["close"] / previous_close - 1.0
        term_1 = np.sin(self._ts_mean(close_return, 5))
        term_2 = self._cs_skew(self.data["vwap"])
        term_3 = self._ts_median(self._resolve_main_in_flow(), 20)
        term_4 = np.log1p(self._ts_max_std(self.data["high"] - self.data["low"], 60, 3))
        term_5 = self._ts_max_mean(self.data["volume"], 20, 5) / self._ts_mean(self.data["volume"], 20).replace(0, np.nan)

        out = -1.0 * (term_1 * term_2 * term_3 + term_4 * term_5)
        return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def factor_041_change_pct_decaymax_volume_standardized(self) -> pd.Series:
        change_pct = self.data["close"] / self._delay(self.data["close"], 1).replace(0, np.nan) - 1.0
        decay = self._ts_decay_linear(change_pct, 20)
        core = decay * np.log1p(self.data["volume"].clip(lower=0.0))
        peaked = self._ts_max(core, 3)
        normalized = self._cs_standardize(peaked)
        out = -1.0 * normalized
        return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def factor_042_round_ema_ma_vol_sharpe_spread(self) -> pd.Series:
        previous_close = self._delay(self.data["close"], 1).replace(0, np.nan)
        returns = self.data["close"] / previous_close - 1.0

        ema10 = self._ts_ema(self.data["close"], 10)
        ma5 = self._ts_mean(self.data["close"], 5).replace(0, np.nan)
        ratio = self._round(ema10 / ma5, 1)

        vol120 = self._rolling_volatility(returns, 120)
        vol20 = self._rolling_volatility(returns, 20)
        sharpe120 = self._rolling_sharpe(returns, 120)
        sharpe20 = self._rolling_sharpe(returns, 20)

        out = ratio * (vol120 - vol20) + (sharpe120 - sharpe20)
        return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def factor_043_rank_normalize_log_regression_residual(self) -> pd.Series:
        inverse_turn_rate = self._inv(self.data.get("turn_rate", self.data["volume"].replace(0, np.nan)))
        residual = self._cs_regression_residual(self.data["close"], inverse_turn_rate)
        logged = np.log1p(residual.clip(lower=-0.999999999))
        ranked = logged.groupby(self.data["date"], sort=False).rank(pct=True)
        normalized = ranked.groupby(self.data["date"], sort=False).transform(
            lambda s: 0.0 if s.dropna().empty else (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > self.eps else 1.0)
        )
        return -1.0 * pd.to_numeric(normalized, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def factor_044_ts_rank_close_maxsum_percentage_avdiff(self) -> pd.Series:
        close_rank = self._ts_rank_pct(self.data["close"], 20)
        max_sum = self._ts_max_sum(self.data["high"], 20, 5)
        percentage = self._ts_quantile(self.data["close"], 20, 50)
        av_diff = self._ts_av_diff(self.data["close"], 20)
        return -1.0 * close_rank * ((max_sum - percentage) + av_diff)

    def factor_045_rank_normalize_log_returns_rocttm(self) -> pd.Series:
        rocttm = self.data.get("factor_rocttm")
        if rocttm is None:
            raise ValueError("factor_045_rank_normalize_log_returns_rocttm requires FACTOR_ROCTTM or ROCTTM.")
        returns = self._ts_returns(self.data["close"], 15, mode=1)
        logged = np.log((returns + rocttm).clip(lower=1e-12))
        return self._cs_rank(self._cs_standardize(pd.Series(logged, index=self.data.index)))

    def factor_046_rank_corr_rank_mean_close_volume_turnrate(self) -> pd.Series:
        close_mean_rank = self._cs_rank(self._ts_mean(self.data["close"], 15))
        volume_mean_rank = self._cs_rank(self._ts_mean(self.data["volume"], 15))
        corr_rank = self._cs_rank(self._ts_corr(close_mean_rank, volume_mean_rank, 10))
        change_pct_rank = self._cs_rank(self._ts_mean(self.data["change_pct"], 15))
        turn_rate = self.data.get("turn_rate")
        if turn_rate is None:
            raise ValueError("factor_046_rank_corr_rank_mean_close_volume_turnrate requires TURN_RATE.")
        turn_rate_rank = self._cs_rank(self._ts_mean(turn_rate, 15))
        volume_rank = self._cs_rank(self._ts_mean(self.data["volume"], 15))
        return -1.0 * corr_rank * change_pct_rank * turn_rate_rank * volume_rank

    def factor_047_turn_rate_mean_ratio(self) -> pd.Series:
        turn_rate = self.data.get("turn_rate")
        if turn_rate is None:
            raise ValueError("factor_047_turn_rate_mean_ratio requires TURN_RATE.")
        short_mean = self._ts_mean(turn_rate, 20)
        long_mean = self._ts_mean(turn_rate, 120)
        return -1.0 * (short_mean / long_mean.replace(0, np.nan))

    def factor_048_close_30d_return(self) -> pd.Series:
        return -1.0 * (self.data["close"] / self._delay(self.data["close"], 30) - 1.0)

    def factor_049_wma_high_open_spread(self) -> pd.Series:
        return -1.0 * self._ts_wma(self.data["high"] - self.data["open"], 5)

    def factor_050_max_drawdown_net_mf_amount_v2(self) -> pd.Series:
        flow = self.data.get("net_mf_amount_v2")
        if flow is None:
            raise ValueError("factor_050_max_drawdown_net_mf_amount_v2 requires NET_MF_AMOUNT_V2.")
        return -1.0 * self._rolling_max_drawdown(flow, 15)

    def factor_051_price_liquidity_preference(self) -> pd.Series:
        return -1.0 * self._cs_rank(self.data["close"]) * self._cs_rank(self.data["volume"])

    def factor_052_volume_stability_adjustment(self) -> pd.Series:
        factor_vol60d = self.data["factor_vol60d"]
        factor_tvsd20d = self.data["factor_tvsd20d"]
        return self._ts_pctchange(factor_vol60d, 5) - self._ts_pctchange(factor_tvsd20d, 5)

    def factor_053_liquidity_quality_filter(self) -> pd.Series:
        return self._cs_rank(self.data["volume"]) * (1.0 - self._cs_rank(self._ts_std(self.data["close"], 10)))

    def factor_054_direction_strength_times_volatility(self) -> pd.Series:
        return -1.0 * self._cs_rank(self._ts_sum(self.data["close"] - self.data["open"], 15)) * self._cs_rank(
            self._ts_std(self.data["close"], 15)
        )

    def factor_055_open_close_momentum_resonance(self) -> pd.Series:
        spread = self._ts_mean(self.data["open"], 10) - self._ts_mean(self.data["close"], 10)
        return spread * self._cs_rank(self._ts_delta(self.data["close"], 10))

    def factor_056_short_term_volatility_adjusted_return(self) -> pd.Series:
        return self._cs_rank(self._ts_sum(self.data["close"] - self.data["open"], 15)) * (
            1.0 - self._cs_rank(self._ts_std(self.data["close"], 15))
        )

    def factor_057_inverse_volatility_covariance(self) -> pd.Series:
        close_rank_std = self._ts_std(self._cs_rank(self.data["close"]), 15)
        return -1.0 * self._ts_cov(self.data["factor_vroc12d"], close_rank_std, 20)

    def factor_058_double_volatility_ratio(self) -> pd.Series:
        numerator = self._ts_std(self.data["reinstatement_chg_60d"], 35)
        denominator = self._ts_std(self.data["factor_tvsd20d"], 35).replace(0, np.nan)
        return numerator / denominator

    def factor_059_momentum_flow_decay(self) -> pd.Series:
        return self._ts_decay_linear(self.data["factor_vroc12d"], 10) + self._ts_decay_linear(
            self.data["main_in_flow_20d_v2"],
            10,
        )

    def factor_060_flow_diff_decay(self) -> pd.Series:
        diff = self._cs_rank(self.data["main_in_flow_20d_v2"]).abs() - self._cs_rank(self.data["slarge_in_flow_v2"]).abs()
        return self._ts_decay_linear(diff, 15)

    def factor_061_momentum_liquidity_flow(self) -> pd.Series:
        return self._ts_decay_linear(self.data["factor_vol60d"], 10) * self._ts_pctchange(
            self.data["main_in_flow_20d_v2"],
            10,
        )

    def factor_062_main_flow_decay_volatility_synergy(self) -> pd.Series:
        return self._ts_decay_linear(self.data["main_in_flow_20d_v2"], 10) * self.data["factor_vol60d"]

    def factor_063_main_slarge_flow_rank_synergy(self) -> pd.Series:
        core = self._ts_rank_pct(self.data["main_in_flow_20d_v2"], 30) * self._ts_decay_linear(
            self.data["slarge_in_flow_v2"],
            15,
        )
        return self._cs_rank(core)

    def factor_064_main_slarge_flow_decay_spread(self) -> pd.Series:
        return self._ts_decay_linear(self.data["main_in_flow_20d_v2"], 5) - self._ts_decay_linear(
            self.data["slarge_in_flow_v2"],
            5,
        )

    def factor_065_cne5_beta_size_ir_spread(self) -> pd.Series:
        beta = self.data.get("factor_cne5_beta")
        size = self.data.get("factor_cne5_size")
        if beta is None or size is None:
            raise ValueError("factor_065_cne5_beta_size_ir_spread requires FACTOR_CNE5_BETA and FACTOR_CNE5_SIZE.")
        return self._ts_ir(beta, 20) - self._ts_ir(size, 20)

    def factor_066_main_flow_volatility_synergy(self) -> pd.Series:
        return self._ts_rank_pct(self.data["main_in_flow_20d_v2"], 10) * self.data["factor_vol60d"]

    def factor_067_flow_rank_decay(self) -> pd.Series:
        spread = self.data["main_in_flow_20d_v2"] - self.data["slarge_in_flow_v2"]
        return self._ts_decay_linear(self._cs_rank(spread), 15)

    def factor_068_flow_decay_pctchange(self) -> pd.Series:
        return self._ts_pctchange(self._ts_decay_linear(self.data["main_in_flow_20d_v2"], 6), 3)

    def factor_069_rank_close_delta_cumulative(self) -> pd.Series:
        ranked_close = self._cs_rank(self.data["close"])
        return -1.0 * self._ts_sum(self._ts_delta(ranked_close, 1), 40)

    def factor_070_abs_reinstatement_flow_momentum(self) -> pd.Series:
        return self.data["reinstatement_chg_60d"].abs() * self._ts_decay_linear(self.data["main_in_flow_20d_v2"], 15)

    def factor_071_main_flow_decay_10d(self) -> pd.Series:
        return self._ts_decay_linear(self.data["main_in_flow_20d_v2"], 10)

    def factor_072_price_momentum_flow_volatility_inverse_coupling(self) -> pd.Series:
        price_momentum = self._cs_rank(self.data["close"] / self._delay(self.data["close"], 15))
        flow_vol = self._cs_rank(self._ts_std(self.data["main_in_flow_v2"], 15))
        return -1.0 * price_momentum * flow_vol

    def factor_073_close_rank_decay_vroc(self) -> pd.Series:
        ranked_close_pct = self._ts_pctchange(self._cs_rank(self.data["close"]), 10)
        return -1.0 * ranked_close_pct * self._ts_decay_linear(self.data["factor_vroc12d"], 60)

    def factor_074_price_main_flow_volatility_negative(self) -> pd.Series:
        return -1.0 * self._cs_rank(self._ts_std(self.data["close"], 10)) * self._cs_rank(
            self._ts_std(self.data["main_in_flow_v2"], 10)
        )

    def factor_075_price_volume_volatility_negative(self) -> pd.Series:
        return -1.0 * self._cs_rank(self._ts_std(self.data["close"], 10)) * self._cs_rank(
            self._ts_std(self.data["volume"], 10)
        )

    def factor_076_close_mean_volume_turnrate_reversal(self) -> pd.Series:
        turn_rate = self.data.get("turn_rate")
        if turn_rate is None:
            raise ValueError("factor_076_close_mean_volume_turnrate_reversal requires TURN_RATE.")
        close_rel = self.data["close"] / self._ts_mean(self.data["close"], 20).replace(0, np.nan)
        volume_rank = self._cs_rank(self._ts_mean(self.data["volume"], 20))
        turn_rate_rank = self._cs_rank(self._ts_mean(turn_rate, 20))
        return -1.0 * self._cs_rank(close_rel) * volume_rank * turn_rate_rank

    def factor_077_liquidity_stability_factor(self) -> pd.Series:
        return -1.0 * self._ts_mean(self._cs_rank(self.data["amount"]), 10)

    def factor_078_signedpower_change_pct_mean(self) -> pd.Series:
        if "change_pct" not in self.data.columns:
            raise ValueError("factor_078_signedpower_change_pct_mean requires CHANGE_PCT.")
        return -1.0 * self._ts_mean(po.signed_power(self.data["change_pct"], 2), 30)

    def factor_079_weighted_price_vwap_volatility(self) -> pd.Series:
        return -1.0 * self._ts_std((self.data["close"] - self.data["vwap"]) * self.data["volume"], 10)

    def factor_080_breakout_reversal_condition(self) -> pd.Series:
        ma35 = self._ts_mean(self.data["close"], 35)
        triggered = ma35 < self.data["close"]
        reverse_move = -1.0 * self._ts_delta(self.data["close"], 10)
        return pd.Series(np.where(triggered, reverse_move, 0.0), index=self.data.index)

    def factor_081_volume_close_covariance_negative(self) -> pd.Series:
        vol_delta = self._ts_delta(self.data["volume"], 1)
        close_delta = self._ts_delta(self.data["close"], 1)
        return -1.0 * self._ts_cov(vol_delta, close_delta, 30)

    def factor_082_reversal_turnrate(self) -> pd.Series:
        if "turn_rate" not in self.data.columns:
            raise ValueError("factor_082_reversal_turnrate requires TURN_RATE.")
        return -1.0 * (self.data["close"] / self._delay(self.data["close"], 5) * self.data["turn_rate"])

    def factor_083_signedpower_turnrate_rank_close(self) -> pd.Series:
        if "turn_rate" not in self.data.columns:
            raise ValueError("factor_083_signedpower_turnrate_rank_close requires TURN_RATE.")
        turn_rate_vol = self._cs_rank(self._ts_std(self.data["turn_rate"], 14))
        close_rank = self._ts_rank_pct(self.data["close"], 30)
        return -1.0 * po.signed_power(turn_rate_vol, 2) * close_rank

    def factor_084_volume_amount_relative_strength(self) -> pd.Series:
        volume_sum_rank = self._cs_rank(self._ts_sum(self.data["volume"], 30))
        amount_sum_rank = self._cs_rank(self._ts_sum(self.data["amount"], 30)).replace(0, np.nan)
        return volume_sum_rank / amount_sum_rank

    def factor_085_regression_residual_volatility(self) -> pd.Series:
        residual = self._cs_regression_residual(self.data["close"], self.data["volume"])
        return -1.0 * self._ts_std(residual, 20)


def factor_001_volume_flow_sine_skew(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_001_volume_flow_sine_skew")


def factor_041_change_pct_decaymax_volume_standardized(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_041_change_pct_decaymax_volume_standardized")


def factor_042_round_ema_ma_vol_sharpe_spread(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_042_round_ema_ma_vol_sharpe_spread")


def factor_043_rank_normalize_log_regression_residual(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_043_rank_normalize_log_regression_residual")


def factor_044_ts_rank_close_maxsum_percentage_avdiff(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_044_ts_rank_close_maxsum_percentage_avdiff")


def factor_045_rank_normalize_log_returns_rocttm(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_045_rank_normalize_log_returns_rocttm")


def factor_046_rank_corr_rank_mean_close_volume_turnrate(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_046_rank_corr_rank_mean_close_volume_turnrate")


def factor_047_turn_rate_mean_ratio(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_047_turn_rate_mean_ratio")


def factor_048_close_30d_return(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_048_close_30d_return")


def factor_049_wma_high_open_spread(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_049_wma_high_open_spread")


def factor_050_max_drawdown_net_mf_amount_v2(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_050_max_drawdown_net_mf_amount_v2")


def factor_051_price_liquidity_preference(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_051_price_liquidity_preference")


def factor_052_volume_stability_adjustment(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_052_volume_stability_adjustment")


def factor_053_liquidity_quality_filter(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_053_liquidity_quality_filter")


def factor_054_direction_strength_times_volatility(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_054_direction_strength_times_volatility")


def factor_055_open_close_momentum_resonance(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_055_open_close_momentum_resonance")


def factor_056_short_term_volatility_adjusted_return(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_056_short_term_volatility_adjusted_return")


def factor_057_inverse_volatility_covariance(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_057_inverse_volatility_covariance")


def factor_058_double_volatility_ratio(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_058_double_volatility_ratio")


def factor_059_momentum_flow_decay(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_059_momentum_flow_decay")


def factor_060_flow_diff_decay(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_060_flow_diff_decay")


def factor_061_momentum_liquidity_flow(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_061_momentum_liquidity_flow")


def factor_062_main_flow_decay_volatility_synergy(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_062_main_flow_decay_volatility_synergy")


def factor_063_main_slarge_flow_rank_synergy(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_063_main_slarge_flow_rank_synergy")


def factor_064_main_slarge_flow_decay_spread(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_064_main_slarge_flow_decay_spread")


def factor_065_cne5_beta_size_ir_spread(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_065_cne5_beta_size_ir_spread")


def factor_066_main_flow_volatility_synergy(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_066_main_flow_volatility_synergy")


def factor_067_flow_rank_decay(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_067_flow_rank_decay")


def factor_068_flow_decay_pctchange(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_068_flow_decay_pctchange")


def factor_069_rank_close_delta_cumulative(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_069_rank_close_delta_cumulative")


def factor_070_abs_reinstatement_flow_momentum(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_070_abs_reinstatement_flow_momentum")


def factor_071_main_flow_decay_10d(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_071_main_flow_decay_10d")


def factor_072_price_momentum_flow_volatility_inverse_coupling(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_072_price_momentum_flow_volatility_inverse_coupling")


def factor_073_close_rank_decay_vroc(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_073_close_rank_decay_vroc")


def factor_074_price_main_flow_volatility_negative(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_074_price_main_flow_volatility_negative")


def factor_075_price_volume_volatility_negative(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_075_price_volume_volatility_negative")


def factor_076_close_mean_volume_turnrate_reversal(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_076_close_mean_volume_turnrate_reversal")


def factor_077_liquidity_stability_factor(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_077_liquidity_stability_factor")


def factor_078_signedpower_change_pct_mean(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_078_signedpower_change_pct_mean")


def factor_079_weighted_price_vwap_volatility(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_079_weighted_price_vwap_volatility")


def factor_080_breakout_reversal_condition(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_080_breakout_reversal_condition")


def factor_081_volume_close_covariance_negative(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_081_volume_close_covariance_negative")


def factor_082_reversal_turnrate(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_082_reversal_turnrate")


def factor_083_signedpower_turnrate_rank_close(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_083_signedpower_turnrate_rank_close")


def factor_084_volume_amount_relative_strength(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_084_volume_amount_relative_strength")


def factor_085_regression_residual_volatility(data: pd.DataFrame) -> pd.DataFrame:
    engine = PracticalFactorEngine(data)
    return engine.compute("factor_085_regression_residual_volatility")


def available_practical_factors() -> tuple[str, ...]:
    return (
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


PRACTICAL_FACTOR_SPECS: tuple[PracticalFactorSpec, ...] = (
    PracticalFactorSpec(
        name="factor_001_volume_flow_sine_skew",
        description=(
            "-1 * (sin(mean(close / delay(close, 1) - 1, 5)) * cs_skew(vwap) * "
            "median(main_in_flow_days_10d_v2, 20) + log1p(max_std(high - low, 60, 3)) * "
            "(max_mean(volume, 20, 5) / mean(volume, 20)))"
        ),
        source=(
            "-1 * (SIN(TS_MEAN(AF_CLOSE/TS_DELAY(AF_CLOSE,1)-1,5)) * "
            "CS_SKEW(AF_VWAP) * TS_MEDIAN(MAIN_IN_FLOW_DAYS_10D_V2,20) + "
            "LOG(1+TS_MAX_STD(AF_HIGH-AF_LOW,60,3)) * "
            "(TS_MAX_MEAN(VOLUME,20,5)/TS_MEAN(VOLUME,20)))"
        ),
        builder=PracticalFactorEngine.factor_001_volume_flow_sine_skew,
    ),
    PracticalFactorSpec(
        name="factor_041_change_pct_decaymax_volume_standardized",
        description=(
            "-1 * standardize(max(ts_decay_linear(change_pct, 20) * log1p(volume), 3))"
        ),
        source=(
            "-1 * (NORMALIZE[1](TS_MAX[2](TS_DECAY_LINEAR[3](CHANGE_PCT,20) * "
            "LOG(VOLUME+1),3),STANDARDIZE=1))"
        ),
        builder=PracticalFactorEngine.factor_041_change_pct_decaymax_volume_standardized,
    ),
    PracticalFactorSpec(
        name="factor_042_round_ema_ma_vol_sharpe_spread",
        description=(
            "round(ema(close, 10) / ma(close, 5), 1) * (volatility_120d - volatility_20d) + "
            "(sharpe_120d - sharpe_20d)"
        ),
        source="ROUND[1](FACTOR_EMA10D / FACTOR_MA5D) * (FACTOR_VOL120D - FACTOR_VOL20D) + (FACTOR_SHARPE120D - FACTOR_SHARPE20D)",
        builder=PracticalFactorEngine.factor_042_round_ema_ma_vol_sharpe_spread,
    ),
    PracticalFactorSpec(
        name="factor_043_rank_normalize_log_regression_residual",
        description=(
            "-1 * rank_normalize(log1p(regression_residual(close ~ inverse_turn_rate)))"
        ),
        source=(
            "-1 * RANK_NORMALIZE(S_LOG_LP[1](CS_REGRESSION[2](AF_CLOSE,-1/(TURN_RATE),"
            "DUMMIES=0, OUT_TYPE=0,WITH_ONE_COL=1,FILL_PREDICT=1)))"
        ),
        builder=PracticalFactorEngine.factor_043_rank_normalize_log_regression_residual,
    ),
    PracticalFactorSpec(
        name="factor_044_ts_rank_close_maxsum_percentage_avdiff",
        description=(
            "-1 * ts_rank(close, 20) * ((max_sum(high, 20, 5) - percentile(close, 20, 50)) + av_diff(close, 20))"
        ),
        source=(
            "-1 * (TS_RANK[1](CLOSE,20) * ((TS_MAX_SUM2(HIGH,20,5) - TS_PERCENTAGE3(CLOSE,20,50)) + "
            "TS_AV_DIFF4(CLOSE,20)))"
        ),
        builder=PracticalFactorEngine.factor_044_ts_rank_close_maxsum_percentage_avdiff,
    ),
    PracticalFactorSpec(
        name="factor_045_rank_normalize_log_returns_rocttm",
        description="rank(normalize(log(ts_returns(close, 15) + factor_rocttm)))",
        source="1 * (RANK(NORMALIZE(LOG(TS_RETURNS(CLOSE, 15, MODE=1) + FACTOR_ROCTTM))))",
        builder=PracticalFactorEngine.factor_045_rank_normalize_log_returns_rocttm,
    ),
    PracticalFactorSpec(
        name="factor_046_rank_corr_rank_mean_close_volume_turnrate",
        description=(
            "-1 * rank(corr(rank(mean(close, 15)), rank(mean(volume, 15)), 10)) * "
            "rank(mean(change_pct, 15)) * rank(mean(turn_rate, 15)) * rank(mean(volume, 15))"
        ),
        source=(
            "-1*RANK(TS_CORR(RANK(TS_MEAN(CLOSE,15)), RANK(TS_MEAN(VOLUME,15)),10))* "
            "RANK(TS_MEAN(CHANGE_PCT,15))* RANK(TS_MEAN(TURN_RATE,15))* RANK(TS_MEAN(VOLUME,15))"
        ),
        builder=PracticalFactorEngine.factor_046_rank_corr_rank_mean_close_volume_turnrate,
    ),
    PracticalFactorSpec(
        name="factor_047_turn_rate_mean_ratio",
        description="-mean(turn_rate, 20) / mean(turn_rate, 120)",
        source="-(TS_MEAN(TURN_RATE, 20) / TS_MEAN(TURN_RATE, 120))",
        builder=PracticalFactorEngine.factor_047_turn_rate_mean_ratio,
    ),
    PracticalFactorSpec(
        name="factor_048_close_30d_return",
        description="-1 * (close / delay(close, 30) - 1)",
        source="-(AF_CLOSE/[1]DELAY(AF_CLOSE,30)-1)",
        builder=PracticalFactorEngine.factor_048_close_30d_return,
    ),
    PracticalFactorSpec(
        name="factor_049_wma_high_open_spread",
        description="-wma(high - open, 5)",
        source="-(TS_WMA((HIGH-OPEN),5))",
        builder=PracticalFactorEngine.factor_049_wma_high_open_spread,
    ),
    PracticalFactorSpec(
        name="factor_050_max_drawdown_net_mf_amount_v2",
        description="-max_drawdown(net_mf_amount_v2, 15)",
        source="TS_MAX_DRAWDOWN(NET_MF_AMOUNT_V2,15) * (-1)",
        builder=PracticalFactorEngine.factor_050_max_drawdown_net_mf_amount_v2,
    ),
    PracticalFactorSpec(
        name="factor_051_price_liquidity_preference",
        description="-rank(close) * rank(volume)",
        source="-1 * RANK(CLOSE) * RANK(VOLUME)",
        builder=PracticalFactorEngine.factor_051_price_liquidity_preference,
    ),
    PracticalFactorSpec(
        name="factor_052_volume_stability_adjustment",
        description="ts_percentage(factor_vol60d, 5) - ts_percentage(factor_tvsd20d, 5)",
        source="TS_PERCENTAGE(FACTOR_VOL60D,5) - TS_PERCENTAGE(FACTOR_TVSD20D,5)",
        builder=PracticalFactorEngine.factor_052_volume_stability_adjustment,
    ),
    PracticalFactorSpec(
        name="factor_053_liquidity_quality_filter",
        description="rank(volume) * (1 - rank(ts_std(close, 10)))",
        source="RANK[1](VOLUME) * (1 - RANK(TS_STDDEV[2](CLOSE,10)))",
        builder=PracticalFactorEngine.factor_053_liquidity_quality_filter,
    ),
    PracticalFactorSpec(
        name="factor_054_direction_strength_times_volatility",
        description="-rank(ts_sum(close - open, 15)) * rank(ts_std(close, 15))",
        source="-1 * RANK[1](TS_SUM[2](CLOSE-OPEN,15)) * RANK(TS_STDDEV[3](CLOSE,15))",
        builder=PracticalFactorEngine.factor_054_direction_strength_times_volatility,
    ),
    PracticalFactorSpec(
        name="factor_055_open_close_momentum_resonance",
        description="(ts_mean(open, 10) - ts_mean(close, 10)) * rank(ts_delta(close, 10))",
        source="(TS_MEAN[1](OPEN,10) - TS_MEAN(CLOSE,10)) * RANK[2](TS_DELTA[3](CLOSE,10))",
        builder=PracticalFactorEngine.factor_055_open_close_momentum_resonance,
    ),
    PracticalFactorSpec(
        name="factor_056_short_term_volatility_adjusted_return",
        description="rank(ts_sum(close - open, 15)) * (1 - rank(ts_std(close, 15)))",
        source="TS_SUM(CLOSE-OPEN,15) * (1 - RANK(TS_STDDEV(CLOSE,15)))",
        builder=PracticalFactorEngine.factor_056_short_term_volatility_adjusted_return,
    ),
    PracticalFactorSpec(
        name="factor_057_inverse_volatility_covariance",
        description="-covariance(vroc12d, ts_std(rank(close), 15), 20)",
        source="TS_COVARIANCE(FACTOR_VROC12D, TS_STDDEV(RANK(CLOSE), 15), 20) * (-1)",
        builder=PracticalFactorEngine.factor_057_inverse_volatility_covariance,
    ),
    PracticalFactorSpec(
        name="factor_058_double_volatility_ratio",
        description="ts_std(reinstatement_chg_60d, 35) / ts_std(factor_tvsd20d, 35)",
        source="TS_STDDEV(REINSTATEMENT_CHG_60D,35) / TS_STDDEV(FACTOR_TVSD20D,35)",
        builder=PracticalFactorEngine.factor_058_double_volatility_ratio,
    ),
    PracticalFactorSpec(
        name="factor_059_momentum_flow_decay",
        description="ts_decay_linear(vroc12d, 10) + ts_decay_linear(main_in_flow_20d_v2, 10)",
        source="TS_DECAY_LINEAR(FACTOR_VROC12D,10) + TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,10)",
        builder=PracticalFactorEngine.factor_059_momentum_flow_decay,
    ),
    PracticalFactorSpec(
        name="factor_060_flow_diff_decay",
        description="ts_decay_linear(abs(rank(main_in_flow_20d_v2)) - abs(rank(slarge_in_flow_v2)), 15)",
        source="TS_DECAY_LINEAR(ABS(RANK(MAIN_IN_FLOW_20D_V2))-ABS(RANK(SLARGE_IN_FLOW_V2)),15)",
        builder=PracticalFactorEngine.factor_060_flow_diff_decay,
    ),
    PracticalFactorSpec(
        name="factor_061_momentum_liquidity_flow",
        description="ts_decay_linear(factor_vol60d, 10) * ts_percentage(main_in_flow_20d_v2, 10)",
        source="TS_DECAY_LINEAR(FACTOR_VOL60D,10) * TS_PERCENTAGE(MAIN_IN_FLOW_20D_V2,10)",
        builder=PracticalFactorEngine.factor_061_momentum_liquidity_flow,
    ),
    PracticalFactorSpec(
        name="factor_062_main_flow_decay_volatility_synergy",
        description="ts_decay_linear(main_in_flow_20d_v2, 10) * factor_vol60d",
        source="TS_DECAY_LINEAR(MAIN_IN_FLOW_20D_V2,10) * FACTOR_VOL60D",
        builder=PracticalFactorEngine.factor_062_main_flow_decay_volatility_synergy,
    ),
    PracticalFactorSpec(
        name="factor_063_main_slarge_flow_rank_synergy",
        description="rank(ts_percentage(main_in_flow_20d_v2, 30) * ts_decay_linear(slarge_in_flow_v2, 15))",
        source="RANK[1](TS_PERCENTAGE[2](MAIN_IN_FLOW_20D_V2,30) * TS_DECAY_LINEAR[3](SLARGE_IN_FLOW_V2,15))",
        builder=PracticalFactorEngine.factor_063_main_slarge_flow_rank_synergy,
    ),
    PracticalFactorSpec(
        name="factor_064_main_slarge_flow_decay_spread",
        description="ts_decay_linear(main_in_flow_20d_v2, 5) - ts_decay_linear(slarge_in_flow_v2, 5)",
        source="(TS_DECAY_LINEAR[1](MAIN_IN_FLOW_20D_V2,5) - TS_DECAY_LINEAR(SLARGE_IN_FLOW_V2,5))",
        builder=PracticalFactorEngine.factor_064_main_slarge_flow_decay_spread,
    ),
    PracticalFactorSpec(
        name="factor_065_cne5_beta_size_ir_spread",
        description="ts_ir(factor_cne5_beta, 20) - ts_ir(factor_cne5_size, 20)",
        source="TS_IR(FACTOR_CNE5_BETA,20) - TS_IR(FACTOR_CNE5_SIZE,20)",
        builder=PracticalFactorEngine.factor_065_cne5_beta_size_ir_spread,
    ),
    PracticalFactorSpec(
        name="factor_066_main_flow_volatility_synergy",
        description="ts_percentage(main_in_flow_20d_v2, 10) * factor_vol60d",
        source="TS_PERCENTAGE[1](MAIN_IN_FLOW_20D_V2,10) * FACTOR_VOL60D",
        builder=PracticalFactorEngine.factor_066_main_flow_volatility_synergy,
    ),
    PracticalFactorSpec(
        name="factor_067_flow_rank_decay",
        description="ts_decay_linear(rank(main_in_flow_20d_v2 - slarge_in_flow_v2), 15)",
        source="TS_DECAY_LINEAR[1](RANK[2](MAIN_IN_FLOW_20D_V2-SLARGE_IN_FLOW_V2),15)",
        builder=PracticalFactorEngine.factor_067_flow_rank_decay,
    ),
    PracticalFactorSpec(
        name="factor_068_flow_decay_pctchange",
        description="ts_percentage(ts_decay_linear(main_in_flow_20d_v2, 6), 3)",
        source="TS_PERCENTAGE[1](TS_DECAY_LINEAR[2](MAIN_IN_FLOW_20D_V2,6),3)",
        builder=PracticalFactorEngine.factor_068_flow_decay_pctchange,
    ),
    PracticalFactorSpec(
        name="factor_069_rank_close_delta_cumulative",
        description="-1 * ts_sum(ts_delta(rank(close), 1), 40)",
        source="-1*TS_SUM[1](TS_DELTA[2](RANK[3](AF_CLOSE),1),40)",
        builder=PracticalFactorEngine.factor_069_rank_close_delta_cumulative,
    ),
    PracticalFactorSpec(
        name="factor_070_abs_reinstatement_flow_momentum",
        description="abs(reinstatement_chg_60d) * ts_decay_linear(main_in_flow_20d_v2, 15)",
        source="(ABS[1](REINSTATEMENT_CHG_60D)* TS_DECAY_LINEAR[2](MAIN_IN_FLOW_20D_V2,15))",
        builder=PracticalFactorEngine.factor_070_abs_reinstatement_flow_momentum,
    ),
    PracticalFactorSpec(
        name="factor_071_main_flow_decay_10d",
        description="ts_decay_linear(main_in_flow_20d_v2, 10)",
        source="TS_DECAY_LINEAR[1](MAIN_IN_FLOW_20D_V2,10)",
        builder=PracticalFactorEngine.factor_071_main_flow_decay_10d,
    ),
    PracticalFactorSpec(
        name="factor_072_price_momentum_flow_volatility_inverse_coupling",
        description="-rank(close / delay(close, 15)) * rank(ts_std(main_in_flow_v2, 15))",
        source="RANK[1](AF_CLOSE/DELAY[2](AF_CLOSE,15))* RANK(TS_STDDEV[3](MAIN_IN_FLOW_V2,15))*(-1)",
        builder=PracticalFactorEngine.factor_072_price_momentum_flow_volatility_inverse_coupling,
    ),
    PracticalFactorSpec(
        name="factor_073_close_rank_decay_vroc",
        description="-ts_percentage(rank(close), 10) * ts_decay_linear(factor_vroc12d, 60)",
        source="-TS_PERCENTAGE[1](RANK[2](CLOSE),10)* TS_DECAY_LINEAR[3](FACTOR_VROC12D,60)",
        builder=PracticalFactorEngine.factor_073_close_rank_decay_vroc,
    ),
    PracticalFactorSpec(
        name="factor_074_price_main_flow_volatility_negative",
        description="-rank(ts_std(close, 10)) * rank(ts_std(main_in_flow_v2, 10))",
        source="-1*RANK[1](TS_STDDEV[1](CLOSE,10))* RANK[2](TS_STDDEV(MAIN_IN_FLOW_V2,10))",
        builder=PracticalFactorEngine.factor_074_price_main_flow_volatility_negative,
    ),
    PracticalFactorSpec(
        name="factor_075_price_volume_volatility_negative",
        description="-rank(ts_std(close, 10)) * rank(ts_std(volume, 10))",
        source="-1 * RANK(TS_STDDEV[2](CLOSE,10))*RANK(TS_STDDEV(VOLUME,10))",
        builder=PracticalFactorEngine.factor_075_price_volume_volatility_negative,
    ),
    PracticalFactorSpec(
        name="factor_076_close_mean_volume_turnrate_reversal",
        description="-rank(close / mean(close, 20)) * rank(mean(volume, 20)) * rank(mean(turn_rate, 20))",
        source="RANK[1](CLOSE/TS_MEAN(CLOSE,20))* RANK(TS_MEAN(VOLUME,20))* RANK(TS_MEAN[2](TURN_RATE,20)) * (-1)",
        builder=PracticalFactorEngine.factor_076_close_mean_volume_turnrate_reversal,
    ),
    PracticalFactorSpec(
        name="factor_077_liquidity_stability_factor",
        description="-mean(rank(amount), 10)",
        source="-TS_MEAN[1](RANK[2](AMOUNT),10)",
        builder=PracticalFactorEngine.factor_077_liquidity_stability_factor,
    ),
    PracticalFactorSpec(
        name="factor_078_signedpower_change_pct_mean",
        description="-mean(signedpower(change_pct, 2), 30)",
        source="(-1*TS_MEAN[1](SIGNEDPOWER(CHANGE_PCT,2),30))",
        builder=PracticalFactorEngine.factor_078_signedpower_change_pct_mean,
    ),
    PracticalFactorSpec(
        name="factor_079_weighted_price_vwap_volatility",
        description="-std((close - vwap) * volume, 10)",
        source="-1*TS_STDDEV((CLOSE-VWAP) *VOLUME, 10)",
        builder=PracticalFactorEngine.factor_079_weighted_price_vwap_volatility,
    ),
    PracticalFactorSpec(
        name="factor_080_breakout_reversal_condition",
        description="if(close > mean(close, 35), -delta(close, 10), 0)",
        source="(((TS_SUM[1](CLOSE[2], 35) / 35) < CLOSE) ? (-1 * DELTA(CLOSE,10)) : 0)",
        builder=PracticalFactorEngine.factor_080_breakout_reversal_condition,
    ),
    PracticalFactorSpec(
        name="factor_081_volume_close_covariance_negative",
        description="-covariance(delta(volume, 1), delta(close, 1), 30)",
        source="TS_COVARIANCE[1](DELTA(VOLUME[2],1),DELTA(CLOSE[3], 1), 30) * (-1)",
        builder=PracticalFactorEngine.factor_081_volume_close_covariance_negative,
    ),
    PracticalFactorSpec(
        name="factor_082_reversal_turnrate",
        description="-(close / delay(close, 5) * turn_rate)",
        source="-(AF_CLOSE/DELAY(AF_CLOSE, 5)*TURN_RATE)",
        builder=PracticalFactorEngine.factor_082_reversal_turnrate,
    ),
    PracticalFactorSpec(
        name="factor_083_signedpower_turnrate_rank_close",
        description="-signed_power(rank(ts_std(turn_rate, 14)), 2) * ts_rank(close, 30)",
        source="-1*SIGNED_POWER(RANK(TS_STDDEV(TURN_RATE, 14)),2)*TS_RANK(CLOSE,30)",
        builder=PracticalFactorEngine.factor_083_signedpower_turnrate_rank_close,
    ),
    PracticalFactorSpec(
        name="factor_084_volume_amount_relative_strength",
        description="rank(sum(volume, 30)) / rank(sum(amount, 30))",
        source="RANK(TS_SUM(VOLUME,30))/RANK(TS_SUM(AMOUNT, 30))",
        builder=PracticalFactorEngine.factor_084_volume_amount_relative_strength,
    ),
    PracticalFactorSpec(
        name="factor_085_regression_residual_volatility",
        description="-std(cs_regression_residual(close, volume), 20)",
        source="-TS_STDDEV[1](CS_REGRESSION[2](CLOSE, VOLUME, OUT_TYPE=0), 20)",
        builder=PracticalFactorEngine.factor_085_regression_residual_volatility,
    ),
)


__all__ = [
    "PracticalFactorEngine",
    "PracticalFactorSpec",
    "PRACTICAL_FACTOR_SPECS",
    "available_practical_factors",
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
]
