# strategies.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.linear_model import LinearRegression
except Exception as e:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from e


# ============================================================
# Strategy interface
# ============================================================
class BaseStrategy:
    """
    A strategy must provide:
      - name: unique strategy name
      - build(...) -> (monthly_returns, exposures_by_date)
          monthly_returns: Series(index=month_end, name=name)
          exposures_by_date: dict[dt]->Series(index=assets_all) representing asset weights
    NOTE:
      - In strategy-level allocator mode we only need monthly_returns (R).
      - exposures_by_date remains available for future asset-level mapping / audit.
    """
    name: str

    def build(
        self,
        prices_daily_close: pd.DataFrame,
        ohlc_daily: Optional[Dict[str, pd.DataFrame]],
        returns_m_all: pd.DataFrame,
        assets_all: List[str],
        start: str,
        end: str,
    ) -> Tuple[pd.Series, Dict[pd.Timestamp, pd.Series]]:
        raise NotImplementedError


# ============================================================
# Strategy implementations
# ============================================================
class SingleAssetStrategy(BaseStrategy):
    """ETF itself as a strategy: return=its monthly return, exposure=100% that asset."""
    def __init__(self, asset: str):
        self.asset = asset
        self.name = asset

    def build(self, prices_daily_close, ohlc_daily, returns_m_all, assets_all, start, end):
        if self.asset not in returns_m_all.columns:
            r = pd.Series(index=returns_m_all.index, dtype=float, name=self.name)
        else:
            r = returns_m_all[self.asset].copy()
            r.name = self.name

        exp = {}
        for dt in r.index:
            v = pd.Series(0.0, index=assets_all)
            if self.asset in v.index:
                v.loc[self.asset] = 1.0
            exp[dt] = v
        return r, exp


class StockReversalStrategy(BaseStrategy):
    """
    Reversal: last-month return rank bottom quantile, EW long.
    Signal uses shift(1) (no look-ahead).
    """
    def __init__(self, stock_universe: List[str], name="Reversal", bottom_q=0.2):
        self.name = name
        self.bottom_q = bottom_q
        self.stock_universe = stock_universe

    def build(self, prices_daily_close, ohlc_daily, returns_m_all, assets_all, start, end):
        stocks = [s for s in self.stock_universe if s in returns_m_all.columns and s in assets_all]
        if len(stocks) == 0:
            r = pd.Series(index=returns_m_all.index, dtype=float, name=self.name)
            exp = {dt: pd.Series(0.0, index=assets_all) for dt in r.index}
            return r, exp

        R_stock = returns_m_all[stocks].copy()
        signal = R_stock.shift(1)  # no look-ahead
        rank = signal.rank(axis=1, pct=True)
        picks = (rank <= self.bottom_q)

        r_list = []
        exp = {}
        for dt in R_stock.index:
            row = R_stock.loc[dt]
            chosen = picks.loc[dt][picks.loc[dt]].index.tolist()
            if len(chosen) == 0:
                r_list.append(0.0)
                exp[dt] = pd.Series(0.0, index=assets_all)
            else:
                w = 1.0 / len(chosen)
                r_list.append(float((row[chosen] * w).sum()))
                v = pd.Series(0.0, index=assets_all)
                v.loc[chosen] = w
                exp[dt] = v

        r = pd.Series(r_list, index=R_stock.index, name=self.name)
        return r, exp


class ETFMomBlendStrategy(BaseStrategy):
    """
    3-factor ETF rotation (bias+slope+efficiency) with threshold switching.
    Uses ETF daily OHLC dict, produces monthly returns and monthly exposures (single ETF held).
    No look-ahead: month t return decided by position chosen at month t-1 end.
    """
    def __init__(
        self,
        etf_list: List[str],
        name="Momentum",
        threshold: float = 1.5,
        BIAS_N: int = 25,
        MOMENTUM_DAY: int = 25,
        SLOPE_N: int = 25,
        EFFICIENCY_N: int = 25,
        w_bias: float = 0.2,
        w_slope: float = 0.3,
        w_eff: float = 0.5
    ):
        self.name = name
        self.etf_list = etf_list
        self.threshold = threshold
        self.BIAS_N = BIAS_N
        self.MOMENTUM_DAY = MOMENTUM_DAY
        self.SLOPE_N = SLOPE_N
        self.EFFICIENCY_N = EFFICIENCY_N
        self.w_bias = w_bias
        self.w_slope = w_slope
        self.w_eff = w_eff

        self._score_df = None
        self._hold_next = None

    def _bias_mom(self, close: pd.Series) -> float:
        if close is None or len(close) < max(self.BIAS_N, self.MOMENTUM_DAY):
            return np.nan
        ma = close.rolling(self.BIAS_N, min_periods=self.BIAS_N).mean()
        bias = (close / ma).dropna()
        if len(bias) < self.MOMENTUM_DAY:
            return np.nan
        recent = bias.iloc[-self.MOMENTUM_DAY:]
        base = float(recent.iloc[0])
        if not np.isfinite(base) or base == 0:
            return np.nan
        x = np.arange(self.MOMENTUM_DAY).reshape(-1, 1)
        y = (recent / base).values.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        return float(lr.coef_[0]) * 10000.0

    def _slope_mom(self, close: pd.Series) -> float:
        if close is None or len(close) < self.SLOPE_N:
            return np.nan
        w = close.iloc[-self.SLOPE_N:]
        base = float(w.iloc[0])
        if not np.isfinite(base) or base <= 0:
            return np.nan
        y = (w / base).values.reshape(-1, 1)
        x = np.arange(1, self.SLOPE_N + 1).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        slope = float(lr.coef_[0])
        r2 = float(lr.score(x, y))
        return 10000.0 * slope * r2

    def _eff_mom(self, ohlc: pd.DataFrame) -> float:
        if ohlc is None or len(ohlc) < self.EFFICIENCY_N:
            return np.nan
        w = ohlc.iloc[-self.EFFICIENCY_N:].copy()
        pivot = (w["open"] + w["high"] + w["low"] + w["close"]) / 4.0
        pivot = pivot.ffill()
        if pivot.isna().any():
            return np.nan
        p0, p1 = float(pivot.iloc[0]), float(pivot.iloc[-1])
        if p0 <= 0 or p1 <= 0 or (not np.isfinite(p0)) or (not np.isfinite(p1)):
            return np.nan
        momentum = 100.0 * np.log(p1 / p0)
        direction = abs(np.log(p1) - np.log(p0))
        volatility = np.log(pivot).diff().abs().sum()
        if (not np.isfinite(volatility)) or volatility <= 0:
            return np.nan
        eff_ratio = direction / volatility
        return float(momentum * eff_ratio)

    def _score(self, ohlc: pd.DataFrame) -> float:
        close = ohlc["close"]
        b = self._bias_mom(close)
        s = self._slope_mom(close)
        e = self._eff_mom(ohlc)
        return self.w_bias * b + self.w_slope * s + self.w_eff * e

    def build(self, prices_daily_close, ohlc_daily, returns_m_all, assets_all, start, end):
        assert ohlc_daily is not None, "ETFMomBlendStrategy requires ohlc_daily dict"

        # monthly close returns for ETFs
        close_daily = pd.DataFrame({e: ohlc_daily[e]["close"] for e in self.etf_list}).loc[start:end]
        close_m = close_daily.resample("M").last()
        ret_m = close_m.pct_change()

        month_ends = close_m.index

        # score at each month end
        score_df = pd.DataFrame(index=month_ends, columns=self.etf_list, dtype=float)
        for me in month_ends:
            for e in self.etf_list:
                o = ohlc_daily[e].loc[:me].dropna(how="any")
                score_df.loc[me, e] = self._score(o)

        # threshold switching -> hold_next (decides next month)
        hold_next = {}
        current = None
        for me in month_ends:
            s = score_df.loc[me].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                hold_next[me] = current
                continue
            best = s.idxmax()
            if current is None:
                current = best
            else:
                cur_score = float(s.loc[current]) if current in s.index else np.nan
                best_score = float(s.loc[best])
                if (not np.isfinite(cur_score)) or (np.isfinite(best_score) and best_score >= self.threshold * cur_score):
                    current = best
            hold_next[me] = current

        hold_next = pd.Series(hold_next, index=month_ends, name="hold_next_month")
        pos_for_month = hold_next.shift(1)

        # build monthly returns + exposures
        r_list = []
        exp = {}
        for me in month_ends:
            pos = pos_for_month.get(me, None)
            if pos is None or pos not in ret_m.columns:
                r_list.append(np.nan)
                exp[me] = pd.Series(0.0, index=assets_all)
            else:
                rr = ret_m.loc[me, pos] if me in ret_m.index else np.nan
                r_list.append(float(rr) if np.isfinite(rr) else np.nan)
                v = pd.Series(0.0, index=assets_all)
                if pos in v.index:
                    v.loc[pos] = 1.0
                exp[me] = v

        r = pd.Series(r_list, index=month_ends, name=self.name)

        # store for debugging (optional)
        self._score_df = score_df
        self._hold_next = hold_next

        return r, exp


# ============================================================
# Strategy Bundle
# ============================================================
@dataclass
class StrategyBundleOutput:
    R: pd.DataFrame
    H_by_date: Dict[pd.Timestamp, pd.DataFrame]
    assets_all: List[str]
    strategies: List[str]


class StrategyBundle:
    """
    Builds a unified strategy universe:
      - R: (T,K) monthly returns matrix
      - H_by_date: dt -> (N,K) exposures
    """
    def __init__(self, strategies: List[BaseStrategy], align_mode: str = "intersection"):
        self._strategies = strategies
        if align_mode not in ("intersection", "union"):
            raise ValueError("align_mode must be 'intersection' or 'union'")
        self.align_mode = align_mode

    def build(
        self,
        prices_daily_close: pd.DataFrame,
        ohlc_daily: Optional[Dict[str, pd.DataFrame]],
        returns_m_all: pd.DataFrame,
        assets_all: List[str],
        start: str,
        end: str
    ) -> StrategyBundleOutput:

        R_list = []
        exp_list = []
        names = []

        for s in self._strategies:
            r_s, exp_s = s.build(
                prices_daily_close=prices_daily_close,
                ohlc_daily=ohlc_daily,
                returns_m_all=returns_m_all,
                assets_all=assets_all,
                start=start,
                end=end
            )
            names.append(s.name)
            R_list.append(r_s)
            exp_list.append(exp_s)

        # align dates
        if self.align_mode == "intersection":
            idx = R_list[0].index
            for r in R_list[1:]:
                idx = idx.intersection(r.index)
        else:
            idx = R_list[0].index
            for r in R_list[1:]:
                idx = idx.union(r.index)

        idx = idx.sort_values()

        R = pd.concat([r.reindex(idx) for r in R_list], axis=1)
        R.columns = names

        # production-safe default: keep only full-data months
        R = R.dropna(how="any")

        H_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
        for dt in R.index:
            H = pd.DataFrame(0.0, index=assets_all, columns=names)
            for j, name in enumerate(names):
                v = exp_list[j].get(dt, None)
                if v is None:
                    continue
                v = v.reindex(assets_all).fillna(0.0)
                H[name] = v.values
            H_by_date[dt] = H

        return StrategyBundleOutput(R=R, H_by_date=H_by_date, assets_all=assets_all, strategies=names)
