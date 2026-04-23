from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


def _round_window(value: float) -> int:
    return max(int(round(float(value))), 1)


@dataclass(frozen=True)
class GTJA191Columns:
    date: str = "date"
    code: str = "symbol"
    index_open: str = "index_open"
    index_close: str = "index_close"


def _as_pandas_frame(frame: object) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    to_pandas = getattr(frame, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return pd.DataFrame(frame)


class GTJA191Engine:
    FIELD_SPEC: ClassVar[dict[str, tuple[str, ...]]] = {
        "required": ("date", "symbol", "open", "high", "low", "close", "volume"),
        "optional": ("vwap", "return", "amount", "index_open", "index_close", "benchmark_open", "benchmark_close", "mkt", "smb", "hml"),
    }
    FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "date_": "date",
        "tradetime": "date",
        "code": "symbol",
        "securityid": "symbol",
        "ticker": "symbol",
        "vol": "volume",
        "returns": "return",
        "market_return": "mkt",
        "fama_mkt": "mkt",
        "fama3_mkt": "mkt",
        "mkt_return": "mkt",
        "fama_smb": "smb",
        "fama3_smb": "smb",
        "fama_hml": "hml",
        "fama3_hml": "hml",
        "benchmark_open": "index_open",
        "benchmark_close": "index_close",
        "banchmarkindexopen": "index_open",
        "banchmarkindexclose": "index_close",
        "benchmarkindexopen": "index_open",
        "benchmarkindexclose": "index_close",
    }

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = self._prepare_input(data)
        self.eps = 1e-8

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        rename_map = {source: target for source, target in self.FIELD_ALIASES.items() if source in frame.columns and target not in frame.columns}
        frame = frame.rename(columns=rename_map)

        required = set(self.FIELD_SPEC["required"])
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"gtja191 input is missing columns: {sorted(missing)}")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
        frame["symbol"] = frame["symbol"].astype(str)
        frame = frame.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)

        numeric_columns = [column for column in [*self.FIELD_SPEC["required"][2:], *self.FIELD_SPEC["optional"]] if column in frame.columns]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "vwap" not in frame.columns:
            frame["vwap"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
        else:
            frame["vwap"] = frame["vwap"].fillna((frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0)

        if "amount" not in frame.columns:
            frame["amount"] = frame["close"] * frame["volume"]

        frame["return"] = frame.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)

        if "index_open" in frame.columns:
            frame["index_open"] = pd.to_numeric(frame["index_open"], errors="coerce")
        if "index_close" in frame.columns:
            frame["index_close"] = pd.to_numeric(frame["index_close"], errors="coerce")
        if "benchmark_open" in frame.columns and "index_open" not in frame.columns:
            frame["index_open"] = pd.to_numeric(frame["benchmark_open"], errors="coerce")
        if "benchmark_close" in frame.columns and "index_close" not in frame.columns:
            frame["index_close"] = pd.to_numeric(frame["benchmark_close"], errors="coerce")
        if "mkt" not in frame.columns:
            frame["mkt"] = np.nan
        if "smb" not in frame.columns:
            frame["smb"] = np.nan
        if "hml" not in frame.columns:
            frame["hml"] = np.nan

        return frame

    def _rank(self, x: pd.Series) -> pd.Series:
        return x.groupby(self.data["date"]).rank(pct=True)

    def _delay(self, x: pd.Series, d: int) -> pd.Series:
        return x.groupby(self.data["symbol"]).shift(_round_window(d))

    def _delta(self, x: pd.Series, d: int) -> pd.Series:
        return x.groupby(self.data["symbol"]).diff(_round_window(d))

    def _rolling_last_rank(self, values: np.ndarray) -> float:
        series = pd.Series(values)
        return float(series.rank(pct=True).iloc[-1])

    def _correlation(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            series_x = pd.Series(x.loc[indexer].to_numpy(), index=indexer)
            series_y = pd.Series(y.loc[indexer].to_numpy(), index=indexer)
            pieces.append(series_x.rolling(window).corr(series_y))
        result = pd.concat(pieces).sort_index()
        return result.replace([np.inf, -np.inf], np.nan)

    def _covariance(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            series_x = pd.Series(x.loc[indexer].to_numpy(), index=indexer)
            series_y = pd.Series(y.loc[indexer].to_numpy(), index=indexer)
            pieces.append(series_x.rolling(window).cov(series_y))
        return pd.concat(pieces).sort_index()

    def _scale(self, x: pd.Series, a: float = 1.0) -> pd.Series:
        return x.groupby(self.data["date"]).transform(lambda s: s * a / (s.abs().sum() + self.eps))

    def _decay_linear(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        weights = np.arange(1, window + 1, dtype=float)
        result = (
            x.groupby(self.data["symbol"])
            .rolling(window)
            .apply(lambda s: float(np.dot(s, weights) / weights.sum()), raw=True)
            .reset_index(level=0, drop=True)
        )
        result.index = x.index
        return result

    def _ts_min(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).min().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_max(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).max().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_argmin(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).apply(lambda s: float(np.argmin(s)), raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_argmax(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).apply(lambda s: float(np.argmax(s)), raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _ts_rank(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).apply(self._rolling_last_rank, raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _sum(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).sum().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _mean(self, x: pd.Series, d: int) -> pd.Series:
        return self._sum(x, d) / _round_window(d)

    def _product(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).apply(np.prod, raw=True).reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _stddev(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        result = x.groupby(self.data["symbol"]).rolling(window).std().reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _count(self, condition: pd.Series, d: int) -> pd.Series:
        return self._sum(condition.astype(float), d)

    def _ma(self, x: pd.Series, d: int) -> pd.Series:
        return self._mean(x, d)

    def _adv(self, d: int) -> pd.Series:
        window = _round_window(d)
        result = self.data["volume"].groupby(self.data["symbol"]).rolling(window).mean().reset_index(level=0, drop=True)
        result.index = self.data.index
        return result

    def _bool(self, condition: pd.Series) -> pd.Series:
        return condition.astype(float)

    def _sma(self, x: pd.Series, n: int, m: int) -> pd.Series:
        window = _round_window(n)
        smooth = _round_window(m)
        alpha = smooth / window

        def _apply(series: pd.Series) -> pd.Series:
            values = series.to_numpy(dtype=float)
            out = np.full_like(values, np.nan, dtype=float)
            prev = np.nan
            for idx, value in enumerate(values):
                if np.isnan(value):
                    out[idx] = np.nan
                    continue
                if np.isnan(prev):
                    prev = value
                else:
                    prev = alpha * value + (1 - alpha) * prev
                out[idx] = prev
            return pd.Series(out, index=series.index)

        result = x.groupby(self.data["symbol"], sort=False).apply(_apply)
        if isinstance(result.index, pd.MultiIndex):
            result = result.reset_index(level=0, drop=True)
        result.index = x.index
        return result

    def _wma(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        weights = np.arange(1, window + 1, dtype=float)
        result = (
            x.groupby(self.data["symbol"])
            .rolling(window)
            .apply(lambda s: float(np.dot(s, weights) / weights.sum()), raw=True)
            .reset_index(level=0, drop=True)
        )
        result.index = x.index
        return result

    def _sumif(self, values: pd.Series, condition: pd.Series, d: int) -> pd.Series:
        if not isinstance(condition, pd.Series):
            condition = pd.Series(condition, index=values.index)
        masked = values.where(condition.astype(bool), 0.0)
        return self._sum(masked, d)

    def _regbeta(self, y: pd.Series, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            series_y = pd.Series(y.loc[indexer].to_numpy(), index=indexer)
            series_x = pd.Series(x.loc[indexer].to_numpy(), index=indexer)

            def _beta(window_y: np.ndarray) -> float:
                if len(window_y) < window:
                    return float("nan")
                window_x = series_x.loc[window_y.index].to_numpy(dtype=float)
                window_y_values = window_y.to_numpy(dtype=float)
                x_mean = np.nanmean(window_x)
                y_mean = np.nanmean(window_y_values)
                x_var = np.nanvar(window_x)
                if np.isnan(x_var) or x_var == 0:
                    return float("nan")
                cov = np.nanmean((window_x - x_mean) * (window_y_values - y_mean))
                return float(cov / x_var)

            pieces.append(series_y.rolling(window).apply(lambda arr: _beta(pd.Series(arr)), raw=False))
        result = pd.concat(pieces).sort_index()
        result.index = y.index
        return result

    def _regbeta_to_sequence(self, y: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            series_y = pd.Series(y.loc[indexer].to_numpy(), index=indexer)
            seq = pd.Series(np.arange(1, len(series_y) + 1, dtype=float), index=indexer)

            def _beta(window_y: pd.Series) -> float:
                window_index = window_y.index
                window_x = seq.loc[window_index].to_numpy(dtype=float)
                window_y_values = window_y.to_numpy(dtype=float)
                x_mean = np.nanmean(window_x)
                y_mean = np.nanmean(window_y_values)
                x_var = np.nanvar(window_x)
                if np.isnan(x_var) or x_var == 0:
                    return float("nan")
                cov = np.nanmean((window_x - x_mean) * (window_y_values - y_mean))
                return float(cov / x_var)

            pieces.append(series_y.rolling(window).apply(lambda arr: _beta(pd.Series(arr, index=seq.index[: len(arr)])), raw=False))
        result = pd.concat(pieces).sort_index()
        result.index = y.index
        return result

    def _highday(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            series = pd.Series(x.loc[indexer].to_numpy(), index=indexer)
            def _days_since_high(values: np.ndarray) -> float:
                if len(values) == 0:
                    return float("nan")
                return float(len(values) - 1 - int(np.nanargmax(values)))
            pieces.append(series.rolling(window).apply(_days_since_high, raw=True))
        result = pd.concat(pieces).sort_index()
        result.index = x.index
        return result

    def _lowday(self, x: pd.Series, d: int) -> pd.Series:
        window = _round_window(d)
        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            series = pd.Series(x.loc[indexer].to_numpy(), index=indexer)
            def _days_since_low(values: np.ndarray) -> float:
                if len(values) == 0:
                    return float("nan")
                return float(len(values) - 1 - int(np.nanargmin(values)))
            pieces.append(series.rolling(window).apply(_days_since_low, raw=True))
        result = pd.concat(pieces).sort_index()
        result.index = x.index
        return result

    def _cross_sectional_max(self, x: pd.Series) -> pd.Series:
        return x.groupby(self.data["date"]).transform("max")

    def _cross_sectional_min(self, x: pd.Series) -> pd.Series:
        return x.groupby(self.data["date"]).transform("min")

    def _finalize(self, values: pd.Series, factor_name: str) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "date_": self.data["date"],
                "code": self.data["symbol"],
                factor_name: pd.to_numeric(values, errors="coerce"),
            }
        )
        return result.dropna(subset=[factor_name]).reset_index(drop=True)

    def factor_names(self) -> list[str]:
        return [f"alpha_{i:03d}" for i in range(1, 192)]

    def implemented_alpha_ids(self) -> list[int]:
        return sorted(
            int(name.split("_", 1)[1])
            for name in dir(self)
            if name.startswith("alpha_") and name[6:].isdigit() and callable(getattr(self, name))
        )

    def alpha_1(self) -> pd.Series:
        return -self._correlation(self._rank(self._delta(np.log(self.data["volume"].replace(0, np.nan)), 1)), self._rank((self.data["close"] - self.data["open"]) / (self.data["open"] + self.eps)), 6)

    def alpha_2(self) -> pd.Series:
        return -self._delta((((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"])) / (self.data["high"] - self.data["low"] + self.eps)), 1)

    def alpha_3(self) -> pd.Series:
        prev_close = self._delay(self.data["close"], 1)
        current = self.data["close"]
        lower = np.minimum(self.data["low"], prev_close)
        upper = np.maximum(self.data["high"], prev_close)
        delta = pd.Series(
            np.where(
                current == prev_close,
                0.0,
                np.where(current > prev_close, current - lower, current - upper),
            ),
            index=self.data.index,
        )
        return self._sum(delta, 6)

    def alpha_4(self) -> pd.Series:
        mean8 = self._mean(self.data["close"], 8)
        std8 = self._stddev(self.data["close"], 8)
        mean2 = self._mean(self.data["close"], 2)
        adv20 = self._adv(20)
        alpha = pd.Series(
            np.where(
                mean8.add(std8) < mean2,
                -1.0,
                np.where(mean2 < mean8.sub(std8), 1.0, np.where((self.data["volume"] / (adv20 + self.eps)) >= 1.0, 1.0, -1.0)),
            ),
            index=self.data.index,
        )
        alpha.loc[mean8.isna() | std8.isna() | mean2.isna() | adv20.isna()] = np.nan
        return alpha

    def alpha_5(self) -> pd.Series:
        return -self._ts_max(self._correlation(self._ts_rank(self.data["volume"], 5), self._ts_rank(self.data["high"], 5), 5), 3)

    def alpha_6(self) -> pd.Series:
        return -self._rank(np.sign(self._delta(self.data["open"] * 0.85 + self.data["high"] * 0.15, 4)))

    def alpha_7(self) -> pd.Series:
        left = self._rank(self._ts_max(self.data["vwap"] - self.data["close"], 3))
        right = self._rank(self._ts_min(self.data["vwap"] - self.data["close"], 3))
        return (left + right) * self._rank(self._delta(self.data["volume"], 3))

    def alpha_8(self) -> pd.Series:
        inner = (((self.data["high"] + self.data["low"]) / 2.0) * 0.2) + (self.data["vwap"] * 0.8)
        return -self._rank(self._delta(inner, 4))

    def alpha_9(self) -> pd.Series:
        hl_mid = (self.data["high"] + self.data["low"]) / 2.0
        prev_mid = (self._delay(self.data["high"], 1) + self._delay(self.data["low"], 1)) / 2.0
        numerator = (hl_mid - prev_mid) * (self.data["high"] - self.data["low"])
        denominator = self.data["volume"].replace(0, np.nan)
        return self._sma(numerator / denominator, 7, 2)

    def alpha_10(self) -> pd.Series:
        ret_std = self._stddev(self.data["return"], 20)
        base = pd.Series(np.where(self.data["return"] < 0, ret_std, self.data["close"]), index=self.data.index)
        return self._rank(self._ts_max(base.pow(2), 5))

    def alpha_11(self) -> pd.Series:
        numerator = ((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"]))
        denominator = (self.data["high"] - self.data["low"]).replace(0, np.nan)
        return self._sum((numerator / denominator) * self.data["volume"], 6)

    def alpha_12(self) -> pd.Series:
        left = self._rank(self.data["open"] - (self._sum(self.data["vwap"], 10) / 10.0))
        right = -self._rank((self.data["close"] - self.data["vwap"]).abs())
        return left * right

    def alpha_13(self) -> pd.Series:
        return np.sqrt(self.data["high"] * self.data["low"]) - self.data["vwap"]

    def alpha_14(self) -> pd.Series:
        return self.data["close"] - self._delay(self.data["close"], 5)

    def alpha_15(self) -> pd.Series:
        return self.data["open"] / (self._delay(self.data["close"], 1) + self.eps) - 1.0

    def alpha_16(self) -> pd.Series:
        return -self._ts_max(self._rank(self._correlation(self._rank(self.data["volume"]), self._rank(self.data["vwap"]), 5)), 5)

    def alpha_17(self) -> pd.Series:
        return self._rank(self.data["vwap"] - self._ts_max(self.data["vwap"], 15)).pow(self._delta(self.data["close"], 5))

    def alpha_18(self) -> pd.Series:
        return self.data["close"] / (self._delay(self.data["close"], 5) + self.eps)

    def alpha_19(self) -> pd.Series:
        prev_close = self._delay(self.data["close"], 5)
        delta = self.data["close"] - prev_close
        return pd.Series(
            np.where(
                self.data["close"] < prev_close,
                delta / (prev_close + self.eps),
                np.where(self.data["close"] == prev_close, 0.0, delta / (self.data["close"] + self.eps)),
            ),
            index=self.data.index,
        )

    def alpha_20(self) -> pd.Series:
        return (self.data["close"] - self._delay(self.data["close"], 6)) / (self._delay(self.data["close"], 6) + self.eps) * 100.0

    def alpha_21(self) -> pd.Series:
        return self._regbeta_to_sequence(self._mean(self.data["close"], 6), 6)

    def alpha_22(self) -> pd.Series:
        normalized = (self.data["close"] - self._mean(self.data["close"], 6)) / (self._mean(self.data["close"], 6) + self.eps)
        inner = normalized - self._delay(normalized, 3)
        return self._sma(inner, 12, 1)

    def alpha_23(self) -> pd.Series:
        close_std = self._stddev(self.data["close"], 20)
        prev_close = self._delay(self.data["close"], 1)
        condition = self.data["close"] > prev_close
        left = self._sma(close_std.where(condition, 0.0), 20, 1)
        right_up = self._sma(close_std.where(condition, 0.0), 20, 1)
        right_down = self._sma(close_std.where(~condition, 0.0), 20, 1)
        alpha = left / (right_up + right_down + self.eps) * 100.0
        alpha.loc[close_std.isna()] = np.nan
        return alpha

    def alpha_24(self) -> pd.Series:
        return self._sma(self.data["close"] - self._delay(self.data["close"], 5), 5, 1)

    def alpha_25(self) -> pd.Series:
        return self._rank(self._delta(self.data["close"], 7) * (1 - self._rank(self._decay_linear(self.data["volume"] / (self._adv(20) + self.eps), 9)))) * -(1 + self._rank(self._sum(self.data["return"], 250)))

    def alpha_26(self) -> pd.Series:
        return (self._sum(self.data["close"], 7) / 7.0) - self.data["close"] + self._correlation(self.data["vwap"], self._delay(self.data["close"], 5), 230)

    def alpha_27(self) -> pd.Series:
        return self._wma(
            (self.data["close"] - self._delay(self.data["close"], 3)) / (self._delay(self.data["close"], 3) + self.eps) * 100.0
            + (self.data["close"] - self._delay(self.data["close"], 6)) / (self._delay(self.data["close"], 6) + self.eps) * 100.0,
            12,
        )

    def alpha_28(self) -> pd.Series:
        tp = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        num = (self.data["close"] - self._ts_min(self.data["low"], 9)) / (self._ts_max(self.data["high"], 9) - self._ts_min(self.data["low"], 9) + self.eps) * 100.0
        first = 3 * self._sma(num, 3, 1)
        second = 2 * self._sma(self._sma((self.data["close"] - self._ts_min(self.data["low"], 9)) / (self._ts_max(self.data["high"], 9) - self._ts_min(self.data["low"], 9) + self.eps) * 100.0, 3, 1), 3, 1)
        return first - second

    def alpha_29(self) -> pd.Series:
        return (self.data["close"] - self._delay(self.data["close"], 6)) / (self._delay(self.data["close"], 6) + self.eps) * self.data["volume"]

    def alpha_30(self) -> pd.Series:
        required = [self.data.get("mkt"), self.data.get("smb"), self.data.get("hml")]
        if any(series is None or series.isna().all() for series in required):
            return pd.Series(np.nan, index=self.data.index, dtype=float)

        def _rolling_residual(group: pd.DataFrame) -> pd.Series:
            y = group["return"].to_numpy(dtype=float)
            x = np.column_stack(
                [
                    np.ones(len(group), dtype=float),
                    group["mkt"].to_numpy(dtype=float),
                    group["smb"].to_numpy(dtype=float),
                    group["hml"].to_numpy(dtype=float),
                ]
            )
            out = np.full(len(group), np.nan, dtype=float)
            window = 60
            for idx in range(window - 1, len(group)):
                y_win = y[idx - window + 1 : idx + 1]
                x_win = x[idx - window + 1 : idx + 1]
                mask = np.isfinite(y_win) & np.isfinite(x_win).all(axis=1)
                if mask.sum() < 10:
                    continue
                x_use = x_win[mask]
                y_use = y_win[mask]
                try:
                    beta, *_ = np.linalg.lstsq(x_use, y_use, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                x_last = x[idx]
                if not np.isfinite(x_last).all():
                    continue
                out[idx] = float(y[idx] - x_last @ beta)
            return pd.Series(out, index=group.index)

        pieces: list[pd.Series] = []
        for _, indexer in self.data.groupby("symbol", sort=False).groups.items():
            group = self.data.loc[indexer, ["return", "mkt", "smb", "hml"]].copy()
            series = _rolling_residual(group)
            series.index = indexer
            pieces.append(series)
        residual = pd.concat(pieces).sort_index()
        residual.index = self.data.index
        return self._wma(residual.pow(2), 20)

    def alpha_31(self) -> pd.Series:
        return (self.data["close"] - self._mean(self.data["close"], 12)) / (self._mean(self.data["close"], 12) + self.eps) * 100.0

    def alpha_32(self) -> pd.Series:
        return -self._sum(self._rank(self._correlation(self._rank(self.data["high"]), self._rank(self.data["volume"]), 3)), 3)

    def alpha_33(self) -> pd.Series:
        low_5 = self._ts_min(self.data["low"], 5)
        rank_term = self._rank((self._sum(self.data["return"], 240) - self._sum(self.data["return"], 20)) / 220.0)
        return ((-low_5) + self._delay(low_5, 5)) * rank_term * self._ts_rank(self.data["volume"], 5)

    def alpha_34(self) -> pd.Series:
        return self._mean(self.data["close"], 12) / (self.data["close"] + self.eps)

    def alpha_35(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["open"], 1), 15))
        right = self._rank(self._decay_linear(self._correlation(self.data["volume"], self.data["open"], 17), 7))
        return -pd.concat([left, right], axis=1).min(axis=1)

    def alpha_36(self) -> pd.Series:
        return self._rank(self._sum(self._correlation(self._rank(self.data["volume"]), self._rank(self.data["vwap"]), 6), 6))

    def alpha_37(self) -> pd.Series:
        return -self._rank((self._sum(self.data["open"], 5) * self._sum(self.data["return"], 5)) - self._delay(self._sum(self.data["open"], 5) * self._sum(self.data["return"], 5), 10))

    def alpha_38(self) -> pd.Series:
        high_20_mean = self._mean(self.data["high"], 20)
        return pd.Series(np.where(high_20_mean < self.data["high"], -self._delta(self.data["high"], 2), 0.0), index=self.data.index)

    def alpha_39(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["close"], 2), 8))
        right = self._rank(self._decay_linear(self._correlation((self.data["vwap"] * 0.3) + (self.data["open"] * 0.7), self._sum(self._mean(self.data["volume"], 180), 37), 14), 12))
        return -(left - right)

    def alpha_40(self) -> pd.Series:
        up_volume = self._sum(pd.Series(np.where(self.data["close"] > self._delay(self.data["close"], 1), self.data["volume"], 0.0), index=self.data.index), 26)
        down_volume = self._sum(pd.Series(np.where(self.data["close"] <= self._delay(self.data["close"], 1), self.data["volume"], 0.0), index=self.data.index), 26)
        return up_volume / (down_volume + self.eps) * 100.0

    def alpha_41(self) -> pd.Series:
        return -self._rank(self._ts_max(self._delta(self.data["vwap"], 3), 5))

    def alpha_42(self) -> pd.Series:
        return -(self._rank(self._stddev(self.data["high"], 10)) * self._correlation(self.data["high"], self.data["volume"], 10))

    def alpha_43(self) -> pd.Series:
        return self._sum(
            pd.Series(
                np.where(
                    self.data["close"] > self._delay(self.data["close"], 1),
                    self.data["volume"],
                    np.where(self.data["close"] < self._delay(self.data["close"], 1), -self.data["volume"], 0.0),
                ),
                index=self.data.index,
            ),
            6,
        )

    def alpha_44(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._correlation(self.data["low"], self._mean(self.data["volume"], 10), 7), 6), 4)
        right = self._ts_rank(self._decay_linear(self._delta(self.data["vwap"], 3), 10), 15)
        return left + right

    def alpha_45(self) -> pd.Series:
        return self._rank(self._delta((self.data["close"] * 0.6) + (self.data["open"] * 0.4), 1)) * self._rank(self._correlation(self.data["vwap"], self._mean(self.data["volume"], 150), 15))

    def alpha_46(self) -> pd.Series:
        return (self._mean(self.data["close"], 3) + self._mean(self.data["close"], 6) + self._mean(self.data["close"], 12) + self._mean(self.data["close"], 24)) / (4.0 * self.data["close"] + self.eps)

    def alpha_47(self) -> pd.Series:
        return self._sma((self._ts_max(self.data["high"], 6) - self.data["close"]) / (self._ts_max(self.data["high"], 6) - self._ts_min(self.data["low"], 6) + self.eps) * 100.0, 9, 1)

    def alpha_48(self) -> pd.Series:
        sign_chain = np.sign((self.data["close"] - self._delay(self.data["close"], 1)) + (self._delay(self.data["close"], 1) - self._delay(self.data["close"], 2)) + (self._delay(self.data["close"], 2) - self._delay(self.data["close"], 3)))
        numerator = self._rank(sign_chain) * self._sum(self.data["volume"], 5)
        denominator = self._sum(self.data["volume"], 20) + self.eps
        return -(numerator / denominator)

    def alpha_49(self) -> pd.Series:
        prev_high = self._delay(self.data["high"], 1)
        prev_low = self._delay(self.data["low"], 1)
        prev_sum = prev_high + prev_low
        up = pd.Series(
            np.where(
                (self.data["high"] + self.data["low"]) >= prev_sum,
                0.0,
                np.maximum(np.abs(self.data["high"] - prev_high), np.abs(self.data["low"] - prev_low)),
            ),
            index=self.data.index,
        )
        down = pd.Series(
            np.where(
                (self.data["high"] + self.data["low"]) <= prev_sum,
                0.0,
                np.maximum(np.abs(self.data["high"] - prev_high), np.abs(self.data["low"] - prev_low)),
            ),
            index=self.data.index,
        )
        return self._sum(up, 12) / (self._sum(up, 12) + self._sum(down, 12) + self.eps)

    def alpha_50(self) -> pd.Series:
        prev_high = self._delay(self.data["high"], 1)
        prev_low = self._delay(self.data["low"], 1)
        prev_sum = prev_high + prev_low
        up = pd.Series(
            np.where(
                (self.data["high"] + self.data["low"]) <= prev_sum,
                0.0,
                np.maximum(np.abs(self.data["high"] - prev_high), np.abs(self.data["low"] - prev_low)),
            ),
            index=self.data.index,
        )
        down = pd.Series(
            np.where(
                (self.data["high"] + self.data["low"]) >= prev_sum,
                0.0,
                np.maximum(np.abs(self.data["high"] - prev_high), np.abs(self.data["low"] - prev_low)),
            ),
            index=self.data.index,
        )
        return self._sum(up, 12) / (self._sum(up, 12) + self._sum(down, 12) + self.eps) - self._sum(down, 12) / (self._sum(up, 12) + self._sum(down, 12) + self.eps)

    def alpha_51(self) -> pd.Series:
        prev_high = self._delay(self.data["high"], 1)
        prev_low = self._delay(self.data["low"], 1)
        prev_sum = prev_high + prev_low
        up = pd.Series(
            np.where(
                (self.data["high"] + self.data["low"]) <= prev_sum,
                0.0,
                np.maximum(np.abs(self.data["high"] - prev_high), np.abs(self.data["low"] - prev_low)),
            ),
            index=self.data.index,
        )
        down = pd.Series(
            np.where(
                (self.data["high"] + self.data["low"]) >= prev_sum,
                0.0,
                np.maximum(np.abs(self.data["high"] - prev_high), np.abs(self.data["low"] - prev_low)),
            ),
            index=self.data.index,
        )
        return self._sum(up, 12) / (self._sum(up, 12) + self._sum(down, 12) + self.eps)

    def alpha_52(self) -> pd.Series:
        typical = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        up = self._sum(np.maximum(0.0, self.data["high"] - self._delay(typical, 1)), 26)
        down = self._sum(np.maximum(0.0, self._delay(typical, 1) - self.data["low"]), 26)
        return up / (down + self.eps) * 100.0

    def alpha_53(self) -> pd.Series:
        return self._count(self.data["close"] > self._delay(self.data["close"], 1), 12) / 12.0 * 100.0

    def alpha_54(self) -> pd.Series:
        return -self._rank(self._stddev((self.data["close"] - self.data["open"]).abs(), 20) + (self.data["close"] - self.data["open"]) + self._correlation(self.data["close"], self.data["open"], 10))

    def alpha_55(self) -> pd.Series:
        high_delay_close = (self.data["high"] - self._delay(self.data["close"], 1)).abs()
        low_delay_close = (self.data["low"] - self._delay(self.data["close"], 1)).abs()
        high_delay_low = (self.data["high"] - self._delay(self.data["low"], 1)).abs()
        delay_close_delay_open = (self._delay(self.data["close"], 1) - self._delay(self.data["open"], 1)).abs()
        cond1 = (high_delay_close > low_delay_close) & (high_delay_close > high_delay_low)
        cond2 = (low_delay_close > high_delay_low) & (low_delay_close > high_delay_close)
        cond3 = (high_delay_low >= high_delay_close) & (high_delay_low >= low_delay_close)
        part0 = 16.0 * (self.data["close"] + (self.data["close"] - self.data["open"]) / 2.0 - self._delay(self.data["open"], 1))
        part1 = pd.Series(np.nan, index=self.data.index, dtype=float)
        part1[cond1] = high_delay_close[cond1] + low_delay_close[cond1] / 2.0 + delay_close_delay_open[cond1] / 4.0
        part1[cond2] = low_delay_close[cond2] + high_delay_close[cond2] / 2.0 + delay_close_delay_open[cond2] / 4.0
        part1[cond3] = high_delay_low[cond3] + delay_close_delay_open[cond3] / 4.0
        part2 = pd.concat([high_delay_close, low_delay_close], axis=1).max(axis=1)
        return self._sum((part0 / (part1 + self.eps)) * part2, 20)

    def alpha_56(self) -> pd.Series:
        left = self._rank(self.data["open"] - self._ts_min(self.data["open"], 12))
        corr_term = self._correlation(self._sum((self.data["high"] + self.data["low"]) / 2.0, 19), self._sum(self._mean(self.data["volume"], 40), 19), 13)
        right = self._rank(self._rank(corr_term).pow(5))
        return self._bool(left < right)

    def alpha_57(self) -> pd.Series:
        return self._sma((self.data["close"] - self._ts_min(self.data["low"], 9)) / (self._ts_max(self.data["high"], 9) - self._ts_min(self.data["low"], 9) + self.eps) * 100.0, 3, 1)

    def alpha_58(self) -> pd.Series:
        return self._count(self.data["close"] > self._delay(self.data["close"], 1), 20) / 20.0 * 100.0

    def alpha_59(self) -> pd.Series:
        prev_close = self._delay(self.data["close"], 1)
        delta = self.data["close"] - prev_close
        piece = delta.where(self.data["close"] != prev_close, 0.0)
        return self._sum(piece.where(self.data["close"] > prev_close, self.data["close"] - np.maximum(self.data["low"], prev_close)), 20)

    def alpha_60(self) -> pd.Series:
        numerator = ((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"]))
        denominator = (self.data["high"] - self.data["low"]).replace(0, np.nan)
        return self._sum((numerator / denominator) * self.data["volume"], 20)

    def alpha_61(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["vwap"], 1), 12))
        right = self._rank(self._decay_linear(self._rank(self._correlation(self.data["low"], self._mean(self.data["volume"], 80), 8)), 17))
        return -np.maximum(left, right)

    def alpha_62(self) -> pd.Series:
        return -self._correlation(self.data["high"], self._rank(self.data["volume"]), 5)

    def alpha_63(self) -> pd.Series:
        up = self._sma(np.maximum(self.data["close"] - self._delay(self.data["close"], 1), 0.0), 6, 1)
        down = self._sma(np.abs(self.data["close"] - self._delay(self.data["close"], 1)), 6, 1)
        return up / (down + self.eps) * 100.0

    def alpha_64(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 4), 4))
        right = self._rank(self._decay_linear(self._ts_max(self._correlation(self._rank(self.data["close"]), self._rank(self._mean(self.data["volume"], 60)), 4), 13), 14))
        return -np.maximum(left, right)

    def alpha_65(self) -> pd.Series:
        return self._mean(self.data["close"], 6) / (self.data["close"] + self.eps)

    def alpha_66(self) -> pd.Series:
        mean6 = self._mean(self.data["close"], 6)
        return (self.data["close"] - mean6) / (mean6 + self.eps) * 100.0

    def alpha_67(self) -> pd.Series:
        up = self._sma(np.maximum(self.data["close"] - self._delay(self.data["close"], 1), 0.0), 24, 1)
        down = self._sma(np.abs(self.data["close"] - self._delay(self.data["close"], 1)), 24, 1)
        return up / (down + self.eps) * 100.0

    def alpha_68(self) -> pd.Series:
        return self._sma(
            (((self.data["high"] + self.data["low"]) / 2.0) - ((self._delay(self.data["high"], 1) + self._delay(self.data["low"], 1)) / 2.0))
            * (self.data["high"] - self.data["low"])
            / (self.data["volume"] + self.eps),
            15,
            2,
        )

    def alpha_69(self) -> pd.Series:
        dtm = pd.Series(np.where(self.data["open"] <= self._delay(self.data["open"], 1), 0.0, np.maximum(self.data["high"] - self.data["open"], self.data["open"] - self._delay(self.data["open"], 1))), index=self.data.index)
        dbm = pd.Series(np.where(self.data["open"] >= self._delay(self.data["open"], 1), 0.0, np.maximum(self.data["open"] - self.data["low"], self.data["open"] - self._delay(self.data["open"], 1))), index=self.data.index)
        sum_dtm = self._sum(dtm, 20)
        sum_dbm = self._sum(dbm, 20)
        result = pd.Series(np.nan, index=self.data.index, dtype=float)
        gt = sum_dtm > sum_dbm
        lt = sum_dtm < sum_dbm
        eq = sum_dtm == sum_dbm
        result[gt] = (sum_dtm[gt] - sum_dbm[gt]) / (sum_dtm[gt] + self.eps)
        result[eq] = 0.0
        result[lt] = (sum_dtm[lt] - sum_dbm[lt]) / (sum_dbm[lt] + self.eps)
        return result

    def alpha_70(self) -> pd.Series:
        return self._stddev(self.data["amount"], 6)

    def alpha_71(self) -> pd.Series:
        mean24 = self._mean(self.data["close"], 24)
        return (self.data["close"] - mean24) / (mean24 + self.eps) * 100.0

    def alpha_72(self) -> pd.Series:
        return self._sma((self._ts_max(self.data["high"], 6) - self.data["close"]) / (self._ts_max(self.data["high"], 6) - self._ts_min(self.data["low"], 6) + self.eps) * 100.0, 15, 1)

    def alpha_73(self) -> pd.Series:
        left = self._ts_rank(self._decay_linear(self._decay_linear(self._correlation(self.data["close"], self.data["volume"], 10), 16), 4), 5)
        right = self._rank(self._decay_linear(self._correlation(self.data["vwap"], self._mean(self.data["volume"], 30), 4), 3))
        return -(left - right)

    def alpha_74(self) -> pd.Series:
        left = self._rank(self._correlation(self._sum((self.data["low"] * 0.35) + (self.data["vwap"] * 0.65), 20), self._sum(self._mean(self.data["volume"], 40), 20), 7))
        right = self._rank(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 6))
        return left + right

    def alpha_75(self) -> pd.Series:
        benchmark_open = self.data.get("index_open")
        benchmark_close = self.data.get("index_close")
        if benchmark_open is None or benchmark_close is None:
            return pd.Series(np.nan, index=self.data.index, dtype=float)
        numerator = self._count((self.data["close"] > self.data["open"]) & (benchmark_close < benchmark_open), 50)
        denominator = self._count(benchmark_close < benchmark_open, 50)
        return numerator / (denominator + self.eps)

    def alpha_76(self) -> pd.Series:
        return self._stddev(np.abs(self.data["close"] / self._delay(self.data["close"], 1) - 1.0) / (self.data["volume"] + self.eps), 20) / (self._mean(np.abs(self.data["close"] / self._delay(self.data["close"], 1) - 1.0) / (self.data["volume"] + self.eps), 20) + self.eps)

    def alpha_77(self) -> pd.Series:
        left = self._rank(self._decay_linear((((self.data["high"] + self.data["low"]) / 2.0) + self.data["high"] - (self.data["vwap"] + self.data["high"])), 20))
        right = self._rank(self._decay_linear(self._correlation(((self.data["high"] + self.data["low"]) / 2.0), self._mean(self.data["volume"], 40), 3), 6))
        return np.minimum(left, right)

    def alpha_78(self) -> pd.Series:
        typical = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        return (typical - self._mean(typical, 12)) / (0.015 * self._mean(np.abs(self.data["close"] - self._mean(typical, 12)), 12) + self.eps)

    def alpha_79(self) -> pd.Series:
        up = self._sma(np.maximum(self.data["close"] - self._delay(self.data["close"], 1), 0.0), 12, 1)
        down = self._sma(np.abs(self.data["close"] - self._delay(self.data["close"], 1)), 12, 1)
        return up / (down + self.eps) * 100.0

    def alpha_80(self) -> pd.Series:
        return (self.data["volume"] - self._delay(self.data["volume"], 5)) / (self._delay(self.data["volume"], 5) + self.eps) * 100.0

    def alpha_81(self) -> pd.Series:
        return self._sma(self.data["volume"], 21, 2)

    def alpha_82(self) -> pd.Series:
        return self._sma((self._ts_max(self.data["high"], 6) - self.data["close"]) / (self._ts_max(self.data["high"], 6) - self._ts_min(self.data["low"], 6) + self.eps) * 100.0, 20, 1)

    def alpha_83(self) -> pd.Series:
        return -self._rank(self._covariance(self._rank(self.data["high"]), self._rank(self.data["volume"]), 5))

    def alpha_84(self) -> pd.Series:
        part = pd.Series(np.where(self.data["close"] > self._delay(self.data["close"], 1), self.data["volume"], np.where(self.data["close"] < self._delay(self.data["close"], 1), -self.data["volume"], 0.0)), index=self.data.index)
        return self._sum(part, 20)

    def alpha_85(self) -> pd.Series:
        return self._ts_rank(self.data["volume"] / (self._mean(self.data["volume"], 20) + self.eps), 20) * self._ts_rank(-self._delta(self.data["close"], 7), 8)

    def alpha_86(self) -> pd.Series:
        a = ((self._delay(self.data["close"], 20) - self._delay(self.data["close"], 10)) / 10.0) - ((self._delay(self.data["close"], 10) - self.data["close"]) / 10.0)
        result = pd.Series(np.nan, index=self.data.index, dtype=float)
        result[a > 0.25] = -1.0
        result[a < 0.0] = 1.0
        mid = (a >= 0.0) & (a <= 0.25)
        result[mid] = -1.0 * (self.data["close"] - self._delay(self.data["close"], 1))
        return result

    def alpha_87(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["vwap"], 4), 7))
        right = self._ts_rank(self._decay_linear((((self.data["low"] * 1.0) - self.data["vwap"]) / (self.data["open"] - ((self.data["high"] + self.data["low"]) / 2.0) + self.eps)), 11), 7)
        return -(left + right)

    def alpha_88(self) -> pd.Series:
        return (self.data["close"] - self._delay(self.data["close"], 20)) / (self._delay(self.data["close"], 20) + self.eps) * 100.0

    def alpha_89(self) -> pd.Series:
        a = self._sma(self.data["close"], 13, 2)
        b = self._sma(self.data["close"], 27, 2)
        return 2 * (a - b - self._sma(a - b, 10, 2))

    def alpha_90(self) -> pd.Series:
        return -self._rank(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 5))

    def alpha_91(self) -> pd.Series:
        return -(self._rank(self.data["close"] - self._ts_max(self.data["close"], 5)) * self._rank(self._correlation(self._mean(self.data["volume"], 40), self.data["low"], 5)))

    def alpha_92(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["close"] * 0.35 + self.data["vwap"] * 0.65, 2), 3))
        right = self._ts_rank(self._decay_linear(np.abs(self._correlation(self._mean(self.data["volume"], 180), self.data["close"], 13)), 5), 15)
        return -np.maximum(left, right)

    def alpha_93(self) -> pd.Series:
        part = pd.Series(np.where(self.data["open"] >= self._delay(self.data["open"], 1), 0.0, np.maximum(self.data["open"] - self.data["low"], self.data["open"] - self._delay(self.data["open"], 1))), index=self.data.index)
        return self._sum(part, 20)

    def alpha_94(self) -> pd.Series:
        part = pd.Series(np.where(self.data["close"] > self._delay(self.data["close"], 1), self.data["volume"], np.where(self.data["close"] < self._delay(self.data["close"], 1), -self.data["volume"], 0.0)), index=self.data.index)
        return self._sum(part, 30)

    def alpha_95(self) -> pd.Series:
        return self._stddev(self.data["amount"], 20)

    def alpha_96(self) -> pd.Series:
        return self._sma(self._sma((self.data["close"] - self._ts_min(self.data["low"], 9)) / (self._ts_max(self.data["high"], 9) - self._ts_min(self.data["low"], 9) + self.eps) * 100.0, 3, 1), 3, 1)

    def alpha_97(self) -> pd.Series:
        return self._stddev(self.data["volume"], 10)

    def alpha_98(self) -> pd.Series:
        cond = self._delta(self._sum(self.data["close"], 100) / 100.0, 100) / (self._delay(self.data["close"], 100) + self.eps) <= 0.05
        result = pd.Series(np.nan, index=self.data.index, dtype=float)
        result[cond] = -(self.data["close"] - self._ts_min(self.data["close"], 100))
        result[~cond] = -self._delta(self.data["close"], 3)[~cond]
        return result

    def alpha_99(self) -> pd.Series:
        return -self._rank(self._covariance(self._rank(self.data["close"]), self._rank(self.data["volume"]), 5))

    def alpha_100(self) -> pd.Series:
        return self._stddev(self.data["volume"], 20)

    def alpha_101(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["close"], self._sum(self._mean(self.data["volume"], 30), 37), 15))
        right = self._rank(self._correlation(self._rank(self.data["high"] * 0.1 + self.data["vwap"] * 0.9), self._rank(self.data["volume"]), 11))
        return pd.Series(np.where(left < right, -1.0, 0.0), index=self.data.index)

    def alpha_102(self) -> pd.Series:
        return self._sma(np.maximum(self.data["volume"] - self._delay(self.data["volume"], 1), 0.0), 6, 1) / (self._sma(np.abs(self.data["volume"] - self._delay(self.data["volume"], 1)), 6, 1) + self.eps) * 100.0

    def alpha_103(self) -> pd.Series:
        return ((20 - self._lowday(self.data["low"], 20)) / 20.0) * 100.0

    def alpha_104(self) -> pd.Series:
        return -(self._delta(self._correlation(self.data["high"], self.data["volume"], 5), 5) * self._rank(self._stddev(self.data["close"], 20)))

    def alpha_105(self) -> pd.Series:
        return -self._correlation(self._rank(self.data["open"]), self._rank(self.data["volume"]), 10)

    def alpha_106(self) -> pd.Series:
        return self.data["close"] - self._delay(self.data["close"], 20)

    def alpha_107(self) -> pd.Series:
        return (-self._rank(self.data["open"] - self._delay(self.data["high"], 1)) * self._rank(self.data["open"] - self._delay(self.data["close"], 1)) * self._rank(self.data["open"] - self._delay(self.data["low"], 1)))

    def alpha_108(self) -> pd.Series:
        return -(self._rank(self.data["high"] - self._ts_min(self.data["high"], 2)) ** self._rank(self._correlation(self.data["vwap"], self._mean(self.data["volume"], 120), 6)))

    def alpha_109(self) -> pd.Series:
        return self._sma(self.data["high"] - self.data["low"], 10, 2) / (self._sma(self._sma(self.data["high"] - self.data["low"], 10, 2), 10, 2) + self.eps)

    def alpha_110(self) -> pd.Series:
        return self._sum(np.maximum(self.data["high"] - self._delay(self.data["close"], 1), 0.0), 20) / (self._sum(np.maximum(self._delay(self.data["close"], 1) - self.data["low"], 0.0), 20) + self.eps) * 100.0

    def alpha_111(self) -> pd.Series:
        numerator = self.data["volume"] * ((self.data["close"] - self.data["low"]) - (self.data["high"] - self.data["close"])) / (self.data["high"] - self.data["low"] + self.eps)
        return self._sma(numerator, 11, 2) - self._sma(numerator, 4, 2)

    def alpha_112(self) -> pd.Series:
        change = self.data["close"] - self._delay(self.data["close"], 1)
        up = pd.Series(np.where(change > 0, change, 0.0), index=self.data.index)
        down = pd.Series(np.where(change < 0, np.abs(change), 0.0), index=self.data.index)
        return (self._sum(up, 12) - self._sum(down, 12)) / (self._sum(up, 12) + self._sum(down, 12) + self.eps) * 100.0

    def alpha_113(self) -> pd.Series:
        return -(self._rank(self._sum(self._delay(self.data["close"], 5), 20) / 20.0) * self._correlation(self.data["close"], self.data["volume"], 2) * self._rank(self._correlation(self._sum(self.data["close"], 5), self._sum(self.data["close"], 20), 2)))

    def alpha_114(self) -> pd.Series:
        base = (self.data["high"] - self.data["low"]) / (self._sum(self.data["close"], 5) / 5.0 + self.eps)
        return (self._rank(self._delay(base, 2)) * self._rank(self._rank(self.data["volume"]))) / (base / (self.data["vwap"] - self.data["close"] + self.eps))

    def alpha_115(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["high"] * 0.9 + self.data["close"] * 0.1, self._mean(self.data["volume"], 30), 10))
        right = self._rank(self._correlation(self._ts_rank((self.data["high"] + self.data["low"]) / 2.0, 4), self._ts_rank(self.data["volume"], 10), 7))
        return left ** right

    def alpha_116(self) -> pd.Series:
        return self._regbeta_to_sequence(self.data["close"], 20)

    def alpha_117(self) -> pd.Series:
        return self._ts_rank(self.data["volume"], 32) * (1 - self._ts_rank(self.data["close"] + self.data["high"] - self.data["low"], 16)) * (1 - self._ts_rank(self.data["return"], 32))

    def alpha_118(self) -> pd.Series:
        return self._sum(self.data["high"] - self.data["open"], 20) / (self._sum(self.data["open"] - self.data["low"], 20) + self.eps) * 100.0

    def alpha_119(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._correlation(self.data["vwap"], self._sum(self._mean(self.data["volume"], 5), 26), 5), 7))
        right = self._rank(self._decay_linear(self._ts_rank(self._ts_min(self._correlation(self._rank(self.data["open"]), self._rank(self._mean(self.data["volume"], 15)), 21), 9), 7), 8))
        return left - right

    def alpha_120(self) -> pd.Series:
        return self._rank(self.data["vwap"] - self.data["close"]) / (self._rank(self.data["vwap"] + self.data["close"]) + self.eps)

    def alpha_121(self) -> pd.Series:
        return -(self._rank(self.data["vwap"] - self._ts_min(self.data["vwap"], 12)) ** self._ts_rank(self._correlation(self._ts_rank(self.data["vwap"], 20), self._ts_rank(self._mean(self.data["volume"], 60), 2), 18), 3))

    def alpha_122(self) -> pd.Series:
        sma1 = self._sma(np.log(self.data["close"]), 13, 2)
        sma2 = self._sma(sma1, 13, 2)
        sma3 = self._sma(sma2, 13, 2)
        return (sma3 - self._delay(sma3, 1)) / (self._delay(sma3, 1) + self.eps)

    def alpha_123(self) -> pd.Series:
        left = self._rank(self._correlation(self._sum((self.data["high"] + self.data["low"]) / 2.0, 20), self._sum(self._mean(self.data["volume"], 60), 20), 9))
        right = self._rank(self._correlation(self.data["low"], self.data["volume"], 6))
        return pd.Series(np.where(left < right, -1.0, 0.0), index=self.data.index)

    def alpha_124(self) -> pd.Series:
        return (self.data["close"] - self.data["vwap"]) / (self._decay_linear(self._rank(self._ts_max(self.data["close"], 30)), 2) + self.eps)

    def alpha_125(self) -> pd.Series:
        return self._rank(self._decay_linear(self._correlation(self.data["vwap"], self._mean(self.data["volume"], 80), 17), 20)) / (self._rank(self._decay_linear(self._delta(self.data["close"] * 0.5 + self.data["vwap"] * 0.5, 3), 16)) + self.eps)

    def alpha_126(self) -> pd.Series:
        return (self.data["close"] + self.data["high"] + self.data["low"]) / 3.0

    def alpha_127(self) -> pd.Series:
        return np.sqrt(self._mean((100.0 * (self.data["close"] - self._ts_max(self.data["close"], 12)) / (self._ts_max(self.data["close"], 12) + self.eps)) ** 2, 12))

    def alpha_128(self) -> pd.Series:
        typical = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        up = pd.Series(np.where(typical > self._delay(typical, 1), typical * self.data["volume"], 0.0), index=self.data.index)
        down = pd.Series(np.where(typical < self._delay(typical, 1), typical * self.data["volume"], 0.0), index=self.data.index)
        return 100.0 - (100.0 / (1.0 + self._sum(up, 14) / (self._sum(down, 14) + self.eps)))

    def alpha_129(self) -> pd.Series:
        change = self.data["close"] - self._delay(self.data["close"], 1)
        part = pd.Series(np.where(change < 0, np.abs(change), 0.0), index=self.data.index)
        return self._sum(part, 12)

    def alpha_130(self) -> pd.Series:
        return self._rank(self._decay_linear(self._correlation(((self.data["high"] + self.data["low"]) / 2.0), self._mean(self.data["volume"], 40), 9), 10)) / (self._rank(self._decay_linear(self._correlation(self._rank(self.data["vwap"]), self._rank(self.data["volume"]), 7), 3)) + self.eps)

    def alpha_131(self) -> pd.Series:
        return self._rank(self._delta(self.data["vwap"], 1)) ** self._ts_rank(self._correlation(self.data["close"], self._mean(self.data["volume"], 50), 18), 18)

    def alpha_132(self) -> pd.Series:
        return self._mean(self.data["amount"], 20)

    def alpha_133(self) -> pd.Series:
        return ((20 - self._highday(self.data["high"], 20)) / 20.0) * 100.0 - ((20 - self._lowday(self.data["low"], 20)) / 20.0) * 100.0

    def alpha_134(self) -> pd.Series:
        return (self.data["close"] - self._delay(self.data["close"], 12)) / (self._delay(self.data["close"], 12) + self.eps) * self.data["volume"]

    def alpha_135(self) -> pd.Series:
        return self._sma(self._delay(self.data["close"] / (self._delay(self.data["close"], 20) + self.eps), 1), 20, 1)

    def alpha_136(self) -> pd.Series:
        return -self._rank(self._delta(self.data["return"], 3)) * self._correlation(self.data["open"], self.data["volume"], 10)

    def alpha_137(self) -> pd.Series:
        a = np.abs(self.data["high"] - self._delay(self.data["close"], 1))
        b = np.abs(self.data["low"] - self._delay(self.data["close"], 1))
        c = np.abs(self.data["high"] - self._delay(self.data["low"], 1))
        d = np.abs(self._delay(self.data["close"], 1) - self._delay(self.data["open"], 1))
        cond1 = (a > b) & (a > c)
        cond2 = (b > c) & (b > a)
        denom = pd.Series(np.nan, index=self.data.index, dtype=float)
        denom[cond1] = a[cond1] + b[cond1] / 2.0 + d[cond1] / 4.0
        denom[cond2] = b[cond2] + a[cond2] / 2.0 + d[cond2] / 4.0
        denom[~cond1 & ~cond2] = c[~cond1 & ~cond2] + d[~cond1 & ~cond2] / 4.0
        numer = 16.0 * (self.data["close"] - self._delay(self.data["close"], 1) + (self.data["close"] - self.data["open"]) / 2.0 + self._delay(self.data["close"], 1) - self._delay(self.data["open"], 1))
        return numer / (denom * np.maximum(a, b) + self.eps)

    def alpha_138(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["low"] * 0.7 + self.data["vwap"] * 0.3, 3), 20))
        right = self._ts_rank(self._decay_linear(self._ts_rank(self._correlation(self._ts_rank(self.data["low"], 8), self._ts_rank(self._mean(self.data["volume"], 60), 17), 5), 19), 16), 7)
        return -(left - right)

    def alpha_139(self) -> pd.Series:
        return -self._correlation(self.data["open"], self.data["volume"], 10)

    def alpha_140(self) -> pd.Series:
        left = self._rank(self._decay_linear((self._rank(self.data["open"]) + self._rank(self.data["low"])) - (self._rank(self.data["high"]) + self._rank(self.data["close"])), 8))
        right = self._ts_rank(self._decay_linear(self._correlation(self._ts_rank(self.data["close"], 8), self._ts_rank(self._mean(self.data["volume"], 60), 20), 8), 7), 3)
        return np.minimum(left, right)

    def alpha_141(self) -> pd.Series:
        return -self._rank(self._correlation(self._rank(self.data["high"]), self._rank(self._mean(self.data["volume"], 15)), 9))

    def alpha_142(self) -> pd.Series:
        return (-self._rank(self._ts_rank(self.data["close"], 10)) * self._rank(self._delta(self._delta(self.data["close"], 1), 1)) * self._rank(self._ts_rank(self.data["volume"] / (self._mean(self.data["volume"], 20) + self.eps), 5)))

    def alpha_143(self) -> pd.Series:
        return pd.Series(np.nan, index=self.data.index, dtype=float)

    def alpha_144(self) -> pd.Series:
        cond = self.data["close"] < self._delay(self.data["close"], 1)
        part = (np.abs(self.data["close"] / (self._delay(self.data["close"], 1) + self.eps) - 1.0)) / (self.data["amount"] + self.eps)
        return self._sumif(part, cond, 20) / (self._count(cond, 20) + self.eps)

    def alpha_145(self) -> pd.Series:
        return (self._mean(self.data["volume"], 9) - self._mean(self.data["volume"], 26)) / (self._mean(self.data["volume"], 12) + self.eps) * 100.0

    def alpha_146(self) -> pd.Series:
        ret = (self.data["close"] - self._delay(self.data["close"], 1)) / (self._delay(self.data["close"], 1) + self.eps)
        sma_ret = self._sma(ret, 61, 2)
        centered = ret - sma_ret
        return self._mean(centered, 20) * centered / (self._sma(centered ** 2, 61, 2) + self.eps)

    def alpha_147(self) -> pd.Series:
        return self._regbeta_to_sequence(self._mean(self.data["close"], 12), 12)

    def alpha_148(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["open"], self._sum(self._mean(self.data["volume"], 60), 9), 6))
        right = self._rank(self.data["open"] - self._ts_min(self.data["open"], 14))
        return pd.Series(np.where(left < right, -1.0, 0.0), index=self.data.index)

    def alpha_149(self) -> pd.Series:
        benchmark_open = self.data.get("index_open")
        benchmark_close = self.data.get("index_close")
        if benchmark_open is None or benchmark_close is None:
            return pd.Series(np.nan, index=self.data.index, dtype=float)
        cond = benchmark_close < self._delay(benchmark_close, 1)
        stock_ret = self.data["close"] / (self._delay(self.data["close"], 1) + self.eps) - 1.0
        bench_ret = benchmark_close / (self._delay(benchmark_close, 1) + self.eps) - 1.0
        stock = stock_ret.where(cond, np.nan)
        bench = bench_ret.where(cond, np.nan)
        return self._regbeta(stock, bench, 252)

    def alpha_150(self) -> pd.Series:
        return (self.data["close"] + self.data["high"] + self.data["low"]) / 3.0 * self.data["volume"]

    def alpha_151(self) -> pd.Series:
        return self._sma(self.data["close"] - self._delay(self.data["close"], 20), 20, 1)

    def alpha_152(self) -> pd.Series:
        inner = self._sma(self._delay(self.data["close"] / (self._delay(self.data["close"], 9) + self.eps), 1), 9, 1)
        return self._sma(self._mean(self._delay(inner, 1), 12) - self._mean(self._delay(inner, 1), 26), 9, 1)

    def alpha_153(self) -> pd.Series:
        return (self._mean(self.data["close"], 3) + self._mean(self.data["close"], 6) + self._mean(self.data["close"], 12) + self._mean(self.data["close"], 24)) / 4.0

    def alpha_154(self) -> pd.Series:
        return pd.Series(np.where((self.data["vwap"] - self._ts_min(self.data["vwap"], 16)) < self._correlation(self.data["vwap"], self._mean(self.data["volume"], 180), 18), 1.0, 0.0), index=self.data.index)

    def alpha_155(self) -> pd.Series:
        a = self._sma(self.data["volume"], 13, 2)
        b = self._sma(self.data["volume"], 27, 2)
        return a - b - self._sma(a - b, 10, 2)

    def alpha_156(self) -> pd.Series:
        left = self._rank(self._decay_linear(self._delta(self.data["vwap"], 5), 3))
        base = ((self._delta(self.data["open"] * 0.15 + self.data["low"] * 0.85, 2) / (self.data["open"] * 0.15 + self.data["low"] * 0.85 + self.eps)) * -1.0)
        right = self._rank(self._decay_linear(base, 3))
        return -np.maximum(left, right)

    def alpha_157(self) -> pd.Series:
        inner = -1 * self._rank(self._delta(self.data["close"] - 1, 5))
        term = self._ts_min(self._rank(self._rank(self._sum(np.maximum(self._rank(self._rank(np.log(self._sum(inner, 2) + self.eps))), 1), 1))), 5)
        return self._ts_min(self._product(self._rank(self._rank(np.log(self._sum(term, 1) + self.eps))), 1), 5) + self._ts_rank(self._delay(-1 * self.data["return"], 6), 5)

    def alpha_158(self) -> pd.Series:
        return ((self.data["high"] - self._sma(self.data["close"], 15, 2)) - (self.data["low"] - self._sma(self.data["close"], 15, 2))) / (self.data["close"] + self.eps)

    def alpha_159(self) -> pd.Series:
        base1 = (self.data["close"] - self._sum(self._sum(np.minimum(self.data["low"], self._delay(self.data["close"], 1)), 6), 1)) / (self._sum(np.maximum(self.data["high"], self._delay(self.data["close"], 1)) - np.minimum(self.data["low"], self._delay(self.data["close"], 1)), 6) + self.eps)
        base2 = (self.data["close"] - self._sum(self._sum(np.minimum(self.data["low"], self._delay(self.data["close"], 1)), 12), 1)) / (self._sum(np.maximum(self.data["high"], self._delay(self.data["close"], 1)) - np.minimum(self.data["low"], self._delay(self.data["close"], 1)), 12) + self.eps)
        base3 = (self.data["close"] - self._sum(self._sum(np.minimum(self.data["low"], self._delay(self.data["close"], 1)), 24), 1)) / (self._sum(np.maximum(self.data["high"], self._delay(self.data["close"], 1)) - np.minimum(self.data["low"], self._delay(self.data["close"], 1)), 24) + self.eps)
        return (base1 * 12 * 24 + base2 * 6 * 24 + base3 * 6 * 24) * 100.0 / (6 * 12 + 6 * 24 + 12 * 24)

    def alpha_160(self) -> pd.Series:
        return self._sma(pd.Series(np.where(self.data["close"] <= self._delay(self.data["close"], 1), self._stddev(self.data["close"], 20), 0.0), index=self.data.index), 20, 1)

    def alpha_161(self) -> pd.Series:
        return self._mean(np.maximum(np.maximum(self.data["high"] - self.data["low"], np.abs(self._delay(self.data["close"], 1) - self.data["high"])), np.abs(self._delay(self.data["close"], 1) - self.data["low"])), 12)

    def alpha_162(self) -> pd.Series:
        base = self._sma(np.maximum(self.data["close"] - self._delay(self.data["close"], 1), 0.0), 12, 1) / (self._sma(np.abs(self.data["close"] - self._delay(self.data["close"], 1)), 12, 1) + self.eps) * 100.0
        return (base - self._ts_min(base, 12)) / (self._ts_max(base, 12) - self._ts_min(base, 12) + self.eps)

    def alpha_163(self) -> pd.Series:
        return self._rank((-1 * self.data["return"]) * self._mean(self.data["volume"], 20) * self.data["vwap"] * (self.data["high"] - self.data["close"]))

    def alpha_164(self) -> pd.Series:
        cond = self.data["close"] > self._delay(self.data["close"], 1)
        base = pd.Series(np.where(cond, 1.0 / (self.data["close"] - self._delay(self.data["close"], 1) + self.eps), 1.0), index=self.data.index)
        return self._sma((base - self._ts_min(base, 12)) / (self.data["high"] - self.data["low"] + self.eps) * 100.0, 13, 2)

    def alpha_165(self) -> pd.Series:
        centered = self.data["close"] - self._mean(self.data["close"], 48)
        rolling = self._sum(centered, 48)
        cross_max = self._cross_sectional_max(rolling)
        cross_min = self._cross_sectional_min(rolling)
        std48 = self._stddev(self.data["close"], 48)
        return -(1.0 / (std48 + self.eps)).div(cross_min + self.eps) - cross_max

    def alpha_166(self) -> pd.Series:
        ret = self.data["close"] / (self._delay(self.data["close"], 1) + self.eps) - 1.0
        centered = ret - self._mean(ret, 20)
        numerator = -20.0 * (20 - 1) ** 1.5 * self._sum(centered, 20)
        denominator = ((20 - 1) * (20 - 2) * (self._sum(ret ** 2, 20)) ** 1.5)
        return numerator / (denominator + self.eps)

    def alpha_167(self) -> pd.Series:
        change = self.data["close"] - self._delay(self.data["close"], 1)
        part = pd.Series(np.where(change > 0, change, 0.0), index=self.data.index)
        return self._sum(part, 12)

    def alpha_168(self) -> pd.Series:
        return -self.data["volume"] / (self._mean(self.data["volume"], 20) + self.eps)

    def alpha_169(self) -> pd.Series:
        inner = self._sma(self.data["close"] - self._delay(self.data["close"], 1), 9, 1)
        return self._sma(self._mean(self._delay(inner, 1), 12) - self._mean(self._delay(inner, 1), 26), 10, 1)

    def alpha_170(self) -> pd.Series:
        left = ((self._rank(1.0 / (self.data["close"] + self.eps)) * self.data["volume"]) / (self._mean(self.data["volume"], 20) + self.eps))
        right = (self.data["high"] * self._rank(self.data["high"] - self.data["close"])) / (self._sum(self.data["high"], 5) / 5.0 + self.eps)
        return (left * right) - self._rank(self.data["vwap"] - self._delay(self.data["vwap"], 5))

    def alpha_171(self) -> pd.Series:
        return (-1.0 * ((self.data["low"] - self.data["close"]) * (self.data["open"] ** 5))) / (((self.data["close"] - self.data["high"]) * (self.data["close"] ** 5)) + self.eps)

    def alpha_172(self) -> pd.Series:
        tr = np.maximum(np.maximum(self.data["high"] - self.data["low"], np.abs(self.data["high"] - self._delay(self.data["close"], 1))), np.abs(self.data["low"] - self._delay(self.data["close"], 1)))
        hd = self.data["high"] - self._delay(self.data["high"], 1)
        ld = self._delay(self.data["low"], 1) - self.data["low"]
        cond1 = (ld > 0) & (ld > hd)
        cond2 = (hd > 0) & (hd > ld)
        part1 = pd.Series(np.where(cond1, ld, 0.0), index=self.data.index)
        part2 = pd.Series(np.where(cond2, hd, 0.0), index=self.data.index)
        term1 = self._sum(part1, 14) * 100.0 / (self._sum(tr, 14) + self.eps)
        term2 = self._sum(part2, 14) * 100.0 / (self._sum(tr, 14) + self.eps)
        return self._mean(np.abs(term1 - term2) / (term1 + term2 + self.eps) * 100.0, 6)

    def alpha_173(self) -> pd.Series:
        return 3 * self._sma(self.data["close"], 13, 2) - 2 * self._sma(self._sma(self.data["close"], 13, 2), 13, 2) + self._sma(self._sma(self._sma(np.log(self.data["close"]), 13, 2), 13, 2), 13, 2)

    def alpha_174(self) -> pd.Series:
        cond = self.data["close"] > self._delay(self.data["close"], 1)
        part = pd.Series(np.where(cond, self._stddev(self.data["close"], 20), 0.0), index=self.data.index)
        return self._sma(part, 20, 1)

    def alpha_175(self) -> pd.Series:
        return self._mean(np.maximum(np.maximum(self.data["high"] - self.data["low"], np.abs(self._delay(self.data["close"], 1) - self.data["high"])), np.abs(self._delay(self.data["close"], 1) - self.data["low"])), 6)

    def alpha_176(self) -> pd.Series:
        return self._correlation(self._rank((self.data["close"] - self._ts_min(self.data["low"], 12)) / (self._ts_max(self.data["high"], 12) - self._ts_min(self.data["low"], 12) + self.eps)), self._rank(self.data["volume"]), 6)

    def alpha_177(self) -> pd.Series:
        return ((20 - self._highday(self.data["high"], 20)) / 20.0) * 100.0

    def alpha_178(self) -> pd.Series:
        return (self.data["close"] - self._delay(self.data["close"], 1)) / (self._delay(self.data["close"], 1) + self.eps) * self.data["volume"]

    def alpha_179(self) -> pd.Series:
        left = self._rank(self._correlation(self.data["vwap"], self.data["volume"], 4))
        right = self._rank(self._correlation(self._rank(self.data["low"]), self._rank(self._mean(self.data["volume"], 50)), 12))
        return left * right

    def alpha_180(self) -> pd.Series:
        cond = self._mean(self.data["volume"], 20) < self.data["volume"]
        result = pd.Series(np.nan, index=self.data.index, dtype=float)
        result[cond] = (-1.0 * self._ts_rank(np.abs(self._delta(self.data["close"], 7)), 60) * self._bool(np.sign(self._delta(self.data["close"], 7))))
        result[~cond] = -self.data["volume"]
        return result

    def alpha_181(self) -> pd.Series:
        if "index_close" not in self.data.columns:
            return pd.Series(np.nan, index=self.data.index, dtype=float)
        stock_ret = self.data["close"] / (self._delay(self.data["close"], 1) + self.eps) - 1.0
        bench_ret = self.data["index_close"] / (self._delay(self.data["index_close"], 1) + self.eps) - 1.0
        return self._sum((stock_ret - self._mean(stock_ret, 20)) - (bench_ret - self._mean(bench_ret, 20)) ** 2, 20) / (self._sum((bench_ret - self._mean(bench_ret, 20)) ** 3, 20) + self.eps)

    def alpha_182(self) -> pd.Series:
        if "index_open" not in self.data.columns or "index_close" not in self.data.columns:
            return pd.Series(np.nan, index=self.data.index, dtype=float)
        cond = ((self.data["close"] > self.data["open"]) & (self.data["index_close"] > self.data["index_open"])) | ((self.data["close"] < self.data["open"]) & (self.data["index_close"] < self.data["index_open"]))
        return self._count(cond, 20) / 20.0

    def alpha_183(self) -> pd.Series:
        rolling = self._sum(self.data["close"] - self._mean(self.data["close"], 24), 24)
        return -(1.0 / (self._stddev(self.data["close"], 24) + self.eps)).div(self._cross_sectional_min(rolling) + self.eps) - self._cross_sectional_max(rolling)

    def alpha_184(self) -> pd.Series:
        return self._rank(self._correlation(self._delay(self.data["open"] - self.data["close"], 1), self.data["close"], 200)) + self._rank(self.data["open"] - self.data["close"])

    def alpha_185(self) -> pd.Series:
        return self._rank(-((1.0 - (self.data["open"] / (self.data["close"] + self.eps))) ** 2))

    def alpha_186(self) -> pd.Series:
        tr = np.maximum(np.maximum(self.data["high"] - self.data["low"], np.abs(self.data["high"] - self._delay(self.data["close"], 1))), np.abs(self.data["low"] - self._delay(self.data["close"], 1)))
        hd = self.data["high"] - self._delay(self.data["high"], 1)
        ld = self._delay(self.data["low"], 1) - self.data["low"]
        cond1 = (ld > 0) & (ld > hd)
        cond2 = (hd > 0) & (hd > ld)
        part1 = pd.Series(np.where(cond1, ld, 0.0), index=self.data.index)
        part2 = pd.Series(np.where(cond2, hd, 0.0), index=self.data.index)
        term1 = self._sum(part1, 14) * 100.0 / (self._sum(tr, 14) + self.eps)
        term2 = self._sum(part2, 14) * 100.0 / (self._sum(tr, 14) + self.eps)
        core = np.abs(term1 - term2) / (term1 + term2 + self.eps) * 100.0
        return (self._mean(core, 6) + self._delay(self._mean(core, 6), 6)) / 2.0

    def alpha_187(self) -> pd.Series:
        part = pd.Series(np.where(self.data["open"] <= self._delay(self.data["open"], 1), 0.0, np.maximum(self.data["high"] - self.data["open"], self.data["open"] - self._delay(self.data["open"], 1))), index=self.data.index)
        return self._sum(part, 20)

    def alpha_188(self) -> pd.Series:
        return ((self.data["high"] - self.data["low"] - self._sma(self.data["high"] - self.data["low"], 11, 2)) / (self._sma(self.data["high"] - self.data["low"], 11, 2) + self.eps)) * 100.0

    def alpha_189(self) -> pd.Series:
        return self._mean(np.abs(self.data["close"] - self._mean(self.data["close"], 6)), 6)

    def alpha_190(self) -> pd.Series:
        change_1 = self.data["close"] / (self._delay(self.data["close"], 1) + self.eps) - 1.0
        change_19 = (self.data["close"] / (self._delay(self.data["close"], 19) + self.eps)) ** (1.0 / 20.0) - 1.0
        cond_low = change_1 < change_19
        cond_high = change_1 > change_19
        numerator = self._count(change_1 > change_19, 20) - 1.0
        lower = self._sumif((change_1 - change_19) ** 2, cond_low, 20)
        upper = self._count(cond_low, 20) * self._sumif((change_1 - change_19) ** 2, cond_high, 20)
        return np.log((numerator + self.eps) * (lower + self.eps) / (upper + self.eps))

    def alpha_191(self) -> pd.Series:
        return self._correlation(self._mean(self.data["volume"], 20), self.data["low"], 5) + (self.data["high"] + self.data["low"]) / 2.0 - self.data["close"]

    def _alpha_methods(self) -> dict[int, Callable[[], pd.Series]]:
        return {i: getattr(self, f"alpha_{i}") for i in self.implemented_alpha_ids()}

    def compute_series(self, alpha_id: int) -> pd.Series:
        if alpha_id < 1 or alpha_id > 191:
            raise ValueError("alpha_id must be in [1, 191]")
        methods = self._alpha_methods()
        if alpha_id not in methods:
            raise NotImplementedError(
                f"gtja191 alpha_{alpha_id:03d} is not implemented yet; "
                f"currently implemented ids are {self.implemented_alpha_ids()}"
            )
        series = methods[alpha_id]().rename(f"alpha_{alpha_id:03d}")
        return series

    def compute(self, alpha_id: int) -> pd.DataFrame:
        factor_name = f"alpha_{alpha_id:03d}"
        result = self._finalize(self.compute_series(alpha_id), factor_name)
        return result

    def compute_all(self, alpha_ids: Iterable[int] | None = None) -> pd.DataFrame:
        selected = list(alpha_ids) if alpha_ids is not None else self.implemented_alpha_ids()
        frame = self.data[["date", "symbol"]].rename(columns={"date": "date_", "symbol": "code"}).copy()
        factor_columns: dict[str, pd.Series] = {}
        for alpha_id in selected:
            factor_name = f"alpha_{alpha_id:03d}"
            factor_columns[factor_name] = pd.to_numeric(self.compute_series(alpha_id), errors="coerce")
        if not factor_columns:
            return frame
        result = pd.concat([frame, pd.DataFrame(factor_columns, index=self.data.index)], axis=1)
        return result

    def compute_matrix(self, alpha_id: int) -> pd.DataFrame:
        factor_name = f"alpha_{alpha_id:03d}"
        result = self.compute(alpha_id)
        return result.pivot(index="date_", columns="code", values=factor_name).sort_index()


def gtja191_factor_names() -> list[str]:
    return [f"alpha_{i:03d}" for i in range(1, 192)]
