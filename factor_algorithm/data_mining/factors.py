from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from tiger_factors.utils import panel_ops as po

@dataclass(frozen=True)
class DataMiningFactorSpec:
    name: str
    description: str
    source: str
    builder: Callable[["DataMiningEngine"], pd.Series]


class DataMiningEngine:
    """Cleaned data-mining factor engine based on the suggested alpha formulas.

    The engine works on long-form OHLCV panels with columns:
    `date_` or `date`, `code` or `symbol`, `open`, `high`, `low`, `close`, `volume`.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = self._prepare_input(data)
        self.eps = 1e-8

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        rename_map = {}
        if "date_" in frame.columns:
            rename_map["date_"] = "date"
        if "code" in frame.columns:
            rename_map["code"] = "symbol"
        frame = frame.rename(columns=rename_map)

        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"data_mining input is missing columns: {sorted(missing)}")

        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame["symbol"] = frame["symbol"].astype(str)
        frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

        for column in ["open", "high", "low", "close", "volume", "vwap"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "vwap" not in frame.columns:
            frame["vwap"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0

        return frame

    def _group_apply(self, series: pd.Series, func: Callable[[pd.Series], pd.Series]) -> pd.Series:
        result = (
            series.groupby(self.data["symbol"], sort=False)
            .transform(func)
            .rename(series.name)
        )
        result.index = series.index
        return result

    def _delay(self, series: pd.Series, periods: int) -> pd.Series:
        return po.delay(self.data, series, periods)

    def _rank(self, series: pd.Series) -> pd.Series:
        return po.cs_rank(self.data, series, date_col="date", pct=True)

    def _ts_sum(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_sum(self.data, series, window)

    def _ts_min(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_min(self.data, series, window)

    def _ts_var(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_var(self.data, series, window)

    def _ts_ema(self, series: pd.Series, span: int) -> pd.Series:
        return po.ts_ema(self.data, series, span)

    def _ts_wma(self, series: pd.Series, window: int) -> pd.Series:
        return po.ts_wma(self.data, series, window)

    def _ts_pctchange(self, series: pd.Series, periods: int) -> pd.Series:
        return po.ts_pctchange(self.data, series, periods)

    def _ts_corr(self, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        return po.ts_corr(self.data, left, right, window)

    def _ts_cov(self, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        return po.ts_cov(self.data, left, right, window)

    def _inv(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        out = 1.0 / numeric.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def _s_log1p(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.Series(np.log1p(numeric), index=series.index)

    def _finalize(self, values: pd.Series, factor_name: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "date_": self.data["date"],
                "code": self.data["symbol"],
                factor_name: pd.to_numeric(values, errors="coerce"),
            }
        )
        return frame.dropna(subset=[factor_name]).reset_index(drop=True)

    def compute(self, name: str) -> pd.DataFrame:
        try:
            builder = getattr(self, name)
        except AttributeError as exc:
            raise KeyError(f"Unknown data mining factor: {name}") from exc
        values = builder()
        if not isinstance(values, pd.Series):
            raise TypeError(f"Factor {name!r} must return a pandas Series.")
        return self._finalize(values, name)

    def compute_all(self, names: list[str] | None = None) -> pd.DataFrame:
        selected = names or available_factors()
        frames: list[pd.DataFrame] = []
        for name in selected:
            frames.append(self.compute(name))
        if not frames:
            return pd.DataFrame(columns=["date_", "code"])
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["date_", "code"], how="outer")
        return merged.sort_values(["date_", "code"]).reset_index(drop=True)

    # Cleaned factors from the suggested list
    def factor_002_intraday_strength(self) -> pd.Series:
        return (self.data["close"] - self.data["open"]) / self.data["open"].replace(0, np.nan)

    def factor_006_price_volume_var40(self) -> pd.Series:
        ratio = self.data["close"] / self._ts_wma(self.data["volume"], 5).replace(0, np.nan)
        return self._ts_var(ratio, 40)

    def factor_007_volume_ema40(self) -> pd.Series:
        return self._ts_ema(self.data["volume"], 40)

    def factor_008_close_low_var10(self) -> pd.Series:
        return self._ts_var(self.data["close"] * self.data["low"], 10)

    def factor_010_vwap_corr_inverse_high_lag10(self) -> pd.Series:
        corr = self._ts_corr(self.data["vwap"], self._inv(self.data["high"]), 30)
        return self._delay(corr, 10)

    def factor_011_vwap_over_low_min10(self) -> pd.Series:
        return 2.0 * self.data["vwap"] / self._ts_min(self.data["low"], 10).replace(0, np.nan)

    def factor_012_volume_sum5(self) -> pd.Series:
        return self._ts_sum(self.data["volume"], 5)

    def factor_013_volume(self) -> pd.Series:
        return self.data["volume"]

    def factor_021_log1p_inverse_turnover_pressure(self) -> pd.Series:
        raw = 10.0 / (self.data["volume"] * (self.data["high"] + 2.0))
        return self._s_log1p(raw)

    def factor_024_close_over_high(self) -> pd.Series:
        return self.data["close"] / self.data["high"].replace(0, np.nan)

    def factor_025_open_high_cov5_wma5(self) -> pd.Series:
        cov = self._ts_cov(self.data["open"], self.data["high"], 5)
        return self._ts_wma(cov, 5)

    def factor_029_vwap_minus_high(self) -> pd.Series:
        return self.data["vwap"] - self.data["high"]

    def factor_031_nested_cov_open(self) -> pd.Series:
        nested = self._ts_cov(self.data["vwap"], self.data["low"], 30)
        return self._ts_cov(self.data["open"], nested, 5)

    def factor_032_low_pctchange_momentum(self) -> pd.Series:
        smoothed_low = self._ts_wma(self.data["low"], 10)
        return self._ts_pctchange(self._ts_min(smoothed_low, 50), 20)

    def factor_040_volume_high_corr_strength(self) -> pd.Series:
        corr = self._ts_corr(self.data["volume"], self.data["high"], 20)
        return corr * self._ts_sum(self.data["volume"], 30)


FACTOR_SPECS: tuple[DataMiningFactorSpec, ...] = (
    DataMiningFactorSpec(
        name="factor_002_intraday_strength",
        description="Clean intraday strength proxy: (close - open) / open.",
        source="(close/((open-10.0)*0.01))",
        builder=DataMiningEngine.factor_002_intraday_strength,
    ),
    DataMiningFactorSpec(
        name="factor_006_price_volume_var40",
        description="Variance of close divided by 5-day WMA of volume.",
        source="ts_var((close/ts_wma(volume,5)),40)",
        builder=DataMiningEngine.factor_006_price_volume_var40,
    ),
    DataMiningFactorSpec(
        name="factor_007_volume_ema40",
        description="40-day exponential moving average of volume.",
        source="ts_sum((-30.0-ts_ema(volume,40)),30)",
        builder=DataMiningEngine.factor_007_volume_ema40,
    ),
    DataMiningFactorSpec(
        name="factor_008_close_low_var10",
        description="10-day variance of the close-low product.",
        source="ts_var(((close*low)--30.0),10)",
        builder=DataMiningEngine.factor_008_close_low_var10,
    ),
    DataMiningFactorSpec(
        name="factor_010_vwap_corr_inverse_high_lag10",
        description="Lagged 30-day correlation of vwap and inverse high.",
        source="Ref(ts_corr(vwap,Inv(high),30),10)",
        builder=DataMiningEngine.factor_010_vwap_corr_inverse_high_lag10,
    ),
    DataMiningFactorSpec(
        name="factor_011_vwap_over_low_min10",
        description="2 * vwap divided by the 10-day low minimum.",
        source="(vwap/ts_min((low/2.0),10))",
        builder=DataMiningEngine.factor_011_vwap_over_low_min10,
    ),
    DataMiningFactorSpec(
        name="factor_012_volume_sum5",
        description="5-day rolling volume sum.",
        source="(ts_sum((volume*30.0),5)-10.0)",
        builder=DataMiningEngine.factor_012_volume_sum5,
    ),
    DataMiningFactorSpec(
        name="factor_013_volume",
        description="Raw trading volume.",
        source="volume",
        builder=DataMiningEngine.factor_013_volume,
    ),
    DataMiningFactorSpec(
        name="factor_021_log1p_inverse_turnover_pressure",
        description="Log1p of inverse volume-price pressure.",
        source="S_log1p((((volume*(high+2.0))/10.0)**-1.0))",
        builder=DataMiningEngine.factor_021_log1p_inverse_turnover_pressure,
    ),
    DataMiningFactorSpec(
        name="factor_024_close_over_high",
        description="Close price relative to the intraday high.",
        source="(close/(high+-1.0))",
        builder=DataMiningEngine.factor_024_close_over_high,
    ),
    DataMiningFactorSpec(
        name="factor_025_open_high_cov5_wma5",
        description="5-day covariance of open and high, then WMA-smoothed.",
        source="ts_wma(ts_cov((-5.0*open),((high*0.5)+Inv(close)),5),5)",
        builder=DataMiningEngine.factor_025_open_high_cov5_wma5,
    ),
    DataMiningFactorSpec(
        name="factor_029_vwap_minus_high",
        description="VWAP minus high.",
        source="(vwap-(high+-1.0))",
        builder=DataMiningEngine.factor_029_vwap_minus_high,
    ),
    DataMiningFactorSpec(
        name="factor_031_nested_cov_open",
        description="Nested covariance of open against a 30-day vwap/low covariance.",
        source="ts_cov((((ts_cov(vwap,low,30)*-0.01)**-2.0)*open),open,5)",
        builder=DataMiningEngine.factor_031_nested_cov_open,
    ),
    DataMiningFactorSpec(
        name="factor_032_low_pctchange_momentum",
        description="20-day pct change of a 50-day min over smoothed low.",
        source="ts_pctchange(ts_min((-5.0+((ts_wma(low,10)-0.01)+1.0)),50),20)",
        builder=DataMiningEngine.factor_032_low_pctchange_momentum,
    ),
    DataMiningFactorSpec(
        name="factor_040_volume_high_corr_strength",
        description="20-day volume/high correlation times 30-day volume strength.",
        source="(ts_corr(volume,(high+-1.0),20)*ts_sum(volume,30))",
        builder=DataMiningEngine.factor_040_volume_high_corr_strength,
    ),
)


def available_factors() -> list[str]:
    return [spec.name for spec in FACTOR_SPECS]


def factor_spec(name: str) -> DataMiningFactorSpec:
    lookup = {spec.name: spec for spec in FACTOR_SPECS}
    try:
        return lookup[name]
    except KeyError as exc:
        raise KeyError(f"Unknown data mining factor: {name}") from exc


def compute_factor(data: pd.DataFrame, name: str) -> pd.DataFrame:
    return DataMiningEngine(data).compute(name)


def _create_factor_function(name: str):
    def _factor(data: pd.DataFrame, **kwargs):
        del kwargs
        return DataMiningEngine(data).compute(name)

    _factor.__name__ = name
    _factor.__qualname__ = name
    _factor.__doc__ = factor_spec(name).description
    return _factor


for _factor_name in available_factors():
    globals()[_factor_name] = _create_factor_function(_factor_name)


__all__ = [
    "DataMiningFactorSpec",
    "DataMiningEngine",
    "FACTOR_SPECS",
    "available_factors",
    "compute_factor",
    "factor_spec",
    *available_factors(),
]
