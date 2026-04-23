from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketBreathingColumns:
    date: str = "date_"
    code: str = "code"
    iv_surface_level: str = "iv_surface_level"
    iv_surface_skew: str = "iv_surface_skew"
    iv_surface_curvature: str = "iv_surface_curvature"
    iv_term_slope: str = "iv_term_slope"
    option_spread: str = "option_spread"
    option_volume: str = "option_volume"
    option_open_interest: str = "option_open_interest"
    realized_volatility: str = "realized_volatility"


@dataclass(frozen=True, slots=True)
class MarketBreathingResult:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    metadata: dict[str, object]


def market_breathing_factor_names() -> list[str]:
    return ["market_breathing"]


def _as_frame(data: object) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return pd.DataFrame(data)


def _rolling_zscore(series: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    min_periods = max(3, min(window, 5))
    mean = series.groupby(groups, sort=False).transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    std = series.groupby(groups, sort=False).transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=0))
    return (series - mean) / std.replace(0.0, np.nan)


class MarketBreathingEngine:
    """Experimental market breathing factor.

    The engine expects self-named columns that can be generated directly from
    an options analytics pipeline:

    - ``date_``
    - ``code``
    - ``iv_surface_level``
    - ``iv_surface_skew``
    - ``iv_surface_curvature``
    - ``iv_term_slope``
    - ``option_spread``
    - ``option_volume``
    - optional ``option_open_interest``
    - optional ``realized_volatility``

    The factor is intended to capture regime pressure and "breathing" changes
    in the implied-volatility surface rather than a fixed trade rule.
    """

    FIELD_SPEC: ClassVar[dict[str, tuple[str, ...]]] = {
        "required": (
            "date_",
            "code",
            "iv_surface_level",
            "iv_surface_skew",
            "iv_surface_curvature",
            "iv_term_slope",
            "option_spread",
            "option_volume",
        ),
        "optional": ("option_open_interest", "realized_volatility"),
    }

    FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "date": "date_",
        "tradetime": "date_",
        "symbol": "code",
        "securityid": "code",
        "iv_level": "iv_surface_level",
        "iv_skew": "iv_surface_skew",
        "iv_curvature": "iv_surface_curvature",
        "term_slope": "iv_term_slope",
        "spread": "option_spread",
        "volume": "option_volume",
        "open_interest": "option_open_interest",
        "realized_vol": "realized_volatility",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        columns: MarketBreathingColumns | None = None,
        window: int = 20,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.columns = columns or MarketBreathingColumns()
        self.window = max(int(window), 3)
        self.weights = dict(
            weights
            or {
                "curvature_change": 0.30,
                "skew_change": 0.20,
                "level_change": 0.15,
                "term_slope_change": -0.15,
                "spread_change": -0.20,
                "volume_change": 0.20,
                "open_interest_change": 0.10,
                "realized_volatility": -0.10,
            }
        )
        self.data = self._prepare_input(data)

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = _as_frame(data)
        rename_map: dict[str, str] = {}
        for source, target in self.FIELD_ALIASES.items():
            if source in frame.columns and target not in frame.columns:
                rename_map[source] = target
        frame = frame.rename(columns=rename_map)

        missing = [column for column in self.FIELD_SPEC["required"] if column not in frame.columns]
        if missing:
            raise ValueError(f"market_breathing input is missing columns: {missing!r}")

        frame[self.columns.date] = pd.to_datetime(frame[self.columns.date], errors="coerce").dt.tz_localize(None)
        frame[self.columns.code] = frame[self.columns.code].astype(str)
        frame = frame.sort_values([self.columns.code, self.columns.date], kind="stable").reset_index(drop=True)

        for column in [*self.FIELD_SPEC["required"][2:], *self.FIELD_SPEC["optional"]]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if self.columns.option_open_interest not in frame.columns:
            frame[self.columns.option_open_interest] = np.nan
        if self.columns.realized_volatility not in frame.columns:
            frame[self.columns.realized_volatility] = np.nan

        return frame

    def _feature_frame(self) -> pd.DataFrame:
        frame = self.data.copy()
        code = self.columns.code

        frame["_curvature_change"] = frame.groupby(code, sort=False)[self.columns.iv_surface_curvature].diff()
        frame["_skew_change"] = frame.groupby(code, sort=False)[self.columns.iv_surface_skew].diff()
        frame["_level_change"] = frame.groupby(code, sort=False)[self.columns.iv_surface_level].diff()
        frame["_term_slope_change"] = frame.groupby(code, sort=False)[self.columns.iv_term_slope].diff()
        frame["_spread_change"] = frame.groupby(code, sort=False)[self.columns.option_spread].diff()
        frame["_volume_change"] = frame.groupby(code, sort=False)[self.columns.option_volume].diff()
        frame["_open_interest_change"] = frame.groupby(code, sort=False)[self.columns.option_open_interest].diff()

        frame["curvature_change_z"] = _rolling_zscore(frame["_curvature_change"], frame[code], self.window)
        frame["skew_change_z"] = _rolling_zscore(frame["_skew_change"], frame[code], self.window)
        frame["level_change_z"] = _rolling_zscore(frame["_level_change"], frame[code], self.window)
        frame["term_slope_change_z"] = _rolling_zscore(frame["_term_slope_change"], frame[code], self.window)
        frame["spread_change_z"] = _rolling_zscore(frame["_spread_change"], frame[code], self.window)
        frame["volume_change_z"] = _rolling_zscore(frame["_volume_change"], frame[code], self.window)
        frame["open_interest_change_z"] = _rolling_zscore(frame["_open_interest_change"], frame[code], self.window)
        frame["realized_volatility_z"] = _rolling_zscore(frame[self.columns.realized_volatility], frame[code], self.window)

        return frame

    def compute(self) -> MarketBreathingResult:
        frame = self._feature_frame()
        component_columns = (
            "curvature_change_z",
            "skew_change_z",
            "level_change_z",
            "term_slope_change_z",
            "spread_change_z",
            "volume_change_z",
            "open_interest_change_z",
            "realized_volatility_z",
        )
        frame["value"] = 0.0
        for component, weight in self.weights.items():
            column_name = {
                "curvature_change": "curvature_change_z",
                "skew_change": "skew_change_z",
                "level_change": "level_change_z",
                "term_slope_change": "term_slope_change_z",
                "spread_change": "spread_change_z",
                "volume_change": "volume_change_z",
                "open_interest_change": "open_interest_change_z",
                "realized_volatility": "realized_volatility_z",
            }.get(component)
            if column_name is None:
                continue
            frame["value"] = frame["value"] + pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0) * float(weight)

        output = frame.loc[:, [self.columns.date, self.columns.code, "value", *component_columns]].copy()
        output = output.rename(columns={self.columns.date: "date_", self.columns.code: "code"})
        output["factor_name"] = "market_breathing"
        output["value"] = pd.to_numeric(output["value"], errors="coerce")
        return MarketBreathingResult(
            frame=output.reset_index(drop=True),
            feature_columns=component_columns,
            metadata={
                "window": self.window,
                "weights": dict(self.weights),
                "factor_name": "market_breathing",
            },
        )

    def compute_matrix(self) -> pd.DataFrame:
        result = self.compute().frame
        return result.pivot(index="date_", columns="code", values="value").sort_index()


