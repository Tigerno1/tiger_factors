from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


def alpha158_feature_config(config: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    from qlib.contrib.data.loader import Alpha158DL

    if config is None:
        return Alpha158DL.get_feature_config()
    return Alpha158DL.get_feature_config(config)


def alpha360_feature_config() -> tuple[list[str], list[str]]:
    from qlib.contrib.data.loader import Alpha360DL

    return Alpha360DL.get_feature_config()


def available_qlib_factor_sets() -> tuple[str, ...]:
    return ("alpha158", "alpha360")


@dataclass(frozen=True)
class Alpha158FactorSet:
    """Alpha158 is a predefined Qlib factor family, not a single factor.

    The default feature set contains 158 derived features built from OHLCV
    inputs. The exact formulas are Qlib-compatible and are exposed here as a
    Tiger-native factor set definition.
    """

    config: dict[str, Any] | None = None

    @property
    def feature_exprs(self) -> list[str]:
        return alpha158_feature_config(self.config)[0] if self.config is not None else alpha158_feature_config()[0]

    @property
    def feature_names(self) -> list[str]:
        return alpha158_feature_config(self.config)[1] if self.config is not None else alpha158_feature_config()[1]


@dataclass(frozen=True)
class Alpha360FactorSet:
    """Alpha360 is a predefined Qlib factor family, not a single factor.

    The default feature set contains 360 normalized OHLCV history features
    over a 60-day lookback window.
    """

    @property
    def feature_exprs(self) -> list[str]:
        return alpha360_feature_config()[0]

    @property
    def feature_names(self) -> list[str]:
        return alpha360_feature_config()[1]


class QlibAlphaFactorSetEngine:
    """Tiger wrapper around Qlib's predefined alpha feature families.

    This engine does two things:
    1. exposes the exact Alpha158 / Alpha360 feature configurations
    2. fetches the corresponding factor panels from an initialized Qlib provider

    The returned panel is always normalized to Tiger's long-form format:
    `date_`, `code`, plus one column per factor.
    """

    def __init__(
        self,
        *,
        provider_uri: str | Path,
        region: str = "us",
        instruments: str = "all",
        start_time: str | pd.Timestamp | None = None,
        end_time: str | pd.Timestamp | None = None,
        fit_start_time: str | pd.Timestamp | None = None,
        fit_end_time: str | pd.Timestamp | None = None,
    ) -> None:
        self.provider_uri = str(provider_uri)
        self.region = region
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self._initialized = False

    def initialize(self) -> None:
        import qlib

        qlib.init(provider_uri=self.provider_uri, region=self.region)
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _build_handler(
        self,
        family: Literal["alpha158", "alpha360"],
    ):
        self._ensure_initialized()
        if family == "alpha158":
            from qlib.contrib.data.handler import Alpha158

            return Alpha158(
                instruments=self.instruments,
                start_time=self.start_time,
                end_time=self.end_time,
                fit_start_time=self.fit_start_time,
                fit_end_time=self.fit_end_time,
            )
        if family == "alpha360":
            from qlib.contrib.data.handler import Alpha360

            return Alpha360(
                instruments=self.instruments,
                start_time=self.start_time,
                end_time=self.end_time,
                fit_start_time=self.fit_start_time,
                fit_end_time=self.fit_end_time,
            )
        raise ValueError(f"Unknown Qlib alpha family: {family}")

    @staticmethod
    def _normalize_fetch_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["date_", "code"])

        out = frame.copy()

        if isinstance(out.index, pd.MultiIndex):
            out = out.reset_index()

        rename_map = {}
        for candidate, target in (
            ("datetime", "date_"),
            ("date", "date_"),
            ("instrument", "code"),
            ("symbol", "code"),
            ("code", "code"),
        ):
            if candidate in out.columns and candidate != target:
                rename_map[candidate] = target
        out = out.rename(columns=rename_map)

        if "date_" not in out.columns:
            for column in out.columns:
                if str(column).lower() in {"datetime", "date"}:
                    out = out.rename(columns={column: "date_"})
                    break
        if "code" not in out.columns:
            for column in out.columns:
                if str(column).lower() in {"instrument", "symbol"}:
                    out = out.rename(columns={column: "code"})
                    break

        if "date_" in out.columns:
            out["date_"] = pd.to_datetime(out["date_"], errors="coerce").dt.tz_localize(None)
        if "code" in out.columns:
            out["code"] = out["code"].astype(str)

        feature_columns = [column for column in out.columns if column not in {"date_", "code"}]
        for column in feature_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        return out.sort_values(["date_", "code"]).reset_index(drop=True)

    def fetch_panel(
        self,
        family: Literal["alpha158", "alpha360"],
        *,
        selector: slice | str | pd.Timestamp | None = None,
        col_set: str = "feature",
        data_key: Literal["raw", "infer", "learn"] = "raw",
        squeeze: bool = False,
        proc_func=None,
    ) -> pd.DataFrame:
        handler = self._build_handler(family)
        raw = handler.fetch(
            selector=selector if selector is not None else slice(None, None, None),
            level="datetime",
            col_set=col_set,
            data_key=data_key,
            squeeze=squeeze,
            proc_func=proc_func,
        )
        if isinstance(raw, pd.Series):
            raw = raw.to_frame()
        return self._normalize_fetch_frame(raw)

    def fetch_alpha158(
        self,
        *,
        selector: slice | str | pd.Timestamp | None = None,
        col_set: str = "feature",
        data_key: Literal["raw", "infer", "learn"] = "raw",
        squeeze: bool = False,
        proc_func=None,
    ) -> pd.DataFrame:
        return self.fetch_panel(
            "alpha158",
            selector=selector,
            col_set=col_set,
            data_key=data_key,
            squeeze=squeeze,
            proc_func=proc_func,
        )

    def fetch_alpha360(
        self,
        *,
        selector: slice | str | pd.Timestamp | None = None,
        col_set: str = "feature",
        data_key: Literal["raw", "infer", "learn"] = "raw",
        squeeze: bool = False,
        proc_func=None,
    ) -> pd.DataFrame:
        return self.fetch_panel(
            "alpha360",
            selector=selector,
            col_set=col_set,
            data_key=data_key,
            squeeze=squeeze,
            proc_func=proc_func,
        )

    def feature_spec(self, family: Literal["alpha158", "alpha360"]) -> tuple[list[str], list[str]]:
        if family == "alpha158":
            return alpha158_feature_config()
        if family == "alpha360":
            return alpha360_feature_config()
        raise ValueError(f"Unknown Qlib alpha family: {family}")
