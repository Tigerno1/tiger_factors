from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from tiger_factors.factor_algorithm.alpha101.engine import Alpha101Engine
from tiger_factors.factor_algorithm.alpha101.engine import NeutralizationColumns


@dataclass(frozen=True)
class Sunday100PlusColumns:
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


def _round_window(value: float) -> int:
    return max(int(round(float(value))), 1)


class Sunday100PlusEngine(Alpha101Engine):
    """
    Sunday100+ factor engine scaffold.

    The public API mirrors the Alpha101 / GTJA191 engine shape:
    - normalize the input frame
    - expose time-series / cross-sectional helpers
    - compute registered factor formulas by alpha id

    The formula registry is intentionally explicit so the engine can be
    populated from a paper/source-of-truth later without changing the wiring.
    """

    FIELD_SPEC: ClassVar[dict[str, tuple[str, ...]]] = {
        "required": ("date", "symbol", "open", "high", "low", "close", "volume"),
        "optional": (
            "vwap",
            "return",
            "amount",
            "market_value",
            "shares_outstanding",
            "index_open",
            "index_close",
            "benchmark_open",
            "benchmark_close",
            "mkt",
            "smb",
            "hml",
        ),
        "classification": ("sector", "industry", "subindustry"),
        "dropped": ("adj_close", "dividend", "price_adjustment_factor"),
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

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        neutralization_columns: NeutralizationColumns | None = None,
        formula_registry: Mapping[int, Callable[["Sunday100PlusEngine"], pd.Series]] | None = None,
        formula_descriptions: Mapping[int, str] | None = None,
    ) -> None:
        self.neutralization_columns = neutralization_columns or NeutralizationColumns()
        self._formula_registry: dict[int, Callable[["Sunday100PlusEngine"], pd.Series]] = {}
        self._formula_descriptions: dict[int, str] = {}
        if formula_registry:
            self.register_formulas(formula_registry, descriptions=formula_descriptions)
        self.data = self._prepare_input(data)
        self.eps = 1e-8

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = _as_pandas_frame(data)
        rename_map = {}
        for source, target in self.FIELD_ALIASES.items():
            if source in frame.columns and target not in frame.columns:
                rename_map[source] = target
        for source, target in (
            (self.neutralization_columns.sector, "sector"),
            (self.neutralization_columns.industry, "industry"),
            (self.neutralization_columns.subindustry, "subindustry"),
        ):
            if source in frame.columns and source != target:
                rename_map[source] = target
        frame = frame.rename(columns=rename_map)

        required = set(self.FIELD_SPEC["required"])
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"sunday100plus input is missing columns: {sorted(missing)}")

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

        frame["return"] = frame.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)

        if "shares_outstanding" in frame.columns:
            frame["market_value"] = frame["close"] * frame["shares_outstanding"]
        elif "market_value" not in frame.columns:
            frame["market_value"] = frame["close"] * frame["volume"]

        if "amount" not in frame.columns:
            frame["amount"] = frame["close"] * frame["volume"]

        frame = frame.drop(columns=list(self.FIELD_SPEC["dropped"]), errors="ignore")

        def _fill_group_column(target: str, fallbacks: tuple[str, ...]) -> None:
            if target in frame.columns:
                series = frame[target]
            else:
                series = None
                for fallback in fallbacks:
                    if fallback in frame.columns:
                        series = frame[fallback]
                        break
                if series is None:
                    frame[target] = "unknown"
                    return
            frame[target] = pd.Series(series, index=frame.index).astype("string").fillna("unknown")

        _fill_group_column("sector", ("industry", "subindustry"))
        _fill_group_column("industry", ("subindustry", "sector"))
        _fill_group_column("subindustry", ("industry", "sector"))

        if "index_open" in frame.columns:
            frame["index_open"] = pd.to_numeric(frame["index_open"], errors="coerce")
        if "index_close" in frame.columns:
            frame["index_close"] = pd.to_numeric(frame["index_close"], errors="coerce")
        if "benchmark_open" in frame.columns and "index_open" not in frame.columns:
            frame["index_open"] = pd.to_numeric(frame["benchmark_open"], errors="coerce")
        if "benchmark_close" in frame.columns and "index_close" not in frame.columns:
            frame["index_close"] = pd.to_numeric(frame["benchmark_close"], errors="coerce")
        for column in ("mkt", "smb", "hml"):
            if column not in frame.columns:
                frame[column] = np.nan

        return frame

    def register_formula(
        self,
        alpha_id: int,
        formula: Callable[["Sunday100PlusEngine"], pd.Series],
        *,
        description: str | None = None,
    ) -> None:
        alpha_id = int(alpha_id)
        self._formula_registry[alpha_id] = formula
        if description is not None:
            self._formula_descriptions[alpha_id] = description

    def register_formulas(
        self,
        formulas: Mapping[int, Callable[["Sunday100PlusEngine"], pd.Series]],
        *,
        descriptions: Mapping[int, str] | None = None,
    ) -> None:
        for alpha_id, formula in formulas.items():
            self.register_formula(alpha_id, formula, description=None if descriptions is None else descriptions.get(int(alpha_id)))

    def formula_description(self, alpha_id: int) -> str:
        alpha_id = int(alpha_id)
        if alpha_id in self._formula_descriptions:
            return self._formula_descriptions[alpha_id]
        return f"Sunday100+ alpha_{alpha_id:03d}"

    def formula_descriptions(self) -> dict[int, str]:
        return {alpha_id: self.formula_description(alpha_id) for alpha_id in self.registered_alpha_ids()}

    def registered_alpha_ids(self) -> list[int]:
        return sorted(self._formula_registry)

    def factor_names(self) -> list[str]:
        return [f"alpha_{i:03d}" for i in self.registered_alpha_ids()]

    def compute_series(self, alpha_id: int) -> pd.Series:
        alpha_id = int(alpha_id)
        if alpha_id not in self._formula_registry:
            raise NotImplementedError(
                f"sunday100plus alpha_{alpha_id:03d} is not registered yet; "
                f"currently registered ids are {self.registered_alpha_ids()}"
            )
        series = self._formula_registry[alpha_id](self).rename(f"alpha_{alpha_id:03d}")
        return pd.to_numeric(series, errors="coerce")

    def compute(self, alpha_id: int) -> pd.DataFrame:
        factor_name = f"alpha_{int(alpha_id):03d}"
        values = self.compute_series(alpha_id)
        result = self._finalize(values, factor_name)
        result.attrs["description"] = self.formula_description(alpha_id)
        return result

    def compute_all(self, alpha_ids: Iterable[int] | None = None) -> pd.DataFrame:
        selected = list(alpha_ids) if alpha_ids is not None else self.registered_alpha_ids()
        frame = self.data[["date", "symbol"]].rename(columns={"date": "date_", "symbol": "code"}).copy()
        factor_columns: dict[str, pd.Series] = {}
        for alpha_id in selected:
            factor_name = f"alpha_{int(alpha_id):03d}"
            factor_columns[factor_name] = pd.to_numeric(self.compute_series(int(alpha_id)), errors="coerce")
        if not factor_columns:
            return frame
        result = pd.concat([frame, pd.DataFrame(factor_columns, index=self.data.index)], axis=1)
        result.attrs["alpha_descriptions"] = {f"alpha_{alpha_id:03d}": self.formula_description(alpha_id) for alpha_id in selected}
        return result

    def compute_matrix(self, alpha_id: int) -> pd.DataFrame:
        factor_name = f"alpha_{int(alpha_id):03d}"
        result = self.compute(alpha_id)
        return result.pivot(index="date_", columns="code", values=factor_name).sort_index()


def sunday100plus_factor_names() -> list[str]:
    return [f"alpha_{i:03d}" for i in range(1, 101)]
