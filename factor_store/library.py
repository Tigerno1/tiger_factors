from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

import tiger_factors.factor_store.store as factor_store_store
from tiger_factors.utils.accelerators import momentum_12m_1m as accelerated_momentum_12m_1m
from tiger_factors.utils.accelerators import rolling_std as accelerated_rolling_std
from tiger_factors.factor_algorithm.alpha101 import Alpha101Engine, NeutralizationColumns
from tiger_factors.factor_algorithm.gtja191 import GTJA191Engine
from tiger_factors.factor_algorithm.sunday100plus import Sunday100PlusEngine
from tiger_factors.utils.asof_align import align_fundamental_point_in_time
from tiger_factors.factor_store.conf import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store.spec import FactorSpec
from tiger_factors.factor_store.spec import OthersSpec
from tiger_factors.factor_store.store import DatasetSaveResult
from tiger_factors.factor_store.store import FactorStore
from tiger_factors.factor_store.store import _to_polars_frame
from tiger_reference.adjustments import adj_df

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAMA_FAMILY_SHORT_NAMES: dict[int, tuple[str, ...]] = {
    3: ("mkt", "smb", "hml"),
    5: ("mkt", "smb", "hml", "rmw", "cma"),
    6: ("mkt", "smb", "hml", "rmw", "cma", "umd"),
}

FAMA_FAMILY_LONG_NAMES: dict[int, tuple[str, ...]] = {
    3: ("market", "size", "value"),
    5: ("market", "size", "value", "profitability", "investment"),
    6: ("market", "size", "value", "profitability", "investment", "momentum"),
}

FAMA_NAME_ALIASES: dict[str, str] = {
    "market": "mkt",
    "size": "smb",
    "value": "hml",
    "profitability": "rmw",
    "investment": "cma",
    "momentum": "umd",
    "mkt": "mkt",
    "smb": "smb",
    "hml": "hml",
    "rmw": "rmw",
    "cma": "cma",
    "umd": "umd",
}

@dataclass(frozen=True)
class FactorResult:
    name: str
    data: pd.DataFrame
    parquet_path: Path
    metadata_path: Path


class ProviderAdapter(Protocol):
    def fetch_price_data(
        self,
        *,
        provider: str,
        region: str,
        sec_type: str,
        freq: str,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> pd.DataFrame: ...

    def fetch_fundamental_data(
        self,
        *,
        provider: str,
        name: str,
        region: str,
        sec_type: str,
        freq: str,
        variant: str | None,
        codes: list[str] | None,
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> pd.DataFrame: ...

    def fetch_dataset(
        self,
        *,
        provider: str,
        name: str,
        region: str,
        sec_type: str,
        freq: str | None,
        filters: dict[str, object],
        as_ex: bool | None = None,
    ) -> pd.DataFrame: ...


class TigerAPIAdapter:
    @staticmethod
    def _price_dataset_name(provider: str) -> str:
        if str(provider).strip().lower() in {"simfin", "yahoo", "eod"}:
            return "eod_price"
        return "price"

    def _fetch(self, *args, **kwargs):
        try:
            from tiger_api.sdk.client import fetch as tiger_fetch
        except Exception as exc:
            raise ImportError(
                "tiger_api is not available. Install tiger_api or register a custom provider adapter."
            ) from exc
        return tiger_fetch(*args, **kwargs)

    def fetch_price_data(
        self,
        *,
        provider: str,
        region: str,
        sec_type: str,
        freq: str,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        dataset_name = self._price_dataset_name(provider)
        try:
            return self._fetch(
                provider,
                dataset_name,
                region=region,
                sec_type=sec_type,
                freq=freq,
                codes=codes,
                start=start,
                end=end,
                as_ex=as_ex,
                return_type="df",
            )
        except TypeError:
            return self._fetch(
                provider,
                dataset_name,
                region=region,
                sec_type=sec_type,
                freq=freq,
                codes=codes,
                as_ex=as_ex,
                return_type="df",
            )

    def fetch_fundamental_data(
        self,
        *,
        provider: str,
        name: str,
        region: str,
        sec_type: str,
        freq: str,
        variant: str | None,
        codes: list[str] | None,
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        try:
            return self._fetch(
                provider,
                name,
                region=region,
                sec_type=sec_type,
                freq=freq,
                variant=variant,
                codes=codes,
                start=start,
                end=end,
                as_ex=as_ex,
                return_type="df",
            )
        except TypeError:
            return self._fetch(
                provider,
                name,
                region=region,
                sec_type=sec_type,
                freq=freq,
                variant=variant,
                codes=codes,
                as_ex=as_ex,
                return_type="df",
            )

    def fetch_dataset(
        self,
        *,
        provider: str,
        name: str,
        region: str,
        sec_type: str,
        freq: str | None,
        filters: dict[str, object],
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        return self._fetch(
            provider,
            name,
            region=region,
            sec_type=sec_type,
            freq=freq,
            as_ex=as_ex,
            return_type="df",
            **filters,
        )


def _normalize_fama_family(family: int | str) -> int:
    if isinstance(family, int):
        family_id = family
    else:
        token = str(family).strip().lower()
        if token.startswith("fama"):
            token = token[4:]
        family_id = int(token)
    if family_id not in FAMA_FAMILY_SHORT_NAMES:
        raise ValueError(f"unsupported Fama family: {family!r}")
    return family_id


def _fama_factor_names(family: int | str, *, style: str = "abbr") -> tuple[str, ...]:
    family_id = _normalize_fama_family(family)
    style_key = str(style).strip().lower()
    if style_key in {"abbr", "short", "canonical"}:
        return FAMA_FAMILY_SHORT_NAMES[family_id]
    if style_key in {"long", "descriptive"}:
        return FAMA_FAMILY_LONG_NAMES[family_id]
    raise ValueError(f"unsupported Fama name style: {style!r}")


def _fama_alias_candidates(name: str) -> tuple[str, ...]:
    token = str(name).strip().lower()
    candidates = [token]
    alias = FAMA_NAME_ALIASES.get(token)
    if alias is not None:
        candidates.append(alias)
    reverse_aliases = {value: key for key, value in FAMA_NAME_ALIASES.items()}
    reverse = reverse_aliases.get(token)
    if reverse is not None:
        candidates.append(reverse)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _quote_sql_literal(value: object) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _quote_sql_identifier(value: object) -> str:
    text = str(value).replace('"', '""')
    return f'"{text}"'


def _resolve_fama_column(frame: pd.DataFrame, target_name: str) -> str | None:
    for candidate in _fama_alias_candidates(target_name):
        if candidate in frame.columns:
            return candidate
    return None


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_dates(values) -> pd.Series:
    converted = pd.to_datetime(values, utc=True)
    if isinstance(converted, pd.Series):
        return converted.dt.tz_localize(None)
    if isinstance(converted, pd.DatetimeIndex):
        return converted.tz_localize(None)
    return pd.Series(pd.DatetimeIndex(converted).tz_localize(None))


def to_long_factor(frame: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date_", "code", factor_name])
    long_df = (
        frame.rename_axis(index="date_")
        .reset_index()
        .melt(id_vars="date_", var_name="code", value_name=factor_name)
    )
    long_df["date_"] = normalize_dates(long_df["date_"])
    long_df["code"] = long_df["code"].astype(str)
    return long_df.dropna(subset=[factor_name]).sort_values(["date_", "code"]).reset_index(drop=True)


def to_long_series(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date_", "code", value_name])
    long_df = (
        frame.rename_axis(index="date_")
        .reset_index()
        .melt(id_vars="date_", var_name="code", value_name=value_name)
    )
    long_df["date_"] = normalize_dates(long_df["date_"])
    long_df["code"] = long_df["code"].astype(str)
    return long_df.sort_values(["date_", "code"]).reset_index(drop=True)


def _adjust_ohlcv_for_alpha101(df: pd.DataFrame) -> pd.DataFrame:
    """Build an Alpha101-ready price frame with optional adj_close-based adjustment."""
    if df.empty:
        return df.copy()

    frame = df.copy()
    if "code" not in frame.columns or "date_" not in frame.columns:
        raise ValueError("price data must contain code and date_ columns")

    return adj_df(
        frame,
        drop_adj_close=True,
        history=False,
    )


class TigerFactorLibrary:
    def __init__(
        self,
        *,
        store: FactorStore | None = None,
        output_dir: str | Path | None = None,
        region: str = "us",
        sec_type: str = "stock",
        price_provider: str = "yahoo",
        verbose: bool = True,
        provider_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> None:
        if store is not None and output_dir is not None:
            raise ValueError("pass either store or output_dir, not both")
        if store is None:
            store = FactorStore(output_dir)
        self.store = store
        self.output_dir = self.store.root_dir
        ensure_dir(self.output_dir)
        self.region = region
        self.sec_type = sec_type
        self.price_provider = price_provider
        self.verbose = verbose
        self.provider_adapters = dict(provider_adapters or {})
        self._default_adapter: ProviderAdapter = TigerAPIAdapter()

    def register_provider_adapter(self, provider: str, adapter: ProviderAdapter) -> None:
        self.provider_adapters[str(provider)] = adapter

    def _adapter_for(self, provider: str) -> ProviderAdapter:
        return self.provider_adapters.get(provider, self._default_adapter)

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _filter_by_date_range(df: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
        if df.empty or "date_" not in df.columns:
            return df
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return df.loc[(df["date_"] >= start_ts) & (df["date_"] <= end_ts)]

    def fetch_price_data(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        provider: str | None = None,
        freq: str = "1d",
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        provider = provider or self.price_provider
        rows = self._adapter_for(provider).fetch_price_data(
            provider=provider,
            region=self.region,
            sec_type=self.sec_type,
            freq=freq,
            codes=codes,
            start=start,
            end=end,
            as_ex=as_ex,
        )
        df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if df.empty:
            return df
        if "code" not in df.columns or "date_" not in df.columns:
            raise KeyError("price data must contain code and date_ columns")
        df["date_"] = normalize_dates(df["date_"])
        df["code"] = df["code"].astype(str)
        for column in ("open", "high", "low", "close", "adj_close", "volume", "dividend", "shares_outstanding"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        df = self._filter_by_date_range(df, start=start, end=end)
        return df.sort_values(["date_", "code"]).reset_index(drop=True)

    def fetch_fundamental_data(
        self,
        *,
        name: str,
        freq: str,
        variant: str | None = None,
        start: str,
        end: str,
        codes: list[str] | None = None,
        provider: str = "simfin",
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        rows = self._adapter_for(provider).fetch_fundamental_data(
            provider=provider,
            name=name,
            region=self.region,
            sec_type=self.sec_type,
            freq=freq,
            variant=variant,
            codes=codes,
            start=start,
            end=end,
            as_ex=as_ex,
        )
        df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if df.empty:
            return df
        df["code"] = df["code"].astype(str)
        if freq == "static":
            if "code" not in df.columns:
                raise KeyError("static fundamental data must contain code column")
            return df.sort_values(["code"]).reset_index(drop=True)
        if "date_" not in df.columns:
            if "publish_date" in df.columns:
                df["date_"] = normalize_dates(df["publish_date"])
            else:
                raise KeyError("fundamental data must contain date_ or publish_date columns")
        df["date_"] = normalize_dates(df["date_"])
        df = self._filter_by_date_range(df, start=start, end=end)
        return df.sort_values(["date_", "code"]).reset_index(drop=True)

    def fetch_dataset(
        self,
        *,
        provider: str,
        name: str,
        freq: str | None = None,
        as_ex: bool | None = None,
        **filters,
    ) -> pd.DataFrame:
        rows = self._adapter_for(provider).fetch_dataset(
            provider=provider,
            name=name,
            region=self.region,
            sec_type=self.sec_type,
            freq=freq,
            filters=filters,
            as_ex=as_ex,
        )
        df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if df.empty:
            return df
        if "code" in df.columns:
            df["code"] = df["code"].astype(str)
        if "date_" not in df.columns:
            sort_columns = ["code"] if "code" in df.columns else list(df.columns[:1])
            return df.sort_values(sort_columns).reset_index(drop=True)
        df["date_"] = normalize_dates(df["date_"])
        sort_columns = ["date_"]
        if "code" in df.columns:
            sort_columns.append("code")
        return df.sort_values(sort_columns).reset_index(drop=True)

    def resolve_universe_codes(
        self,
        *,
        provider: str,
        dataset: str = "companies",
        freq: str | None = "1d",
        code_column: str | None = None,
        limit: int | None = None,
        sort_by: str | None = None,
        ascending: bool = True,
        filters: dict[str, object] | None = None,
        as_ex: bool | None = None,
    ) -> list[str]:
        if dataset == "companies" and freq != "static":
            freq = "static"
        universe = self.fetch_dataset(
            provider=provider,
            name=dataset,
            freq=freq,
            as_ex=as_ex,
            **(filters or {}),
        )
        if universe.empty:
            return []

        source = universe.copy()
        if sort_by and sort_by in source.columns:
            source = source.sort_values(sort_by, ascending=ascending)

        candidate_columns = [
            code_column,
            "code",
        ]
        selected_column = next((column for column in candidate_columns if column and column in source.columns), None)
        if selected_column is None:
            raise KeyError(
                f"Could not resolve a code column from dataset {dataset!r}; tried {candidate_columns}."
            )

        codes: list[str] = []
        for value in source[selected_column].dropna().tolist():
            normalized = str(value).strip().upper()
            if normalized and normalized != "NAN":
                codes.append(normalized)

        deduped = list(dict.fromkeys(codes))
        if limit is not None:
            return deduped[: int(limit)]
        return deduped

    def price_panel(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        provider: str | None = None,
        field: str = "close",
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        price_df = self.fetch_price_data(codes=codes, start=start, end=end, provider=provider, as_ex=as_ex)
        if price_df.empty:
            return pd.DataFrame()
        if field not in price_df.columns:
            raise KeyError(f"Price field {field!r} not found in fetched data.")
        return price_df.pivot(index="date_", columns="code", values=field).sort_index().ffill()

    def load_factor_panel(
        self,
        *,
        factor_name: str,
        provider: str = "tiger",
        freq: str = "1d",
        variant: str | None = None,
        codes: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        engine: str = "pandas",
    ) -> pd.DataFrame | object:
        """Load a stored factor as a wide date x code panel.

        The underlying factor file is expected to be stored in the Tiger factor
        store as a standard long table with ``date_``, ``code`` and ``value``
        columns. The result is pivoted into a wide panel suitable for joins and
        backtests.
        """
        store = self.store
        spec = FactorSpec(
            provider=provider,
            region=self.region,
            sec_type=self.sec_type,
            freq=freq,
            table_name=str(factor_name).strip().lower(),
            variant=variant,
        )
        factor_df = store.get_factor(spec, start=start, end=end, engine="pandas")
        if not isinstance(factor_df, pd.DataFrame) or factor_df.empty:
            empty = pd.DataFrame()
            return _to_polars_frame(empty) if engine == "polars" else empty
        required = {"date_", "code", "value"}
        missing = sorted(required - set(factor_df.columns))
        if missing:
            raise ValueError(f"factor panel has unexpected columns; missing {missing}")

        panel = (
            factor_df.loc[:, ["date_", "code", "value"]]
            .copy()
            .assign(
                date_=pd.to_datetime(factor_df["date_"], errors="coerce").dt.tz_localize(None),
                code=factor_df["code"].astype(str),
                value=pd.to_numeric(factor_df["value"], errors="coerce"),
            )
            .dropna(subset=["date_", "code", "value"])
            .pivot_table(index="date_", columns="code", values="value", aggfunc="last")
            .sort_index()
        )
        if codes is not None:
            normalized_codes = list(dict.fromkeys(map(str, codes)))
            panel = panel.reindex(columns=normalized_codes)
        if start is not None or end is not None:
            start_ts = pd.Timestamp(start) if start is not None else panel.index.min()
            end_ts = pd.Timestamp(end) if end is not None else panel.index.max()
            panel = panel.loc[(panel.index >= start_ts) & (panel.index <= end_ts)]
        if engine == "polars":
            return _to_polars_frame(panel.rename_axis(index="date_").reset_index())
        return panel

    def load_factor_long(
        self,
        *,
        factor_name: str,
        provider: str = "tiger",
        freq: str = "1d",
        variant: str | None = None,
        codes: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load a stored factor as a long date / code / value frame."""
        panel = self.load_factor_panel(
            factor_name=factor_name,
            provider=provider,
            freq=freq,
            variant=variant,
            codes=codes,
            start=start,
            end=end,
            engine="pandas",
        )
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            return pd.DataFrame(columns=["date_", "code", factor_name])
        long_df = (
            panel.rename_axis(index="date_")
            .reset_index()
            .melt(id_vars="date_", var_name="code", value_name=factor_name)
        )
        long_df["date_"] = pd.to_datetime(long_df["date_"], errors="coerce").dt.tz_localize(None)
        long_df["code"] = long_df["code"].astype(str)
        long_df[factor_name] = pd.to_numeric(long_df[factor_name], errors="coerce")
        long_df = long_df.dropna(subset=["date_", "code", factor_name])
        return long_df.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)

    def load_factor_panels(
        self,
        *,
        factor_names: Iterable[str],
        provider: str = "tiger",
        freq: str = "1d",
        variant: str | None = None,
        codes: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load several stored factors as wide panels keyed by factor name."""
        panels: dict[str, pd.DataFrame] = {}
        for factor_name in factor_names:
            panel = self.load_factor_panel(
                factor_name=factor_name,
                provider=provider,
                freq=freq,
                variant=variant,
                codes=codes,
                start=start,
                end=end,
                engine="pandas",
            )
            if isinstance(panel, pd.DataFrame):
                panels[str(factor_name)] = panel
        return panels

    @staticmethod
    def _query_backend_available() -> bool:
        return factor_store_store.ibis is not None and factor_store_store.duckdb is not None

    def _factor_spec(
        self,
        *,
        factor_name: str,
        provider: str,
        freq: str,
        variant: str | None,
    ) -> FactorSpec:
        return FactorSpec(
            provider=provider,
            region=self.region,
            sec_type=self.sec_type,
            freq=freq,
            table_name=str(factor_name).strip().lower(),
            variant=variant,
        )

    def _build_multi_factor_frame_sql(
        self,
        *,
        factor_specs: Sequence[tuple[str, FactorSpec, Sequence[str]]],
        codes: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> str:
        if not factor_specs:
            raise ValueError("factor_specs must not be empty")

        ctes: list[str] = []
        for idx, (factor_name, spec, paths) in enumerate(factor_specs):
            path_list = ", ".join(_quote_sql_literal(path) for path in paths)
            conditions: list[str] = []
            if start is not None:
                conditions.append(f"{spec.date_column} >= TIMESTAMP {_quote_sql_literal(pd.Timestamp(start))}")
            if end is not None:
                conditions.append(f"{spec.date_column} <= TIMESTAMP {_quote_sql_literal(pd.Timestamp(end))}")
            if codes:
                code_list = ", ".join(_quote_sql_literal(code) for code in dict.fromkeys(map(str, codes)))
                conditions.append(f"{spec.code_column or 'code'} IN ({code_list})")
            where_clause = ""
            if conditions:
                where_clause = "\n    WHERE " + " AND ".join(conditions)

            ctes.append(
                "\n".join(
                    [
                        f"factor_{idx} AS (",
                        "    SELECT",
                        f"        CAST({spec.date_column} AS TIMESTAMP) AS date_,",
                        f"        CAST({spec.code_column or 'code'} AS VARCHAR) AS code,",
                        f"        CAST({getattr(spec, 'value_column', 'value')} AS DOUBLE) AS {_quote_sql_identifier(factor_name)}",
                        f"    FROM read_parquet([{path_list}], union_by_name=true){where_clause}",
                        ")",
                    ]
                )
            )
        select_lines = ["SELECT", "    base.date_,", "    base.code,"]
        projected_columns = []
        for idx, (factor_name, _, _) in enumerate(factor_specs):
            table_alias = "base" if idx == 0 else f"factor_{idx}"
            projected_columns.append(f"{table_alias}.{_quote_sql_identifier(factor_name)}")
        select_lines.extend(
            f"    {column}{',' if idx < len(projected_columns) - 1 else ''}"
            for idx, column in enumerate(projected_columns)
        )

        join_lines = ["FROM factor_0 AS base"]
        for idx in range(1, len(factor_specs)):
            join_lines.append(
                "INNER JOIN factor_{idx} USING (date_, code)".format(idx=idx)
            )

        return "\n".join(
            [
                "WITH",
                ",\n".join(ctes),
                *select_lines,
                *join_lines,
                "ORDER BY base.date_, base.code",
            ]
        )

    def _load_factor_frame_via_query(
        self,
        *,
        factor_names: Sequence[str],
        provider: str,
        freq: str,
        variant: str | None,
        codes: list[str] | None,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        store = self.store
        factor_specs: list[tuple[str, FactorSpec, Sequence[str]]] = []
        for factor_name in factor_names:
            spec = self._factor_spec(
                factor_name=factor_name,
                provider=provider,
                freq=freq,
                variant=variant,
            )
            paths = store._dataset_paths_for_query(spec, start=start, end=end)
            if not paths:
                continue
            factor_specs.append((factor_name, spec, paths))

        if not factor_specs:
            return pd.DataFrame(columns=["date_", "code", *factor_names])

        sql = self._build_multi_factor_frame_sql(
            factor_specs=factor_specs,
            codes=codes,
            start=start,
            end=end,
        )
        con = factor_store_store.ibis.duckdb.connect()
        frame = con.sql(sql).to_pandas()
        if frame.empty:
            return pd.DataFrame(columns=["date_", "code", *[name for name, _, _ in factor_specs]])

        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce").dt.tz_localize(None)
        frame["code"] = frame["code"].astype(str)
        for factor_name, _, _ in factor_specs:
            if factor_name in frame.columns:
                frame[factor_name] = pd.to_numeric(frame[factor_name], errors="coerce")
        frame = frame.dropna(subset=["date_", "code"])
        return frame.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)

    def load_factor_frame(
        self,
        *,
        factor_names: Iterable[str],
        provider: str = "tiger",
        freq: str = "1d",
        variant: str | None = None,
        codes: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load several stored factors and merge them into a long research frame."""
        factor_names = [str(name) for name in factor_names]
        if self._query_backend_available():
            return self._load_factor_frame_via_query(
                factor_names=factor_names,
                provider=provider,
                freq=freq,
                variant=variant,
                codes=codes,
                start=start,
                end=end,
            )

        panels = self.load_factor_panels(
            factor_names=factor_names,
            provider=provider,
            freq=freq,
            variant=variant,
            codes=codes,
            start=start,
            end=end,
        )
        if not panels:
            return pd.DataFrame(columns=["date_", "code", *factor_names])

        merged: pd.DataFrame | None = None
        for factor_name in factor_names:
            panel = panels.get(factor_name)
            if not isinstance(panel, pd.DataFrame) or panel.empty:
                continue
            long_df = (
                panel.rename_axis(index="date_")
                .reset_index()
                .melt(id_vars="date_", var_name="code", value_name=factor_name)
            )
            long_df["date_"] = pd.to_datetime(long_df["date_"], errors="coerce").dt.tz_localize(None)
            long_df["code"] = long_df["code"].astype(str)
            long_df[factor_name] = pd.to_numeric(long_df[factor_name], errors="coerce")
            long_df = long_df.dropna(subset=["date_", "code", factor_name])
            if merged is None:
                merged = long_df
            else:
                merged = merged.merge(long_df, on=["date_", "code"], how="inner")

        if merged is None:
            return pd.DataFrame(columns=["date_", "code", *factor_names])
        return merged.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)

    def align_fundamental_to_trading_dates(
        self,
        fundamentals: pd.DataFrame,
        trading_dates: pd.Index,
        *,
        value_columns: list[str],
        use_point_in_time: bool = False,
        availability_column: str | None = None,
        lag_sessions: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """
        Align fundamental-like data onto the trading-date axis.

        The default behavior keeps fundamentals one trading session behind
        prices, even when the caller does not request a full point-in-time
        alignment. That lets simple panel joins avoid vectorization while still
        preserving the conservative "available one session later" rule.
        """
        if use_point_in_time:
            return align_fundamental_point_in_time(
                fundamentals,
                trading_dates,
                value_columns=value_columns,
                availability_column=availability_column,
                lag_sessions=lag_sessions,
            )

        return align_fundamental_point_in_time(
            fundamentals,
            trading_dates,
            value_columns=value_columns,
            availability_column=None,
            lag_sessions=lag_sessions,
        )

    def align_entity_to_trading_dates(
        self,
        entities: pd.DataFrame,
        trading_dates: pd.Index,
        *,
        value_columns: list[str],
    ) -> dict[str, pd.DataFrame]:
        if entities.empty:
            return {column: pd.DataFrame(index=trading_dates) for column in value_columns}

        if "date_" not in entities.columns:
            static_entities = entities.copy()
            if "code" not in static_entities.columns:
                return {column: pd.DataFrame(index=trading_dates) for column in value_columns}
            static_entities["code"] = static_entities["code"].astype(str)
            aligned: dict[str, pd.DataFrame] = {}
            index = pd.DatetimeIndex(trading_dates)
            for column in value_columns:
                if column not in static_entities.columns:
                    aligned[column] = pd.DataFrame(index=index)
                    continue
                series = static_entities[["code", column]].dropna(subset=["code"]).groupby("code")[column].last()
                wide = pd.DataFrame(index=index, columns=series.index)
                for code, value in series.items():
                    wide[code] = value
                aligned[column] = wide
            return aligned

        aligned: dict[str, pd.DataFrame] = {}
        for column in value_columns:
            if column not in entities.columns:
                aligned[column] = pd.DataFrame(index=trading_dates)
                continue
            subset = entities[["date_", "code", column]].copy()
            subset["code"] = subset["code"].astype(str)
            wide = subset.pivot_table(index="date_", columns="code", values=column, aggfunc="last").sort_index()
            wide = wide.reindex(pd.DatetimeIndex(trading_dates), method="ffill")
            aligned[column] = wide
        return aligned

    def build_alpha101_input(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        classification_provider: str = "simfin",
        classification_dataset: str = "companies",
        lookback_days: int = 400,
        neutralization_columns: NeutralizationColumns | None = None,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        normalized_codes = list(dict.fromkeys(map(str, codes)))
        buffer_start = str((pd.Timestamp(start) - pd.Timedelta(days=lookback_days)).date())
        price_df = self.fetch_price_data(
            codes=normalized_codes,
            start=buffer_start,
            end=end,
            provider=price_provider,
            as_ex=as_ex,
        )
        if price_df.empty:
            return pd.DataFrame(columns=["date_", "code", "open", "high", "low", "close", "volume"])

        adjusted_price_df = _adjust_ohlcv_for_alpha101(price_df)
        keep_columns = [
            column
            for column in [
                "date_",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "dividend",
                "shares_outstanding",
                "market_value",
            ]
            if column in adjusted_price_df.columns
        ]
        base = adjusted_price_df[keep_columns].copy()
        trading_dates = pd.DatetimeIndex(sorted(base["date_"].dropna().unique()))

        neutralization_columns = neutralization_columns or NeutralizationColumns()
        companies = self.fetch_fundamental_data(
            provider=classification_provider,
            name=classification_dataset,
            freq="static" if classification_dataset == "companies" else "1d",
            codes=normalized_codes,
            start=buffer_start,
            end=end,
        )

        category_columns = [
            neutralization_columns.sector,
            neutralization_columns.industry,
            neutralization_columns.subindustry,
        ]
        aligned_categories = self.align_entity_to_trading_dates(
            companies,
            trading_dates,
            value_columns=category_columns,
        )
        aligned_numeric = self.align_fundamental_to_trading_dates(
            companies,
            trading_dates,
            value_columns=["shares_basic"],
        )

        merged = base.copy()
        for column_name, wide in aligned_categories.items():
            long_df = to_long_series(wide.reindex(columns=normalized_codes), column_name)
            merged = merged.merge(long_df, on=["date_", "code"], how="left")

        shares = aligned_numeric.get("shares_basic", pd.DataFrame(index=trading_dates))
        shares_long = to_long_series(shares.reindex(columns=normalized_codes), "shares_basic")
        merged = merged.merge(shares_long, on=["date_", "code"], how="left")

        if "shares_outstanding" in merged.columns:
            if "shares_basic" in merged.columns:
                merged["shares_outstanding"] = merged["shares_outstanding"].fillna(merged["shares_basic"])
        elif "shares_basic" in merged.columns:
            merged["shares_outstanding"] = merged["shares_basic"]

        if "shares_outstanding" in merged.columns:
            merged["market_value"] = merged["close"] * merged["shares_outstanding"]
        elif "shares_basic" in merged.columns:
            merged["market_value"] = merged["close"] * merged["shares_basic"].fillna(merged["volume"])
        else:
            merged["market_value"] = merged["close"] * merged["volume"]

        if "adj_close" in merged.columns:
            merged = merged.drop(columns=["adj_close"])
        merged = merged.sort_values(["code", "date_"]).reset_index(drop=True)
        return merged

    def build_gtja191_input(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        benchmark_code: str | None = None,
        benchmark_provider: str | None = None,
        include_fama3: bool = False,
        fama_provider: str = "simfin",
        fama3_panel: pd.DataFrame | None = None,
        lookback_days: int = 400,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        normalized_codes = list(dict.fromkeys(map(str, codes)))
        buffer_start = str((pd.Timestamp(start) - pd.Timedelta(days=lookback_days)).date())
        price_df = self.fetch_price_data(
            codes=normalized_codes,
            start=buffer_start,
            end=end,
            provider=price_provider,
            as_ex=as_ex,
        )
        if price_df.empty:
            return pd.DataFrame(columns=["date_", "code", "open", "high", "low", "close", "volume", "vwap", "index_open", "index_close"])

        adjusted_price_df = _adjust_ohlcv_for_alpha101(price_df)
        keep_columns = [
            column
            for column in [
                "date_",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "dividend",
                "shares_outstanding",
                "market_value",
            ]
            if column in adjusted_price_df.columns
        ]
        base = adjusted_price_df[keep_columns].copy()
        base["date_"] = pd.to_datetime(base["date_"], errors="coerce").dt.tz_localize(None)
        base["code"] = base["code"].astype(str)

        if benchmark_code is not None:
            benchmark_provider = benchmark_provider or price_provider or self.price_provider
            benchmark_df = self.fetch_price_data(
                codes=[str(benchmark_code)],
                start=buffer_start,
                end=end,
                provider=benchmark_provider,
                as_ex=as_ex,
            )
            if not benchmark_df.empty:
                benchmark_adjusted = _adjust_ohlcv_for_alpha101(benchmark_df)
                benchmark_frame = (
                    benchmark_adjusted[["date_", "open", "close"]]
                    .dropna(subset=["date_"])
                    .sort_values(["date_"], kind="stable")
                    .groupby("date_", as_index=False)
                    .agg({"open": "last", "close": "last"})
                    .rename(columns={"open": "index_open", "close": "index_close"})
                )
                base = base.merge(benchmark_frame, on="date_", how="left")

        if "index_open" not in base.columns:
            base["index_open"] = np.nan
        if "index_close" not in base.columns:
            base["index_close"] = np.nan

        if "vwap" not in base.columns:
            base["vwap"] = (base["open"] + base["high"] + base["low"] + base["close"]) / 4.0
        else:
            base["vwap"] = pd.to_numeric(base["vwap"], errors="coerce")
            base["vwap"] = base["vwap"].fillna((base["open"] + base["high"] + base["low"] + base["close"]) / 4.0)

        base["return"] = base.groupby("code", sort=False)["close"].pct_change(fill_method=None)

        if include_fama3:
            fama3 = fama3_panel
            if fama3 is None:
                fama3 = self.build_fama3_panel(
                    codes=normalized_codes,
                    start=buffer_start,
                    end=end,
                    price_provider=price_provider,
                    fama_provider=fama_provider,
                    as_ex=as_ex,
                )
            if not fama3.empty:
                fama_long = fama3.rename(columns={"mkt": "mkt", "smb": "smb", "hml": "hml"}).copy()
                base = base.merge(fama_long, on="date_", how="left")
        for column in ("mkt", "smb", "hml"):
            if column not in base.columns:
                base[column] = np.nan
        base = base.sort_values(["code", "date_"], kind="stable").reset_index(drop=True)
        return base

    def build_sunday100plus_input(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        classification_provider: str = "simfin",
        classification_dataset: str = "companies",
        lookback_days: int = 400,
        neutralization_columns: NeutralizationColumns | None = None,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        return self.build_alpha101_input(
            codes=codes,
            start=start,
            end=end,
            price_provider=price_provider,
            classification_provider=classification_provider,
            classification_dataset=classification_dataset,
            lookback_days=lookback_days,
            neutralization_columns=neutralization_columns,
            as_ex=as_ex,
        )

    def build_fama3_panel(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        fama_provider: str = "simfin",
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        normalized_codes = list(dict.fromkeys(map(str, codes)))
        buffer_start = str((pd.Timestamp(start) - pd.Timedelta(days=540)).date())
        price_df = self.fetch_price_data(
            codes=normalized_codes,
            start=buffer_start,
            end=end,
            provider=price_provider,
            as_ex=as_ex,
        )
        if price_df.empty:
            return pd.DataFrame(columns=["date_", "mkt", "smb", "hml"])

        adjusted_price_df = _adjust_ohlcv_for_alpha101(price_df)
        close = (
            adjusted_price_df.loc[:, ["date_", "code", "close"]]
            .pivot(index="date_", columns="code", values="close")
            .sort_index()
            .reindex(columns=normalized_codes)
        )
        returns = close.pct_change(fill_method=None)
        mkt = returns.mean(axis=1, skipna=True)

        balance = self.fetch_fundamental_data(
            name="balance_sheet",
            freq="1q",
            start=buffer_start,
            end=end,
            codes=normalized_codes,
            provider=fama_provider,
            as_ex=as_ex,
        )
        companies = self.fetch_fundamental_data(
            name="companies",
            freq="static",
            start=buffer_start,
            end=end,
            codes=normalized_codes,
            provider=fama_provider,
            as_ex=as_ex,
        )
        if balance.empty:
            raise ValueError("build_fama3_panel requires balance_sheet data with total_equity to compute HML.")
        trading_dates = pd.DatetimeIndex(close.index)
        share_source = (
            self.align_entity_to_trading_dates(companies, trading_dates, value_columns=["shares_basic"]).get("shares_basic", pd.DataFrame(index=close.index))
            if not companies.empty
            else pd.DataFrame(index=close.index)
        )
        shares = share_source.reindex(index=close.index, columns=normalized_codes)
        companies_aligned = self.align_fundamental_to_trading_dates(balance, trading_dates, value_columns=["total_equity"])["total_equity"]
        if shares.empty and "shares_outstanding" in adjusted_price_df.columns:
            shares = (
                adjusted_price_df.loc[:, ["date_", "code", "shares_outstanding"]]
                .pivot(index="date_", columns="code", values="shares_outstanding")
                .sort_index()
                .reindex(index=close.index, columns=normalized_codes)
            )
        if shares.empty:
            shares = pd.DataFrame(index=close.index, columns=normalized_codes)

        mcap = close * shares.reindex(index=close.index, columns=normalized_codes)
        mcap = mcap.ffill()
        mcap_lag = mcap.shift(1)
        book_equity = companies_aligned.reindex(index=close.index, columns=normalized_codes)
        book_to_market = book_equity.div(mcap_lag.replace(0, np.nan))

        size_cut = mcap_lag.median(axis=1)
        value_cut = book_to_market.median(axis=1)

        small_mask = mcap_lag.le(size_cut, axis=0)
        big_mask = mcap_lag.gt(size_cut, axis=0)
        high_mask = book_to_market.ge(value_cut, axis=0)
        low_mask = book_to_market.lt(value_cut, axis=0)

        smb = returns.where(small_mask).mean(axis=1, skipna=True) - returns.where(big_mask).mean(axis=1, skipna=True)
        hml = returns.where(high_mask).mean(axis=1, skipna=True) - returns.where(low_mask).mean(axis=1, skipna=True)

        fama3 = pd.DataFrame(
            {
                "date_": pd.DatetimeIndex(close.index),
                "mkt": pd.to_numeric(mkt, errors="coerce").to_numpy(),
                "smb": pd.to_numeric(smb, errors="coerce").to_numpy(),
                "hml": pd.to_numeric(hml, errors="coerce").to_numpy(),
            }
        )
        fama3 = fama3.dropna(subset=["date_"]).sort_values("date_", kind="stable").reset_index(drop=True)
        return fama3

    def fama_factor_names(self, family: int | str = 3, *, style: str = "abbr") -> tuple[str, ...]:
        """Return the canonical or descriptive names for a Fama family."""
        return _fama_factor_names(family, style=style)

    def save_fama_family(
        self,
        frame: pd.DataFrame,
        *,
        family: int | str = 3,
        provider: str = "tiger",
        variant: str | None = "fama",
        name_style: str = "abbr",
        force_updated: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> tuple[DatasetSaveResult, ...]:
        """Save a Fama-family wide frame as separate factor files.

        This is a thin convenience wrapper around ``save_factor(...)``.
        """
        family_id = _normalize_fama_family(family)
        target_names = _fama_factor_names(family_id, style=name_style)
        normalized = frame.copy()
        if "date_" not in normalized.columns:
            raise ValueError("Fama frame must contain a 'date_' column")
        normalized["date_"] = pd.to_datetime(normalized["date_"], errors="coerce").dt.tz_localize(None)
        normalized = normalized.dropna(subset=["date_"]).sort_values("date_", kind="stable").reset_index(drop=True)

        store = self.store
        results: list[DatasetSaveResult] = []
        for factor_name in target_names:
            source_column = _resolve_fama_column(normalized, factor_name)
            if source_column is None:
                raise ValueError(
                    f"Fama frame is missing required factor column for {factor_name!r}; "
                    f"available columns={list(normalized.columns)!r}"
                )
            factor_frame = normalized.loc[:, ["date_", source_column]].copy()
            factor_frame["code"] = factor_name
            factor_frame = factor_frame.rename(columns={source_column: "value"})[["date_", "code", "value"]]
            spec = FactorSpec(
                provider=provider,
                region=self.region,
                sec_type=self.sec_type,
                freq="1d",
                table_name=factor_name,
                variant=variant,
            )
            combined_metadata = dict(metadata or {})
            combined_metadata.setdefault("family", f"fama{family_id}")
            combined_metadata.setdefault("factor_name", factor_name)
            combined_metadata.setdefault("name_style", name_style)
            result = store.save_factor(
                spec,
                factor_frame,
                force_updated=force_updated,
                metadata=combined_metadata,
            )
            results.append(result)
        return tuple(results)

    def load_fama_family(
        self,
        *,
        family: int | str = 3,
        provider: str = "tiger",
        variant: str | None = "fama",
        name_style: str = "abbr",
        engine: str = "pandas",
    ) -> pd.DataFrame | object:
        """Load a Fama-family panel from factor files."""
        family_id = _normalize_fama_family(family)
        target_names = _fama_factor_names(family_id, style=name_style)
        store = self.store
        frames: list[pd.DataFrame] = []
        for factor_name in target_names:
            loaded: pd.DataFrame | object | None = None
            loaded_name: str | None = None
            for candidate_name in _fama_alias_candidates(factor_name):
                spec = FactorSpec(
                    provider=provider,
                    region=self.region,
                    sec_type=self.sec_type,
                    freq="1d",
                    table_name=candidate_name,
                    variant=variant,
                )
                candidate = store.get_factor(spec, engine="pandas")
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    loaded = candidate
                    loaded_name = candidate_name
                    break
            if not isinstance(loaded, pd.DataFrame) or loaded.empty:
                frames = []
                break
            if "value" not in loaded.columns or "date_" not in loaded.columns:
                raise ValueError(
                    f"loaded Fama factor {loaded_name or factor_name!r} has unexpected columns: {list(loaded.columns)!r}"
                )
            series = (
                loaded.loc[:, ["date_", "value"]]
                .dropna(subset=["date_"])
                .drop_duplicates(subset=["date_"], keep="last")
                .sort_values("date_", kind="stable")
                .rename(columns={"value": factor_name})
            )
            frames.append(series)

        if frames:
            panel = frames[0]
            for frame_part in frames[1:]:
                panel = panel.merge(frame_part, on="date_", how="outer", sort=True)
            panel = panel.sort_values("date_", kind="stable").reset_index(drop=True)
            if engine == "polars":
                return _to_polars_frame(panel)
            return panel

        if family_id == 3 and name_style in {"abbr", "short", "canonical"}:
            legacy_spec = OthersSpec(
                provider=provider,
                region=self.region,
                sec_type=self.sec_type,
                freq="1d",
                table_name="fama3",
                variant=None,
            )
            legacy = store.get_fama3(legacy_spec, engine="pandas")
            if isinstance(legacy, pd.DataFrame) and not legacy.empty:
                legacy = legacy.rename(columns={column: column.lower() for column in legacy.columns})
                columns = [column for column in ["date_", "mkt", "smb", "hml"] if column in legacy.columns]
                legacy = legacy.loc[:, columns]
                if engine == "polars":
                    return _to_polars_frame(legacy)
                return legacy

        empty = pd.DataFrame(columns=["date_", *target_names])
        return _to_polars_frame(empty) if engine == "polars" else empty

    def save_fama3(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        fama_provider: str = "simfin",
        table_name: str = "fama3",
        variant: str | None = "fama",
        name_style: str = "abbr",
        as_ex: bool | None = None,
        force_updated: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> DatasetSaveResult:
        """Build and persist the canonical Fama 3-factor panel.

        This is a thin convenience wrapper around ``save_factor(...)``.
        """
        panel = self.build_fama3_panel(
            codes=codes,
            start=start,
            end=end,
            price_provider=price_provider,
            fama_provider=fama_provider,
            as_ex=as_ex,
        )
        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault("family", "fama3")
        combined_metadata.setdefault("price_provider", price_provider or self.price_provider)
        combined_metadata.setdefault("fama_provider", fama_provider)
        combined_metadata.setdefault("codes", len(codes))
        combined_metadata.setdefault("table_name", table_name)
        combined_metadata.setdefault("variant", variant)
        combined_metadata.setdefault("provider", "tiger")
        results = self.save_fama_family(
            panel,
            family=3,
            provider="tiger",
            variant=variant,
            name_style=name_style,
            force_updated=force_updated,
            metadata=combined_metadata,
        )
        if not results:
            raise RuntimeError("save_fama3 produced no outputs")
        return results[0]

    def save_fama5(
        self,
        *,
        frame: pd.DataFrame,
        provider: str = "tiger",
        variant: str | None = "fama",
        name_style: str = "abbr",
        force_updated: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> tuple[DatasetSaveResult, ...]:
        """Persist a Fama 5-factor wide frame as factor files.

        This is a thin convenience wrapper around ``save_factor(...)``.
        """
        return self.save_fama_family(
            frame,
            family=5,
            provider=provider,
            variant=variant,
            name_style=name_style,
            force_updated=force_updated,
            metadata=metadata,
        )

    def save_fama6(
        self,
        *,
        frame: pd.DataFrame,
        provider: str = "tiger",
        variant: str | None = "fama",
        name_style: str = "abbr",
        force_updated: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> tuple[DatasetSaveResult, ...]:
        """Persist a Fama 6-factor wide frame as factor files.

        This is a thin convenience wrapper around ``save_factor(...)``.
        """
        return self.save_fama_family(
            frame,
            family=6,
            provider=provider,
            variant=variant,
            name_style=name_style,
            force_updated=force_updated,
            metadata=metadata,
        )

    def load_fama3(
        self,
        *,
        table_name: str = "fama3",
        variant: str | None = "fama",
        provider: str = "tiger",
        name_style: str = "abbr",
        engine: str = "pandas",
    ) -> pd.DataFrame | object:
        """Load the canonical Fama 3-factor panel."""
        panel = self.load_fama_family(
            family=3,
            provider=provider,
            variant=variant,
            name_style=name_style,
            engine=engine,
        )
        if isinstance(panel, pd.DataFrame) and not panel.empty and table_name and table_name != "fama3":
            panel = panel.copy()
            panel.attrs["table_name"] = table_name
        return panel

    def load_fama5(
        self,
        *,
        provider: str = "tiger",
        variant: str | None = "fama",
        name_style: str = "abbr",
        engine: str = "pandas",
    ) -> pd.DataFrame | object:
        """Load a Fama 5-factor panel."""
        return self.load_fama_family(
            family=5,
            provider=provider,
            variant=variant,
            name_style=name_style,
            engine=engine,
        )

    def load_fama6(
        self,
        *,
        provider: str = "tiger",
        variant: str | None = "fama",
        name_style: str = "abbr",
        engine: str = "pandas",
    ) -> pd.DataFrame | object:
        """Load a Fama 6-factor panel."""
        return self.load_fama_family(
            family=6,
            provider=provider,
            variant=variant,
            name_style=name_style,
            engine=engine,
        )

    def save_factor(
        self,
        *,
        factor_name: str,
        factor_df: pd.DataFrame,
        spec: FactorSpec | None = None,
        force_updated: bool = False,
        metadata: dict | None = None,
    ) -> FactorResult:
        frame = factor_df.copy()
        value_column = factor_name if factor_name in frame.columns else "value"
        if value_column not in frame.columns:
            candidate_columns = [column for column in frame.columns if column not in {"date_", "code"}]
            if len(candidate_columns) == 1:
                value_column = candidate_columns[0]
            else:
                raise ValueError(
                    "factor frame must contain either the factor column or a single value column; "
                    f"got columns={list(frame.columns)!r}"
                )
        if value_column != "value":
            frame = frame.rename(columns={value_column: "value"})
        if not {"date_", "code", "value"}.issubset(frame.columns):
            missing = sorted({"date_", "code", "value"} - set(frame.columns))
            raise ValueError(f"factor frame is missing required columns: {missing}")

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("factor_name", factor_name)
        resolved_spec = spec or FactorSpec(
            region=str(metadata_payload.pop("region", self.region)),
            sec_type=str(metadata_payload.pop("sec_type", self.sec_type)),
            freq=str(metadata_payload.pop("freq", "1d")),
            table_name=factor_name,
            variant=metadata_payload.pop("variant", None),
            provider=metadata_payload.pop("provider", None),
        )
        data_store = self.store
        storage_result = data_store.save_factor(
            resolved_spec,
            frame.loc[:, ["date_", "code", "value"]],
            force_updated=force_updated,
            metadata=metadata_payload,
        )
        self.log(f"saved factor {factor_name} -> {storage_result.files[0]}")
        return FactorResult(
            name=factor_name,
            data=factor_df,
            parquet_path=storage_result.files[0],
            metadata_path=storage_result.manifest_path,
        )

    def momentum_12m_1m(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> FactorResult:
        close = self.price_panel(codes=codes, start=start, end=end, field="close", as_ex=as_ex)
        factor = close.apply(lambda column: accelerated_momentum_12m_1m(column, 252, 21), axis=0)
        long_df = to_long_factor(factor, "momentum_12m_1m")
        return self.save_factor(
            factor_name="momentum_12m_1m",
            factor_df=long_df,
            metadata={"provider": self.price_provider, "family": "price", "window_long": 252, "window_short": 21},
        )

    def volatility_3m(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> FactorResult:
        close = self.price_panel(codes=codes, start=start, end=end, field="close", as_ex=as_ex)
        returns = close.pct_change()
        factor = returns.apply(lambda column: accelerated_rolling_std(column, 63), axis=0)
        long_df = to_long_factor(factor, "volatility_3m")
        return self.save_factor(
            factor_name="volatility_3m",
            factor_df=long_df,
            metadata={"provider": self.price_provider, "family": "price", "window": 63},
        )

    def size_market_cap(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> FactorResult:
        close = self.price_panel(codes=codes, start=start, end=end, field="close", as_ex=as_ex)
        trading_dates = close.index
        companies = self.fetch_fundamental_data(name="companies", freq="static", start=start, end=end, codes=codes)
        aligned = self.align_fundamental_to_trading_dates(companies, trading_dates, value_columns=["shares_basic"])
        shares = aligned["shares_basic"].reindex(columns=close.columns)
        factor = close * shares
        long_df = to_long_factor(factor, "size_market_cap")
        return self.save_factor(
            factor_name="size_market_cap",
            factor_df=long_df,
            metadata={"provider": "simfin", "family": "fundamental_price_mix"},
        )

    def quality_composite(
        self,
        *,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> FactorResult:
        buffer_start = str((pd.Timestamp(start) - pd.Timedelta(days=500)).date())
        close = self.price_panel(codes=codes, start=buffer_start, end=end, field="close", as_ex=as_ex)
        trading_dates = close.index

        balance = self.fetch_fundamental_data(name="balance_sheet", freq="1q", start=buffer_start, end=end, codes=codes, as_ex=as_ex)
        income = self.fetch_fundamental_data(name="income_statement", freq="1q", start=buffer_start, end=end, codes=codes, as_ex=as_ex)
        cashflow = self.fetch_fundamental_data(name="cashflow_statement", freq="1q", start=buffer_start, end=end, codes=codes, as_ex=as_ex)

        bal = self.align_fundamental_to_trading_dates(
            balance,
            trading_dates,
            value_columns=["total_assets", "total_equity", "total_liabilities", "total_current_assets", "total_current_liabilities"],
        )
        inc = self.align_fundamental_to_trading_dates(income, trading_dates, value_columns=["net_income", "revenue"])
        cfo = self.align_fundamental_to_trading_dates(cashflow, trading_dates, value_columns=["net_cfo"])

        total_assets = bal["total_assets"].reindex(columns=close.columns)
        total_equity = bal["total_equity"].reindex(columns=close.columns)
        total_liabilities = bal["total_liabilities"].reindex(columns=close.columns)
        current_assets = bal["total_current_assets"].reindex(columns=close.columns)
        current_liabilities = bal["total_current_liabilities"].reindex(columns=close.columns)
        net_income = inc["net_income"].reindex(columns=close.columns)
        revenue = inc["revenue"].reindex(columns=close.columns)
        net_cfo = cfo["net_cfo"].reindex(columns=close.columns)

        components = {
            "roa": net_income / total_assets.replace(0, np.nan),
            "roe": net_income / total_equity.replace(0, np.nan),
            "current_ratio": current_assets / current_liabilities.replace(0, np.nan),
            "cash_conversion": net_cfo / net_income.abs().replace(0, np.nan),
            "asset_turnover": revenue / total_assets.replace(0, np.nan),
            "leverage": -(total_liabilities / total_assets.replace(0, np.nan)),
        }

        zscores: list[pd.DataFrame] = []
        for frame in components.values():
            zscores.append(frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1, ddof=0).replace(0, np.nan), axis=0))
        composite = pd.concat(zscores).groupby(level=0).mean()
        composite = composite.loc[composite.index >= pd.Timestamp(start)]
        long_df = to_long_factor(composite, "quality_composite")
        return self.save_factor(
            factor_name="quality_composite",
            factor_df=long_df,
            metadata={"provider": "simfin+yahoo", "family": "fundamental", "components": list(components.keys())},
        )

    def alpha101(
        self,
        *,
        alpha_id: int,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        classification_provider: str = "simfin",
        classification_dataset: str = "companies",
        lookback_days: int = 400,
        neutralization_columns: NeutralizationColumns | None = None,
        as_ex: bool | None = None,
    ) -> FactorResult:
        factor_name = f"alpha_{alpha_id:03d}"
        alpha_input = self.build_alpha101_input(
            codes=codes,
            start=start,
            end=end,
            price_provider=price_provider,
            classification_provider=classification_provider,
            classification_dataset=classification_dataset,
            lookback_days=lookback_days,
            neutralization_columns=neutralization_columns,
            as_ex=as_ex,
        )
        engine = Alpha101Engine(alpha_input, neutralization_columns=neutralization_columns)
        factor_df = self._filter_by_date_range(engine.compute(alpha_id), start=start, end=end)
        return self.save_factor(
            factor_name=factor_name,
            factor_df=factor_df,
            metadata={
                "provider": price_provider or self.price_provider,
                "classification_provider": classification_provider,
                "classification_dataset": classification_dataset,
                "family": "alpha101",
                "alpha_id": int(alpha_id),
                "lookback_days": int(lookback_days),
            },
        )

    def alpha101_parallel(
        self,
        *,
        alpha_ids: list[int] | str | None = None,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        classification_provider: str = "simfin",
        classification_dataset: str = "companies",
        compute_workers: int | None = None,
        save_workers: int | None = None,
        save_factors: bool = True,
        as_ex: bool | None = None,
    ):
        from tiger_factors.factor_maker.vectorization.indicators import Alpha101IndicatorTransformer

        transformer = Alpha101IndicatorTransformer(
            calendar="XNYS",
            start=start,
            end=end,
            universe_provider="github",
            universe_name="sp500_constituents",
            price_provider=price_provider or self.price_provider,
            price_name="eod_price" if (price_provider or self.price_provider).lower() == "simfin" else "price",
            classification_provider=classification_provider,
            classification_company_name=classification_dataset,
            classification_industry_name="industry",
            region=self.region,
            sec_type=self.sec_type,
        )
        return transformer.compute_all_alpha101_parallel(
            alpha_ids=alpha_ids,
            codes=codes,
            start=start,
            end=end,
            dividends=False,
            history=False,
            output_dir=self.output_dir,
            compute_workers=compute_workers,
            save_workers=save_workers,
            save_factors=save_factors,
            as_ex=as_ex,
        )

    def gtja191(
        self,
        *,
        alpha_id: int,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        benchmark_code: str | None = None,
        benchmark_provider: str | None = None,
        fama_provider: str = "simfin",
        use_cached_fama3: bool = True,
        fama3_table_name: str = "fama3",
        fama3_variant: str | None = "fama",
        lookback_days: int = 400,
        as_ex: bool | None = None,
    ) -> FactorResult:
        """Compute GTJA191 factors.

        Alpha 030 consumes the canonical Fama 3-factor cache by default:
        ``mkt__fama.parquet``, ``smb__fama.parquet``, and ``hml__fama.parquet``.
        """
        factor_name = f"alpha_{alpha_id:03d}"
        fama3_panel = None
        if int(alpha_id) == 30 and use_cached_fama3:
            try:
                fama3_panel = self.load_fama3(
                    table_name=fama3_table_name,
                    variant=fama3_variant,
                    provider="tiger",
                    engine="pandas",
                )
                self.log(f"loaded cached fama3 panel from factor store: {fama3_table_name}")
            except FileNotFoundError:
                fama3_panel = None
        gtja_input = self.build_gtja191_input(
            codes=codes,
            start=start,
            end=end,
            price_provider=price_provider,
            benchmark_code=benchmark_code,
            benchmark_provider=benchmark_provider,
            include_fama3=(int(alpha_id) == 30),
            fama_provider=fama_provider,
            fama3_panel=fama3_panel,
            lookback_days=lookback_days,
            as_ex=as_ex,
        )
        engine = GTJA191Engine(gtja_input)
        factor_df = self._filter_by_date_range(engine.compute(alpha_id), start=start, end=end)
        return self.save_factor(
            factor_name=factor_name,
            factor_df=factor_df,
            metadata={
                "provider": "gtja",
                "price_provider": price_provider or self.price_provider,
                "benchmark_provider": benchmark_provider or price_provider or self.price_provider,
                "benchmark_code": benchmark_code,
                "family": "gtja191",
                "alpha_id": int(alpha_id),
                "lookback_days": int(lookback_days),
            },
        )

    def sunday100plus(
        self,
        *,
        alpha_id: int,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        classification_provider: str = "simfin",
        classification_dataset: str = "companies",
        lookback_days: int = 400,
        neutralization_columns: NeutralizationColumns | None = None,
        formula_registry: Mapping[int, Callable[[Sunday100PlusEngine], pd.Series]] | None = None,
        formula_descriptions: Mapping[int, str] | None = None,
        as_ex: bool | None = None,
    ) -> FactorResult:
        factor_name = f"alpha_{alpha_id:03d}"
        sunday_input = self.build_sunday100plus_input(
            codes=codes,
            start=start,
            end=end,
            price_provider=price_provider,
            classification_provider=classification_provider,
            classification_dataset=classification_dataset,
            lookback_days=lookback_days,
            neutralization_columns=neutralization_columns,
            as_ex=as_ex,
        )
        engine = Sunday100PlusEngine(
            sunday_input,
            neutralization_columns=neutralization_columns,
            formula_registry=formula_registry,
            formula_descriptions=formula_descriptions,
        )
        factor_df = self._filter_by_date_range(engine.compute(alpha_id), start=start, end=end)
        return self.save_factor(
            factor_name=factor_name,
            factor_df=factor_df,
            metadata={
                "provider": "sunday100plus",
                "price_provider": price_provider or self.price_provider,
                "classification_provider": classification_provider,
                "classification_dataset": classification_dataset,
                "family": "sunday100plus",
                "alpha_id": int(alpha_id),
                "lookback_days": int(lookback_days),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-factor parquet files from tiger_api data.")
    parser.add_argument("--factor", required=True)
    parser.add_argument("--family", choices=["alpha101", "gtja191", "sunday100plus"], default="alpha101")
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", default=str(DEFAULT_FACTOR_STORE_ROOT))
    parser.add_argument("--price-provider", default="yahoo")
    parser.add_argument("--fama-provider", default="simfin")
    parser.add_argument("--as-ex", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = TigerFactorLibrary(
        output_dir=args.output_dir,
        price_provider=args.price_provider,
        verbose=not args.quiet,
    )
    builders: dict[str, Callable[..., FactorResult]] = {
        "momentum_12m_1m": library.momentum_12m_1m,
        "volatility_3m": library.volatility_3m,
        "size_market_cap": library.size_market_cap,
        "quality_composite": library.quality_composite,
    }
    if args.factor in builders:
        result = builders[args.factor](codes=args.codes, start=args.start, end=args.end, as_ex=args.as_ex)
    elif args.factor.startswith("alpha_") and args.factor.split("_")[-1].isdigit():
        alpha_id = int(args.factor.split("_")[-1])
        if args.family == "gtja191":
            result = library.gtja191(
                alpha_id=alpha_id,
                codes=args.codes,
                start=args.start,
                end=args.end,
                price_provider=args.price_provider,
                fama_provider=args.fama_provider,
                as_ex=args.as_ex,
            )
        elif args.family == "sunday100plus":
            result = library.sunday100plus(
                alpha_id=alpha_id,
                codes=args.codes,
                start=args.start,
                end=args.end,
                price_provider=args.price_provider,
                as_ex=args.as_ex,
            )
        else:
            result = library.alpha101(
                alpha_id=alpha_id,
                codes=args.codes,
                start=args.start,
                end=args.end,
                price_provider=args.price_provider,
                as_ex=args.as_ex,
            )
    else:
        raise ValueError(f"Unsupported factor: {args.factor}")
    print(
        json.dumps(
            {
                "factor": result.name,
                "rows": int(len(result.data)),
                "parquet_path": str(result.parquet_path),
                "metadata_path": str(result.metadata_path),
            },
            indent=2,
            default=str,
        )
    )
