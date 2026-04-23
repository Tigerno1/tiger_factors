from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from tiger_factors.factor_algorithm.alpha101 import Alpha101Engine
from tiger_factors.factor_algorithm.alpha101 import build_code_industry_frame
from tiger_api.sdk.client import fetch_codes
from tiger_api.sdk.client import fetch_data
from tiger_factors.utils.calculation.types import Interval
from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.utils.factor_parallel import run_factor_tasks_parallel
from tiger_reference.adjustments import adj_df


DEFAULT_SP500_CONSTITUENTS_REQUEST = {
    "provider": "github",
    "region": "us",
    "sec_type": "stock",
    "freq": "1d",
    "name": "sp500_constituents",
    "variant": None,
}

DEFAULT_SP500_PRICE_REQUEST = {
    "provider": "simfin",
    "region": "us",
    "sec_type": "stock",
    "freq": "1d",
    "name": "eod_price",
    "variant": None,
}

DEFAULT_SIMFIN_COMPANIES_REQUEST = {
    "provider": "simfin",
    "region": "us",
    "sec_type": "stock",
    "freq": "static",
    "name": "companies",
    "variant": None,
}

DEFAULT_SIMFIN_INDUSTRY_REQUEST = {
    "provider": "simfin",
    "region": "us",
    "sec_type": "stock",
    "freq": "static",
    "name": "industry",
    "variant": None,
}

DEFAULT_ALPHA101_DATA_STORE_ROOT = Path("/Volumes/Quant_Disk/factors")

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "dividend",
    "volume",
    "shares_outstanding",
]


@dataclass(frozen=True)
class Alpha101IndicatorResult:
    codes: list[str]
    raw_price_frame: pd.DataFrame
    calendar_frame: pd.DataFrame
    adjusted_frame: pd.DataFrame
    companies_frame: pd.DataFrame
    industry_frame: pd.DataFrame
    classification_frame: pd.DataFrame
    alpha_input_frame: pd.DataFrame
    factor_frame: pd.DataFrame
    alpha_ids: list[int]
    as_of: date | None = None
    saved_factor_paths: dict[str, str] | None = None
    saved_metadata_paths: dict[str, str] | None = None
    saved_adjusted_price_path: str | None = None


def _normalize_code_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, pd.DataFrame):
        for candidate in ("code", "symbol", "ticker"):
            if candidate in values.columns:
                values = values[candidate].tolist()
                break
        else:
            return []
    elif isinstance(values, pd.Series):
        values = values.tolist()
    elif isinstance(values, str):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = list(values)

    codes: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        codes.append(normalized)
    return codes


def _coerce_datetime(value: date | str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _resolve_alpha_ids(alpha_ids: Sequence[int] | str | None) -> list[int]:
    if alpha_ids is None:
        return list(range(1, 102))
    if isinstance(alpha_ids, str):
        normalized = alpha_ids.strip().lower()
        if normalized == "all":
            return list(range(1, 102))
        raise ValueError("alpha_ids must be 'all' or a sequence of integers.")
    return [int(alpha_id) for alpha_id in alpha_ids]


def _build_alpha101_metadata(*, provider: str, classification_provider: str, classification_dataset: str, alpha_id: int, execution_mode: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "classification_provider": classification_provider,
        "classification_dataset": classification_dataset,
        "family": "alpha101",
        "alpha_id": int(alpha_id),
        "parallel": execution_mode,
    }


def _save_adjusted_price_frame(
    *,
    data_store: FactorStore,
    adjusted_frame: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> str:
    storage_frame = adjusted_frame.copy()
    if "close" in storage_frame.columns:
        storage_frame["adj_close"] = pd.to_numeric(storage_frame["close"], errors="coerce")
    spec = AdjPriceSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        variant="fwd",
        provider="tiger",
    )
    result = data_store.save_adj_price(
        spec,
        storage_frame,
        force_updated=True,
        metadata=metadata,
    )
    return str(result.files[0])


def _save_alpha101_factor_frame(
    *,
    data_store: FactorStore,
    factor_name: str,
    factor_df: pd.DataFrame,
    metadata: dict[str, Any] | None,
) -> tuple[str, str]:
    storage_frame = factor_df.copy()
    if "value" not in storage_frame.columns:
        if factor_name not in storage_frame.columns:
            raise KeyError(f"factor frame must contain '{factor_name}' or 'value'")
        storage_frame = storage_frame.rename(columns={factor_name: "value"})
    spec = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name=factor_name,
        variant=None,
        provider="tiger",
    )
    result = data_store.save_factor(
        spec,
        storage_frame,
        force_updated=True,
        metadata=metadata,
    )
    return str(result.files[0]), str(result.manifest_path)


_PARALLEL_ALPHA101_ENGINE: Alpha101Engine | None = None


def _parallel_alpha101_init(
    alpha_input_frame: pd.DataFrame,
    neutralization_columns: Any | None,
) -> None:
    global _PARALLEL_ALPHA101_ENGINE
    _PARALLEL_ALPHA101_ENGINE = Alpha101Engine(
        alpha_input_frame,
        neutralization_columns=neutralization_columns,
    )


def _parallel_alpha101_compute(alpha_id: int) -> tuple[int, pd.DataFrame]:
    if _PARALLEL_ALPHA101_ENGINE is None:
        raise RuntimeError("Parallel Alpha101 engine was not initialized.")
    return int(alpha_id), _PARALLEL_ALPHA101_ENGINE.compute(int(alpha_id))


class Alpha101IndicatorTransformer:
    """Build an Alpha101-ready panel using the shared factor/pipeline path."""

    def __init__(
        self,
        *,
        calendar: str = "XNYS",
        start: date | str | pd.Timestamp,
        end: date | str | pd.Timestamp | None = None,
        interval: Interval | None = None,
        lag: int = 1,
        universe_provider: str = "github",
        universe_name: str = "sp500_constituents",
        price_provider: str = "simfin",
        price_name: str = "eod_price",
        classification_provider: str = "simfin",
        classification_company_name: str = "companies",
        classification_industry_name: str = "industry",
        region: str = "us",
        sec_type: str = "stock",
    ) -> None:
        self.calendar = calendar
        self._start = pd.Timestamp(start).date()
        self._end = pd.Timestamp(end).date() if end is not None else pd.Timestamp.today(tz="UTC").date()
        self.interval = interval
        self.lag = int(lag)
        self.universe_provider = universe_provider
        self.universe_name = universe_name
        self.price_provider = price_provider
        self.price_name = price_name
        self.classification_provider = classification_provider
        self.classification_company_name = classification_company_name
        self.classification_industry_name = classification_industry_name
        self.region = region
        self.sec_type = sec_type

    @property
    def start(self) -> date:
        return self._start

    @property
    def end(self) -> date:
        return self._end

    def resolve_codes(
        self,
        *,
        as_of: date | str | pd.Timestamp | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        as_ex: bool | None = None,
    ) -> list[str]:
        request = dict(DEFAULT_SP500_CONSTITUENTS_REQUEST)
        request.update(
            {
                "provider": self.universe_provider,
                "name": self.universe_name,
                "region": self.region,
                "sec_type": self.sec_type,
            }
        )
        if as_of is not None:
            request["at"] = pd.Timestamp(as_of).date()
        else:
            if start is not None:
                request["start"] = pd.Timestamp(start).date()
            if end is not None:
                request["end"] = pd.Timestamp(end).date()
            if "start" not in request and "end" not in request:
                request["at"] = self.start
        if as_ex is not None:
            request["as_ex"] = as_ex
        codes = fetch_codes(**request)
        return _normalize_code_list(codes)

    def fetch_price_data(
        self,
        *,
        codes: Sequence[str],
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        request = dict(DEFAULT_SP500_PRICE_REQUEST)
        request.update(
            {
                "provider": self.price_provider,
                "name": self.price_name,
                "region": self.region,
                "sec_type": self.sec_type,
                "codes": _normalize_code_list(codes),
                "return_type": "df",
            }
        )
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        request["start"] = pd.Timestamp(start).date()
        request["end"] = pd.Timestamp(end).date()
        if as_ex is not None:
            request["as_ex"] = as_ex

        try:
            rows = fetch_data(**request)
        except TypeError:
            request.pop("start", None)
            request.pop("end", None)
            request["at"] = pd.Timestamp(start).date()
            rows = fetch_data(**request)

        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if frame.empty:
            return frame
        if "date_" not in frame.columns and "date" in frame.columns:
            frame = frame.rename(columns={"date": "date_"})
        if "code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "code"})
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.upper()
        if "date_" in frame.columns:
            frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        for column in PRICE_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["date_", "code"]).sort_values(["date_", "code"]).reset_index(drop=True)

    def fetch_classification_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        company_request = dict(DEFAULT_SIMFIN_COMPANIES_REQUEST)
        company_request.update(
            {
                "provider": self.classification_provider,
                "name": self.classification_company_name,
                "region": self.region,
                "sec_type": self.sec_type,
                "return_type": "df",
            }
        )
        industry_request = dict(DEFAULT_SIMFIN_INDUSTRY_REQUEST)
        industry_request.update(
            {
                "provider": self.classification_provider,
                "name": self.classification_industry_name,
                "region": self.region,
                "sec_type": self.sec_type,
                "return_type": "df",
            }
        )

        companies = fetch_data(**company_request)
        industries = fetch_data(**industry_request)

        companies_frame = companies.copy() if isinstance(companies, pd.DataFrame) else pd.DataFrame(companies)
        industries_frame = industries.copy() if isinstance(industries, pd.DataFrame) else pd.DataFrame(industries)
        if companies_frame.empty or industries_frame.empty:
            empty_classification = pd.DataFrame(columns=["code", "industry_id", "industry", "sector"])
            return empty_classification, companies_frame, industries_frame

        classification = build_code_industry_frame(companies_frame, industries_frame)
        if classification.empty:
            empty_classification = pd.DataFrame(columns=["code", "industry_id", "industry", "sector"])
            return empty_classification, companies_frame, industries_frame

        classification["code"] = classification["code"].astype(str).str.upper()
        keep_columns = [column for column in ["code", "industry_id", "industry", "sector"] if column in classification.columns]
        classification = classification[keep_columns].drop_duplicates(subset=["code"]).sort_values(["code"]).reset_index(drop=True)
        return classification, companies_frame, industries_frame

    def merge_classification(
        self,
        frame: pd.DataFrame,
        classification_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        if classification_frame.empty:
            result = frame.copy()
            for column in ("industry_id", "industry", "sector"):
                if column not in result.columns:
                    result[column] = pd.NA
            return result

        result = frame.merge(classification_frame, on="code", how="left")
        for column in ("industry_id", "industry", "sector"):
            if column not in result.columns:
                result[column] = pd.NA
        return result

    def merge_calendar(
        self,
        price_frame: pd.DataFrame,
        *,
        codes: Sequence[str],
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = price_frame.copy()
        if frame.empty:
            return frame
        if "date_" in frame.columns:
            frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce").dt.tz_localize(None)
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.upper()
        normalized_codes = set(_normalize_code_list(codes))
        if normalized_codes and "code" in frame.columns:
            frame = frame[frame["code"].isin(normalized_codes)].copy()
        if start is not None:
            frame = frame[frame["date_"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date_"] <= pd.Timestamp(end)]
        return frame.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)

    def adjust_prices(
        self,
        frame: pd.DataFrame,
        *,
        dividends: bool = False,
        history: bool = False,
    ) -> pd.DataFrame:
        adjusted = adj_df(
            frame,
            drop_adj_close=False,
            dividends=dividends,
            history=history,
        )
        return adjusted if isinstance(adjusted, pd.DataFrame) else pd.DataFrame(adjusted)

    def build_alpha101_input(self, adjusted_frame: pd.DataFrame) -> pd.DataFrame:
        frame = adjusted_frame.copy()
        drop_columns = [
            "exchange_date_",
            "eff_at",
            "available_time",
            "trading_day",
            "step_index",
            "step_kind",
            "session_open",
            "session_close",
            "is_session_open",
            "is_session_close",
        ]
        frame = frame.drop(columns=[column for column in drop_columns if column in frame.columns], errors="ignore")

        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.upper()
        if "date_" not in frame.columns:
            raise KeyError("adjusted frame must contain date_.")
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce").dt.tz_localize(None)

        for column in ("open", "high", "low", "close", "volume", "dividend", "shares_outstanding"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "vwap" not in frame.columns:
            frame["vwap"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
        else:
            frame["vwap"] = pd.to_numeric(frame["vwap"], errors="coerce")
            frame["vwap"] = frame["vwap"].fillna((frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0)

        frame["return"] = frame.groupby("code", sort=False)["close"].pct_change(fill_method=None)

        if "shares_outstanding" in frame.columns:
            frame["market_value"] = frame["close"] * frame["shares_outstanding"]
        else:
            frame["market_value"] = frame["close"] * frame["volume"]

        return frame.sort_values(["code", "date_"], kind="stable").reset_index(drop=True)

    def compute_alpha101(
        self,
        alpha_id: int | str,
        *,
        codes: Sequence[str] | None = None,
        as_of: date | str | pd.Timestamp | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        dividends: bool = False,
        history: bool = False,
        save_factors: bool = False,
        save_adj_price: bool = False,
        output_dir: str | Path | None = None,
        compute_workers: int | None = None,
        save_workers: int | None = None,
        as_ex: bool | None = None,
    ) -> Alpha101IndicatorResult:
        if isinstance(alpha_id, str) and alpha_id.strip().lower() == "all":
            return self.compute_all_alpha101_parallel(
                alpha_ids="all",
                codes=codes,
                as_of=as_of,
                start=start,
                end=end,
                dividends=dividends,
                history=history,
                output_dir=output_dir,
                compute_workers=compute_workers,
                save_workers=save_workers,
                save_factors=save_factors,
                save_adj_price=save_adj_price,
                as_ex=as_ex,
            )

        resolved_alpha_id = int(alpha_id)
        resolved_start = self.start if start is None else pd.Timestamp(start).date()
        resolved_end = self.end if end is None else pd.Timestamp(end).date()
        data_store = FactorStore(output_dir or DEFAULT_ALPHA101_DATA_STORE_ROOT)
        resolved_codes = _normalize_code_list(codes) if codes is not None else self.resolve_codes(
            as_of=as_of,
            start=resolved_start,
            end=resolved_end,
            as_ex=as_ex,
        )
        if not resolved_codes:
            raise ValueError("No codes resolved for Alpha101 input.")

        raw_price_frame = self.fetch_price_data(codes=resolved_codes, start=resolved_start, end=resolved_end, as_ex=as_ex)
        if raw_price_frame.empty:
            raise ValueError("No price rows returned for Alpha101 input.")

        classification_frame, companies_frame, industry_frame = self.fetch_classification_data()
        calendar_frame = self.merge_calendar(raw_price_frame, codes=resolved_codes, start=resolved_start, end=resolved_end)
        adjusted_frame = self.adjust_prices(calendar_frame, dividends=dividends, history=history)
        adjusted_frame = self.merge_classification(adjusted_frame, classification_frame)
        alpha_input_frame = self.build_alpha101_input(adjusted_frame)
        engine = Alpha101Engine(alpha_input_frame)
        factor_frame = engine.compute(resolved_alpha_id)
        saved_adjusted_price_path = (
            _save_adjusted_price_frame(
                data_store=data_store,
                adjusted_frame=adjusted_frame,
                metadata={
                    "provider": self.price_provider,
                    "classification_provider": self.classification_provider,
                    "classification_dataset": self.classification_company_name,
                    "family": "alpha101",
                    "execution_mode": "single",
                    "output_root": str(data_store.root_dir),
                },
            )
            if save_adj_price
            else None
        )
        saved_factor_paths: dict[str, str] | None = None
        saved_metadata_paths: dict[str, str] | None = None
        if save_factors:
            factor_name = f"alpha_{resolved_alpha_id:03d}"
            metadata = _build_alpha101_metadata(
                provider=self.price_provider,
                classification_provider=self.classification_provider,
                classification_dataset=self.classification_company_name,
                alpha_id=resolved_alpha_id,
                execution_mode="single",
            )
            factor_path, metadata_path = _save_alpha101_factor_frame(
                data_store=data_store,
                factor_name=factor_name,
                factor_df=factor_frame,
                metadata=metadata,
            )
            saved_factor_paths = {factor_name: factor_path}
            saved_metadata_paths = {factor_name: metadata_path}
        return Alpha101IndicatorResult(
            codes=resolved_codes,
            raw_price_frame=raw_price_frame,
            calendar_frame=calendar_frame,
            adjusted_frame=adjusted_frame,
            companies_frame=companies_frame,
            industry_frame=industry_frame,
            classification_frame=classification_frame,
            alpha_input_frame=alpha_input_frame,
            factor_frame=factor_frame,
            alpha_ids=[resolved_alpha_id],
            as_of=pd.Timestamp(as_of).date() if as_of is not None else None,
            saved_factor_paths=saved_factor_paths,
            saved_metadata_paths=saved_metadata_paths,
            saved_adjusted_price_path=saved_adjusted_price_path,
        )

    def compute_all_alpha101(
        self,
        alpha_ids: Sequence[int] | str | None = None,
        *,
        codes: Sequence[str] | None = None,
        as_of: date | str | pd.Timestamp | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        dividends: bool = False,
        history: bool = False,
        save_factors: bool = False,
        save_adj_price: bool = False,
        output_dir: str | Path | None = None,
        as_ex: bool | None = None,
    ) -> Alpha101IndicatorResult:
        resolved_alpha_ids = _resolve_alpha_ids(alpha_ids)
        resolved_start = self.start if start is None else pd.Timestamp(start).date()
        resolved_end = self.end if end is None else pd.Timestamp(end).date()
        data_store = FactorStore(output_dir or DEFAULT_ALPHA101_DATA_STORE_ROOT)
        resolved_codes = _normalize_code_list(codes) if codes is not None else self.resolve_codes(
            as_of=as_of,
            start=resolved_start,
            end=resolved_end,
            as_ex=as_ex,
        )
        if not resolved_codes:
            raise ValueError("No codes resolved for Alpha101 input.")

        raw_price_frame = self.fetch_price_data(codes=resolved_codes, start=resolved_start, end=resolved_end, as_ex=as_ex)
        if raw_price_frame.empty:
            raise ValueError("No price rows returned for Alpha101 input.")

        classification_frame, companies_frame, industry_frame = self.fetch_classification_data()
        calendar_frame = self.merge_calendar(raw_price_frame, codes=resolved_codes, start=resolved_start, end=resolved_end)
        adjusted_frame = self.adjust_prices(calendar_frame, dividends=dividends, history=history)
        adjusted_frame = self.merge_classification(adjusted_frame, classification_frame)
        alpha_input_frame = self.build_alpha101_input(adjusted_frame)
        engine = Alpha101Engine(alpha_input_frame)
        factor_frame = engine.compute_all(alpha_ids=resolved_alpha_ids)
        saved_adjusted_price_path = (
            _save_adjusted_price_frame(
                data_store=data_store,
                adjusted_frame=adjusted_frame,
                metadata={
                    "provider": self.price_provider,
                    "classification_provider": self.classification_provider,
                    "classification_dataset": self.classification_company_name,
                    "family": "alpha101",
                    "execution_mode": "single",
                    "output_root": str(data_store.root_dir),
                },
            )
            if save_adj_price
            else None
        )
        saved_factor_paths: dict[str, str] | None = None
        saved_metadata_paths: dict[str, str] | None = None
        if save_factors:
            saved_factor_paths = {}
            saved_metadata_paths = {}
            for alpha_id in resolved_alpha_ids:
                factor_name = f"alpha_{alpha_id:03d}"
                factor_subset = factor_frame[["date_", "code", factor_name]].copy()
                metadata = _build_alpha101_metadata(
                    provider=self.price_provider,
                    classification_provider=self.classification_provider,
                    classification_dataset=self.classification_company_name,
                    alpha_id=alpha_id,
                    execution_mode="single",
                )
                factor_path, metadata_path = _save_alpha101_factor_frame(
                    data_store=data_store,
                    factor_name=factor_name,
                    factor_df=factor_subset,
                    metadata=metadata,
                )
                saved_factor_paths[factor_name] = factor_path
                saved_metadata_paths[factor_name] = metadata_path
        return Alpha101IndicatorResult(
            codes=resolved_codes,
            raw_price_frame=raw_price_frame,
            calendar_frame=calendar_frame,
            adjusted_frame=adjusted_frame,
            companies_frame=companies_frame,
            industry_frame=industry_frame,
            classification_frame=classification_frame,
            alpha_input_frame=alpha_input_frame,
            factor_frame=factor_frame,
            alpha_ids=resolved_alpha_ids,
            as_of=pd.Timestamp(as_of).date() if as_of is not None else None,
            saved_factor_paths=saved_factor_paths,
            saved_metadata_paths=saved_metadata_paths,
            saved_adjusted_price_path=saved_adjusted_price_path,
        )

    def compute_all_alpha101_parallel(
        self,
        alpha_ids: Sequence[int] | str | None = None,
        *,
        codes: Sequence[str] | None = None,
        as_of: date | str | pd.Timestamp | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        dividends: bool = False,
        history: bool = False,
        output_dir: str | Path | None = None,
        compute_workers: int | None = None,
        save_workers: int | None = None,
        save_factors: bool = False,
        save_adj_price: bool = False,
        as_ex: bool | None = None,
    ) -> Alpha101IndicatorResult:
        resolved_alpha_ids = _resolve_alpha_ids(alpha_ids)
        resolved_start = self.start if start is None else pd.Timestamp(start).date()
        resolved_end = self.end if end is None else pd.Timestamp(end).date()
        resolved_codes = _normalize_code_list(codes) if codes is not None else self.resolve_codes(
            as_of=as_of,
            start=resolved_start,
            end=resolved_end,
            as_ex=as_ex,
        )
        if not resolved_codes:
            raise ValueError("No codes resolved for Alpha101 input.")

        raw_price_frame = self.fetch_price_data(codes=resolved_codes, start=resolved_start, end=resolved_end, as_ex=as_ex)
        if raw_price_frame.empty:
            raise ValueError("No price rows returned for Alpha101 input.")

        classification_frame, companies_frame, industry_frame = self.fetch_classification_data()
        calendar_frame = self.merge_calendar(raw_price_frame, codes=resolved_codes, start=resolved_start, end=resolved_end)
        adjusted_frame = self.adjust_prices(calendar_frame, dividends=dividends, history=history)
        adjusted_frame = self.merge_classification(adjusted_frame, classification_frame)
        alpha_input_frame = self.build_alpha101_input(adjusted_frame)
        data_store = FactorStore(output_dir or DEFAULT_ALPHA101_DATA_STORE_ROOT)
        saved_adjusted_price_path = (
            _save_adjusted_price_frame(
                data_store=data_store,
                adjusted_frame=adjusted_frame,
                metadata={
                    "provider": self.price_provider,
                    "classification_provider": self.classification_provider,
                    "classification_dataset": self.classification_company_name,
                    "family": "alpha101",
                    "execution_mode": "parallel",
                    "output_root": str(data_store.root_dir),
                },
            )
            if save_adj_price
            else None
        )

        parallel_result = run_factor_tasks_parallel(
            resolved_alpha_ids,
            compute_fn=_parallel_alpha101_compute,
            compute_workers=compute_workers,
            save_workers=save_workers,
            save=False,
            output_dir=output_dir,
            factor_name_for_task=lambda alpha_id: f"alpha_{alpha_id:03d}",
            metadata_for_task=lambda alpha_id, execution_mode: _build_alpha101_metadata(
                provider=self.price_provider,
                classification_provider=self.classification_provider,
                classification_dataset=self.classification_company_name,
                alpha_id=alpha_id,
                execution_mode=execution_mode,
            ),
            initializer=_parallel_alpha101_init,
            initargs=(alpha_input_frame, None),
        )

        computed_frames = parallel_result.computed_frames
        saved_factor_paths: dict[str, str] | None = None
        saved_metadata_paths: dict[str, str] | None = None

        if not computed_frames:
            raise RuntimeError("No alpha101 factors were computed.")

        factor_frame = pd.DataFrame({"date_": [], "code": []})
        for alpha_id in resolved_alpha_ids:
            factor_name = f"alpha_{alpha_id:03d}"
            frame = computed_frames[alpha_id][["date_", "code", factor_name]].copy()
            factor_frame = frame if factor_frame.empty else factor_frame.merge(frame, on=["date_", "code"], how="left")
        factor_frame = factor_frame.sort_values(["code", "date_"], kind="stable").reset_index(drop=True)

        if save_factors:
            saved_factor_paths = {}
            saved_metadata_paths = {}
            for alpha_id in resolved_alpha_ids:
                factor_name = f"alpha_{alpha_id:03d}"
                factor_subset = computed_frames[alpha_id][["date_", "code", factor_name]].copy()
                metadata = _build_alpha101_metadata(
                    provider=self.price_provider,
                    classification_provider=self.classification_provider,
                    classification_dataset=self.classification_company_name,
                    alpha_id=alpha_id,
                    execution_mode=parallel_result.execution_mode,
                )
                factor_path, metadata_path = _save_alpha101_factor_frame(
                    data_store=data_store,
                    factor_name=factor_name,
                    factor_df=factor_subset,
                    metadata=metadata,
                )
                saved_factor_paths[factor_name] = factor_path
                saved_metadata_paths[factor_name] = metadata_path

        return Alpha101IndicatorResult(
            codes=resolved_codes,
            raw_price_frame=raw_price_frame,
            calendar_frame=calendar_frame,
            adjusted_frame=adjusted_frame,
            companies_frame=companies_frame,
            industry_frame=industry_frame,
            classification_frame=classification_frame,
            alpha_input_frame=alpha_input_frame,
            factor_frame=factor_frame,
            alpha_ids=resolved_alpha_ids,
            as_of=pd.Timestamp(as_of).date() if as_of is not None else None,
            saved_factor_paths=saved_factor_paths,
            saved_metadata_paths=saved_metadata_paths,
            saved_adjusted_price_path=saved_adjusted_price_path,
        )

    def run(
        self,
        alpha_id: int | str,
        *,
        codes: Sequence[str] | None = None,
        as_of: date | str | pd.Timestamp | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        dividends: bool = False,
        history: bool = False,
        save_factors: bool = False,
        save_adj_price: bool = False,
        output_dir: str | Path | None = None,
        compute_workers: int | None = None,
        save_workers: int | None = None,
        as_ex: bool | None = None,
    ) -> Alpha101IndicatorResult:
        return self.compute_alpha101(
            alpha_id,
            codes=codes,
            as_of=as_of,
            start=start,
            end=end,
            dividends=dividends,
            history=history,
            save_factors=save_factors,
            save_adj_price=save_adj_price,
            output_dir=output_dir,
            compute_workers=compute_workers,
            save_workers=save_workers,
            as_ex=as_ex,
        )


__all__ = [
    "Alpha101IndicatorResult",
    "Alpha101IndicatorTransformer",
    "DEFAULT_SP500_CONSTITUENTS_REQUEST",
    "DEFAULT_SP500_PRICE_REQUEST",
]
