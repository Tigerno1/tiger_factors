from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tiger_api.const import normalize_frequency
from tiger_api.const import normalize_identifier
from tiger_api.const import normalize_provider
from tiger_api.const import normalize_region
from tiger_api.const import normalize_sec_type
from tiger_api.const import normalize_variant


DEFAULT_ADJ_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


# =========================
# Data Schema Layer
# =========================


@dataclass(frozen=True, slots=True)
class BaseDataSchema:
    date_column: str = "date_"

    def _empty_frame(self, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=list(columns))

    def _ensure_columns(
        self,
        frame: pd.DataFrame,
        required: tuple[str, ...],
        *,
        label: str,
    ) -> None:
        missing = sorted(column for column in required if column not in frame.columns)
        if missing:
            raise ValueError(f"{label} frame is missing required columns: {missing}")

    def _normalize_datetime(self, frame: pd.DataFrame, column: str) -> None:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.tz_localize(None)

    def _normalize_numeric(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    def _raise_if_all_rows_invalid(self, original: pd.DataFrame, normalized: pd.DataFrame, *, label: str) -> None:
        if not original.empty and normalized.empty:
            raise ValueError(f"{label} frame contains no valid rows after normalization")


@dataclass(frozen=True, slots=True)
class FactorSchema(BaseDataSchema):
    code_column: str = "code"
    value_column: str = "value"

    def columns(self) -> tuple[str, str, str]:
        return self.code_column, self.date_column, self.value_column

    def normalize(
        self,
        frame: pd.DataFrame,
        *,
        factor_name: str | None = None,
    ) -> pd.DataFrame:
        if frame.empty:
            return self._empty_frame(self.columns())

        normalized = frame.copy()
        self._ensure_columns(
            normalized,
            (self.code_column, self.date_column),
            label="factor",
        )

        value_column = self.value_column
        if value_column not in normalized.columns:
            if factor_name and factor_name in normalized.columns:
                value_column = factor_name
            else:
                candidates = [
                    column
                    for column in normalized.columns
                    if column not in {self.code_column, self.date_column}
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "factor frame must contain exactly one value column besides "
                        f"{self.code_column!r}/{self.date_column!r}; got {candidates!r}"
                    )
                value_column = candidates[0]

        normalized = normalized.loc[:, [self.code_column, self.date_column, value_column]].rename(
            columns={value_column: self.value_column}
        )
        normalized[self.code_column] = normalized[self.code_column].astype(str)
        self._normalize_datetime(normalized, self.date_column)
        self._normalize_numeric(normalized, (self.value_column,))
        normalized = normalized.dropna(subset=[self.code_column, self.date_column, self.value_column])

        self._raise_if_all_rows_invalid(frame, normalized, label="factor")

        normalized = normalized.drop_duplicates(
            subset=[self.code_column, self.date_column],
            keep="last",
        )
        return normalized.sort_values(
            [self.code_column, self.date_column],
            kind="stable",
        ).reset_index(drop=True)


FactorData = FactorSchema
FactorFrameSchema = FactorSchema


@dataclass(frozen=True, slots=True)
class AdjPriceSchema(BaseDataSchema):
    code_column: str = "code"
    core_columns: tuple[str, ...] = DEFAULT_ADJ_PRICE_COLUMNS

    def columns(self) -> tuple[str, ...]:
        return (self.code_column, self.date_column, *self.core_columns)

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self._empty_frame(self.columns())

        normalized = frame.copy()
        self._ensure_columns(
            normalized,
            (self.code_column, self.date_column, *self.core_columns),
            label="adj price",
        )

        normalized[self.code_column] = normalized[self.code_column].astype(str)
        self._normalize_datetime(normalized, self.date_column)
        self._normalize_numeric(normalized, self.core_columns)

        ordered_columns = [
            self.code_column,
            self.date_column,
            *[column for column in self.core_columns if column in normalized.columns],
            *[
                column
                for column in normalized.columns
                if column not in {self.code_column, self.date_column, *self.core_columns}
            ],
        ]
        normalized = normalized.loc[:, ordered_columns]
        normalized = normalized.dropna(subset=[self.code_column, self.date_column])

        self._raise_if_all_rows_invalid(frame, normalized, label="adj price")

        normalized = normalized.drop_duplicates(
            subset=[self.code_column, self.date_column],
            keep="last",
        )
        return normalized.sort_values(
            [self.code_column, self.date_column],
            kind="stable",
        ).reset_index(drop=True)


AdjPriceData = AdjPriceSchema


@dataclass(frozen=True, slots=True)
class MacroSchema(BaseDataSchema):
    value_column: str = "value"

    def columns(self) -> tuple[str, str]:
        return self.date_column, self.value_column

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self._empty_frame(self.columns())

        normalized = frame.copy()
        self._ensure_columns(
            normalized,
            (self.date_column, self.value_column),
            label="macro",
        )
        self._normalize_datetime(normalized, self.date_column)
        self._normalize_numeric(normalized, (self.value_column,))
        normalized = normalized.loc[:, [self.date_column, self.value_column]]
        normalized = normalized.dropna(subset=[self.date_column, self.value_column])

        self._raise_if_all_rows_invalid(frame, normalized, label="macro")

        return normalized.sort_values(
            [self.date_column],
            kind="stable",
        ).reset_index(drop=True)


MacroData = MacroSchema


# =========================
# Dataset Spec Layer
# =========================


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    region: str
    freq: str
    table_name: str
    variant: str | None = None
    provider: str = "tiger"
    date_column: str = "date_"
    code_column: str | None = "code"

    def kind(self) -> str:
        raise NotImplementedError

    def storage_columns(self) -> tuple[str, ...]:
        raise NotImplementedError

    def path_segments(self) -> tuple[str, ...]:
        raise NotImplementedError

    def data_stem(self) -> str:
        return self.table_name if self.variant is None else f"{self.table_name}__{self.variant}"

    def variant_suffix(self) -> str:
        return "" if self.variant is None else f"__{self.variant}"

    def key_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        if self.code_column:
            columns.append(self.code_column)
        columns.append(self.date_column)
        return tuple(columns)

    def dataset_dir(self, root: str | Path) -> Path:
        return Path(root).joinpath(*self.path_segments(), self.data_stem())

    def data_path(self, root: str | Path, *, part: str | None = None) -> Path:
        filename = self.data_stem()
        if part is not None:
            filename = f"{filename}__{part}"
        return self.dataset_dir(root) / f"{filename}.parquet"

    def meta_path(self, root: str | Path) -> Path:
        return self.dataset_dir(root) / "meta.json"
    
    def manifest_path(self, root: str | Path) -> Path:
        return self.dataset_dir(root) / "manifest.json"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind()
        return payload

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", normalize_provider(self.provider, known_only=True))
        object.__setattr__(self, "region", normalize_region(self.region, known_only=True))
        object.__setattr__(self, "freq", normalize_frequency(self.freq, known_only=True))
        object.__setattr__(self, "table_name", normalize_identifier(self.table_name, "table_name"))

        if self.code_column is not None:
            object.__setattr__(self, "code_column", normalize_identifier(self.code_column, "code_column"))
        object.__setattr__(self, "date_column", normalize_identifier(self.date_column, "date_column"))

        if self.variant is not None:
            object.__setattr__(self, "variant", normalize_variant(self.variant))


@dataclass(frozen=True, slots=True)
class SecTypeDatasetSpec(DatasetSpec):
    sec_type: str = "stock"

    def __post_init__(self) -> None:
        DatasetSpec.__post_init__(self)
        object.__setattr__(self, "sec_type", normalize_sec_type(self.sec_type, known_only=True))

    def path_segments(self) -> tuple[str, ...]:
        return (
            self.kind(),
            self.provider,
            self.region,
            self.sec_type,
            self.freq,
        )


@dataclass(frozen=True, slots=True)
class FactorSpec(SecTypeDatasetSpec):
    value_column: str = "value"
    group: str | None = None 

    def kind(self) -> str:
        return "factor"

    def storage_columns(self) -> tuple[str, ...]:
        return (self.code_column or "code", self.date_column, self.value_column)
    
    def path_segments(self) -> tuple[str, ...]:
        parts = [
            self.kind(),
            self.provider,
            self.region,
            self.sec_type,
            self.freq,
        ]
        if self.group is not None:
            parts.append(self.group)
        return tuple(parts)
    
    def __post_init__(self) -> None:
        SecTypeDatasetSpec.__post_init__(self)
        object.__setattr__(self, "value_column", normalize_identifier(self.value_column, "value_column"))
        if self.group is not None:
            object.__setattr__(self, "group", normalize_identifier(self.group, "group"))


@dataclass(frozen=True, slots=True)
class AdjPriceSpec(SecTypeDatasetSpec):
    table_name: str = "adj_price"
    columns: tuple[str, ...] = DEFAULT_ADJ_PRICE_COLUMNS

    def kind(self) -> str:
        return "price"

    def storage_columns(self) -> tuple[str, ...]:
        return (self.code_column or "code", self.date_column, *self.columns)

    def __post_init__(self) -> None:
        SecTypeDatasetSpec.__post_init__(self)
        object.__setattr__(
            self,
            "columns",
            tuple(normalize_identifier(column, "price_column") for column in self.columns),
        )


@dataclass(frozen=True, slots=True)
class MacroSpec(SecTypeDatasetSpec):
    sec_type: str = "micro"
    code_column: str | None = None
    value_column: str = "value"

    def kind(self) -> str:
        return "macro"

    def storage_columns(self) -> tuple[str, ...]:
        return (self.date_column, self.value_column)

    def __post_init__(self) -> None:
        SecTypeDatasetSpec.__post_init__(self)
        object.__setattr__(self, "value_column", normalize_identifier(self.value_column, "value_column"))


@dataclass(frozen=True, slots=True)
class OthersSpec(SecTypeDatasetSpec):
    def kind(self) -> str:
        return "others"

    def storage_columns(self) -> tuple[str, ...]:
        return self.key_columns()


__all__ = [
    "DEFAULT_ADJ_PRICE_COLUMNS",
    # schema
    "BaseDataSchema",
    "FactorSchema",
    "FactorData",
    "FactorFrameSchema",
    "AdjPriceSchema",
    "AdjPriceData",
    "MacroSchema",
    "MacroData",
    # spec
    "DatasetSpec",
    "SecTypeDatasetSpec",
    "FactorSpec",
    "AdjPriceSpec",
    "MacroSpec",
    "OthersSpec",
]