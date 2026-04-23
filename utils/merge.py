from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


def _coerce_frame(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, pd.Series):
        return rows.to_frame().reset_index()
    to_pandas = getattr(rows, "to_pandas", None)
    if callable(to_pandas):
        try:
            frame = to_pandas()
            if isinstance(frame, pd.DataFrame):
                return frame.copy()
        except Exception:
            pass
    return pd.DataFrame(rows)


def _normalize_aliases(frame: pd.DataFrame, *, join_keys: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    columns_by_lower = {str(column).lower(): column for column in result.columns}
    rename_map: dict[str, str] = {}

    for key in join_keys:
        if key in result.columns:
            continue
        candidates: tuple[str, ...]
        lower = str(key).lower()
        if lower in {"code", "symbol", "ticker", "code_"}:
            candidates = ("code", "symbol", "ticker", "code_", "Code", "Ticker")
        elif lower in {"date", "date_"}:
            candidates = ("date_", "date", "datetime", "Date", "Date_")
        elif lower == "industry_id":
            candidates = ("industry_id", "industryid", "IndustryId", "industryID")
        else:
            candidates = (key,)

        for candidate in candidates:
            actual = columns_by_lower.get(str(candidate).lower())
            if actual is not None:
                rename_map[actual] = key
                break

    if rename_map:
        result = result.rename(columns=rename_map)
    return result


def _normalize_join_columns(frame: pd.DataFrame, *, join_keys: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for key in join_keys:
        lower = str(key).lower()
        if lower in {"code", "symbol", "ticker", "code_"}:
            result[key] = result[key]
        elif lower in {"date", "date_"}:
            result[key] = pd.to_datetime(result[key], errors="coerce")
            if hasattr(result[key].dt, "tz_localize"):
                result[key] = result[key].dt.tz_localize(None)
    return result


def _prepare_frame(
    rows: Any,
    *,
    join_keys: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    frame = _coerce_frame(rows)
    if frame.empty:
        return frame
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame = _normalize_aliases(frame, join_keys=join_keys)

    missing = [key for key in join_keys if key not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing join columns: {missing}")

    frame = _normalize_join_columns(frame, join_keys=join_keys)
    value_columns = [column for column in frame.columns if column not in join_keys]
    if prefix:
        frame = frame.rename(columns={column: f"{prefix}__{column}" for column in value_columns})
    return frame


def _merge_frames(
    frames: Sequence[Any],
    *,
    join_keys: Sequence[str],
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not frames:
        raise ValueError("frames must not be empty.")

    merged: pd.DataFrame | None = None
    normalized_keys = tuple(str(key) for key in join_keys)

    for index, rows in enumerate(frames):
        name = names[index] if names is not None and index < len(names) else f"dataset_{index}"
        frame = _prepare_frame(rows, join_keys=normalized_keys, prefix=name)
        if frame.empty:
            continue
        frame = frame.dropna(subset=list(normalized_keys)).copy()
        frame = frame.sort_values(list(normalized_keys), kind="stable")
        frame = frame.drop_duplicates(subset=list(normalized_keys), keep="last")
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, how="left", on=list(normalized_keys), sort=False)

    if merged is None:
        merged = pd.DataFrame(columns=list(normalized_keys))

    return merged.reset_index(drop=True)


def merge_code_frames(
    frames: Sequence[Any],
    *,
    code_column: str = "code",
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    return _merge_frames(frames, join_keys=(code_column,), names=names)


def merge_date_frames(
    frames: Sequence[Any],
    *,
    date_column: str = "date_",
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    return _merge_frames(frames, join_keys=(date_column,), names=names)


def merge_code_date_frames(
    frames: Sequence[Any],
    *,
    date_column: str = "date_",
    code_column: str = "code",
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    return _merge_frames(frames, join_keys=(date_column, code_column), names=names)


def merge_other_frames(
    frames: Sequence[Any],
    *,
    join_keys: Sequence[str],
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    normalized_keys = tuple(str(key) for key in join_keys)
    if not normalized_keys:
        raise ValueError("join_keys must not be empty.")
    return _merge_frames(frames, join_keys=normalized_keys, names=names)


def merge_by_keys(
    frames: Sequence[Any],
    *,
    join_keys: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("frames must not be empty.")

    normalized_keys = tuple(str(key) for key in join_keys)
    if not normalized_keys:
        raise ValueError("join_keys must not be empty.")

    merged: pd.DataFrame | None = None
    for rows in frames:
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        if merged is None:
            merged = frame
        else:
            overlap = [
                column
                for column in frame.columns
                if column not in normalized_keys and column in merged.columns
            ]
            if overlap:
                frame = frame.drop(columns=overlap)
            merged = merged.merge(frame, how="left", on=list(normalized_keys), sort=False)

    if merged is None:
        merged = pd.DataFrame(columns=list(normalized_keys))

    return merged.reset_index(drop=True)


def merge_frames(
    frames: Sequence[Any],
    *,
    mode: str,
    join_keys: Sequence[str] | None = None,
    date_column: str = "date_",
    code_column: str = "code",
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    if mode == "code":
        return merge_code_frames(frames, code_column=code_column, names=names)
    if mode == "date":
        return merge_date_frames(frames, date_column=date_column, names=names)
    if mode == "code_date":
        return merge_code_date_frames(frames, date_column=date_column, code_column=code_column, names=names)
    if mode == "other":
        if not join_keys:
            raise ValueError("mode='other' requires join_keys.")
        return merge_other_frames(frames, join_keys=join_keys, names=names)
    raise ValueError("mode must be one of: code, date, code_date, other")


__all__ = [
    "merge_code_date_frames",
    "merge_by_keys",
    "merge_code_frames",
    "merge_date_frames",
    "merge_frames",
    "merge_other_frames",
]
