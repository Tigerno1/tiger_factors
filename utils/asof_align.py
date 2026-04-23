from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_reference.calendar import apply_session_lag, normalize_sessions, session_index_on_or_after


def align_fundamental_point_in_time(
    fundamentals: pd.DataFrame,
    trading_dates: pd.Index,
    *,
    value_columns: list[str],
    code_column: str = "code",
    date_column: str = "date_",
    availability_column: str | None = None,
    lag_sessions: int = 1,
) -> dict[str, pd.DataFrame]:
    sessions = normalize_sessions(trading_dates)
    if fundamentals.empty:
        return {column: pd.DataFrame(index=sessions) for column in value_columns}

    source = fundamentals.copy()
    event_col = availability_column if availability_column and availability_column in source.columns else date_column
    if event_col not in source.columns or code_column not in source.columns:
        return {column: pd.DataFrame(index=sessions) for column in value_columns}

    source[event_col] = pd.to_datetime(source[event_col], errors="coerce")
    source = source.dropna(subset=[event_col, code_column])
    if source.empty:
        return {column: pd.DataFrame(index=sessions) for column in value_columns}

    positions = session_index_on_or_after(source[event_col], sessions)
    positions = apply_session_lag(positions, lag_sessions=lag_sessions, session_count=len(sessions))
    valid = positions >= 0
    source = source.loc[valid].copy()
    if source.empty:
        return {column: pd.DataFrame(index=sessions) for column in value_columns}

    source["effective_session"] = sessions.take(positions[valid])
    source[code_column] = source[code_column].astype(str)

    aligned: dict[str, pd.DataFrame] = {}
    for column in value_columns:
        if column not in source.columns:
            aligned[column] = pd.DataFrame(index=sessions)
            continue

        subset = source[["effective_session", code_column, column]].copy()
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        wide = subset.pivot_table(
            index="effective_session",
            columns=code_column,
            values=column,
            aggfunc="last",
        ).sort_index()
        wide = wide.reindex(sessions).ffill()
        aligned[column] = wide

    return aligned
