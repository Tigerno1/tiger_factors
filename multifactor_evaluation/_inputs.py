from __future__ import annotations

import pandas as pd


def coerce_factor_series(factor: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(factor, pd.Series):
        if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
            raise ValueError("factor series must be indexed by (date, code).")
        return factor.sort_index()

    if {"date_", "code"}.issubset(factor.columns):
        value_columns = [col for col in factor.columns if col not in {"date_", "code"}]
        if len(value_columns) != 1:
            raise ValueError("long factor frame must contain exactly one value column besides date_ and code.")
        return (
            factor.copy()
            .assign(date_=pd.to_datetime(factor["date_"], errors="coerce"))
            .dropna(subset=["date_"])
            .set_index(["date_", "code"])[value_columns[0]]
            .sort_index()
        )

    if factor.index.nlevels == 1:
        return factor.sort_index().stack(future_stack=True).sort_index()

    raise ValueError("Unsupported factor input format.")


def coerce_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame.")

    if {"date_", "code"}.issubset(prices.columns):
        candidate_columns = [col for col in ["close", "adj_close", "price", "value"] if col in prices.columns]
        if not candidate_columns:
            raise ValueError("long price frame must include one of close/adj_close/price/value.")
        value_column = candidate_columns[0]
        frame = (
            prices.copy()
            .assign(date_=pd.to_datetime(prices["date_"], errors="coerce"))
            .dropna(subset=["date_"])
            .pivot(index="date_", columns="code", values=value_column)
            .sort_index()
        )
        frame.index.name = "date_"
        return frame

    frame = prices.sort_index().copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()]
    frame.index.name = frame.index.name or "date_"
    return frame


def coerce_labels_frame(labels: pd.DataFrame) -> pd.DataFrame:
    if {"date_", "code"}.issubset(labels.columns):
        frame = labels.copy()
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_"]).set_index(["date_", "code"]).sort_index()
        return frame

    if isinstance(labels.index, pd.MultiIndex) and labels.index.nlevels == 2:
        frame = labels.copy().sort_index()
        if frame.index.names != ["date_", "code"]:
            frame.index = frame.index.set_names(["date_", "code"])
        return frame

    raise ValueError("labels must use a (date_, code) MultiIndex or date_/code columns.")


__all__ = [
    "coerce_factor_series",
    "coerce_labels_frame",
    "coerce_price_panel",
]
