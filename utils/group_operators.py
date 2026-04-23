from __future__ import annotations

import numpy as np
import pandas as pd


def group_min(x: pd.Series, group: pd.Series) -> pd.Series:
    return x.groupby(group).transform("min")


def group_max(x: pd.Series, group: pd.Series) -> pd.Series:
    return x.groupby(group).transform("max")


def group_mean(x: pd.Series, weight: pd.Series | None = None, group: pd.Series | None = None) -> pd.Series:
    if weight is None:
        return x.groupby(group).transform("mean")

    def weighted_mean(g):
        return np.average(g[0], weights=g[1])

    return x.groupby(group).transform(lambda s: weighted_mean((s, weight.loc[s.index])))


def group_rank(x: pd.Series, group: pd.Series) -> pd.Series:
    return x.groupby(group).rank(pct=True)


def group_backfill(x: pd.Series, group: pd.Series, d: int, std: float = 4.0) -> pd.Series:
    def fill(s):
        s = s.copy()
        rolling_std = s.rolling(d, min_periods=1).std()
        mask = s.isna() | (np.abs(s - s.mean()) > std * rolling_std)
        return s.fillna(method="bfill") if mask.any() else s

    return x.groupby(group).transform(fill)


def group_scale(x: pd.Series, group: pd.Series) -> pd.Series:
    return x.groupby(group).transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def group_zscore(x: pd.Series, group: pd.Series) -> pd.Series:
    return x.groupby(group).transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def group_neutralize(x: pd.Series, group: pd.Series) -> pd.Series:
    def neutralize_group(s):
        return s - s.mean()

    return x.groupby(group).transform(neutralize_group)


def group_cartesian_product(g1: pd.Series, g2: pd.Series) -> pd.Series:
    return g1.astype(str) + "_" + g2.astype(str)


def combo_a(alpha: pd.Series, nlength: int = 250, mode: str = "algo1") -> pd.Series:
    """
    适用于 combo 平台的组合函数，常用于因子合成/处理。
    """
    if mode == "algo1":
        return alpha.rolling(nlength, min_periods=1).mean()
    raise ValueError("Only 'algo1' mode is supported")


__all__ = [
    "combo_a",
    "group_backfill",
    "group_cartesian_product",
    "group_max",
    "group_mean",
    "group_min",
    "group_neutralize",
    "group_rank",
    "group_scale",
    "group_zscore",
]
