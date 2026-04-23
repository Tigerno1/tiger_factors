from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def factor_correlation_matrix(
    factors: dict[str, pd.DataFrame],
    *,
    standardize: bool = True,
) -> pd.DataFrame:
    names = list(factors.keys())
    corr = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)

    transformed = {
        name: (cross_sectional_zscore(panel) if standardize else panel)
        for name, panel in factors.items()
    }

    for i, left_name in enumerate(names):
        left = transformed[left_name].stack()
        for j in range(i + 1, len(names)):
            right_name = names[j]
            right = transformed[right_name].stack()
            joined = pd.concat([left.rename("x"), right.rename("y")], axis=1).dropna()
            value = float(joined["x"].corr(joined["y"])) if not joined.empty else np.nan
            corr.loc[left_name, right_name] = value
            corr.loc[right_name, left_name] = value
    return corr


def greedy_select_by_correlation(
    scores: dict[str, float],
    corr: pd.DataFrame,
    threshold: float,
) -> list[str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[str] = []

    for name, _ in ordered:
        if not selected:
            selected.append(name)
            continue
        too_correlated = any(abs(float(corr.loc[name, picked])) >= threshold for picked in selected)
        if not too_correlated:
            selected.append(name)

    return selected


def blend_factor_panels(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
    *,
    standardize: bool = True,
) -> pd.DataFrame:
    if not weights:
        raise ValueError("weights must not be empty.")

    selected = [name for name in weights.keys() if name in factors]
    if not selected:
        raise ValueError("No selected factor exists in factor panels.")

    panels: list[pd.DataFrame] = []
    for name in selected:
        panel = factors[name]
        panel = cross_sectional_zscore(panel) if standardize else panel
        panels.append(panel * float(weights[name]))

    return pd.concat(panels).groupby(level=0).sum()
