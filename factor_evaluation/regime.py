from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tiger_factors.factor_evaluation.utils import price_frame_to_wide


def _coerce_price_panel(
    prices: pd.DataFrame,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> pd.DataFrame:
    frame = prices.copy()
    if {date_column, code_column, price_column}.issubset(frame.columns):
        panel = price_frame_to_wide(
            frame,
            date_column=date_column,
            code_column=code_column,
            price_column=price_column,
        )
    else:
        panel = frame
    panel.index = pd.to_datetime(panel.index, errors="coerce")
    panel = panel[~panel.index.isna()].sort_index()
    panel.columns = panel.columns.astype(str)
    return panel.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prepare_feature_matrix(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = features.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    if feature_columns is None:
        feature_columns = [str(column) for column in frame.columns]
    selected = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    clean = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("No valid regime features remain after dropping NaNs.")
    return frame, clean


def _label_regimes(
    states: pd.Series,
    *,
    reference: pd.Series | None = None,
    label_map: dict[int, str] | None = None,
) -> pd.Series:
    states = states.astype(int)
    if label_map is not None:
        return states.map(label_map).rename("group")

    ordered_states = list(pd.Index(states.unique()).sort_values())
    if reference is not None:
        stats = reference.groupby(states).mean().sort_values()
        ordered_states = [int(state) for state in stats.index.tolist()]

    if len(ordered_states) == 2:
        names = ["Bear", "Bull"]
    elif len(ordered_states) == 3:
        names = ["Bear", "Sideways", "Bull"]
    else:
        names = [f"Regime {idx + 1}" for idx in range(len(ordered_states))]

    inferred_map = {state: name for state, name in zip(ordered_states, names)}
    return states.map(inferred_map).rename("group")


def build_market_regime_features(
    prices: pd.DataFrame,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
    vol_window: int = 20,
    momentum_window: int = 20,
) -> pd.DataFrame:
    panel = _coerce_price_panel(
        prices,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )
    returns = panel.pct_change(fill_method=None)
    market_return = returns.mean(axis=1)
    market_price = panel.mean(axis=1)
    features = pd.DataFrame(
        {
            "market_return": market_return,
            "market_vol": market_return.rolling(vol_window).std(),
            "market_momentum": market_price.pct_change(momentum_window),
        }
    )
    return features.dropna(how="all")


def build_kmeans_regime_labels(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    n_regimes: int = 3,
    random_state: int = 42,
    label_map: dict[int, str] | None = None,
    order_by: str = "market_return",
) -> pd.Series:
    frame, clean = _prepare_feature_matrix(features, feature_columns=feature_columns)
    scaled = StandardScaler().fit_transform(clean.to_numpy(dtype=float))
    model = KMeans(n_clusters=int(n_regimes), random_state=random_state, n_init="auto")
    states = pd.Series(model.fit_predict(scaled), index=clean.index, name="state")
    reference = clean[order_by] if order_by in clean.columns else None
    labels = _label_regimes(states, reference=reference, label_map=label_map)
    return labels.reindex(frame.index)


def build_hmm_regime_labels(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    n_regimes: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    label_map: dict[int, str] | None = None,
    order_by: str = "market_return",
) -> pd.Series:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "hmmlearn is required for build_hmm_regime_labels. Install it with `.venv/bin/python -m pip install hmmlearn`."
        ) from exc

    frame, clean = _prepare_feature_matrix(features, feature_columns=feature_columns)
    scaled = StandardScaler().fit_transform(clean.to_numpy(dtype=float))
    model = GaussianHMM(
        n_components=int(n_regimes),
        covariance_type=covariance_type,
        n_iter=int(n_iter),
        random_state=random_state,
    )
    model.fit(scaled)
    states = pd.Series(model.predict(scaled), index=clean.index, name="state")
    reference = clean[order_by] if order_by in clean.columns else None
    labels = _label_regimes(states, reference=reference, label_map=label_map)
    return labels.reindex(frame.index)


def expand_date_regime_to_group_labels(
    reference: pd.DataFrame | pd.MultiIndex,
    regime_labels: pd.Series | pd.DataFrame,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    group_column: str = "group",
) -> pd.DataFrame:
    if isinstance(reference, pd.MultiIndex):
        pairs = pd.DataFrame(reference.tolist(), columns=[date_column, code_column])
    else:
        if date_column not in reference.columns or code_column not in reference.columns:
            raise ValueError(f"reference must contain {date_column} and {code_column}")
        pairs = reference[[date_column, code_column]].drop_duplicates().copy()
    pairs[date_column] = pd.to_datetime(pairs[date_column], errors="coerce")
    pairs[code_column] = pairs[code_column].astype(str)
    pairs = pairs.dropna(subset=[date_column])

    if isinstance(regime_labels, pd.Series):
        regime_frame = regime_labels.rename(group_column).reset_index()
        if regime_frame.shape[1] != 2:
            raise ValueError("regime_labels series must be indexed by date.")
        regime_frame.columns = [date_column, group_column]
    else:
        regime_frame = regime_labels.copy()
        if group_column not in regime_frame.columns:
            if regime_frame.index.name is not None and regime_frame.shape[1] == 1:
                regime_frame = regime_frame.rename(columns={regime_frame.columns[0]: group_column}).reset_index()
            else:
                raise ValueError(f"regime_labels dataframe must contain {group_column}")
        if date_column not in regime_frame.columns:
            regime_frame = regime_frame.reset_index()
    regime_frame[date_column] = pd.to_datetime(regime_frame[date_column], errors="coerce")
    regime_frame[group_column] = regime_frame[group_column].astype(str)
    regime_frame = regime_frame[[date_column, group_column]].dropna().drop_duplicates(subset=[date_column])

    return pairs.merge(regime_frame, on=date_column, how="left")[[date_column, code_column, group_column]]


def build_kmeans_regime_group_labels(
    reference: pd.DataFrame | pd.MultiIndex,
    features: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    return expand_date_regime_to_group_labels(reference, build_kmeans_regime_labels(features, **kwargs))


def build_hmm_regime_group_labels(
    reference: pd.DataFrame | pd.MultiIndex,
    features: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    return expand_date_regime_to_group_labels(reference, build_hmm_regime_labels(features, **kwargs))


__all__ = [
    "build_hmm_regime_group_labels",
    "build_hmm_regime_labels",
    "build_kmeans_regime_group_labels",
    "build_kmeans_regime_labels",
    "build_market_regime_features",
    "expand_date_regime_to_group_labels",
]
