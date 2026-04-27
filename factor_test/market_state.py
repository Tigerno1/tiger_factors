from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MarketStateResult:
    method: str
    state_column: str
    table: pd.DataFrame
    state_counts: dict[str, int]
    state_transition_matrix: pd.DataFrame
    current_state: str | None
    mean_return_by_state: dict[str, float] | None = None
    volatility_by_state: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "state_column": self.state_column,
            "state_counts": self.state_counts,
            "state_transition_matrix": self.state_transition_matrix.to_dict(),
            "current_state": self.current_state,
            "mean_return_by_state": self.mean_return_by_state,
            "volatility_by_state": self.volatility_by_state,
            "table": self.table.to_dict(orient="records"),
        }


def _coerce_price_panel(
    prices: pd.DataFrame,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> pd.DataFrame:
    frame = pd.DataFrame(prices).copy()
    if {date_column, code_column, price_column}.issubset(frame.columns):
        panel = frame.pivot_table(index=date_column, columns=code_column, values=price_column, aggfunc="last")
    else:
        panel = frame
    panel.index = pd.to_datetime(panel.index, errors="coerce")
    panel = panel.loc[~panel.index.isna()].sort_index()
    panel.columns = panel.columns.astype(str)
    return panel.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_market_state_features(
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
            "market_vol": market_return.rolling(int(max(vol_window, 2))).std(),
            "market_momentum": market_price.pct_change(int(max(momentum_window, 2))),
        }
    )
    return features.dropna(how="all")


def _prepare_feature_matrix(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(features).copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    if feature_columns is None:
        feature_columns = [str(column) for column in frame.columns]
    selected = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    clean = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("No valid market-state features remain after dropping NaNs.")
    return frame, clean


def _label_states(
    states: pd.Series,
    *,
    reference: pd.Series | None = None,
    label_map: dict[int, str] | None = None,
) -> pd.Series:
    states = states.astype(int)
    if label_map is not None:
        return states.map(label_map).rename("state")

    ordered_states = list(pd.Index(states.unique()).sort_values())
    if reference is not None:
        stats = reference.groupby(states).mean().sort_values()
        ordered_states = [int(state) for state in stats.index.tolist()]

    if len(ordered_states) == 2:
        names = ["bear", "bull"]
    elif len(ordered_states) == 3:
        names = ["bear", "sideways", "bull"]
    else:
        names = [f"state_{idx + 1}" for idx in range(len(ordered_states))]

    inferred_map = {state: name for state, name in zip(ordered_states, names)}
    return states.map(inferred_map).rename("state")


def build_kmeans_market_state_labels(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    n_states: int = 3,
    random_state: int = 42,
    label_map: dict[int, str] | None = None,
    order_by: str = "market_return",
) -> pd.Series:
    frame, clean = _prepare_feature_matrix(features, feature_columns=feature_columns)
    scaled = StandardScaler().fit_transform(clean.to_numpy(dtype=float))
    model = KMeans(n_clusters=int(n_states), random_state=random_state, n_init="auto")
    states = pd.Series(model.fit_predict(scaled), index=clean.index, name="state_id")
    reference = clean[order_by] if order_by in clean.columns else None
    labels = _label_states(states, reference=reference, label_map=label_map)
    return labels.reindex(frame.index)


def build_hmm_market_state_labels(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    n_states: int = 3,
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
            "hmmlearn is required for build_hmm_market_state_labels. Install it with `.venv/bin/python -m pip install hmmlearn`."
        ) from exc

    frame, clean = _prepare_feature_matrix(features, feature_columns=feature_columns)
    scaled = StandardScaler().fit_transform(clean.to_numpy(dtype=float))
    model = GaussianHMM(
        n_components=int(n_states),
        covariance_type=covariance_type,
        n_iter=int(n_iter),
        random_state=random_state,
    )
    model.fit(scaled)
    states = pd.Series(model.predict(scaled), index=clean.index, name="state_id")
    reference = clean[order_by] if order_by in clean.columns else None
    labels = _label_states(states, reference=reference, label_map=label_map)
    return labels.reindex(frame.index)


def expand_date_market_state_to_group_labels(
    reference: pd.DataFrame | pd.MultiIndex,
    state_labels: pd.Series | pd.DataFrame,
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

    if isinstance(state_labels, pd.Series):
        state_frame = state_labels.rename(group_column).reset_index()
        if state_frame.shape[1] != 2:
            raise ValueError("state_labels series must be indexed by date.")
        state_frame.columns = [date_column, group_column]
    else:
        state_frame = state_labels.copy()
        if group_column not in state_frame.columns:
            if state_frame.index.name is not None and state_frame.shape[1] == 1:
                state_frame = state_frame.rename(columns={state_frame.columns[0]: group_column}).reset_index()
            else:
                raise ValueError(f"state_labels dataframe must contain {group_column}")
        if date_column not in state_frame.columns:
            state_frame = state_frame.reset_index()
    state_frame[date_column] = pd.to_datetime(state_frame[date_column], errors="coerce")
    state_frame[group_column] = state_frame[group_column].astype(str)
    state_frame = state_frame[[date_column, group_column]].dropna().drop_duplicates(subset=[date_column])
    return pairs.merge(state_frame, on=date_column, how="left")[[date_column, code_column, group_column]]


def _state_transition_matrix(state_labels: pd.Series) -> pd.DataFrame:
    series = pd.Series(state_labels).dropna().astype(str)
    if series.empty or len(series) < 2:
        return pd.DataFrame()
    states = sorted(series.unique().tolist())
    transition = pd.DataFrame(0.0, index=states, columns=states)
    previous = series.iloc[:-1].to_numpy()
    current = series.iloc[1:].to_numpy()
    for left, right in zip(previous, current, strict=False):
        transition.loc[str(left), str(right)] += 1.0
    row_sums = transition.sum(axis=1).replace(0.0, np.nan)
    transition = transition.div(row_sums, axis=0).fillna(0.0)
    return transition


def market_state_test(
    prices: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    n_states: int = 3,
    use_hmm: bool = True,
    random_state: int = 42,
    label_map: dict[int, str] | None = None,
) -> MarketStateResult:
    features = build_market_state_features(prices)
    if use_hmm:
        labels = build_hmm_market_state_labels(
            features,
            feature_columns=feature_columns,
            n_states=n_states,
            random_state=random_state,
            label_map=label_map,
        )
        method = "hmm"
    else:
        labels = build_kmeans_market_state_labels(
            features,
            feature_columns=feature_columns,
            n_states=n_states,
            random_state=random_state,
            label_map=label_map,
        )
        method = "kmeans"

    state_counts = labels.dropna().astype(str).value_counts().sort_index().to_dict()
    transition_matrix = _state_transition_matrix(labels)
    current_state = None
    non_null = labels.dropna()
    if not non_null.empty:
        current_state = str(non_null.iloc[-1])

    market_returns = features.get("market_return")
    mean_return_by_state: dict[str, float] | None = None
    volatility_by_state: dict[str, float] | None = None
    if market_returns is not None:
        aligned = pd.concat(
            [market_returns.rename("market_return"), labels.rename("state")],
            axis=1,
        ).dropna()
        if not aligned.empty:
            mean_return_by_state = aligned.groupby("state")["market_return"].mean().to_dict()
            volatility_by_state = aligned.groupby("state")["market_return"].std(ddof=0).to_dict()

    table = pd.DataFrame(
        {
            "date_": labels.index,
            "state": labels.to_numpy(),
        }
    )
    return MarketStateResult(
        method=method,
        state_column="state",
        table=table,
        state_counts={str(key): int(value) for key, value in state_counts.items()},
        state_transition_matrix=transition_matrix,
        current_state=current_state,
        mean_return_by_state=None if mean_return_by_state is None else {str(k): float(v) for k, v in mean_return_by_state.items()},
        volatility_by_state=None if volatility_by_state is None else {str(k): float(v) for k, v in volatility_by_state.items()},
    )


__all__ = [
    "MarketStateResult",
    "build_market_state_features",
    "build_hmm_market_state_labels",
    "build_kmeans_market_state_labels",
    "expand_date_market_state_to_group_labels",
    "market_state_test",
]
