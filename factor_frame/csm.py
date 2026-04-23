from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CSMConfig:
    """Configuration for a cross-sectional selection model.

    The model expects a long panel with at least:

    - ``date_``: rebalance date
    - ``code``: instrument code
    - feature columns listed in ``feature_columns``
    - an optional label column such as ``forward_return`` for fitting
    """

    feature_columns: tuple[str, ...]
    label_column: str = "forward_return"
    date_column: str = "date_"
    code_column: str = "code"
    feature_transform: str = "zscore"
    fit_method: str = "rank_ic"
    min_group_size: int = 5
    winsorize_limits: tuple[float, float] | None = (0.01, 0.99)
    normalize_score_by_date: bool = False
    score_clip: tuple[float, float] | None = None
    learning_rate: float = 0.05
    max_iter: int = 250
    l2_reg: float = 1e-3
    temperature: float = 1.0
    pairwise_max_pairs: int = 256


@dataclass(frozen=True)
class CSMFeatureStat:
    feature: str
    ic_mean: float
    ic_std: float
    rank_ic_mean: float
    rank_ic_std: float
    observations: int
    weight: float


@dataclass(frozen=True)
class CSMResult:
    score_frame: pd.DataFrame
    feature_stats: pd.DataFrame
    weights: dict[str, float]
    selection_frame: pd.DataFrame | None = None


def _ensure_frame(frame: pd.DataFrame, name: str = "frame") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return frame.copy()


def build_csm_training_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    label_column: str = "forward_return",
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.DataFrame:
    """Build a clean long frame suitable for CSM training.

    This is a thin convenience helper for factor_frame outputs or any other
    long research table that already follows the canonical ``date_`` / ``code``
    layout.
    """

    data = _ensure_frame(frame, "frame")
    required = {date_column, code_column, label_column, *feature_columns}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[code_column] = data[code_column].astype(str)
    for column in feature_columns:
        data[column] = _to_numeric_series(data[column])
    data[label_column] = _to_numeric_series(data[label_column])
    data = data.dropna(subset=[date_column, code_column]).copy()
    return data.loc[:, [date_column, code_column, *feature_columns, label_column]].sort_values(
        [date_column, code_column]
    ).reset_index(drop=True)


def infer_csm_feature_columns(
    frame: pd.DataFrame,
    *,
    label_column: str = "forward_return",
    date_column: str = "date_",
    code_column: str = "code",
    exclude_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Infer numeric feature columns from a Tiger-style long research frame.

    Columns used for identifiers or labels are excluded automatically. Remaining
    columns are kept when they contain at least one finite numeric value after
    coercion.
    """

    data = _ensure_frame(frame, "frame")
    excluded = {date_column, code_column, label_column}
    if exclude_columns is not None:
        excluded.update(exclude_columns)

    feature_columns: list[str] = []
    for column in data.columns:
        if column in excluded:
            continue
        numeric = _to_numeric_series(data[column])
        if numeric.notna().any():
            feature_columns.append(column)
    return tuple(feature_columns)


def _resolve_csm_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None,
    *,
    label_column: str,
    date_column: str,
    code_column: str,
) -> tuple[str, ...]:
    if feature_columns is not None:
        return tuple(feature_columns)
    return infer_csm_feature_columns(
        frame,
        label_column=label_column,
        date_column=date_column,
        code_column=code_column,
    )


def _to_numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _validate_config(config: CSMConfig) -> None:
    if not config.feature_columns:
        raise ValueError("feature_columns cannot be empty.")
    if config.feature_transform not in {"zscore", "rank", "minmax", "raw"}:
        raise ValueError("feature_transform must be one of: zscore, rank, minmax, raw.")
    if config.fit_method not in {"rank_ic", "ic", "regression", "ranknet", "listnet"}:
        raise ValueError("fit_method must be one of: rank_ic, ic, regression, ranknet, listnet.")
    if config.winsorize_limits is not None:
        lower, upper = config.winsorize_limits
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("winsorize_limits must satisfy 0 <= lower < upper <= 1.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if config.max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if config.l2_reg < 0:
        raise ValueError("l2_reg must be non-negative.")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if config.pairwise_max_pairs <= 0:
        raise ValueError("pairwise_max_pairs must be positive.")


def _winsorize_group(series: pd.Series, limits: tuple[float, float] | None) -> pd.Series:
    numeric = _to_numeric_series(series)
    if limits is None:
        return numeric
    lower, upper = limits
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return numeric.clip(lower=lo, upper=hi)


def _transform_group(series: pd.Series, method: str) -> pd.Series:
    numeric = _to_numeric_series(series)
    valid = numeric.dropna()
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    if valid.empty:
        return out
    if method == "raw":
        return numeric
    if method == "rank":
        out.loc[valid.index] = valid.rank(method="average", pct=True) - 0.5
        return out
    if method == "minmax":
        lo = float(valid.min())
        hi = float(valid.max())
        if np.isclose(lo, hi):
            out.loc[valid.index] = 0.5
            return out
        out.loc[valid.index] = (valid - lo) / (hi - lo) - 0.5
        return out
    if method == "zscore":
        std = float(valid.std(ddof=0))
        if std <= 1e-12:
            out.loc[valid.index] = valid - valid.mean()
        else:
            out.loc[valid.index] = (valid - valid.mean()) / std
        return out
    raise ValueError(f"Unsupported transform method: {method!r}")


def _groupwise_transform(
    frame: pd.DataFrame,
    *,
    date_column: str,
    feature_columns: tuple[str, ...],
    feature_transform: str,
    winsorize_limits: tuple[float, float] | None,
) -> pd.DataFrame:
    transformed = pd.DataFrame(index=frame.index)
    for feature in feature_columns:
        grouped = frame.groupby(date_column, sort=True)[feature]
        transformed[feature] = grouped.transform(lambda s: _transform_group(_winsorize_group(s, winsorize_limits), feature_transform))
    return transformed


def _groupwise_metric(
    frame: pd.DataFrame,
    *,
    date_column: str,
    feature: str,
    label_column: str,
    feature_transform: str,
    winsorize_limits: tuple[float, float] | None,
    min_group_size: int,
) -> tuple[float, float, float, float, int]:
    ic_values: list[float] = []
    rank_ic_values: list[float] = []
    observations = 0
    for _, group in frame.groupby(date_column, sort=True):
        subset = group[[feature, label_column]].dropna()
        if len(subset) < min_group_size:
            continue
        x = _transform_group(_winsorize_group(subset[feature], winsorize_limits), feature_transform)
        y = _transform_group(_winsorize_group(subset[label_column], winsorize_limits), "rank")
        x = x.dropna()
        y = y.reindex(x.index).dropna()
        common_index = x.index.intersection(y.index)
        if len(common_index) < min_group_size:
            continue
        x = x.loc[common_index]
        y = y.loc[common_index]
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            continue
        ic = x.corr(y, method="pearson")
        rank_ic = _transform_group(subset[feature], "rank").corr(_transform_group(subset[label_column], "rank"), method="pearson")
        if pd.notna(ic):
            ic_values.append(float(ic))
        if pd.notna(rank_ic):
            rank_ic_values.append(float(rank_ic))
        observations += len(common_index)
    ic_mean = float(np.mean(ic_values)) if ic_values else np.nan
    ic_std = float(np.std(ic_values, ddof=0)) if ic_values else np.nan
    rank_ic_mean = float(np.mean(rank_ic_values)) if rank_ic_values else np.nan
    rank_ic_std = float(np.std(rank_ic_values, ddof=0)) if rank_ic_values else np.nan
    return ic_mean, ic_std, rank_ic_mean, rank_ic_std, observations


def _prepare_rank_groups(
    frame: pd.DataFrame,
    *,
    date_column: str,
    feature_columns: tuple[str, ...],
    label_column: str,
    feature_transform: str,
    winsorize_limits: tuple[float, float] | None,
    min_group_size: int,
) -> list[tuple[pd.DataFrame, pd.Series]]:
    groups: list[tuple[pd.DataFrame, pd.Series]] = []
    for _, group in frame.groupby(date_column, sort=True):
        subset = group.loc[:, list(feature_columns) + [label_column]].dropna()
        if len(subset) < min_group_size:
            continue
        x = pd.DataFrame(index=subset.index)
        for feature in feature_columns:
            x[feature] = _transform_group(
                _winsorize_group(subset[feature], winsorize_limits),
                feature_transform,
            )
        y = _transform_group(_winsorize_group(subset[label_column], winsorize_limits), "rank")
        valid = x.notna().all(axis=1) & y.notna()
        x = x.loc[valid].astype(float)
        y = y.loc[valid].astype(float)
        if len(x) < min_group_size:
            continue
        if any(x[column].nunique(dropna=True) < 2 for column in x.columns):
            continue
        if y.nunique(dropna=True) < 2:
            continue
        groups.append((x, y))
    return groups


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    denom = exp.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / denom


def _fit_ranknet_weights(
    groups: list[tuple[pd.DataFrame, pd.Series]],
    *,
    feature_columns: tuple[str, ...],
    learning_rate: float,
    max_iter: int,
    l2_reg: float,
    pairwise_max_pairs: int,
) -> pd.Series:
    weights = np.zeros(len(feature_columns), dtype=float)
    if not groups:
        return pd.Series(weights, index=feature_columns, dtype=float)

    for _ in range(max_iter):
        grad = np.zeros_like(weights)
        pair_count = 0
        for x_frame, y in groups:
            x = x_frame.to_numpy(dtype=float)
            yv = y.to_numpy(dtype=float)
            if len(x) < 2:
                continue
            scores = x @ weights
            order = np.argsort(-yv, kind="mergesort")
            x = x[order]
            yv = yv[order]
            scores = scores[order]

            pair_limit = min(pairwise_max_pairs, len(x) * (len(x) - 1) // 2)
            if pair_limit <= 0:
                continue
            seen = 0
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    if yv[i] <= yv[j]:
                        continue
                    diff = x[i] - x[j]
                    margin = float(np.dot(weights, diff))
                    prob = 1.0 / (1.0 + np.exp(margin))
                    grad -= prob * diff
                    pair_count += 1
                    seen += 1
                    if seen >= pair_limit:
                        break
                if seen >= pair_limit:
                    break
        grad += l2_reg * weights
        if pair_count > 0:
            grad /= float(pair_count)
        weights -= learning_rate * grad
        if np.linalg.norm(grad) < 1e-6:
            break
        if not np.all(np.isfinite(weights)):
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(weights, index=feature_columns, dtype=float)


def _fit_listnet_weights(
    groups: list[tuple[pd.DataFrame, pd.Series]],
    *,
    feature_columns: tuple[str, ...],
    learning_rate: float,
    max_iter: int,
    l2_reg: float,
    temperature: float,
) -> pd.Series:
    weights = np.zeros(len(feature_columns), dtype=float)
    if not groups:
        return pd.Series(weights, index=feature_columns, dtype=float)

    for _ in range(max_iter):
        grad = np.zeros_like(weights)
        group_count = 0
        for x_frame, y in groups:
            x = x_frame.to_numpy(dtype=float)
            yv = y.to_numpy(dtype=float)
            if len(x) < 2:
                continue
            target = _softmax(yv, temperature=temperature)
            scores = x @ weights
            pred = _softmax(scores, temperature=temperature)
            grad += (x.T @ (pred - target)) / float(temperature)
            group_count += 1
        grad += l2_reg * weights
        if group_count > 0:
            grad /= float(group_count)
        weights -= learning_rate * grad
        if np.linalg.norm(grad) < 1e-6:
            break
        if not np.all(np.isfinite(weights)):
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(weights, index=feature_columns, dtype=float)


class CSMModel:
    """Cross-sectional ranking model for CSM-style stock selection.

    The model is intentionally light-weight:

    - fit on a long panel with ``date_`` / ``code`` / features / forward return
    - learn signed feature weights from IC / rank IC or regression
    - score each cross-section
    - rank and select top/bottom names per rebalance date

    It is designed to live in ``tiger_factors.factor_frame`` so it can feed
    both research and later strategy layers.
    """

    def __init__(self, config: CSMConfig) -> None:
        _validate_config(config)
        self.config = config
        self.feature_stats_: pd.DataFrame = pd.DataFrame()
        self.weights_: dict[str, float] = {}
        self.is_fitted_: bool = False

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.config.feature_columns

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
        *,
        require_label: bool,
        label_column: str | None = None,
    ) -> pd.DataFrame:
        data = _ensure_frame(frame, "frame")
        required = {self.config.date_column, self.config.code_column, *self.feature_columns}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(f"Missing required columns: {', '.join(missing)}")

        label_name = label_column or self.config.label_column
        if require_label and label_name not in data.columns:
            raise KeyError(f"Missing required label column: {label_name}")

        data[self.config.date_column] = pd.to_datetime(data[self.config.date_column], errors="coerce")
        data[self.config.code_column] = data[self.config.code_column].astype(str)
        for column in self.feature_columns:
            data[column] = _to_numeric_series(data[column])
        if label_name in data.columns:
            data[label_name] = _to_numeric_series(data[label_name])
        data = data.dropna(subset=[self.config.date_column, self.config.code_column]).copy()
        return data.sort_values([self.config.date_column, self.config.code_column]).reset_index(drop=True)

    def fit(self, frame: pd.DataFrame, *, label_column: str | None = None) -> "CSMModel":
        data = self._prepare_frame(frame, require_label=True, label_column=label_column)
        label_name = label_column or self.config.label_column

        rows: list[CSMFeatureStat] = []
        raw_weights: dict[str, float] = {}

        for feature in self.feature_columns:
            ic_mean, ic_std, rank_ic_mean, rank_ic_std, observations = _groupwise_metric(
                data,
                date_column=self.config.date_column,
                feature=feature,
                label_column=label_name,
                feature_transform=self.config.feature_transform,
                winsorize_limits=self.config.winsorize_limits,
                min_group_size=self.config.min_group_size,
            )
            raw_value = float(rank_ic_mean) if pd.notna(rank_ic_mean) else 0.0
            if self.config.fit_method == "ic":
                raw_value = float(ic_mean) if pd.notna(ic_mean) else raw_value

            rows.append(
                CSMFeatureStat(
                    feature=feature,
                    ic_mean=ic_mean,
                    ic_std=ic_std,
                    rank_ic_mean=rank_ic_mean,
                    rank_ic_std=rank_ic_std,
                    observations=observations,
                    weight=raw_value,
                )
            )
            raw_weights[feature] = raw_value

        stats = pd.DataFrame([stat.__dict__ for stat in rows])
        weights = pd.Series(raw_weights, dtype=float).replace([np.inf, -np.inf], np.nan)
        weights = weights.fillna(0.0)

        if self.config.fit_method in {"regression", "ranknet", "listnet"}:
            rank_groups = _prepare_rank_groups(
                data,
                date_column=self.config.date_column,
                feature_columns=self.feature_columns,
                label_column=label_name,
                feature_transform=self.config.feature_transform,
                winsorize_limits=self.config.winsorize_limits,
                min_group_size=self.config.min_group_size,
            )

        if self.config.fit_method == "regression":
            transformed = _groupwise_transform(
                data,
                date_column=self.config.date_column,
                feature_columns=self.feature_columns,
                feature_transform=self.config.feature_transform,
                winsorize_limits=self.config.winsorize_limits,
            )
            target = _groupwise_transform(
                data.rename(columns={label_name: "__csm_target__"}),
                date_column=self.config.date_column,
                feature_columns=("__csm_target__",),
                feature_transform=self.config.feature_transform,
                winsorize_limits=self.config.winsorize_limits,
            )["__csm_target__"]
            design = transformed.replace([np.inf, -np.inf], np.nan)
            fit_frame = pd.concat([design, target.rename(label_name)], axis=1).dropna()
            if len(fit_frame) >= len(self.feature_columns):
                x = fit_frame.loc[:, list(self.feature_columns)].to_numpy(dtype=float)
                y = fit_frame[label_name].to_numpy(dtype=float)
                beta, *_ = np.linalg.lstsq(x, y, rcond=None)
                weights = pd.Series(beta, index=self.feature_columns, dtype=float)
        elif self.config.fit_method == "ranknet":
            weights = _fit_ranknet_weights(
                rank_groups,
                feature_columns=self.feature_columns,
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                l2_reg=self.config.l2_reg,
                pairwise_max_pairs=self.config.pairwise_max_pairs,
            )
        elif self.config.fit_method == "listnet":
            weights = _fit_listnet_weights(
                rank_groups,
                feature_columns=self.feature_columns,
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                l2_reg=self.config.l2_reg,
                temperature=self.config.temperature,
            )

        scale = float(weights.abs().sum())
        if not np.isfinite(scale) or scale <= 0:
            weights = pd.Series(1.0 / len(self.feature_columns), index=self.feature_columns, dtype=float)
        else:
            weights = weights / scale

        stats["weight"] = stats["feature"].map(weights.to_dict()).astype(float)
        self.feature_stats_ = stats.reset_index(drop=True)
        self.weights_ = weights.to_dict()
        self.is_fitted_ = True
        return self

    def _score_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("CSMModel must be fitted before scoring.")
        transformed = _groupwise_transform(
            frame,
            date_column=self.config.date_column,
            feature_columns=self.feature_columns,
            feature_transform=self.config.feature_transform,
            winsorize_limits=self.config.winsorize_limits,
        )
        score = pd.Series(0.0, index=frame.index, dtype=float)
        for feature in self.feature_columns:
            score = score.add(transformed[feature].fillna(0.0) * float(self.weights_.get(feature, 0.0)), fill_value=0.0)
        return score

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = self._prepare_frame(frame, require_label=False)
        scored = data.copy()
        scored["csm_score"] = self._score_features(scored)
        if self.config.normalize_score_by_date:
            scored["csm_score"] = scored.groupby(self.config.date_column)["csm_score"].transform(
                lambda s: _transform_group(s, "zscore")
            )
        if self.config.score_clip is not None:
            scored["csm_score"] = scored["csm_score"].clip(self.config.score_clip[0], self.config.score_clip[1])
        scored["csm_rank"] = scored.groupby(self.config.date_column)["csm_score"].rank(method="average", ascending=False)
        return scored

    def score_panel(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a wide factor panel of model scores.

        The returned frame is indexed by ``date_`` and columns are security codes,
        which makes it directly consumable by ``multifactor_evaluation`` helpers
        such as ``run_factor_backtest``.
        """

        scored = self.predict(frame)
        panel = scored.pivot_table(
            index=self.config.date_column,
            columns=self.config.code_column,
            values="csm_score",
            aggfunc="last",
        )
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.sort_index()
        panel.columns = panel.columns.astype(str)
        return panel

    def rank(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.predict(frame)

    def select(
        self,
        frame: pd.DataFrame,
        *,
        top_n: int = 10,
        bottom_n: int = 0,
        long_only: bool = True,
    ) -> pd.DataFrame:
        scored = self.predict(frame)
        if top_n <= 0 and bottom_n <= 0:
            return scored.iloc[0:0].copy()

        groups = scored.groupby(self.config.date_column, sort=True)
        selections: list[pd.DataFrame] = []
        for _, group in groups:
            long_sel = group.nlargest(top_n, "csm_score") if top_n > 0 else group.iloc[0:0]
            if bottom_n > 0 and not long_only:
                short_sel = group.nsmallest(bottom_n, "csm_score")
                long_sel = long_sel.assign(csm_side=1.0, csm_target_weight=1.0 / max(top_n, 1))
                short_sel = short_sel.assign(csm_side=-1.0, csm_target_weight=-1.0 / max(bottom_n, 1))
                selections.append(pd.concat([long_sel, short_sel], axis=0, ignore_index=True))
            else:
                long_sel = long_sel.assign(csm_side=1.0, csm_target_weight=1.0 / max(top_n, 1))
                selections.append(long_sel)

        if not selections:
            return scored.iloc[0:0].copy()
        return pd.concat(selections, axis=0, ignore_index=True).sort_values(
            [self.config.date_column, "csm_score", self.config.code_column], ascending=[True, False, True]
        ).reset_index(drop=True)

    def fit_predict(self, frame: pd.DataFrame, *, label_column: str | None = None) -> CSMResult:
        self.fit(frame, label_column=label_column)
        scored = self.predict(frame)
        return CSMResult(score_frame=scored, feature_stats=self.feature_stats_.copy(), weights=dict(self.weights_))

    def fit_score_panel(self, frame: pd.DataFrame, *, label_column: str | None = None) -> pd.DataFrame:
        """Fit the model and return a wide score panel."""

        self.fit(frame, label_column=label_column)
        return self.score_panel(frame)

    def selection_panel(
        self,
        frame: pd.DataFrame,
        *,
        top_n: int = 10,
        bottom_n: int = 0,
        long_only: bool = True,
    ) -> pd.DataFrame:
        """Return a wide panel of target weights from the selected names."""

        selected = self.select(frame, top_n=top_n, bottom_n=bottom_n, long_only=long_only)
        if selected.empty:
            return pd.DataFrame()
        panel = selected.pivot_table(
            index=self.config.date_column,
            columns=self.config.code_column,
            values="csm_target_weight",
            aggfunc="sum",
        )
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.sort_index().fillna(0.0)
        panel.index.name = self.config.date_column
        panel.columns = panel.columns.astype(str)
        return panel

    def fit_selection_panel(
        self,
        frame: pd.DataFrame,
        *,
        label_column: str | None = None,
        top_n: int = 10,
        bottom_n: int = 0,
        long_only: bool = True,
    ) -> pd.DataFrame:
        """Fit the model and return a wide selection panel."""

        self.fit(frame, label_column=label_column)
        return self.selection_panel(frame, top_n=top_n, bottom_n=bottom_n, long_only=long_only)

    def to_strategy(
        self,
        *,
        top_n: int = 10,
        bottom_n: int = 0,
        long_only: bool = True,
    ) -> Callable[[Any], pd.DataFrame]:
        """Build a FactorFrame strategy callable from a fitted CSM model."""

        def _strategy(ctx: Any) -> pd.DataFrame:
            frame = ctx.combined_frame
            return self.select(frame, top_n=top_n, bottom_n=bottom_n, long_only=long_only)

        return _strategy


def build_csm_model(
    feature_columns: tuple[str, ...],
    *,
    label_column: str = "forward_return",
    date_column: str = "date_",
    code_column: str = "code",
    feature_transform: str = "zscore",
    fit_method: str = "rank_ic",
    min_group_size: int = 5,
    winsorize_limits: tuple[float, float] | None = (0.01, 0.99),
    normalize_score_by_date: bool = False,
    score_clip: tuple[float, float] | None = None,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    l2_reg: float = 1e-3,
    temperature: float = 1.0,
    pairwise_max_pairs: int = 256,
) -> CSMModel:
    return CSMModel(
        CSMConfig(
            feature_columns=tuple(feature_columns),
            label_column=label_column,
            date_column=date_column,
            code_column=code_column,
            feature_transform=feature_transform,
            fit_method=fit_method,
            min_group_size=min_group_size,
            winsorize_limits=winsorize_limits,
            normalize_score_by_date=normalize_score_by_date,
            score_clip=score_clip,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_reg=l2_reg,
            temperature=temperature,
            pairwise_max_pairs=pairwise_max_pairs,
        )
    )


def _build_csm_model_from_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None,
    *,
    label_column: str,
    date_column: str,
    code_column: str,
    feature_transform: str,
    fit_method: str,
    min_group_size: int,
    winsorize_limits: tuple[float, float] | None,
    normalize_score_by_date: bool,
    score_clip: tuple[float, float] | None,
    learning_rate: float,
    max_iter: int,
    l2_reg: float,
    temperature: float,
    pairwise_max_pairs: int,
) -> CSMModel:
    resolved_features = _resolve_csm_feature_columns(
        frame,
        feature_columns,
        label_column=label_column,
        date_column=date_column,
        code_column=code_column,
    )
    return build_csm_model(
        resolved_features,
        label_column=label_column,
        date_column=date_column,
        code_column=code_column,
        feature_transform=feature_transform,
        fit_method=fit_method,
        min_group_size=min_group_size,
        winsorize_limits=winsorize_limits,
        normalize_score_by_date=normalize_score_by_date,
        score_clip=score_clip,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_reg=l2_reg,
        temperature=temperature,
        pairwise_max_pairs=pairwise_max_pairs,
    )
