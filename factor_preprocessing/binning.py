from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from tiger_factors.factor_preprocessing._core import coerce_factor_panel
from tiger_factors.factor_preprocessing._core import coerce_target_panel


def _bin_unsupervised_series(
    values: pd.Series,
    *,
    method: str,
    n_bins: int,
    labels: list[Any] | None = None,
    duplicates: str = "drop",
) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    if valid.empty or valid.nunique(dropna=True) <= 1:
        return pd.Series(np.nan, index=series.index)

    normalized_method = str(method).strip().lower()
    bins = max(2, min(int(n_bins), int(valid.nunique(dropna=True))))

    if normalized_method == "quantile":
        ranked = valid.rank(method="first")
        codes = pd.qcut(ranked, q=bins, labels=False, duplicates=duplicates)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        result.loc[codes.index] = codes.astype(float)
        if labels is not None:
            mapping = {idx: label for idx, label in enumerate(labels[: int(codes.max()) + 1])}
            return result.map(mapping)
        return result

    if normalized_method == "uniform":
        codes = pd.cut(valid, bins=bins, labels=False, duplicates=duplicates, include_lowest=True)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        result.loc[codes.index] = codes.astype(float)
        if labels is not None:
            mapping = {idx: label for idx, label in enumerate(labels[: int(codes.max()) + 1])}
            return result.map(mapping)
        return result

    if normalized_method == "kmeans":
        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
        transformed = discretizer.fit_transform(valid.to_numpy(dtype=float).reshape(-1, 1)).ravel()
        result = pd.Series(np.nan, index=series.index, dtype=float)
        result.loc[valid.index] = transformed
        return result if labels is None else result.map({idx: label for idx, label in enumerate(labels)})

    raise ValueError("Unsupported unsupervised binning method.")


def _tree_thresholds(feature: pd.Series, target: pd.Series, *, max_bins: int, random_state: int | None = 42) -> np.ndarray:
    x = pd.to_numeric(feature, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    frame = pd.concat([x.rename("_x"), y.rename("_y")], axis=1).dropna()
    if frame.empty or frame["_x"].nunique() <= 1:
        return np.array([], dtype=float)

    X = frame["_x"].to_numpy(dtype=float).reshape(-1, 1)
    yv = frame["_y"].to_numpy(dtype=float)
    unique_y = pd.unique(frame["_y"].dropna())
    use_classifier = len(unique_y) <= 2 and set(pd.Series(unique_y).dropna().tolist()).issubset({0.0, 1.0})

    if use_classifier:
        tree = DecisionTreeClassifier(
            max_leaf_nodes=max(2, int(max_bins)),
            min_samples_leaf=max(1, int(np.ceil(len(frame) * 0.05))),
            random_state=random_state,
        )
    else:
        tree = DecisionTreeRegressor(
            max_leaf_nodes=max(2, int(max_bins)),
            min_samples_leaf=max(1, int(np.ceil(len(frame) * 0.05))),
            random_state=random_state,
        )

    tree.fit(X, yv)
    thresholds = tree.tree_.threshold
    thresholds = thresholds[np.isfinite(thresholds) & (thresholds != -2)]
    return np.unique(np.sort(thresholds))


def _chi_square_2x2(left: tuple[float, float], right: tuple[float, float]) -> float:
    table = np.array([left, right], dtype=float)
    total = table.sum()
    if total <= 0:
        return 0.0
    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = ((table - expected) ** 2) / expected
    return float(np.nansum(chi))


def _chi_merge_thresholds(
    feature: pd.Series,
    target: pd.Series,
    *,
    max_bins: int,
    positive_label: Any = 1,
    significance_level: float = 3.841,
) -> np.ndarray:
    x = pd.to_numeric(feature, errors="coerce")
    y = pd.Series(target).reindex(x.index)
    frame = pd.concat([x.rename("_x"), y.rename("_y")], axis=1).dropna().sort_values("_x")
    if frame.empty or frame["_x"].nunique() <= 1:
        return np.array([], dtype=float)

    bad = (frame["_y"] == positive_label).astype(int)
    grouped = (
        pd.DataFrame({"x": frame["_x"], "bad": bad})
        .groupby("x", sort=True)
        .agg(total=("bad", "size"), bad=("bad", "sum"))
    )
    bins: list[dict[str, float]] = []
    for value, row in grouped.iterrows():
        bins.append(
            {
                "start": float(value),
                "end": float(value),
                "good": float(row["total"] - row["bad"]),
                "bad": float(row["bad"]),
                "total": float(row["total"]),
            }
        )

    if len(bins) <= 1:
        return np.array([], dtype=float)

    def merge(i: int) -> None:
        left = bins[i]
        right = bins[i + 1]
        bins[i] = {
            "start": left["start"],
            "end": right["end"],
            "good": left["good"] + right["good"],
            "bad": left["bad"] + right["bad"],
            "total": left["total"] + right["total"],
        }
        del bins[i + 1]

    while len(bins) > 1:
        chi_values = [
            _chi_square_2x2((bins[i]["good"], bins[i]["bad"]), (bins[i + 1]["good"], bins[i + 1]["bad"]))
            for i in range(len(bins) - 1)
        ]
        min_idx = int(np.argmin(chi_values))
        if len(bins) > max_bins or chi_values[min_idx] < significance_level:
            merge(min_idx)
        else:
            break

    thresholds = []
    for left, right in zip(bins[:-1], bins[1:]):
        thresholds.append((left["end"] + right["start"]) / 2.0)
    return np.array(sorted(set(thresholds)), dtype=float)


def bin_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "quantile",
    n_bins: int = 5,
    labels: list[Any] | None = None,
    axis: int = 1,
    target: pd.Series | pd.DataFrame | None = None,
    positive_label: Any = 1,
    significance_level: float = 3.841,
    duplicates: str = "drop",
    random_state: int | None = 42,
) -> pd.Series | pd.DataFrame:
    """Discretize factor values into bins."""

    normalized_method = str(method).strip().lower()

    if isinstance(values, pd.Series):
        if normalized_method in {"quantile", "uniform", "kmeans"}:
            return _bin_unsupervised_series(values, method=normalized_method, n_bins=n_bins, labels=labels, duplicates=duplicates)
        if normalized_method == "decision_tree":
            if target is None or not isinstance(target, pd.Series):
                raise ValueError("target series is required for decision_tree binning.")
            thresholds = _tree_thresholds(values, target, max_bins=n_bins, random_state=random_state)
            bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
            codes = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=False, include_lowest=True)
            return codes if labels is None else codes.map({idx: label for idx, label in enumerate(labels)})
        if normalized_method == "chi_merge":
            if target is None or not isinstance(target, pd.Series):
                raise ValueError("target series is required for chi_merge binning.")
            thresholds = _chi_merge_thresholds(
                values,
                target,
                max_bins=n_bins,
                positive_label=positive_label,
                significance_level=significance_level,
            )
            bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
            codes = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=False, include_lowest=True)
            return codes if labels is None else codes.map({idx: label for idx, label in enumerate(labels)})
        raise ValueError("Unsupported binning method.")

    frame = coerce_factor_panel(values)
    target_frame = coerce_target_panel(target)

    if normalized_method in {"quantile", "uniform", "kmeans"}:
        return frame.apply(
            lambda row: _bin_unsupervised_series(row, method=normalized_method, n_bins=n_bins, labels=labels, duplicates=duplicates),
            axis=1,
        )

    if normalized_method in {"decision_tree", "chi_merge"}:
        if target_frame is None:
            raise ValueError("target is required for supervised binning.")

        def _bin_row(row: pd.Series) -> pd.Series:
            row_target = target_frame.loc[row.name].reindex(row.index)
            if normalized_method == "decision_tree":
                thresholds = _tree_thresholds(row, row_target, max_bins=n_bins, random_state=random_state)
            else:
                thresholds = _chi_merge_thresholds(
                    row,
                    row_target,
                    max_bins=n_bins,
                    positive_label=positive_label,
                    significance_level=significance_level,
                )
            bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
            codes = pd.cut(pd.to_numeric(row, errors="coerce"), bins=bins, labels=False, include_lowest=True)
            if labels is None:
                return codes
            mapping = {idx: label for idx, label in enumerate(labels)}
            return codes.map(mapping)

        return frame.apply(_bin_row, axis=1)

    raise ValueError("Unsupported binning method.")


def woe_encode_binned(
    binned: pd.Series,
    target: pd.Series,
    *,
    positive_label: Any = 1,
    smoothing: float = 0.5,
) -> pd.Series:
    """WOE encode a binned feature."""

    frame = pd.concat([pd.Series(binned, name="_bin"), pd.Series(target, name="_target")], axis=1).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=binned.index)

    bad = (frame["_target"] == positive_label).astype(int)
    good = 1 - bad
    total_good = float(good.sum())
    total_bad = float(bad.sum())
    stats = pd.DataFrame({"good": good, "bad": bad, "bin": frame["_bin"]}).groupby("bin").sum()
    stats["woe"] = np.log(((stats["good"] + smoothing) / (total_good + smoothing)) / ((stats["bad"] + smoothing) / (total_bad + smoothing)))
    return pd.Series(frame["_bin"].map(stats["woe"]), index=frame.index).reindex(binned.index)


def target_encode_binned(
    binned: pd.Series,
    target: pd.Series,
    *,
    smoothing: float = 1.0,
) -> pd.Series:
    """Target encode a binned feature using smoothed mean target."""

    frame = pd.concat([pd.Series(binned, name="_bin"), pd.Series(target, name="_target")], axis=1).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=binned.index)

    global_mean = float(pd.to_numeric(frame["_target"], errors="coerce").mean())
    stats = frame.groupby("_bin")["_target"].agg(["mean", "count"])
    stats["encoded"] = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return pd.Series(frame["_bin"].map(stats["encoded"]), index=frame.index).reindex(binned.index)


def onehot_encode_binned(binned: pd.Series, *, prefix: str = "bin") -> pd.DataFrame:
    """One-hot encode a binned feature."""

    return pd.get_dummies(binned, prefix=prefix, dummy_na=False)


__all__ = [
    "bin_factor_panel",
    "onehot_encode_binned",
    "target_encode_binned",
    "woe_encode_binned",
]
