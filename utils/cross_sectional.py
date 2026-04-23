from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_quantile_bounds(lower: float, upper: float) -> None:
    if not 0 <= lower < upper <= 1:
        raise ValueError("Expected 0 <= lower < upper <= 1.")


def _safe_divide(numerator: pd.DataFrame | pd.Series, denominator: pd.DataFrame | pd.Series):
    out = numerator / denominator
    if isinstance(out, pd.DataFrame):
        return out.replace([np.inf, -np.inf], np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _divide_with_axis(numerator: pd.DataFrame, denominator: pd.Series, axis: int) -> pd.DataFrame:
    """Divide a DataFrame by a Series aligned on rows (axis=1 case) or columns."""
    out = numerator.div(denominator, axis=0 if axis == 1 else 1)
    return out.replace([np.inf, -np.inf], np.nan)


def winsorize_quantile(
    values: pd.Series | pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = 1,
) -> pd.Series | pd.DataFrame:
    """Clip values using per-row or per-column quantiles."""
    _validate_quantile_bounds(lower, upper)
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")

    if isinstance(numeric, pd.Series):
        lo = numeric.quantile(lower)
        hi = numeric.quantile(upper)
        return numeric.clip(lower=lo, upper=hi)

    lo = numeric.quantile(lower, axis=axis)
    hi = numeric.quantile(upper, axis=axis)

    if axis == 1:
        lo_aligned = pd.DataFrame(np.repeat(lo.values[:, None], numeric.shape[1], axis=1), index=numeric.index, columns=numeric.columns)
        hi_aligned = pd.DataFrame(np.repeat(hi.values[:, None], numeric.shape[1], axis=1), index=numeric.index, columns=numeric.columns)
    else:
        lo_aligned = pd.DataFrame(np.repeat(lo.values[None, :], numeric.shape[0], axis=0), index=numeric.index, columns=numeric.columns)
        hi_aligned = pd.DataFrame(np.repeat(hi.values[None, :], numeric.shape[0], axis=0), index=numeric.index, columns=numeric.columns)

    return numeric.clip(lower=lo_aligned, upper=hi_aligned)


def winsorize_mad(values: pd.Series | pd.DataFrame, n_mad: float = 3.0, axis: int = 1) -> pd.Series | pd.DataFrame:
    """Robust outlier clipping based on median absolute deviation."""
    if n_mad <= 0:
        raise ValueError("n_mad must be positive.")
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")

    if isinstance(numeric, pd.Series):
        med = numeric.median()
        mad = (numeric - med).abs().median()
        scale = mad * 1.4826
        return numeric.clip(med - n_mad * scale, med + n_mad * scale)

    med = numeric.median(axis=axis)
    mad = numeric.sub(med, axis=0 if axis == 1 else 1).abs().median(axis=axis)
    scale = mad * 1.4826

    if axis == 1:
        center = pd.DataFrame(np.repeat(med.values[:, None], numeric.shape[1], axis=1), index=numeric.index, columns=numeric.columns)
        band = pd.DataFrame(np.repeat((n_mad * scale).values[:, None], numeric.shape[1], axis=1), index=numeric.index, columns=numeric.columns)
    else:
        center = pd.DataFrame(np.repeat(med.values[None, :], numeric.shape[0], axis=0), index=numeric.index, columns=numeric.columns)
        band = pd.DataFrame(np.repeat((n_mad * scale).values[None, :], numeric.shape[0], axis=0), index=numeric.index, columns=numeric.columns)

    return numeric.clip(lower=center - band, upper=center + band)


def winsorize_cross_section(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "mad",
    axis: int = 1,
    lower: float = 0.01,
    upper: float = 0.99,
    n_mad: float = 5.0,
) -> pd.Series | pd.DataFrame:
    """Winsorize a cross-section using a named method."""
    normalized_method = str(method).strip().lower()
    if normalized_method in {"quantile", "percentile"}:
        return winsorize_quantile(values, lower=lower, upper=upper, axis=axis)
    if normalized_method == "mad":
        return winsorize_mad(values, n_mad=n_mad, axis=axis)
    raise ValueError("method must be 'mad' or 'quantile'.")


def demean(values: pd.Series | pd.DataFrame, axis: int = 1) -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        return numeric - numeric.mean()
    return numeric.sub(numeric.mean(axis=axis), axis=0 if axis == 1 else 1)


def zscore(
    values: pd.Series | pd.DataFrame,
    axis: int = 1,
    ddof: int = 0,
    clip: tuple[float, float] | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")

    if isinstance(numeric, pd.Series):
        std = numeric.std(ddof=ddof)
        out = (numeric - numeric.mean()) / std
    else:
        mu = numeric.mean(axis=axis)
        sigma = numeric.std(axis=axis, ddof=ddof)
        centered = numeric.sub(mu, axis=0 if axis == 1 else 1)
        out = _divide_with_axis(centered, sigma, axis=axis)

    if clip is not None:
        out = out.clip(lower=clip[0], upper=clip[1])
    return out


def robust_zscore(
    values: pd.Series | pd.DataFrame,
    axis: int = 1,
    clip: tuple[float, float] | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")

    if isinstance(numeric, pd.Series):
        med = numeric.median()
        mad = (numeric - med).abs().median() * 1.4826
        out = (numeric - med) / mad
    else:
        med = numeric.median(axis=axis)
        mad = numeric.sub(med, axis=0 if axis == 1 else 1).abs().median(axis=axis) * 1.4826
        centered = numeric.sub(med, axis=0 if axis == 1 else 1)
        out = _divide_with_axis(centered, mad, axis=axis)

    if clip is not None:
        out = out.clip(lower=clip[0], upper=clip[1])
    return out


def rank_pct(values: pd.Series | pd.DataFrame, axis: int = 1, method: str = "average") -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        return numeric.rank(method=method, pct=True)
    return numeric.rank(axis=axis, method=method, pct=True)


def rank_centered(values: pd.Series | pd.DataFrame, axis: int = 1, method: str = "average") -> pd.Series | pd.DataFrame:
    return rank_pct(values, axis=axis, method=method) - 0.5


def minmax_scale(
    values: pd.Series | pd.DataFrame,
    axis: int = 1,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> pd.Series | pd.DataFrame:
    lo, hi = feature_range
    if hi <= lo:
        raise ValueError("feature_range must satisfy high > low.")

    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")

    if isinstance(numeric, pd.Series):
        mn = numeric.min()
        mx = numeric.max()
        scaled = _safe_divide(numeric - mn, mx - mn)
        return lo + scaled * (hi - lo)

    mn = numeric.min(axis=axis)
    mx = numeric.max(axis=axis)
    spread = mx - mn
    centered = numeric.sub(mn, axis=0 if axis == 1 else 1)
    scaled = _divide_with_axis(centered, spread, axis=axis)
    return lo + scaled * (hi - lo)


def l1_normalize(values: pd.Series | pd.DataFrame, axis: int = 1) -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        denom = numeric.abs().sum()
        return _safe_divide(numeric, denom)
    denom = numeric.abs().sum(axis=axis)
    return _divide_with_axis(numeric, denom, axis=axis)


def l2_normalize(values: pd.Series | pd.DataFrame, axis: int = 1) -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        denom = np.sqrt((numeric**2).sum())
        return _safe_divide(numeric, denom)
    denom = np.sqrt((numeric**2).sum(axis=axis))
    return _divide_with_axis(numeric, denom, axis=axis)


def normalize_cross_section(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "zscore",
    axis: int = 1,
    ddof: int = 0,
    clip: tuple[float, float] | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    rank_method: str = "average",
) -> pd.Series | pd.DataFrame:
    """Normalize a cross-section using a named method."""
    normalized_method = str(method).strip().lower()

    if normalized_method == "zscore":
        return zscore(values, axis=axis, ddof=ddof, clip=clip)
    if normalized_method in {"robust_zscore", "robust-zscore", "robust"}:
        return robust_zscore(values, axis=axis, clip=clip)
    if normalized_method == "demean":
        return demean(values, axis=axis)
    if normalized_method in {"rank", "rank_pct", "rank-pct"}:
        return rank_pct(values, axis=axis, method=rank_method)
    if normalized_method in {"rank_centered", "rank-centered"}:
        return rank_centered(values, axis=axis, method=rank_method)
    if normalized_method in {"minmax", "min_max"}:
        return minmax_scale(values, axis=axis, feature_range=feature_range)
    if normalized_method == "l1":
        return l1_normalize(values, axis=axis)
    if normalized_method == "l2":
        return l2_normalize(values, axis=axis)
    raise ValueError(
        "method must be one of 'zscore', 'robust_zscore', 'demean', "
        "'rank', 'rank_centered', 'minmax', 'l1', or 'l2'."
    )


def neutralize_by_group(
    values: pd.Series,
    groups: pd.Series,
    method: str = "demean",
) -> pd.Series:
    """Remove group effects from a cross-section."""
    numeric = pd.to_numeric(values, errors="coerce")
    aligned_groups = groups.reindex(numeric.index)

    if method == "demean":
        return numeric - numeric.groupby(aligned_groups).transform("mean")
    if method == "zscore":
        mu = numeric.groupby(aligned_groups).transform("mean")
        sigma = numeric.groupby(aligned_groups).transform(lambda series: series.std(ddof=0))
        return _safe_divide(numeric - mu, sigma)
    raise ValueError("method must be 'demean' or 'zscore'.")


def group_demean(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Demean values within each group."""
    return neutralize_by_group(values, groups, method="demean")


def group_zscore(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Z-score values within each group using population std."""
    return neutralize_by_group(values, groups, method="zscore")


def group_dummy_regression_residual(
    values: pd.Series,
    groups: pd.Series,
    *,
    drop_first: bool = True,
    add_intercept: bool = True,
) -> pd.Series:
    """Residualize a cross-section on group dummy variables."""
    numeric = pd.to_numeric(values, errors="coerce")
    aligned_groups = groups.reindex(numeric.index)
    frame = pd.concat([numeric.rename("_y"), aligned_groups.rename("_group")], axis=1).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=numeric.index, dtype=float)

    dummies = pd.get_dummies(frame["_group"], drop_first=drop_first, dtype=float)
    if dummies.empty:
        return frame["_y"] - frame["_y"].mean()
    return orthogonalize(frame["_y"], dummies, add_intercept=add_intercept).reindex(numeric.index)


def industry_regression_residual(
    values: pd.Series,
    industries: pd.Series,
    *,
    drop_first: bool = True,
    add_intercept: bool = True,
) -> pd.Series:
    """Residualize a cross-section on industry dummy variables."""
    return group_dummy_regression_residual(
        values,
        industries,
        drop_first=drop_first,
        add_intercept=add_intercept,
    )


def industry_size_regression_residual(
    values: pd.Series,
    industries: pd.Series,
    market_cap: pd.Series,
    *,
    drop_first: bool = True,
    add_intercept: bool = True,
    log_transform_size: bool = True,
) -> pd.Series:
    """Residualize a cross-section on industry dummies plus size exposure."""
    numeric = pd.to_numeric(values, errors="coerce")
    aligned_industries = industries.reindex(numeric.index)
    aligned_size = pd.to_numeric(market_cap.reindex(numeric.index), errors="coerce")
    if log_transform_size:
        aligned_size = np.log(aligned_size.replace(0, np.nan))
    aligned_size = pd.Series(aligned_size, index=numeric.index).replace([np.inf, -np.inf], np.nan)

    frame = pd.concat(
        [
            numeric.rename("_y"),
            aligned_industries.rename("_industry"),
            aligned_size.rename("_size"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=numeric.index, dtype=float)

    dummies = pd.get_dummies(frame["_industry"], drop_first=drop_first, dtype=float)
    exposures = pd.concat([dummies, frame["_size"]], axis=1)
    return orthogonalize(frame["_y"], exposures, add_intercept=add_intercept).reindex(numeric.index)


def neutralize_cross_section(
    values: pd.Series | pd.DataFrame,
    *,
    groups: pd.Series | None = None,
    exposures: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame] | None = None,
    method: str = "group_zscore",
    axis: int = 1,
    add_intercept: bool = True,
) -> pd.Series | pd.DataFrame:
    """Neutralize a cross-section by group or regression exposures."""
    normalized_method = str(method).strip().lower()

    if normalized_method in {"group_demean", "demean"}:
        if groups is None:
            raise ValueError("groups is required for group neutralization.")
        if isinstance(values, pd.DataFrame):
            if axis != 1:
                raise ValueError("group neutralization for DataFrame currently requires axis=1.")
            return values.apply(lambda row: neutralize_by_group(row, groups.loc[row.name], method="demean"), axis=1)
        return neutralize_by_group(values, groups, method="demean")

    if normalized_method in {"group_zscore", "zscore"}:
        if groups is None:
            raise ValueError("groups is required for group neutralization.")
        if isinstance(values, pd.DataFrame):
            if axis != 1:
                raise ValueError("group neutralization for DataFrame currently requires axis=1.")
            return values.apply(lambda row: neutralize_by_group(row, groups.loc[row.name], method="zscore"), axis=1)
        return neutralize_by_group(values, groups, method="zscore")

    if normalized_method in {"regression", "residual", "orthogonalize"}:
        if exposures is None:
            raise ValueError("exposures is required for regression neutralization.")
        if isinstance(values, pd.Series):
            if isinstance(exposures, list):
                exposure_frame = pd.concat(exposures, axis=1)
            elif isinstance(exposures, pd.Series):
                exposure_frame = exposures.to_frame()
            else:
                exposure_frame = exposures
            return orthogonalize(values, exposure_frame, add_intercept=add_intercept)

        exposure_list = exposures if isinstance(exposures, list) else [exposures]
        prepared_exposures = []
        for exposure in exposure_list:
            if isinstance(exposure, pd.Series):
                prepared_exposures.append(pd.DataFrame([exposure.reindex(values.columns)], index=values.index))
            else:
                prepared_exposures.append(exposure)
        return cs_neutralize(values, exposures=prepared_exposures, add_intercept=add_intercept)

    raise ValueError(
        "method must be one of 'group_demean', 'group_zscore', or 'regression'."
    )


def preprocess_cross_section(
    values: pd.Series | pd.DataFrame,
    *,
    winsorize_method: str | None = "mad",
    normalize_method: str | None = "zscore",
    neutralize_method: str | None = None,
    axis: int = 1,
    groups: pd.Series | pd.DataFrame | None = None,
    exposures: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
    n_mad: float = 5.0,
    ddof: int = 0,
    clip: tuple[float, float] | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    rank_method: str = "average",
    add_intercept: bool = True,
) -> pd.Series | pd.DataFrame:
    """Run winsorize -> normalize -> neutralize on a cross-section."""
    out = values
    if winsorize_method is not None:
        out = winsorize_cross_section(out, method=winsorize_method, axis=axis, lower=lower, upper=upper, n_mad=n_mad)
    if normalize_method is not None:
        out = normalize_cross_section(
            out,
            method=normalize_method,
            axis=axis,
            ddof=ddof,
            clip=clip,
            feature_range=feature_range,
            rank_method=rank_method,
        )
    if neutralize_method is not None:
        out = neutralize_cross_section(
            out,
            groups=groups,
            exposures=exposures,
            method=neutralize_method,
            axis=axis,
            add_intercept=add_intercept,
        )
    return out


def orthogonalize(target: pd.Series, exposures: pd.DataFrame, add_intercept: bool = True) -> pd.Series:
    """Return residuals of target regressed on exposures."""
    y = pd.to_numeric(target, errors="coerce")
    X = exposures.apply(pd.to_numeric, errors="coerce")

    frame = pd.concat([y.rename("_y"), X], axis=1).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=target.index, dtype=float)

    yv = frame["_y"].to_numpy(dtype=float)
    xv = frame.drop(columns=["_y"]).to_numpy(dtype=float)
    if add_intercept:
        xv = np.c_[np.ones(len(frame)), xv]

    beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    fitted = xv @ beta
    resid = pd.Series(yv - fitted, index=frame.index, dtype=float)
    return resid.reindex(target.index)


def bucketize_quantiles(values: pd.Series, q: int = 10, labels: list[int] | None = None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if labels is None:
        labels = list(range(1, q + 1))
    ranked = numeric.rank(method="average", pct=True)
    bins = np.linspace(0, 1, q + 1)
    # include_lowest ensures the smallest rank lands in the first bucket.
    return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True)


def col_minmax_pos(col: pd.Series, mask: pd.Series | None = None) -> pd.Series:
    numeric = pd.to_numeric(col, errors="coerce")
    active = numeric.where(mask) if mask is not None else numeric
    vmin = active.min(skipna=True)
    vmax = active.max(skipna=True)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        out = pd.Series(0.5, index=numeric.index, dtype=float)
    else:
        out = (numeric - vmin) / (vmax - vmin)
    if mask is not None:
        out = out.where(mask, 0.0)
    return out.fillna(0.0)


def col_minmax_neg(col: pd.Series, mask: pd.Series | None = None) -> pd.Series:
    return 1.0 - col_minmax_pos(col, mask)


def cs_minmax_pos(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    row_min = numeric.min(axis=1)
    row_max = numeric.max(axis=1)
    spread = row_max - row_min
    out = numeric.sub(row_min, axis=0).div(spread.replace(0, np.nan), axis=0)
    equal_rows = spread.eq(0) & numeric.notna().any(axis=1)
    if equal_rows.any():
        out.loc[equal_rows, :] = 0.5
    return out


def cs_minmax_neg(df: pd.DataFrame) -> pd.DataFrame:
    return 1.0 - cs_minmax_pos(df)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return rank_pct(df, axis=1)


def cs_zscore(df: pd.DataFrame, ddof: int = 0) -> pd.DataFrame:
    return zscore(df, axis=1, ddof=ddof)


def cs_winsorize(df: pd.DataFrame, low: float = 0.01, high: float = 0.99) -> pd.DataFrame:
    return winsorize_quantile(df, lower=low, upper=high, axis=1)


def cs_winsorize_mad(df: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    return winsorize_mad(df, n_mad=k, axis=1)


def cs_neutralize(
    y: pd.DataFrame,
    exposures: list[pd.DataFrame] | None = None,
    add_intercept: bool = True,
) -> pd.DataFrame:
    target = y.apply(pd.to_numeric, errors="coerce")
    if not exposures:
        return target.sub(target.mean(axis=1), axis=0)

    prepared_exposures = [exposure.reindex_like(target).apply(pd.to_numeric, errors="coerce") for exposure in exposures]
    residuals = pd.DataFrame(index=target.index, columns=target.columns, dtype="float64")

    for date in target.index:
        y_row = target.loc[date]
        mask = y_row.notna()
        if int(mask.sum()) < 5:
            continue

        matrices = []
        for exposure in prepared_exposures:
            exposure_row = exposure.loc[date, mask]
            matrices.append(exposure_row.to_numpy(dtype=float).reshape(-1, 1))

        if not matrices:
            continue

        design = np.concatenate(matrices, axis=1)
        finite_mask = np.isfinite(design).all(axis=1) & np.isfinite(y_row[mask].to_numpy(dtype=float))
        if int(finite_mask.sum()) < max(2, design.shape[1] + int(add_intercept)):
            continue

        design = design[finite_mask]
        values = y_row[mask].to_numpy(dtype=float)[finite_mask]
        asset_index = y_row[mask].index[finite_mask]

        if add_intercept:
            design = np.c_[np.ones((design.shape[0], 1)), design]

        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ beta
        residuals.loc[date, asset_index] = values - fitted

    return residuals


def make_onehot_by_group(groups: pd.DataFrame | pd.Series, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(groups, pd.Series):
        dummies = pd.get_dummies(groups)
        if columns is not None:
            dummies = dummies.reindex(index=columns).fillna(0.0)
        return dummies.T
    raise ValueError("Dynamic groups should be expanded per date with pd.get_dummies(groups.loc[t]).")


def neutralize_industry_size(
    y: pd.DataFrame,
    industry: pd.DataFrame,
    log_mktcap: pd.DataFrame,
    drop_first: bool = True,
    add_intercept: bool = True,
    winsor_k: float = 5.0,
) -> pd.DataFrame:
    y_clean = cs_winsorize_mad(y, k=winsor_k)
    size_clean = cs_winsorize_mad(log_mktcap, k=winsor_k)
    size_z = cs_zscore(size_clean)
    residuals = pd.DataFrame(index=y.index, columns=y.columns, dtype="float64")

    for date in y.index:
        y_row = y_clean.loc[date]
        industry_row = industry.loc[date]
        size_row = size_z.loc[date]
        mask = y_row.notna() & industry_row.notna() & size_row.notna()
        if int(mask.sum()) < 5:
            continue

        dummies = pd.get_dummies(industry_row[mask], drop_first=drop_first)
        design = np.hstack([dummies.to_numpy(dtype=float), size_row[mask].to_numpy(dtype=float).reshape(-1, 1)])
        if add_intercept:
            design = np.c_[np.ones((design.shape[0], 1)), design]

        beta, *_ = np.linalg.lstsq(design, y_row[mask].to_numpy(dtype=float), rcond=None)
        residuals.loc[date, mask] = y_row[mask].to_numpy(dtype=float) - (design @ beta)

    return residuals


def make_industry_dummies(
    df: pd.DataFrame,
    industry_col: str = "industry",
    all_industries: list[str] | None = None,
    baseline: str | None = None,
    prefix: str = "ind",
    drop_first: bool = True,
    keep_full_set: bool = True,
    dtype: str = "int64",
) -> pd.DataFrame:
    if industry_col not in df.columns:
        raise KeyError(f"{industry_col!r} not in df.columns")

    if all_industries is None:
        all_industries = sorted(pd.Index(df[industry_col].dropna().unique()).astype(str).tolist())

    if baseline is not None and baseline in all_industries:
        all_industries = [baseline] + [category for category in all_industries if category != baseline]

    categories = pd.Categorical(df[industry_col].astype("string"), categories=all_industries, ordered=True)
    dummies = pd.get_dummies(categories, prefix=prefix, drop_first=drop_first)

    if keep_full_set:
        if drop_first:
            expected = [f"{prefix}_{category}" for category in all_industries[1:]]
        else:
            expected = [f"{prefix}_{category}" for category in all_industries]
        dummies = dummies.reindex(columns=expected, fill_value=0)

    return dummies.astype(dtype)
