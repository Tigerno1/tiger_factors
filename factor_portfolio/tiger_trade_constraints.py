from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TigerTradeConstraintConfig:
    """Practical trading constraints for stock-level factor portfolios."""

    min_price: float | None = 1.0
    min_market_cap: float | None = None
    market_cap_window: int = 1
    min_volume: float | None = None
    volume_window: int = 20
    min_dollar_volume: float | None = 1_000_000.0
    dollar_volume_window: int = 20
    require_price: bool = True
    require_volume: bool = False
    exclude_halted: bool = True
    require_shortable_for_short: bool = True
    max_single_name_weight: float | None = 0.05
    max_industry_weight: float | None = 0.30
    min_eligible_assets: int = 1
    normalize_after_constraints: bool = True


@dataclass(frozen=True)
class TigerTradeConstraintData:
    """Wide date x code data used by the trading constraint layer."""

    close: pd.DataFrame | pd.Series
    volume: pd.DataFrame | pd.Series | Mapping[str, float] | None = None
    dollar_volume: pd.DataFrame | pd.Series | Mapping[str, float] | None = None
    market_cap: pd.DataFrame | pd.Series | Mapping[str, float] | None = None
    industry: pd.DataFrame | pd.Series | Mapping[str, str] | None = None
    shortable: pd.DataFrame | pd.Series | Mapping[str, bool] | None = None
    halted: pd.DataFrame | pd.Series | Mapping[str, bool] | None = None


@dataclass(frozen=True)
class TigerTradeConstraintResult:
    values: pd.DataFrame
    eligible_mask: pd.DataFrame
    summary: pd.DataFrame
    config: TigerTradeConstraintConfig


def _first_value_column(frame: pd.DataFrame, preferred: str | None) -> str:
    if preferred is not None and preferred in frame.columns:
        return preferred
    candidates = [column for column in frame.columns if column not in {"date_", "code"}]
    if len(candidates) != 1:
        raise ValueError("long constraint data must have one value column, or pass a known value column")
    return str(candidates[0])


def _coerce_panel(
    data: pd.DataFrame | pd.Series | Mapping[str, object] | None,
    *,
    index: pd.DatetimeIndex | None = None,
    columns: pd.Index | list[str] | tuple[str, ...] | None = None,
    value_column: str | None = None,
    numeric: bool,
    ffill: bool = True,
) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()

    if isinstance(data, Mapping):
        data = pd.Series(dict(data))

    if isinstance(data, pd.Series):
        if isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 2:
            names = list(data.index.names)
            date_level = names.index("date_") if "date_" in names else 0
            code_level = names.index("code") if "code" in names else 1
            wide = data.unstack(code_level)
            if date_level != 0:
                wide.index = wide.index.get_level_values(date_level)
        else:
            if index is None:
                raise ValueError("static Series constraint data requires a target datetime index")
            series = data.copy()
            series.index = series.index.astype(str)
            if columns is None:
                target_columns = pd.Index(series.index.astype(str))
            elif isinstance(columns, pd.Index):
                target_columns = columns.astype(str)
            else:
                target_columns = pd.Index(list(map(str, columns)))
            wide = pd.DataFrame(index=index, columns=target_columns)
            for code in target_columns:
                if code in series.index:
                    wide.loc[:, code] = series.loc[code]
    elif isinstance(data, pd.DataFrame):
        if {"date_", "code"}.issubset(data.columns):
            value = _first_value_column(data, value_column)
            frame = data.loc[:, ["date_", "code", value]].copy()
            frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce").dt.tz_localize(None)
            frame["code"] = frame["code"].astype(str)
            wide = frame.dropna(subset=["date_", "code"]).pivot_table(
                index="date_",
                columns="code",
                values=value,
                aggfunc="last",
            )
        else:
            wide = data.copy()
    else:
        raise TypeError("constraint data must be a pandas object, mapping, or None")

    wide = wide.copy()
    if not isinstance(wide.index, pd.DatetimeIndex):
        wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide = wide.loc[~wide.index.isna()].sort_index()
    wide.index = pd.DatetimeIndex(wide.index).tz_localize(None)
    wide.columns = wide.columns.astype(str)

    if index is not None:
        wide = wide.reindex(pd.DatetimeIndex(index))
        if ffill:
            with pd.option_context("future.no_silent_downcasting", True):
                wide = wide.ffill().infer_objects(copy=False)
    if columns is not None:
        wide = wide.reindex(columns=pd.Index(columns).astype(str))
    if numeric:
        wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide


def _coerce_bool_panel(
    data: pd.DataFrame | pd.Series | Mapping[str, bool] | None,
    *,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    default: bool,
) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame(default, index=index, columns=columns)
    panel = _coerce_panel(data, index=index, columns=columns, numeric=False, ffill=True)
    return panel.fillna(default).astype(bool)


def _rolling_mean(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    window = max(int(window), 1)
    if window <= 1:
        return panel
    min_periods = max(1, min(window, 5))
    return panel.rolling(window=window, min_periods=min_periods).mean()


def _align_mask(mask: pd.DataFrame, *, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    aligned = mask.reindex(index=pd.DatetimeIndex(index), method="ffill").reindex(columns=columns.astype(str))
    return aligned.fillna(False).astype(bool)


def _industry_labels_for_date(
    industry: pd.DataFrame | pd.Series | Mapping[str, str] | None,
    *,
    date_: pd.Timestamp,
    columns: pd.Index,
) -> pd.Series:
    if industry is None:
        return pd.Series(index=columns.astype(str), dtype=object)
    if isinstance(industry, Mapping):
        labels = pd.Series(dict(industry), dtype=object)
    elif isinstance(industry, pd.Series):
        if isinstance(industry.index, pd.MultiIndex):
            panel = _coerce_panel(industry, index=pd.DatetimeIndex([date_]), columns=columns, numeric=False)
            return panel.iloc[-1].reindex(columns.astype(str))
        labels = industry.copy()
    elif isinstance(industry, pd.DataFrame):
        panel = _coerce_panel(industry, index=pd.DatetimeIndex([date_]), columns=columns, numeric=False)
        return panel.iloc[-1].reindex(columns.astype(str))
    else:
        raise TypeError("industry must be a pandas object, mapping, or None")
    labels.index = labels.index.astype(str)
    return labels.reindex(columns.astype(str))


def build_tradeable_universe_mask(
    data: TigerTradeConstraintData,
    config: TigerTradeConstraintConfig | None = None,
    *,
    side: str = "long",
) -> pd.DataFrame:
    """Return a boolean date x code mask of assets that pass trading filters."""

    config = config or TigerTradeConstraintConfig()
    side_key = str(side).strip().lower()
    if side_key not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")

    close = _coerce_panel(data.close, numeric=True, ffill=False)
    if close.empty:
        return pd.DataFrame(dtype=bool)

    index = pd.DatetimeIndex(close.index)
    columns = close.columns.astype(str)
    mask = pd.DataFrame(True, index=index, columns=columns)

    if config.require_price:
        valid_price = np.isfinite(close) & close.gt(0.0)
        if config.min_price is not None:
            valid_price &= close.ge(float(config.min_price))
        mask &= valid_price.fillna(False)

    volume = _coerce_panel(data.volume, index=index, columns=columns, value_column="volume", numeric=True, ffill=False)
    if config.require_volume and not volume.empty:
        mask &= (np.isfinite(volume) & volume.gt(0.0)).fillna(False)
    if config.min_volume is not None and not volume.empty:
        average_volume = _rolling_mean(volume, config.volume_window)
        mask &= average_volume.ge(float(config.min_volume)).fillna(False)

    dollar_volume = _coerce_panel(
        data.dollar_volume,
        index=index,
        columns=columns,
        value_column="dollar_volume",
        numeric=True,
        ffill=False,
    )
    if dollar_volume.empty or dollar_volume.isna().all(axis=None):
        if not volume.empty:
            dollar_volume = close * volume
    if config.min_dollar_volume is not None and not dollar_volume.empty:
        average_dollar_volume = _rolling_mean(dollar_volume, config.dollar_volume_window)
        mask &= average_dollar_volume.ge(float(config.min_dollar_volume)).fillna(False)

    market_cap = _coerce_panel(data.market_cap, index=index, columns=columns, value_column="market_cap", numeric=True)
    if config.min_market_cap is not None and not market_cap.empty:
        average_market_cap = _rolling_mean(market_cap, config.market_cap_window)
        mask &= average_market_cap.ge(float(config.min_market_cap)).fillna(False)

    if config.exclude_halted:
        halted = _coerce_bool_panel(data.halted, index=index, columns=columns, default=False)
        mask &= ~halted

    if side_key == "short" and config.require_shortable_for_short and data.shortable is not None:
        shortable = _coerce_bool_panel(data.shortable, index=index, columns=columns, default=False)
        mask &= shortable

    min_assets = max(int(config.min_eligible_assets), 0)
    if min_assets > 1:
        enough_assets = mask.sum(axis=1) >= min_assets
        mask.loc[~enough_assets, :] = False

    return mask.astype(bool)


def summarize_trade_constraints(mask: pd.DataFrame) -> pd.DataFrame:
    """Summarize the eligible universe count and ratio through time."""

    if mask.empty:
        return pd.DataFrame(columns=["eligible_count", "universe_count", "eligible_ratio"])
    eligible_count = mask.sum(axis=1).astype(int)
    universe_count = pd.Series(mask.shape[1], index=mask.index, dtype=int)
    summary = pd.DataFrame(
        {
            "eligible_count": eligible_count,
            "universe_count": universe_count,
            "eligible_ratio": eligible_count / universe_count.replace(0, np.nan),
        }
    )
    summary.index.name = "date_"
    return summary


def apply_trade_constraints_to_scores(
    scores: pd.DataFrame | pd.Series,
    data: TigerTradeConstraintData,
    config: TigerTradeConstraintConfig | None = None,
    *,
    side: str = "long",
) -> TigerTradeConstraintResult:
    """Mask factor scores before candidate selection."""

    score_panel = _coerce_panel(scores, numeric=True, ffill=False)
    mask = build_tradeable_universe_mask(data, config=config, side=side)
    eligible = _align_mask(mask, index=pd.DatetimeIndex(score_panel.index), columns=score_panel.columns)
    filtered = score_panel.where(eligible)
    return TigerTradeConstraintResult(
        values=filtered,
        eligible_mask=eligible,
        summary=summarize_trade_constraints(eligible),
        config=config or TigerTradeConstraintConfig(),
    )


def _normalize_nonnegative_with_caps(
    weights: pd.Series,
    *,
    target_gross: float,
    max_weight: float | None,
    labels: pd.Series,
    max_industry_weight: float | None,
) -> pd.Series:
    clean = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    clean = clean.loc[clean > 0.0]
    result = pd.Series(0.0, index=weights.index.astype(str), dtype=float)
    target = max(float(target_gross), 0.0)
    if clean.empty or target <= 1e-12:
        return result

    asset_cap = target if max_weight is None else max(float(max_weight), 0.0)
    group_cap = None if max_industry_weight is None else max(float(max_industry_weight), 0.0)
    aligned_labels = labels.reindex(clean.index.astype(str))
    group_keys = pd.Series(
        [
            str(value) if pd.notna(value) else f"__ungrouped__:{code}"
            for code, value in aligned_labels.items()
        ],
        index=clean.index.astype(str),
    )

    remaining = clean.copy()
    remaining_target = target
    for _ in range(len(clean) * 2 + 2):
        total = float(remaining.sum())
        if total <= 1e-12 or remaining_target <= 1e-12:
            break
        proposed = remaining / total * remaining_target
        capacities = pd.Series(asset_cap, index=remaining.index, dtype=float) - result.reindex(remaining.index).fillna(0.0)
        if group_cap is not None:
            used_by_group = result.reindex(clean.index).fillna(0.0).groupby(group_keys).sum()
            for code in remaining.index:
                group = group_keys.loc[code]
                capacities.loc[code] = min(capacities.loc[code], group_cap - float(used_by_group.get(group, 0.0)))
        capacities = capacities.clip(lower=0.0)
        active = capacities > 1e-12
        if not bool(active.any()):
            break
        active_index = active.loc[active].index
        proposed = proposed.loc[active_index]
        capacities = capacities.loc[active_index]
        allocation = proposed.clip(upper=capacities)
        allocated = float(allocation.sum())
        if allocated <= 1e-12:
            break
        result.loc[allocation.index] += allocation
        remaining_target -= allocated
        if remaining_target <= 1e-12:
            break
        remaining = remaining.loc[active_index]
    return result


def _apply_industry_cap(
    weights: pd.Series,
    *,
    labels: pd.Series,
    max_industry_weight: float | None,
) -> pd.Series:
    if max_industry_weight is None or labels.empty:
        return weights
    cap = max(float(max_industry_weight), 0.0)
    if cap <= 0.0:
        return weights * 0.0

    result = weights.copy()
    aligned_labels = labels.reindex(result.index.astype(str))
    for industry_name, members in aligned_labels.dropna().groupby(aligned_labels.dropna()).groups.items():
        member_index = pd.Index(members)
        gross = float(result.loc[member_index].abs().sum())
        if gross > cap + 1e-12:
            result.loc[member_index] *= cap / gross
    return result


def _constrain_weight_row(
    row: pd.Series,
    *,
    date_: pd.Timestamp,
    data: TigerTradeConstraintData,
    config: TigerTradeConstraintConfig,
    target_long_gross: float,
    target_short_gross: float,
) -> pd.Series:
    columns = row.index.astype(str)
    labels = _industry_labels_for_date(data.industry, date_=date_, columns=columns)

    positive = row.clip(lower=0.0)
    negative = (-row.clip(upper=0.0))

    if config.normalize_after_constraints:
        positive = _normalize_nonnegative_with_caps(
            positive,
            target_gross=target_long_gross,
            max_weight=config.max_single_name_weight,
            labels=labels,
            max_industry_weight=config.max_industry_weight,
        )
        negative = _normalize_nonnegative_with_caps(
            negative,
            target_gross=target_short_gross,
            max_weight=config.max_single_name_weight,
            labels=labels,
            max_industry_weight=config.max_industry_weight,
        )
    elif config.max_single_name_weight is not None:
        positive = positive.clip(upper=float(config.max_single_name_weight))
        negative = negative.clip(upper=float(config.max_single_name_weight))

    constrained = positive - negative
    if not config.normalize_after_constraints:
        constrained = _apply_industry_cap(
            constrained,
            labels=labels,
            max_industry_weight=config.max_industry_weight,
        )

    if config.max_single_name_weight is not None:
        cap = float(config.max_single_name_weight)
        constrained = constrained.clip(lower=-cap, upper=cap)
    return constrained.reindex(columns).fillna(0.0)


def apply_trade_constraints_to_weights(
    weights: pd.DataFrame | pd.Series,
    data: TigerTradeConstraintData,
    config: TigerTradeConstraintConfig | None = None,
) -> pd.DataFrame:
    """Apply tradability, single-name, shortability, and industry caps to weights."""

    config = config or TigerTradeConstraintConfig()
    weight_panel = _coerce_panel(weights, numeric=True, ffill=False).fillna(0.0)
    if weight_panel.empty:
        return weight_panel

    long_mask = _align_mask(
        build_tradeable_universe_mask(data, config=config, side="long"),
        index=pd.DatetimeIndex(weight_panel.index),
        columns=weight_panel.columns,
    )
    short_mask = _align_mask(
        build_tradeable_universe_mask(data, config=config, side="short"),
        index=pd.DatetimeIndex(weight_panel.index),
        columns=weight_panel.columns,
    )

    original = weight_panel.copy()
    constrained = weight_panel.clip(lower=0.0).where(long_mask, 0.0) + weight_panel.clip(upper=0.0).where(short_mask, 0.0)

    rows: list[pd.Series] = []
    for date_, row in constrained.iterrows():
        target_long = float(original.loc[date_].clip(lower=0.0).sum())
        target_short = float((-original.loc[date_].clip(upper=0.0)).sum())
        cleaned = _constrain_weight_row(
            row,
            date_=pd.Timestamp(date_),
            data=data,
            config=config,
            target_long_gross=target_long,
            target_short_gross=target_short,
        )
        rows.append(cleaned.rename(pd.Timestamp(date_)))

    result = pd.DataFrame(rows).sort_index().reindex(columns=weight_panel.columns).fillna(0.0)
    result.index.name = weight_panel.index.name or "date_"
    return result
