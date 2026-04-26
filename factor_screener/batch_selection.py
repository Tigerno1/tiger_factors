from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tiger_factors.factor_screener._evaluation_io import load_factor_panel
from tiger_factors.factor_screener._evaluation_io import load_ic_series
from tiger_factors.factor_screener._evaluation_io import load_return_series
from tiger_factors.factor_screener.factor_screener import FactorScreener
from tiger_factors.factor_screener.factor_screener import FactorScreenerResult
from tiger_factors.factor_screener.factor_screener import FactorScreenerSpec
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series


class FactorSelectionMode:
    CORRELATION = "correlation"
    CONDITIONAL = "conditional"
    RETURN_GAIN = "return_gain"

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return (cls.CORRELATION, cls.CONDITIONAL, cls.RETURN_GAIN)

    @classmethod
    def normalize(cls, value: str | None) -> str:
        normalized = cls.CORRELATION if value is None else str(value).strip().lower()
        if normalized in cls.choices():
            return normalized
        raise ValueError(f"selection_mode must be one of: {', '.join(cls.choices())}")


@dataclass(frozen=True)
class FactorReturnGainSelectionConfig:
    metric_fields: tuple[str, ...] = ("directional_fitness", "directional_ic_ir", "directional_sharpe")
    metric_weights: tuple[float, ...] = (0.50, 0.25, 0.25)
    metric_fallback_field: str = "selected_score"
    metric_penalty_fields: tuple[str, ...] = ("turnover", "max_drawdown")
    metric_penalty_weights: tuple[float, ...] = (0.20, 0.10)
    ann_return_weight: float = 1.0
    sharpe_weight: float = 1.0
    max_drawdown_weight: float = 0.5
    win_rate_weight: float = 0.0
    corr_weight: float = 0.50
    min_improvement: float = 0.0
    min_base_objective: float | None = None

    @classmethod
    def metric_focused(cls) -> FactorReturnGainSelectionConfig:
        return cls(
            metric_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            metric_weights=(0.60, 0.25, 0.15),
            ann_return_weight=0.75,
            sharpe_weight=0.75,
            max_drawdown_weight=0.35,
            corr_weight=0.35,
        )

    @classmethod
    def return_focused(cls) -> FactorReturnGainSelectionConfig:
        return cls(
            metric_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            metric_weights=(0.35, 0.25, 0.15),
            ann_return_weight=1.25,
            sharpe_weight=1.25,
            max_drawdown_weight=0.65,
            corr_weight=0.50,
        )

    @classmethod
    def robust(cls) -> FactorReturnGainSelectionConfig:
        return cls(
            metric_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            metric_weights=(0.45, 0.30, 0.25),
            ann_return_weight=0.90,
            sharpe_weight=0.90,
            max_drawdown_weight=0.80,
            win_rate_weight=0.10,
            corr_weight=0.60,
        )

    @classmethod
    def balanced(cls) -> FactorReturnGainSelectionConfig:
        return cls()


def resolve_selection_mode(selection_mode: str | None) -> str:
    return FactorSelectionMode.normalize(selection_mode)


def resolve_return_gain_config(
    config: FactorReturnGainSelectionConfig | None,
    preset: str | None,
) -> FactorReturnGainSelectionConfig:
    if config is not None:
        return config
    if preset is None:
        return FactorReturnGainSelectionConfig()
    normalized = str(preset).strip().lower()
    if normalized in {"balanced", "default"}:
        return FactorReturnGainSelectionConfig.balanced()
    if normalized in {"metric", "metric_focused"}:
        return FactorReturnGainSelectionConfig.metric_focused()
    if normalized in {"return", "return_focused"}:
        return FactorReturnGainSelectionConfig.return_focused()
    if normalized in {"robust", "stable"}:
        return FactorReturnGainSelectionConfig.robust()
    raise ValueError(
        "preset must be one of: balanced, metric_focused, return_focused, robust"
    )


def annualized_return_stats(returns: pd.Series, annual_trading_days: int = 252) -> dict[str, float]:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    cumulative = (1.0 + series).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (annual_trading_days / max(len(series), 1)) - 1.0)
    ann_vol = float(series.std(ddof=0) * np.sqrt(annual_trading_days))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((series > 0).mean())
    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def return_objective(returns: pd.Series, config: FactorReturnGainSelectionConfig) -> float:
    stats = annualized_return_stats(returns)
    return (
        float(config.ann_return_weight) * float(stats["ann_return"])
        + float(config.sharpe_weight) * float(stats["sharpe"])
        - float(config.max_drawdown_weight) * abs(float(stats["max_drawdown"]))
        + float(config.win_rate_weight) * float(stats["win_rate"])
    )


def metric_objective(frame: pd.DataFrame, config: FactorReturnGainSelectionConfig) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if len(config.metric_fields) != len(config.metric_weights):
        raise ValueError("metric_fields and metric_weights must have the same length")
    if len(config.metric_penalty_fields) != len(config.metric_penalty_weights):
        raise ValueError("metric_penalty_fields and metric_penalty_weights must have the same length")

    metric_score = pd.Series(0.0, index=frame.index, dtype=float)
    metric_weight_total = 0.0
    for field, weight in zip(config.metric_fields, config.metric_weights):
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        metric_score = metric_score.add(values.fillna(0.0) * float(weight), fill_value=0.0)
        metric_weight_total += float(weight)
    if metric_weight_total > 0:
        metric_score = metric_score / metric_weight_total
    elif config.metric_fallback_field in frame.columns:
        metric_score = pd.to_numeric(frame[config.metric_fallback_field], errors="coerce")
    elif "fitness" in frame.columns:
        metric_score = pd.to_numeric(frame["fitness"], errors="coerce")

    metric_penalty = pd.Series(0.0, index=frame.index, dtype=float)
    for field, weight in zip(config.metric_penalty_fields, config.metric_penalty_weights):
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce").abs()
        metric_penalty = metric_penalty.add(values.fillna(0.0) * float(weight), fill_value=0.0)

    return metric_score - metric_penalty


def item_label(item, index: int) -> str:
    label = str(item.label).strip() if item.label is not None else ""
    return label or f"spec_{index + 1}"


def spec_signature(spec: FactorScreenerSpec) -> str:
    return ":".join(
        [
            str(spec.selection_threshold),
            str(spec.selection_score_field),
            str(spec.correlation_method),
            str(spec.ic_correlation_method),
        ]
    )


def tag_frame(
    frame: pd.DataFrame,
    *,
    batch_index: int,
    batch_label: str,
    batch_signature: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    tagged = frame.copy()
    tagged["batch_index"] = batch_index
    tagged["batch_label"] = batch_label
    tagged["batch_signature"] = batch_signature
    if "factor_name" in tagged.columns:
        tagged["batch_factor_key"] = tagged["factor_name"].astype(str).map(lambda name: f"{batch_label}::{name}")
    return tagged


def select_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame
    if "selected" in selected.columns:
        selected = selected.loc[selected["selected"].fillna(False)]
    elif "usable" in selected.columns:
        selected = selected.loc[selected["usable"].fillna(False)]
    return selected.copy()


def factor_frame_to_panel(frame: pd.DataFrame, *, factor_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if "date_" in normalized.columns:
        normalized["date_"] = pd.to_datetime(normalized["date_"], errors="coerce")
    if "code" in normalized.columns:
        normalized["code"] = normalized["code"].astype(str)

    if {"date_", "code"}.issubset(normalized.columns):
        value_candidates = [
            column
            for column in normalized.columns
            if column not in {"date_", "code"}
            and pd.api.types.is_numeric_dtype(normalized[column])
        ]
        if not value_candidates:
            return pd.DataFrame()
        value_column = "value" if "value" in value_candidates else value_candidates[0]
        panel = normalized.pivot_table(index="date_", columns="code", values=value_column, aggfunc="last")
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.loc[~panel.index.isna()].sort_index()
        panel.columns = panel.columns.astype(str)
        panel.columns.name = factor_name
        return panel

    if isinstance(normalized.index, pd.DatetimeIndex):
        panel = normalized.copy()
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.loc[~panel.index.isna()].sort_index()
        panel.columns = panel.columns.astype(str)
        panel.columns.name = factor_name
        return panel

    return pd.DataFrame()


def score_series(
    frame: pd.DataFrame,
    *,
    score_field: str,
) -> pd.Series:
    if frame.empty or "factor_name" not in frame.columns:
        return pd.Series(dtype=float)

    if score_field in frame.columns:
        values = pd.to_numeric(frame[score_field], errors="coerce")
    elif score_field == "selected_score" and "selected_score" in frame.columns:
        values = pd.to_numeric(frame["selected_score"], errors="coerce")
    elif "selected_score" in frame.columns:
        values = pd.to_numeric(frame["selected_score"], errors="coerce")
    elif "fitness" in frame.columns:
        values = pd.to_numeric(frame["fitness"], errors="coerce").abs()
    else:
        values = pd.Series(1.0, index=frame.index, dtype=float)

    values.index = frame["factor_name"].astype(str)
    return values.replace([pd.NA, pd.NaT], pd.NA)


def primary_return_mode(result: FactorScreenerResult) -> str:
    available_modes = sorted(
        {str(key.split(":", 1)[1]) for key in result.return_series if ":" in key}
    )
    if "long_short" in available_modes:
        return "long_short"
    if "long_only" in available_modes:
        return "long_only"
    return "long_short"


def extract_primary_return_series(
    result: FactorScreenerResult,
    *,
    primary_mode: str,
) -> dict[str, pd.Series]:
    extracted: dict[str, pd.Series] = {}
    for key, series in result.return_series.items():
        if not key.endswith(f":{primary_mode}"):
            continue
        factor_name = key.split(":", 1)[0]
        if isinstance(series, pd.Series) and not series.empty:
            extracted[factor_name] = series.copy()
    return extracted


def combined_return_panel(results: list[tuple[str, FactorScreenerResult]]) -> pd.DataFrame:
    from collections import Counter

    column_counts: Counter[str] = Counter()
    for _, result in results:
        column_counts.update(str(column) for column in result.return_panel.columns)

    frames: dict[str, pd.Series] = {}
    for batch_label, result in results:
        panel = result.return_panel.copy()
        if panel.empty:
            continue
        for column in panel.columns:
            name = str(column)
            composite_key = name if column_counts[name] == 1 else f"{batch_label}::{name}"
            frames[composite_key] = panel[column]
    if not frames:
        return pd.DataFrame()
    frame = pd.DataFrame(frames).sort_index()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    return frame


def build_result_return_artifacts(
    result: FactorScreenerResult,
) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.DataFrame, list[str]]:
    return dict(result.return_series), result.return_long.copy(), result.return_panel.copy(), list(result.missing_return_factors)


def collect_candidate_panels(
    screeners: list[tuple[str, FactorScreener]],
    selection_summary: pd.DataFrame,
    *,
    score_field: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, float]]:
    candidate_panels: dict[str, pd.DataFrame] = {}
    candidate_rows: list[pd.DataFrame] = []
    candidate_scores: dict[str, float] = {}

    for label, screener in screeners:
        frame = selection_summary.loc[selection_summary["batch_label"] == label].copy()
        if frame.empty or "factor_name" not in frame.columns:
            continue
        frame = select_candidate_frame(frame)
        if frame.empty:
            continue

        factor_names = frame["factor_name"].astype(str).tolist()
        scores = score_series(frame, score_field=score_field)

        for factor_name in factor_names:
            factor_spec = next((spec for spec in screener.factor_specs if spec.table_name == factor_name), None)
            if factor_spec is None:
                continue
            raw_frame = screener.store.get_factor(
                factor_spec,
                engine="pandas",
            )
            panel = factor_frame_to_panel(raw_frame if isinstance(raw_frame, pd.DataFrame) else pd.DataFrame(), factor_name=factor_name)
            if panel.empty:
                continue
            composite_key = f"{label}::{factor_name}"
            candidate_panels[composite_key] = panel
            score = scores.get(factor_name)
            candidate_scores[composite_key] = float(score) if pd.notna(score) else 0.0

        if "batch_factor_key" in frame.columns:
            candidate_rows.append(
                frame.assign(batch_factor_key=frame["batch_factor_key"].astype(str))
            )
        else:
            candidate_rows.append(
                frame.assign(batch_factor_key=frame["factor_name"].astype(str).map(lambda name: f"{label}::{name}"))
            )

    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    return candidate_panels, candidates, candidate_scores


def collect_candidate_returns(
    screeners: list[tuple[str, FactorScreener]],
    results: list[tuple[str, FactorScreenerResult]],
    selection_summary: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    candidate_returns: dict[str, pd.Series] = {}
    factor_to_label: dict[str, str] = {}

    result_lookup = {label: result for label, result in results}
    for label, screener in screeners:
        frame = selection_summary.loc[selection_summary["batch_label"] == label].copy()
        if frame.empty or "factor_name" not in frame.columns:
            continue
        frame = select_candidate_frame(frame)
        if frame.empty:
            continue

        result = result_lookup.get(label)
        if result is None:
            continue

        primary_mode = primary_return_mode(result)
        extracted = extract_primary_return_series(result, primary_mode=primary_mode)
        for factor_name, series in extracted.items():
            batch_key = f"{label}::{factor_name}"
            candidate_returns[batch_key] = series
            factor_to_label[batch_key] = label

    return candidate_returns, factor_to_label


def greedy_select_by_return_gain(
    candidate_returns: dict[str, pd.Series],
    candidate_scores: dict[str, float],
    candidate_metrics: pd.DataFrame,
    *,
    config: FactorReturnGainSelectionConfig,
) -> list[str]:
    if not candidate_returns:
        return []

    ordered = sorted(
        candidate_returns.keys(),
        key=lambda key: (float(candidate_scores.get(key, 0.0)), key),
        reverse=True,
    )
    selected: list[str] = []
    current_objective = float("-inf")

    metric_frame = candidate_metrics.copy()
    if not metric_frame.empty and "batch_factor_key" in metric_frame.columns:
        metric_frame = metric_frame.set_index("batch_factor_key", drop=False)
    metric_frame = metric_frame.reindex(candidate_returns.keys())
    metric_scores = metric_objective(metric_frame, config)
    metric_scores.index = metric_frame.index

    for key in ordered:
        trial = selected + [key]
        frame = pd.DataFrame({name: candidate_returns[name] for name in trial}).sort_index()
        combined = frame.mean(axis=1, skipna=True).dropna()
        if combined.empty:
            continue
        metric_component = float(pd.to_numeric(metric_scores.reindex(trial), errors="coerce").dropna().mean()) if trial else 0.0
        return_component = return_objective(combined, config)
        corr_penalty = 0.0
        if len(trial) > 1:
            corr_frame = pd.DataFrame({name: candidate_returns[name] for name in trial}).corr().abs()
            pair_values: list[float] = []
            for idx, left in enumerate(trial):
                for right in trial[idx + 1 :]:
                    if left in corr_frame.index and right in corr_frame.columns:
                        value = corr_frame.loc[left, right]
                        if pd.notna(value):
                            pair_values.append(float(value))
            corr_penalty = float(np.mean(pair_values)) if pair_values else 0.0
        trial_objective = metric_component + return_component - float(config.corr_weight) * corr_penalty

        if not selected:
            if config.min_base_objective is not None and trial_objective < config.min_base_objective:
                continue
            if trial_objective >= config.min_improvement:
                selected.append(key)
                current_objective = trial_objective
            continue

        if trial_objective >= current_objective + config.min_improvement:
            selected.append(key)
            current_objective = trial_objective

    return selected


__all__ = [
    "FactorSelectionMode",
    "FactorReturnGainSelectionConfig",
    "resolve_selection_mode",
    "resolve_return_gain_config",
    "annualized_return_stats",
    "return_objective",
    "metric_objective",
    "item_label",
    "spec_signature",
    "tag_frame",
    "select_candidate_frame",
    "factor_frame_to_panel",
    "score_series",
    "primary_return_mode",
    "extract_primary_return_series",
    "combined_return_panel",
    "build_result_return_artifacts",
    "collect_candidate_panels",
    "collect_candidate_returns",
    "greedy_select_by_return_gain",
]
