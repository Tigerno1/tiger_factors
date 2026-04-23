from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from typing import Mapping
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSelectionConfig:
    cost_rate: float = 0.001
    periods_per_year: int = 252
    strong_ic: float = 0.02
    medium_ic: float = 0.01
    weak_ic: float = 0.005
    positive_sign_threshold: float = 0.2
    negative_sign_threshold: float = -0.2
    high_turnover: float = 0.4
    too_high_turnover: float = 0.6


@dataclass(frozen=True)
class FactorMetricFilterConfig:
    min_fitness: float | None = 0.10
    min_ic_mean: float | None = 0.01
    min_rank_ic_mean: float | None = 0.01
    min_sharpe: float | None = 0.40
    max_turnover: float | None = 0.50
    min_decay_score: float | None = 0.20
    min_capacity_score: float | None = 0.20
    max_correlation_penalty: float | None = 0.60
    min_regime_robustness: float | None = 0.60
    min_out_of_sample_stability: float | None = 0.60
    sort_field: str = "fitness"
    tie_breaker_field: str = "ic_ir"


@dataclass(frozen=True)
class FactorSummaryTableConfig:
    x_metric: str = "directional_ic_ir"
    y_metric: str = "directional_sharpe"
    score_fields: tuple[str, ...] = ("directional_fitness", "directional_ic_ir", "directional_sharpe")
    score_weights: tuple[float, ...] = (0.5, 0.25, 0.25)


@dataclass(frozen=True)
class FactorFilterConfig:
    min_abs_mean_ic: float = 0.005
    min_abs_ic_ir: float = 0.05
    min_net_sharpe: float = 0.30
    max_avg_turnover: float = 0.70
    max_max_drawdown: float = 0.50
    required_direction: str | None = None
    strong_ic: float = 0.02
    medium_ic: float = 0.01
    weak_ic: float = 0.005
    weight_ic: float = 4.0
    weight_ic_ir: float = 2.0
    weight_net_sharpe: float = 3.0
    weight_turnover_penalty: float = 2.0
    weight_drawdown_penalty: float = 1.0


def _resolve_config(config: FactorSelectionConfig | None) -> FactorSelectionConfig:
    return config if config is not None else FactorSelectionConfig()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.notna() & (denominator > 0)
    out.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return out


def _field_higher_is_better(field_name: str) -> bool:
    normalized = str(field_name).strip().lower()
    lower_better_markers = ("turnover", "penalty", "drawdown", "cost", "volatility", "risk")
    return not any(marker in normalized for marker in lower_better_markers)


def _normalize_rank_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    if valid.empty:
        return out
    if len(valid) == 1:
        out.loc[valid.index[0]] = 1.0
        return out

    ranks = valid.rank(method="average", ascending=higher_is_better)
    scaled = (ranks - 1.0) / (len(valid) - 1.0)
    out.loc[valid.index] = scaled
    return out


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def _record_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        if len(value) != 1:
            raise ValueError("DataFrame values must contain exactly one row.")
        return value.iloc[0].to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    raise TypeError(f"Unsupported metric record type: {type(value)!r}")


def factor_metric_frame(results: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        frame = results.copy()
        if "factor_name" not in frame.columns:
            index_name = frame.index.name
            frame = frame.reset_index()
            if index_name and index_name in frame.columns:
                frame = frame.rename(columns={index_name: "factor_name"})
            elif "index" in frame.columns:
                frame = frame.rename(columns={"index": "factor_name"})
            else:
                frame.insert(0, "factor_name", frame.index.astype(str))
        return frame

    rows: list[dict[str, Any]] = []
    for factor_name, value in results.items():
        row = _record_from_value(value)
        row.setdefault("factor_name", factor_name)
        rows.append(row)
    return pd.DataFrame(rows)


def screen_factor_metrics(
    results: Mapping[str, Any] | pd.DataFrame,
    *,
    config: FactorMetricFilterConfig | None = None,
) -> pd.DataFrame:
    cfg = config if config is not None else FactorMetricFilterConfig()
    frame = factor_metric_frame(results)
    if frame.empty:
        return frame

    required = {"factor_name", "ic_mean", "rank_ic_mean", "sharpe", "turnover", "fitness"}
    _require_columns(frame, required)

    filtered = frame.copy()
    filtered["ic_mean"] = pd.to_numeric(filtered["ic_mean"], errors="coerce")
    filtered["rank_ic_mean"] = pd.to_numeric(filtered["rank_ic_mean"], errors="coerce")
    filtered["fitness"] = pd.to_numeric(filtered["fitness"], errors="coerce")
    filtered["sharpe"] = pd.to_numeric(filtered["sharpe"], errors="coerce")
    filtered["ic_ir"] = pd.to_numeric(filtered.get("ic_ir"), errors="coerce") if "ic_ir" in filtered.columns else np.nan
    filtered["ic_mean_abs"] = filtered["ic_mean"].abs()
    filtered["rank_ic_mean_abs"] = filtered["rank_ic_mean"].abs()
    filtered["direction_hint"] = np.where(filtered["ic_mean"] < 0, "reverse_factor", "use_as_is")
    filtered["directional_fitness"] = filtered["fitness"].abs()
    filtered["directional_ic_ir"] = filtered["ic_ir"].abs() if "ic_ir" in filtered.columns else np.nan
    filtered["directional_sharpe"] = filtered["sharpe"].abs()

    usable_flags: list[bool] = []
    failed_rules: list[list[str]] = []
    for _, row in filtered.iterrows():
        row_failed: list[str] = []
        fitness = float(row.get("directional_fitness", np.nan))
        ic_mean_abs = float(row.get("ic_mean_abs", np.nan))
        rank_ic_mean_abs = float(row.get("rank_ic_mean_abs", np.nan))
        sharpe = float(row.get("directional_sharpe", np.nan))
        turnover = float(row.get("turnover", np.nan))
        decay_score = float(row.get("decay_score", np.nan))
        capacity_score = float(row.get("capacity_score", np.nan))
        correlation_penalty = float(row.get("correlation_penalty", np.nan))
        regime_robustness = float(row.get("regime_robustness", np.nan))
        out_of_sample_stability = float(row.get("out_of_sample_stability", np.nan))

        if cfg.min_fitness is not None and pd.notna(fitness) and fitness < cfg.min_fitness:
            row_failed.append(f"abs_fitness<{cfg.min_fitness}")
        if cfg.min_ic_mean is not None and pd.notna(ic_mean_abs) and ic_mean_abs < cfg.min_ic_mean:
            row_failed.append(f"abs_ic_mean<{cfg.min_ic_mean}")
        if cfg.min_rank_ic_mean is not None and pd.notna(rank_ic_mean_abs) and rank_ic_mean_abs < cfg.min_rank_ic_mean:
            row_failed.append(f"abs_rank_ic_mean<{cfg.min_rank_ic_mean}")
        if cfg.min_sharpe is not None and pd.notna(sharpe) and sharpe < cfg.min_sharpe:
            row_failed.append(f"abs_sharpe<{cfg.min_sharpe}")
        if cfg.max_turnover is not None and pd.notna(turnover) and turnover > cfg.max_turnover:
            row_failed.append(f"turnover>{cfg.max_turnover}")
        if cfg.min_decay_score is not None and pd.notna(decay_score) and decay_score < cfg.min_decay_score:
            row_failed.append(f"decay_score<{cfg.min_decay_score}")
        if cfg.min_capacity_score is not None and pd.notna(capacity_score) and capacity_score < cfg.min_capacity_score:
            row_failed.append(f"capacity_score<{cfg.min_capacity_score}")
        if cfg.max_correlation_penalty is not None and pd.notna(correlation_penalty) and correlation_penalty > cfg.max_correlation_penalty:
            row_failed.append(f"correlation_penalty>{cfg.max_correlation_penalty}")
        if cfg.min_regime_robustness is not None and pd.notna(regime_robustness) and regime_robustness < cfg.min_regime_robustness:
            row_failed.append(f"regime_robustness<{cfg.min_regime_robustness}")
        if cfg.min_out_of_sample_stability is not None and pd.notna(out_of_sample_stability) and out_of_sample_stability < cfg.min_out_of_sample_stability:
            row_failed.append(f"out_of_sample_stability<{cfg.min_out_of_sample_stability}")

        usable_flags.append(not row_failed)
        failed_rules.append(row_failed)

    def _sort_metric(frame: pd.DataFrame, field: str) -> pd.Series:
        if field == "fitness":
            return pd.to_numeric(frame["directional_fitness"], errors="coerce")
        if field == "ic_ir":
            return pd.to_numeric(frame["directional_ic_ir"], errors="coerce")
        if field == "sharpe":
            return pd.to_numeric(frame["directional_sharpe"], errors="coerce")
        if field in {"ic_mean", "rank_ic_mean"}:
            return pd.to_numeric(frame[field], errors="coerce").abs()
        return pd.to_numeric(frame.get(field), errors="coerce")

    filtered["usable"] = usable_flags
    filtered["failed_rules"] = failed_rules

    return (
        filtered.assign(
            _usable_rank=filtered["usable"].fillna(False).astype(int),
            _sort_field=_sort_metric(filtered, cfg.sort_field),
            _tie_breaker=_sort_metric(filtered, cfg.tie_breaker_field),
        )
        .sort_values(
            by=["_usable_rank", "_sort_field", "_tie_breaker", "factor_name"],
            ascending=[False, False, False, True],
            na_position="last",
        )
        .drop(columns=["_usable_rank", "_sort_field", "_tie_breaker"])
        .reset_index(drop=True)
    )


def build_factor_summary_table(
    frame: pd.DataFrame,
    *,
    config: FactorSummaryTableConfig | None = None,
) -> pd.DataFrame:
    cfg = config if config is not None else FactorSummaryTableConfig()
    if frame.empty:
        return frame.copy()

    required = {"factor_name", cfg.x_metric, cfg.y_metric}
    _require_columns(frame, required)
    if len(cfg.score_fields) != len(cfg.score_weights):
        raise ValueError("score_fields and score_weights must have the same length.")

    out = frame.copy().reset_index(drop=True)
    x_values = pd.to_numeric(out[cfg.x_metric], errors="coerce")
    y_values = pd.to_numeric(out[cfg.y_metric], errors="coerce")
    out["x_metric"] = cfg.x_metric
    out["y_metric"] = cfg.y_metric
    out["x_value"] = x_values
    out["y_value"] = y_values
    out["x_score"] = _normalize_rank_score(x_values, higher_is_better=_field_higher_is_better(cfg.x_metric))
    out["y_score"] = _normalize_rank_score(y_values, higher_is_better=_field_higher_is_better(cfg.y_metric))

    metric_score_map: dict[str, pd.Series] = {}
    combined = pd.Series(0.0, index=out.index, dtype=float)
    combined_weight = pd.Series(0.0, index=out.index, dtype=float)
    for field_name, weight in zip(cfg.score_fields, cfg.score_weights, strict=False):
        if field_name not in out.columns:
            continue
        weight_value = float(weight)
        if weight_value == 0:
            continue
        component = _normalize_rank_score(
            out[field_name],
            higher_is_better=_field_higher_is_better(field_name),
        )
        metric_score_map[field_name] = component
        out[f"{field_name}_score"] = component
        valid = component.notna()
        combined.loc[valid] = combined.loc[valid] + component.loc[valid] * weight_value
        combined_weight.loc[valid] = combined_weight.loc[valid] + weight_value

    out["combined_score"] = combined.where(combined_weight > 0, np.nan) / combined_weight.replace(0.0, np.nan)
    out["combined_score_norm"] = _normalize_rank_score(out["combined_score"], higher_is_better=True)

    ordered_metrics: list[tuple[str, str]] = [
        (cfg.x_metric, "x"),
        (cfg.y_metric, "y"),
    ]
    for field_name in cfg.score_fields:
        if field_name not in {cfg.x_metric, cfg.y_metric}:
            ordered_metrics.append((field_name, "score_component"))

    records: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        factor_name = row["factor_name"]
        usable = bool(row["usable"]) if "usable" in out.columns and pd.notna(row.get("usable")) else None
        for metric_index, (metric_name, metric_role) in enumerate(ordered_metrics):
            if metric_name not in out.columns:
                continue
            metric_value = row.get(metric_name)
            if metric_name == cfg.x_metric:
                metric_score = row.get("x_score")
            elif metric_name == cfg.y_metric:
                metric_score = row.get("y_score")
            else:
                metric_score = row.get(f"{metric_name}_score", np.nan)
            records.append(
                {
                    "factor_name": factor_name,
                    "metric_name": metric_name,
                    "metric_role": metric_role,
                    "metric_order": metric_index,
                    "metric_value": metric_value,
                    "metric_score": metric_score,
                    "combined_score": row.get("combined_score"),
                    "combined_score_norm": row.get("combined_score_norm"),
                    "x_metric": row.get("x_metric"),
                    "y_metric": row.get("y_metric"),
                    "usable": usable,
                }
            )

    long_frame = pd.DataFrame(records)
    if long_frame.empty:
        return long_frame

    sort_columns = ["usable", "combined_score", "factor_name", "metric_order", "metric_name"]
    sort_ascending = [False, False, True, True, True]
    if "usable" not in long_frame.columns:
        sort_columns = ["combined_score", "factor_name", "metric_order", "metric_name"]
        sort_ascending = [False, True, True, True]

    return (
        long_frame.sort_values(
            by=sort_columns,
            ascending=sort_ascending,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def add_cost_analysis(
    result: pd.DataFrame,
    *,
    cost_rate: float | None = None,
    config: FactorSelectionConfig | None = None,
) -> pd.DataFrame:
    cfg = _resolve_config(config)
    effective_cost_rate = float(cfg.cost_rate if cost_rate is None else cost_rate)
    out = result.copy()
    _require_columns(out, {"ann_return", "ann_vol", "avg_turnover"})

    out["cost_penalty"] = out["avg_turnover"] * effective_cost_rate * float(cfg.periods_per_year)
    out["net_ann_return"] = out["ann_return"] - out["cost_penalty"]
    out["net_sharpe"] = _safe_divide(out["net_ann_return"], out["ann_vol"])
    return out


def evaluate_factor(
    result: pd.DataFrame,
    *,
    config: FactorSelectionConfig | None = None,
) -> dict[str, Any]:
    cfg = _resolve_config(config)
    frame = result.copy()
    _require_columns(frame, {"horizon", "mean_ic", "avg_turnover", "sharpe"})

    valid_ic = frame["mean_ic"].dropna()
    sharpe_col = "net_sharpe" if "net_sharpe" in frame.columns else "sharpe"
    valid_sharpe = frame[sharpe_col].abs().dropna()

    if valid_ic.empty:
        return {
            "usable": False,
            "reason": "No valid IC observations",
            "direction": "unknown",
            "quality": "unknown",
            "tradeability": "unknown",
            "sign_score": float("nan"),
            "ic_strength": float("nan"),
            "best_ic_horizon": None,
            "best_sharpe_horizon": None,
            "selected_sharpe_column": sharpe_col,
            "recommendation": "DO NOT USE",
        }

    sign_score = float(np.sign(valid_ic).mean())
    if sign_score < cfg.negative_sign_threshold:
        direction = "reverse_factor"
    elif sign_score > cfg.positive_sign_threshold:
        direction = "use_as_is"
    else:
        direction = "unstable"

    ic_strength = float(valid_ic.abs().mean())
    if ic_strength > cfg.strong_ic:
        quality = "strong"
    elif ic_strength > cfg.medium_ic:
        quality = "medium"
    elif ic_strength > cfg.weak_ic:
        quality = "weak_but_usable"
    else:
        quality = "too_weak"

    best_ic_idx = frame["mean_ic"].abs().idxmax()
    best_ic_horizon = int(frame.loc[best_ic_idx, "horizon"]) if pd.notna(best_ic_idx) else None

    if valid_sharpe.empty:
        best_sharpe_horizon = None
    else:
        best_sharpe_horizon = int(frame.loc[frame[sharpe_col].abs().idxmax(), "horizon"])

    avg_turnover = float(frame["avg_turnover"].mean()) if "avg_turnover" in frame else float("nan")
    if pd.isna(avg_turnover):
        tradeability = "unknown"
    elif avg_turnover > cfg.too_high_turnover:
        tradeability = "too_high_turnover"
    elif avg_turnover > cfg.high_turnover:
        tradeability = "high_turnover"
    else:
        tradeability = "ok"

    usable = (quality != "too_weak") and (direction != "unstable")

    if not usable:
        recommendation = "DO NOT USE"
    elif best_sharpe_horizon is not None:
        recommendation = f"{direction}, use {best_sharpe_horizon}D holding"
    elif best_ic_horizon is not None:
        recommendation = f"{direction}, use {best_ic_horizon}D holding"
    else:
        recommendation = direction

    return {
        "usable": usable,
        "reason": None,
        "direction": direction,
        "quality": quality,
        "tradeability": tradeability,
        "sign_score": sign_score,
        "ic_strength": ic_strength,
        "best_ic_horizon": best_ic_horizon,
        "best_sharpe_horizon": best_sharpe_horizon,
        "selected_sharpe_column": sharpe_col,
        "recommendation": recommendation,
    }


def screen_factor_results(
    results_by_factor: dict[str, pd.DataFrame],
    *,
    config: FactorSelectionConfig | None = None,
    apply_costs: bool = True,
) -> pd.DataFrame:
    cfg = _resolve_config(config)
    rows: list[dict[str, Any]] = []

    for factor_name, raw_result in results_by_factor.items():
        enriched = add_cost_analysis(raw_result, config=cfg) if apply_costs else raw_result.copy()
        evaluation = evaluate_factor(enriched, config=cfg)

        summary: dict[str, Any] = {"factor_name": factor_name, **evaluation}
        selected_horizon = evaluation["best_sharpe_horizon"] or evaluation["best_ic_horizon"]
        summary["selected_horizon"] = selected_horizon

        if selected_horizon is not None and "horizon" in enriched.columns:
            selected = enriched.loc[enriched["horizon"] == selected_horizon]
            if not selected.empty:
                selected_row = selected.iloc[0]
                summary["selected_mean_ic"] = selected_row.get("mean_ic")
                summary["selected_sharpe"] = selected_row.get("sharpe")
                summary["selected_net_sharpe"] = selected_row.get("net_sharpe")
                summary["selected_ann_return"] = selected_row.get("ann_return")
                summary["selected_net_ann_return"] = selected_row.get("net_ann_return")
                summary["selected_avg_turnover"] = selected_row.get("avg_turnover")

        rows.append(summary)

    summary_frame = pd.DataFrame(rows)
    if summary_frame.empty:
        return summary_frame

    usable_rank = summary_frame["usable"].fillna(False).astype(int)
    net_sharpe_rank = pd.to_numeric(summary_frame.get("selected_net_sharpe"), errors="coerce").abs()
    sharpe_rank = pd.to_numeric(summary_frame.get("selected_sharpe"), errors="coerce").abs()
    ic_rank = pd.to_numeric(summary_frame.get("selected_mean_ic"), errors="coerce").abs()

    return summary_frame.assign(
        _usable_rank=usable_rank,
        _net_sharpe_rank=net_sharpe_rank,
        _sharpe_rank=sharpe_rank,
        _ic_rank=ic_rank,
    ).sort_values(
        by=["_usable_rank", "_net_sharpe_rank", "_sharpe_rank", "_ic_rank", "factor_name"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).drop(columns=["_usable_rank", "_net_sharpe_rank", "_sharpe_rank", "_ic_rank"]).reset_index(drop=True)


def classify_quality(abs_mean_ic: float, config: FactorFilterConfig | None = None) -> str:
    cfg = config if config is not None else FactorFilterConfig()
    if abs_mean_ic >= cfg.strong_ic:
        return "strong"
    if abs_mean_ic >= cfg.medium_ic:
        return "medium"
    if abs_mean_ic >= cfg.weak_ic:
        return "weak_but_usable"
    return "too_weak"


def compute_factor_score(best_row: pd.Series, config: FactorFilterConfig | None = None) -> float:
    cfg = config if config is not None else FactorFilterConfig()
    mean_ic = abs(float(best_row.get("mean_ic", np.nan)))
    ic_ir = abs(float(best_row.get("ic_ir", np.nan)))
    net_sharpe = abs(float(best_row.get("net_sharpe", best_row.get("sharpe", np.nan))))
    turnover = float(best_row.get("avg_turnover", np.nan))
    max_drawdown = abs(float(best_row.get("max_drawdown", np.nan)))

    score = 0.0
    if pd.notna(mean_ic):
        score += cfg.weight_ic * mean_ic * 100.0
    if pd.notna(ic_ir):
        score += cfg.weight_ic_ir * ic_ir
    if pd.notna(net_sharpe):
        score += cfg.weight_net_sharpe * net_sharpe
    if pd.notna(turnover):
        score -= cfg.weight_turnover_penalty * turnover
    if pd.notna(max_drawdown):
        score -= cfg.weight_drawdown_penalty * max_drawdown
    return float(score)


def evaluate_factor_with_filter(
    result: pd.DataFrame,
    config: FactorFilterConfig | None = None,
) -> dict[str, Any]:
    cfg = config if config is not None else FactorFilterConfig()
    if result.empty:
        return {"usable": False, "reason": "empty_result"}

    _require_columns(result, {"horizon", "mean_ic", "ic_ir", "avg_turnover", "max_drawdown"})

    valid_ic = result["mean_ic"].dropna()
    if valid_ic.empty:
        return {"usable": False, "reason": "no_valid_ic"}

    sign_score = float(np.sign(valid_ic).mean())
    if sign_score < -0.2:
        direction = "negative"
        recommendation_direction = "reverse_factor"
    elif sign_score > 0.2:
        direction = "positive"
        recommendation_direction = "use_as_is"
    else:
        direction = "unstable"
        recommendation_direction = "unstable"

    best_col = "net_sharpe" if "net_sharpe" in result.columns else "sharpe"
    if best_col not in result.columns:
        raise KeyError(f"Missing required columns: {best_col}")
    valid_best = result[best_col].abs().dropna()
    if valid_best.empty:
        return {"usable": False, "reason": "no_valid_sharpe"}

    best_idx = valid_best.idxmax()
    best_row = result.loc[best_idx]
    best_horizon = int(best_row["horizon"])
    mean_ic = float(best_row["mean_ic"])
    abs_mean_ic = abs(mean_ic)
    ic_ir = abs(float(best_row["ic_ir"]))
    best_sharpe = abs(float(best_row[best_col]))
    avg_turnover = float(best_row["avg_turnover"])
    max_drawdown = abs(float(best_row["max_drawdown"]))
    quality = classify_quality(abs_mean_ic, cfg)

    failed_rules: list[str] = []
    if abs_mean_ic < cfg.min_abs_mean_ic:
        failed_rules.append(f"abs_mean_ic<{cfg.min_abs_mean_ic}")
    if ic_ir < cfg.min_abs_ic_ir:
        failed_rules.append(f"abs_ic_ir<{cfg.min_abs_ic_ir}")
    if best_sharpe < cfg.min_net_sharpe:
        failed_rules.append(f"abs_net_sharpe<{cfg.min_net_sharpe}")
    if avg_turnover > cfg.max_avg_turnover:
        failed_rules.append(f"avg_turnover>{cfg.max_avg_turnover}")
    if max_drawdown > cfg.max_max_drawdown:
        failed_rules.append(f"max_drawdown>{cfg.max_max_drawdown}")
    if cfg.required_direction is not None and direction != cfg.required_direction:
        failed_rules.append(f"direction!={cfg.required_direction}")
    if direction == "unstable":
        failed_rules.append("direction_unstable")

    usable = len(failed_rules) == 0
    recommendation = (
        f"{recommendation_direction}, use {best_horizon}D holding" if usable else "REJECT"
    )

    return {
        "usable": usable,
        "reason": None if usable else "filtered",
        "failed_rules": failed_rules,
        "direction": direction,
        "recommendation_direction": recommendation_direction,
        "quality": quality,
        "best_horizon": best_horizon,
        "best_mean_ic": mean_ic,
        "best_abs_mean_ic": abs_mean_ic,
        "best_ic_ir": ic_ir,
        "best_net_sharpe": best_sharpe,
        "best_avg_turnover": avg_turnover,
        "best_max_drawdown": max_drawdown,
        "score": compute_factor_score(best_row, cfg),
        "recommendation": recommendation,
    }


__all__ = [
    "FactorFilterConfig",
    "FactorMetricFilterConfig",
    "FactorSelectionConfig",
    "FactorSummaryTableConfig",
    "add_cost_analysis",
    "build_factor_summary_table",
    "classify_quality",
    "factor_metric_frame",
    "compute_factor_score",
    "evaluate_factor",
    "evaluate_factor_with_filter",
    "screen_factor_metrics",
    "screen_factor_results",
]
