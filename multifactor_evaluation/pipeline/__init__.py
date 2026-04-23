from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation import create_native_full_tear_sheet
from tiger_factors.factor_evaluation import evaluate_factor_panel
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.screening import FactorSummaryTableConfig
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics
from tiger_factors.multifactor_evaluation.reporting.summary_table import build_factor_summary_table
from tiger_factors.multifactor_evaluation.selection import greedy_select_by_correlation


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


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    total = float(sum(weights.values()))
    if total == 0:
        equal = 1.0 / len(weights)
        return {key: equal for key in weights}
    return {key: float(value) / total for key, value in weights.items()}


def score_to_weights(
    scores: dict[str, float],
    *,
    selected: list[str] | None = None,
    method: str = "positive",
    temperature: float = 1.0,
) -> dict[str, float]:
    names = selected or list(scores.keys())
    if not names:
        return {}

    values = np.array([float(scores.get(name, 0.0)) for name in names], dtype=float)

    if method == "equal":
        raw = {name: 1.0 for name in names}
        return normalize_weights(raw)

    if method == "softmax":
        temp = max(float(temperature), 1e-6)
        logits = values / temp
        logits = logits - np.nanmax(logits)
        expv = np.exp(logits)
        raw = {name: float(value) for name, value in zip(names, expv)}
        return normalize_weights(raw)

    if method == "positive":
        shifted = values - np.nanmin(values)
        shifted = shifted + 1e-12
        raw = {name: float(value) for name, value in zip(names, shifted)}
        return normalize_weights(raw)

    raise ValueError("method must be one of: 'equal', 'positive', 'softmax'.")


def apply_weight_bounds(
    weights: dict[str, float],
    *,
    min_weight: float | None = None,
    max_weight: float | None = None,
    total: float = 1.0,
    max_iter: int = 128,
    tol: float = 1e-9,
) -> dict[str, float]:
    if not weights:
        return {}

    names = list(weights.keys())
    base = np.array([float(weights[name]) for name in names], dtype=float)
    base = np.where(np.isfinite(base), np.maximum(base, 0.0), 0.0)

    target_total = float(total)
    if target_total <= 0:
        raise ValueError("total must be positive.")

    if base.sum() <= tol:
        base[:] = 1.0 / len(base)
    else:
        base = base / base.sum()

    current = base * target_total
    min_w = 0.0 if min_weight is None else float(min_weight)
    max_w = None if max_weight is None else float(max_weight)

    if min_w < 0:
        raise ValueError("min_weight must be non-negative.")
    if max_w is not None and max_w <= 0:
        raise ValueError("max_weight must be positive.")
    if max_w is not None and max_w + tol < min_w:
        raise ValueError("max_weight must be >= min_weight.")
    if min_w * len(names) > target_total + tol:
        raise ValueError("min_weight is infeasible for the number of selected weights.")
    if max_w is not None and max_w * len(names) < target_total - tol:
        raise ValueError("max_weight is infeasible for the number of selected weights.")

    locked = np.zeros(len(names), dtype=bool)
    for _ in range(max_iter):
        changed = False

        if min_weight is not None:
            low_mask = (~locked) & (current < min_w - tol)
            if low_mask.any():
                current[low_mask] = min_w
                locked[low_mask] = True
                changed = True

        if max_w is not None:
            high_mask = (~locked) & (current > max_w + tol)
            if high_mask.any():
                current[high_mask] = max_w
                locked[high_mask] = True
                changed = True

        free = ~locked
        if not free.any():
            break

        residual = target_total - float(current[locked].sum())
        if residual < -tol:
            raise ValueError("Weight bounds are infeasible under the requested total.")

        free_base = base[free]
        if free_base.sum() <= tol:
            current[free] = residual / float(free.sum())
        else:
            current[free] = residual * free_base / free_base.sum()

        if not changed and abs(float(current.sum()) - target_total) <= tol:
            break

    return {name: float(value) for name, value in zip(names, current)}


def summarize_factor_evaluation(panel: pd.DataFrame, forward_returns: pd.DataFrame) -> dict[str, float]:
    evaluation = evaluate_factor_panel(panel, forward_returns)
    payload = evaluation.__dict__
    return {key: float(value) for key, value in payload.items()}


def score_factor_panels(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    *,
    score_field: str = "fitness",
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}

    for name, panel in factors.items():
        summary = summarize_factor_evaluation(panel, forward_returns)
        summaries[name] = summary
        scores[name] = float(summary.get(score_field, 0.0))

    return scores, summaries


@dataclass(frozen=True)
class FactorPipelineConfig:
    forward_days: int = 21
    top_n_initial: int = 20
    corr_threshold: float = 0.75
    score_field: str = "fitness"
    selection_score_field: str = "ic_ir"
    min_ic_mean: float | None = None
    min_rank_ic_mean: float | None = None
    min_sharpe: float | None = None
    weight_method: str = "positive"
    weight_temperature: float = 1.0
    min_factor_weight: float | None = None
    max_factor_weight: float | None = None
    summary_x_metric: str = "directional_ic_ir"
    summary_y_metric: str = "directional_sharpe"
    summary_score_fields: tuple[str, ...] = ("directional_fitness", "directional_ic_ir", "directional_sharpe")
    summary_score_weights: tuple[float, ...] = (0.5, 0.25, 0.25)
    standardize: bool = True
    long_pct: float = 0.2
    long_short: bool = True
    rebalance_freq: str = "ME"
    annual_trading_days: int = 252
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    persist_outputs: bool = False


@dataclass(frozen=True)
class FactorPipelineResult:
    summary: pd.DataFrame
    correlation_matrix: pd.DataFrame
    selected_factors: list[str]
    selected_scores: dict[str, float]
    factor_weights: dict[str, float]
    combined_factor: pd.DataFrame
    forward_returns: pd.DataFrame
    backtest_returns: pd.DataFrame
    backtest_stats: dict[str, dict[str, float]]
    report_dir: Path | None = None
    score_table: pd.DataFrame | None = None


def _common_factor_universe(
    factors: Mapping[str, pd.DataFrame],
    close_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not factors:
        raise ValueError("factors must not be empty.")

    close = close_panel.sort_index().copy()
    common_columns = close.columns
    for panel in factors.values():
        common_columns = common_columns.intersection(panel.columns)

    if close.index.empty or common_columns.empty:
        raise ValueError("No overlapping dates or assets across factor panels and close panel.")

    close = close.loc[:, common_columns].sort_index().ffill()
    aligned: dict[str, pd.DataFrame] = {}
    for name, panel in factors.items():
        aligned[name] = panel.reindex(index=close.index, columns=common_columns)
    return close, aligned


def _metric_filter_config(config: FactorPipelineConfig) -> FactorMetricFilterConfig:
    return FactorMetricFilterConfig(
        min_ic_mean=config.min_ic_mean,
        min_rank_ic_mean=config.min_rank_ic_mean,
        min_sharpe=config.min_sharpe,
        max_turnover=None,
        sort_field=config.score_field,
        tie_breaker_field=config.selection_score_field,
    )


def _summary_table_config(config: FactorPipelineConfig) -> FactorSummaryTableConfig:
    return FactorSummaryTableConfig(
        x_metric=config.summary_x_metric,
        y_metric=config.summary_y_metric,
        score_fields=config.summary_score_fields,
        score_weights=config.summary_score_weights,
    )


def _annualized_return_stats(returns: pd.Series, annual_trading_days: int = 252) -> dict[str, float]:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }
    cumulative = (1.0 + series).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (annual_trading_days / max(len(series), 1)) - 1.0)
    ann_vol = float(series.std(ddof=0) * np.sqrt(annual_trading_days))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = float(ann_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    win_rate = float((series > 0).mean())
    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_return": total_return,
    }


def run_factor_backtest(
    factor_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    long_pct: float = 0.2,
    rebalance_freq: str = "ME",
    long_short: bool = True,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    close = close_panel.ffill().sort_index()
    daily_returns = close.pct_change(fill_method=None)

    factor_eom = factor_panel.sort_index().resample(rebalance_freq).last()
    if start is not None:
        factor_eom = factor_eom.loc[factor_eom.index >= pd.Timestamp(start)]
    if end is not None:
        factor_eom = factor_eom.loc[factor_eom.index <= pd.Timestamp(end)]
    rebalance_dates = factor_eom.index

    records: list[dict[str, float | int | pd.Timestamp]] = []
    position_records: list[dict[str, float | pd.Timestamp]] = []
    previous_weights: dict[str, float] = {}
    friction_bps = max(float(transaction_cost_bps), 0.0) + max(float(slippage_bps), 0.0)
    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance = rebalance_dates[i + 1]
        signal = factor_eom.loc[rebalance_date].dropna()
        if len(signal) < 2:
            continue
        n_long = max(1, int(len(signal) * long_pct))
        n_short = max(1, n_long)
        long_codes = signal.nlargest(n_long).index.tolist()
        short_codes = signal.nsmallest(n_short).index.tolist()

        long_weights = apply_weight_bounds({code: 1.0 for code in long_codes}, total=0.5)
        short_weights = apply_weight_bounds({code: 1.0 for code in short_codes}, total=0.5)
        target_weights = dict(long_weights)
        if long_short:
            target_weights.update({code: -weight for code, weight in short_weights.items()})

        all_codes = sorted(set(previous_weights) | set(target_weights))
        turnover = 0.5 * sum(abs(float(target_weights.get(code, 0.0)) - float(previous_weights.get(code, 0.0))) for code in all_codes)
        rebalance_cost = turnover * friction_bps / 10000.0
        previous_weights = target_weights

        holding_mask = (daily_returns.index > rebalance_date) & (daily_returns.index <= next_rebalance)
        holding_slice = daily_returns.loc[holding_mask]
        first_day = True
        for date, row in holding_slice.iterrows():
            long_return = float(sum(float(row.get(code, 0.0)) * weight for code, weight in long_weights.items() if pd.notna(row.get(code, np.nan))))
            short_return = float(sum(float(row.get(code, 0.0)) * weight for code, weight in short_weights.items() if pd.notna(row.get(code, np.nan))))
            portfolio_return = float(sum(float(row.get(code, 0.0)) * weight for code, weight in target_weights.items() if pd.notna(row.get(code, np.nan))))
            cost = rebalance_cost if first_day else 0.0
            portfolio_return -= cost
            first_day = False
            benchmark_return = float(row.mean(skipna=True))
            records.append(
                {
                    "date": date,
                    "portfolio": portfolio_return,
                    "benchmark": benchmark_return,
                    "long_return": long_return,
                    "short_return": short_return,
                    "n_long": n_long,
                    "n_short": n_short,
                    "turnover": turnover if cost > 0 else 0.0,
                    "cost": cost,
                }
            )
            position_record: dict[str, float | pd.Timestamp] = {"date": date}
            position_record.update({code: float(weight) for code, weight in target_weights.items()})
            position_record["cash"] = float(1.0 - sum(target_weights.values()))
            position_records.append(position_record)

    if not records:
        backtest = pd.DataFrame(columns=["portfolio", "benchmark", "long_return", "short_return", "n_long", "n_short", "turnover", "cost"])
        backtest.attrs["positions"] = pd.DataFrame()
        backtest.attrs["close_panel"] = close
        return backtest, {
            "portfolio": _annualized_return_stats(pd.Series(dtype=float), annual_trading_days),
            "benchmark": _annualized_return_stats(pd.Series(dtype=float), annual_trading_days),
        }

    backtest = pd.DataFrame(records).set_index("date").sort_index()
    backtest.index = pd.DatetimeIndex(backtest.index)
    positions_frame = pd.DataFrame.from_records(position_records)
    if not positions_frame.empty:
        positions_frame["date"] = pd.to_datetime(positions_frame["date"], errors="coerce")
        positions_frame = positions_frame.dropna(subset=["date"]).set_index("date").sort_index()
        positions_frame.index = pd.DatetimeIndex(positions_frame.index, name="date_")
        positions_frame = positions_frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positions_frame.columns = positions_frame.columns.astype(str)
    backtest.attrs["positions"] = positions_frame
    backtest.attrs["close_panel"] = close
    stats = {
        "portfolio": _annualized_return_stats(backtest["portfolio"], annual_trading_days),
        "benchmark": _annualized_return_stats(backtest["benchmark"], annual_trading_days),
    }
    return backtest, stats


def screen_factor_panels(
    factor_panels: Mapping[str, pd.DataFrame],
    close_panel: pd.DataFrame,
    *,
    config: FactorPipelineConfig | None = None,
    output_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    report_factor_name: str = "combined_factor",
) -> FactorPipelineResult:
    if not factor_panels:
        raise ValueError("factor_panels must not be empty.")

    active = config or FactorPipelineConfig()
    close, aligned_factors = _common_factor_universe(factor_panels, close_panel)
    forward_returns = close.pct_change(active.forward_days, fill_method=None).shift(-active.forward_days)

    _, summaries = score_factor_panels(aligned_factors, forward_returns, score_field=active.score_field)
    summary = pd.DataFrame(summaries).T
    summary.index.name = "factor"
    if summary.empty:
        raise ValueError("No factor could be evaluated.")

    summary = screen_factor_metrics(summary, config=_metric_filter_config(active))
    if summary.empty:
        summary = pd.DataFrame(summaries).T.sort_values(active.score_field, ascending=False)
    elif "factor_name" in summary.columns:
        summary = summary.set_index("factor_name", drop=False)

    score_table = build_factor_summary_table(summary, config=_summary_table_config(active))

    top_candidates = [str(name) for name in summary["factor_name"].head(active.top_n_initial).tolist() if name in aligned_factors]
    candidate_panels = {name: aligned_factors[name] for name in top_candidates}
    if not candidate_panels:
        raise ValueError("No candidate factors survived the initial screening.")

    scores = {name: float(summary.loc[name, active.selection_score_field]) for name in top_candidates if name in summary.index}
    correlation_matrix = factor_correlation_matrix(candidate_panels, standardize=active.standardize)
    selected_factors = greedy_select_by_correlation(scores, correlation_matrix, active.corr_threshold)
    if not selected_factors:
        selected_factors = top_candidates[:1]

    selected_scores = {name: float(summary.loc[name, active.selection_score_field]) for name in selected_factors}
    factor_weights = score_to_weights(
        selected_scores,
        selected=selected_factors,
        method=active.weight_method,
        temperature=active.weight_temperature,
    )
    factor_weights = apply_weight_bounds(
        factor_weights,
        min_weight=active.min_factor_weight,
        max_weight=active.max_factor_weight,
        total=1.0,
    )
    selected_panels = {name: aligned_factors[name] for name in selected_factors}
    combined_factor = blend_factor_panels(selected_panels, factor_weights, standardize=active.standardize)
    backtest, backtest_stats = run_factor_backtest(
        combined_factor,
        close,
        long_pct=active.long_pct,
        rebalance_freq=active.rebalance_freq,
        long_short=active.long_short,
        annual_trading_days=active.annual_trading_days,
        transaction_cost_bps=active.transaction_cost_bps,
        slippage_bps=active.slippage_bps,
    )

    report_path: Path | None = None
    if active.persist_outputs and report_dir is not None:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        create_native_full_tear_sheet(
            report_factor_name,
            combined_factor,
            forward_returns.loc[combined_factor.index.intersection(forward_returns.index)],
            output_dir=report_path,
            portfolio_returns=backtest["portfolio"] if "portfolio" in backtest else None,
            benchmark_returns=backtest["benchmark"] if "benchmark" in backtest else None,
        )

    if active.persist_outputs and output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target / "factor_summary.csv")
        score_table.to_csv(target / "factor_score_table.csv", index=False)
        score_table.to_parquet(target / "factor_score_table.parquet", index=False)
        correlation_matrix.to_csv(target / "factor_correlation.csv")
        pd.Series(selected_factors, name="factor").to_csv(target / "selected_factors.csv", index=False)
        pd.Series(factor_weights, name="weight").to_csv(target / "factor_weights.csv")
        combined_factor.to_csv(target / "combined_factor.csv")
        backtest.to_csv(target / "backtest_returns.csv")
        pd.DataFrame(backtest_stats).T.to_csv(target / "backtest_stats.csv")

    return FactorPipelineResult(
        summary=summary,
        score_table=score_table,
        correlation_matrix=correlation_matrix,
        selected_factors=selected_factors,
        selected_scores=selected_scores,
        factor_weights=factor_weights,
        combined_factor=combined_factor,
        forward_returns=forward_returns,
        backtest_returns=backtest,
        backtest_stats=backtest_stats,
        report_dir=report_path,
    )


__all__ = [
    "FactorPipelineConfig",
    "FactorPipelineResult",
    "run_factor_backtest",
    "screen_factor_panels",
]
