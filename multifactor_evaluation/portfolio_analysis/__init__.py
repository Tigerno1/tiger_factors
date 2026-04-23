from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from tiger_factors.utils.cross_sectional import (
    group_zscore,
    industry_size_regression_residual,
    orthogonalize,
)

try:  # pragma: no cover - optional dependency path
    from scipy import stats
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    stats = None


@dataclass(frozen=True)
class StrategyComparisonConfig:
    annual_trading_days: int = 252
    ranking_metric: str = "sharpe_ratio"


@dataclass(frozen=True)
class ComprehensiveScoringConfig:
    factor_weights: dict[str, float] | None = None
    strategy_weights: dict[str, float] | None = None
    portfolio_weights: dict[str, float] | None = None


def _get_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C+"
    if score >= 40:
        return "C"
    return "D"


def _ensure_dataframe(frame: pd.DataFrame, name: str = "frame") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return frame.copy()


def _to_numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _position_weight_series(
    positions: pd.DataFrame,
    *,
    key_column: str = "stock_code",
    weight_column: str = "weight",
) -> pd.Series:
    frame = _ensure_dataframe(positions, "positions")
    if key_column not in frame.columns:
        if frame.index.name == key_column:
            frame = frame.reset_index()
        else:
            raise ValueError(f"positions is missing required column: {key_column}")
    if weight_column not in frame.columns:
        raise ValueError(f"positions is missing required column: {weight_column}")

    weights = _to_numeric_series(frame[weight_column]).fillna(0.0)
    codes = frame[key_column].astype(str)
    series = pd.Series(weights.values, index=codes, name=weight_column)
    return series.groupby(level=0).sum().sort_index()


def _position_market_cap_series(
    positions: pd.DataFrame,
    *,
    key_column: str = "stock_code",
    market_cap_column: str = "market_cap",
) -> pd.Series:
    frame = _ensure_dataframe(positions, "positions")
    if key_column not in frame.columns:
        if frame.index.name == key_column:
            frame = frame.reset_index()
        else:
            raise ValueError(f"positions is missing required column: {key_column}")
    if market_cap_column not in frame.columns:
        raise ValueError(f"positions is missing required column: {market_cap_column}")

    caps = _to_numeric_series(frame[market_cap_column]).fillna(0.0).clip(lower=0.0)
    codes = frame[key_column].astype(str)
    series = pd.Series(caps.values, index=codes, name=market_cap_column)
    return series.groupby(level=0).sum().sort_index()


def _coerce_factor_series(value: Any, *, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        series = _to_numeric_series(value)
        if series.index.dtype == object or series.index.name is not None or len(series.index) == len(index):
            return series.reindex(index)
        return pd.Series(series.values, index=index[: len(series)], dtype=float).reindex(index)

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return pd.Series(dtype=float, index=index)
        if value.shape[0] == 1:
            return _to_numeric_series(value.iloc[0]).reindex(index)
        if value.shape[1] == 1:
            return _to_numeric_series(value.iloc[:, 0]).reindex(index)
        return _to_numeric_series(value.squeeze()).reindex(index)

    if np.isscalar(value):
        return pd.Series(float(value), index=index, dtype=float)

    raise TypeError(f"Unsupported factor value type: {type(value)!r}")


def calculate_industry_exposure(
    positions: pd.DataFrame,
    *,
    industry_column: str = "industry",
    weight_column: str = "weight",
) -> dict[str, Any]:
    frame = _ensure_dataframe(positions, "positions")
    if industry_column not in frame.columns:
        return {"error": f"data is missing required column: {industry_column}"}
    if weight_column not in frame.columns:
        return {"error": f"data is missing required column: {weight_column}"}

    weights = _to_numeric_series(frame[weight_column]).fillna(0.0)
    industry_weights = frame.assign(_weight=weights).groupby(industry_column)["_weight"].sum().sort_values(ascending=False)
    gross_weight = float(weights.abs().sum())
    if gross_weight > 0:
        exposure = industry_weights / gross_weight
    else:
        exposure = industry_weights.astype(float)

    abs_exposure = exposure.abs()
    return {
        "industry_exposure": exposure.to_dict(),
        "gross_industry_exposure": abs_exposure.to_dict(),
        "gross_weight": gross_weight,
        "net_weight": float(weights.sum()),
        "max_exposure": float(abs_exposure.max()) if not abs_exposure.empty else 0.0,
        "min_exposure": float(abs_exposure.min()) if not abs_exposure.empty else 0.0,
        "concentration": float(abs_exposure.std(ddof=0)) if len(abs_exposure) > 1 else 0.0,
        "top3_concentration": float(abs_exposure.nlargest(3).sum()) if not abs_exposure.empty else 0.0,
    }


def calculate_market_cap_weights(
    positions: pd.DataFrame,
    *,
    key_column: str = "stock_code",
    market_cap_column: str = "market_cap",
) -> dict[str, Any]:
    frame = _ensure_dataframe(positions, "positions")
    market_caps = _position_market_cap_series(frame, key_column=key_column, market_cap_column=market_cap_column)
    total_market_cap = float(market_caps.sum())
    if total_market_cap > 0:
        weights = market_caps / total_market_cap
    elif len(market_caps) > 0:
        weights = pd.Series(1.0 / len(market_caps), index=market_caps.index, dtype=float)
    else:
        weights = pd.Series(dtype=float)
    return {
        "market_cap_weights": weights.to_dict(),
        "market_caps": market_caps.to_dict(),
        "total_market_cap": total_market_cap,
    }


def neutralize_market_cap(
    df: pd.DataFrame,
    factor_name: str,
    *,
    market_cap_column: str = "market_cap",
) -> pd.Series:
    frame = _ensure_dataframe(df, "df")
    if factor_name not in frame.columns:
        raise ValueError(f"df is missing required column: {factor_name}")
    if market_cap_column not in frame.columns:
        raise ValueError(f"df is missing required column: {market_cap_column}")

    valid = frame[[factor_name, market_cap_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2:
        raise ValueError("valid data is insufficient for market cap neutralization")

    factor = _to_numeric_series(valid[factor_name]).astype(float)
    market_cap = _to_numeric_series(valid[market_cap_column]).astype(float)
    log_market_cap = np.log(market_cap.replace(0, np.nan))
    log_market_cap = pd.Series(log_market_cap, index=valid.index).replace([np.inf, -np.inf], np.nan)
    log_market_cap = log_market_cap.fillna(log_market_cap.mean())
    residual = orthogonalize(factor, log_market_cap.rename("log_market_cap").to_frame())

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid.index] = pd.to_numeric(residual, errors="coerce")
    return result


def neutralize_industry(
    df: pd.DataFrame,
    factor_name: str,
    *,
    industry_column: str = "industry",
) -> pd.Series:
    frame = _ensure_dataframe(df, "df")
    if factor_name not in frame.columns:
        raise ValueError(f"df is missing required column: {factor_name}")
    if industry_column not in frame.columns:
        raise ValueError(f"df is missing required column: {industry_column}")

    valid = frame[[factor_name, industry_column]].replace([np.inf, -np.inf], np.nan).dropna(subset=[factor_name])
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid.index] = group_zscore(_to_numeric_series(valid[factor_name]), valid[industry_column])
    return result


def neutralize_both(
    df: pd.DataFrame,
    factor_name: str,
    *,
    market_cap_column: str = "market_cap",
    industry_column: str = "industry",
) -> pd.Series:
    frame = _ensure_dataframe(df, "df")
    if factor_name not in frame.columns:
        raise ValueError(f"df is missing required column: {factor_name}")
    if market_cap_column not in frame.columns:
        raise ValueError(f"df is missing required column: {market_cap_column}")
    if industry_column not in frame.columns:
        raise ValueError(f"df is missing required column: {industry_column}")

    valid = frame[[factor_name, market_cap_column, industry_column]].replace([np.inf, -np.inf], np.nan).dropna(subset=[factor_name])
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid.index] = industry_size_regression_residual(
        _to_numeric_series(valid[factor_name]),
        valid[industry_column],
        _to_numeric_series(valid[market_cap_column]),
    )
    return result


def calculate_factor_exposure(
    positions: pd.DataFrame,
    factor_data: Mapping[str, pd.Series | pd.DataFrame | float | int],
    *,
    stock_code_column: str = "stock_code",
    weight_column: str = "weight",
) -> dict[str, Any]:
    frame = _ensure_dataframe(positions, "positions")
    if stock_code_column not in frame.columns:
        if frame.index.name == stock_code_column:
            frame = frame.reset_index()
        else:
            raise ValueError(f"positions is missing required column: {stock_code_column}")
    if weight_column not in frame.columns:
        raise ValueError(f"positions is missing required column: {weight_column}")

    weights = _to_numeric_series(frame[weight_column]).fillna(0.0)
    stock_weights = pd.Series(weights.values, index=frame[stock_code_column].astype(str), name=weight_column)
    stock_weights = stock_weights.groupby(level=0).sum().sort_index()
    gross_weight = float(stock_weights.abs().sum())

    exposures: dict[str, float] = {}
    for factor_name, values in factor_data.items():
        try:
            factor_series = _coerce_factor_series(values, index=stock_weights.index)
            aligned = pd.concat([stock_weights.rename("weight"), factor_series.rename("factor")], axis=1).dropna()
            if aligned.empty:
                continue
            denom = gross_weight if gross_weight > 0 else float(aligned["weight"].abs().sum())
            exposure = float((aligned["weight"] * aligned["factor"]).sum() / denom) if denom > 0 else 0.0
            exposures[str(factor_name)] = exposure
        except Exception:
            continue

    abs_exposures = {name: abs(value) for name, value in exposures.items()}
    return {
        "factor_exposures": exposures,
        "max_exposure": float(max(abs_exposures.values())) if abs_exposures else 0.0,
        "gross_weight": gross_weight,
    }


def _calculate_gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.astype(float))
    total = float(sorted_values.sum())
    if total <= 0:
        return 0.0
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)


def calculate_concentration(
    positions: pd.DataFrame,
    *,
    weight_column: str = "weight",
) -> dict[str, float]:
    frame = _ensure_dataframe(positions, "positions")
    if weight_column not in frame.columns:
        return {"error": f"data is missing required column: {weight_column}"}

    weights = _to_numeric_series(frame[weight_column]).abs().dropna()
    if weights.empty:
        return {
            "top10_concentration": 0.0,
            "herfindahl_index": 0.0,
            "gini_coefficient": 0.0,
        }

    total = float(weights.sum())
    normalized = weights / total if total > 0 else weights
    top10_concentration = float(normalized.sort_values(ascending=False).head(10).sum())
    herfindahl_index = float((normalized ** 2).sum())
    gini_coefficient = float(_calculate_gini(normalized.values))
    return {
        "top10_concentration": top10_concentration,
        "herfindahl_index": herfindahl_index,
        "gini_coefficient": gini_coefficient,
    }


def calculate_turnover(
    current_positions: pd.DataFrame,
    previous_positions: pd.DataFrame,
    *,
    key_column: str = "stock_code",
    weight_column: str = "weight",
) -> dict[str, float]:
    current = _position_weight_series(current_positions, key_column=key_column, weight_column=weight_column)
    previous = _position_weight_series(previous_positions, key_column=key_column, weight_column=weight_column)
    all_keys = current.index.union(previous.index)
    current = current.reindex(all_keys, fill_value=0.0)
    previous = previous.reindex(all_keys, fill_value=0.0)
    weight_change = (current - previous).abs()
    turnover = float(0.5 * weight_change.sum())
    gross_change = float(weight_change.sum())
    return {
        "turnover": turnover,
        "gross_change": gross_change,
        "n_current": float((current != 0).sum()),
        "n_previous": float((previous != 0).sum()),
    }


def calculate_risk_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    annual_trading_days: int = 252,
) -> dict[str, float]:
    series = _to_numeric_series(pd.Series(returns)).dropna()
    if series.empty:
        metrics = {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "downside_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
        }
        if benchmark_returns is not None:
            metrics.update({
                "tracking_error": 0.0,
                "beta": 0.0,
                "alpha": 0.0,
                "r2": 0.0,
                "information_ratio": 0.0,
            })
        return metrics

    cumulative = (1.0 + series).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (annual_trading_days / max(len(series), 1)) - 1.0)
    annual_volatility = float(series.std(ddof=0) * np.sqrt(annual_trading_days))
    downside = series[series < 0]
    downside_volatility = float(downside.std(ddof=0) * np.sqrt(annual_trading_days)) if not downside.empty else 0.0
    sharpe_ratio = float(annual_return / annual_volatility) if annual_volatility > 1e-12 else 0.0
    sortino_ratio = float(annual_return / downside_volatility) if downside_volatility > 1e-12 else 0.0
    win_rate = float((series > 0).mean())
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    var_95 = float(series.quantile(0.05))
    cvar_95 = float(series[series <= var_95].mean()) if not series[series <= var_95].empty else var_95

    metrics: dict[str, float] = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "downside_volatility": downside_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }

    if benchmark_returns is not None:
        benchmark = _to_numeric_series(pd.Series(benchmark_returns))
        aligned = pd.concat([series.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
        if aligned.empty:
            metrics.update({
                "tracking_error": 0.0,
                "beta": 0.0,
                "alpha": 0.0,
                "r2": 0.0,
                "information_ratio": 0.0,
            })
        else:
            excess = aligned["portfolio"] - aligned["benchmark"]
            tracking_error = float(excess.std(ddof=0) * np.sqrt(annual_trading_days))
            benchmark_var = float(aligned["benchmark"].var(ddof=0))
            beta = float(aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_var) if benchmark_var > 1e-12 else 0.0
            alpha_daily = float(aligned["portfolio"].mean() - beta * aligned["benchmark"].mean())
            alpha = float(alpha_daily * annual_trading_days)
            corr = aligned["portfolio"].corr(aligned["benchmark"])
            information_ratio = float(excess.mean() / excess.std(ddof=0) * np.sqrt(annual_trading_days)) if excess.std(ddof=0) > 1e-12 else 0.0
            metrics.update({
                "tracking_error": tracking_error,
                "beta": beta,
                "alpha": alpha,
                "r2": float(corr ** 2) if pd.notna(corr) else 0.0,
                "information_ratio": information_ratio,
            })

    return metrics


def score_factor(
    factor_metrics: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cfg_weights = dict(weights) if weights is not None else {
        "ic": 0.35,
        "ir": 0.30,
        "stability": 0.20,
        "turnover": 0.15,
    }

    ic_mean = abs(float(factor_metrics.get("ic_mean", 0.0)))
    ir = factor_metrics.get("ir", factor_metrics.get("ic_ir", 0.0))
    ir = abs(float(ir if ir is not None else 0.0))
    stability = factor_metrics.get("stability_score", factor_metrics.get("regime_robustness", 0.8))
    stability = float(stability if stability is not None else 0.8)
    turnover = float(factor_metrics.get("turnover", 0.3))

    ic_score = min(ic_mean * 400.0, 100.0)
    ir_score = min(ir * 40.0, 100.0)
    stability_score = stability * 100.0
    turnover_score = max(100.0 - turnover * 200.0, 0.0)

    total_score = (
        cfg_weights.get("ic", 0.0) * ic_score
        + cfg_weights.get("ir", 0.0) * ir_score
        + cfg_weights.get("stability", 0.0) * stability_score
        + cfg_weights.get("turnover", 0.0) * turnover_score
    )

    details = {
        "ic_score": float(ic_score),
        "ir_score": float(ir_score),
        "stability_score": float(stability_score),
        "turnover_score": float(turnover_score),
    }
    return {
        "total_score": round(float(total_score), 2),
        "grade": _get_grade(float(total_score)),
        "details": details,
        "weights": cfg_weights,
    }


def score_strategy(
    strategy_metrics: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cfg_weights = dict(weights) if weights is not None else {
        "return": 0.30,
        "risk": 0.25,
        "efficiency": 0.20,
        "stability": 0.15,
        "cost": 0.10,
    }

    annual_return = float(strategy_metrics.get("annual_return", 0.0))
    max_drawdown = float(strategy_metrics.get("max_drawdown", 0.2))
    sharpe_ratio = float(strategy_metrics.get("sharpe_ratio", 0.0))
    win_rate = float(strategy_metrics.get("win_rate", 0.5))
    turnover = float(strategy_metrics.get("turnover", 0.5))

    return_score = min(max(annual_return / 0.2 * 100.0, 0.0), 100.0)
    risk_score = max(100.0 - abs(max_drawdown) / 0.1 * 100.0, 0.0)
    efficiency_score = min(sharpe_ratio / 2.0 * 100.0, 100.0)
    stability_score = win_rate * 100.0
    cost_score = max(100.0 - turnover * 100.0, 0.0)

    total_score = (
        cfg_weights.get("return", 0.0) * return_score
        + cfg_weights.get("risk", 0.0) * risk_score
        + cfg_weights.get("efficiency", 0.0) * efficiency_score
        + cfg_weights.get("stability", 0.0) * stability_score
        + cfg_weights.get("cost", 0.0) * cost_score
    )

    details = {
        "return_score": float(return_score),
        "risk_score": float(risk_score),
        "efficiency_score": float(efficiency_score),
        "stability_score": float(stability_score),
        "cost_score": float(cost_score),
    }
    return {
        "total_score": round(float(total_score), 2),
        "grade": _get_grade(float(total_score)),
        "details": details,
        "weights": cfg_weights,
    }


def score_portfolio(
    portfolio_metrics: Mapping[str, Any],
    benchmark_metrics: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cfg_weights = dict(weights) if weights is not None else {
        "return": 0.35,
        "risk": 0.30,
        "diversification": 0.20,
        "efficiency": 0.15,
    }

    annual_return = float(portfolio_metrics.get("annual_return", 0.0))
    volatility = float(portfolio_metrics.get("volatility", portfolio_metrics.get("annual_volatility", 0.15)))
    max_drawdown = float(portfolio_metrics.get("max_drawdown", 0.1))
    sharpe_ratio = float(portfolio_metrics.get("sharpe_ratio", 0.0))
    concentration = float(portfolio_metrics.get("herfindahl_index", portfolio_metrics.get("concentration", 0.1)))

    return_score = min(max(annual_return / 0.15 * 100.0, 0.0), 100.0)
    if benchmark_metrics:
        benchmark_return = float(benchmark_metrics.get("annual_return", 0.0))
        excess_return = annual_return - benchmark_return
        return_score = min(max(excess_return / 0.05 * 100.0, 0.0), 100.0)

    risk_score = max(100.0 - (volatility / 0.2 * 50.0 + abs(max_drawdown) / 0.15 * 50.0), 0.0)
    diversification_score = max(100.0 - concentration * 100.0, 0.0)
    efficiency_score = min(sharpe_ratio / 2.0 * 100.0, 100.0)

    total_score = (
        cfg_weights.get("return", 0.0) * return_score
        + cfg_weights.get("risk", 0.0) * risk_score
        + cfg_weights.get("diversification", 0.0) * diversification_score
        + cfg_weights.get("efficiency", 0.0) * efficiency_score
    )

    details = {
        "return_score": float(return_score),
        "risk_score": float(risk_score),
        "diversification_score": float(diversification_score),
        "efficiency_score": float(efficiency_score),
    }
    return {
        "total_score": round(float(total_score), 2),
        "grade": _get_grade(float(total_score)),
        "details": details,
        "weights": cfg_weights,
    }


def _extract_returns_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value
    elif isinstance(value, pd.DataFrame):
        for candidate in ("portfolio", "returns", "return", "equity_curve"):
            if candidate in value.columns:
                series = value[candidate]
                break
        else:
            numeric = value.select_dtypes(include=[np.number])
            if numeric.empty:
                raise ValueError("Strategy data frame does not contain a numeric returns column.")
            series = numeric.iloc[:, 0]
    else:
        raise TypeError(f"Unsupported strategy value type: {type(value)!r}")

    series = _to_numeric_series(pd.Series(series)).dropna()
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()


def compare_and_rank(
    items: list[Mapping[str, Any]],
    scoring_type: str = "strategy",
) -> list[dict[str, Any]]:
    if not items:
        return []

    scored: list[dict[str, Any]] = []
    for item in items:
        metrics = item.get("metrics", {})
        name = str(item.get("name", "Unknown"))
        if scoring_type == "factor":
            score_result = score_factor(metrics)
        elif scoring_type == "strategy":
            score_result = score_strategy(metrics)
        elif scoring_type == "portfolio":
            score_result = score_portfolio(metrics)
        else:
            raise ValueError("scoring_type must be one of: 'factor', 'strategy', 'portfolio'.")
        scored.append(
            {
                "name": name,
                "metrics": metrics,
                "score": score_result["total_score"],
                "grade": score_result["grade"],
                "details": score_result["details"],
                "weights": score_result["weights"],
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
    return scored


def compare_strategies(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
    *,
    benchmark_returns: pd.Series | None = None,
    annual_trading_days: int = 252,
    ranking_metric: str = "sharpe_ratio",
) -> dict[str, Any]:
    if not strategies:
        raise ValueError("strategies must not be empty.")

    returns_map = {str(name): _extract_returns_series(value) for name, value in strategies.items()}

    metrics_by_strategy = {
        name: calculate_risk_metrics(returns, benchmark_returns=benchmark_returns, annual_trading_days=annual_trading_days)
        for name, returns in returns_map.items()
    }
    metrics_table = pd.DataFrame(metrics_by_strategy).T

    if ranking_metric not in metrics_table.columns:
        raise ValueError(f"ranking_metric '{ranking_metric}' is not available in strategy metrics.")

    ranking = {
        ranking_metric: metrics_table[ranking_metric].rank(ascending=False, method="min").to_dict(),
        "annual_return": metrics_table["annual_return"].rank(ascending=False, method="min").to_dict(),
        "max_drawdown": metrics_table["max_drawdown"].rank(ascending=True, method="min").to_dict(),
        "overall": {},
    }
    overall_score: dict[str, float] = {}
    for strategy in metrics_table.index:
        overall_score[strategy] = float(
            ranking[ranking_metric][strategy]
            + ranking["annual_return"][strategy]
            + ranking["max_drawdown"][strategy]
        ) / 3.0
    ranking["overall"] = {name: rank for rank, (name, _) in enumerate(sorted(overall_score.items(), key=lambda item: item[1]), start=1)}

    pairwise_tests: dict[str, Any] = {}
    names = list(returns_map.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            aligned = pd.concat(
                [returns_map[left_name].rename("left"), returns_map[right_name].rename("right")],
                axis=1,
            ).dropna()
            if aligned.empty:
                continue
            if stats is not None and len(aligned) >= 2:
                t_stat, p_value = stats.ttest_ind(aligned["left"], aligned["right"], equal_var=False)
                paired_t_stat, paired_p_value = stats.ttest_rel(aligned["left"], aligned["right"])
            else:  # pragma: no cover - fallback without scipy
                t_stat = p_value = paired_t_stat = paired_p_value = np.nan
            pairwise_tests[f"{left_name}_vs_{right_name}"] = {
                "independent_t_test": {
                    "statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "significant": bool(p_value < 0.05) if pd.notna(p_value) else False,
                },
                "paired_t_test": {
                    "statistic": float(paired_t_stat) if pd.notna(paired_t_stat) else np.nan,
                    "p_value": float(paired_p_value) if pd.notna(paired_p_value) else np.nan,
                    "significant": bool(paired_p_value < 0.05) if pd.notna(paired_p_value) else False,
                },
                "correlation": float(aligned["left"].corr(aligned["right"])) if len(aligned) > 1 else np.nan,
            }

    correlation_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for left_name in names:
        for right_name in names:
            left = returns_map[left_name]
            right = returns_map[right_name]
            aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
            correlation_matrix.loc[left_name, right_name] = float(aligned["left"].corr(aligned["right"])) if len(aligned) > 1 else np.nan

    return {
        "returns": returns_map,
        "metrics_table": metrics_table,
        "correlation_matrix": correlation_matrix,
        "pairwise_tests": pairwise_tests,
        "ranking": ranking,
    }


def calculate_combined_factor_score(
    factor_data: Mapping[str, pd.Series | pd.DataFrame | float | int],
    weights: Mapping[str, float],
    *,
    normalize: bool = True,
) -> pd.Series:
    if not factor_data:
        return pd.Series(dtype=float)
    if not weights:
        raise ValueError("weights must not be empty.")

    common_index: pd.Index | None = None
    series_map: dict[str, pd.Series] = {}
    scalar_map: dict[str, float] = {}
    for name, value in factor_data.items():
        key = str(name)
        if isinstance(value, pd.Series):
            series = _to_numeric_series(value).dropna()
        elif isinstance(value, pd.DataFrame):
            if value.empty:
                series = pd.Series(dtype=float)
            elif value.shape[1] == 1:
                series = _to_numeric_series(value.iloc[:, 0]).dropna()
            elif value.shape[0] == 1:
                series = _to_numeric_series(value.iloc[0]).dropna()
            else:
                squeezed = value.squeeze()
                if isinstance(squeezed, pd.Series):
                    series = _to_numeric_series(squeezed).dropna()
                else:
                    raise ValueError("DataFrame factor values must be reducible to a single Series.")
        elif np.isscalar(value):
            scalar_map[key] = float(value)
            continue
        else:
            raise TypeError(f"Unsupported factor value type: {type(value)!r}")

        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)
        series_map[key] = series

    if common_index is None or len(common_index) == 0:
        return pd.Series(dtype=float)

    combined = pd.Series(0.0, index=common_index, dtype=float)
    for name, weight in weights.items():
        key = str(name)
        if key in series_map:
            aligned = pd.to_numeric(series_map[key].reindex(common_index), errors="coerce")
        elif key in scalar_map:
            aligned = pd.Series(scalar_map[key], index=common_index, dtype=float)
        else:
            continue
        if normalize:
            mean = aligned.mean()
            std = aligned.std(ddof=0)
            if std > 1e-12:
                aligned = (aligned - mean) / std
            else:
                aligned = aligned - mean
        combined = combined.add(aligned.fillna(0.0) * float(weight), fill_value=0.0)

    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    combined.name = "composite_score"
    return combined.sort_index()


def optimize_factor_weights(
    factor_returns: pd.DataFrame,
    *,
    method: str = "equal_weight",
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    if factor_returns.empty:
        return {"weights": {}, "method": method, "error": "因子收益率为空"}

    returns = factor_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_factors = len(returns.columns)
    extra_info: dict[str, Any] = {}

    if method == "equal_weight":
        weights = pd.Series(1.0 / n_factors, index=returns.columns)
        extra_info["note"] = "等权重分配"
    elif method == "ic_weight":
        factor_stats: dict[str, dict[str, float]] = {}
        for factor in returns.columns:
            factor_series = returns[factor].dropna()
            if factor_series.empty:
                continue
            mean_return = float(factor_series.mean())
            std_return = float(factor_series.std(ddof=0))
            ir = mean_return / std_return if std_return > 0 else 0.0
            factor_stats[str(factor)] = {"mean": mean_return, "std": std_return, "ir": ir}
        ir_values = pd.Series({factor: max(0.0, stats["ir"]) for factor, stats in factor_stats.items()}, dtype=float)
        if ir_values.sum() <= 0:
            weights = pd.Series(1.0 / n_factors, index=returns.columns)
        else:
            weights = ir_values / ir_values.sum()
            weights = weights.reindex(returns.columns, fill_value=0.0)
        extra_info["factor_stats"] = factor_stats
    elif method == "risk_parity":
        volatilities = returns.std(ddof=0).replace(0, np.nan)
        inv_vol = 1.0 / volatilities
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if inv_vol.sum() <= 0:
            weights = pd.Series(1.0 / n_factors, index=returns.columns)
        else:
            weights = inv_vol / inv_vol.sum()
        extra_info["note"] = "按波动率倒数分配"
    elif method == "max_sharpe":
        mean_returns = returns.mean()
        std_returns = returns.std(ddof=0).replace(0, np.nan)
        sharpe_ratios = mean_returns / std_returns * np.sqrt(252)
        sharpe_ratios = sharpe_ratios.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        positive_sharpe = sharpe_ratios[sharpe_ratios > 0]
        if positive_sharpe.empty:
            weights = pd.Series(1.0 / n_factors, index=returns.columns)
        else:
            weights = pd.Series(0.0, index=returns.columns, dtype=float)
            weights.update(positive_sharpe / positive_sharpe.sum())
        extra_info["sharpe_ratios"] = sharpe_ratios.to_dict()
    elif method == "min_variance":
        cov_matrix = returns.cov(ddof=0)
        variances = pd.Series(np.diag(cov_matrix), index=cov_matrix.index)
        inv_var = 1.0 / (variances + 1e-8)
        inv_var = inv_var.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if inv_var.sum() <= 0:
            weights = pd.Series(1.0 / n_factors, index=returns.columns)
        else:
            weights = inv_var / inv_var.sum()
        extra_info["note"] = "基于方差倒数的简化最小方差"
    else:
        raise ValueError("method must be one of: equal_weight, ic_weight, risk_parity, max_sharpe, min_variance")

    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    total = float(weights.sum())
    if np.isfinite(total) and abs(total) > 1e-12:
        weights = weights / total

    mean_returns = returns.mean()
    cov_matrix = returns.cov(ddof=0)
    weights_vector = weights.reindex(returns.columns, fill_value=0.0).astype(float)
    weighted_return = float((weights_vector * mean_returns).sum() * 252)
    portfolio_variance = float(np.dot(weights_vector.T, np.dot(cov_matrix.values, weights_vector.values))) if not cov_matrix.empty else 0.0
    weighted_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
    sharpe_ratio = float((weighted_return - risk_free_rate) / weighted_volatility) if weighted_volatility > 1e-12 else 0.0

    result = {
        "weights": weights_vector.to_dict(),
        "method": method,
        "expected_return": weighted_return,
        "expected_volatility": weighted_volatility,
        "sharpe_ratio": sharpe_ratio,
    }
    result.update(extra_info)
    return result


def compare_weight_methods(
    factor_returns: pd.DataFrame,
    methods: list[str] | None = None,
    *,
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    if methods is None:
        methods = ["equal_weight", "ic_weight", "risk_parity", "max_sharpe", "min_variance"]

    results: dict[str, Any] = {}
    for method in methods:
        optimization_result = optimize_factor_weights(
            factor_returns,
            method=method,
            risk_free_rate=risk_free_rate,
        )
        if "error" not in optimization_result:
            results[method] = {
                "annual_return": optimization_result["expected_return"],
                "volatility": optimization_result["expected_volatility"],
                "sharpe_ratio": (
                    optimization_result["expected_return"] / optimization_result["expected_volatility"]
                    if optimization_result["expected_volatility"] > 0
                    else 0.0
                ),
            }
    return results


def generate_comparison_report(comparison_result: dict[str, Any]) -> str:
    report = "# 策略对比报告\n\n"
    metrics_table = comparison_result.get("metrics_table", pd.DataFrame())
    if isinstance(metrics_table, pd.DataFrame) and not metrics_table.empty:
        report += "## 指标对比\n\n"
        report += metrics_table.to_string()
    ranking = comparison_result.get("ranking", {})
    if ranking:
        report += "\n\n## 排名\n\n"
        overall = ranking.get("overall", {})
        for strategy, rank in sorted(overall.items(), key=lambda item: item[1]):
            report += f"- {rank}. {strategy}\n"
    tests = comparison_result.get("pairwise_tests", {})
    if tests:
        report += "\n\n## 统计显著性检验\n\n"
        for test_name, test_result in tests.items():
            report += f"### {test_name}\n"
            report += f"- 独立T检验 p值: {test_result['independent_t_test']['p_value']}\n"
            report += f"- 配对T检验 p值: {test_result['paired_t_test']['p_value']}\n"
            report += f"- 相关系数: {test_result['correlation']}\n"
    return report


def analyze_portfolio_comprehensive(
    positions: pd.DataFrame,
    returns: pd.Series,
    *,
    factor_data: Mapping[str, pd.Series | pd.DataFrame | float | int] | None = None,
    benchmark_returns: pd.Series | None = None,
    previous_positions: pd.DataFrame | None = None,
    industry_column: str = "industry",
    stock_code_column: str = "stock_code",
    weight_column: str = "weight",
    annual_trading_days: int = 252,
) -> dict[str, Any]:
    frame = _ensure_dataframe(positions, "positions")
    exposure = calculate_industry_exposure(
        frame,
        industry_column=industry_column,
        weight_column=weight_column,
    )
    concentration = calculate_concentration(frame, weight_column=weight_column)
    risk_metrics = calculate_risk_metrics(
        returns,
        benchmark_returns=benchmark_returns,
        annual_trading_days=annual_trading_days,
    )
    factor_exposure = (
        calculate_factor_exposure(
            frame,
            factor_data,
            stock_code_column=stock_code_column,
            weight_column=weight_column,
        )
        if factor_data
        else {"factor_exposures": {}, "max_exposure": 0.0, "gross_weight": 0.0}
    )
    turnover = (
        calculate_turnover(
            frame,
            previous_positions,
            key_column=stock_code_column,
            weight_column=weight_column,
        )
        if previous_positions is not None
        else None
    )

    return {
        "industry_exposure": exposure,
        "factor_exposure": factor_exposure,
        "concentration": concentration,
        "risk_metrics": risk_metrics,
        "turnover": turnover,
    }


__all__ = [
    "ComprehensiveScoringConfig",
    "StrategyComparisonConfig",
    "analyze_portfolio_comprehensive",
    "calculate_concentration",
    "calculate_factor_exposure",
    "calculate_industry_exposure",
    "calculate_market_cap_weights",
    "calculate_risk_metrics",
    "calculate_turnover",
    "calculate_combined_factor_score",
    "compare_strategies",
    "compare_and_rank",
    "compare_weight_methods",
    "generate_comparison_report",
    "neutralize_both",
    "neutralize_industry",
    "neutralize_market_cap",
    "optimize_factor_weights",
    "score_factor",
    "score_portfolio",
    "score_strategy",
]
