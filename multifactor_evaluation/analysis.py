from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation.portfolio_analysis import analyze_portfolio_comprehensive
from tiger_factors.multifactor_evaluation.portfolio_analysis import calculate_risk_metrics
from tiger_factors.multifactor_evaluation.portfolio_analysis import calculate_turnover
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import compare as compare_returns
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import best
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import cpc_index
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import common_sense_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import consecutive_losses
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import consecutive_wins
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import expected_shortfall
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import gain_to_pain_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import information_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import kelly_criterion
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import max_drawdown as quantstats_max_drawdown
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import omega
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import drawdown_details
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import monthly_returns
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import payoff_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import probabilistic_adjusted_sortino_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import probabilistic_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import probabilistic_sharpe_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import probabilistic_sortino_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import profit_factor
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import profit_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import recovery_factor
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import r2
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import risk_return_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import serenity_index
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import smart_sharpe
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import smart_sortino
from tiger_factors.multifactor_evaluation.reporting.trades import extract_round_trips
from tiger_factors.utils.returns_analysis import clean_returns
from tiger_factors.utils.returns_analysis import drawdown_series
from tiger_factors.utils.returns_analysis import monthly_returns_heatmap
from tiger_factors.utils.returns_analysis import rolling_beta
from tiger_factors.utils.returns_analysis import rolling_sharpe
from tiger_factors.utils.returns_analysis import rolling_sortino
from tiger_factors.utils.returns_analysis import rolling_volatility
from tiger_factors.utils.returns_analysis import max_drawdown
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import calmar
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import cagr
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import cvar
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import exposure
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import kurtosis
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import tail_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import treynor_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import ulcer_index
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import ulcer_performance_index
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import var
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import value_at_risk
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import win_loss_ratio
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import skew
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import worst


def _to_series(values: pd.Series | pd.DataFrame | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.empty:
            return pd.Series(dtype=float)
        if "returns" in values.columns:
            values = values["returns"]
        elif "return" in values.columns:
            values = values["return"]
        elif "portfolio" in values.columns:
            values = values["portfolio"]
        else:
            values = values.select_dtypes(include=[np.number]).iloc[:, 0] if not values.select_dtypes(include=[np.number]).empty else values.iloc[:, 0]
    return clean_returns(pd.Series(values))


def _normalize_positions_frame(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None:
        return pd.DataFrame()
    frame = pd.DataFrame(positions).copy()
    if frame.empty:
        return frame
    long_key_candidates = ("code", "stock_code", "symbol")
    long_key_column = next((column for column in long_key_candidates if column in frame.columns), None)
    if "weight" in frame.columns and ("date_" in frame.columns or "date" in frame.columns) and long_key_column is not None:
        date_column = "date_" if "date_" in frame.columns else "date"
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, long_key_column])
        frame[long_key_column] = frame[long_key_column].astype(str)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
        wide = frame.pivot_table(index=date_column, columns=long_key_column, values="weight", aggfunc="sum").sort_index()
        wide.index = pd.DatetimeIndex(wide.index, name="date_")
        wide.columns = wide.columns.astype(str)
        return wide
    if {"date_", "code", "weight"}.issubset(frame.columns):
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_", "code"])
        frame["code"] = frame["code"].astype(str)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
        wide = frame.pivot_table(index="date_", columns="code", values="weight", aggfunc="sum").sort_index()
        wide.index = pd.DatetimeIndex(wide.index, name="date_")
        wide.columns = wide.columns.astype(str)
        return wide
    if "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.set_index("date_")
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    numeric.columns = numeric.columns.astype(str)
    return numeric


def _gross_leverage_series(positions: pd.DataFrame | None) -> pd.Series:
    frame = _normalize_positions_frame(positions)
    if frame.empty:
        return pd.Series(dtype=float, name="gross_leverage")
    asset_cols = [column for column in frame.columns if column != "cash"]
    if not asset_cols:
        return pd.Series(dtype=float, name="gross_leverage")
    gross = frame[asset_cols].abs().sum(axis=1)
    gross.name = "gross_leverage"
    return gross.sort_index()


def _latest_holdings_long(positions: pd.DataFrame | None) -> pd.DataFrame:
    frame = _normalize_positions_frame(positions)
    if frame.empty:
        return pd.DataFrame(columns=["date_", "stock_code", "weight"])
    latest = frame.iloc[-1].drop(labels=["cash"], errors="ignore")
    latest = pd.to_numeric(latest, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if latest.empty:
        return pd.DataFrame(columns=["date_", "stock_code", "weight"])
    date_value = pd.Timestamp(frame.index[-1])
    holdings = pd.DataFrame(
        {
            "date_": date_value,
            "stock_code": latest.index.astype(str),
            "weight": latest.astype(float).values,
        }
    )
    return holdings.sort_values("weight", ascending=False).reset_index(drop=True)


def _frame_turnover(current_positions: pd.DataFrame, previous_positions: pd.DataFrame) -> dict[str, float]:
    current = _normalize_positions_frame(current_positions)
    previous = _normalize_positions_frame(previous_positions)
    if current.empty or previous.empty:
        return {"turnover": 0.0, "gross_change": 0.0, "n_current": float(len(current.columns)), "n_previous": float(len(previous.columns))}
    all_columns = current.columns.union(previous.columns)
    current = current.reindex(columns=all_columns, fill_value=0.0)
    previous = previous.reindex(columns=all_columns, fill_value=0.0)
    weight_change = (current.iloc[-1] - previous.iloc[-1]).abs()
    turnover = float(0.5 * weight_change.sum())
    gross_change = float(weight_change.sum())
    return {
        "turnover": turnover,
        "gross_change": gross_change,
        "n_current": float((current.iloc[-1] != 0).sum()),
        "n_previous": float((previous.iloc[-1] != 0).sum()),
    }


def _daily_transaction_totals(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["txn_volume", "txn_shares"])
    frame = pd.DataFrame(transactions).copy()
    if "dt" in frame.columns:
        frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
        frame = frame.set_index("dt")
    elif "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.set_index("date_")
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"), name="date_")
    frame = frame.loc[~frame.index.isna()].sort_index()
    if "txn_dollars" not in frame.columns and {"amount", "price"}.issubset(frame.columns):
        frame["txn_dollars"] = -pd.to_numeric(frame["amount"], errors="coerce") * pd.to_numeric(frame["price"], errors="coerce")
    txn_volume = pd.to_numeric(frame.get("txn_dollars"), errors="coerce").abs().groupby(frame.index.normalize()).sum()
    txn_shares = pd.to_numeric(frame.get("amount"), errors="coerce").abs().groupby(frame.index.normalize()).sum()
    out = pd.DataFrame({"txn_volume": txn_volume, "txn_shares": txn_shares})
    out.index.name = "date_"
    return out.sort_index()


@dataclass(frozen=True)
class ReturnAnalysisResult:
    returns: pd.Series
    benchmark_returns: pd.Series | None
    summary: dict[str, float]
    metric_table: pd.DataFrame
    compare_table: pd.DataFrame
    monthly_returns_heatmap: pd.DataFrame
    drawdown: pd.Series
    drawdown_details: pd.DataFrame
    monthly_returns: pd.Series
    rolling_sharpe: pd.Series
    rolling_sortino: pd.Series
    rolling_volatility: pd.Series
    rolling_beta: pd.Series
    live_start_date: pd.Timestamp | None = None
    pre_live_summary: dict[str, float] | None = None
    live_summary: dict[str, float] | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "return_count": int(len(self.returns)),
            "benchmark_count": int(len(self.benchmark_returns)) if self.benchmark_returns is not None else 0,
            "summary": self.summary,
            "metric_rows": int(len(self.metric_table)),
            "compare_rows": int(len(self.compare_table)),
            "monthly_heatmap_rows": int(len(self.monthly_returns_heatmap)),
            "monthly_heatmap_cols": int(len(self.monthly_returns_heatmap.columns)),
            "drawdown_rows": int(len(self.drawdown_details)),
            "monthly_return_rows": int(len(self.monthly_returns)),
            "rolling_sharpe_rows": int(len(self.rolling_sharpe)),
            "rolling_sortino_rows": int(len(self.rolling_sortino)),
            "rolling_volatility_rows": int(len(self.rolling_volatility)),
            "rolling_beta_rows": int(len(self.rolling_beta)),
            "live_start_date": None if self.live_start_date is None else self.live_start_date.isoformat(),
            "pre_live_summary": self.pre_live_summary,
            "live_summary": self.live_summary,
        }


@dataclass(frozen=True)
class PositionAnalysisResult:
    positions: pd.DataFrame
    gross_leverage: pd.Series
    concentration: dict[str, float]
    turnover: dict[str, float] | None
    risk_metrics: dict[str, float] | None
    industry_exposure: dict[str, Any]
    factor_exposure: dict[str, Any]
    comprehensive: dict[str, Any]
    latest_holdings: pd.DataFrame

    def to_summary(self) -> dict[str, Any]:
        return {
            "position_rows": int(len(self.positions)),
            "gross_leverage_mean": float(self.gross_leverage.mean()) if not self.gross_leverage.empty else 0.0,
            "gross_leverage_max": float(self.gross_leverage.max()) if not self.gross_leverage.empty else 0.0,
            "concentration": self.concentration,
            "turnover": self.turnover,
            "risk_metrics": self.risk_metrics,
            "industry_exposure": self.industry_exposure,
            "factor_exposure": self.factor_exposure,
            "latest_holdings_rows": int(len(self.latest_holdings)),
        }


@dataclass(frozen=True)
class TransactionAnalysisResult:
    transactions: pd.DataFrame
    daily_totals: pd.DataFrame
    round_trips: pd.DataFrame
    round_trip_summary: pd.DataFrame
    summary: dict[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "transaction_count": int(len(self.transactions)),
            "daily_transaction_rows": int(len(self.daily_totals)),
            "round_trip_count": int(len(self.round_trips)),
            "round_trip_summary_rows": int(len(self.round_trip_summary)),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class MultifactorAnalysisResult:
    returns: ReturnAnalysisResult | None = None
    positions: PositionAnalysisResult | None = None
    transactions: TransactionAnalysisResult | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "returns": None if self.returns is None else self.returns.to_summary(),
            "positions": None if self.positions is None else self.positions.to_summary(),
            "transactions": None if self.transactions is None else self.transactions.to_summary(),
        }


def analyze_returns(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    live_start_date: str | pd.Timestamp | None = None,
    annual_trading_days: int = 252,
    rolling_window: int = 126,
) -> ReturnAnalysisResult:
    portfolio = _to_series(returns)
    benchmark = _to_series(benchmark_returns) if benchmark_returns is not None else None
    summary = calculate_risk_metrics(portfolio, benchmark_returns=benchmark, annual_trading_days=annual_trading_days)
    compare_table = compare_returns(portfolio, benchmark) if benchmark is not None else pd.DataFrame()
    dd = drawdown_series(portfolio)
    dd_details = drawdown_details(portfolio)
    monthly = monthly_returns(portfolio)
    monthly_heatmap = monthly_returns_heatmap(portfolio)
    rolling_window = max(int(rolling_window), 2)

    if len(portfolio) >= 2:
        rolling_sh = rolling_sharpe(portfolio, rolling_window, annualization=annual_trading_days)
        rolling_so = rolling_sortino(portfolio, rolling_window, annualization=annual_trading_days)
        rolling_vol = rolling_volatility(portfolio, rolling_window, annualization=annual_trading_days)
    else:
        rolling_sh = pd.Series(dtype=float)
        rolling_so = pd.Series(dtype=float)
        rolling_vol = pd.Series(dtype=float)

    if benchmark is not None and len(portfolio) >= 2:
        rolling_b = rolling_beta(portfolio, benchmark, rolling_window)
    else:
        rolling_b = pd.Series(dtype=float)

    parsed_live_start = pd.Timestamp(live_start_date) if live_start_date is not None else None
    pre_live_summary: dict[str, float] | None = None
    live_summary: dict[str, float] | None = None
    if parsed_live_start is not None and not portfolio.empty:
        pre_live = portfolio.loc[portfolio.index < parsed_live_start]
        live = portfolio.loc[portfolio.index >= parsed_live_start]
        if not pre_live.empty:
            pre_live_summary = calculate_risk_metrics(pre_live, benchmark_returns=benchmark.loc[benchmark.index < parsed_live_start] if benchmark is not None else None, annual_trading_days=annual_trading_days)
        if not live.empty:
            live_summary = calculate_risk_metrics(live, benchmark_returns=benchmark.loc[benchmark.index >= parsed_live_start] if benchmark is not None else None, annual_trading_days=annual_trading_days)

    metric_table = pd.DataFrame(
        {
            "metric": [
                "annual_return",
                "annual_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "smart_sharpe",
                "smart_sortino",
                "calmar_ratio",
                "max_drawdown",
                "var_95",
                "cvar_95",
                "expected_shortfall",
                "cagr",
                "best_period_return",
                "worst_period_return",
                "consecutive_wins",
                "consecutive_losses",
                "profit_factor",
                "common_sense_ratio",
                "cpc_index",
                "gain_to_pain_ratio",
                "payoff_ratio",
                "win_loss_ratio",
                "profit_ratio",
                "recovery_factor",
                "serenity_index",
                "risk_return_ratio",
                "omega",
                "kelly_criterion",
                "tail_ratio",
                "skew",
                "kurtosis",
                "exposure",
                "win_rate",
                "ulcer_index",
                "ulcer_performance_index",
                "probabilistic_sharpe_ratio",
                "probabilistic_sortino_ratio",
                "probabilistic_adjusted_sortino_ratio",
                "probabilistic_ratio",
            ],
            "value": [
                summary.get("annual_return", 0.0),
                summary.get("annual_volatility", 0.0),
                summary.get("sharpe_ratio", 0.0),
                summary.get("sortino_ratio", 0.0),
                smart_sharpe(portfolio, annualization=annual_trading_days),
                smart_sortino(portfolio, annualization=annual_trading_days),
                calmar(portfolio, annualization=annual_trading_days),
                quantstats_max_drawdown(portfolio),
                var(portfolio),
                cvar(portfolio),
                expected_shortfall(portfolio),
                cagr(portfolio, annualization=annual_trading_days),
                best(portfolio),
                worst(portfolio),
                consecutive_wins(portfolio),
                consecutive_losses(portfolio),
                profit_factor(portfolio),
                common_sense_ratio(portfolio),
                cpc_index(portfolio),
                gain_to_pain_ratio(portfolio),
                payoff_ratio(portfolio),
                win_loss_ratio(portfolio),
                profit_ratio(portfolio),
                recovery_factor(portfolio),
                serenity_index(portfolio),
                risk_return_ratio(portfolio),
                omega(portfolio),
                kelly_criterion(portfolio),
                tail_ratio(portfolio),
                skew(portfolio),
                kurtosis(portfolio),
                exposure(portfolio),
                summary.get("win_rate", 0.0),
                ulcer_index(portfolio),
                ulcer_performance_index(portfolio),
                probabilistic_sharpe_ratio(portfolio),
                probabilistic_sortino_ratio(portfolio),
                probabilistic_adjusted_sortino_ratio(portfolio),
                probabilistic_ratio(portfolio),
            ],
        }
    ).set_index("metric")
    if benchmark is not None:
        metric_table.loc["beta", "value"] = summary.get("beta", 0.0)
        metric_table.loc["alpha", "value"] = summary.get("alpha", 0.0)
        metric_table.loc["r2", "value"] = summary.get("r2", 0.0)
        metric_table.loc["information_ratio", "value"] = summary.get("information_ratio", 0.0)
        metric_table.loc["treynor_ratio", "value"] = treynor_ratio(portfolio, benchmark)
        metric_table.loc["r_squared", "value"] = r2(portfolio, benchmark)

    return ReturnAnalysisResult(
        returns=portfolio,
        benchmark_returns=benchmark,
        summary=summary,
        metric_table=metric_table,
        compare_table=compare_table,
        monthly_returns_heatmap=monthly_heatmap,
        drawdown=dd,
        drawdown_details=dd_details,
        monthly_returns=monthly,
        rolling_sharpe=rolling_sh,
        rolling_sortino=rolling_so,
        rolling_volatility=rolling_vol,
        rolling_beta=rolling_b,
        live_start_date=parsed_live_start,
        pre_live_summary=pre_live_summary,
        live_summary=live_summary,
    )


def analyze_positions(
    positions: pd.DataFrame,
    *,
    returns: pd.Series | pd.DataFrame | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    previous_positions: pd.DataFrame | None = None,
    factor_data: Mapping[str, pd.Series | pd.DataFrame | float | int] | None = None,
    industry_column: str = "industry",
    stock_code_column: str = "stock_code",
    weight_column: str = "weight",
    annual_trading_days: int = 252,
) -> PositionAnalysisResult:
    positions_frame = _normalize_positions_frame(positions)
    gross_leverage = _gross_leverage_series(positions_frame)
    concentration: dict[str, float] = {}
    if not positions_frame.empty and {"cash"}.issubset(positions_frame.columns) is False:
        pass
    if not positions_frame.empty and positions_frame.shape[1] > 0:
        abs_weights = positions_frame.drop(columns=["cash"], errors="ignore").abs()
        row_sums = abs_weights.sum(axis=1)
        normalized = abs_weights.div(row_sums.replace(0.0, np.nan), axis=0).fillna(0.0)
        if not normalized.empty:
            top10 = normalized.apply(lambda row: row.sort_values(ascending=False).head(10).sum(), axis=1)
            concentration = {
                "mean_top10_concentration": float(top10.mean()) if not top10.empty else 0.0,
                "max_top10_concentration": float(top10.max()) if not top10.empty else 0.0,
                "mean_gross_leverage": float(gross_leverage.mean()) if not gross_leverage.empty else 0.0,
                "max_gross_leverage": float(gross_leverage.max()) if not gross_leverage.empty else 0.0,
            }

    turnover = None
    if previous_positions is not None:
        turnover = _frame_turnover(positions, previous_positions)

    risk_metrics = None
    benchmark_series = _to_series(benchmark_returns) if benchmark_returns is not None else None
    if returns is not None:
        risk_metrics = calculate_risk_metrics(_to_series(returns), benchmark_returns=benchmark_series, annual_trading_days=annual_trading_days)

    comprehensive: dict[str, Any] = {}
    industry_exposure: dict[str, Any] = {}
    factor_exposure: dict[str, Any] = {}
    latest_holdings = _latest_holdings_long(positions)
    if not positions_frame.empty:
        if returns is None:
            returns_series = pd.Series(dtype=float)
        else:
            returns_series = _to_series(returns)
        if not latest_holdings.empty:
            portfolio_input = latest_holdings.copy()
            portfolio_input["stock_code"] = portfolio_input["stock_code"].astype(str)
            portfolio_input["weight"] = pd.to_numeric(portfolio_input["weight"], errors="coerce").fillna(0.0)
            comprehensive = analyze_portfolio_comprehensive(
                portfolio_input,
                returns_series,
                factor_data=factor_data,
                benchmark_returns=benchmark_series,
                previous_positions=_latest_holdings_long(previous_positions) if previous_positions is not None else None,
                industry_column=industry_column,
                stock_code_column=stock_code_column,
                weight_column=weight_column,
                annual_trading_days=annual_trading_days,
            )
        industry_exposure = comprehensive.get("industry_exposure", {})
        factor_exposure = comprehensive.get("factor_exposure", {})

    return PositionAnalysisResult(
        positions=positions_frame,
        gross_leverage=gross_leverage,
        concentration=concentration,
        turnover=turnover,
        risk_metrics=risk_metrics,
        industry_exposure=industry_exposure,
        factor_exposure=factor_exposure,
        comprehensive=comprehensive,
        latest_holdings=latest_holdings,
    )


def analyze_transactions(
    transactions: pd.DataFrame,
    *,
    positions: pd.DataFrame | None = None,
    returns: pd.Series | pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: Mapping[str, pd.Series | pd.DataFrame | float | int] | None = None,
    capital_base: float = 1_000_000.0,
) -> TransactionAnalysisResult:
    frame = pd.DataFrame(transactions).copy()
    if frame.empty:
        empty = pd.DataFrame(columns=["value"]).rename_axis("metric")
        return TransactionAnalysisResult(
            transactions=frame,
            daily_totals=pd.DataFrame(columns=["txn_volume", "txn_shares"]),
            round_trips=pd.DataFrame(columns=["symbol", "pnl", "open_dt", "close_dt", "duration", "long", "rt_returns", "amount", "open_price", "close_price"]),
            round_trip_summary=empty,
            summary={
                "transaction_count": 0,
                "round_trip_count": 0,
                "gross_transaction_volume": 0.0,
                "net_transaction_volume": 0.0,
            },
        )

    if "dt" in frame.columns:
        frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
        frame = frame.set_index("dt")
    elif "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.set_index("date_")
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"), name="date_")
    frame = frame.loc[~frame.index.isna()].sort_index()
    if "symbol" not in frame.columns and "code" in frame.columns:
        frame = frame.rename(columns={"code": "symbol"})
    if "txn_dollars" not in frame.columns and {"amount", "price"}.issubset(frame.columns):
        frame["txn_dollars"] = -pd.to_numeric(frame["amount"], errors="coerce") * pd.to_numeric(frame["price"], errors="coerce")

    daily_totals = _daily_transaction_totals(frame)
    round_trips = extract_round_trips(frame)
    round_trip_summary = pd.DataFrame(columns=["value"]).rename_axis("metric")
    if not round_trips.empty:
        pnl = pd.to_numeric(round_trips["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rt_returns = pd.to_numeric(round_trips["rt_returns"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        duration_days = pd.to_timedelta(round_trips["duration"], errors="coerce").dt.total_seconds() / 86400.0
        summary = {
            "total_round_trips": float(len(round_trips)),
            "percent_profitable": float((pnl > 0).mean()) if not pnl.empty else 0.0,
            "winning_round_trips": float((pnl > 0).sum()),
            "losing_round_trips": float((pnl < 0).sum()),
            "total_profit": float(pnl.sum()) if not pnl.empty else 0.0,
            "gross_profit": float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0,
            "gross_loss": float(pnl[pnl < 0].sum()) if not pnl.empty else 0.0,
            "profit_factor": float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if not pnl[pnl < 0].empty else np.nan,
            "avg_trade_net_profit": float(pnl.mean()) if not pnl.empty else 0.0,
            "avg_return_all": float(rt_returns.mean()) if not rt_returns.empty else 0.0,
            "avg_duration_days": float(duration_days.mean()) if not duration_days.empty else 0.0,
            "median_duration_days": float(duration_days.median()) if not duration_days.empty else 0.0,
        }
        round_trip_summary = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
        round_trip_summary.index.name = "metric"
    else:
        summary = {
            "total_round_trips": 0.0,
            "percent_profitable": 0.0,
            "winning_round_trips": 0.0,
            "losing_round_trips": 0.0,
            "total_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_net_profit": 0.0,
            "avg_return_all": 0.0,
            "avg_duration_days": 0.0,
            "median_duration_days": 0.0,
        }

    if positions is not None and not frame.empty:
        positions_frame = _normalize_positions_frame(positions)
        if not positions_frame.empty:
            turnover = _frame_turnover(positions_frame.tail(1), positions_frame.head(1))
            summary.update({f"turnover_{key}": float(value) for key, value in turnover.items() if isinstance(value, (int, float, np.floating))})

    summary.update(
        {
            "transaction_count": float(len(frame)),
            "symbol_count": float(frame["symbol"].nunique()) if "symbol" in frame.columns else 0.0,
            "gross_transaction_volume": float(pd.to_numeric(frame.get("txn_dollars"), errors="coerce").abs().sum()) if "txn_dollars" in frame.columns else 0.0,
            "net_transaction_volume": float(pd.to_numeric(frame.get("txn_dollars"), errors="coerce").sum()) if "txn_dollars" in frame.columns else 0.0,
        }
    )

    return TransactionAnalysisResult(
        transactions=frame,
        daily_totals=daily_totals,
        round_trips=round_trips,
        round_trip_summary=round_trip_summary,
        summary=summary,
    )


def analyze_multifactor(
    *,
    returns: pd.Series | pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: Mapping[str, pd.Series | pd.DataFrame | float | int] | None = None,
    previous_positions: pd.DataFrame | None = None,
    live_start_date: str | pd.Timestamp | None = None,
    annual_trading_days: int = 252,
    rolling_window: int = 126,
) -> MultifactorAnalysisResult:
    return_result = None
    position_result = None
    transaction_result = None
    if returns is not None:
        return_result = analyze_returns(
            returns,
            benchmark_returns=benchmark_returns,
            live_start_date=live_start_date,
            annual_trading_days=annual_trading_days,
            rolling_window=rolling_window,
        )
    if positions is not None:
        position_result = analyze_positions(
            positions,
            returns=returns,
            benchmark_returns=benchmark_returns,
            previous_positions=previous_positions,
            factor_data=factor_data,
            annual_trading_days=annual_trading_days,
        )
    if transactions is not None:
        transaction_result = analyze_transactions(
            transactions,
            positions=positions,
            returns=returns,
            market_data=market_data,
            close_panel=close_panel,
            factor_data=factor_data,
        )
    return MultifactorAnalysisResult(
        returns=return_result,
        positions=position_result,
        transactions=transaction_result,
    )


__all__ = [
    "best",
    "calmar",
    "cagr",
    "common_sense_ratio",
    "consecutive_losses",
    "consecutive_wins",
    "cpc_index",
    "MultifactorAnalysisResult",
    "PositionAnalysisResult",
    "ReturnAnalysisResult",
    "TransactionAnalysisResult",
    "analyze_multifactor",
    "analyze_positions",
    "analyze_returns",
    "analyze_transactions",
    "compare_returns",
    "expected_shortfall",
    "gain_to_pain_ratio",
    "drawdown_details",
    "information_ratio",
    "kelly_criterion",
    "omega",
    "payoff_ratio",
    "probabilistic_adjusted_sortino_ratio",
    "probabilistic_ratio",
    "probabilistic_sharpe_ratio",
    "probabilistic_sortino_ratio",
    "profit_factor",
    "profit_ratio",
    "recovery_factor",
    "r2",
    "risk_return_ratio",
    "serenity_index",
    "monthly_returns_heatmap",
    "monthly_returns",
    "cvar",
    "exposure",
    "kurtosis",
    "quantstats_max_drawdown",
    "smart_sharpe",
    "smart_sortino",
    "rolling_beta",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_volatility",
    "skew",
    "tail_ratio",
    "treynor_ratio",
    "ulcer_index",
    "ulcer_performance_index",
    "var",
    "value_at_risk",
    "win_loss_ratio",
    "worst",
    "calculate_risk_metrics",
    "calculate_turnover",
    "analyze_portfolio_comprehensive",
    "max_drawdown",
]
