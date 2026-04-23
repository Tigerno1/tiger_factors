from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation.common.matplotlib_config import configure_matplotlib
from tiger_factors.multifactor_evaluation.common.parquet_utils import to_parquet_clean
from tiger_factors.multifactor_evaluation.common.plotting import finalize_quantstats_axis
from tiger_factors.multifactor_evaluation.common.plotting import save_quantstats_figure
from tiger_factors.multifactor_evaluation.common.results import TearSheetResultMixin
from tiger_factors.multifactor_evaluation.reporting.html_report import render_trade_report_html

configure_matplotlib()

import matplotlib.pyplot as plt
import seaborn as sns

from tiger_factors.multifactor_evaluation.portfolio_analysis import analyze_portfolio_comprehensive


@dataclass(frozen=True)
class PortfolioTradeAnalysisResult(TearSheetResultMixin):
    output_dir: Path
    figure_paths: list[Path]
    transactions_path: Path | None
    round_trips_path: Path | None
    capacity_summary_path: Path | None
    factor_attribution_path: Path | None
    transaction_summary_path: Path | None
    round_trip_summary_path: Path | None
    report_name: str

    def _table_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        if self.transactions_path is not None:
            paths["transactions"] = self.transactions_path
        if self.round_trips_path is not None:
            paths["round_trips"] = self.round_trips_path
        if self.capacity_summary_path is not None:
            paths["capacity_summary"] = self.capacity_summary_path
        if self.factor_attribution_path is not None:
            paths["factor_attribution"] = self.factor_attribution_path
        if self.transaction_summary_path is not None:
            paths["summary"] = self.transaction_summary_path
            paths["transaction_summary"] = self.transaction_summary_path
        if self.round_trip_summary_path is not None:
            paths["round_trip_summary"] = self.round_trip_summary_path
        return paths

    def report(self) -> str | None:
        report_path = self.output_dir / f"{self.report_name}_trade_report.html"
        return report_path.stem if report_path.exists() else None

    def preferred_table_order(self) -> list[str]:
        return [
            "summary",
            "transaction_summary",
            "transactions",
            "round_trip_summary",
            "round_trips",
            "capacity_summary",
            "factor_attribution",
        ]

    def get_report(self, *, open_browser: bool = True) -> Path:
        tables = {}
        for name in self.ordered_tables():
            tables[name] = self.get_table(name)
        html_path = render_trade_report_html(
            output_dir=self.output_dir,
            report_name=f"{self.report_name}_trade_report",
            tables=tables,
            figure_paths=self.figure_paths,
            open_browser=open_browser,
            subtitle=f"Trade analysis for {self.report_name}",
        )
        return html_path

    def to_summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["figure_paths"] = [str(path) for path in self.figure_paths]
        payload["transactions_path"] = str(self.transactions_path) if self.transactions_path is not None else None
        payload["round_trips_path"] = str(self.round_trips_path) if self.round_trips_path is not None else None
        payload["capacity_summary_path"] = (
            str(self.capacity_summary_path) if self.capacity_summary_path is not None else None
        )
        payload["factor_attribution_path"] = (
            str(self.factor_attribution_path) if self.factor_attribution_path is not None else None
        )
        payload["transaction_summary_path"] = (
            str(self.transaction_summary_path) if self.transaction_summary_path is not None else None
        )
        payload["round_trip_summary_path"] = (
            str(self.round_trip_summary_path) if self.round_trip_summary_path is not None else None
        )
        return payload


def _safe_numeric(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        cleaned = pd.to_numeric(values, errors="coerce")
    else:
        cleaned = pd.DataFrame(values).apply(lambda col: pd.to_numeric(col, errors="coerce"))
    return cleaned.replace([np.inf, -np.inf], np.nan)


def _ensure_datetime_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    obj = frame.copy()
    obj.index = pd.DatetimeIndex(pd.to_datetime(obj.index, errors="coerce"))
    obj = obj.loc[~obj.index.isna()].sort_index()
    return obj


def _normalize_positions_for_trading(positions: pd.DataFrame | None) -> pd.DataFrame | None:
    if positions is None:
        return None
    frame = pd.DataFrame(positions).copy()
    if frame.empty:
        return frame
    if "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_"]).set_index("date_")
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"), name="date_")
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame.columns = frame.columns.astype(str)
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "cash" not in numeric.columns:
        asset_cols = [column for column in numeric.columns if column != "cash"]
        numeric["cash"] = 1.0 - numeric[asset_cols].sum(axis=1)
    return numeric.sort_index()


def _normalize_close_panel(close_panel: pd.DataFrame | None) -> pd.DataFrame | None:
    if close_panel is None:
        return None
    frame = pd.DataFrame(close_panel).copy()
    if frame.empty:
        return frame
    frame = _ensure_datetime_index(frame)
    frame.columns = frame.columns.astype(str)
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return frame


def _normalize_transactions_frame(transactions: pd.DataFrame | None) -> pd.DataFrame | None:
    if transactions is None:
        return None
    frame = pd.DataFrame(transactions).copy()
    if frame.empty:
        return frame
    if "symbol" not in frame.columns and "code" in frame.columns:
        frame = frame.rename(columns={"code": "symbol"})
    if "dt" in frame.columns:
        frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
        frame = frame.dropna(subset=["dt"]).set_index("dt")
    elif "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_"]).set_index("date_")
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"), name="dt")
    frame = frame.loc[~frame.index.isna()].sort_index()
    if "symbol" not in frame.columns or "amount" not in frame.columns or "price" not in frame.columns:
        raise ValueError("transactions must contain symbol, amount and price columns.")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if "commission" in frame.columns:
        frame["commission"] = pd.to_numeric(frame["commission"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["amount", "price"])
    if "txn_dollars" not in frame.columns:
        frame["txn_dollars"] = -frame["amount"] * frame["price"]
    else:
        frame["txn_dollars"] = pd.to_numeric(frame["txn_dollars"], errors="coerce").fillna(-frame["amount"] * frame["price"])
    return frame.sort_index()


def synthesize_transactions_from_positions(
    positions: pd.DataFrame,
    close_panel: pd.DataFrame | None,
    *,
    capital_base: float = 1_000_000.0,
) -> pd.DataFrame:
    positions_frame = _normalize_positions_for_trading(positions)
    close_frame = _normalize_close_panel(close_panel)
    if positions_frame is None or positions_frame.empty:
        return pd.DataFrame(columns=["symbol", "amount", "price", "txn_dollars"])
    asset_cols = [column for column in positions_frame.columns if column != "cash"]
    if not asset_cols:
        return pd.DataFrame(columns=["symbol", "amount", "price", "txn_dollars"])

    if close_frame is not None and not close_frame.empty:
        common_index = positions_frame.index.intersection(close_frame.index)
        common_columns = [column for column in asset_cols if column in close_frame.columns]
        if common_index.empty or not common_columns:
            return pd.DataFrame(columns=["symbol", "amount", "price", "txn_dollars"])
        positions_frame = positions_frame.reindex(index=common_index, columns=common_columns + (["cash"] if "cash" in positions_frame.columns else [])).fillna(0.0)
        close_frame = close_frame.reindex(index=common_index, columns=common_columns).ffill().bfill()
        asset_cols = common_columns
    else:
        close_frame = pd.DataFrame(1.0, index=positions_frame.index, columns=asset_cols)

    dollar_positions = positions_frame[asset_cols].multiply(float(capital_base))
    shares = dollar_positions.div(close_frame.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    txn_shares = shares.diff()
    if not txn_shares.empty:
        txn_shares.iloc[0] = shares.iloc[0]

    records: list[dict[str, Any]] = []
    for dt, row in txn_shares.iterrows():
        for symbol, amount in row.items():
            amount = float(amount)
            if abs(amount) <= 1e-12:
                continue
            price = float(close_frame.loc[dt, symbol]) if symbol in close_frame.columns else 1.0
            records.append(
                {
                    "dt": dt,
                    "symbol": symbol,
                    "amount": amount,
                    "price": price,
                    "txn_dollars": -amount * price,
                }
            )

    if not records:
        return pd.DataFrame(columns=["symbol", "amount", "price", "txn_dollars"])
    txn_frame = pd.DataFrame.from_records(records).set_index("dt").sort_index()
    txn_frame.index = pd.DatetimeIndex(txn_frame.index, name="dt")
    txn_frame["symbol"] = txn_frame["symbol"].astype(str)
    return txn_frame


def _daily_transaction_totals(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["txn_volume", "txn_shares"])
    frame = transactions.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="dt")
    daily = pd.DataFrame(
        {
            "txn_volume": frame["txn_dollars"].abs().groupby(frame.index.normalize()).sum(),
            "txn_shares": frame["amount"].abs().groupby(frame.index.normalize()).sum(),
        }
    )
    daily.index.name = "date_"
    return daily.sort_index()


def _transaction_summary(transactions: pd.DataFrame, positions: pd.DataFrame | None = None, *, capital_base: float = 1_000_000.0) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["value"]).rename_axis("metric")
    daily = _daily_transaction_totals(transactions)
    summary = {
        "txn_count": float(len(transactions)),
        "txn_volume_total": float(transactions["txn_dollars"].abs().sum()),
        "txn_volume_mean": float(transactions["txn_dollars"].abs().mean()),
        "txn_volume_median": float(transactions["txn_dollars"].abs().median()),
        "txn_volume_max": float(transactions["txn_dollars"].abs().max()),
        "txn_shares_total": float(transactions["amount"].abs().sum()),
        "buy_count": float((transactions["amount"] > 0).sum()),
        "sell_count": float((transactions["amount"] < 0).sum()),
        "daily_txn_volume_mean": float(daily["txn_volume"].mean()) if not daily.empty else 0.0,
        "daily_txn_volume_max": float(daily["txn_volume"].max()) if not daily.empty else 0.0,
    }
    if positions is not None and not positions.empty:
        pos = _normalize_positions_for_trading(positions)
        if pos is not None and not pos.empty:
            gross_book = pos.drop(columns=["cash"], errors="ignore").abs().sum(axis=1) * float(capital_base)
            turnover = daily["txn_volume"].reindex(gross_book.index, fill_value=0.0).div(gross_book.rolling(2).mean().fillna(gross_book / 2).replace(0.0, np.nan))
            summary["turnover_mean"] = float(turnover.dropna().mean()) if not turnover.dropna().empty else 0.0
            summary["turnover_max"] = float(turnover.dropna().max()) if not turnover.dropna().empty else 0.0
    table = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
    table.index.name = "metric"
    return table


def _daily_txn_by_symbol(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["symbol", "amount", "txn_volume", "date_"])
    frame = transactions.copy()
    frame["date_"] = pd.DatetimeIndex(frame.index).normalize()
    grouped = (
        frame.assign(abs_amount=frame["amount"].abs(), abs_value=frame["txn_dollars"].abs())
        .groupby(["date_", "symbol"], as_index=False)
        .agg(amount=("abs_amount", "sum"), txn_volume=("abs_value", "sum"), price=("price", "last"))
    )
    return grouped.sort_values(["date_", "symbol"], ignore_index=True)


def _plot_transaction_volume(transactions: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    daily = _daily_transaction_totals(transactions)
    if daily.empty:
        return None
    plt.figure(figsize=(12, 4))
    daily["txn_volume"].plot(kind="bar", color="#2a6fdb")
    path = output_dir / f"{report_name}_txn_daily_volume.png"
    finalize_quantstats_axis(plt.gca(), title="Daily transaction volume", ylabel="Txn volume")
    save_quantstats_figure(path)
    return path


def _plot_transaction_time_hist(transactions: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    if transactions.empty:
        return None
    hours = pd.DatetimeIndex(transactions.index).hour + pd.DatetimeIndex(transactions.index).minute / 60.0
    if len(hours) == 0:
        return None
    plt.figure(figsize=(10, 4))
    sns.histplot(hours, bins=24, color="#6c757d")
    finalize_quantstats_axis(plt.gca(), title="Transaction time histogram", xlabel="Hour of day")
    path = output_dir / f"{report_name}_txn_time_hist.png"
    save_quantstats_figure(path)
    return path


def _plot_transaction_turnover(transactions: pd.DataFrame, positions: pd.DataFrame | None, *, output_dir: Path, report_name: str) -> Path | None:
    if transactions.empty or positions is None or positions.empty:
        return None
    pos = _normalize_positions_for_trading(positions)
    if pos is None or pos.empty:
        return None
    daily = _daily_transaction_totals(transactions)
    gross_book = pos.drop(columns=["cash"], errors="ignore").abs().sum(axis=1)
    denom = gross_book.rolling(2).mean().fillna(gross_book / 2).replace(0.0, np.nan)
    turnover = daily["txn_volume"].reindex(gross_book.index, fill_value=0.0).div(denom)
    turnover = turnover.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if turnover.empty:
        return None
    plt.figure(figsize=(12, 4))
    turnover.plot(color="#1f7a8c", linewidth=1.8)
    plt.axhline(turnover.mean(), color="#c44536", linestyle="--", linewidth=1.0, alpha=0.8, label="mean")
    path = output_dir / f"{report_name}_turnover.png"
    finalize_quantstats_axis(plt.gca(), title="Daily turnover", ylabel="Turnover", legend=True)
    save_quantstats_figure(path)
    return path


def _annualized_return(series: pd.Series, annualization: int = 252) -> float:
    clean = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    safe = clean.clip(lower=-0.999999, upper=1e6)
    log_growth = float(np.log1p(safe).sum())
    annual_log = log_growth * (annualization / max(len(clean), 1))
    if not np.isfinite(annual_log):
        return 0.0
    if annual_log > 700:
        return float("inf")
    if annual_log < -700:
        return -1.0
    return float(np.exp(annual_log) - 1.0)


def _slippage_adjusted_returns(
    returns: pd.Series,
    transactions: pd.DataFrame,
    *,
    capital_base: float = 1_000_000.0,
    slippage_bps: float = 10.0,
) -> pd.Series:
    clean_returns = pd.to_numeric(pd.Series(returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean_returns.empty or transactions.empty:
        return clean_returns
    portfolio_value = float(capital_base) * (1.0 + clean_returns.fillna(0.0)).cumprod()
    daily_volume = _daily_transaction_totals(transactions)["txn_volume"].reindex(clean_returns.index, fill_value=0.0)
    penalty = daily_volume * (float(slippage_bps) / 10000.0)
    adjusted = clean_returns.copy()
    adjusted = adjusted - (penalty / portfolio_value.replace(0.0, np.nan)).fillna(0.0)
    return adjusted.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _plot_slippage_sweep(
    returns: pd.Series,
    transactions: pd.DataFrame,
    *,
    output_dir: Path,
    report_name: str,
    capital_base: float = 1_000_000.0,
) -> Path | None:
    if transactions.empty:
        return None
    sweep = np.array([0, 5, 10, 20, 50, 100], dtype=float)
    rows = []
    for bps in sweep:
        adjusted = _slippage_adjusted_returns(returns, transactions, capital_base=capital_base, slippage_bps=float(bps))
        annual_return = float((1.0 + adjusted.fillna(0.0)).prod() ** (252 / max(len(adjusted), 1)) - 1.0) if not adjusted.empty else 0.0
        volatility = float(adjusted.std(ddof=0) * np.sqrt(252)) if not adjusted.empty else 0.0
        sharpe = float(annual_return / volatility) if volatility > 1e-12 else 0.0
        rows.append({"slippage_bps": float(bps), "annual_return": annual_return, "volatility": volatility, "sharpe": sharpe})
    frame = pd.DataFrame.from_records(rows)
    plt.figure(figsize=(10, 4))
    frame.set_index("slippage_bps")["annual_return"].plot(marker="o", color="#2a6fdb")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    path = output_dir / f"{report_name}_slippage_sweep.png"
    finalize_quantstats_axis(plt.gca(), title="Slippage sweep", xlabel="Slippage (bps)", ylabel="Annual return")
    save_quantstats_figure(path)
    frame.to_csv(output_dir / f"{report_name}_slippage_sweep.csv", index=False)
    return path


@dataclass(frozen=True)
class _OpenLot:
    qty: float
    price: float
    open_dt: pd.Timestamp
    direction: int


def extract_round_trips(transactions: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_transactions_frame(transactions)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "pnl", "open_dt", "close_dt", "duration", "long", "rt_returns", "amount", "open_price", "close_price"])

    roundtrips: list[dict[str, Any]] = []
    tol = 1e-12
    for symbol, symbol_txns in frame.groupby("symbol"):
        open_lots: deque[_OpenLot] = deque()
        for dt, row in symbol_txns.sort_index().iterrows():
            amount = float(row["amount"])
            if abs(amount) <= tol or not np.isfinite(amount):
                continue
            price = float(row["price"])
            direction = 1 if amount > 0 else -1
            remaining = abs(amount)
            while remaining > tol:
                if open_lots and open_lots[0].direction != direction:
                    lot = open_lots[0]
                    matched = min(remaining, lot.qty)
                    pnl = matched * (price - lot.price) if lot.direction > 0 else matched * (lot.price - price)
                    invested = matched * abs(lot.price)
                    roundtrips.append(
                        {
                            "symbol": symbol,
                            "pnl": float(pnl),
                            "open_dt": lot.open_dt,
                            "close_dt": dt,
                            "duration": dt - lot.open_dt,
                            "long": bool(lot.direction > 0),
                            "rt_returns": float(pnl / invested) if invested > tol else np.nan,
                            "amount": float(matched),
                            "open_price": float(lot.price),
                            "close_price": float(price),
                        }
                    )
                    lot = _OpenLot(qty=lot.qty - matched, price=lot.price, open_dt=lot.open_dt, direction=lot.direction)
                    open_lots.popleft()
                    if lot.qty > tol:
                        open_lots.appendleft(lot)
                    remaining -= matched
                else:
                    open_lots.append(_OpenLot(qty=remaining, price=price, open_dt=dt, direction=direction))
                    remaining = 0.0
    return pd.DataFrame.from_records(roundtrips)


def _round_trip_summary(round_trips: pd.DataFrame) -> pd.DataFrame:
    if round_trips.empty:
        return pd.DataFrame(columns=["value"]).rename_axis("metric")
    pnl = pd.to_numeric(round_trips["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    returns = pd.to_numeric(round_trips["rt_returns"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    duration_days = pd.to_timedelta(round_trips["duration"], errors="coerce").dt.total_seconds() / 86400.0
    summary = {
        "total_round_trips": float(len(round_trips)),
        "percent_profitable": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        "winning_round_trips": float((pnl > 0).sum()),
        "losing_round_trips": float((pnl < 0).sum()),
        "even_round_trips": float((pnl == 0).sum()),
        "total_profit": float(pnl.sum()) if not pnl.empty else 0.0,
        "gross_profit": float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0,
        "gross_loss": float(pnl[pnl < 0].sum()) if not pnl.empty else 0.0,
        "profit_factor": float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if not pnl[pnl < 0].empty else np.nan,
        "avg_trade_net_profit": float(pnl.mean()) if not pnl.empty else 0.0,
        "avg_winning_trade": float(pnl[pnl > 0].mean()) if not pnl[pnl > 0].empty else 0.0,
        "avg_losing_trade": float(pnl[pnl < 0].mean()) if not pnl[pnl < 0].empty else 0.0,
        "avg_return_all": float(returns.mean()) if not returns.empty else 0.0,
        "median_return_all": float(returns.median()) if not returns.empty else 0.0,
        "avg_duration_days": float(duration_days.mean()) if not duration_days.empty else 0.0,
        "median_duration_days": float(duration_days.median()) if not duration_days.empty else 0.0,
        "longest_duration_days": float(duration_days.max()) if not duration_days.empty else 0.0,
        "shortest_duration_days": float(duration_days.min()) if not duration_days.empty else 0.0,
    }
    table = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
    table.index.name = "metric"
    return table


def _plot_round_trip_lifetimes(round_trips: pd.DataFrame, *, output_dir: Path, report_name: str, top_n: int = 20) -> Path | None:
    if round_trips.empty:
        return None
    sample = round_trips.sort_values("pnl", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    if sample.empty:
        return None
    sample["open_dt"] = pd.to_datetime(sample["open_dt"], errors="coerce")
    sample["close_dt"] = pd.to_datetime(sample["close_dt"], errors="coerce")
    plt.figure(figsize=(12, max(4, 0.35 * len(sample))))
    for i, (_, row) in enumerate(sample.iterrows()):
        color = "#2a6fdb" if row["pnl"] >= 0 else "#c44536"
        plt.hlines(i, row["open_dt"], row["close_dt"], color=color, linewidth=2.0)
        plt.plot([row["open_dt"], row["close_dt"]], [i, i], marker="o", color=color, markersize=4)
    plt.yticks(range(len(sample)), sample["symbol"].astype(str).tolist())
    path = output_dir / f"{report_name}_round_trip_lifetimes.png"
    finalize_quantstats_axis(plt.gca(), title="Round-trip lifetimes", xlabel="Date")
    save_quantstats_figure(path)
    return path


def _plot_round_trip_pnl(round_trips: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    if round_trips.empty:
        return None
    pnl = pd.to_numeric(round_trips["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    rt_returns = pd.to_numeric(round_trips["rt_returns"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if pnl.empty and rt_returns.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not pnl.empty:
        sns.histplot(pnl, ax=axes[0], bins=20, color="#2a6fdb", kde=True)
        finalize_quantstats_axis(axes[0], title="PnL per round-trip", xlabel="PnL")
    if not rt_returns.empty:
        sns.histplot(rt_returns * 100.0, ax=axes[1], bins=20, color="#6c757d", kde=True)
        finalize_quantstats_axis(axes[1], title="Round-trip returns", xlabel="Return (%)")
    path = output_dir / f"{report_name}_round_trip_pnl.png"
    save_quantstats_figure(path)
    return path


def _plot_round_trip_profit_attribution(round_trips: pd.DataFrame, *, output_dir: Path, report_name: str, top_n: int = 15) -> Path | None:
    if round_trips.empty:
        return None
    attribution = round_trips.groupby("symbol")["pnl"].sum().sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
    if attribution.empty:
        return None
    plt.figure(figsize=(10, max(4, 0.35 * len(attribution))))
    attribution.sort_values().plot(kind="barh", color="#2a6fdb")
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    path = output_dir / f"{report_name}_round_trip_profit_attribution.png"
    finalize_quantstats_axis(plt.gca(), title="Profit attribution", xlabel="PnL")
    save_quantstats_figure(path)
    return path


def _plot_prob_profit_trade(round_trips: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    if round_trips.empty:
        return None
    profitable = pd.to_numeric(round_trips["pnl"], errors="coerce") > 0
    counts = pd.Series({"profitable": int(profitable.sum()), "unprofitable": int((~profitable).sum())})
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar", color=["#2a6fdb", "#c44536"])
    path = output_dir / f"{report_name}_round_trip_profitable.png"
    finalize_quantstats_axis(plt.gca(), title="Probability of profitable trade", ylabel="Count")
    save_quantstats_figure(path)
    return path


def _market_data_to_long(market_data: pd.DataFrame | None) -> pd.DataFrame | None:
    if market_data is None:
        return None
    frame = pd.DataFrame(market_data).copy()
    if frame.empty:
        return frame
    if {"date_", "symbol", "price", "volume"}.issubset(frame.columns):
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_", "symbol"])
        return frame[["date_", "symbol", "price", "volume"]].copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if {"price", "volume"}.issubset(set(map(str, frame.columns.get_level_values(0)))):
            price = frame.xs("price", axis=1, level=0)
            volume = frame.xs("volume", axis=1, level=0)
            price.index = pd.DatetimeIndex(pd.to_datetime(price.index, errors="coerce"), name="date_")
            volume.index = pd.DatetimeIndex(pd.to_datetime(volume.index, errors="coerce"), name="date_")
            price = price.reset_index().melt(id_vars="date_", var_name="symbol", value_name="price")
            volume = volume.reset_index().melt(id_vars="date_", var_name="symbol", value_name="volume")
            merged = price.merge(volume, on=["date_", "symbol"], how="inner")
            return merged.dropna(subset=["date_", "symbol"])
    if isinstance(frame.index, pd.MultiIndex):
        levels = [str(level) for level in frame.index.get_level_values(-1).unique()]
        if "price" in levels and "volume" in levels:
            price = frame.xs("price", level=-1)
            volume = frame.xs("volume", level=-1)
            price.index = pd.DatetimeIndex(pd.to_datetime(price.index, errors="coerce"), name="date_")
            volume.index = pd.DatetimeIndex(pd.to_datetime(volume.index, errors="coerce"), name="date_")
            price = price.reset_index().melt(id_vars="date_", var_name="symbol", value_name="price")
            volume = volume.reset_index().melt(id_vars="date_", var_name="symbol", value_name="volume")
            return price.merge(volume, on=["date_", "symbol"], how="inner").dropna(subset=["date_", "symbol"])
    return None


def _capacity_daily_transactions(transactions: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    daily_txns = _daily_txn_by_symbol(transactions)
    if daily_txns.empty:
        return pd.DataFrame(columns=["date_", "symbol", "amount", "price", "volume", "max_pct_bar_consumed"])
    market_long = _market_data_to_long(market_data)
    if market_long is None or market_long.empty:
        return pd.DataFrame(columns=["date_", "symbol", "amount", "price", "volume", "max_pct_bar_consumed"])
    market_long["date_"] = pd.to_datetime(market_long["date_"], errors="coerce")
    market_long["symbol"] = market_long["symbol"].astype(str)
    merged = daily_txns.merge(market_long, on=["date_", "symbol"], how="inner", suffixes=("", "_bar"))
    if merged.empty:
        return pd.DataFrame(columns=["date_", "symbol", "amount", "price", "volume", "max_pct_bar_consumed"])
    merged["max_pct_bar_consumed"] = merged["amount"].div(merged["volume"].replace(0.0, np.nan)).mul(100.0)
    return merged.sort_values(["date_", "symbol"], ignore_index=True)


def _days_to_liquidate_positions(
    positions: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    max_bar_consumption: float = 0.2,
    capital_base: float = 1_000_000.0,
    mean_volume_window: int = 5,
) -> pd.DataFrame:
    pos = _normalize_positions_for_trading(positions)
    market_long = _market_data_to_long(market_data)
    if pos is None or pos.empty or market_long is None or market_long.empty:
        return pd.DataFrame()
    market_long["date_"] = pd.to_datetime(market_long["date_"], errors="coerce")
    market_long["symbol"] = market_long["symbol"].astype(str)
    market_long["dollar_volume"] = pd.to_numeric(market_long["price"], errors="coerce") * pd.to_numeric(market_long["volume"], errors="coerce")
    dollar_volume = market_long.pivot_table(index="date_", columns="symbol", values="dollar_volume", aggfunc="mean").sort_index()
    dollar_volume = dollar_volume.reindex(index=pos.index).ffill()
    roll_mean_dv = dollar_volume.rolling(window=mean_volume_window, min_periods=1).mean().shift(1)
    roll_mean_dv = roll_mean_dv.replace(0.0, np.nan)
    positions_alloc = pos.drop(columns=["cash"], errors="ignore").abs()
    if positions_alloc.empty:
        return pd.DataFrame()
    days = (positions_alloc * float(capital_base)) / (float(max_bar_consumption) * roll_mean_dv)
    return days.iloc[mean_volume_window:].replace([np.inf, -np.inf], np.nan)


def _max_days_to_liquidate_by_ticker(days_to_liquidate: pd.DataFrame, positions: pd.DataFrame | None = None) -> pd.DataFrame:
    if days_to_liquidate.empty:
        return pd.DataFrame(columns=["days_to_liquidate", "date_", "position_alloc"])
    worst = pd.DataFrame()
    worst["days_to_liquidate"] = days_to_liquidate.max(axis=0)
    if positions is not None:
        pos = _normalize_positions_for_trading(positions)
        if pos is not None and not pos.empty:
            pos_alloc = pos.drop(columns=["cash"], errors="ignore").abs()
            for symbol in worst.index:
                if symbol in pos_alloc.columns:
                    idx = days_to_liquidate[symbol].idxmax()
                    worst.loc[symbol, "date_"] = idx
                    worst.loc[symbol, "position_alloc"] = float(pos_alloc.loc[idx, symbol]) if idx in pos_alloc.index else np.nan
    return worst


def _get_low_liquidity_transactions(capacity_daily: pd.DataFrame) -> pd.DataFrame:
    if capacity_daily.empty:
        return pd.DataFrame(columns=["date_", "max_pct_bar_consumed"])
    max_bar = capacity_daily.sort_values("max_pct_bar_consumed", ascending=False).groupby("symbol").first().reset_index()
    return max_bar[["symbol", "date_", "max_pct_bar_consumed"]].sort_values("max_pct_bar_consumed", ascending=False)


def _apply_slippage_penalty(
    returns: pd.Series,
    txn_daily: pd.DataFrame,
    *,
    simulate_starting_capital: float,
    backtest_starting_capital: float,
    impact: float = 0.1,
) -> pd.Series:
    clean = pd.to_numeric(pd.Series(returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or txn_daily.empty:
        return clean
    mult = float(simulate_starting_capital) / float(backtest_starting_capital)
    txn_daily = txn_daily.copy()
    txn_daily.index = pd.DatetimeIndex(pd.to_datetime(txn_daily["date_"], errors="coerce"))
    simulate_traded_shares = abs(mult * txn_daily["amount"])
    simulate_traded_dollars = txn_daily["price"] * simulate_traded_shares
    simulate_pct_volume_used = simulate_traded_shares / txn_daily["volume"].replace(0.0, np.nan)
    penalties = simulate_pct_volume_used.pow(2) * float(impact) * simulate_traded_dollars
    daily_penalty = penalties.groupby(penalties.index.normalize()).sum()
    daily_penalty = daily_penalty.reindex(clean.index).fillna(0.0)
    portfolio_value = (1.0 + clean).cumprod() * float(backtest_starting_capital)
    return clean - (daily_penalty / portfolio_value.replace(0.0, np.nan)).fillna(0.0)


def _plot_capacity_sweep(
    returns: pd.Series,
    txn_daily: pd.DataFrame,
    *,
    output_dir: Path,
    report_name: str,
    backtest_starting_capital: float = 1_000_000.0,
) -> Path | None:
    if txn_daily.empty:
        return None
    capitals = np.logspace(5, 9, 10)
    rows = []
    for capital in capitals:
        adjusted = _apply_slippage_penalty(
            returns,
            txn_daily,
            simulate_starting_capital=float(capital),
            backtest_starting_capital=backtest_starting_capital,
        )
        annual_return = _annualized_return(adjusted)
        annual_vol = float(adjusted.std(ddof=0) * np.sqrt(252)) if not adjusted.empty else 0.0
        rows.append(
            {
                "capital": float(capital),
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "sharpe": float(annual_return / annual_vol) if annual_vol > 1e-12 else 0.0,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    plt.figure(figsize=(10, 4))
    plt.plot(frame["capital"], frame["annual_return"], marker="o", color="#2a6fdb")
    plt.xscale("log")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    path = output_dir / f"{report_name}_capacity_sweep.png"
    finalize_quantstats_axis(plt.gca(), title="Capacity sweep", xlabel="Simulated starting capital", ylabel="Annual return")
    plt.xscale("log")
    save_quantstats_figure(path)
    frame.to_csv(output_dir / f"{report_name}_capacity_sweep.csv", index=False)
    return path


def create_trade_report(
    returns: pd.Series,
    *,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    output_dir: str | Path,
    report_name: str = "portfolio",
    capital_base: float = 1_000_000.0,
) -> PortfolioTradeAnalysisResult | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions_frame = _normalize_positions_for_trading(positions)
    close_frame = _normalize_close_panel(close_panel)
    txn_frame = _normalize_transactions_frame(transactions)
    if txn_frame is None and positions_frame is not None and not positions_frame.empty and close_frame is not None:
        txn_frame = synthesize_transactions_from_positions(positions_frame, close_frame, capital_base=capital_base)
    if txn_frame is None or txn_frame.empty:
        return None

    stem = f"{report_name}_" if report_name else ""
    figure_paths: list[Path] = []
    transactions_path = output_dir / f"{stem}transactions.parquet"
    to_parquet_clean(txn_frame, transactions_path)
    transaction_summary = _transaction_summary(txn_frame, positions_frame, capital_base=capital_base)
    transaction_summary_path = output_dir / f"{stem}transaction_summary.csv"
    transaction_summary.to_csv(transaction_summary_path)

    for maybe_path in (
        _plot_transaction_volume(txn_frame, output_dir=output_dir, report_name=report_name),
        _plot_transaction_time_hist(txn_frame, output_dir=output_dir, report_name=report_name),
        _plot_transaction_turnover(txn_frame, positions_frame, output_dir=output_dir, report_name=report_name),
        _plot_slippage_sweep(returns, txn_frame, output_dir=output_dir, report_name=report_name, capital_base=capital_base),
    ):
        if maybe_path is not None:
            figure_paths.append(maybe_path)

    round_trips = extract_round_trips(txn_frame)
    round_trips_path = None
    round_trip_summary_path = None
    if not round_trips.empty:
        round_trips_path = output_dir / f"{stem}round_trips.parquet"
        to_parquet_clean(round_trips, round_trips_path)
        round_trip_summary = _round_trip_summary(round_trips)
        round_trip_summary_path = output_dir / f"{stem}round_trip_summary.csv"
        round_trip_summary.to_csv(round_trip_summary_path)
        for maybe_path in (
            _plot_round_trip_lifetimes(round_trips, output_dir=output_dir, report_name=report_name),
            _plot_round_trip_pnl(round_trips, output_dir=output_dir, report_name=report_name),
            _plot_round_trip_profit_attribution(round_trips, output_dir=output_dir, report_name=report_name),
            _plot_prob_profit_trade(round_trips, output_dir=output_dir, report_name=report_name),
        ):
            if maybe_path is not None:
                figure_paths.append(maybe_path)

    capacity_summary_path = None
    factor_attribution_path = None
    if market_data is not None:
        capacity_daily = _capacity_daily_transactions(txn_frame, market_data)
        if not capacity_daily.empty:
            capacity_summary = _get_low_liquidity_transactions(capacity_daily)
            capacity_summary_path = output_dir / f"{stem}capacity_summary.csv"
            capacity_summary.to_csv(capacity_summary_path, index=False)
            days_to_liquidate = _days_to_liquidate_positions(
                positions_frame if positions_frame is not None else pd.DataFrame(),
                market_data,
                capital_base=capital_base,
            )
            if not days_to_liquidate.empty:
                days_path = output_dir / f"{stem}days_to_liquidate.parquet"
                to_parquet_clean(days_to_liquidate, days_path)
                plot_table = _max_days_to_liquidate_by_ticker(days_to_liquidate, positions_frame)
                if not plot_table.empty:
                    plt.figure(figsize=(10, max(4, 0.35 * len(plot_table))))
                    plot_table["days_to_liquidate"].sort_values().plot(kind="barh", color="#c44536")
                    path = output_dir / f"{report_name}_days_to_liquidate.png"
                    finalize_quantstats_axis(plt.gca(), title="Max days to liquidate by ticker", xlabel="Days")
                    save_quantstats_figure(path)
                    figure_paths.append(path)
            maybe_path = _plot_capacity_sweep(returns, capacity_daily, output_dir=output_dir, report_name=report_name, backtest_starting_capital=capital_base)
            if maybe_path is not None:
                figure_paths.append(maybe_path)

    if factor_data and positions_frame is not None and not positions_frame.empty:
        latest = positions_frame.iloc[-1].drop(labels=["cash"], errors="ignore")
        if not latest.empty:
            factor_positions = pd.DataFrame(
                {
                    "stock_code": latest.index.astype(str),
                    "weight": pd.to_numeric(latest, errors="coerce").fillna(0.0).values,
                }
            )
            attribution = analyze_portfolio_comprehensive(
                factor_positions,
                returns,
                factor_data=factor_data,
                benchmark_returns=None,
            )
            factor_attribution_path = output_dir / f"{stem}factor_attribution_summary.csv"
            pd.DataFrame(
                {
                    "factor": list(attribution["factor_exposure"]["factor_exposures"].keys()),
                    "exposure": list(attribution["factor_exposure"]["factor_exposures"].values()),
                }
            ).to_csv(factor_attribution_path, index=False)
            if attribution["factor_exposure"]["factor_exposures"]:
                factor_frame = pd.Series(attribution["factor_exposure"]["factor_exposures"]).sort_values()
                plt.figure(figsize=(10, max(4, 0.35 * len(factor_frame))))
                factor_frame.plot(kind="barh", color="#5a189a")
                plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
                path = output_dir / f"{report_name}_factor_attribution.png"
                finalize_quantstats_axis(plt.gca(), title="Factor attribution", xlabel="Exposure")
                save_quantstats_figure(path)
                figure_paths.append(path)

    return PortfolioTradeAnalysisResult(
        output_dir=output_dir,
        figure_paths=figure_paths,
        transactions_path=transactions_path,
        round_trips_path=round_trips_path,
        capacity_summary_path=capacity_summary_path,
        factor_attribution_path=factor_attribution_path,
        transaction_summary_path=transaction_summary_path,
        round_trip_summary_path=round_trip_summary_path,
        report_name=report_name,
    )


def build_trade_analysis(
    returns: pd.Series,
    *,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    output_dir: str | Path,
    report_name: str = "portfolio",
    capital_base: float = 1_000_000.0,
) -> PortfolioTradeAnalysisResult | None:
    return create_trade_report(
        returns,
        positions=positions,
        transactions=transactions,
        market_data=market_data,
        close_panel=close_panel,
        factor_data=factor_data,
        output_dir=output_dir,
        report_name=report_name,
        capital_base=capital_base,
    )


def create_trade_tear_sheet(
    returns: pd.Series,
    *,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    output_dir: str | Path,
    report_name: str = "portfolio",
    capital_base: float = 1_000_000.0,
) -> PortfolioTradeAnalysisResult | None:
    return create_trade_report(
        returns,
        positions=positions,
        transactions=transactions,
        market_data=market_data,
        close_panel=close_panel,
        factor_data=factor_data,
        output_dir=output_dir,
        report_name=report_name,
        capital_base=capital_base,
    )


__all__ = [
    "PortfolioTradeAnalysisResult",
    "create_trade_report",
    "create_trade_tear_sheet",
    "build_trade_analysis",
    "extract_round_trips",
    "synthesize_transactions_from_positions",
]
