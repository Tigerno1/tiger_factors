from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def _safe_series_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(frame) < 3:
        return float("nan")
    x = pd.to_numeric(frame["left"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["right"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def _normalize_event_return_table(
    average_cumulative_return_by_quantile: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    if isinstance(average_cumulative_return_by_quantile, pd.Series):
        df = average_cumulative_return_by_quantile.to_frame()
    else:
        df = average_cumulative_return_by_quantile.copy()
    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex) and "factor_quantile" in df.index.names:
        if "mean" in df.index.get_level_values(-1):
            df = df.xs("mean", level=-1)
        df = df.unstack("factor_quantile")

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(df.columns, pd.MultiIndex) and "factor_quantile" in df.columns.names:
        level = df.columns.names.index("factor_quantile")
        df.columns = df.columns.get_level_values(level)

    try:
        df.columns = [int(c) for c in df.columns]
    except Exception:
        pass
    return df


def _event_return_spread_series(
    average_cumulative_return_by_quantile: pd.DataFrame | pd.Series,
    *,
    top_quantile: int | None = None,
    bottom_quantile: int | None = None,
) -> pd.Series:
    df = _normalize_event_return_table(average_cumulative_return_by_quantile)
    if df.empty or len(df.columns) == 0:
        return pd.Series(dtype=float, name="spread")

    if top_quantile is None or top_quantile not in df.columns:
        top_label = df.columns[-1]
    else:
        top_label = top_quantile
    if bottom_quantile is None or bottom_quantile not in df.columns:
        bottom_label = df.columns[0]
    else:
        bottom_label = bottom_quantile

    spread = pd.to_numeric(df[top_label], errors="coerce") - pd.to_numeric(df[bottom_label], errors="coerce")
    spread = spread.sort_index()

    raw_period_index = pd.Series(
        [item[0] if isinstance(item, tuple) and item else item for item in spread.index],
        index=spread.index,
    )
    period_index = pd.to_numeric(raw_period_index, errors="coerce")
    spread = spread.copy()
    spread.index = period_index
    return spread.dropna().sort_index()


@dataclass(frozen=True)
class HorizonResult:
    horizon: int
    mean_ic: float
    ic_std: float
    ic_ir: float
    mean_spread: float
    spread_std: float
    spread_ir: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    rank_autocorr: float
    direction: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class HoldingPeriodAnalyzer:
    """
    Analyze factor performance across multiple holding periods.

    Parameters
    ----------
    factor_panel
        Wide factor panel indexed by ``date_`` with asset codes in columns.
    price_panel
        Wide adjusted price panel indexed by ``date_`` with asset codes in columns.
    """

    def __init__(
        self,
        factor_panel: pd.DataFrame | pd.Series,
        price_panel: pd.DataFrame,
        *,
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
        min_names: int | None = None,
    ) -> None:
        if isinstance(factor_panel, pd.Series):
            if not isinstance(factor_panel.index, pd.MultiIndex) or factor_panel.index.nlevels != 2:
                raise ValueError("factor_panel series must be indexed by (date, asset).")
            self.factor_panel = factor_panel.sort_index().unstack().sort_index()
        else:
            self.factor_panel = factor_panel.sort_index()
        self.price_panel = price_panel.sort_index()
        self.quantiles = int(max(quantiles, 2))
        self.periods_per_year = int(max(periods_per_year, 1))
        self.long_short_pct = float(min(max(long_short_pct, 0.01), 0.5))
        self.min_names = None if min_names is None else int(max(min_names, 1))

        common_dates = self.factor_panel.index.intersection(self.price_panel.index)
        common_codes = self.factor_panel.columns.intersection(self.price_panel.columns)
        self.factor_panel = self.factor_panel.loc[common_dates, common_codes]
        self.price_panel = self.price_panel.loc[common_dates, common_codes]

    def _minimum_names(self, fallback: int) -> int:
        if self.min_names is None:
            return int(fallback)
        return int(max(self.min_names, 1))

    def _forward_returns(self, horizon: int) -> pd.DataFrame:
        horizon = int(max(horizon, 1))
        return self.price_panel.shift(-horizon).div(self.price_panel).sub(1.0)

    def _cross_sectional_ic(self, factor_t: pd.Series, ret_t: pd.Series) -> float:
        frame = pd.concat([factor_t.rename("factor"), ret_t.rename("ret")], axis=1).dropna()
        if len(frame) < self._minimum_names(5):
            return float("nan")
        return _safe_series_corr(frame["factor"], frame["ret"], method="spearman")

    def _daily_ic_series(self, forward_returns: pd.DataFrame) -> pd.Series:
        values: list[float] = []
        for dt in self.factor_panel.index:
            if dt not in forward_returns.index:
                values.append(float("nan"))
                continue
            values.append(self._cross_sectional_ic(self.factor_panel.loc[dt], forward_returns.loc[dt]))
        return pd.Series(values, index=self.factor_panel.index, name="ic")

    def _assign_quantiles(self, factor_t: pd.Series) -> pd.Series:
        clean = factor_t.dropna()
        out = pd.Series(index=factor_t.index, dtype=float)
        if len(clean) < max(self.quantiles, self._minimum_names(self.quantiles)):
            return out
        try:
            quantiles = pd.qcut(clean.rank(method="first"), self.quantiles, labels=False) + 1
            out.loc[clean.index] = quantiles.astype(float)
            return out
        except ValueError:
            return out

    def _daily_spread_series(self, forward_returns: pd.DataFrame) -> pd.Series:
        spreads: list[float] = []
        for dt in self.factor_panel.index:
            if dt not in forward_returns.index:
                spreads.append(float("nan"))
                continue
            factor_t = self.factor_panel.loc[dt]
            ret_t = forward_returns.loc[dt]
            quantiles = self._assign_quantiles(factor_t)
            frame = pd.concat([quantiles.rename("q"), ret_t.rename("ret")], axis=1).dropna()
            if frame.empty:
                spreads.append(float("nan"))
                continue
            top = frame.loc[frame["q"] == self.quantiles, "ret"].mean()
            bottom = frame.loc[frame["q"] == 1, "ret"].mean()
            spreads.append(float(top - bottom))
        return pd.Series(spreads, index=self.factor_panel.index, name="spread")

    def _daily_ls_return_series(self, horizon: int) -> tuple[pd.Series, pd.Series]:
        forward_returns = self._forward_returns(horizon)
        long_short: list[float] = []
        turnover: list[float] = []

        prev_long: set[str] | None = None
        prev_short: set[str] | None = None

        for dt in self.factor_panel.index:
            if dt not in forward_returns.index:
                long_short.append(float("nan"))
                turnover.append(float("nan"))
                continue

            factor_t = self.factor_panel.loc[dt]
            ret_t = forward_returns.loc[dt]
            frame = pd.concat([factor_t.rename("factor"), ret_t.rename("ret")], axis=1).dropna()
            if len(frame) < self._minimum_names(10):
                long_short.append(float("nan"))
                turnover.append(float("nan"))
                continue

            bucket_size = max(1, int(len(frame) * self.long_short_pct))
            ranked = frame.sort_values("factor")
            short_names = set(ranked.index[:bucket_size].astype(str))
            long_names = set(ranked.index[-bucket_size:].astype(str))

            long_ret = frame.loc[list(long_names), "ret"].mean()
            short_ret = frame.loc[list(short_names), "ret"].mean()
            long_short.append(float(long_ret - short_ret))

            if prev_long is None or prev_short is None:
                turnover.append(float("nan"))
            else:
                long_turn = 1.0 - len(long_names & prev_long) / max(1, len(long_names))
                short_turn = 1.0 - len(short_names & prev_short) / max(1, len(short_names))
                turnover.append(float((long_turn + short_turn) / 2.0))

            prev_long = long_names
            prev_short = short_names

        return (
            pd.Series(long_short, index=self.factor_panel.index, name=f"ls_{int(horizon)}"),
            pd.Series(turnover, index=self.factor_panel.index, name=f"turnover_{int(horizon)}"),
        )

    def _rank_autocorr(self, lag: int = 1) -> float:
        lag = int(max(lag, 1))
        values: list[float] = []
        dates = self.factor_panel.index
        for i in range(lag, len(dates)):
            t0 = dates[i - lag]
            t1 = dates[i]
            frame = pd.concat(
                [self.factor_panel.loc[t0].rename("x"), self.factor_panel.loc[t1].rename("y")],
                axis=1,
            ).dropna()
            if len(frame) < self._minimum_names(5):
                continue
            values.append(_safe_series_corr(frame["x"], frame["y"], method="spearman"))
        if not values:
            return float("nan")
        return float(np.nanmean(values))

    @staticmethod
    def _max_drawdown(cumulative_returns: pd.Series) -> float:
        if cumulative_returns.empty:
            return float("nan")
        peak = cumulative_returns.cummax()
        drawdown = cumulative_returns.div(peak).sub(1.0)
        return float(drawdown.min())

    def analyze_horizon(self, horizon: int) -> HorizonResult:
        horizon = int(max(horizon, 1))
        forward_returns = self._forward_returns(horizon)
        ic_series = self._daily_ic_series(forward_returns).dropna()
        spread_series = self._daily_spread_series(forward_returns).dropna()
        ls_series, turnover_series = self._daily_ls_return_series(horizon)
        ls_series = ls_series.dropna()
        turnover_series = turnover_series.dropna()

        mean_ic = float(ic_series.mean()) if not ic_series.empty else float("nan")
        ic_std = float(ic_series.std()) if not ic_series.empty else float("nan")
        ic_ir = mean_ic / ic_std if pd.notna(ic_std) and ic_std > 0 else float("nan")

        mean_spread = float(spread_series.mean()) if not spread_series.empty else float("nan")
        spread_std = float(spread_series.std()) if not spread_series.empty else float("nan")
        spread_ir = mean_spread / spread_std if pd.notna(spread_std) and spread_std > 0 else float("nan")

        ann_return = float(ls_series.mean() * self.periods_per_year / horizon) if not ls_series.empty else float("nan")
        ann_vol = float(ls_series.std() * np.sqrt(self.periods_per_year / horizon)) if not ls_series.empty else float("nan")
        sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else float("nan")

        cumulative = (1.0 + ls_series.fillna(0.0)).cumprod()
        max_drawdown = self._max_drawdown(cumulative)
        direction = "positive" if pd.notna(mean_ic) and mean_ic > 0 else "negative"

        return HorizonResult(
            horizon=horizon,
            mean_ic=mean_ic,
            ic_std=ic_std,
            ic_ir=ic_ir,
            mean_spread=mean_spread,
            spread_std=spread_std,
            spread_ir=spread_ir,
            ann_return=ann_return,
            ann_vol=ann_vol,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            avg_turnover=float(turnover_series.mean()) if not turnover_series.empty else float("nan"),
            rank_autocorr=self._rank_autocorr(lag=min(horizon, 5)),
            direction=direction,
        )

    def run(self, horizons: Iterable[int]) -> pd.DataFrame:
        rows = [self.analyze_horizon(int(h)).to_dict() for h in horizons]
        result = pd.DataFrame(rows)
        if result.empty:
            return result
        return result.sort_values("horizon").reset_index(drop=True)

    def plot_advanced(self, result: pd.DataFrame, *, output_path: str | Path | None = None) -> plt.Figure | Path:
        figure, axes = plt.subplots(2, 3, figsize=(16, 8))
        draw_horizon_advanced(result, axes)
        figure.tight_layout()
        if output_path is None:
            return figure

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_summary(
        self,
        result: pd.DataFrame,
        *,
        output_path: str | Path | None = None,
    ) -> plt.Figure | Path:
        figure, ax = plt.subplots(figsize=(10, 3.6))
        draw_horizon_summary(result, ax)
        figure.tight_layout()

        if output_path is None:
            return figure

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return path


def summarize_best_horizon(result: pd.DataFrame) -> dict[str, int | str | float]:
    summary: dict[str, int | str | float] = {}
    if result.empty or "horizon" not in result.columns:
        summary["suggested_direction"] = "use_as_is"
        return summary

    valid_ic = result["mean_ic"].dropna()
    valid_ic_ir = result["ic_ir"].dropna()
    valid_sharpe = result["sharpe"].dropna()

    if not valid_ic.empty:
        summary["best_ic_horizon"] = int(result.loc[valid_ic.abs().idxmax(), "horizon"])
    if not valid_ic_ir.empty:
        summary["best_ic_ir_horizon"] = int(result.loc[valid_ic_ir.abs().idxmax(), "horizon"])
    if not valid_sharpe.empty:
        summary["best_sharpe_horizon"] = int(result.loc[valid_sharpe.idxmax(), "horizon"])

    sign_score = float(np.sign(valid_ic).mean()) if not valid_ic.empty else float("nan")
    summary["suggested_direction"] = "reverse_factor" if pd.notna(sign_score) and sign_score < 0 else "use_as_is"
    return summary


def select_best_holding_period_from_event_returns(
    average_cumulative_return_by_quantile: pd.DataFrame | pd.Series,
    *,
    top_quantile: int | None = None,
    bottom_quantile: int | None = None,
    min_period: int = 1,
    max_period: int | None = None,
) -> dict[str, float | int | str]:
    """
    Pick a holding period from event-return cumulative spread paths.

    The helper is intentionally conservative: it accepts the common Alphalens-style
    return tables and returns a small summary dict that can be attached to the
    event returns sheet payload.
    """
    df = _normalize_event_return_table(average_cumulative_return_by_quantile)
    if df.empty:
        return {
            "best_holding_period": -1,
            "best_cumulative_spread": float("nan"),
            "best_score": float("nan"),
            "direction": "unknown",
        }

    if df.empty or len(df.columns) == 0:
        return {
            "best_holding_period": -1,
            "best_cumulative_spread": float("nan"),
            "best_score": float("nan"),
            "direction": "unknown",
        }

    if top_quantile is None or top_quantile not in df.columns:
        top_label = df.columns[-1]
    else:
        top_label = top_quantile
    if bottom_quantile is None or bottom_quantile not in df.columns:
        bottom_label = df.columns[0]
    else:
        bottom_label = bottom_quantile

    spread = _event_return_spread_series(df, top_quantile=top_label, bottom_quantile=bottom_label)
    period_index = spread.index
    valid_mask = period_index >= min_period
    if max_period is not None:
        valid_mask &= period_index <= max_period
    valid = spread.loc[valid_mask].copy()
    valid = valid.dropna()

    if valid.empty:
        return {
            "best_holding_period": -1,
            "best_cumulative_spread": float("nan"),
            "best_score": float("nan"),
            "direction": "unknown",
        }

    running_peak = valid.cummax()
    drawdown = valid - running_peak
    score = valid / (1.0 + drawdown.abs())

    best_pos = int(score.to_numpy(dtype=float).argmax())
    best_idx = score.index[best_pos]
    best_spread = float(valid.iloc[best_pos])
    direction = "use_as_is" if best_spread >= 0 else "reverse_factor"

    return {
        "best_holding_period": int(best_idx),
        "best_cumulative_spread": best_spread,
        "best_score": float(score.iloc[best_pos]),
        "direction": direction,
    }


def plot_best_holding_period_selection(
    average_cumulative_return_by_quantile: pd.DataFrame | pd.Series,
    *,
    top_quantile: int | None = None,
    bottom_quantile: int | None = None,
    best_holding_period: dict[str, float | int | str] | None = None,
    min_period: int = 1,
    max_period: int | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))

    spread = _event_return_spread_series(
        average_cumulative_return_by_quantile,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )
    if spread.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No best holding period data", ha="center", va="center", transform=ax.transAxes)
        return ax

    valid = spread.loc[spread.index >= min_period].copy()
    if max_period is not None:
        valid = valid.loc[valid.index <= max_period]
    valid = valid.dropna()
    if valid.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid holding period data", ha="center", va="center", transform=ax.transAxes)
        return ax

    selected = best_holding_period or select_best_holding_period_from_event_returns(
        average_cumulative_return_by_quantile,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        min_period=min_period,
        max_period=max_period,
    )
    best_period = selected.get("best_holding_period", -1)
    best_spread = selected.get("best_cumulative_spread", float("nan"))

    ax.plot(valid.index, valid.values, marker="o", markersize=4, color="#4C72B0", linewidth=1.6)
    if pd.notna(best_period) and int(best_period) >= 0:
        ax.axvline(int(best_period), color="#C44E52", linestyle="--", alpha=0.75)
    if pd.notna(best_spread):
        ax.scatter([best_period], [best_spread], color="#C44E52", s=30, zorder=3)
    ax.set_title("Best Holding Period")
    ax.set_xlabel("Holding Period")
    ax.set_ylabel("Cumulative Spread")
    ax.grid(True, alpha=0.25)
    return ax


def draw_horizon_advanced(result: pd.DataFrame, axes: np.ndarray | list[Axes] | list[list[Axes]]) -> None:
    grid = np.asarray(axes)
    grid = grid.reshape(2, 3)
    if result.empty or not {"horizon", "mean_ic", "ic_ir", "mean_spread", "sharpe", "avg_turnover", "rank_autocorr"}.issubset(result.columns):
        for ax in grid.flat:
            ax.axis("off")
            ax.text(0.5, 0.5, "No horizon data", ha="center", va="center", transform=ax.transAxes)
        return
    grid[0, 0].plot(result["horizon"], result["mean_ic"], marker="o")
    grid[0, 0].set_title("Mean IC")
    grid[0, 1].plot(result["horizon"], result["ic_ir"], marker="o")
    grid[0, 1].set_title("IC IR")
    grid[0, 2].plot(result["horizon"], result["mean_spread"], marker="o")
    grid[0, 2].set_title("Mean Spread")
    grid[1, 0].plot(result["horizon"], result["sharpe"], marker="o")
    grid[1, 0].set_title("Sharpe")
    grid[1, 1].plot(result["horizon"], result["avg_turnover"], marker="o")
    grid[1, 1].set_title("Turnover")
    grid[1, 2].plot(result["horizon"], result["rank_autocorr"], marker="o")
    grid[1, 2].set_title("Rank Autocorrelation")
    for ax in grid.flat:
        ax.axhline(0.0, linewidth=1.0, color="black", alpha=0.4)
        ax.grid(True, alpha=0.2)


def draw_horizon_summary(result: pd.DataFrame, ax: Axes) -> None:
    summary = summarize_best_horizon(result)
    ax.axis("off")
    if result.empty or "horizon" not in result.columns:
        rows = [["Horizon", "No data"]]
        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        ax.set_title("Holding Period Summary", fontsize=13, pad=12)
        return
    rows = [
        ["Best IC Horizon", summary.get("best_ic_horizon", "")],
        ["Best IC IR Horizon", summary.get("best_ic_ir_horizon", "")],
        ["Best Sharpe Horizon", summary.get("best_sharpe_horizon", "")],
        ["Suggested Direction", summary.get("suggested_direction", "")],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Holding Period Summary", fontsize=13, pad=12)


__all__ = [
    "HoldingPeriodAnalyzer",
    "HorizonResult",
    "draw_horizon_advanced",
    "draw_horizon_summary",
    "plot_best_holding_period_selection",
    "select_best_holding_period_from_event_returns",
    "summarize_best_horizon",
]
