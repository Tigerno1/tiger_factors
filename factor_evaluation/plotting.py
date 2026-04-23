from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from typing import Callable
from typing import Iterable

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tiger_matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from tiger_factors.factor_evaluation.horizon import draw_horizon_advanced
from tiger_factors.factor_evaluation.horizon import draw_horizon_summary
from tiger_factors.factor_evaluation.horizon import plot_best_holding_period_selection
from tiger_factors.factor_evaluation.utils import period_to_label

DECIMAL_TO_BPS = 10000
RETURNS_OVERVIEW_PALETTE = sns.color_palette("Set2", n_colors=8)
OVERVIEW_FIG_WIDTH = 12.0
OVERVIEW_TITLE_FONTSIZE = 15
OVERVIEW_SECTION_FONTSIZE = 10
OVERVIEW_HSPACE = 0.38
OVERVIEW_WSPACE = 0.12


class GridFigure:
    """Lightweight alphalens-compatible grid figure helper.

    The historical Alphalens helper exposes ``next_cell`` and ``next_row`` for
    sequential subplot allocation.  The Tiger test suite only relies on that
    contract, so this implementation keeps the behaviour intentionally small and
    dependency-free.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        figsize: tuple[float, float] | None = None,
        dpi: int = 150,
        tight_layout: bool = True,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")
        self.rows = int(rows)
        self.cols = int(cols)
        self._tight_layout = tight_layout
        self.fig = plt.figure(figsize=figsize or (max(1.0, cols * 4.0), max(1.0, rows * 3.0)), dpi=dpi)
        self._grid = self.fig.add_gridspec(self.rows, self.cols)
        self._row = 0
        self._col = 0
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("GridFigure has already been closed.")

    def _advance_cell(self) -> None:
        self._col += 1
        if self._col >= self.cols:
            self._col = 0
            self._row += 1

    def next_cell(self):
        self._ensure_open()
        if self._row >= self.rows:
            return None
        axis = self.fig.add_subplot(self._grid[self._row, self._col])
        self._advance_cell()
        return axis

    def next_row(self):
        self._ensure_open()
        if self._row >= self.rows:
            return None
        axis = self.fig.add_subplot(self._grid[self._row, :])
        self._row += 1
        self._col = 0
        return axis

    def close(self):
        if self._closed:
            return None
        if self._tight_layout:
            try:
                self.fig.tight_layout()
            except Exception:
                pass
        self._closed = True
        plt.close(self.fig)
        return self.fig


def _returns_palette_for_labels(labels: Iterable[object]) -> list[tuple[float, float, float]]:
    label_list = [str(label) for label in labels]
    base_labels = ["1D", "5D", "10D"]
    color_map = {
        "1D": RETURNS_OVERVIEW_PALETTE[0],
        "5D": RETURNS_OVERVIEW_PALETTE[1],
        "10D": RETURNS_OVERVIEW_PALETTE[2],
    }
    palette: list[tuple[float, float, float]] = []
    for idx, label in enumerate(label_list):
        if label in color_map:
            palette.append(color_map[label])
        elif idx < len(base_labels) and base_labels[idx] in color_map:
            palette.append(color_map[base_labels[idx]])
        else:
            palette.append(RETURNS_OVERVIEW_PALETTE[idx % len(RETURNS_OVERVIEW_PALETTE)])
    return palette


def _turnover_period_labels(frame: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for column in frame.columns:
        text = str(column)
        if text.startswith("top_"):
            label = text[len("top_") :]
        elif text.startswith("bottom_"):
            label = text[len("bottom_") :]
        else:
            return []
        if label not in labels:
            labels.append(label)
    if not labels:
        return []
    if all(f"top_{label}" in frame.columns and f"bottom_{label}" in frame.columns for label in labels):
        return labels
    return []


def _safe_stat_series(ic_data: pd.DataFrame, func) -> pd.Series:
    values: dict[str, float] = {}
    for column in ic_data.columns:
        series = _clean_numeric_series(ic_data[column])
        values[str(column)] = float(func(series)) if len(series) >= 3 else float("nan")
    return pd.Series(values)


def _safe_ttest_stat(ic_data: pd.DataFrame, attr: str) -> pd.Series:
    values: dict[str, float] = {}
    for column in ic_data.columns:
        series = _clean_numeric_series(ic_data[column])
        if len(series) < 2:
            values[str(column)] = float("nan")
            continue
        result = stats.ttest_1samp(series, 0.0, nan_policy="omit")
        values[str(column)] = float(getattr(result, attr))
    return pd.Series(values)


def _get_ax(ax=None, *, figsize=(10, 4)):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax


def _finalize_ax(ax, *, title: str = "", grid: bool = True):
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return ax


def _safe_legend(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        return ax.legend(*args, **kwargs)
    return None


def _render_table_axes(ax, table: pd.DataFrame, *, title: str | None = None, font_size: int = 8) -> None:
    ax.axis("off")
    if title:
        ax.set_title(title, pad=12)
    table_artist = ax.table(
        cellText=table.fillna("").round(4).values,
        rowLabels=[str(index) for index in table.index],
        colLabels=table.columns.tolist(),
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(font_size)
    table_artist.scale(1.0, 1.25)


def _monthly_rolling_mean(series: pd.Series, *, fallback_window: int = 22, min_periods: int = 3) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if cleaned.empty:
        return cleaned
    if isinstance(cleaned.index, pd.DatetimeIndex):
        return cleaned.rolling("30D", min_periods=min(min_periods, len(cleaned))).mean()
    return cleaned.rolling(window=fallback_window, min_periods=min(min_periods, len(cleaned))).mean()


def _clean_numeric_series(
    series: pd.Series,
    *,
    dropna: bool = True,
    sort_index: bool = False,
    fillna: float | None = None,
) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if fillna is not None:
        clean = clean.fillna(fillna)
    elif dropna:
        clean = clean.dropna()
    if sort_index:
        clean = clean.sort_index()
    return clean


def _prepare_panel_axes(
    num_plots: int,
    ax=None,
    *,
    ncols: int = 3,
    width: float = 18.0,
    height: float = 6.0,
    sharey: bool = False,
):
    axes = ax
    if axes is None:
        rows = ((max(int(num_plots), 1) - 1) // max(int(ncols), 1)) + 1
        _, axes = plt.subplots(
            rows,
            max(int(ncols), 1),
            figsize=(width, rows * height),
            squeeze=False,
            sharey=sharey,
        )
        axes = axes.flatten()
    elif not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    return list(axes)


def _finalize_panel_axes(axes, rendered_axes):
    for current_ax in axes[len(rendered_axes) :]:
        current_ax.set_visible(False)
    if len(rendered_axes) == 1:
        return rendered_axes[0]
    return rendered_axes


def _normalize_quantile_wide_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        return frame.unstack("factor_quantile")
    if isinstance(frame.index, pd.MultiIndex) and "factor_quantile" in frame.index.names:
        return frame.unstack("factor_quantile")
    return frame.copy()


def save_stability_overview(
    turnover_frame: pd.DataFrame,
    rank_autocorr: pd.DataFrame,
    horizon_result: pd.DataFrame,
    output: Path,
    *,
    title: str = "Stability and Horizon Overview",
) -> Path | None:
    if turnover_frame.empty and rank_autocorr.empty and horizon_result.empty:
        return None

    turnover_rows = len(_turnover_period_labels(turnover_frame)) if not turnover_frame.empty else 0
    rank_rows = 1 if not rank_autocorr.empty else 0
    horizon_rows = 2 if not horizon_result.empty else 0
    summary_rows = 1 if not horizon_result.empty else 0

    total_rows = turnover_rows + rank_rows + horizon_rows + summary_rows
    fig = plt.figure(figsize=(OVERVIEW_FIG_WIDTH, max(5.0 * total_rows, 8.0)))
    gs = gridspec.GridSpec(max(total_rows, 1), 3, figure=fig, hspace=OVERVIEW_HSPACE, wspace=OVERVIEW_WSPACE)

    cursor = 0
    if turnover_rows:
        turnover_axes = [fig.add_subplot(gs[cursor + idx, :]) for idx in range(turnover_rows)]
        plot_top_bottom_quantile_turnover(turnover_frame, ax=turnover_axes)
        cursor += turnover_rows

    if rank_rows:
        rank_ax = fig.add_subplot(gs[cursor, :])
        plot_factor_rank_auto_correlation(rank_autocorr, ax=rank_ax)
        cursor += rank_rows

    if horizon_rows:
        horizon_axes = np.array(
            [
                [fig.add_subplot(gs[cursor, 0]), fig.add_subplot(gs[cursor, 1]), fig.add_subplot(gs[cursor, 2])],
                [fig.add_subplot(gs[cursor + 1, 0]), fig.add_subplot(gs[cursor + 1, 1]), fig.add_subplot(gs[cursor + 1, 2])],
            ]
        )
        draw_horizon_advanced(horizon_result, horizon_axes)
        cursor += horizon_rows

    if summary_rows:
        summary_ax = fig.add_subplot(gs[cursor, :])
        draw_horizon_summary(horizon_result, summary_ax)

    fig.suptitle(title, y=0.995, fontsize=OVERVIEW_TITLE_FONTSIZE)
    fig.subplots_adjust(top=0.975)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


@contextmanager
def plotting_context(context="notebook", font_scale=1.5, rc=None):
    settings = {"lines.linewidth": 1.5}
    if rc is not None:
        settings.update(rc)
    with sns.plotting_context(context=context, font_scale=font_scale, rc=settings):
        yield


@contextmanager
def axes_style(style="darkgrid", rc=None):
    with sns.axes_style(style=style, rc=rc):
        yield


def customize(func):
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop("set_context", True)
        if not set_context:
            return func(*args, **kwargs)
        color_palette = sns.color_palette("colorblind")
        with plotting_context(), axes_style(), color_palette:
            sns.despine(left=True)
            return func(*args, **kwargs)

    return call_w_context


def _period_label(period) -> str:
    return period_to_label(period)


def plot_ic_ts(ic_data: pd.DataFrame, ax=None):
    num_plots = max(len(ic_data.columns), 1)
    axes = _prepare_panel_axes(num_plots, ax, ncols=1, width=12.0, height=4.0)
    ymin, ymax = None, None
    for current_ax, column in zip(axes, ic_data.columns):
        series = ic_data[column].sort_index()
        if series.dropna().empty:
            current_ax.axis("off")
            current_ax.set_title(f"{column} Forward Return IC")
            current_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=current_ax.transAxes)
            continue
        rolling = _monthly_rolling_mean(series)
        series.plot(ax=current_ax, linewidth=0.9, alpha=0.7, color="steelblue")
        rolling.plot(ax=current_ax, linewidth=1.8, alpha=0.85, color="forestgreen")
        current_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        current_ax.set_ylabel("IC")
        current_ax.set_title(f"{column} Forward Return IC")
        current_ax.legend(["IC", "1 month moving avg"], loc="upper right")
        current_ax.text(
            0.05,
            0.95,
            f"Mean {series.mean():.3f}\nStd. {series.std():.3f}",
            transform=current_ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 4},
        )
        lo, hi = current_ax.get_ylim()
        ymin = lo if ymin is None else min(ymin, lo)
        ymax = hi if ymax is None else max(ymax, hi)
    if ymin is not None and ymax is not None:
        for current_ax in axes[:num_plots]:
            current_ax.set_ylim(ymin, ymax)
    return axes[0] if len(axes) == 1 else axes


def plot_ic_rolling(ic_data: pd.DataFrame, ax=None, *, window: int = 22):
    num_plots = max(len(ic_data.columns), 1)
    axes = _prepare_panel_axes(num_plots, ax, ncols=3, width=18.0, height=6.0)
    rendered_axes = []
    ymin, ymax = None, None
    for current_ax, column in zip(axes, ic_data.columns):
        series = _clean_numeric_series(ic_data[column], sort_index=True)
        if series.empty:
            current_ax.axis("off")
            current_ax.set_title(f"{column} Period Rolling IC", fontsize=10, pad=6)
            current_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=current_ax.transAxes)
            rendered_axes.append(current_ax)
            continue
        rolling = _monthly_rolling_mean(series, fallback_window=window, min_periods=max(3, window // 8))
        series.plot(ax=current_ax, linewidth=0.8, alpha=0.28, color="steelblue", label="IC")
        rolling.plot(ax=current_ax, linewidth=1.8, alpha=0.9, color="forestgreen", label="1M moving avg")
        current_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        current_ax.set_title(f"{column} Period Rolling IC", fontsize=10, pad=6)
        current_ax.set_ylabel("IC", fontsize=9, labelpad=4)
        current_ax.set_xlabel("Date", fontsize=9, labelpad=4)
        current_ax.legend(loc="upper right")
        current_ax.text(
            0.05,
            0.95,
            f"Mean {series.mean():.3f}\nStd. {series.std():.3f}",
            transform=current_ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 3},
        )
        lo, hi = current_ax.get_ylim()
        ymin = lo if ymin is None else min(ymin, lo)
        ymax = hi if ymax is None else max(ymax, hi)
        rendered_axes.append(current_ax)
    for current_ax in rendered_axes:
        current_ax.set_ylim(ymin, ymax)
    return _finalize_panel_axes(axes, rendered_axes)


def plot_ic_hist(ic_data: pd.DataFrame, ax=None):
    num_plots = max(len(ic_data.columns), 1)
    axes = _prepare_panel_axes(num_plots, ax, ncols=3, width=18.0, height=6.0)
    rendered_axes = []
    for current_ax, column in zip(axes, ic_data.columns):
        series = _clean_numeric_series(ic_data[column])
        if series.empty:
            current_ax.axis("off")
            current_ax.set_title(f"{column} Period IC", fontsize=10, pad=6)
            current_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=current_ax.transAxes)
            rendered_axes.append(current_ax)
            continue
        sns.histplot(series, kde=True, bins=45, stat="density", color="#4C72B0", ax=current_ax)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if np.isfinite(mean) and np.isfinite(std) and std > 1e-12:
            grid = np.linspace(series.min(), series.max(), 200)
            normal_pdf = stats.norm.pdf(grid, loc=mean, scale=std)
            current_ax.plot(grid, normal_pdf, color="#d62728", linewidth=1.8, label="Normal fit")
        current_ax.set_title(f"{column} Period IC", fontsize=10, pad=6)
        current_ax.set_xlabel("IC", fontsize=9, labelpad=4)
        current_ax.set_ylabel("Density", fontsize=9, labelpad=4)
        current_ax.set_xlim([-1, 1])
        current_ax.text(
            0.05,
            0.95,
            f"Mean {mean:.3f}\nStd. {std:.3f}",
            transform=current_ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 3},
        )
        current_ax.axvline(mean, color="w", linestyle="dashed", linewidth=2)
        current_ax.legend(loc="upper right")
        rendered_axes.append(current_ax)
    return _finalize_panel_axes(axes, rendered_axes)


def plot_ic_missingness(ic_data: pd.DataFrame, ax=None):
    num_plots = max(len(ic_data.columns), 1)
    axes = _prepare_panel_axes(num_plots, ax, ncols=3, width=18.0, height=6.0)
    rendered_axes = []
    for current_ax, column in zip(axes, ic_data.columns):
        series = _clean_numeric_series(ic_data[column], dropna=False)
        valid_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        total_count = valid_count + missing_count
        if total_count == 0:
            current_ax.axis("off")
            current_ax.set_title(f"{column} Period IC Missingness", pad=6)
            current_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=current_ax.transAxes)
            rendered_axes.append(current_ax)
            continue
        colors = [RETURNS_OVERVIEW_PALETTE[0], RETURNS_OVERVIEW_PALETTE[3]]
        bars = current_ax.bar(
            ["Valid", "Missing"],
            [valid_count, missing_count],
            color=colors,
            width=0.62,
        )
        current_ax.set_title(f"{column} Period IC Missingness", fontsize=10, pad=6)
        current_ax.set_ylabel("Count", fontsize=9, labelpad=4)
        current_ax.set_ylim(0, max(total_count * 1.15, 1))
        miss_rate = missing_count / total_count if total_count else 0.0
        current_ax.text(
            0.05,
            0.95,
            f"Total {total_count}\nMissing {missing_count} ({miss_rate:.1%})",
            transform=current_ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 3},
        )
        for bar, value in zip(bars, [valid_count, missing_count]):
            current_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(total_count * 0.015, 0.5),
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        rendered_axes.append(current_ax)
    return _finalize_panel_axes(axes, rendered_axes)


def plot_ic_qq(ic_data: pd.DataFrame, theoretical_dist=stats.norm, ax=None):
    num_plots = max(len(ic_data.columns), 1)
    axes = _prepare_panel_axes(num_plots, ax, ncols=3, width=18.0, height=6.0)

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = "Normal"
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = "T"
    else:
        dist_name = "Theoretical"

    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover - explicit dependency failure
        raise RuntimeError("statsmodels is required for plot_ic_qq") from exc

    rendered_axes = []
    for current_ax, column in zip(axes, ic_data.columns):
        values = _clean_numeric_series(ic_data[column])
        if values.empty:
            current_ax.axis("off")
            current_ax.set_title(f"{column} Period IC {dist_name} Dist. Q-Q", pad=6)
            current_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=current_ax.transAxes)
            rendered_axes.append(current_ax)
            continue
        sm.qqplot(
            values.to_numpy(),
            theoretical_dist,
            fit=True,
            line="45",
            ax=current_ax,
            marker="o",
            markerfacecolor="#2a6fdb",
            markeredgecolor="#2a6fdb",
            markersize=2.8,
            alpha=0.42,
        )
        current_ax.set_title(f"{column} Period IC {dist_name} Dist. Q-Q", pad=6)
        current_ax.set_xlabel(f"{dist_name} Distribution Quantile", labelpad=4)
        current_ax.set_ylabel("Observed Quantile", labelpad=4)
        rendered_axes.append(current_ax)
    return _finalize_panel_axes(axes, rendered_axes)


def plot_monthly_ic_heatmap(ic_data: pd.DataFrame, ax=None):
    frame = ic_data.copy()
    num_plots = max(len(frame.columns), 1)
    axes = _prepare_panel_axes(num_plots, ax, ncols=3, width=18.0, height=6.0)

    if frame.empty:
        current_ax = axes[0]
        return _finalize_ax(current_ax, title="Monthly Mean IC", grid=False)

    frame.index = pd.MultiIndex.from_arrays(
        [frame.index.year, frame.index.month],
        names=["year", "month"],
    )

    rendered_axes = []
    for current_ax, (period, values) in zip(axes, frame.items()):
        heatmap_data = values.unstack()
        if heatmap_data.isna().all().all():
            current_ax.axis("off")
            current_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=current_ax.transAxes)
        else:
            sns.heatmap(
                heatmap_data,
                annot=True,
                alpha=1.0,
                center=0.0,
                annot_kws={"size": 7},
                linewidths=0.01,
                linecolor="white",
                cmap=cm.coolwarm_r,
                cbar=False,
                ax=current_ax,
            )
        current_ax.set_title(f"Monthly Mean {period} Period IC", pad=6)
        current_ax.set_xlabel("")
        current_ax.set_ylabel("")
        rendered_axes.append(current_ax)
    return _finalize_panel_axes(axes, rendered_axes)


def plot_quantile_returns_bar(mean_returns: pd.DataFrame, by_group: bool = False, ylim_percentiles=None, ax=None):
    if ylim_percentiles is not None:
        low = np.nanpercentile(mean_returns.to_numpy(dtype=float), ylim_percentiles[0]) * DECIMAL_TO_BPS
        high = np.nanpercentile(mean_returns.to_numpy(dtype=float), ylim_percentiles[1]) * DECIMAL_TO_BPS
    else:
        low = None
        high = None

    if by_group and isinstance(mean_returns.index, pd.MultiIndex) and "group" in mean_returns.index.names:
        groups = list(mean_returns.index.get_level_values("group").unique())
        axes = _prepare_panel_axes(len(groups), ax, ncols=2, width=12.0, height=4.0, sharey=True)
        for current_ax, group in zip(axes, groups):
            group_frame = mean_returns.xs(group, level="group").multiply(DECIMAL_TO_BPS)
            group_frame.plot(kind="bar", ax=current_ax, title=str(group))
            current_ax.set_xlabel("")
            current_ax.set_ylabel("Mean Return (bps)")
            current_ax.set_ylim(low, high)
        return axes[0] if len(groups) == 1 else axes

    ax = _get_ax(ax, figsize=(10, 4))
    average = mean_returns.multiply(DECIMAL_TO_BPS)
    if isinstance(average, pd.DataFrame):
        colors = _returns_palette_for_labels(average.columns)
        average.plot(kind="bar", ax=ax, color=colors)
    else:
        average.plot(kind="bar", ax=ax, color=RETURNS_OVERVIEW_PALETTE[0])
    ax.set_ylim(low, high)
    ax.set_ylabel("Mean Return (bps)")
    return _finalize_ax(ax, title="Mean Return By Quantile")


def plot_quantile_returns_violin(return_by_q: pd.DataFrame, ylim_percentiles=None, ax=None):
    ax = _get_ax(ax, figsize=(10, 4))
    frame = return_by_q.copy().multiply(DECIMAL_TO_BPS)
    if isinstance(frame.index, pd.MultiIndex):
        frame.columns = frame.columns.set_names("forward_periods")
        forward_periods = [str(column) for column in frame.columns]
        palette = _returns_palette_for_labels(forward_periods)
        violin_data = frame.stack().rename("return").reset_index()
        sns.violinplot(
            data=violin_data,
            x="factor_quantile",
            hue="forward_periods",
            y="return",
            orient="v",
            cut=0,
            inner="quartile",
            hue_order=forward_periods,
            palette=palette,
            ax=ax,
        )
        values = violin_data["return"].to_numpy(dtype=float)
    else:
        data = [frame[column].dropna().to_numpy() for column in frame.columns]
        ax.violinplot(data, showmeans=True, showextrema=False)
        ax.set_xticks(range(1, len(frame.columns) + 1))
        ax.set_xticklabels([str(column) for column in frame.columns])
        values = np.concatenate([v for v in data if len(v) > 0]) if data else np.array([])
    if len(values) > 0 and ylim_percentiles is not None:
        low, high = np.nanpercentile(values, ylim_percentiles)
        ax.set_ylim(low, high)
    ax.set_ylabel("Return (bps)")
    return _finalize_ax(ax, title="Period Wise Return By Factor Quantile")


def plot_mean_quantile_returns_spread_time_series(spread_returns: pd.Series, std_err=None, bandwidth=1, ax=None):
    if isinstance(spread_returns, pd.DataFrame):
        axes = ax
        if axes is None:
            axes = [None for _ in spread_returns.columns]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        ymin, ymax = None, None
        rendered_axes = []
        for current_ax, (name, series) in zip(list(axes), spread_returns.items()):
            std_series = None if std_err is None else std_err[name]
            rendered = plot_mean_quantile_returns_spread_time_series(series, std_err=std_series, bandwidth=bandwidth, ax=current_ax)
            rendered_axes.append(rendered)
            lo, hi = rendered.get_ylim()
            ymin = lo if ymin is None else min(ymin, lo)
            ymax = hi if ymax is None else max(ymax, hi)
        for rendered in rendered_axes:
            rendered.set_ylim([ymin, ymax])
        return rendered_axes

    if spread_returns.isnull().all():
        return ax if ax is not None else _get_ax(None)

    current_ax = _get_ax(ax, figsize=(12, 4))
    values = spread_returns.sort_index().mul(DECIMAL_TO_BPS)
    trend = _monthly_rolling_mean(values)
    base_color = RETURNS_OVERVIEW_PALETTE[3]
    trend_color = RETURNS_OVERVIEW_PALETTE[4]
    values.plot(alpha=0.45, ax=current_ax, lw=0.8, color=base_color)
    trend.plot(color=trend_color, alpha=0.8, ax=current_ax)
    current_ax.legend(["mean returns spread", "1 month moving avg"], loc="upper right")
    if std_err is not None:
        err = std_err.reindex(values.index).mul(DECIMAL_TO_BPS) * float(bandwidth)
        current_ax.fill_between(values.index, values - err, values + err, color=base_color, alpha=0.18)
    current_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    current_ax.set_ylabel("Difference In Quantile Mean Return (bps)")
    current_ax.set_title(f"Top Minus Bottom Quantile Mean Return ({getattr(spread_returns, 'name', '')})".strip())
    return current_ax


def plot_cumulative_returns(factor_returns: pd.Series, period=None, freq=None, title=None, ax=None):
    from tiger_factors.factor_evaluation.performance import cumulative_returns as compute_cumulative_returns

    ax = _get_ax(ax, figsize=(12, 4))
    cumulative = compute_cumulative_returns(factor_returns, period=period, freq=freq) + 1.0
    cumulative.plot(ax=ax, linewidth=2.0, color=RETURNS_OVERVIEW_PALETTE[5])
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Cumulative Returns")
    return _finalize_ax(ax, title=title or f"Portfolio Cumulative Return ({_period_label(period) if period is not None else '1D'} Fwd Period)")


def plot_cumulative_returns_by_quantile(quantile_returns: pd.DataFrame, period=None, freq=None, ax=None):
    from tiger_factors.factor_evaluation.performance import cumulative_returns as compute_cumulative_returns

    ax = _get_ax(ax, figsize=(12, 5))
    ret_wide = _normalize_quantile_wide_frame(quantile_returns)
    cum_ret = ret_wide.apply(lambda values: compute_cumulative_returns(values, period=period, freq=freq) + 1.0)
    cum_ret = cum_ret.loc[:, ::-1]
    cum_ret.plot(ax=ax, linewidth=1.6, color=_returns_palette_for_labels(cum_ret.columns))
    ax.legend(title="Quantile")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.7)
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set_ylabel("Log Cumulative Returns")
    ax.set_yscale("symlog")
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.set_ylim((ymin, ymax))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    return _finalize_ax(ax, title=f"Cumulative Return By Quantile ({_period_label(period) if period is not None else '1D'})")


def plot_top_bottom_quantile_turnover(quantile_turnover: pd.DataFrame, period=1, ax=None):
    if quantile_turnover.empty:
        ax = _get_ax(ax, figsize=(12, 4))
        return _finalize_ax(ax, title=f"{_period_label(period)} Period Top and Bottom Quantile Turnover")

    period_labels: list[str] = []
    period_frames: list[tuple[str, pd.DataFrame]] = []
    for column in quantile_turnover.columns:
        text = str(column)
        if text.startswith("top_"):
            label = text[len("top_") :]
        elif text.startswith("bottom_"):
            label = text[len("bottom_") :]
        else:
            period_labels = []
            period_frames = []
            break
        if label not in period_labels:
            period_labels.append(label)
    if period_labels and all(f"top_{label}" in quantile_turnover.columns and f"bottom_{label}" in quantile_turnover.columns for label in period_labels):
        for label in period_labels:
            turnover = pd.DataFrame(
                {
                    "top quantile turnover": quantile_turnover[f"top_{label}"],
                    "bottom quantile turnover": quantile_turnover[f"bottom_{label}"],
                }
            ).sort_index()
            period_frames.append((label, turnover))

    if period_frames:
        if ax is None:
            _, axes = plt.subplots(len(period_frames), 1, figsize=(14, 4 * len(period_frames)), squeeze=False)
            axis_list = axes.flatten().tolist()
        elif isinstance(ax, (list, np.ndarray)):
            axis_list = list(ax)
        else:
            axis_list = [ax]
        for current_ax, (label, turnover) in zip(axis_list, period_frames):
            turnover.plot(ax=current_ax, linewidth=1.2, alpha=0.75, color=[RETURNS_OVERVIEW_PALETTE[0], RETURNS_OVERVIEW_PALETTE[1]])
            current_ax.set_title(f"{label} Period Top and Bottom Quantile Turnover")
            current_ax.set_ylabel("Proportion Of Names New To Quantile")
            current_ax.legend(loc="best")
            current_ax.grid(True, alpha=0.25)
        return axis_list[0] if len(axis_list) == 1 else axis_list

    ax = _get_ax(ax, figsize=(12, 4))
    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame(
        {
            "top quantile turnover": quantile_turnover[max_quantile],
            "bottom quantile turnover": quantile_turnover[min_quantile],
        }
    )
    turnover.sort_index().plot(ax=ax, linewidth=1.2, alpha=0.6)
    ax.set_ylabel("Proportion Of Names New To Quantile")
    return _finalize_ax(ax, title=f"{_period_label(period)} Period Top and Bottom Quantile Turnover")


def plot_factor_rank_auto_correlation(factor_autocorrelation: pd.Series, period=1, ax=None):
    if isinstance(factor_autocorrelation, pd.DataFrame):
        columns = list(factor_autocorrelation.columns)
        current_ax = _get_ax(ax, figsize=(12, 4))
        palette = _returns_palette_for_labels(columns)
        for idx, column in enumerate(columns):
            series = pd.to_numeric(factor_autocorrelation[column], errors="coerce").dropna()
            if series.empty:
                continue
            label = f"{column} (mean {series.mean():.3f})"
            series.sort_index().plot(ax=current_ax, linewidth=1.5, color=palette[idx], label=label)
        current_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        current_ax.set_ylabel("Autocorrelation Coefficient")
        _safe_legend(current_ax)
        return _finalize_ax(current_ax, title="Factor Rank Autocorrelation")
    ax = _get_ax(ax, figsize=(12, 4))
    factor_autocorrelation.sort_index().plot(ax=ax, linewidth=1.5, color="#2a6fdb")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Autocorrelation Coefficient")
    ax.text(
        0.05,
        0.95,
        f"Mean {pd.to_numeric(factor_autocorrelation, errors='coerce').mean():.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 1.0, "pad": 4},
    )
    return _finalize_ax(ax, title=f"{_period_label(period)} Period Factor Rank Autocorrelation")


def plot_information_table(ic_data: pd.DataFrame, return_df: bool = False, ax=None):
    summary = pd.DataFrame(
        {
            "IC Mean": ic_data.mean(),
            "IC Std.": ic_data.std(ddof=0),
            "Risk-Adjusted IC": ic_data.mean() / ic_data.std(ddof=0).replace(0.0, np.nan),
            "t-stat(IC)": _safe_ttest_stat(ic_data, "statistic"),
            "p-value(IC)": _safe_ttest_stat(ic_data, "pvalue"),
            "IC Skew": _safe_stat_series(ic_data, lambda series: stats.skew(series, nan_policy="omit")),
            "IC Kurtosis": _safe_stat_series(ic_data, lambda series: stats.kurtosis(series, nan_policy="omit")),
        }
    ).round(4)
    if return_df:
        return summary
    ax = _get_ax(ax, figsize=(8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=summary.fillna("").values,
        rowLabels=summary.index.tolist(),
        colLabels=summary.columns.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    return ax


def plot_returns_table(
    alpha_beta: pd.DataFrame,
    mean_quantile_ret=None,
    mean_ret_spread_quant=None,
    return_df: bool = False,
    ax=None,
):
    table_df = alpha_beta.copy()
    if isinstance(mean_quantile_ret, pd.DataFrame):
        table_df.loc["Mean Period Wise Return Top Quantile (bps)"] = mean_quantile_ret.iloc[-1] * DECIMAL_TO_BPS
        table_df.loc["Mean Period Wise Return Bottom Quantile (bps)"] = mean_quantile_ret.iloc[0] * DECIMAL_TO_BPS
    if isinstance(mean_ret_spread_quant, pd.Series):
        table_df.loc["Mean Period Wise Spread (bps)"] = mean_ret_spread_quant.mean() * DECIMAL_TO_BPS
    if return_df:
        return table_df
    ax = _get_ax(ax, figsize=(8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.fillna("").round(4).values,
        rowLabels=table_df.index.tolist(),
        colLabels=table_df.columns.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    return ax


def plot_quantile_statistics_table(factor_data: pd.DataFrame, return_df: bool = False, ax=None):
    quantile_stats = factor_data
    if "factor_quantile" in factor_data.columns and "factor" in factor_data.columns:
        quantile_stats = factor_data.groupby("factor_quantile")["factor"].agg(["min", "max", "mean", "std", "count"])
        quantile_stats["count %"] = quantile_stats["count"] / quantile_stats["count"].sum() * 100.0
    if return_df:
        return quantile_stats
    ax = _get_ax(ax, figsize=(9, 3))
    ax.axis("off")
    table = ax.table(
        cellText=quantile_stats.fillna("").round(4).values,
        rowLabels=[str(index) for index in quantile_stats.index],
        colLabels=quantile_stats.columns.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    return ax


def plot_turnover_table(autocorrelation_data, quantile_turnover=None, return_df: bool = False, ax=None):
    turnover_stats = autocorrelation_data
    auto_corr = None
    if isinstance(quantile_turnover, dict):
        turnover_stats = pd.DataFrame()
        for period in sorted(quantile_turnover.keys()):
            for quantile, p_data in quantile_turnover[period].items():
                turnover_stats.loc[f"Quantile {quantile} Mean Turnover", _period_label(period)] = pd.to_numeric(
                    p_data,
                    errors="coerce",
                ).mean()
        auto_corr = pd.DataFrame()
        for period, p_data in autocorrelation_data.items():
            auto_corr.loc["Mean Factor Rank Autocorrelation", _period_label(period)] = pd.to_numeric(
                p_data,
                errors="coerce",
            ).mean()
    if return_df:
        return (turnover_stats, auto_corr) if auto_corr is not None else turnover_stats
    if auto_corr is not None:
        turnover_stats = pd.concat([turnover_stats, auto_corr], axis=0)
    ax = _get_ax(ax, figsize=(9, 3))
    ax.axis("off")
    table = ax.table(
        cellText=turnover_stats.fillna("").round(4).values,
        rowLabels=[str(index) for index in turnover_stats.index],
        colLabels=turnover_stats.columns.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    return ax


def plot_ic_by_group(group_ic: pd.DataFrame, ax=None):
    ax = _get_ax(ax, figsize=(10, 4))
    group_ic.plot(kind="bar", ax=ax)
    ax.set_title("Information Coefficient By Group")
    ax.set_xlabel("")
    ax.set_xticklabels(group_ic.index, rotation=45)
    return ax


def plot_events_distribution(events: pd.Series, num_bars=50, ax=None):
    ax = _get_ax(ax, figsize=(10, 4))
    if isinstance(events.index, pd.MultiIndex):
        date_level = events.index.names[0] or "date_"
        date_values = pd.to_datetime(events.index.get_level_values(date_level), errors="coerce")
    else:
        date_values = pd.to_datetime(events.index, errors="coerce")
        if date_values.isna().all():
            date_values = pd.to_datetime(events, errors="coerce")
    date_values = pd.DatetimeIndex(date_values).dropna()
    if date_values.empty:
        return _finalize_ax(ax, title="Events Distribution")
    group_interval = (date_values.max() - date_values.min()) / max(int(num_bars), 1)
    if group_interval <= pd.Timedelta(0):
        counts = pd.Series(1, index=date_values).groupby(date_values).sum()
    else:
        counts = pd.Series(1, index=date_values).groupby(pd.Grouper(freq=group_interval)).sum()
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of events")
    ax.set_xlabel("Date")
    return _finalize_ax(ax, title="Distribution of events in time", grid=False)


def plot_quantile_average_cumulative_return(
    average_cumulative_return: pd.DataFrame,
    std_frame: pd.DataFrame | None = None,
    by_quantile: bool = False,
    std_bar: bool = False,
    title: str | None = None,
    ax=None,
):
    frame = average_cumulative_return.copy().multiply(DECIMAL_TO_BPS)
    quantile_level = "factor_quantile" if isinstance(frame.index, pd.MultiIndex) and "factor_quantile" in frame.index.names else None
    quantiles = len(frame.index.get_level_values(quantile_level).unique()) if quantile_level else len(frame.columns)
    palette = _returns_palette_for_labels(range(max(quantiles, 1)))

    if by_quantile:
        axes = _prepare_panel_axes(quantiles, ax, ncols=2, width=18.0, height=6.0)

        rendered_axes = []
        grouped = frame.groupby(level=quantile_level) if quantile_level else [(column, frame[[column]].T) for column in frame.columns]
        for idx, (quantile, q_ret) in enumerate(grouped):
            current_ax = axes[idx]
            mean = q_ret.xs("mean", level=-1) if isinstance(q_ret.index, pd.MultiIndex) and "mean" in q_ret.index.get_level_values(-1) else q_ret
            mean = mean.iloc[0] if isinstance(mean, pd.DataFrame) else mean
            mean.name = f"Quantile {quantile}"
            mean.plot(ax=current_ax, color=palette[idx])
            current_ax.set_ylabel("Mean Return (bps)")
            if std_bar and isinstance(q_ret.index, pd.MultiIndex) and "std" in q_ret.index.get_level_values(-1):
                std = q_ret.xs("std", level=-1)
                std = std.iloc[0] if isinstance(std, pd.DataFrame) else std
                current_ax.errorbar(std.index, mean, yerr=std, fmt="none", ecolor=palette[idx], label="none")
            current_ax.axvline(x=0, color="k", linestyle="--")
            _safe_legend(current_ax)
            rendered_axes.append(current_ax)
        return _finalize_panel_axes(axes, rendered_axes)

    current_ax = _get_ax(ax, figsize=(12, 5))
    grouped = frame.groupby(level=quantile_level) if quantile_level else [(column, frame[[column]].T) for column in frame.columns]
    for idx, (quantile, q_ret) in enumerate(grouped):
        mean = q_ret.xs("mean", level=-1) if isinstance(q_ret.index, pd.MultiIndex) and "mean" in q_ret.index.get_level_values(-1) else q_ret
        mean = mean.iloc[0] if isinstance(mean, pd.DataFrame) else mean
        mean.name = f"Quantile {quantile}"
        mean.plot(ax=current_ax, color=palette[idx])
        if std_bar and isinstance(q_ret.index, pd.MultiIndex) and "std" in q_ret.index.get_level_values(-1):
            std = q_ret.xs("std", level=-1)
            std = std.iloc[0] if isinstance(std, pd.DataFrame) else std
            current_ax.errorbar(std.index, mean, yerr=std, fmt="none", ecolor=palette[idx], label="none")
    current_ax.axvline(x=0, color="k", linestyle="--")
    _safe_legend(current_ax)
    current_ax.set_ylabel("Mean Return (bps)")
    current_ax.set_xlabel("Periods")
    return _finalize_ax(current_ax, title=title or "Average Cumulative Returns by Quantile")


def plot_series_line(
    series: pd.Series,
    *,
    title: str | None = None,
    color: str = "#2a6fdb",
    label: str = "series",
    rolling_window: int = 20,
    ax=None,
):
    ax = _get_ax(ax, figsize=(12, 4))
    clean = _clean_numeric_series(series, sort_index=True)
    if not clean.empty:
        clean.plot(linewidth=1.2, alpha=0.85, label=label, color=color, ax=ax)
        rolling = clean.rolling(max(int(rolling_window), 1)).mean()
        rolling.plot(linewidth=2.0, label=f"{int(max(rolling_window, 1))}D rolling mean", color="#333333", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.25)
    return _finalize_ax(ax, title=title or "")


def plot_series_histogram(
    series: pd.Series,
    *,
    title: str | None = None,
    color: str = "#2a6fdb",
    ax=None,
):
    ax = _get_ax(ax, figsize=(8, 4))
    clean = _clean_numeric_series(series)
    try:
        if clean.empty:
            ax.hist([], bins=1, alpha=0.75, color=color, edgecolor="white")
        elif clean.nunique(dropna=True) <= 1:
            center = float(clean.iloc[0])
            width = max(abs(center) * 0.1, 1e-6)
            ax.hist(clean, bins=[center - width, center + width], alpha=0.75, color=color, edgecolor="white")
        else:
            ax.hist(clean, bins=30, alpha=0.75, color=color, edgecolor="white")
    except ValueError:
        center = float(clean.mean()) if not clean.empty else 0.0
        width = max(abs(center) * 0.1, 1e-6)
        ax.hist(clean, bins=[center - width, center + width], alpha=0.75, color=color, edgecolor="white")
    ax.axvline(clean.mean() if not clean.empty else 0.0, color="black", linestyle="--", linewidth=1.2, label="mean")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    return _finalize_ax(ax, title=title or "")


def plot_drawdown(
    returns: pd.Series,
    *,
    title: str | None = None,
    color: str = "#6a4c93",
    ax=None,
):
    ax = _get_ax(ax, figsize=(12, 4))
    clean = _clean_numeric_series(returns, dropna=False, fillna=0.0, sort_index=True)
    cumulative = (1.0 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    drawdown.index = pd.DatetimeIndex(drawdown.index)
    drawdown.plot(linewidth=1.8, color=color, ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.25)
    return _finalize_ax(ax, title=title or "")


def plot_rolling_sharpe(
    returns: pd.Series,
    *,
    title: str | None = None,
    annualization: int = 252,
    color: str = "#1f7a8c",
    ax=None,
):
    ax = _get_ax(ax, figsize=(12, 4))
    clean = _clean_numeric_series(returns, dropna=False, fillna=0.0, sort_index=True)
    rolling_mean = clean.rolling(int(max(annualization, 1))).mean()
    rolling_std = clean.rolling(int(max(annualization, 1))).std(ddof=0)
    sharpe = (rolling_mean / rolling_std) * np.sqrt(int(max(annualization, 1)))
    sharpe.replace([np.inf, -np.inf], np.nan).plot(linewidth=1.8, color=color, ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.25)
    return _finalize_ax(ax, title=title or "")


def plot_monthly_heatmap(
    returns: pd.Series,
    *,
    title: str | None = None,
    ax=None,
):
    ax = _get_ax(ax, figsize=(12, 4))
    clean = _clean_numeric_series(returns, sort_index=True)
    if clean.empty:
        return _finalize_ax(ax, title=title or "")
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index, errors="coerce")
        clean = clean[~clean.index.isna()]
    if clean.empty:
        return _finalize_ax(ax, title=title or "")
    monthly = clean.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    heatmap = table.pivot(index="year", columns="month", values="return").sort_index()
    if heatmap.empty:
        return _finalize_ax(ax, title=title or "")
    im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Monthly return")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    return _finalize_ax(ax, title=title or "")


def plot_regression_scatter(
    x: pd.Series,
    y: pd.Series,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    title: str | None = None,
    ax=None,
):
    ax = _get_ax(ax, figsize=(7, 5))
    joined = pd.concat([_clean_numeric_series(x, dropna=False), _clean_numeric_series(y, dropna=False)], axis=1).dropna()
    if joined.empty:
        return _finalize_ax(ax, title=title or "")
    x_values = joined.iloc[:, 0].to_numpy(dtype=float)
    y_values = joined.iloc[:, 1].to_numpy(dtype=float)
    ax.scatter(x_values, y_values, s=14, alpha=0.45, color="#2a6fdb", edgecolor="none")
    x_line = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), 100)
    if alpha is not None and beta is not None:
        ax.plot(x_line, alpha + beta * x_line, color="#c44536", linewidth=2.0, label=f"alpha={alpha:.4f}, beta={beta:.4f}")
        _safe_legend(ax)
    ax.grid(True, alpha=0.25)
    return _finalize_ax(ax, title=title or "")


def plot_series_barh(
    series: pd.Series,
    *,
    title: str | None = None,
    color: str = "#2a6fdb",
    xlabel: str | None = None,
    ax=None,
):
    ax = _get_ax(ax, figsize=(12, 5))
    clean = _clean_numeric_series(series).sort_values()
    if not clean.empty:
        clean.plot(kind="barh", color=color, ax=ax)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(True, axis="x", alpha=0.25)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return _finalize_ax(ax, title=title or "")


def save_figure(ax, output: Path) -> None:
    figure = ax.figure if hasattr(ax, "figure") else np.ravel(ax)[0].figure
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


__all__ = [
    "DECIMAL_TO_BPS",
    "axes_style",
    "customize",
    "plot_cumulative_returns",
    "plot_cumulative_returns_by_quantile",
    "plot_events_distribution",
    "plot_factor_rank_auto_correlation",
    "plot_ic_by_group",
    "plot_ic_hist",
    "plot_ic_qq",
    "plot_ic_rolling",
    "plot_ic_ts",
    "plot_information_table",
    "plot_mean_quantile_returns_spread_time_series",
    "plot_monthly_ic_heatmap",
    "plot_quantile_average_cumulative_return",
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_regression_scatter",
    "plot_rolling_sharpe",
    "plot_series_barh",
    "plot_series_histogram",
    "plot_series_line",
    "plot_quantile_statistics_table",
    "plot_returns_table",
    "plot_top_bottom_quantile_turnover",
    "plot_turnover_table",
    "plotting_context",
    "save_figure",
]
