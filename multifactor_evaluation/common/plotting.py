from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


def configure_quantstats_style(*, font_family: str = "DejaVu Sans") -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "font.family": [font_family],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#d0d7de",
            "axes.labelcolor": "#111111",
            "text.color": "#111111",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "axes.titleweight": "regular",
            "axes.titlelocation": "left",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.autolayout": False,
        },
    )


def finalize_quantstats_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str = "",
    ylabel: str | None = None,
    legend: bool = False,
    legend_loc: str = "best",
    percent_y: bool = False,
    percent_x: bool = False,
    x_rotation: int = 0,
) -> None:
    if title is not None:
        ax.set_title(title, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)
    if legend:
        ax.legend(loc=legend_loc, fontsize=9)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if percent_x:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if x_rotation:
        for label in ax.get_xticklabels():
            label.set_rotation(x_rotation)
            label.set_ha("right")
    ax.tick_params(axis="both", labelsize=9)
    sns.despine(ax=ax, trim=True)
    ax.margins(x=0.01, y=0.05)


def save_quantstats_figure(path: Path, *, dpi: int = 150) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
