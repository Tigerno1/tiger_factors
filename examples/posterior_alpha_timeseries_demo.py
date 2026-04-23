"""Rolling Bayesian alpha time-series demo.

This example shows how to turn a noisy alpha series into a trailing-window
posterior-alpha trajectory with empirical-Bayes shrinkage, and compares the
result with the classical rolling OLS alpha.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from tiger_factors.multifactor_evaluation import rolling_bayesian_alpha
from tiger_factors.multifactor_evaluation import dynamic_bayesian_alpha


def _synthetic_alpha_series() -> pd.DataFrame:
    dates = pd.date_range("1960-01-31", periods=780, freq="ME")
    trend = np.linspace(0.0005, 0.0040, len(dates))
    cycle = 0.0012 * np.sin(np.linspace(0.0, 10.0 * np.pi, len(dates)))
    regime = np.where((np.arange(len(dates)) // 120) % 2 == 0, 0.0008, -0.0004)
    noise = np.random.default_rng(7).normal(loc=0.0, scale=0.0010, size=len(dates))
    alpha = trend + cycle + regime + noise
    return pd.DataFrame({"date_": dates, "alpha": alpha})


def main() -> None:
    frame = _synthetic_alpha_series()
    rolling_result = rolling_bayesian_alpha(
        frame,
        date_column="date_",
        alpha_column="alpha",
        window=60,
        min_periods=36,
        alpha=0.05,
    )
    dynamic_result = dynamic_bayesian_alpha(
        frame,
        date_column="date_",
        alpha_column="alpha",
        process_discount=0.985,
        alpha=0.05,
    )
    rolling_table = rolling_result["table"]
    dynamic_table = dynamic_result["table"]

    output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "bayes"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "posterior_alpha_timeseries_demo.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(rolling_table["window_end"], rolling_table["ols_alpha"], color="#8d99ae", linewidth=1.2, label="Rolling OLS Alpha")
    ax1.plot(rolling_table["window_end"], rolling_table["posterior_alpha"], color="#1d3557", linewidth=1.8, label="Rolling Posterior Alpha")
    ax1.plot(dynamic_table["date_"], dynamic_table["posterior_alpha"], color="#2a9d8f", linewidth=2.0, label="Dynamic Posterior Alpha")
    ax1.fill_between(
        dynamic_table["date_"],
        dynamic_table["posterior_alpha"] - 1.96 * dynamic_table["posterior_alpha_sd"],
        dynamic_table["posterior_alpha"] + 1.96 * dynamic_table["posterior_alpha_sd"],
        color="#2a9d8f",
        alpha=0.12,
        label="Dynamic 95% band",
    )
    ax1.set_ylabel("Alpha")
    ax1.set_title("Rolling Bayesian Alpha vs. Dynamic Bayesian Alpha")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.2)

    ax2.bar(rolling_table["window_end"], rolling_table["bayesian_fdr"], width=18, color="#2a9d8f", alpha=0.65, label="Rolling Bayesian FDR")
    ax2.plot(rolling_table["window_end"], rolling_table["bayesian_fwer"], color="#e76f51", linewidth=1.6, label="Rolling Bayesian FWER")
    ax2.plot(dynamic_table["date_"], dynamic_table["posterior_null_prob"], color="#457b9d", linewidth=1.3, linestyle="--", label="Dynamic Null Prob")
    ax2.set_ylabel("Error Rate")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)

    summary = pd.Series(
        {
            "window": rolling_result["result"].window,
            "min_periods": rolling_result["result"].min_periods,
            "rolling_average_ols_alpha": rolling_result["average_ols_alpha"],
            "rolling_average_posterior_alpha": rolling_result["average_posterior_alpha"],
            "rolling_bayesian_fdr": rolling_result["bayesian_fdr"],
            "rolling_bayesian_fwer": rolling_result["bayesian_fwer"],
            "rolling_discovered_count": rolling_result["discovered_count"],
            "dynamic_average_ols_alpha": dynamic_result["average_ols_alpha"],
            "dynamic_average_posterior_alpha": dynamic_result["average_posterior_alpha"],
            "dynamic_bayesian_fdr": dynamic_result["bayesian_fdr"],
            "dynamic_bayesian_fwer": dynamic_result["bayesian_fwer"],
            "dynamic_discovered_count": dynamic_result["discovered_count"],
            "figure_path": str(figure_path),
        }
    )
    print(summary.to_string())
    print("\nrolling table tail:")
    print(rolling_table.tail(8).to_string(index=False))
    print("\ndynamic table tail:")
    print(dynamic_table.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
