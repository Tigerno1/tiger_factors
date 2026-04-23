"""Alpha-hacking posterior demo.

This example compares an in-sample alpha path with a held-out OOS path and
updates the posterior alpha recursively, giving extra weight to OOS when the
estimated hacking bias grows.
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

from tiger_factors.multifactor_evaluation import alpha_hacking_bayesian_update


def _synthetic_hacking_series() -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=360, freq="ME")
    base = 0.001 + np.linspace(0.0, 0.0025, len(dates))
    cycle = 0.0008 * np.sin(np.linspace(0.0, 8.0 * np.pi, len(dates)))
    rng = np.random.default_rng(19)
    in_sample = base + cycle + 0.0010 + rng.normal(0.0, 0.0008, len(dates))
    out_of_sample = base + cycle + rng.normal(0.0, 0.0008, len(dates))
    return pd.DataFrame({"date_": dates, "in_sample_alpha": in_sample, "out_of_sample_alpha": out_of_sample})


def main() -> None:
    frame = _synthetic_hacking_series()
    result = alpha_hacking_bayesian_update(frame, date_column="date_", in_sample_column="in_sample_alpha", out_of_sample_column="out_of_sample_alpha", alpha=0.05)
    table = result["table"]

    output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "bayes"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "alpha_hacking_posterior_demo.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(table["date_"], table["in_sample_alpha"], color="#8d99ae", linewidth=1.2, label="In-sample alpha")
    ax1.plot(table["date_"], table["out_of_sample_alpha"], color="#e76f51", linewidth=1.2, label="OOS alpha")
    ax1.plot(table["date_"], table["posterior_alpha"], color="#1d3557", linewidth=2.0, label="Posterior alpha")
    ax1.fill_between(
        table["date_"],
        table["posterior_alpha"] - 1.96 * table["posterior_alpha_sd"],
        table["posterior_alpha"] + 1.96 * table["posterior_alpha_sd"],
        color="#1d3557",
        alpha=0.15,
        label="Posterior 95% band",
    )
    ax1.set_title("Alpha Hacking Posterior Update")
    ax1.set_ylabel("Alpha")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.2)

    ax2.plot(table["date_"], table["hacking_bias"], color="#2a9d8f", linewidth=1.5, label="Estimated hacking bias")
    ax2.plot(table["date_"], table["out_of_sample_weight"], color="#457b9d", linewidth=1.5, label="OOS weight")
    ax2.set_ylabel("Bias / Weight")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)

    summary = pd.Series(
        {
            "average_in_sample_alpha": result["average_in_sample_alpha"],
            "average_out_of_sample_alpha": result["average_out_of_sample_alpha"],
            "average_posterior_alpha": result["average_posterior_alpha"],
            "average_hacking_bias": result["average_hacking_bias"],
            "bayesian_fdr": result["bayesian_fdr"],
            "bayesian_fwer": result["bayesian_fwer"],
            "discovered_count": result["discovered_count"],
            "figure_path": str(figure_path),
        }
    )
    print(summary.to_string())
    print("\nalpha hacking tail:")
    print(table.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
