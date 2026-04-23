"""Hierarchical Bayesian multiple-testing demo.

This example compares classical multiple-testing adjustments, empirical-Bayes
local-FDR, and a cluster-aware hierarchical Bayes variant.
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

from tiger_factors.multifactor_evaluation import bayesian_fdr
from tiger_factors.multifactor_evaluation import benjamini_hochberg
from tiger_factors.multifactor_evaluation import benjamini_yekutieli
from tiger_factors.multifactor_evaluation import bonferroni_adjust
from tiger_factors.multifactor_evaluation import fit_bayesian_mixture
from tiger_factors.multifactor_evaluation import fit_hierarchical_bayesian_mixture
from tiger_factors.multifactor_evaluation import hierarchical_bayesian_fdr
from tiger_factors.multifactor_evaluation import holm_adjust
from tiger_factors.multifactor_evaluation import storey_qvalues


def _sample_factor_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor_name": [
                "quality",
                "momentum",
                "value",
                "low_risk",
                "profitability",
                "investment",
                "size",
                "accruals",
                "seasonality",
                "liquidity",
            ],
            "p_value": [0.0002, 0.004, 0.018, 0.031, 0.055, 0.12, 0.19, 0.34, 0.58, 0.82],
            "statistic": [4.1, 3.0, 2.4, 2.1, 1.9, 1.3, 1.0, 0.4, 0.1, -0.2],
            "fitness": [2.3, 1.8, 1.1, 0.9, 0.5, 0.2, 0.1, -0.1, -0.2, -0.3],
        }
    )


def _sample_factor_dict() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    panel_a = pd.DataFrame({"A": np.linspace(0.2, 1.2, len(dates)), "B": np.linspace(0.1, 1.0, len(dates))}, index=dates)
    panel_b = pd.DataFrame({"A": np.linspace(0.25, 1.1, len(dates)), "B": np.linspace(0.05, 0.9, len(dates))}, index=dates)
    panel_c = pd.DataFrame({"A": np.linspace(-0.3, 0.2, len(dates)), "B": np.linspace(-0.2, 0.15, len(dates))}, index=dates)
    panel_d = pd.DataFrame({"A": np.linspace(-0.1, 0.1, len(dates)), "B": np.linspace(-0.05, 0.05, len(dates))}, index=dates)
    return {
        "quality": panel_a,
        "momentum": panel_b,
        "value": panel_c,
        "low_risk": panel_d,
    }


def main() -> None:
    frame = _sample_factor_metrics()
    factor_dict = _sample_factor_dict()

    classical = pd.DataFrame(
        {
            "factor_name": frame["factor_name"],
            "bonferroni_reject": bonferroni_adjust(frame, alpha=0.05).table.set_index("factor_name")["rejected"].reindex(frame["factor_name"]).to_numpy(),
            "holm_reject": holm_adjust(frame, alpha=0.05).table.set_index("factor_name")["rejected"].reindex(frame["factor_name"]).to_numpy(),
            "bh_reject": benjamini_hochberg(frame, alpha=0.05).table.set_index("factor_name")["rejected"].reindex(frame["factor_name"]).to_numpy(),
            "by_reject": benjamini_yekutieli(frame, alpha=0.05).table.set_index("factor_name")["rejected"].reindex(frame["factor_name"]).to_numpy(),
            "storey_reject": storey_qvalues(frame, alpha=0.05).table.set_index("factor_name")["rejected"].reindex(frame["factor_name"]).to_numpy(),
        }
    )

    empirical = fit_bayesian_mixture(frame, alpha=0.05)
    hierarchical = fit_hierarchical_bayesian_mixture(frame, alpha=0.05, factor_dict=factor_dict, cluster_threshold=0.75)

    empirical_table = empirical["table"].loc[
        :,
        [
            "factor_name",
            "posterior_null_prob",
            "posterior_signal_prob",
            "posterior_mean",
            "discovered",
        ],
    ]
    hierarchical_table = hierarchical["table"].loc[
        :,
        [
            "factor_name",
            "cluster",
            "hierarchical_null_prob",
            "hierarchical_signal_prob",
            "hierarchical_posterior_mean",
            "hierarchical_discovered",
        ],
    ]

    print("classical adjustments:")
    print(classical.to_string(index=False))
    print("\nempirical Bayes params:")
    print(
        pd.Series(
            {
                "pi0": empirical["pi0"],
                "alt_variance": empirical["alt_variance"],
                "prior_variance": empirical["prior_variance"],
                "bayesian_fdr": bayesian_fdr(empirical["result"]),
                "discovered_count": empirical["result"].discovered_count,
                "expected_false_discoveries": empirical["result"].expected_false_discoveries,
            }
        ).to_string()
    )
    print("\nempirical Bayes posterior table:")
    print(empirical_table.to_string(index=False))
    print("\nhierarchical Bayes params:")
    print(
        pd.Series(
            {
                "factor_pi0": hierarchical["factor_pi0"],
                "cluster_pi0": hierarchical["cluster_pi0"],
                "factor_alt_variance": hierarchical["factor_fit"]["alt_variance"],
                "cluster_alt_variance": hierarchical["result"].cluster_alt_variance,
                "hierarchical_fdr": hierarchical_bayesian_fdr(hierarchical["result"]),
            }
        ).to_string()
    )
    print("\nhierarchical Bayes posterior table:")
    print(hierarchical_table.to_string(index=False))


if __name__ == "__main__":
    main()
