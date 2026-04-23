"""Bayesian multiple-testing demo.

This example compares classical multiple-testing adjustments with the
empirical-Bayes / local-FDR family-level validator.
"""

from __future__ import annotations

from pathlib import Path
import sys

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


def main() -> None:
    frame = _sample_factor_metrics()

    classical = pd.DataFrame(
        {
            "bonferroni_reject": bonferroni_adjust(frame, alpha=0.05).table.set_index("factor_name")["rejected"],
            "holm_reject": holm_adjust(frame, alpha=0.05).table.set_index("factor_name")["rejected"],
            "bh_reject": benjamini_hochberg(frame, alpha=0.05).table.set_index("factor_name")["rejected"],
            "by_reject": benjamini_yekutieli(frame, alpha=0.05).table.set_index("factor_name")["rejected"],
            "storey_reject": storey_qvalues(frame, alpha=0.05).table.set_index("factor_name")["rejected"],
        }
    ).reset_index()

    bayes = fit_bayesian_mixture(frame, alpha=0.05)
    bayes_table = bayes["table"].loc[
        :,
        [
            "factor_name",
            "p_value",
            "statistic",
            "posterior_null_prob",
            "posterior_signal_prob",
            "posterior_mean",
            "discovered",
        ],
    ]

    print("classical adjustments:")
    print(classical.to_string(index=False))
    print("\nbayesian mixture params:")
    print(
        pd.Series(
            {
                "pi0": bayes["pi0"],
                "alt_variance": bayes["alt_variance"],
                "prior_variance": bayes["prior_variance"],
                "bayesian_fdr": bayesian_fdr(bayes["result"]),
                "discovered_count": bayes["result"].discovered_count,
                "expected_false_discoveries": bayes["result"].expected_false_discoveries,
            }
        ).to_string()
    )
    print("\nbayesian posterior table:")
    print(bayes_table.to_string(index=False))


if __name__ == "__main__":
    main()
