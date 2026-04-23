"""Grouped factor engine -> validation -> backtest -> portfolio report demo.

This example shows the full handoff for a family-aware factor workflow:

1. define a few reusable member factors
2. combine them into named factor families with ``FactorGroupEngine``
3. validate the resulting family evidence with single-factor and family-level
   validation helpers
4. run a wide-panel backtest on one of the group scores
5. render a local Tiger portfolio report
"""

from __future__ import annotations

import os
from pathlib import Path
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

from tiger_factors.factor_evaluation import validate_series
from tiger_factors.factor_frame import FactorDefinition
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import FactorGroupEngine
from tiger_factors.factor_frame import FactorGroupSpec
from tiger_factors.multifactor_evaluation import run_factor_backtest
from tiger_factors.multifactor_evaluation import validate_factor_family
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


class _PanelDefinition(FactorDefinition):
    def __init__(self, name: str, panel: pd.DataFrame) -> None:
        super().__init__(name=name)
        self._panel = panel

    def compute(self, ctx, state):  # type: ignore[override]
        return self._panel


def _build_sample_member_panels() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=48)
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rng = np.random.default_rng(17)

    momentum = pd.DataFrame(index=dates, columns=codes, dtype=float)
    value = pd.DataFrame(index=dates, columns=codes, dtype=float)
    quality = pd.DataFrame(index=dates, columns=codes, dtype=float)

    base_levels = np.array([0.6, 0.8, 1.0, 1.2, 0.4, 0.5], dtype=float)
    for i, date in enumerate(dates):
        noise = rng.normal(0.0, 0.04, size=len(codes))
        momentum.loc[date] = base_levels + 0.08 * i + noise
        value.loc[date] = (len(codes) - np.arange(len(codes))) + 0.03 * i + rng.normal(0.0, 0.04, size=len(codes))
        quality.loc[date] = 0.55 * momentum.loc[date].to_numpy(dtype=float) - 0.28 * value.loc[date].to_numpy(dtype=float) + rng.normal(0.0, 0.03, size=len(codes))

    for panel in (momentum, value, quality):
        panel.index.name = "date_"
        panel.columns = panel.columns.astype(str)

    return {
        "momentum": momentum,
        "value": value,
        "quality": quality,
    }


def _build_sample_close_panel(member_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    momentum = member_panels["momentum"]
    value = member_panels["value"]
    quality = member_panels["quality"]

    returns = (
        0.0007
        + 0.0014 * momentum
        - 0.0009 * value
        + 0.0010 * quality
        + pd.DataFrame(rng.normal(0.0, 0.004, size=momentum.shape), index=momentum.index, columns=momentum.columns)
    )
    close_panel = (1.0 + returns.fillna(0.0)).cumprod() * 100.0
    close_panel.index.name = "date_"
    return close_panel


def _build_engine(member_panels: dict[str, pd.DataFrame]) -> FactorGroupEngine:
    registry = FactorDefinitionRegistry()
    registry.register_many(
        _PanelDefinition("momentum", member_panels["momentum"]),
        _PanelDefinition("value", member_panels["value"]),
        _PanelDefinition("quality", member_panels["quality"]),
    )
    engine = FactorGroupEngine(definition_registry=registry)
    engine.register_groups(
        FactorGroupSpec(
            name="growth_family",
            members=("momentum", "quality"),
            weights={"momentum": 0.6, "quality": 0.4},
            combine_method="weighted_sum",
            metadata={"theme": "growth"},
        ),
        FactorGroupSpec(
            name="valuation_family",
            members=("value", "quality"),
            combine_method="rank_mean",
            metadata={"theme": "value"},
        ),
        FactorGroupSpec(
            name="core_family",
            members=("momentum", "value", "quality"),
            combine_method="zscore_mean",
            metadata={"theme": "core"},
        ),
    )
    return engine


def main() -> None:
    output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "grouped_factor_engine_demo"
    member_panels = _build_sample_member_panels()
    close_panel = _build_sample_close_panel(member_panels)
    engine = _build_engine(member_panels)
    result = engine.run(object())

    validation_rows: list[dict[str, object]] = []
    for group_name, panel in result.group_panels.items():
        if panel.empty:
            continue
        daily_mean = panel.mean(axis=1).dropna()
        validation = validate_series(
            daily_mean,
            metric_name=group_name,
            n_bootstrap=64,
            n_permutations=64,
            random_state=13,
        )
        validation_rows.append(
            {
                "factor_name": group_name,
                "p_value": validation.p_value,
                "fitness": abs(validation.observed),
            }
        )

    family_report = validate_factor_family(
        pd.DataFrame(validation_rows),
        method="bh",
        alpha=0.05,
        cluster_threshold=0.65,
        factor_dict=result.member_panels,
    )

    backtest, stats = run_factor_backtest(
        result.group_panels["core_family"],
        close_panel,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=True,
    )

    report = run_portfolio_from_backtest(
        backtest,
        output_dir=output_dir,
        report_name="grouped_factor_engine",
    )

    print("group summary:")
    print(result.summary.to_string(index=False))
    print("\nvalidation table:")
    print(family_report["table"].to_string(index=False))
    print("\nbacktest stats:")
    print(pd.DataFrame(stats).T.to_string())
    if report is not None:
        print("\nportfolio report:")
        print(report.to_summary())


if __name__ == "__main__":
    main()
