"""Run screener first, then feed the selected factors into multifactor evaluation.

This demo shows the intended boundary between the two modules:

1. ``factor_screener`` filters a candidate factor basket
2. ``multifactor_evaluation`` takes the selected factors and builds the
   composite backtest plus report

The script expects the factor evaluation artifacts to already exist in the
local Tiger factor store. It is intentionally explicit so you can swap in your
own factor names, providers, and universe controls.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import FactorMarginalSelectionConfig
from tiger_factors.factor_screener import FactorScreener
from tiger_factors.factor_screener import FactorScreenerSpec
from tiger_factors.factor_allocation import allocate_from_return_panel
from tiger_factors.factor_backtest import run_return_backtest
from tiger_factors.multifactor_evaluation.reporting.analysis_report import create_analysis_report


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "factor_screener_multifactor_demo"
DEFAULT_FACTOR_NAMES = (
    "alpha_021",
    "alpha_030",
    "alpha_047",
    "alpha_005",
    "alpha_076",
    "alpha_068",
    "alpha_066",
    "alpha_092",
)

@dataclass(frozen=True)
class SingleFactorScreeningConfig:
    min_fitness: float | None = 0.10
    min_ic_mean: float | None = 0.01
    min_rank_ic_mean: float | None = 0.01
    min_sharpe: float | None = 0.40
    max_turnover: float | None = 0.50
    min_decay_score: float | None = 0.20
    min_capacity_score: float | None = 0.20
    max_correlation_penalty: float | None = 0.60
    min_regime_robustness: float | None = 0.60
    min_out_of_sample_stability: float | None = 0.60
    sort_field: str = "fitness"
    tie_breaker_field: str = "ic_ir"

    def metric_filter_config(self) -> FactorMetricFilterConfig:
        return FactorMetricFilterConfig(
            min_fitness=self.min_fitness,
            min_ic_mean=self.min_ic_mean,
            min_rank_ic_mean=self.min_rank_ic_mean,
            min_sharpe=self.min_sharpe,
            max_turnover=self.max_turnover,
            min_decay_score=self.min_decay_score,
            min_capacity_score=self.min_capacity_score,
            max_correlation_penalty=self.max_correlation_penalty,
            min_regime_robustness=self.min_regime_robustness,
            min_out_of_sample_stability=self.min_out_of_sample_stability,
            sort_field=self.sort_field,
            tie_breaker_field=self.tie_breaker_field,
        )


@dataclass(frozen=True)
class CorrelationScreeningConfig:
    ic_horizon: int = 1
    ic_min_names: int | None = 10
    marginal_gain_score_fields: tuple[str, ...] = (
        "directional_fitness",
        "directional_ic_ir",
        "directional_sharpe",
    )
    marginal_gain_score_weights: tuple[float, ...] = (0.50, 0.25, 0.25)
    marginal_gain_penalty_fields: tuple[str, ...] = ("turnover", "max_drawdown")
    marginal_gain_penalty_weights: tuple[float, ...] = (0.20, 0.10)
    marginal_gain_corr_weight: float = 0.50
    marginal_gain_min_improvement: float = 0.0
    marginal_gain_min_base_score: float | None = None
    marginal_gain_standardize: bool = True

    def marginal_gain_config(self) -> FactorMarginalSelectionConfig:
        return FactorMarginalSelectionConfig(
            score_fields=self.marginal_gain_score_fields,
            score_weights=self.marginal_gain_score_weights,
            penalty_fields=self.marginal_gain_penalty_fields,
            penalty_weights=self.marginal_gain_penalty_weights,
            corr_weight=self.marginal_gain_corr_weight,
            min_improvement=self.marginal_gain_min_improvement,
            min_base_score=self.marginal_gain_min_base_score,
            standardize=self.marginal_gain_standardize,
        )


FACTOR_ROOT = str(DEFAULT_FACTOR_STORE_ROOT)
FACTOR_PROVIDER = "tiger"
FACTOR_VARIANT: str | None = None
FACTOR_GROUP: str | None = "core"
REGION = "us"
SEC_TYPE = "stock"
FREQ = "1d"
FACTOR_NAMES: tuple[str, ...] = DEFAULT_FACTOR_NAMES
CODES: tuple[str, ...] | None = None
START: str | None = "2020-01-01"
END: str | None = "2024-12-31"
LONG_PCT = 0.20
REBALANCE_FREQ = "ME"
ANNUAL_TRADING_DAYS = 252
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 2.0
OUTPUT_DIR = str(DEFAULT_OUTPUT_DIR)
REPORT_NAME = "factor_screener_multifactor_demo"
OPEN_BROWSER = False


CONFIG_SINGLE_FACTOR = SingleFactorScreeningConfig()
CONFIG_CORRELATION = CorrelationScreeningConfig()

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    factor_names = [str(name) for name in FACTOR_NAMES]
    store = FactorStore(root_dir=FACTOR_ROOT)

    candidate_specs = [
        FactorSpec(
            provider=FACTOR_PROVIDER,
            region=REGION,
            sec_type=SEC_TYPE,
            freq=FREQ,
            table_name=factor_name,
            variant=FACTOR_VARIANT,
            group=FACTOR_GROUP,
        )
        for factor_name in factor_names
    ]

    # 1) Build spec inputs for the screener. The screener will resolve factor
    # evaluation artifacts from the store internally.
    screener_spec = FactorScreenerSpec(
        screening_config=CONFIG_SINGLE_FACTOR.metric_filter_config(),
        selection_threshold=0.75,
        selection_score_field="fitness",
        correlation_method="marginal_gain",
        ic_correlation_method="greedy",
    )

    # 2) Run the screener. It loads factor evaluation artifacts internally and
    # returns the selected specs plus the selected return panel.
    screener_result = FactorScreener(
        screener_spec,
        factor_specs=tuple(candidate_specs),
        store=store,
    ).run()
    screener_result.save_detail(output_dir / "screener")

    selected_factor_specs = screener_result.selected_factor_specs
    return_panel = screener_result.return_panel

    # 3) Allocate directly from the return panel.
    factor_weights = allocate_from_return_panel(return_panel)

    # 4) Backtest directly from the return panel.
    backtest_result = run_return_backtest(
        return_panel,
        weights=factor_weights.to_dict(),
        annual_trading_days=ANNUAL_TRADING_DAYS,
    )

    # 5) Generate an analysis report from the backtest returns only.
    report = create_analysis_report(
        returns=backtest_result["portfolio_returns"],
        benchmark_returns=backtest_result["benchmark_returns"],
        output_dir=output_dir / "analysis",
        report_name=REPORT_NAME,
        open_browser=OPEN_BROWSER,
    )

    manifest = {
        "base_config": {
            "factor_root": FACTOR_ROOT,
            "factor_provider": FACTOR_PROVIDER,
            "factor_variant": FACTOR_VARIANT,
            "factor_group": FACTOR_GROUP,
            "region": REGION,
            "sec_type": SEC_TYPE,
            "freq": FREQ,
            "factor_names": list(FACTOR_NAMES),
            "codes": None if CODES is None else list(CODES),
            "long_pct": LONG_PCT,
            "rebalance_freq": REBALANCE_FREQ,
            "annual_trading_days": ANNUAL_TRADING_DAYS,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "output_dir": OUTPUT_DIR,
            "report_name": REPORT_NAME,
            "open_browser": OPEN_BROWSER,
        },
        "single_factor_screening_config": asdict(CONFIG_SINGLE_FACTOR),
        "correlation_screening_config": asdict(CONFIG_CORRELATION),
        "selection_threshold": screener_spec.selection_threshold,
        "selection_score_field": screener_spec.selection_score_field,
        "correlation_method": screener_spec.correlation_method,
        "ic_correlation_method": screener_spec.ic_correlation_method,
        "factor_names": factor_names,
        "candidate_factor_specs": [asdict(spec) for spec in candidate_specs],
        "selected_factor_names": [spec.table_name for spec in selected_factor_specs],
        "selected_factor_specs": [asdict(spec) for spec in selected_factor_specs],
        "output_dir": str(output_dir),
        "return_panel_columns": list(return_panel.columns),
        "factor_weights": {name: float(weight) for name, weight in factor_weights.items()},
        "raw_screener_summary": screener_result.to_summary(),
        "analysis_report_path": None if report.get_report(open_browser=False) is None else str(report.get_report(open_browser=False)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("screened factors:")
    print(f"  candidates: {factor_names}")
    print(f"  selected specs: {[spec.table_name for spec in selected_factor_specs]}")
    print(f"  return panel columns: {list(return_panel.columns)}")
    print("\nallocation weights:")
    print(pd.Series(factor_weights).to_string())
    print("\nbacktest stats:")
    print(pd.DataFrame(backtest_result["stats"]).T.to_string())
    print("\noutputs:")
    print(f"  screener detail: {output_dir / 'screener'}")
    print(f"  analysis report: {report.get_report(open_browser=False)}")
    print(f"  workflow manifest: {output_dir / 'workflow_manifest.json'}")


if __name__ == "__main__":
    main()
