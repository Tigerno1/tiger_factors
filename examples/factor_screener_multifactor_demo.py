"""Run the total screener workflow, then feed the selected factors into multifactor evaluation.

This demo shows the intended boundary between the layers:

1. ``Screener`` dispatches the factor screener and the correlation screener
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
from tiger_factors.factor_screener import CorrelationScreenerSpec
from tiger_factors.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener import ScreeningEffectivenessSpec
from tiger_factors.factor_screener import Screener
from tiger_factors.factor_allocation import RiskfolioConfig
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
RETURN_MODE = "long_short"
OUTPUT_DIR = str(DEFAULT_OUTPUT_DIR)
REPORT_NAME = "factor_screener_multifactor_demo"
OPEN_BROWSER = False


CONFIG_SINGLE_FACTOR = FactorMetricFilterConfig(
    min_fitness=0.10,
    min_ic_mean=0.01,
    min_rank_ic_mean=0.01,
    min_sharpe=0.40,
    max_turnover=0.50,
    min_decay_score=0.20,
    min_capacity_score=0.20,
    max_correlation_penalty=0.60,
    min_regime_robustness=0.60,
    min_out_of_sample_stability=0.60,
    sort_field="fitness",
    tie_breaker_field="ic_ir",
)
CONFIG_RISKFOLIO = RiskfolioConfig(
    model="Classic",
    rm="MV",
    obj="Sharpe",
    rf=0.0,
    l=2.0,
    hist=True,
    method_mu="hist",
    method_cov="hist",
    max_kelly=False,
    weight_bounds=(0.0, 1.0),
)
CONFIG_CORRELATION_STEPS = (
    CorrelationScreenerSpec(
        evaluation_source="factor",
        method="greedy",
        threshold=0.75,
        score_field="fitness",
    ),
    CorrelationScreenerSpec(
        evaluation_source="ic",
        method="greedy",
        threshold=0.60,
        score_field="fitness",
    ),
    CorrelationScreenerSpec(
        evaluation_source="return",
        method="greedy",
        threshold=0.00,
        score_field="fitness",
    ),
)

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

    # 1) Build spec inputs for the workflow. The workflow will resolve factor
    # evaluation artifacts from the store internally.
    screener_spec = FactorScreenerSpec(
        screening_config=CONFIG_SINGLE_FACTOR,
    )

    correlation_specs = tuple(
        CONFIG_CORRELATION_STEPS
    )

    # 2) Run the total screener. It first applies the factor screener, then
    # dispatches the correlation screener on the surviving factor specs.
    workflow_result = Screener(
        screener_spec,
        correlation_specs,
        factor_specs=tuple(candidate_specs),
        store=store,
        return_mode=RETURN_MODE,
    ).run()
    screening_effectiveness = workflow_result.validate_effectiveness(
        spec=ScreeningEffectivenessSpec(min_retained_ratio=0.8),
    )

    return_panel = workflow_result.return_panel

    # 3) Allocate directly from the return panel.
    factor_weight_series = allocate_from_return_panel(
        return_panel,
        config=CONFIG_RISKFOLIO,
    )
    weights: dict[str, float] = {
        str(name): float(weight)
        for name, weight in factor_weight_series.items()
    }

    # 4) Backtest directly from the return panel.
    backtest_result = run_return_backtest(
        return_panel,
        weights=weights,
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
        "riskfolio_config": asdict(CONFIG_RISKFOLIO),
        "correlation_screening_configs": [asdict(config) for config in CONFIG_CORRELATION_STEPS],
        "factor_names": factor_names,
        "candidate_factor_specs": [asdict(spec) for spec in candidate_specs],
        "selected_factor_names": [spec.table_name for spec in workflow_result.selected_factor_specs],
        "selected_factor_specs": [asdict(spec) for spec in workflow_result.selected_factor_specs],
        "factor_selected_factor_names": [spec.table_name for spec in workflow_result.factor_selected_factor_specs],
        "factor_selected_factor_specs": [asdict(spec) for spec in workflow_result.factor_selected_factor_specs],
        "output_dir": str(output_dir),
        "return_panel_columns": list(return_panel.columns),
        "factor_weights": {name: float(weight) for name, weight in weights.items()},
        "workflow_summary": workflow_result.to_summary(),
        "screening_effectiveness_summary": screening_effectiveness.to_summary(),
        "screening_effectiveness_passed": bool(screening_effectiveness.passed),
        "screening_effectiveness_failed_rules": list(screening_effectiveness.failed_rules),
        "factor_screener_summary": workflow_result.factor_result.to_summary(),
        "correlation_screener_summary": workflow_result.correlation_result.to_summary(),
        "correlation_screener_chain": [result.to_summary() for result in workflow_result.correlation_results],
        "analysis_report_path": None if report.get_report(open_browser=False) is None else str(report.get_report(open_browser=False)),
    }
    screening_effectiveness.save(output_dir / "screener" / "screening_effectiveness")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("screened factors:")
    print(f"  candidates: {factor_names}")
    print(f"  factor-selected specs: {[spec.table_name for spec in workflow_result.factor_selected_factor_specs]}")
    print(f"  selected specs: {[spec.table_name for spec in workflow_result.selected_factor_specs]}")
    print(f"  correlation chain: {[result.spec.evaluation_source + ':' + result.spec.method for result in workflow_result.correlation_results]}")
    print(f"  screening effectiveness passed: {screening_effectiveness.passed}")
    print(f"  return panel columns: {list(return_panel.columns)}")
    print("\nallocation weights:")
    print(factor_weight_series.to_string())
    print("\nbacktest stats:")
    print(pd.DataFrame(backtest_result["stats"]).T.to_string())
    print("\noutputs:")
    print(f"  screener detail: {output_dir / 'screener'}")
    print(f"  screening effectiveness: {output_dir / 'screener' / 'screening_effectiveness'}")
    print(f"  analysis report: {report.get_report(open_browser=False)}")
    print(f"  workflow manifest: {output_dir / 'workflow_manifest.json'}")


if __name__ == "__main__":
    main()
