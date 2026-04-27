"""Alpha101 multifactor demo: select, screen, weight, backtest, local reports.

This demo keeps the candidate set fixed to the balanced Alpha101 shortlist
you picked:

    alpha_021, alpha_030, alpha_047, alpha_005,
    alpha_076, alpha_068, alpha_066, alpha_092

The flow is:

1. load the eight factor panels from the local Tiger factor store
2. load the close panel from the current Tiger price store
3. run multifactor screening on this fixed candidate pool
   - score factors
   - filter by the requested score field
   - remove highly correlated factors
   - assign weights
   - blend the selected factors
   - backtest the blended factor
4. export the outputs
5. optionally generate local positions, trade, and combined portfolio reports

The demo is intentionally small and explicit so it can be used as a starting
point for your own Alpha101 research flow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd

from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_portfolio import summarize_factor_portfolio_holdings
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.pipeline import FactorPipelineConfig
from tiger_factors.multifactor_evaluation.pipeline import screen_factor_panels
from tiger_factors.multifactor_evaluation.common.parquet_utils import to_parquet_clean
from tiger_factors.multifactor_evaluation.common.matplotlib_config import configure_matplotlib
from tiger_factors.multifactor_evaluation.reporting.portfolio import create_position_report
from tiger_factors.multifactor_evaluation.reporting.trades import create_trade_report
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest

configure_matplotlib()


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "alpha101_multifactor_portfolio_demo"
DEFAULT_PROVIDER = "tiger"
DEFAULT_REGION = "us"
DEFAULT_SEC_TYPE = "stock"
DEFAULT_FREQ = "1d"
DEFAULT_VARIANT = None
DEFAULT_FORWARD_DAYS = 21
DEFAULT_CORR_THRESHOLD = 0.75
DEFAULT_WEIGHT_METHOD = "positive"
DEFAULT_WEIGHT_TEMPERATURE = 1.0
DEFAULT_LONG_PCT = 0.20
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_ANNUAL_TRADING_DAYS = 252
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Alpha101 multifactor demo and optional local portfolio reports.")
    parser.add_argument("--factor-root", default=str(DEFAULT_FACTOR_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--sec-type", default=DEFAULT_SEC_TYPE)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--factor-names", nargs="*", default=list(DEFAULT_FACTOR_NAMES))
    parser.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument("--weight-method", choices=("equal", "positive", "softmax"), default=DEFAULT_WEIGHT_METHOD)
    parser.add_argument("--weight-temperature", type=float, default=DEFAULT_WEIGHT_TEMPERATURE)
    parser.add_argument("--long-pct", type=float, default=DEFAULT_LONG_PCT)
    parser.add_argument("--rebalance-freq", default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--annual-trading-days", type=int, default=DEFAULT_ANNUAL_TRADING_DAYS)
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--skip-report", action="store_true", help="Skip generating local portfolio reports.")
    parser.add_argument("--no-persist", action="store_true", help="Do not write any output files.")
    parser.add_argument(
        "--report-factor-name",
        default="alpha101_combo",
        help="Name used in the combined portfolio report output.",
    )
    return parser.parse_args()


def _load_factor_panel(
    store: FactorStore,
    *,
    factor_name: str,
    provider: str,
    region: str,
    sec_type: str,
    freq: str,
    variant: str | None,
) -> pd.DataFrame:
    spec = FactorSpec(
        provider=provider,
        region=region,
        sec_type=sec_type,
        freq=freq,
        table_name=factor_name,
        variant=variant,
    )
    frame = store.get_factor(spec)
    factor_series = coerce_factor_series(frame)
    panel = factor_series.unstack("code").sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = "date_"
    return panel


def _load_close_panel(
    store: FactorStore,
    *,
    provider: str,
    region: str,
    sec_type: str,
    freq: str,
) -> pd.DataFrame:
    price_spec = AdjPriceSpec(provider=provider, region=region, sec_type=sec_type, freq=freq)
    price_frame = store.get_adj_price(price_spec)
    close_panel = coerce_price_panel(price_frame)
    if close_panel.empty:
        raise RuntimeError(f"Empty close panel loaded from store using spec: {price_spec}")
    return close_panel


def _write_outputs(
    output_dir: Path,
    *,
    result,
    holdings_summary: dict[str, object] | None,
    selected_factor_names: list[str],
    factor_names: list[str],
    price_location: str,
    args: argparse.Namespace,
    ) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_returns = result.backtest_returns.copy()
    backtest_returns.attrs = {}
    result.summary.to_csv(output_dir / "alpha101_factor_summary.csv")
    result.score_table.to_csv(output_dir / "alpha101_factor_score_table.csv", index=False)
    to_parquet_clean(result.score_table, output_dir / "alpha101_factor_score_table.parquet", index=False)
    result.correlation_matrix.to_csv(output_dir / "alpha101_factor_correlation_matrix.csv")
    pd.Series(selected_factor_names, name="factor").to_csv(output_dir / "alpha101_selected_factors.csv", index=False)
    pd.Series(result.factor_weights, name="weight").to_csv(output_dir / "alpha101_factor_weights.csv")
    if holdings_summary is not None:
        positions = holdings_summary.get("positions")
        latest_holdings = holdings_summary.get("latest_holdings")
        if isinstance(positions, pd.DataFrame):
            positions.to_csv(output_dir / "alpha101_positions.csv", index=False)
        if isinstance(latest_holdings, pd.DataFrame):
            latest_holdings.to_csv(output_dir / "alpha101_latest_holdings.csv", index=False)
    to_parquet_clean(result.combined_factor, output_dir / "alpha101_combined_factor.parquet")
    to_parquet_clean(backtest_returns, output_dir / "alpha101_backtest_returns.parquet")
    to_parquet_clean(pd.DataFrame(result.backtest_stats).T, output_dir / "alpha101_backtest_stats.parquet")
    pd.DataFrame(
        {
            "factor_name": factor_names,
            "selected": [name in selected_factor_names for name in factor_names],
        }
    ).to_csv(output_dir / "alpha101_candidate_set.csv", index=False)
    (output_dir / "alpha101_demo_manifest.json").write_text(
        json.dumps(
            {
                "factor_root": str(args.factor_root),
                "price_location": price_location,
                "provider": args.provider,
                "region": args.region,
                "sec_type": args.sec_type,
                "freq": args.freq,
                "variant": args.variant,
                "factor_names": factor_names,
                "selected_factors": selected_factor_names,
                "corr_threshold": args.corr_threshold,
                "weight_method": args.weight_method,
                "weight_temperature": args.weight_temperature,
                "long_pct": args.long_pct,
                "rebalance_freq": args.rebalance_freq,
                "annual_trading_days": args.annual_trading_days,
                "transaction_cost_bps": args.transaction_cost_bps,
                "slippage_bps": args.slippage_bps,
                "report_factor_name": args.report_factor_name,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def _run_reports(result, *, output_dir: Path, report_factor_name: str) -> None:
    positions = result.backtest_returns.attrs.get("positions")
    close_panel = result.backtest_returns.attrs.get("close_panel")
    create_position_report(
        positions,
        output_dir=output_dir / "positions",
        report_name=report_factor_name,
    )
    create_trade_report(
        result.backtest_returns["portfolio"],
        positions=positions,
        close_panel=close_panel,
        output_dir=output_dir / "trades",
        report_name=report_factor_name,
    )
    run_portfolio_from_backtest(
        result.backtest_returns,
        output_dir=output_dir / "portfolio",
        report_name=report_factor_name,
    )


def main() -> None:
    args = parse_args()
    factor_root = Path(args.factor_root)
    output_dir = Path(args.output_dir)
    factor_names = [str(name).strip() for name in args.factor_names if str(name).strip()]
    if not factor_names:
        raise RuntimeError("No factor names were provided.")

    store = FactorStore(root_dir=factor_root)
    close_panel = _load_close_panel(
        store,
        provider=args.provider,
        region=args.region,
        sec_type=args.sec_type,
        freq=args.freq,
    )

    factor_panels: dict[str, pd.DataFrame] = {}
    for factor_name in factor_names:
        factor_panels[factor_name] = _load_factor_panel(
            store,
            factor_name=factor_name,
            provider=args.provider,
            region=args.region,
            sec_type=args.sec_type,
            freq=args.freq,
            variant=args.variant,
        )

    config = FactorPipelineConfig(
        forward_days=args.forward_days,
        top_n_initial=len(factor_names),
        corr_threshold=args.corr_threshold,
        score_field="fitness",
        selection_score_field="ic_ir",
        weight_method=args.weight_method,
        weight_temperature=args.weight_temperature,
        min_factor_weight=0.0,
        max_factor_weight=0.30,
        long_pct=args.long_pct,
        long_short=True,
        rebalance_freq=args.rebalance_freq,
        annual_trading_days=args.annual_trading_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        persist_outputs=not args.no_persist,
    )

    result = screen_factor_panels(
        factor_panels=factor_panels,
        close_panel=close_panel,
        config=config,
        output_dir=output_dir if not args.no_persist else None,
        report_dir=output_dir / "report" if not args.no_persist else None,
        report_factor_name=args.report_factor_name,
    )
    selected_panels = {name: factor_panels[name] for name in result.selected_factors if name in factor_panels}
    holdings_summary = summarize_factor_portfolio_holdings(
        selected_panels,
        factor_weights=result.factor_weights,
        long_only=False,
        gross_exposure=1.0,
        standardize=True,
        top_n=20,
    )

    print("Alpha101 multifactor demo complete.")
    print("Candidate factors:")
    print(factor_names)
    print("Selected factors:")
    print(result.selected_factors)
    print("Factor weights:")
    print(result.factor_weights)
    print("Latest holdings:")
    print(holdings_summary["latest_holdings"].to_string(index=False))
    print("Portfolio stats:")
    print(json.dumps(result.backtest_stats["portfolio"], indent=2, ensure_ascii=False, default=str))
    print("Benchmark stats:")
    print(json.dumps(result.backtest_stats["benchmark"], indent=2, ensure_ascii=False, default=str))

    if not args.no_persist:
        _write_outputs(
            output_dir,
            result=result,
            holdings_summary=holdings_summary,
            selected_factor_names=result.selected_factors,
            factor_names=factor_names,
            price_location=f"price/{args.provider}/{args.region}/{args.sec_type}/{args.freq}/adj_price",
            args=args,
        )
        print(f"Saved outputs to: {output_dir}")

    if not args.skip_report:
        _run_reports(result, output_dir=output_dir, report_factor_name=args.report_factor_name)


if __name__ == "__main__":
    main()
