"""Run a full multifactor workflow with 20 Alpha101 factors.

This example intentionally uses the same multifactor entry points as the rest
of the codebase:

- factor generation via ``Alpha101Engine``
- screening and selection via ``screen_factor_panels``
- tear sheets via ``MultifactorEvaluation.full``

It downloads a small cross-sectional universe with yfinance when available and
falls back to the local Tiger/Lean sample data if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.factor_algorithm.alpha101 import Alpha101Engine
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation import FactorPipelineConfig
from tiger_factors.multifactor_evaluation import MultifactorEvaluation
from tiger_factors.multifactor_evaluation import screen_factor_panels
from tiger_factors.multifactor_evaluation.examples.multifactors_factor_research import (
    DEFAULT_UNIVERSE,
    LEAN_DAILY_ROOT,
    _build_local_prices_long,
    _yf_to_long,
    flatten_universe,
    to_wide_factor_panels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full Alpha101 multifactor workflow with 20 factors.",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date for the sample universe.")
    parser.add_argument("--end", default="2024-12-31", help="End date for the sample universe.")
    parser.add_argument("--factor-count", type=int, default=20, help="Number of Alpha101 factors to evaluate.")
    parser.add_argument(
        "--use-local-data",
        action="store_true",
        help="Prefer the local Tiger/Lean sample data instead of yfinance.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for outputs. Defaults to TIGER_TMP_DATA_PATH or the example folder.",
    )
    parser.add_argument(
        "--persist-outputs",
        action="store_true",
        help="Persist intermediate screening outputs and the factor-level report.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated multifactor report in a browser.",
    )
    parser.add_argument("--forward-days", type=int, default=5, help="Forward return horizon for screening.")
    parser.add_argument("--top-n-initial", type=int, default=20, help="Top-N factors kept before correlation filtering.")
    parser.add_argument("--corr-threshold", type=float, default=0.65, help="Maximum correlation when selecting factors.")
    parser.add_argument("--long-pct", type=float, default=0.20, help="Long fraction used in the factor backtest.")
    parser.add_argument("--weight-method", default="softmax", choices=["equal", "positive", "softmax"])
    parser.add_argument("--weight-temp", type=float, default=1.0, help="Temperature used by softmax weighting.")
    parser.add_argument("--transaction-cost-bps", type=float, default=8.0, help="Transaction cost in bps.")
    parser.add_argument("--slippage-bps", type=float, default=4.0, help="Slippage in bps.")
    return parser.parse_args()


def _resolve_output_dir(arg_value: str) -> Path:
    if arg_value:
        return Path(arg_value)
    data_root = Path(os.getenv("TIGER_TMP_DATA_PATH", "/Volumes/Quant_Disk/data/tmp"))
    if not data_root.exists():
        data_root = Path(__file__).resolve().parent / "output"
    return data_root / "alpha101_multifactor_20_factor_demo"


def _prepare_prices_long(
    *,
    start: str,
    end: str,
    use_local_data: bool,
    codes: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if use_local_data:
        sector_map = {code: "Unknown" for code in codes}
        prices_long = _build_local_prices_long(
            codes,
            start=start,
            end=end,
            db_path=str(Path("/Volumes/Quant_Disk/data/yahoo_us_stock.db")),
            price_provider="yahoo",
            classification_provider="simfin",
            classification_dataset="companies",
        )
        if "sector" not in prices_long.columns:
            prices_long["sector"] = prices_long["code"].map(sector_map).fillna("Unknown")
        return prices_long, sector_map

    prices_long = _yf_to_long(codes, start=start, end=end)
    sector_map = {code: sector for sector, sector_codes in DEFAULT_UNIVERSE.items() for code in sector_codes}
    prices_long["sector"] = prices_long["code"].map(sector_map).fillna("Unknown")
    return prices_long, sector_map


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_count = max(int(args.factor_count), 1)
    alpha_ids = list(range(1, factor_count + 1))
    codes, sector_map = flatten_universe(DEFAULT_UNIVERSE)

    print(f"[1/6] Loading universe: {len(codes)} tickers, {args.start} -> {args.end}")
    prices_long, sector_map = _prepare_prices_long(
        start=args.start,
        end=args.end,
        use_local_data=bool(args.use_local_data),
        codes=codes,
    )
    if prices_long.empty:
        raise RuntimeError("No price data available for the example universe.")

    prices_long = prices_long.sort_values(["date_", "code"]).reset_index(drop=True)
    if "volume" not in prices_long.columns:
        prices_long["volume"] = np.nan
    if "vwap" not in prices_long.columns:
        prices_long["vwap"] = prices_long[["open", "high", "low", "close"]].mean(axis=1)

    print(f"[2/6] Computing Alpha101 factors: alpha_001 ... alpha_{factor_count:03d}")
    engine = Alpha101Engine(prices_long)
    all_factors_long = engine.compute_all(alpha_ids=alpha_ids)
    all_factors_long = all_factors_long[
        (all_factors_long["date_"] >= pd.Timestamp(args.start))
        & (all_factors_long["date_"] <= pd.Timestamp(args.end))
    ].reset_index(drop=True)

    factor_panels = to_wide_factor_panels(all_factors_long)
    if not factor_panels:
        raise RuntimeError("No valid factor panels were generated.")

    close_panel = (
        prices_long.pivot_table(index="date_", columns="code", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    if close_panel.empty:
        raise RuntimeError("No close panel could be built from the input data.")

    pipeline_report_dir = output_dir / "factor_report" if args.persist_outputs else None
    pipeline_output_dir = output_dir / "pipeline" if args.persist_outputs else None

    print("[3/6] Screening, de-correlating, and blending the selected factors")
    pipeline_result = screen_factor_panels(
        factor_panels,
        close_panel,
        config=FactorPipelineConfig(
            forward_days=int(args.forward_days),
            top_n_initial=int(args.top_n_initial),
            corr_threshold=float(args.corr_threshold),
            score_field="fitness",
            selection_score_field="ic_ir",
            weight_method=args.weight_method,
            weight_temperature=float(args.weight_temp),
            long_pct=float(args.long_pct),
            long_short=True,
            rebalance_freq="ME",
            annual_trading_days=252,
            transaction_cost_bps=float(args.transaction_cost_bps),
            slippage_bps=float(args.slippage_bps),
            persist_outputs=bool(args.persist_outputs),
        ),
        output_dir=pipeline_output_dir,
        report_dir=pipeline_report_dir,
        report_factor_name="alpha101_combined_factor",
    )

    backtest = pipeline_result.backtest_returns
    positions_frame = backtest.attrs.get("positions") if hasattr(backtest, "attrs") else None
    factor_data = {name: factor_panels[name] for name in pipeline_result.selected_factors if name in factor_panels}
    factor_data["combined_factor"] = pipeline_result.combined_factor

    print("[4/6] Building multifactor tear sheets")
    evaluation = MultifactorEvaluation(
        backtest=backtest,
        positions_frame=positions_frame,
        close_panel_frame=close_panel,
        factor_data=factor_data,
        sector_mappings=sector_map,
        output_dir=output_dir,
        report_name="alpha101_20_factor_demo",
        capital_base=1_000_000.0,
    )
    bundle = evaluation.full(
        backtest,
        output_dir=output_dir,
        report_name="alpha101_20_factor_demo",
        open_browser=bool(args.open_browser),
    )

    summary_payload = {
        "start": args.start,
        "end": args.end,
        "factor_count": factor_count,
        "selected_factors": pipeline_result.selected_factors,
        "factor_weights": pipeline_result.factor_weights,
        "backtest_stats": pipeline_result.backtest_stats,
        "output_dir": str(output_dir),
        "report_path": None if bundle.report_path is None else str(bundle.report_path),
        "summary_dir": str(output_dir / "summary"),
        "positions_dir": str(output_dir / "positions"),
        "trades_dir": str(output_dir / "trades"),
        "portfolio_dir": str(output_dir / "portfolio"),
    }

    if args.persist_outputs:
        (output_dir / "alpha101_multifactor_summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    print("[5/6] Multifactor workflow complete.")
    print(f"Output dir: {output_dir}")
    if bundle.report_path is not None:
        print(f"Combined report: {bundle.report_path}")
    print(f"Selected factors: {pipeline_result.selected_factors}")
    print(f"Factor weights: {json.dumps(pipeline_result.factor_weights, indent=2, default=str)}")
    print("\nPortfolio stats:")
    print(json.dumps(pipeline_result.backtest_stats["portfolio"], indent=2, default=str))
    print("\nSummary payload:")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
