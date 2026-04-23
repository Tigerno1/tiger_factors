"""Run a generic factor-store basket through a composite backtest.

This demo loads an arbitrary set of stored factors, inspects them as a merged
Tiger-style research frame, loads the matching close panel, and runs the
generic multifactor composite backtest with equal weights by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.examples.factor_store_multi_factor_reporting import save_factor_backtest_plot
from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "factor_store_multi_factor_backtest_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generic factor-store basket through a composite backtest.")
    parser.add_argument(
        "--store-root",
        default=str(DEFAULT_FACTOR_STORE_ROOT),
        help="Root directory of the Tiger factor store.",
    )
    parser.add_argument("--factor-provider", default="tiger", help="Provider namespace used when the factors were saved.")
    parser.add_argument("--factor-variant", default=None, help="Optional factor variant used when the factors were saved.")
    parser.add_argument(
        "--factor-names",
        nargs="+",
        default=["BM", "FSCORE", "BMFSCORE"],
        help="Stored factor names to load and composite.",
    )
    parser.add_argument("--price-provider", default="yahoo", help="Price provider used for the matching close panel.")
    parser.add_argument("--codes", nargs="*", default=None, help="Optional explicit universe to keep.")
    parser.add_argument("--start", default=None, help="Optional backtest start date.")
    parser.add_argument("--end", default=None, help="Optional backtest end date.")
    parser.add_argument("--rebalance-freq", default="ME")
    parser.add_argument("--long-pct", type=float, default=0.2)
    parser.add_argument("--long-short", action="store_true", default=True)
    parser.add_argument("--long-only", action="store_true", default=False)
    parser.add_argument("--annual-trading-days", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="factor_store_multi_factor_backtest")
    parser.add_argument(
        "--weights-json",
        default=None,
        help="Optional JSON string or file path with custom factor weights, e.g. '{\"BM\": 0.6, \"FSCORE\": 0.4}'.",
    )
    parser.add_argument("--save-csv", action="store_true", help="Write the merged research frame to CSV.")
    return parser.parse_args()


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def _coverage_summary(frame: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in factor_names:
        if name not in frame.columns:
            rows.append({"factor": name, "non_null": 0, "coverage": 0.0})
            continue
        series = pd.to_numeric(frame[name], errors="coerce")
        rows.append(
            {
                "factor": name,
                "non_null": int(series.notna().sum()),
                "coverage": float(series.notna().mean()) if len(series) else 0.0,
                "mean": float(series.mean(skipna=True)) if series.notna().any() else float("nan"),
                "std": float(series.std(skipna=True)) if series.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _load_weights(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    payload: object
    path = Path(token)
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = json.loads(token)
    if not isinstance(payload, dict):
        raise ValueError("weights-json must decode to a JSON object mapping factor names to weights.")
    weights: dict[str, float] = {}
    for key, raw_value in payload.items():
        weights[str(key)] = float(raw_value)
    return weights


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variant = _normalize_variant(args.factor_variant)
    factor_names = [str(name) for name in args.factor_names]
    custom_weights = _load_weights(args.weights_json)

    library = TigerFactorLibrary(output_dir=args.store_root, price_provider=args.price_provider, verbose=True)
    factor_frame = library.load_factor_frame(
        factor_names=factor_names,
        provider=args.factor_provider,
        variant=variant,
        codes=args.codes,
        start=args.start,
        end=args.end,
    )

    print("merged research frame:")
    print(f"  shape={factor_frame.shape}")
    print(f"  columns={list(factor_frame.columns)}")
    print("\ncoverage summary:")
    print(_coverage_summary(factor_frame, factor_names).to_string(index=False))

    if args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{args.report_name}_factor_frame.csv"
        factor_frame.to_csv(csv_path, index=False)
        print(f"\nmerged research frame saved to: {csv_path}")

    if not factor_frame.empty:
        price_panel = library.price_panel(
            codes=sorted(factor_frame["code"].astype(str).unique().tolist()),
            start=str(pd.to_datetime(factor_frame["date_"]).min().date()),
            end=str(pd.to_datetime(factor_frame["date_"]).max().date()),
            provider=args.price_provider,
            field="close",
        )
    else:
        price_panel = pd.DataFrame()

    if price_panel.empty:
        raise ValueError("Could not load a matching close panel for the selected factor universe.")

    factor_panels = library.load_factor_panels(
        factor_names=factor_names,
        provider=args.factor_provider,
        variant=variant,
        codes=args.codes,
        start=args.start,
        end=args.end,
    )

    if custom_weights is None:
        weights = {name: 1.0 for name in factor_panels}
    else:
        weights = {name: float(custom_weights.get(name, 0.0)) for name in factor_panels}
    result = multi_factor_backtest(
        factor_panels,
        price_panel,
        weights=weights,
        standardize=True,
        rebalance_freq=args.rebalance_freq,
        long_pct=args.long_pct,
        long_short=not args.long_only,
        annual_trading_days=args.annual_trading_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    report = run_portfolio_from_backtest(
        result["backtest"],
        output_dir=output_dir,
        report_name=args.report_name,
    )
    figure_path = save_factor_backtest_plot(
        result["backtest"],
        output_dir=output_dir,
        report_name=args.report_name,
    )

    print("\ncomposite weights:")
    print(pd.Series(result["weights"]).to_string())
    if custom_weights is not None:
        print("\ncustom weights input:")
        print(pd.Series(custom_weights).to_string())
    print("\nbacktest stats:")
    print(pd.DataFrame(result["stats"]).T.to_string())
    if figure_path is not None:
        print(f"\ncomposite equity curve: {figure_path}")
    print("\nportfolio report:")
    if report is not None:
        print(report.to_summary())


if __name__ == "__main__":
    main()
