"""Print the latest stock holdings implied by a selected Alpha101 factor basket.

This is the compact "what stocks and what proportions did my factors map to?"
demo. It loads stored factor panels from the local factor store, blends them
with the requested factor weights, and prints the latest holdings table.
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
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from tiger_factors.factor_portfolio import summarize_factor_portfolio_holdings
from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import TigerFactorLibrary


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "alpha101_holdings_snapshot_demo"
DEFAULT_FACTOR_NAMES = [
    "alpha_001",
    "alpha_002",
    "alpha_003",
    "alpha_004",
    "alpha_005",
    "alpha_006",
    "alpha_007",
    "alpha_008",
    "alpha_009",
    "alpha_010",
    "alpha_011",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the latest holdings implied by an Alpha101 factor basket.")
    parser.add_argument("--store-root", default=str(DEFAULT_FACTOR_STORE_ROOT))
    parser.add_argument("--factor-provider", default="tiger")
    parser.add_argument("--region", default="us")
    parser.add_argument("--sec-type", default="stock")
    parser.add_argument("--freq", default="1d")
    parser.add_argument("--group", default="alpha_101")
    parser.add_argument("--factor-variant", default=None)
    parser.add_argument("--factor-names", nargs="+", default=list(DEFAULT_FACTOR_NAMES))
    parser.add_argument(
        "--weights-json",
        default=None,
        help="Optional JSON string or path with custom factor weights, e.g. '{\"alpha_001\": 0.2, \"alpha_002\": 0.8}'.",
    )
    parser.add_argument("--long-only", action="store_true", help="Build a long-only portfolio.")
    parser.add_argument("--gross-exposure", type=float, default=1.0)
    parser.add_argument("--no-standardize", action="store_true", help="Skip cross-sectional standardization.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-csv", action="store_true", help="Write the latest holdings and full positions to CSV.")
    return parser.parse_args()


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def _load_weights(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    path = Path(token)
    payload = json.loads(path.read_text()) if path.exists() else json.loads(token)
    if not isinstance(payload, dict):
        raise ValueError("weights-json must decode to a JSON object.")
    return {str(key): float(raw_value) for key, raw_value in payload.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variant = _normalize_variant(args.factor_variant)
    factor_names = [str(name) for name in args.factor_names]
    custom_weights = _load_weights(args.weights_json)

    library = TigerFactorLibrary(output_dir=args.store_root, price_provider="yahoo", verbose=True)
    factor_panels = library.load_factor_panels(
        factor_names=factor_names,
        provider=args.factor_provider,
        region=args.region,
        sec_type=args.sec_type,
        freq=args.freq,
        variant=variant,
        group=args.group,
    )
    if not factor_panels:
        raise ValueError("No factor panels could be loaded from the factor store.")

    summary = summarize_factor_portfolio_holdings(
        factor_panels,
        factor_weights=custom_weights,
        long_only=args.long_only,
        gross_exposure=args.gross_exposure,
        standardize=not args.no_standardize,
        top_n=args.top_n,
    )

    print("factor weights:")
    print(pd.Series(summary["factor_weights"]).sort_values(ascending=False).to_string())
    print("\nlatest date:")
    print(summary["latest_date"])
    print("\nlatest holdings:")
    print(summary["latest_holdings"].to_string(index=False))

    if args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary["positions"].to_csv(output_dir / "alpha101_positions.csv", index=False)
        summary["latest_holdings"].to_csv(output_dir / "alpha101_latest_holdings.csv", index=False)
        pd.Series(summary["factor_weights"], name="weight").to_csv(output_dir / "alpha101_factor_weights.csv")
        print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
