"""Example entry point for multifactors screening.

This script shows two common workflows:
1. Build a registry from a root folder that contains per-strategy summary/evaluation.parquet files.
2. Screen an in-memory registry DataFrame with the same filter rules.

Each strategy directory is expected to look like:

    strategies_root/
      strategy_a/
        summary/
          evaluation.parquet
      strategy_b/
        summary/
          evaluation.parquet

The script only screens the factors and prints the surviving rows.
It does not require any plotting or report generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import build_factor_registry_from_root
from tiger_factors.factor_screener import screen_factor_registry


DEFAULT_REGISTRY_ROOT = "/Volumes/Quant_Disk/evaluation/summary"
DEFAULT_OUTPUT_PATH = "/Volumes/Quant_Disk/evaluation/multifactors/multifactors_screened.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multifactors screening example.")
    parser.add_argument(
        "--registry-root",
        default=DEFAULT_REGISTRY_ROOT,
        help="Root directory containing strategy folders with summary/evaluation.parquet.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Optional parquet output path for the screened registry.",
    )
    parser.add_argument("--min-ic-mean", type=float, default=0.01)
    parser.add_argument("--min-rank-ic-mean", type=float, default=0.01)
    parser.add_argument("--min-sharpe", type=float, default=0.40)
    parser.add_argument("--max-turnover", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write the screened output to disk; only print it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = build_factor_registry_from_root(args.registry_root)
    if registry.empty:
        print(f"No strategy summaries found under: {args.registry_root}")
        return

    config = FactorMetricFilterConfig(
        min_ic_mean=args.min_ic_mean,
        min_rank_ic_mean=args.min_rank_ic_mean,
        min_sharpe=args.min_sharpe,
        max_turnover=args.max_turnover,
    )
    screened = screen_factor_registry(registry, config=config)

    columns = [
        "strategy_name",
        "factor_name",
        "fitness",
        "ic_mean",
        "rank_ic_mean",
        "ic_ir",
        "sharpe",
        "turnover",
        "usable",
        "failed_rules",
    ]
    existing_columns = [column for column in columns if column in screened.columns]
    print(screened.loc[:, existing_columns].head(args.top_n).to_string(index=False))

    if not args.no_write:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        screened.to_parquet(output_path, index=False)
        print(f"\nSaved screened registry to: {output_path}")


if __name__ == "__main__":
    main()
