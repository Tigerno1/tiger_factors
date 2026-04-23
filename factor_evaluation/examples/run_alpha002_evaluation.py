from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_store import FactorStore

DEFAULT_FACTOR_PATH = "/Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_015.parquet"
DEFAULT_PRICE_PATH = "/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet"
DEFAULT_OUTPUT_DIR = "/Volumes/Quant_Disk/evaluation/alpha_015_formal"
DEFAULT_HORIZONS = [1, 3, 5, 10, 20]


def _load_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"Unsupported file format: {resolved.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiger evaluation for alpha_015.")
    parser.add_argument("--factor-path", default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--price-path", default=DEFAULT_PRICE_PATH)
    parser.add_argument("--factor-column", default="alpha_015")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-path", default=None, help="Optional group labels file to enable group-aware evaluation.")
    parser.add_argument(
        "--group-labels-cache",
        dest="group_labels_cache",
        action="store_true",
        help="Enable group label caching and cache prewarm when group-path is provided.",
    )
    parser.add_argument(
        "--no-group-labels-cache",
        dest="group_labels_cache",
        action="store_false",
        help="Disable group label caching and cache prewarm.",
    )
    parser.set_defaults(group_labels_cache=True)
    parser.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--horizon-quantiles", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor_df = _load_table(args.factor_path)
    price_df = _load_table(args.price_path)

    evaluation = FactorEvaluationEngine(
        factor_frame=factor_df,
        price_frame=price_df,
        factor_column=args.factor_column,
        group_labels=args.group_path,
        group_labels_cache=args.group_labels_cache,
        factor_store=FactorStore(root_dir=args.output_dir),
    )

    summary = evaluation.evaluate()
    report = evaluation.full(
        horizons=tuple(args.horizons),
        horizon_quantiles=args.horizon_quantiles,
    )
    bundle_summary = evaluation.create_report_bundle_summary(report)

    horizon_result = report.payload["horizon_result"] if report.payload else None
    horizon_summary = report.payload["horizon_summary"] if report.payload else None

    print("summary:")
    print(summary)
    print("\nreport dir:")
    print(report.output_dir)
    print("\nfigures:")
    for name, path in report.figure_paths.items():
        print(f"{name}: {path}")
    print("\ntables:")
    for name, path in report.table_paths.items():
        print(f"{name}: {path}")
    print("\nhorizon result:")
    print(horizon_result)
    print("\nhorizon summary:")
    print(horizon_summary)
    print("\nreport bundle summary:")
    print(bundle_summary.to_dict())


if __name__ == "__main__":
    main()
