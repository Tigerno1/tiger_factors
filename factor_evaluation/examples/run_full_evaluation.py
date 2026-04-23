from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_store import FactorStore

DEFAULT_FACTOR_PATH = "/Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_001.parquet"
DEFAULT_PRICE_PATH = "/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet"
DEFAULT_OUTPUT_DIR = "/Volumes/Quant_Disk/evaluation/alpha_001_formal"
DEFAULT_HORIZONS = [1, 3, 5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full Tiger evaluation bundle.")
    parser.add_argument("--factor-path", default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--price-path", default=DEFAULT_PRICE_PATH)
    parser.add_argument("--factor-column", default="alpha_001")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--horizon-quantiles", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor_df = pd.read_parquet(args.factor_path)
    price_df = pd.read_parquet(args.price_path)

    engine = FactorEvaluationEngine(
        factor_frame=factor_df,
        price_frame=price_df,
        factor_column=args.factor_column,
        factor_store=FactorStore(root_dir=args.output_dir),
    )

    summary = engine.evaluate()
    report = engine.full(
        horizons=tuple(args.horizons),
        horizon_quantiles=args.horizon_quantiles,
    )
    bundle_summary = engine.create_report_bundle_summary(report)

    # horizon_result = report.payload["horizon_result"] if report.payload else None
    # horizon_summary = report.payload["horizon_summary"] if report.payload else None
    # full_manifest = report.output_dir / "manifest.json"
    # horizon_manifest = Path(report.table_paths["horizon_manifest"])

    # print("summary:")
    # print(summary)
    # print("\nreport dir:")
    # print(report.output_dir)
    # print("\nfigures:")
    # for name, path in report.figure_paths.items():
    #     print(f"{name}: {path}")
    # print("\ntables:")
    # for name, path in report.table_paths.items():
    #     print(f"{name}: {path}")
    # print("\nhorizon result:")
    # print(horizon_result)
    # print("\nhorizon summary:")
    # print(horizon_summary)
    # print("\nreport bundle summary:")
    # print(bundle_summary.to_dict())
    # print("\nmanifest:")
    # print(f"full: {full_manifest}")
    # print(f"horizon: {horizon_manifest}")


if __name__ == "__main__":
    main()
