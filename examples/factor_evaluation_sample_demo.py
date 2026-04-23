from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
# os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
# MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.factor_evaluation.evaluation import SingleFactorEvaluation
from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


DEFAULT_FACTOR_NAME = "alpha_101"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data factor evaluation and inspect summary/horizon/full outputs.")
    parser.add_argument(
        "--store-root",
        default=str(DEFAULT_FACTOR_STORE_ROOT),
        help="Root directory used by FactorStore and evaluation outputs.",
    )
    parser.add_argument("--factor-name", default=DEFAULT_FACTOR_NAME, help="Stored Alpha101 factor name to evaluate.")
    parser.add_argument("--provider", default="tiger", help="Factor provider namespace.")
    parser.add_argument("--price-provider", default="tiger", help="Adjusted price provider namespace.")
    parser.add_argument("--region", default="us", help="Dataset region.")
    parser.add_argument("--sec-type", default="stock", help="Security type.")
    parser.add_argument("--freq", default="1d", help="Dataset frequency.")
    parser.add_argument("--variant", default=None, help="Optional factor variant.")
    parser.add_argument("--start", default=None, help="Optional start date filter.")
    parser.add_argument("--end", default=None, help="Optional end date filter.")
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the HTML report in a browser after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FactorStore(root_dir=args.store_root)
    spec = FactorSpec(
        provider=args.provider,
        region=args.region,
        sec_type=args.sec_type,
        freq=args.freq,
        table_name=args.factor_name,
        variant=args.variant,
    )
    price_spec = AdjPriceSpec(
        provider=args.price_provider,
        region=args.region,
        sec_type=args.sec_type,
        freq=args.freq,
    )

    factor_frame = store.get_factor(spec, start=args.start, end=args.end, engine="duckdb")
    if "value" in factor_frame.columns and args.factor_name not in factor_frame.columns:
        factor_frame = factor_frame.rename(columns={"value": args.factor_name})
    price_frame = store.get_adj_price(price_spec, start=args.start, end=args.end)
    if factor_frame.empty:
        raise FileNotFoundError(
            f"Factor frame not found in the provider-layer factor store: {spec}. "
            f"Expected directory: {spec.dataset_dir(store.root_dir)}"
        )
    if price_frame.empty:
        raise FileNotFoundError(f"Adj price frame not found in the default factor store: {price_spec}")

    evaluation = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=args.factor_name,
        spec=spec,
        include_horizon=True,
        horizons=(1, 3, 5, 10, 20),
        horizon_quantiles=5,
    )

    # print("running summary()")
    # evaluation.summary(force_updated=True)
    # print("running horizon()")
    # evaluation.horizon(force_updated=True)
    # print("running full()")
    evaluation.full(force_updated=True)

    # print("\nSUMMARY")
    # print("tables:", evaluation.summary.tables())
    # print("imgs:", evaluation.summary.imgs())
    # print("report:", evaluation.summary.report())
    # summary_table = evaluation.summary.get_table("summary")
    # print(summary_table)

    # print("\nHORIZON")
    # print("tables:", evaluation.horizon.tables())
    # print("imgs:", evaluation.horizon.imgs())
    # print("report:", evaluation.horizon.report())
    # horizon_tables = evaluation.horizon.tables()
    # if horizon_tables:
    #     horizon_table = evaluation.horizon.get_table(horizon_tables[0])
    #     print(horizon_table.head())
    # horizon_imgs = evaluation.horizon.imgs()
    # if horizon_imgs:
    #     horizon_img = evaluation.horizon.get_img(horizon_imgs[0])
    #     print(f"opened horizon image: {horizon_imgs[0]} -> {horizon_img.size}")

    # print("\nFULL")
    # print("tables:", evaluation.full.tables())
    # print("imgs:", evaluation.full.imgs())
    # print("report:", evaluation.full.report())
    # report_path = evaluation.full.get_report(open_browser=args.open_report)
    # print("full report path:", report_path)
    # if evaluation.full.imgs():
    #     full_img = evaluation.full.get_img(evaluation.full.imgs()[0])
    #     print(f"opened full image: {evaluation.full.imgs()[0]} -> {full_img.size}")


if __name__ == "__main__":
    main()
