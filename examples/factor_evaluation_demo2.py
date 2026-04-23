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



def main() -> None:
    start = "2020-01-01"
    end = "2022-12-31"

    provider = "tiger"
    region = "us"
    sec_type = "stock"
    freq = "1d"
    group = "alpha_101"
    table_names = [f"alpha_{i:03d}" for i in range(2, 102)]
    store = FactorStore()
    specs = []
    for table_name in table_names: 
        spec = FactorSpec(
            provider=provider,
            region=region,
            sec_type=sec_type,
            freq=freq,
            group=group,
            table_name=table_name,
        )
        specs.append(spec)

    provider = "simfin"
    price_spec = AdjPriceSpec(
        provider=provider,
        region=region,
        sec_type=sec_type,
        freq=freq,
        table_name = "adj_price"
    )
    for factor_name, spec in zip(table_names, specs): 
        factor_frame = store.get_factor(spec, start=start, end=end, engine="duckdb")
        if "value" in factor_frame.columns and factor_name not in factor_frame.columns:
            factor_frame = factor_frame.rename(columns={"value": factor_name})
        price_frame = store.get_adj_price(price_spec, start=start, end=end)
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
            factor_column=factor_name,
            spec=spec,
            include_horizon=True,
            horizons=(1, 5, 10),
            horizon_quantiles=3,
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
