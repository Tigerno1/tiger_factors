from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors import Alpha101Engine
from tiger_factors.factor_evaluation.evaluation import SingleFactorEvaluation
from tiger_factors.factor_store import AdjPriceSpec, FactorSpec, FactorStore, TigerFactorLibrary


START = "2023-01-01"
END = "2024-12-31"
CODES = ["AAPL", "MSFT", "NVDA", "AMZN"]
PERIODS = (1, 5, 10)
PRICE_PROVIDER = "yahoo"
CLASSIFICATION_PROVIDER = "simfin"
CLASSIFICATION_DATASET = "companies"


def _print_section_assets(store: FactorStore, spec: FactorSpec, section_name: str) -> None:
    accessor = store.evaluation.section(spec, section_name)
    print(f"\n[{section_name}]")
    print(f"tables: {accessor.tables()}")
    print(f"imgs:   {accessor.imgs()}")
    try:
        report_path = accessor.get_report(open_browser=False)
        print(f"report: {report_path}")
    except FileNotFoundError:
        print("report: (not found)")


def main() -> None:
    store = FactorStore()
    library = TigerFactorLibrary(store=store, verbose=False)

    alpha_input = library.build_alpha101_input(
        codes=CODES,
        start=START,
        end=END,
        price_provider=PRICE_PROVIDER,
        classification_provider=CLASSIFICATION_PROVIDER,
        classification_dataset=CLASSIFICATION_DATASET,
    )
    if alpha_input.empty:
        raise RuntimeError("alpha101 input is empty; check codes or date range.")

    alpha_engine = Alpha101Engine(alpha_input)
    alpha_001 = alpha_engine.compute(1)
    alpha_001 = alpha_001.loc[
        (alpha_001["date_"] >= pd.Timestamp(START)) & (alpha_001["date_"] <= pd.Timestamp(END))
    ].reset_index(drop=True)
    factor_series = alpha_001.set_index(["date_", "code"])["alpha_001"].sort_index()
    prices = alpha_input.pivot(index="date_", columns="code", values="close").sort_index()

    factor_spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="alpha_001",
    )
    adj_price_spec = AdjPriceSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
    )
    adj_saved = store.save_adj_price(
        adj_price_spec,
        alpha_input,
        force_updated=True,
    )
    factor_saved = store.save_factor(
        factor_spec,
        alpha_001,
        force_updated=True,
    )

    single = SingleFactorEvaluation(
        factor=factor_series,
        prices=prices,
        factor_column="alpha_001",
        spec=factor_spec,
        periods=PERIODS,
        avgretplot=(1, 3),
    )

    core = single.evaluate()
    single.full(format="all", force_updated=True)
    full_report_path = store.evaluation.full(factor_spec).get_report(open_browser=True)

    print("alpha101 alpha_001 saved factor:")
    print(f"  parquet:  {factor_saved.files[0]}")
    print(f"  factor manifest: {factor_saved.manifest_path}")
    print(f"  adj data: {adj_saved.files[0]}")
    print(f"  adj manifest: {adj_saved.manifest_path}")
    print("\ncore evaluation:")
    print(core)
    print("\nfull report:")
    print(f"  report html: {full_report_path}")

    print("\nFactorStore lookup by module/section:")
    for section_name in ["summary", "returns", "information", "turnover", "event_returns", "horizon"]:
        _print_section_assets(store, factor_spec, section_name)

    summary_df = store.evaluation.summary(factor_spec).get_table()
    mean_ic = store.evaluation.information(factor_spec).get_table("mean_information_coefficient")
    turnover_summary = store.evaluation.turnover(factor_spec).get_table("turnover_summary")
    horizon_result = store.evaluation.horizon(factor_spec).get_table("horizon_result")
    horizon_img = store.evaluation.horizon(factor_spec).get_img("horizon_result")

    print("\nDirect reads from FactorStore:")
    print(f"  summary shape: {summary_df.shape}")
    print(f"  mean_ic shape: {mean_ic.shape}")
    print(f"  turnover_summary shape: {turnover_summary.shape}")
    print(f"  horizon_result shape: {horizon_result.shape}")
    print(f"  horizon_result image size: {horizon_img.size}")
    print(f"  full report html: {full_report_path}")


if __name__ == "__main__":
    main()
