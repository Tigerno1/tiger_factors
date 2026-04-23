"""SP500 -> Alpha101 -> factor store -> evaluation demo.

This is the shortest "research engine" example for the new Alpha101 flow.
It does not use ``TigerFactorLibrary``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import replace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_api.sdk.client import fetch_codes
from tiger_api.sdk.client import fetch_data
from tiger_factors.factor_evaluation.evaluation import SingleFactorEvaluation
from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import OthersSpec
from tiger_factors.utils.merge import merge_by_keys
from tiger_factors.examples.sp500_alpha101_researchengine_strategies import Alpha101ResearchCalculator
from tiger_reference.adjustments import adj_df


START = "2020-01-01"
END = "2024-12-31"
UNIVERSE_PROVIDER = "github"
UNIVERSE_NAME = "sp500_constituents"
PRICE_PROVIDER = "yahoo"
PRICE_NAME = "eod_price"
CLASSIFICATION_PROVIDER = "simfin"
COMPANY_NAME = "companies"
INDUSTRY_NAME = "industry"
FACTOR_SPEC_COMMON = {
    "provider": "tiger",
    "region": "us",
    "sec_type": "stock",
    "freq": "1d",
}

if __name__ == "__main__":
    factor_store = FactorStore()

    # print("[1/5] Fetching S&P 500 universe")
    # codes = fetch_codes(
    #     provider=UNIVERSE_PROVIDER,
    #     name=UNIVERSE_NAME,
    #     region="us",
    #     sec_type="stock",
    #     freq="1d",
    #     at=END,
    # )
    # print(f"  codes: {len(codes)}")

    # print("[2/5] Fetching price data")
    # price_frame = fetch_data(
    #     provider=PRICE_PROVIDER,
    #     name=PRICE_NAME,
    #     region="us",
    #     sec_type="stock",
    #     freq="1d",
    #     codes=codes,
    #     start=START,
    #     end=END,
    #     as_ex=True,
    #     return_type="df",
    # )
    # price_frame = adj_df(price_frame, drop_adj_close=True, dividends=False, history=False)

    # print("[3/5] Fetching company classification data")
    # companies = fetch_data(
    #     provider=CLASSIFICATION_PROVIDER,
    #     name=COMPANY_NAME,
    #     region="us",
    #     sec_type="stock",
    #     freq="static",
    #     codes=codes,
    #     return_type="df",
    # )
    # industries = fetch_data(
    #     provider=CLASSIFICATION_PROVIDER,
    #     name=INDUSTRY_NAME,
    #     region="us",
    #     sec_type="stock",
    #     freq="static",
    #     return_type="df",
    # )
    # classification_frame = merge_by_keys([companies, industries], join_keys=["industry_id"])

    # print("[4/5] Computing Alpha101 with FactorResearchEngine")
    # research = FactorResearchEngine(
    #     freq="1d",
    #     calendar="XNYS",
    #     bday_lag=True,
    #     as_ex=True,
    #     label_side="right",
    #     use_point_in_time=True,
    #     start=START,
    #     end=END,
    # )
    # research.feed_price(price_frame)
    # research.feed(
    #     "classification",
    #     classification_frame,
    #     name="classification",
    #     align_mode="code",
    #     code_column="code",
    # )
    # alpha_strategy = Alpha101ResearchCalculator(
    #     alpha_ids=tuple(range(1, 102)),
    #     spec=FactorSpec(table_name="alpha_001", **FACTOR_SPEC_COMMON),
    #     factor_store=factor_store,
    #     verbose=True,
    #     save=True,
    # )
    # research.add_factor("alpha101", alpha_strategy)
    # research.run()

    # print("[5/5] Evaluating Alpha101 summaries")
    # summary_rows: list[pd.DataFrame] = []
    # for factor_name in alpha_strategy.factor_names:
    #     print(f"    [evaluate {factor_name}] start", flush=True)
    #     factor_spec = replace(alpha_strategy.spec, table_name=factor_name)
    #     factor_frame = factor_store.get_factor(factor_spec)
    #     evaluation_engine = SingleFactorEvaluation(
    #         factor_frame=factor_frame,
    #         price_frame=price_frame,
    #         factor_column=getattr(factor_spec, "value_column", "value"),
    #         spec=factor_spec,
    #     )
    #     evaluation_engine.summary(save=True, force_updated=True)
    #     summary = evaluation_engine.summary().get_table()
    #     summary_rows.append(summary.assign(factor_name=factor_name))
    #     print(f"    [evaluate {factor_name}] done", flush=True)

    # spec = FactorSpec(
    #     table_name="alpha_001",
    #     provider="tiger",
    #     region="us",
    #     sec_type="stock",
    #     freq="1d",
    # )

    # summary_row = factor_store.evaluation.summary(spec).get_table()
    # print(summary_row.head())

    # summary_table = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    summary_spec = OthersSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="alpha101_summary",
        variant="summary",
    )
    # factor_store.save_others(summary_spec, summary_table, force_updated=True)
    loaded_summary_table = factor_store.get_others(summary_spec)
    print("\nsummary table head:")
    loaded_summary_table.to_csv("alpha101_summary.csv", index=False)
    print(loaded_summary_table.head(20))
