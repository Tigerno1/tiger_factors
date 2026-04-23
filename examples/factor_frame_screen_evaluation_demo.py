"""Tiger API SimFin fetch -> pre-screened factor creation -> evaluation demo.

This is the recommended starting point when you want to:

1. choose a universe with common pre-factor screens
2. build factors only on the screened universe
3. evaluate each factor separately after construction

This example shows a practical research flow:

1. fetch real SimFin frames with ``tiger_api.sdk.client.fetch_data``
2. build a mixed universe from company + industry tables
3. register a sector classifier
4. apply common pre-factor screens
5. compute screened factor outputs
6. evaluate each factor with ``SingleFactorEvaluation``

Common screens in day-to-day factor research usually include:

- sector / industry universe filters
- minimum price filters to avoid penny stocks
- minimum history filters so rolling windows are valid
- financial coverage filters to avoid sparse fundamentals

The example keeps those ideas explicit and practical.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SIMFIN_DB_URL = "sqlite:///file:/Volumes/Quant_Disk/tiger_quant/data/simfin_us_stock.db?immutable=1&uri=true"
os.environ.setdefault("TIGER_DB_URL_SIMFIN_US_STOCK", SIMFIN_DB_URL)

from tiger_api.sdk.client import fetch_data
from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_evaluation.evaluation import SingleFactorEvaluation
from tiger_factors.utils.merge import merge_by_keys


FETCH_RETURN_TYPE = "df"
OUTPUT_ROOT = PROJECT_ROOT / "tiger_analysis_outputs" / "tiger_factor_frame_screen_evaluation_demo"


def _fetch_simfin_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    companies = fetch_data(
        provider="simfin",
        name="companies",
        region="us",
        sec_type="stock",
        freq="static",
        return_type=FETCH_RETURN_TYPE,
    )
    industry = fetch_data(
        provider="simfin",
        name="industry",
        region="us",
        sec_type="stock",
        freq="static",
        return_type=FETCH_RETURN_TYPE,
    )
    company_lookup = merge_by_keys([companies, industry], join_keys=["industry_id"])
    if "sector" not in company_lookup.columns:
        raise RuntimeError("company lookup does not contain a sector column")

    tech_codes = (
        company_lookup.loc[company_lookup["sector"].eq("Technology"), "code"].dropna().drop_duplicates().head(6).tolist()
    )
    non_tech_codes = (
        company_lookup.loc[~company_lookup["sector"].eq("Technology"), "code"].dropna().drop_duplicates().head(6).tolist()
    )
    codes = tech_codes + non_tech_codes
    if not codes:
        raise RuntimeError("no codes returned from the mixed universe")

    price_df = fetch_data(
        provider="simfin",
        name="eod_price",
        region="us",
        sec_type="stock",
        freq="1d",
        codes=codes,
        return_type=FETCH_RETURN_TYPE,
    )
    if price_df.empty:
        raise RuntimeError("empty price panel")

    financial_df = fetch_data(
        provider="simfin",
        name="income_statement",
        region="us",
        sec_type="stock",
        freq="1q",
        codes=codes,
        return_type=FETCH_RETURN_TYPE,
    )
    if financial_df.empty:
        raise RuntimeError("empty financial dataset")

    return price_df, financial_df, company_lookup, companies, codes


def _build_factor_research(
    price_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    company_lookup: pd.DataFrame,
) -> FactorResearchEngine:
    research = FactorResearchEngine(
        freq="1d",
        bday_lag=True,
        use_point_in_time=True,
        start="2020-01-01",
        end="2023-01-01",
    )
    research.feed_price(price_df)
    research.feed_financial(financial_df, lag_sessions=1)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.add_classifier(
        "sector",
        lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"],
    )

    # Common pre-factor screens:
    # - sector gate
    # - minimum price floor
    # - enough trading history for rolling windows
    # - financial fields present on the research panel
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("Technology"),
    )
    research.add_screen(
        "price_floor",
        lambda ctx: ctx.feed_wide("price", "close").gt(5.0),
    )
    research.add_screen(
        "history_ok",
        lambda ctx: ctx.feed_wide("price", "close").rolling(20, min_periods=10).count().ge(10),
    )
    research.add_screen(
        "fundamentals_ok",
        lambda ctx: ctx.feed_wide("financial", "net_income").notna()
        & ctx.feed_wide("financial", "revenue").notna(),
    )

    research.add_factor(
        "screened_momentum",
        lambda ctx: ctx.feed_wide("price", "close").pct_change(20, fill_method=None),
    )
    research.add_factor(
        "screened_quality",
        lambda ctx: ctx.feed_wide("financial", "net_income").div(
            ctx.feed_wide("financial", "revenue").replace(0, pd.NA)
        ),
    )
    return research


def _evaluate_factor_frame(
    factor_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    factor_column: str,
) -> None:
    factor_panel = factor_frame.loc[:, ["date_", "code", factor_column]].dropna().copy()
    if factor_panel.empty:
        print(f"{factor_column}: empty after screening")
        return

    research = SingleFactorEvaluation(
        factor_frame=factor_panel,
        price_frame=price_frame,
        factor_column=factor_column,
        include_horizon=False,
    )
    evaluation = research.evaluate()
    research.full()
    report_path = research.full.get_report(open_browser=False)

    print(f"\n[{factor_column}]")
    print(
        {
            "ic_mean": evaluation.ic_mean,
            "ic_ir": evaluation.ic_ir,
            "rank_ic_mean": evaluation.rank_ic_mean,
            "sharpe": evaluation.sharpe,
            "turnover": evaluation.turnover,
            "fitness": evaluation.fitness,
        }
    )
    print("report_dir:", report_path.parent)
    print("table_names:", research.full.tables())
    print("img_names:", research.full.imgs())


def main() -> None:
    price_df, financial_df, company_lookup, companies, codes = _fetch_simfin_inputs()
    research = _build_factor_research(price_df, financial_df, company_lookup)
    result = research.run()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("companies:", companies.shape)
    print("codes:", len(codes))
    print("price_df:", price_df.shape)
    print("financial_df:", financial_df.shape)
    print("company_lookup:", company_lookup.shape)
    print("screen_frames:", sorted(result.screen_frames))
    print("classifier_frames:", sorted(result.classifier_frames))
    print("screen_mask rows:", 0 if result.screen_mask is None else len(result.screen_mask))
    print("screened_combined_frame:", result.combined_frame.shape)
    print("screened_factor_frame:", result.factor_frame.shape)

    for factor_column in ("screened_momentum", "screened_quality"):
        _evaluate_factor_frame(
            result.factor_frame,
            price_df,
            factor_column,
        )


if __name__ == "__main__":
    main()
