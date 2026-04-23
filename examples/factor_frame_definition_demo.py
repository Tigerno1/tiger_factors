"""Tiger API SimFin fetch -> FactorDefinition demo.

This example shows the structured factor-definition flow:

1. fetch real SimFin frames with ``tiger_api.sdk.client.fetch_data``
2. build a code-level company lookup from the fetched SimFin tables
3. register a sector classifier and a tech-only screen
4. register several structured factor definitions
5. run once and print the resulting factor frame

The demo keeps the research story explicit:

- classifier: sector labels
- screen: tech-only universe gate
- FactorDefinition: structured factor recipes
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
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import CrossSectionalResidualDefinition
from tiger_factors.factor_frame import EventDrivenFactorDefinition
from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_frame import IndustryNeutralMomentumDefinition
from tiger_factors.factor_frame import WeightedSumFactorDefinition
from tiger_factors.utils.merge import merge_by_keys


FETCH_RETURN_TYPE = "df"


def _fetch_simfin_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    tech_codes = (
        company_lookup.loc[company_lookup["sector"].eq("Technology"), "code"].dropna().drop_duplicates().head(6).tolist()
    )
    if not tech_codes:
        raise RuntimeError("no technology-sector codes returned")

    price = fetch_data(
        provider="simfin",
        name="eod_price",
        region="us",
        sec_type="stock",
        freq="1d",
        codes=tech_codes,
        return_type=FETCH_RETURN_TYPE,
    )
    if price.empty:
        raise RuntimeError("empty price panel")

    financial = fetch_data(
        provider="simfin",
        name="income_statement",
        region="us",
        sec_type="stock",
        freq="1q",
        codes=tech_codes,
        return_type=FETCH_RETURN_TYPE,
    )
    if financial.empty:
        raise RuntimeError("empty financial dataset")

    events = price.loc[:, ["date_", "code", "close"]].copy()
    events["event_flag"] = events.groupby("code")["close"].transform(
        lambda s: s.pct_change(5, fill_method=None).gt(0.05).astype(float)
    )
    events = events.loc[:, ["date_", "code", "event_flag"]]

    return price, financial, company_lookup, events, companies


def main() -> None:
    price_df, financial_df, company_lookup, event_df, companies = _fetch_simfin_inputs()

    registry = FactorDefinitionRegistry()
    registry.register_many(
        IndustryNeutralMomentumDefinition(
            name="industry_neutral_momentum",
            window=20,
            price_feed="price",
            price_column="close",
            classifier_name="sector",
            classifier_column="sector",
            neutralize_method="demean",
        ),
        CrossSectionalResidualDefinition(
            name="residual_momentum",
            target_feed="price",
            target_column="close",
            window=20,
            regressors=[("financial", "net_income"), ("financial", "revenue")],
        ),
        EventDrivenFactorDefinition(
            name="event_alpha",
            event_feed="events",
            event_column="event_flag",
            price_feed="price",
            price_column="close",
            window=5,
        ),
    )
    registry.register(
        WeightedSumFactorDefinition(
            name="composite_alpha",
            components={
                "industry_neutral_momentum": 0.5,
                "residual_momentum": 0.3,
                "event_alpha": 0.2,
            },
            component_fns={
                "industry_neutral_momentum": registry.get("industry_neutral_momentum"),
                "residual_momentum": registry.get("residual_momentum"),
                "event_alpha": registry.get("event_alpha"),
            },
        )
    )

    research = FactorResearchEngine(
        freq="1d",
        bday_lag=True,
        use_point_in_time=True,
        start="2020-01-01",
        end="2023-01-01",
        definition_registry=registry,
    )
    research.feed_price(price_df)
    research.feed_financial(financial_df, lag_sessions=1)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.feed("events", event_df, align_mode="code_date")
    research.add_classifier(
        "sector",
        lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"],
    )
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("Technology"),
    )

    research.add_definition("industry_neutral_momentum")
    research.add_definition("residual_momentum")
    research.add_definition("event_alpha")
    research.add_definition("composite_alpha")

    result = research.run()

    print("companies:", companies.shape)
    print("price_df:", price_df.shape)
    print("financial_df:", financial_df.shape)
    print("event_df:", event_df.shape)
    print("company_lookup:", company_lookup.shape)
    print("combined_frame:", result.combined_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print("screen_frames:", sorted(result.screen_frames))
    print("classifier_frames:", sorted(result.classifier_frames))
    print("screen_mask rows:", 0 if result.screen_mask is None else len(result.screen_mask))
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
