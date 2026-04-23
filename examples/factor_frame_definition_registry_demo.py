"""Tiger API SimFin fetch -> FactorDefinition registry demo.

This example keeps the registry story intentionally short:

1. fetch real SimFin frames with ``tiger_api.sdk.client.fetch_data``
2. build a code-level company lookup from the fetched SimFin tables
3. register several structured factor definitions with ``register_many(...)``
4. enable them later by name with ``add_definition("...")``
5. run once and print the resulting factor frame

The flow is intentionally compact and shows the registry-first style:

- registry: register definitions once
- research: enable by name later
- factor definitions: structured factor recipes
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SIMFIN_DB_URL = "sqlite:///file:/Volumes/Quant_Disk/tiger_quant/data/simfin_us_stock.db?immutable=1&uri=true"
os.environ.setdefault("TIGER_DB_URL_SIMFIN_US_STOCK", SIMFIN_DB_URL)

from tiger_api.sdk.client import fetch_data
from tiger_factors.factor_frame import CrossSectionalResidualDefinition
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_frame import IndustryNeutralMomentumDefinition
from tiger_factors.utils.merge import merge_by_keys


FETCH_RETURN_TYPE = "df"


def main() -> None:
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

    price_df = fetch_data(
        provider="simfin",
        name="eod_price",
        region="us",
        sec_type="stock",
        freq="1d",
        codes=tech_codes,
        return_type=FETCH_RETURN_TYPE,
    )
    financial_df = fetch_data(
        provider="simfin",
        name="income_statement",
        region="us",
        sec_type="stock",
        freq="1q",
        codes=tech_codes,
        return_type=FETCH_RETURN_TYPE,
    )

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

    result = research.run()

    print("companies:", companies.shape)
    print("industry:", industry.shape)
    print("price_df:", price_df.shape)
    print("financial_df:", financial_df.shape)
    print("company_lookup:", company_lookup.shape)
    print("registry_names:", registry.names())
    print("combined_frame:", result.combined_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print("screen_frames:", sorted(result.screen_frames))
    print("classifier_frames:", sorted(result.classifier_frames))
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
