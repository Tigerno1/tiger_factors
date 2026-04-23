"""Tiger API SimFin fetch -> sector classifier -> screen -> momentum demo.

This example follows the three-layer style:

1. fetch real SimFin frames with ``tiger_api.sdk.client.fetch_data``
2. feed materialized data into ``FactorResearchEngine``
3. register a classifier, a screen, and a factor
4. run once and print the resulting research frames

The flow is intentionally thin and keeps the research semantics explicit:

- classifier: sector labels
- screen: tech-only universe gate
- factor: momentum signal
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
from tiger_factors.utils.merge import merge_by_keys


FETCH_RETURN_TYPE = "df"


def _fetch_simfin_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    if price_df.empty:
        raise RuntimeError("empty price panel")

    return price_df, company_lookup, companies


def main() -> None:
    price_df, company_lookup, companies = _fetch_simfin_inputs()
    research = FactorResearchEngine(
        freq="1d",
        bday_lag=True,
        use_point_in_time=True,
        start="2020-01-01",
        end="2023-01-01",
    )
    research.feed_price(price_df)
    research.feed("company_lookup", company_lookup, align_mode="code", code_column="code")
    research.add_classifier(
        "sector",
        lambda ctx: ctx.feed_frame("company_lookup").set_index("code")["sector"],
    )
    research.add_screen(
        "tech_only",
        lambda ctx: ctx.classifier("sector").set_index("code")["sector"].eq("Technology"),
    )
    research.add_factor(
        "momentum",
        lambda ctx: ctx.feed_wide("price", "close").pct_change(10, fill_method=None),
    )

    result = research.run()
    print("companies:", companies.shape)
    print("price_df:", price_df.shape)
    print("company_lookup:", company_lookup.shape)
    print("combined_frame:", result.combined_frame.shape)
    print("factor_frame:", result.factor_frame.shape)
    print("screen_frames:", sorted(result.screen_frames))
    print("classifier_frames:", sorted(result.classifier_frames))
    print(result.factor_frame.head())


if __name__ == "__main__":
    main()
