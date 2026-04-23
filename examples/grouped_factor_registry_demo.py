"""Grouped factor engine on real structured definitions.

This example shows how the grouped factor engine can reuse a real
``FactorDefinitionRegistry`` instead of synthetic panels:

1. fetch SimFin price / financial / company / event data
2. register structured factor definitions
3. build a ``FactorResearchEngine`` context from real feeds
4. combine definitions into factor families with ``FactorGroupEngine``
5. validate family evidence and run a backtest + portfolio report
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

SIMFIN_DB_URL = "sqlite:///file:/Volumes/Quant_Disk/tiger_quant/data/simfin_us_stock.db?immutable=1&uri=true"
os.environ.setdefault("TIGER_DB_URL_SIMFIN_US_STOCK", SIMFIN_DB_URL)

from tiger_api.sdk.client import fetch_data
from tiger_factors.factor_evaluation import validate_series
from tiger_factors.factor_frame import CrossSectionalResidualDefinition
from tiger_factors.factor_frame import EventDrivenFactorDefinition
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import FactorGroupEngine
from tiger_factors.factor_frame import FactorGroupSpec
from tiger_factors.factor_frame import FactorFrameContext
from tiger_factors.factor_frame import FactorResearchEngine
from tiger_factors.factor_frame import IndustryNeutralMomentumDefinition
from tiger_factors.factor_frame import WeightedSumFactorDefinition
from tiger_factors.multifactor_evaluation import run_factor_backtest
from tiger_factors.multifactor_evaluation import validate_factor_family
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest
from tiger_factors.utils.merge import merge_by_keys


def _fetch_simfin_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    companies = fetch_data(
        provider="simfin",
        name="companies",
        region="us",
        sec_type="stock",
        freq="static",
        return_type="df",
    )
    industry = fetch_data(
        provider="simfin",
        name="industry",
        region="us",
        sec_type="stock",
        freq="static",
        return_type="df",
    )
    company_lookup = merge_by_keys([companies, industry], join_keys=["industry_id"])
    tech_codes = (
        company_lookup.loc[company_lookup["sector"].eq("Technology"), "code"]
        .dropna()
        .drop_duplicates()
        .head(8)
        .tolist()
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
        return_type="df",
    )
    financial = fetch_data(
        provider="simfin",
        name="income_statement",
        region="us",
        sec_type="stock",
        freq="1q",
        codes=tech_codes,
        return_type="df",
    )
    if price.empty or financial.empty:
        raise RuntimeError("simfin fetch returned empty frames")

    events = price.loc[:, ["date_", "code", "close"]].copy()
    events["event_flag"] = events.groupby("code")["close"].transform(
        lambda s: s.pct_change(5, fill_method=None).gt(0.05).astype(float)
    )
    events = events.loc[:, ["date_", "code", "event_flag"]]
    return price, financial, company_lookup, events, companies


def _build_registry() -> FactorDefinitionRegistry:
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
    return registry


def _build_research(
    price_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    company_lookup: pd.DataFrame,
    event_df: pd.DataFrame,
    registry: FactorDefinitionRegistry,
) -> FactorResearchEngine:
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
    return research


def _build_group_engine(registry: FactorDefinitionRegistry) -> FactorGroupEngine:
    engine = FactorGroupEngine(definition_registry=registry)
    engine.register_groups(
        FactorGroupSpec(
            name="signal_family",
            members=("industry_neutral_momentum", "residual_momentum"),
            combine_method="weighted_sum",
            weights={"industry_neutral_momentum": 0.6, "residual_momentum": 0.4},
            metadata={"theme": "signal"},
        ),
        FactorGroupSpec(
            name="event_family",
            members=("event_alpha", "composite_alpha"),
            combine_method="mean",
            metadata={"theme": "event"},
        ),
        FactorGroupSpec(
            name="core_family",
            members=("industry_neutral_momentum", "residual_momentum", "event_alpha", "composite_alpha"),
            combine_method="zscore_mean",
            metadata={"theme": "core"},
        ),
    )
    return engine


def main() -> None:
    output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "grouped_factor_registry_demo"
    price_df, financial_df, company_lookup, event_df, companies = _fetch_simfin_inputs()
    registry = _build_registry()
    research = _build_research(price_df, financial_df, company_lookup, event_df, registry)
    research_result = research.run()
    ctx = FactorFrameContext(
        feeds={feed.name: feed for feed in research_result.feeds},
        combined_frame=research_result.combined_frame,
        config=research.engine.build_context().build_config,
        screen_frames=research_result.screen_frames,
        classifier_frames=research_result.classifier_frames,
        screen_mask=research_result.screen_mask,
    )
    group_engine = _build_group_engine(registry)
    result = group_engine.run(ctx)

    close_panel = (
        price_df.loc[:, ["date_", "code", "close"]]
        .pivot_table(index="date_", columns="code", values="close", aggfunc="last")
        .sort_index()
    )
    close_panel.index = pd.to_datetime(close_panel.index)

    validation_rows: list[dict[str, object]] = []
    for group_name, panel in result.group_panels.items():
        if panel.empty:
            continue
        daily_mean = panel.mean(axis=1).dropna()
        validation = validate_series(
            daily_mean,
            metric_name=group_name,
            n_bootstrap=64,
            n_permutations=64,
            random_state=17,
        )
        validation_rows.append(
            {
                "factor_name": group_name,
                "p_value": validation.p_value,
                "fitness": abs(validation.observed),
            }
        )

    family_report = validate_factor_family(
        pd.DataFrame(validation_rows),
        method="bh",
        alpha=0.05,
        cluster_threshold=0.65,
        factor_dict=result.member_panels,
    )

    backtest, stats = run_factor_backtest(
        result.group_panels["core_family"],
        close_panel,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=True,
    )
    report = run_portfolio_from_backtest(
        backtest,
        output_dir=output_dir,
        report_name="grouped_factor_registry",
    )

    print("companies:", companies.shape)
    print("registry names:", registry.names())
    print("group summary:")
    print(result.summary.to_string(index=False))
    print("\nvalidation table:")
    print(family_report["table"].to_string(index=False))
    print("\nbacktest stats:")
    print(pd.DataFrame(stats).T.to_string())
    if report is not None:
        print("\nportfolio report:")
        print(report.to_summary())


if __name__ == "__main__":
    main()
