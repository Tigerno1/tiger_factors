from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import pandas as pd

from .factor_functions import SignalMetadata
from .factor_functions import available_factors
from .factor_functions import factor_metadata


@dataclass(frozen=True)
class TraditionalFactorGroup:
    name: str
    description: str
    keywords: tuple[str, ...]
    members: tuple[str, ...]


_GROUP_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "analyst",
        "Analyst forecasts, revisions, and recommendation-driven signals.",
        (
            "analyst",
            "forecast",
            "recomm",
            "revision",
            "meanest",
            "uprecomm",
            "recommendation",
        ),
    ),
    (
        "event",
        "Corporate event, announcement, and event-study signals.",
        (
            "announcement",
            "event",
            "spinoff",
            "dividend",
            "merger",
            "acquisition",
            "split",
            "buyback",
            "repurchase",
        ),
    ),
    (
        "issuance",
        "Equity and debt issuance / financing signals.",
        (
            "issuance",
            "share issue",
            "share issues",
            "net stock issues",
            "net equity issuance",
            "net debt issuance",
            "debt issuance",
            "equity issuance",
        ),
    ),
    (
        "distress",
        "Financial distress, failure probability, and bankruptcy-risk signals.",
        (
            "altman",
            "ohlson",
            "kaplan-zingales",
            "kaplan zingales",
            "failure probability",
            "distress",
            "bankrupt",
            "pitroski",
            "piotroski",
            "f-score",
            "f score",
        ),
    ),
    (
        "seasonality",
        "Calendar and seasonal patterns.",
        (
            "seasonality",
            "off-season",
            "off season",
            "momseason",
            "momoffseason",
            "january",
        ),
    ),
    (
        "momentum",
        "Trend-following, reversal, and lagged-return signals.",
        (
            "momentum",
            "reversal",
            "residual momentum",
            "price momentum",
            "highest 5 days of return",
            "lagged returns",
            "lagged return",
            "short-term reversal",
            "long-term reversal",
            "year 1-lagged return",
            "years 2-5 lagged returns",
            "years 6-10 lagged returns",
            "years 11-15 lagged returns",
            "years 16-20 lagged returns",
        ),
    ),
    (
        "liquidity",
        "Liquidity, turnover, spread, and trading-volume signals.",
        (
            "liquidity",
            "illiquidity",
            "turnover",
            "volume",
            "bid-ask",
            "bid ask",
            "zerotrade",
            "zero trade",
            "amihud",
            "share turnover",
            "dollar trading volume",
            "high-low bid-ask spread",
        ),
    ),
    (
        "risk",
        "Beta, volatility, skewness, and tail-risk signals.",
        (
            "beta",
            "volatility",
            "skew",
            "coskew",
            "tail risk",
            "downside",
            "correlation",
            "duration",
            "failure probability",
            "idiosyncratic",
            "market beta",
            "dimson beta",
            "frazzini-pedersen market beta",
        ),
    ),
    (
        "quality",
        "Quality-minus-junk style and broad quality composites.",
        (
            "quality minus junk",
            "quality",
            "qmj",
            "f-score",
            "pitroski",
            "piotroski",
        ),
    ),
    (
        "profitability",
        "Profitability, operating efficiency, and return-on-asset style signals.",
        (
            "profit",
            "return on equity",
            "return on assets",
            "return on net operating assets",
            "gross profits",
            "operating profits",
            "cash-based operating profits",
            "cash based operating profits",
            "operating cash flow",
            "profit margin",
            "earnings persistence",
        ),
    ),
    (
        "growth",
        "Growth, growth surprise, and acceleration signals.",
        (
            "profit growth",
            "sales growth",
            "earnings growth",
            "revenue surprise",
            "earnings surprise",
            "standardized earnings surprise",
            "standardized revenue surprise",
            "tax expense surprise",
            "consecutive quarters with earnings increases",
        ),
    ),
    (
        "investment",
        "Corporate investment, asset expansion, and capital expenditure signals.",
        (
            "investment",
            "asset growth",
            "capex",
            "capital turnover",
            "asset turnover",
            "inventory",
            "r&d",
            "r and d",
            "working capital",
            "net operating assets",
            "abnormal corporate investment",
            "book debt",
            "asset tangibility",
            "labor force efficiency",
            "change in current operating assets",
            "change in current operating liabilities",
            "change in noncurrent operating assets",
            "change in noncurrent operating liabilities",
            "change sales minus change inventory",
            "change sales minus change receivables",
            "change sales minus change sg&a",
            "change sales minus change sga",
            "change ppe and inventory",
            "change gross margin minus change sales",
            "change in long-term investments",
            "change in short-term investments",
            "change in long-term net operating assets",
            "change in net financial assets",
            "change in net noncurrent operating assets",
            "change in operating cash flow to assets",
            "growth in book debt",
        ),
    ),
    (
        "accruals",
        "Accrual and cash-flow quality signals.",
        (
            "accrual",
            "cash flow volatility",
            "operating accrual",
            "total accrual",
            "percent operating accrual",
            "percent total accrual",
            "operating cash flow to assets",
            "operating cash flow-to-market",
        ),
    ),
    (
        "leverage",
        "Leverage, debt burden, and capital-structure signals.",
        (
            "leverage",
            "debt-to-market",
            "debt to market",
            "net debt",
            "book leverage",
            "low leverage",
            "book debt",
        ),
    ),
    (
        "value",
        "Valuation, book-to-market, and price-to-fundamentals signals.",
        (
            "value",
            "book-to-market",
            "book to market",
            "market-to-book",
            "market to book",
            "price-to-book",
            "price to book",
            "earnings-to-price",
            "earnings to price",
            "sales-to-market",
            "sales to market",
            "assets-to-market",
            "assets to market",
            "free cash flow-to-price",
            "free cash flow to price",
            "intrinsic value-to-market",
            "intrinsic value to market",
            "ebitda-to-market enterprise value",
            "book-to-market enterprise value",
            "book-to-market equity",
            "dividend yield",
            "payout yield",
            "net payout yield",
            "sales-to-price",
            "sales to price",
            "price per share",
            "cash-to-assets",
        ),
    ),
    (
        "size",
        "Firm size and market-capitalization signals.",
        (
            "size",
            "market equity",
            "market capitalization",
            "market cap",
            "firm age",
        ),
    ),
    (
        "other",
        "Signals that do not cleanly match a canonical family.",
        (),
    ),
)


def _factor_text(meta: SignalMetadata) -> str:
    parts = [
        meta.name,
        meta.long_description or "",
        meta.detailed_definition or "",
        meta.category or "",
        meta.authors or "",
        meta.journal or "",
    ]
    return " ".join(parts).lower()


def _matches(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


@lru_cache(maxsize=1)
def traditional_factor_groups() -> tuple[TraditionalFactorGroup, ...]:
    members_by_group: dict[str, list[str]] = {name: [] for name, _, _ in _GROUP_RULES}
    for factor_name in available_factors():
        meta = factor_metadata(factor_name)
        text = _factor_text(meta)
        assigned = "other"
        for group_name, _, keywords in _GROUP_RULES:
            if group_name == "other":
                continue
            if _matches(text, keywords):
                assigned = group_name
                break
        members_by_group.setdefault(assigned, []).append(factor_name)

    groups: list[TraditionalFactorGroup] = []
    for group_name, description, keywords in _GROUP_RULES:
        groups.append(
            TraditionalFactorGroup(
                name=group_name,
                description=description,
                keywords=tuple(keywords),
                members=tuple(sorted(members_by_group.get(group_name, []))),
            )
        )
    return tuple(groups)


@lru_cache(maxsize=1)
def traditional_factor_group_index() -> dict[str, tuple[str, ...]]:
    return {group.name: group.members for group in traditional_factor_groups()}


def traditional_factor_group_names() -> tuple[str, ...]:
    return tuple(group.name for group in traditional_factor_groups())


def traditional_factor_group_for_signal(signal_name: str) -> str:
    text = _factor_text(factor_metadata(signal_name))
    for group_name, _, keywords in _GROUP_RULES:
        if group_name == "other":
            continue
        if _matches(text, keywords):
            return group_name
    return "other"


def traditional_factor_group_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in traditional_factor_groups():
        for signal_name in group.members:
            meta = factor_metadata(signal_name)
            rows.append(
                {
                    "signal_name": signal_name,
                    "group_name": group.name,
                    "category": meta.category,
                    "source_bucket": meta.output_dir,
                    "authors": meta.authors,
                    "year": meta.year,
                    "long_description": meta.long_description,
                    "detailed_definition": meta.detailed_definition,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "signal_name",
                "group_name",
                "category",
                "source_bucket",
                "authors",
                "year",
                "long_description",
                "detailed_definition",
            ]
        )
    return frame.sort_values(["group_name", "signal_name"]).reset_index(drop=True)


def traditional_factor_group_summary() -> pd.DataFrame:
    frame = traditional_factor_group_frame()
    if frame.empty:
        return pd.DataFrame(columns=["group_name", "count"])
    summary = (
        frame.groupby("group_name", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "group_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def find_traditional_factor_group(name: str) -> TraditionalFactorGroup:
    normalized = str(name)
    for group in traditional_factor_groups():
        if group.name == normalized:
            return group
    raise KeyError(f"Unknown traditional factor group: {name!r}")


__all__ = [
    "TraditionalFactorGroup",
    "find_traditional_factor_group",
    "traditional_factor_group_for_signal",
    "traditional_factor_group_frame",
    "traditional_factor_group_index",
    "traditional_factor_group_names",
    "traditional_factor_group_summary",
    "traditional_factor_groups",
]
