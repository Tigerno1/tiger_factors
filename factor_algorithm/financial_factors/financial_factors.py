from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import TigerFactorLibrary, normalize_dates, to_long_series
from tiger_factors.factor_store.conf import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.utils import panel_ops as po


META_COLUMNS = {
    "date_",
    "date",
    "code",
    "sector",
    "industry",
    "subindustry",
    "available_at",
    "report_date",
    "publish_date",
    "restated_date",
    "currency",
    "ticker",
    "simfin_id",
    "fiscal_year",
    "fiscal_period",
    "source_dataset",
}


BALANCE_COLUMNS = (
    "shares_basic",
    "shares_diluted",
    "cash_st_investments",
    "interbank_assets",
    "st_lt_investments",
    "accounts_notes_receivable",
    "net_loans",
    "net_fixed_assets",
    "inventories",
    "total_current_assets",
    "ppe_net",
    "lt_investments_receivables",
    "other_lt_assets",
    "total_noncurrent_assets",
    "total_assets",
    "payables_accruals",
    "total_deposits",
    "short_term_debt",
    "total_current_liabilities",
    "long_term_debt",
    "total_noncurrent_liabilities",
    "total_liabilities",
    "preferred_equity",
    "share_cap_apic",
    "treasury_stock",
    "retained_earnings",
    "total_equity",
    "total_liabilities_equity",
)

INCOME_COLUMNS = (
    "shares_basic",
    "shares_diluted",
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "operating_expenses",
    "sga",
    "rnd",
    "pretax_income",
    "income_tax_net",
    "income_cont_ops",
    "net_extraordinary",
    "net_income",
    "net_income_common",
    "da",
    "non_operating_income",
    "interest_expense_net",
    "pretax_income_adj",
    "abnormal_gains_losses",
)

CASHFLOW_COLUMNS = (
    "shares_basic",
    "shares_diluted",
    "net_income_starting_line",
    "da",
    "non_cash_items",
    "net_cfo",
    "dividends_paid",
    "cash_repayment_debt",
    "cash_repurchase_equity",
    "net_cff",
    "fx_effect",
    "net_change_cash",
    "provision_loan_losses",
    "change_working_capital",
    "change_fixed_assets_intangibles",
    "net_change_loans_interbank",
    "net_cash_acq_divest",
    "net_cfi",
    "net_change_investments",
)

DERIVED_COLUMNS = (
    "gross_profit_margin",
    "operating_margin",
    "net_profit_margin",
    "roe",
    "roa",
    "fcf_net_income",
    "current_ratio",
    "liabilities_to_equity_ratio",
    "debt_ratio",
    "basic_eps",
    "diluted_eps",
    "sales_per_share",
    "equity_per_share",
    "fcf_per_share",
    "dividends_per_share",
    "piotroski_f_score",
)


@dataclass(frozen=True)
class FinancialFactorBundleResult:
    name: str
    data: pd.DataFrame
    parquet_path: Path
    metadata_path: Path
    payload: dict[str, object]


class FinancialFactorEngine:
    """Compact SimFin financial-factor engine.

    The engine loads SimFin statements into a trading-date panel and then
    expands a curated set of base fundamentals through a small transform grid.
    The concrete subclasses only switch the statement frequency:

    - quarterly
    - annual
    - TTM
    """

    statement_freq: str = "1q"
    output_tag: str = "quarterly"
    include_derived_dataset: bool = False
    piotroski_lag_sessions: int = 63
    def __init__(
        self,
        *,
        library: TigerFactorLibrary,
        codes: list[str],
        start: str,
        end: str,
        variant: str | None = None,
        price_provider: str | None = None,
        classification_provider: str = "simfin",
        classification_dataset: str = "companies",
        output_dir: str | Path = DEFAULT_FACTOR_STORE_ROOT / "financial_factors",
        lookback_days: int = 540,
        monthly_output: bool = True,
        max_factors: int | None = None,
        as_ex: bool | None = None,
    ) -> None:
        self.library = library
        self.codes = list(dict.fromkeys(map(str, codes)))
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.variant = variant
        self.price_provider = price_provider or library.price_provider
        self.classification_provider = classification_provider
        self.classification_dataset = classification_dataset
        self.output_dir = Path(output_dir)
        self.lookback_days = int(lookback_days)
        self.monthly_output = bool(monthly_output)
        self.max_factors = None if max_factors is None else int(max_factors)
        self.as_ex = as_ex

        self._trading_dates: pd.DatetimeIndex | None = None
        self._panel: pd.DataFrame | None = None
        self._price_panel: pd.DataFrame | None = None
        self._valuation_panel: pd.DataFrame | None = None
        self._base_metric_map: OrderedDict[str, pd.Series] | None = None
        self._valuation_metric_map: OrderedDict[str, pd.Series] | None = None

    @property
    def variant_name(self) -> str:
        return self.variant or "base"

    @property
    def buffer_start(self) -> pd.Timestamp:
        return self.start - pd.Timedelta(days=self.lookback_days)

    @property
    def trading_dates(self) -> pd.DatetimeIndex:
        if self._trading_dates is None:
            close = self.library.price_panel(
                codes=self.codes,
                start=str(self.buffer_start.date()),
                end=str(self.end.date()),
                provider=self.price_provider,
                field="close",
                as_ex=self.as_ex,
            )
            self._trading_dates = pd.DatetimeIndex(close.index)
        return self._trading_dates

    def _series(self, name: str, default: float = np.nan) -> pd.Series:
        if self.panel.empty or name not in self.panel.columns:
            return pd.Series(default, index=self.panel.index, dtype=float)
        return pd.to_numeric(self.panel[name], errors="coerce")

    @staticmethod
    def _frame_series(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
        if frame.empty or name not in frame.columns:
            return pd.Series(default, index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce")

    def _safe_div(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denom = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
        numer = pd.to_numeric(numerator, errors="coerce")
        return (numer / denom).replace([np.inf, -np.inf], np.nan)

    def _safe_add(self, *series: pd.Series) -> pd.Series:
        out = pd.Series(0.0, index=self.panel.index, dtype=float)
        for item in series:
            out = out.add(pd.to_numeric(item, errors="coerce"), fill_value=0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def _safe_sub(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    def _safe_mul(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return (pd.to_numeric(left, errors="coerce") * pd.to_numeric(right, errors="coerce")).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    def _safe_log1p(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        numeric = numeric.where(numeric > -1.0)
        return pd.Series(np.log1p(numeric), index=numeric.index).replace([np.inf, -np.inf], np.nan)

    def _group_cs_rank(self, values: pd.Series, group_col: str) -> pd.Series:
        if group_col not in self.panel.columns:
            return pd.Series(np.nan, index=self.panel.index, dtype=float)
        temp = pd.DataFrame(
            {
                "date_": self.panel["date_"],
                "group": self.panel[group_col].astype(str),
                "value": pd.to_numeric(values, errors="coerce"),
            }
        )
        ranked = temp.groupby(["date_", "group"], sort=False)["value"].transform(lambda s: s.rank(pct=True))
        ranked.index = values.index
        return ranked

    def _group_cs_zscore(self, values: pd.Series, group_col: str) -> pd.Series:
        if group_col not in self.panel.columns:
            return pd.Series(np.nan, index=self.panel.index, dtype=float)
        temp = pd.DataFrame(
            {
                "date_": self.panel["date_"],
                "group": self.panel[group_col].astype(str),
                "value": pd.to_numeric(values, errors="coerce"),
            }
        )

        def _z(group: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(group, errors="coerce")
            std = float(numeric.std(ddof=0))
            if not np.isfinite(std) or std == 0.0:
                return pd.Series(0.0, index=group.index)
            return (numeric - float(numeric.mean())) / std

        zscores = temp.groupby(["date_", "group"], sort=False)["value"].transform(_z)
        zscores.index = values.index
        return zscores

    def _build_trading_base(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_product([self.trading_dates, self.codes], names=["date_", "code"])
        base = index.to_frame(index=False)
        base["date_"] = normalize_dates(base["date_"])
        base["code"] = base["code"].astype(str)
        base["date"] = base["date_"]
        base["symbol"] = base["code"]
        return base

    def _fetch_aligned_columns(
        self,
        *,
        name: str,
        freq: str,
        value_columns: Iterable[str],
    ) -> dict[str, pd.DataFrame]:
        fundamentals = self.library.fetch_fundamental_data(
            provider="simfin",
            name=name,
            freq=freq,
            variant=self.variant,
            codes=self.codes,
            start=str(self.buffer_start.date()),
            end=str(self.end.date()),
            as_ex=self.as_ex,
        )
        return self.library.align_fundamental_to_trading_dates(
            fundamentals,
            self.trading_dates,
            value_columns=list(value_columns),
            use_point_in_time=True,
            availability_column="publish_date",
            lag_sessions=1,
        )

    def _merge_wide_columns(
        self,
        base: pd.DataFrame,
        aligned: dict[str, pd.DataFrame],
        columns: Iterable[str],
    ) -> pd.DataFrame:
        merged = base
        for column in columns:
            wide = aligned.get(column)
            if wide is None or wide.empty:
                continue
            long_df = to_long_series(wide.reindex(columns=self.codes), column)
            merged = merged.merge(long_df, on=["date_", "code"], how="left")
        return merged

    def _build_panel(self) -> pd.DataFrame:
        base = self._build_trading_base()
        statement_groups = [
            ("balance_sheet", self.statement_freq, BALANCE_COLUMNS),
            ("income_statement", self.statement_freq, INCOME_COLUMNS),
            ("cashflow_statement", self.statement_freq, CASHFLOW_COLUMNS),
        ]
        if self.include_derived_dataset:
            statement_groups.append(("derived_figures_ratios", "1y", DERIVED_COLUMNS))

        panel = base
        for dataset_name, freq, columns in statement_groups:
            aligned = self._fetch_aligned_columns(name=dataset_name, freq=freq, value_columns=columns)
            panel = self._merge_wide_columns(panel, aligned, columns)

        companies = self.library.fetch_fundamental_data(
            provider=self.classification_provider,
            name=self.classification_dataset,
            freq="static",
            variant=None,
            codes=self.codes,
            start=str(self.buffer_start.date()),
            end=str(self.end.date()),
            as_ex=self.as_ex,
        )
        if not companies.empty and "code" in companies.columns:
            categories = ["sector", "industry", "subindustry"]
            aligned_categories = self.library.align_entity_to_trading_dates(
                companies,
                self.trading_dates,
                value_columns=categories,
            )
            for column in categories:
                wide = aligned_categories.get(column)
                if wide is None or wide.empty:
                    continue
                panel = panel.merge(to_long_series(wide.reindex(columns=self.codes), column), on=["date_", "code"], how="left")

        panel = panel.sort_values(["code", "date_"]).reset_index(drop=True)
        panel["date_"] = normalize_dates(panel["date_"])
        panel["date"] = panel["date_"]

        # If SimFin returns more numeric columns, keep them available for the
        # automatic base-metric registry without hard-coding every one.
        for column in panel.columns:
            if column in META_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(panel[column]):
                panel[column] = pd.to_numeric(panel[column], errors="coerce")
        return panel

    def _build_price_panel(self) -> pd.DataFrame:
        price_df = self.library.fetch_price_data(
            codes=self.codes,
            start=str(self.buffer_start.date()),
            end=str(self.end.date()),
            provider=self.price_provider,
            as_ex=self.as_ex,
        )
        if price_df.empty:
            return pd.DataFrame(columns=["date_", "code"])

        frame = price_df.copy()
        if "date_" in frame.columns:
            frame["date_"] = normalize_dates(frame["date_"])
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str)
        keep_columns = [
            column
            for column in [
                "date_",
                "code",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "dividend",
                "shares_outstanding",
                "market_value",
            ]
            if column in frame.columns
        ]
        frame = frame[keep_columns].copy()
        frame = frame.sort_values(["code", "date_"]).reset_index(drop=True)
        return frame

    @property
    def price_panel_frame(self) -> pd.DataFrame:
        if self._price_panel is None:
            self._price_panel = self._build_price_panel()
        return self._price_panel

    def _build_valuation_panel(self) -> pd.DataFrame:
        panel = self.panel.copy()
        price = self.price_panel_frame
        if price.empty:
            return panel

        merged = panel.merge(price, on=["date_", "code"], how="left", suffixes=("", "_price"))
        merged["shares_for_valuation"] = merged.get("shares_outstanding")
        if "shares_basic" in merged.columns:
            merged["shares_for_valuation"] = merged["shares_for_valuation"].fillna(merged["shares_basic"])
        if "close" in merged.columns:
            merged["market_cap"] = self._safe_mul(merged["close"], merged["shares_for_valuation"])
        else:
            merged["market_cap"] = np.nan
        merged["book_value"] = pd.to_numeric(merged.get("total_equity"), errors="coerce")
        merged["sales"] = pd.to_numeric(merged.get("revenue"), errors="coerce")
        earnings = pd.to_numeric(merged.get("net_income_common"), errors="coerce")
        if "net_income" in merged.columns:
            earnings = earnings.fillna(pd.to_numeric(merged["net_income"], errors="coerce"))
        merged["earnings"] = earnings
        merged["cfo"] = pd.to_numeric(merged.get("net_cfo"), errors="coerce")
        merged["fcf"] = pd.to_numeric(merged.get("fcf"), errors="coerce")
        merged["ebitda"] = pd.to_numeric(merged.get("ebitda"), errors="coerce")
        merged["enterprise_value"] = self._safe_add(
            merged["market_cap"],
            pd.to_numeric(merged.get("total_debt"), errors="coerce"),
            -pd.to_numeric(merged.get("cash_st_investments"), errors="coerce"),
        )
        merged["price_to_book"] = self._safe_div(merged["market_cap"], merged["book_value"])
        merged["price_to_sales"] = self._safe_div(merged["market_cap"], merged["sales"])
        merged["price_to_earnings"] = self._safe_div(merged["market_cap"], merged["earnings"])
        merged["price_to_cashflow"] = self._safe_div(merged["market_cap"], merged["cfo"])
        merged["price_to_fcf"] = self._safe_div(merged["market_cap"], merged["fcf"])
        merged["price_to_ebitda"] = self._safe_div(merged["market_cap"], merged["ebitda"])
        merged["ev_to_sales"] = self._safe_div(merged["enterprise_value"], merged["sales"])
        merged["ev_to_earnings"] = self._safe_div(merged["enterprise_value"], merged["earnings"])
        merged["ev_to_cashflow"] = self._safe_div(merged["enterprise_value"], merged["cfo"])
        merged["ev_to_fcf"] = self._safe_div(merged["enterprise_value"], merged["fcf"])
        merged["ev_to_ebitda"] = self._safe_div(merged["enterprise_value"], merged["ebitda"])
        merged["earnings_yield"] = self._safe_div(merged["earnings"], merged["market_cap"])
        merged["sales_yield"] = self._safe_div(merged["sales"], merged["market_cap"])
        merged["cashflow_yield"] = self._safe_div(merged["cfo"], merged["market_cap"])
        merged["fcf_yield"] = self._safe_div(merged["fcf"], merged["market_cap"])
        dividends_paid = (
            pd.to_numeric(merged["dividends_paid"], errors="coerce")
            if "dividends_paid" in merged.columns
            else pd.Series(np.nan, index=merged.index, dtype=float)
        )
        merged["dividend_yield"] = self._safe_div(dividends_paid.abs(), merged["market_cap"])
        merged["book_to_market"] = self._safe_div(merged["book_value"], merged["market_cap"])
        merged["sales_to_market"] = self._safe_div(merged["sales"], merged["market_cap"])
        merged["cash_to_market"] = self._safe_div(pd.to_numeric(merged.get("cash_st_investments"), errors="coerce"), merged["market_cap"])
        merged["debt_to_market"] = self._safe_div(pd.to_numeric(merged.get("total_debt"), errors="coerce"), merged["market_cap"])
        merged["equity_to_market"] = self._safe_div(pd.to_numeric(merged.get("total_equity"), errors="coerce"), merged["market_cap"])
        merged["market_cap_per_share"] = self._safe_div(merged["market_cap"], merged["shares_for_valuation"])
        merged["book_value_per_share"] = self._safe_div(merged["book_value"], merged["shares_for_valuation"])
        merged["earnings_per_share"] = self._safe_div(merged["earnings"], merged["shares_for_valuation"])
        merged["cashflow_per_share"] = self._safe_div(merged["cfo"], merged["shares_for_valuation"])
        merged["fcf_per_share"] = self._safe_div(merged["fcf"], merged["shares_for_valuation"])
        merged["dividends_per_share"] = self._safe_div(self._series("dividends_paid").abs().reindex(merged.index), merged["shares_for_valuation"])
        merged["price_to_book_share"] = self._safe_div(merged["market_cap_per_share"], merged["book_value_per_share"])
        merged["price_to_sales_share"] = self._safe_div(merged["market_cap_per_share"], self._safe_div(merged["sales"], merged["shares_for_valuation"]))
        merged["price_to_earnings_share"] = self._safe_div(merged["market_cap_per_share"], merged["earnings_per_share"])
        merged["price_to_cashflow_share"] = self._safe_div(merged["market_cap_per_share"], merged["cashflow_per_share"])
        merged["price_to_fcf_share"] = self._safe_div(merged["market_cap_per_share"], merged["fcf_per_share"])

        merged = merged.sort_values(["code", "date_"]).reset_index(drop=True)
        return merged

    @property
    def valuation_panel(self) -> pd.DataFrame:
        if self._valuation_panel is None:
            self._valuation_panel = self._build_valuation_panel()
        return self._valuation_panel

    def source_profile(self) -> dict[str, object]:
        """Describe the SimFin input contract used by this engine.

        Periodic inputs:
        - balance / income / cashflow statements at the engine's statement_freq
        - annual-only derived figures for the annual engine
        """
        return {
            "alignment_inputs": {
                "classification_dataset": self.classification_dataset,
                "classification_provider": self.classification_provider,
                "price_provider": self.price_provider,
                "trading_calendar": "price-derived",
                "financial_availability_column": "publish_date",
                "financial_lag_sessions": 1,
            },
            "periodic_inputs": {
                "statement_freq": self.statement_freq,
                "balance_sheet": self.statement_freq,
                "income_statement": self.statement_freq,
                "cashflow_statement": self.statement_freq,
                "derived_figures_ratios": "1y" if self.include_derived_dataset else None,
            },
            "price_inputs": {
                "close_field": "close",
                "shares_field": "shares_outstanding",
                "valuation_alignment": "same_trading_day_after_financial_lag",
            },
            "period_label": self.output_tag,
            "variant": self.variant_name,
        }

    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = self._build_panel()
        return self._panel

    def _base_metric_builders(self) -> OrderedDict[str, pd.Series]:
        panel = self.panel
        metrics: OrderedDict[str, pd.Series] = OrderedDict()

        # User-requested core financial indicators.
        metrics["total_debt"] = self._safe_add(self._series("short_term_debt"), self._series("long_term_debt"))
        metrics["working_capital"] = self._safe_sub(self._series("total_current_assets"), self._series("total_current_liabilities"))
        metrics["current_ratio"] = self._safe_div(self._series("total_current_assets"), self._series("total_current_liabilities"))
        metrics["quick_ratio"] = self._safe_div(
            self._safe_add(self._series("cash_st_investments"), self._series("accounts_notes_receivable")),
            self._series("total_current_liabilities"),
        )
        metrics["cash_ratio"] = self._safe_div(self._series("cash_st_investments"), self._series("total_current_liabilities"))
        metrics["debt_ratio"] = self._safe_div(self._series("total_liabilities"), self._series("total_assets"))
        metrics["liabilities_to_equity_ratio"] = self._safe_div(self._series("total_liabilities"), self._series("total_equity"))
        metrics["debt_to_equity"] = self._safe_div(self._series("total_liabilities"), self._series("total_equity"))
        metrics["equity_multiplier"] = self._safe_div(self._series("total_assets"), self._series("total_equity"))
        metrics["equity_to_assets"] = self._safe_div(self._series("total_equity"), self._series("total_assets"))
        metrics["cash_to_assets"] = self._safe_div(self._series("cash_st_investments"), self._series("total_assets"))
        metrics["cash_to_equity"] = self._safe_div(self._series("cash_st_investments"), self._series("total_equity"))
        metrics["cash_to_debt"] = self._safe_div(self._series("cash_st_investments"), metrics["total_debt"])
        metrics["asset_turnover"] = self._safe_div(self._series("revenue"), self._series("total_assets"))
        metrics["operating_margin"] = self._safe_div(self._series("operating_income"), self._series("revenue"))
        gross_profit = self._series("gross_profit")
        if gross_profit.isna().all() and "cost_of_revenue" in panel.columns:
            gross_profit = self._safe_sub(self._series("revenue"), self._series("cost_of_revenue"))
        metrics["gross_profit_margin"] = self._safe_div(gross_profit, self._series("revenue"))
        metrics["net_profit_margin"] = self._safe_div(self._series("net_income"), self._series("revenue"))
        metrics["roe"] = self._safe_div(self._series("net_income"), self._series("total_equity"))
        metrics["roa"] = self._safe_div(self._series("net_income"), self._series("total_assets"))
        metrics["ebitda"] = self._safe_add(self._series("operating_income"), self._series("da"))
        metrics["ebitda_margin"] = self._safe_div(metrics["ebitda"], self._series("revenue"))
        metrics["fcf"] = self._safe_add(self._series("net_cfo"), self._series("net_cfi"))
        metrics["fcf_margin"] = self._safe_div(metrics["fcf"], self._series("revenue"))
        metrics["fcf_net_income"] = self._safe_div(metrics["fcf"], self._series("net_income").abs())
        metrics["fcf_to_assets"] = self._safe_div(metrics["fcf"], self._series("total_assets"))
        metrics["fcf_to_equity"] = self._safe_div(metrics["fcf"], self._series("total_equity"))
        metrics["basic_eps"] = self._safe_div(self._series("net_income_common").fillna(self._series("net_income")), self._series("shares_basic"))
        metrics["diluted_eps"] = self._safe_div(self._series("net_income_common").fillna(self._series("net_income")), self._series("shares_diluted"))
        metrics["sales_per_share"] = self._safe_div(self._series("revenue"), self._series("shares_basic"))
        metrics["equity_per_share"] = self._safe_div(self._series("total_equity"), self._series("shares_basic"))
        metrics["fcf_per_share"] = self._safe_div(metrics["fcf"], self._series("shares_basic"))
        metrics["dividends_per_share"] = self._safe_div(self._series("dividends_paid").abs(), self._series("shares_basic"))
        metrics["retained_earnings_to_equity"] = self._safe_div(self._series("retained_earnings"), self._series("total_equity"))
        metrics["cfo_to_assets"] = self._safe_div(self._series("net_cfo"), self._series("total_assets"))
        metrics["cfo_to_income"] = self._safe_div(self._series("net_cfo"), self._series("net_income").abs())
        metrics["accrual_ratio"] = self._safe_div(self._safe_sub(self._series("net_income"), self._series("net_cfo")), self._series("total_assets"))
        metrics["bank_deposit_ratio"] = self._safe_div(self._series("total_deposits"), self._series("total_assets"))
        metrics["bank_loan_ratio"] = self._safe_div(self._series("net_loans"), self._series("total_assets"))
        metrics["bank_interbank_ratio"] = self._safe_div(self._series("interbank_assets"), self._series("total_assets"))
        metrics["bank_liquidity_gap"] = self._safe_sub(
            self._safe_add(self._series("cash_st_investments"), self._series("interbank_assets")),
            self._series("total_deposits"),
        )
        metrics["retained_earnings_growth"] = self._pct_change(self._series("retained_earnings"), self._statement_lag)
        metrics["asset_growth"] = self._pct_change(self._series("total_assets"), self._statement_lag)
        metrics["equity_growth"] = self._pct_change(self._series("total_equity"), self._statement_lag)
        metrics["revenue_growth"] = self._pct_change(self._series("revenue"), self._statement_lag)
        metrics["net_income_growth"] = self._pct_change(self._series("net_income"), self._statement_lag)
        metrics["cfo_growth"] = self._pct_change(self._series("net_cfo"), self._statement_lag)
        metrics["debt_growth"] = self._pct_change(metrics["total_debt"], self._statement_lag)
        metrics["liabilities_growth"] = self._pct_change(self._series("total_liabilities"), self._statement_lag)
        metrics["profitability_spread"] = self._safe_sub(metrics["roe"], metrics["roa"])
        metrics["debt_service_coverage"] = self._safe_div(self._series("net_cfo"), metrics["total_debt"])

        # Built from the aligned panel, so it works across quarterly / annual / TTM.
        metrics["gross_margin_change"] = self._safe_sub(
            metrics["gross_profit_margin"],
            self._delay(metrics["gross_profit_margin"], self._statement_lag),
        )
        metrics["operating_margin_change"] = self._safe_sub(
            metrics["operating_margin"],
            self._delay(metrics["operating_margin"], self._statement_lag),
        )
        metrics["current_ratio_change"] = self._safe_sub(
            metrics["current_ratio"],
            self._delay(metrics["current_ratio"], self._statement_lag),
        )
        metrics["debt_ratio_change"] = self._safe_sub(
            metrics["debt_ratio"],
            self._delay(metrics["debt_ratio"], self._statement_lag),
        )

        metrics["piotroski_f_score"] = self._piotroski_score()
        return metrics

    @property
    def _statement_lag(self) -> int:
        return 63 if self.statement_freq == "1q" else 252

    def _delay(self, values: pd.Series, periods: int) -> pd.Series:
        return po.delay(self.panel, values, periods)

    def _pct_change(self, values: pd.Series, periods: int) -> pd.Series:
        return po.ts_pctchange(self.panel, values, periods)

    def _piotroski_score(self) -> pd.Series:
        lag = self._statement_lag
        score = pd.Series(0.0, index=self.panel.index, dtype=float)

        conditions = [
            self._series("net_income") > 0,
            self._series("net_cfo") > 0,
            self._safe_sub(self._series("roa"), self._delay(self._series("roa"), lag)) > 0,
            self._series("net_cfo") > self._series("net_income"),
            self._safe_sub(self._delay(self._series("debt_ratio"), lag), self._series("debt_ratio")) > 0,
            self._safe_sub(self._series("current_ratio"), self._delay(self._series("current_ratio"), lag)) > 0,
            self._safe_sub(self._delay(self._series("shares_basic"), lag), self._series("shares_basic")) >= 0,
            self._safe_sub(self._series("gross_profit_margin"), self._delay(self._series("gross_profit_margin"), lag)) > 0,
            self._safe_sub(self._series("asset_turnover"), self._delay(self._series("asset_turnover"), lag)) > 0,
        ]

        for cond in conditions:
            score = score.add(cond.fillna(False).astype(float), fill_value=0.0)
        return score.clip(lower=0.0, upper=9.0)

    def _make_group_transform(self, group_col: str, kind: str) -> Callable[[pd.Series], pd.Series]:
        if group_col not in self.panel.columns:
            return lambda values: pd.Series(np.nan, index=values.index, dtype=float)

        def _rank(values: pd.Series) -> pd.Series:
            return self._group_cs_rank(values, group_col)

        def _zscore(values: pd.Series) -> pd.Series:
            return self._group_cs_zscore(values, group_col)

        return _rank if kind == "rank" else _zscore

    def _transform_registry(self) -> OrderedDict[str, Callable[[pd.Series], pd.Series]]:
        registry: OrderedDict[str, Callable[[pd.Series], pd.Series]] = OrderedDict()
        registry["raw"] = lambda values: pd.to_numeric(values, errors="coerce")
        registry["lag1"] = lambda values: po.delay(self.panel, values, 1)
        registry["lag2"] = lambda values: po.delay(self.panel, values, 2)
        registry["delta1"] = lambda values: po.ts_delta(self.panel, values, 1)
        registry["pct1"] = lambda values: po.ts_pctchange(self.panel, values, 1)
        registry["ma3"] = lambda values: po.ts_mean(self.panel, values, 3)
        registry["ma6"] = lambda values: po.ts_mean(self.panel, values, 6)
        registry["std6"] = lambda values: po.ts_std(self.panel, values, 6)
        registry["z12"] = self._ts_zscore12
        registry["cs_rank"] = lambda values: po.cs_rank(self.panel, values, date_col="date_", pct=True)
        registry["cs_zscore"] = lambda values: po.cs_standardize(self.panel, values, date_col="date_", eps=1e-8)
        if "sector" in self.panel.columns:
            registry["sector_rank"] = self._make_group_transform("sector", "rank")
            registry["sector_zscore"] = self._make_group_transform("sector", "zscore")
        if "industry" in self.panel.columns:
            registry["industry_rank"] = self._make_group_transform("industry", "rank")
            registry["industry_zscore"] = self._make_group_transform("industry", "zscore")
        if "subindustry" in self.panel.columns:
            registry["subindustry_rank"] = self._make_group_transform("subindustry", "rank")
            registry["subindustry_zscore"] = self._make_group_transform("subindustry", "zscore")
        return registry

    def _ts_zscore12(self, values: pd.Series) -> pd.Series:
        mean = po.ts_mean(self.panel, values, 12)
        std = po.ts_std(self.panel, values, 12).replace(0, np.nan)
        z = (pd.to_numeric(values, errors="coerce") - mean) / std
        return z.replace([np.inf, -np.inf], np.nan)

    def _metric_registry_from_panel(
        self,
        *,
        panel: pd.DataFrame,
        explicit_builders: OrderedDict[str, pd.Series],
        cache: OrderedDict[str, pd.Series] | None,
    ) -> OrderedDict[str, pd.Series]:
        if cache is not None:
            return cache

        registry: OrderedDict[str, pd.Series] = OrderedDict()
        for name, series in explicit_builders.items():
            registry[name] = pd.to_numeric(series, errors="coerce")

        numeric_columns = [
            column
            for column in panel.columns
            if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(panel[column])
        ]
        for column in sorted(dict.fromkeys(numeric_columns)):
            if column in registry:
                continue
            registry[column] = pd.to_numeric(panel[column], errors="coerce")
        return registry

    def _base_metric_registry(self) -> OrderedDict[str, pd.Series]:
        if self._base_metric_map is None:
            self._base_metric_map = self._metric_registry_from_panel(
                panel=self.panel,
                explicit_builders=self._base_metric_builders(),
                cache=None,
            )
        return self._base_metric_map

    def _valuation_metric_builders(self) -> OrderedDict[str, pd.Series]:
        panel = self.valuation_panel
        metrics: OrderedDict[str, pd.Series] = OrderedDict()

        shares = self._frame_series(panel, "shares_for_valuation")
        if shares.isna().all() and "shares_basic" in panel.columns:
            shares = pd.to_numeric(panel["shares_basic"], errors="coerce")
        close = self._frame_series(panel, "close")
        market_cap = self._safe_mul(close, shares)
        metrics["close_price"] = close
        metrics["shares_for_valuation"] = shares
        metrics["market_cap"] = market_cap
        metrics["book_value"] = self._frame_series(panel, "book_value")
        metrics["sales"] = self._frame_series(panel, "sales")
        metrics["earnings"] = self._frame_series(panel, "earnings")
        metrics["cfo"] = self._frame_series(panel, "cfo")
        fcf = self._frame_series(panel, "fcf")
        if fcf.isna().all():
            fcf = self._safe_add(self._frame_series(panel, "net_cfo"), self._frame_series(panel, "net_cfi"))
        metrics["fcf"] = fcf
        ebitda = self._frame_series(panel, "ebitda")
        if ebitda.isna().all():
            ebitda = self._safe_add(self._frame_series(panel, "operating_income"), self._frame_series(panel, "da"))
        metrics["ebitda"] = ebitda
        metrics["enterprise_value"] = self._frame_series(panel, "enterprise_value")
        metrics["price_to_book"] = self._frame_series(panel, "price_to_book")
        metrics["price_to_sales"] = self._frame_series(panel, "price_to_sales")
        metrics["price_to_earnings"] = self._frame_series(panel, "price_to_earnings")
        metrics["price_to_cashflow"] = self._frame_series(panel, "price_to_cashflow")
        metrics["price_to_fcf"] = self._frame_series(panel, "price_to_fcf")
        metrics["price_to_ebitda"] = self._frame_series(panel, "price_to_ebitda")
        metrics["ev_to_sales"] = self._frame_series(panel, "ev_to_sales")
        metrics["ev_to_earnings"] = self._frame_series(panel, "ev_to_earnings")
        metrics["ev_to_cashflow"] = self._frame_series(panel, "ev_to_cashflow")
        metrics["ev_to_fcf"] = self._frame_series(panel, "ev_to_fcf")
        metrics["ev_to_ebitda"] = self._frame_series(panel, "ev_to_ebitda")
        metrics["earnings_yield"] = self._frame_series(panel, "earnings_yield")
        metrics["sales_yield"] = self._frame_series(panel, "sales_yield")
        metrics["cashflow_yield"] = self._frame_series(panel, "cashflow_yield")
        metrics["fcf_yield"] = self._frame_series(panel, "fcf_yield")
        metrics["dividend_yield"] = self._frame_series(panel, "dividend_yield")
        metrics["book_to_market"] = self._frame_series(panel, "book_to_market")
        metrics["sales_to_market"] = self._frame_series(panel, "sales_to_market")
        metrics["cash_to_market"] = self._frame_series(panel, "cash_to_market")
        metrics["debt_to_market"] = self._frame_series(panel, "debt_to_market")
        metrics["equity_to_market"] = self._frame_series(panel, "equity_to_market")
        metrics["market_cap_per_share"] = self._frame_series(panel, "market_cap_per_share")
        metrics["book_value_per_share"] = self._frame_series(panel, "book_value_per_share")
        metrics["earnings_per_share"] = self._frame_series(panel, "earnings_per_share")
        metrics["cashflow_per_share"] = self._frame_series(panel, "cashflow_per_share")
        metrics["fcf_per_share"] = self._frame_series(panel, "fcf_per_share")
        metrics["dividends_per_share"] = self._frame_series(panel, "dividends_per_share")
        metrics["price_to_book_share"] = self._frame_series(panel, "price_to_book_share")
        metrics["price_to_sales_share"] = self._frame_series(panel, "price_to_sales_share")
        metrics["price_to_earnings_share"] = self._frame_series(panel, "price_to_earnings_share")
        metrics["price_to_cashflow_share"] = self._frame_series(panel, "price_to_cashflow_share")
        metrics["price_to_fcf_share"] = self._frame_series(panel, "price_to_fcf_share")
        return metrics

    def _valuation_metric_registry(self) -> OrderedDict[str, pd.Series]:
        if self._valuation_metric_map is None:
            self._valuation_metric_map = self._metric_registry_from_panel(
                panel=self.valuation_panel,
                explicit_builders=self._valuation_metric_builders(),
                cache=None,
            )
        return self._valuation_metric_map

    def _available_factor_names_from_registry(
        self,
        registry: OrderedDict[str, pd.Series],
        max_factors: int | None = None,
    ) -> list[str]:
        limit = None if max_factors is None else int(max_factors)
        if limit is None:
            limit = self.max_factors
        names: list[str] = []
        transforms = self._transform_registry()
        for metric_name in registry.keys():
            for transform_name in transforms.keys():
                names.append(f"{metric_name}__{transform_name}")
                if limit is not None and len(names) >= limit:
                    return names
        return names

    def available_factor_names(self, max_factors: int | None = None) -> list[str]:
        return self._available_factor_names_from_registry(self._base_metric_registry(), max_factors=max_factors)

    def available_valuation_factor_names(self, max_factors: int | None = None) -> list[str]:
        return self._available_factor_names_from_registry(self._valuation_metric_registry(), max_factors=max_factors)

    def compute_factor(self, name: str) -> pd.Series:
        return self._compute_factor_from_registry(name, self._base_metric_registry())

    def compute_valuation_factor(self, name: str) -> pd.Series:
        return self._compute_factor_from_registry(name, self._valuation_metric_registry())

    def _compute_factor_from_registry(self, name: str, registry: OrderedDict[str, pd.Series]) -> pd.Series:
        if "__" not in name:
            raise KeyError(f"Unsupported financial factor name: {name}")
        metric_name, transform_name = name.split("__", 1)
        transforms = self._transform_registry()
        if metric_name not in registry:
            raise KeyError(f"Unknown base metric: {metric_name}")
        if transform_name not in transforms:
            raise KeyError(f"Unknown transform: {transform_name}")
        return transforms[transform_name](registry[metric_name]).rename(name)

    def compute_factor_frame(self, max_factors: int | None = None) -> pd.DataFrame:
        frame = self.panel[["date_", "code"]].copy()
        factor_names = self.available_factor_names(max_factors=max_factors)
        factor_values = {name: self.compute_factor(name) for name in factor_names}
        if factor_values:
            factor_series = [series.rename(name) for name, series in factor_values.items()]
            frame = pd.concat([frame] + factor_series, axis=1)
        if self.monthly_output:
            frame = self._downsample_to_month_end(frame)
        return frame.reset_index(drop=True)

    def compute_valuation_factor_frame(self, max_factors: int | None = None) -> pd.DataFrame:
        frame = self.valuation_panel[["date_", "code"]].copy()
        factor_names = self.available_valuation_factor_names(max_factors=max_factors)
        factor_values = {name: self.compute_valuation_factor(name) for name in factor_names}
        if factor_values:
            factor_series = [series.rename(name) for name, series in factor_values.items()]
            frame = pd.concat([frame] + factor_series, axis=1)
        if self.monthly_output:
            frame = self._downsample_to_month_end(frame)
        return frame.reset_index(drop=True)

    def save_factor_files(
        self,
        *,
        name: str,
        factor_frame: pd.DataFrame,
    ) -> dict[str, object]:
        bundle_dir = self.output_dir / self.output_tag / self.variant_name / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        factor_columns = [column for column in factor_frame.columns if column not in {"date_", "code"}]
        data_store = FactorStore(self.output_dir)
        files: list[dict[str, object]] = []
        for idx, column in enumerate(factor_columns, start=1):
            factor_name = f"{idx:03d}_{column}"
            spec = FactorSpec(
                region=self.library.region,
                sec_type=self.library.sec_type,
                freq=self.statement_freq,
                table_name=column,
                variant=self.variant_name,
            )
            save_result = data_store.save_factor(
                spec,
                factor_frame[["date_", "code", column]].rename(columns={column: "value"}),
                metadata={
                    "bundle_name": name,
                    "bundle_family": "financial",
                    "factor_index": idx,
                    "factor_name": column,
                    "statement_freq": self.statement_freq,
                    "variant": self.variant_name,
                    "source_profile": self.source_profile(),
                },
            )
            files.append(
                {
                    "index": idx,
                    "factor": column,
                    "parquet_path": str(save_result.files[0]),
                    "manifest_path": str(save_result.manifest_path),
                    "dataset_dir": str(save_result.dataset_dir),
                    "rows": int(len(factor_frame)),
                }
            )
        payload: dict[str, object] = {
            "name": name,
            "statement_freq": self.statement_freq,
            "variant": self.variant_name,
            "rows": int(len(factor_frame)),
            "factor_count": int(len(factor_columns)),
            "codes": int(factor_frame["code"].nunique()) if "code" in factor_frame.columns else 0,
            "date_min": str(factor_frame["date_"].min()) if not factor_frame.empty else None,
            "date_max": str(factor_frame["date_"].max()) if not factor_frame.empty else None,
            "files": files,
            "source_profile": self.source_profile(),
        }
        manifest_path = bundle_dir / f"{name}_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["manifest_path"] = str(manifest_path)
        return payload

    def save_valuation_factor_files(
        self,
        *,
        name: str,
        factor_frame: pd.DataFrame,
    ) -> dict[str, object]:
        payload = self.save_factor_files(name=name, factor_frame=factor_frame)
        payload["family"] = "valuation"
        return payload

    def _downsample_to_month_end(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        if "date_" not in frame.columns or "code" not in frame.columns:
            return frame

        work = frame.copy()
        work["month_"] = pd.to_datetime(work["date_"]).dt.to_period("M")
        work = work.sort_values(["code", "date_"])
        idx = work.groupby(["code", "month_"], sort=False)["date_"].idxmax()
        work = work.loc[idx].drop(columns=["month_"])
        return work.sort_values(["date_", "code"]).reset_index(drop=True)

    def save_bundle(
        self,
        *,
        name: str,
        factor_frame: pd.DataFrame,
    ) -> FinancialFactorBundleResult:
        payload = self.save_factor_files(name=name, factor_frame=factor_frame)
        bundle_dir = self.output_dir / self.output_tag / self.variant_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = bundle_dir / f"{name}.parquet"
        metadata_path = bundle_dir / f"{name}.json"
        factor_frame.to_parquet(parquet_path, index=False)
        bundle_payload: dict[str, object] = {
            "name": name,
            "statement_freq": self.statement_freq,
            "variant": self.variant_name,
            "rows": int(len(factor_frame)),
            "factors": int(max(0, len(factor_frame.columns) - 2)),
            "factor_names": [column for column in factor_frame.columns if column not in {"date_", "code"}][:20],
            "factor_count": int(max(0, len(factor_frame.columns) - 2)),
            "codes": int(factor_frame["code"].nunique()) if "code" in factor_frame.columns else 0,
            "date_min": str(factor_frame["date_"].min()) if not factor_frame.empty else None,
            "date_max": str(factor_frame["date_"].max()) if not factor_frame.empty else None,
            "source_profile": self.source_profile(),
        }
        bundle_payload["canonical_factor_manifest"] = payload.get("manifest_path")
        bundle_payload["canonical_factor_files"] = payload.get("files")
        metadata_path.write_text(json.dumps(bundle_payload, indent=2, default=str), encoding="utf-8")
        return FinancialFactorBundleResult(
            name=name,
            data=factor_frame,
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            payload=bundle_payload,
        )

    def save_valuation_bundle(
        self,
        *,
        name: str,
        factor_frame: pd.DataFrame,
    ) -> FinancialFactorBundleResult:
        payload = self.save_valuation_factor_files(name=name, factor_frame=factor_frame)
        bundle_dir = self.output_dir / self.output_tag / self.variant_name / "valuation"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = bundle_dir / f"{name}.parquet"
        metadata_path = bundle_dir / f"{name}.json"
        factor_frame.to_parquet(parquet_path, index=False)
        bundle_payload: dict[str, object] = {
            "name": name,
            "family": "valuation",
            "statement_freq": self.statement_freq,
            "variant": self.variant_name,
            "rows": int(len(factor_frame)),
            "factors": int(max(0, len(factor_frame.columns) - 2)),
            "factor_names": [column for column in factor_frame.columns if column not in {"date_", "code"}][:20],
            "factor_count": int(max(0, len(factor_frame.columns) - 2)),
            "codes": int(factor_frame["code"].nunique()) if "code" in factor_frame.columns else 0,
            "date_min": str(factor_frame["date_"].min()) if not factor_frame.empty else None,
            "date_max": str(factor_frame["date_"].max()) if not factor_frame.empty else None,
            "source_profile": self.source_profile(),
        }
        bundle_payload["canonical_factor_manifest"] = payload.get("manifest_path")
        bundle_payload["canonical_factor_files"] = payload.get("files")
        metadata_path.write_text(json.dumps(bundle_payload, indent=2, default=str), encoding="utf-8")
        return FinancialFactorBundleResult(
            name=name,
            data=factor_frame,
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            payload=bundle_payload,
        )


class QuarterlyFinancialFactorEngine(FinancialFactorEngine):
    statement_freq = "1q"
    output_tag = "quarterly"


class AnnualFinancialFactorEngine(FinancialFactorEngine):
    statement_freq = "1y"
    output_tag = "annual"
    include_derived_dataset = True
    piotroski_lag_sessions = 252


class TTMFinancialFactorEngine(FinancialFactorEngine):
    statement_freq = "ttm"
    output_tag = "ttm"
    piotroski_lag_sessions = 252


__all__ = [
    "FinancialFactorBundleResult",
    "FinancialFactorEngine",
    "QuarterlyFinancialFactorEngine",
    "AnnualFinancialFactorEngine",
    "TTMFinancialFactorEngine",
]
