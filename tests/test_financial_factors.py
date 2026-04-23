from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.factor_algorithm.financial_factors import (
    AnnualFinancialFactorEngine,
    QuarterlyFinancialFactorEngine,
    TTMFinancialFactorEngine,
)
from tiger_factors.factor_algorithm.financial_factors import record_financial_factors
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_algorithm.valuation_factors import ValuationFactorEngine
from tiger_factors.factor_algorithm.valuation_factors import record_valuation_factors


def _price_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    frame = pd.DataFrame(
        {
            "AAA": [100 + idx * 0.5 for idx in range(len(dates))],
            "BBB": [80 + idx * 0.4 for idx in range(len(dates))],
        },
        index=dates,
    )
    return frame


def _price_rows() -> pd.DataFrame:
    dates = pd.date_range("2023-12-25", periods=130, freq="D")
    rows = []
    for code, base in [("AAA", 100.0), ("BBB", 80.0)]:
        for idx, dt in enumerate(dates):
            close = base + idx * 0.5
            rows.append(
                {
                    "date_": dt,
                    "code": code,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "adj_close": close * 0.98,
                    "volume": 1_000_000 + idx * 1000,
                    "shares_outstanding": 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def _statement_rows(name: str, variant: str | None = None) -> pd.DataFrame:
    dates = pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30"])
    rows = []
    for code, scale in [("AAA", 1.0), ("BBB", 1.25)]:
        for idx, dt in enumerate(dates):
            step = idx + 1
            base = scale * step
            row = {
                "date_": dt,
                "publish_date": dt + pd.Timedelta(days=7),
                "report_date": dt,
                "code": code,
                "ticker": code,
                "simfin_id": 1000 + idx,
                "currency": "USD",
                "fiscal_year": 2024,
                "fiscal_period": "Q1",
            }
            if name == "companies":
                row.update(
                    {
                        "sector": "tech" if code == "AAA" else "finance",
                        "industry": "software" if code == "AAA" else "bank",
                        "subindustry": "app" if code == "AAA" else "retail_bank",
                    }
                )
            elif name == "balance_sheet":
                row.update(
                    {
                        "shares_basic": 1000.0 * base,
                        "shares_diluted": 1020.0 * base,
                        "cash_st_investments": 500.0 * base,
                        "interbank_assets": 50.0 * base,
                        "st_lt_investments": 40.0 * base,
                        "accounts_notes_receivable": 120.0 * base,
                        "net_loans": 300.0 * base,
                        "net_fixed_assets": 400.0 * base,
                        "inventories": 80.0 * base,
                        "total_current_assets": 1100.0 * base,
                        "ppe_net": 350.0 * base,
                        "lt_investments_receivables": 60.0 * base,
                        "other_lt_assets": 70.0 * base,
                        "total_noncurrent_assets": 900.0 * base,
                        "total_assets": 2000.0 * base,
                        "payables_accruals": 150.0 * base,
                        "total_deposits": 220.0 * base,
                        "short_term_debt": 100.0 * base,
                        "total_current_liabilities": 700.0 * base,
                        "long_term_debt": 200.0 * base,
                        "total_noncurrent_liabilities": 500.0 * base,
                        "total_liabilities": 1200.0 * base,
                        "preferred_equity": 10.0 * base,
                        "share_cap_apic": 300.0 * base,
                        "treasury_stock": 5.0 * base,
                        "retained_earnings": 400.0 * base,
                        "total_equity": 800.0 * base,
                        "total_liabilities_equity": 2000.0 * base,
                    }
                )
            elif name == "income_statement":
                row.update(
                    {
                        "shares_basic": 1000.0 * base,
                        "shares_diluted": 1020.0 * base,
                        "revenue": 1500.0 * base,
                        "cost_of_revenue": 600.0 * base,
                        "gross_profit": 900.0 * base,
                        "operating_income": 250.0 * base,
                        "operating_expenses": 180.0 * base,
                        "sga": 120.0 * base,
                        "rnd": 90.0 * base,
                        "pretax_income": 220.0 * base,
                        "income_tax_net": 40.0 * base,
                        "income_cont_ops": 170.0 * base,
                        "net_extraordinary": 0.0,
                        "net_income": 180.0 * base,
                        "net_income_common": 170.0 * base,
                        "da": 30.0 * base,
                        "non_operating_income": 15.0 * base,
                        "interest_expense_net": 10.0 * base,
                        "pretax_income_adj": 225.0 * base,
                        "abnormal_gains_losses": 2.0 * base,
                    }
                )
            elif name == "cashflow_statement":
                row.update(
                    {
                        "shares_basic": 1000.0 * base,
                        "shares_diluted": 1020.0 * base,
                        "net_income_starting_line": 180.0 * base,
                        "da": 30.0 * base,
                        "non_cash_items": 12.0 * base,
                        "net_cfo": 210.0 * base,
                        "dividends_paid": 20.0 * base,
                        "cash_repayment_debt": 25.0 * base,
                        "cash_repurchase_equity": 18.0 * base,
                        "net_cff": -40.0 * base,
                        "fx_effect": 1.0 * base,
                        "net_change_cash": 120.0 * base,
                        "provision_loan_losses": 5.0 * base,
                        "change_working_capital": 14.0 * base,
                        "change_fixed_assets_intangibles": 22.0 * base,
                        "net_change_loans_interbank": 12.0 * base,
                        "net_cash_acq_divest": -6.0 * base,
                        "net_cfi": -50.0 * base,
                        "net_change_investments": -12.0 * base,
                    }
                )
            elif name == "derived_figures_ratios":
                row.update(
                    {
                        "shares_basic": 1000.0 * base,
                        "shares_diluted": 1020.0 * base,
                        "ebitda": 280.0 * base,
                        "total_debt": 300.0 * base,
                        "fcf": 160.0 * base,
                        "gross_profit_margin": 0.60,
                        "operating_margin": 0.16,
                        "net_profit_margin": 0.12,
                        "roe": 0.22,
                        "roa": 0.10,
                        "fcf_net_income": 0.88,
                        "current_ratio": 1.5,
                        "liabilities_to_equity_ratio": 1.4,
                        "debt_ratio": 0.60,
                        "basic_eps": 0.2 * base,
                        "diluted_eps": 0.19 * base,
                        "sales_per_share": 1.5 * base,
                        "equity_per_share": 0.8 * base,
                        "fcf_per_share": 0.16 * base,
                        "dividends_per_share": 0.02 * base,
                        "piotroski_f_score": 7.0 - idx * 0.5,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


class FakeFinancialLibrary(TigerFactorLibrary):
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir=output_dir, verbose=False, price_provider="yahoo")
        self.calls: list[tuple[str, str, str | None]] = []

    def fetch_price_data(self, *, codes, start, end, provider=None, freq="1d", as_ex=None):
        frame = _price_rows()
        if codes is not None:
            frame = frame.loc[frame["code"].isin(codes)].copy()
        return frame

    def price_panel(self, *, codes, start, end, provider=None, field="close", as_ex=None):
        return _price_panel().reindex(columns=codes)

    def fetch_fundamental_data(self, *, name, freq, variant=None, start, end, codes=None, provider="simfin", as_ex=None):
        self.calls.append((name, freq, variant))
        return _statement_rows(name, variant=variant)


def test_quarterly_financial_engine_builds_factor_bundle(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    engine = QuarterlyFinancialFactorEngine(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-04-30",
        output_dir=tmp_path,
        monthly_output=True,
    )

    assert len(engine.available_factor_names()) >= 500
    assert engine.compute_factor("current_ratio__raw").notna().any()
    assert engine.compute_factor("roe__raw").notna().any()
    assert engine.compute_factor("piotroski_f_score__raw").notna().any()

    factor_frame = engine.compute_factor_frame(max_factors=20)
    assert {"date_", "code"}.issubset(factor_frame.columns)
    assert len(factor_frame.columns) == 22
    bundle = engine.save_bundle(name="quarterly_base_financial_factors", factor_frame=factor_frame)
    assert bundle.parquet_path.exists()
    assert bundle.metadata_path.exists()


def test_annual_engine_uses_derived_dataset(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    engine = AnnualFinancialFactorEngine(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        variant="bank",
        output_dir=tmp_path,
        monthly_output=True,
    )

    assert engine.compute_factor("gross_profit_margin__raw").notna().any()
    assert engine.compute_factor("basic_eps__raw").notna().any()
    factor_frame = engine.compute_factor_frame(max_factors=15)
    assert not factor_frame.empty
    bundle = engine.save_bundle(name="annual_bank_financial_factors", factor_frame=factor_frame)
    assert bundle.parquet_path.exists()
    assert bundle.payload["variant"] == "bank"


def test_ttm_engine_works_for_insurance_variant(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    engine = TTMFinancialFactorEngine(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        variant="insurance",
        output_dir=tmp_path,
        monthly_output=True,
    )

    factor_frame = engine.compute_factor_frame(max_factors=10)
    assert not factor_frame.empty
    assert engine.variant_name == "insurance"
    assert any(call[2] == "insurance" for call in library.calls if call[0] != "companies")


def test_valuation_factor_frame_builds_from_price_and_fundamentals(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    engine = AnnualFinancialFactorEngine(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        output_dir=tmp_path,
        monthly_output=True,
    )

    assert len(engine.available_valuation_factor_names()) >= 714
    assert engine.compute_valuation_factor("market_cap__raw").notna().any()
    assert engine.compute_valuation_factor("price_to_book__raw").notna().any()
    valuation_frame = engine.compute_valuation_factor_frame()
    assert {"date_", "code"}.issubset(valuation_frame.columns)
    assert not valuation_frame.empty
    valuation_bundle = engine.save_valuation_bundle(
        name="annual_valuation_factors",
        factor_frame=valuation_frame,
    )
    assert valuation_bundle.parquet_path.exists()
    assert valuation_bundle.metadata_path.exists()


def test_quarterly_valuation_factor_frame_works_without_derived_dataset(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    engine = QuarterlyFinancialFactorEngine(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        output_dir=tmp_path,
        monthly_output=True,
    )

    assert engine.compute_valuation_factor("price_to_sales__raw").notna().any()
    valuation_frame = engine.compute_valuation_factor_frame(max_factors=12)
    assert not valuation_frame.empty
    assert {"date_", "code"}.issubset(valuation_frame.columns)


def test_valuation_engine_module_wrapper_works(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    engine = ValuationFactorEngine(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        output_dir=tmp_path,
        monthly_output=True,
    )

    assert len(engine.available_valuation_factor_names()) >= 714
    frame = engine.compute_valuation_factor_frame()
    assert not frame.empty
    assert engine.available_valuation_factor_names()


def test_financial_factor_recorder_writes_manifests(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    runs = record_financial_factors(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        output_root=tmp_path,
        monthly_output=True,
        max_factors=12,
        variants={"base": None},
        ensure_domain=False,
    )

    assert len(runs) == 3
    assert (tmp_path / "financial_factors_manifest.json").exists()
    assert all(Path(run["parquet_path"]).exists() for run in runs)


def test_valuation_factor_recorder_writes_manifests(tmp_path):
    library = FakeFinancialLibrary(str(tmp_path))
    runs = record_valuation_factors(
        library=library,
        codes=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        output_root=tmp_path,
        monthly_output=True,
        max_factors=12,
        variants={"base": None},
        ensure_domain=False,
    )

    assert len(runs) == 3
    assert (tmp_path / "valuation_factors_manifest.json").exists()
    assert all(Path(run["parquet_path"]).exists() for run in runs)
