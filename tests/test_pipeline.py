from __future__ import annotations

import pandas as pd

from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_maker.pipeline import (
    Pipeline,
    PipelineEngine,
    Returns,
    RollingStdDev,
    SimFinBalanceSheet,
    SimFinIncomeStatement,
    USEquityPricing,
)


class FakeLibrary(TigerFactorLibrary):
    def __init__(self) -> None:
        super().__init__(output_dir="tiger_tmp/test_factors", verbose=False)
        self.price_provider_calls: list[str | None] = []
        self.fundamental_provider_calls: list[str] = []

    def price_panel(self, *, codes, start, end, provider=None, field="close", as_ex=None):
        self.price_provider_calls.append(provider)
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        data = {
            "AAPL": [10, 11, 12, 13, 14, 15],
            "MSFT": [20, 19, 18, 19, 20, 21],
        }
        frame = pd.DataFrame(data, index=dates)
        if field == "volume":
            frame = frame * 100
        return frame.reindex(columns=codes)

    def fetch_fundamental_data(self, *, name, freq, variant=None, start, end, codes=None, provider="simfin", as_ex=None):
        self.fundamental_provider_calls.append(provider)
        dates = pd.to_datetime(["2024-01-01", "2024-01-04"])
        rows = []
        for code in codes or ["AAPL", "MSFT"]:
            for dt in dates:
                is_late_update = dt == pd.Timestamp("2024-01-04")
                rows.append(
                    {
                        "date_": dt,
                        "available_at": dt + pd.Timedelta(days=1),
                        "code": code,
                        "total_assets": (130.0 if is_late_update else 100.0) if code == "AAPL" else (150.0 if is_late_update else 120.0),
                        "total_equity": (80.0 if is_late_update else 50.0) if code == "AAPL" else (90.0 if is_late_update else 60.0),
                        "net_income": (20.0 if is_late_update else 10.0) if code == "AAPL" else (24.0 if is_late_update else 12.0),
                    }
                )
        return pd.DataFrame(rows)


def test_pipeline_engine_runs_basic_terms():
    library = FakeLibrary()
    engine = PipelineEngine(library=library, trading_calendar="XNYS")
    pipeline = Pipeline(
        columns={
            "ret_3": Returns(window_length=3),
            "vol_3": RollingStdDev(inputs=[USEquityPricing.column("close")], window_length=3),
            "roe_like": SimFinIncomeStatement.column("net_income") / SimFinBalanceSheet.column("total_equity"),
        },
        screen=USEquityPricing.column("close").notnull(),
    )

    result = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    assert {"date_", "code", "ret_3", "vol_3", "roe_like"}.issubset(result.columns)
    assert len(result) == 8
    assert list(result["date_"].drop_duplicates()) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]


def test_pipeline_filter_top_selects_assets():
    library = FakeLibrary()
    engine = PipelineEngine(library=library, as_ex=True)
    momentum = Returns(window_length=3)
    pipeline = Pipeline(
        columns={"momentum": momentum},
        screen=momentum.top(1),
    )

    result = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    non_null = result.dropna(subset=["momentum"])
    counts = non_null.groupby("date_")["code"].nunique()
    assert (counts <= 1).all()


def test_pipeline_can_use_explicit_trading_calendar():
    library = FakeLibrary()
    engine = PipelineEngine(library=library, trading_calendar="XNYS", calendar_source="pandas", as_ex=True)
    pipeline = Pipeline(
        columns={"close": USEquityPricing.column("close")},
        screen=USEquityPricing.column("close").notnull(),
    )

    result = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    assert len(result["date_"].unique()) == 4


def test_pipeline_as_ex_does_not_infer_calendar():
    library = FakeLibrary()
    engine = PipelineEngine(library=library, calendar_source="pandas", as_ex=True)
    pipeline = Pipeline(
        columns={"close": USEquityPricing.column("close")},
        screen=USEquityPricing.column("close").notnull(),
    )

    result = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    assert engine.trading_calendar is None
    assert len(result["date_"].unique()) == 6


def test_pipeline_explicit_calendar_overrides_as_ex_default():
    library = FakeLibrary()
    engine = PipelineEngine(library=library, trading_calendar="XHKG", calendar_source="pandas", as_ex=True)

    assert engine.trading_calendar == "XHKG"


def test_pipeline_provider_overrides_are_applied():
    library = FakeLibrary()
    engine = PipelineEngine(
        library=library,
        as_ex=True,
        provider_overrides={
            "source_type:price": "simfin",
            "source_type:fundamental": "simfin",
        },
    )
    pipeline = Pipeline(
        columns={
            "close": USEquityPricing.column("close"),
            "roe_like": SimFinIncomeStatement.column("net_income") / SimFinBalanceSheet.column("total_equity"),
        },
        screen=USEquityPricing.column("close").notnull(),
    )

    _ = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    assert "simfin" in [provider for provider in library.price_provider_calls if provider is not None]
    assert "simfin" in library.fundamental_provider_calls


def test_pipeline_fundamentals_are_lagged_by_default():
    library = FakeLibrary()
    engine = PipelineEngine(library=library, trading_calendar="XNYS")
    pipeline = Pipeline(
        columns={"roe_like": SimFinIncomeStatement.column("net_income") / SimFinBalanceSheet.column("total_equity")},
        screen=USEquityPricing.column("close").notnull(),
    )

    result = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    aapl = result[result["code"] == "AAPL"].set_index("date_")["roe_like"]
    assert pd.isna(aapl.loc[pd.Timestamp("2024-01-02")])
    assert aapl.loc[pd.Timestamp("2024-01-03")] == 0.2
    assert aapl.loc[pd.Timestamp("2024-01-04")] == 0.2
    assert aapl.loc[pd.Timestamp("2024-01-05")] == 0.25


def test_pipeline_can_use_fundamental_availability_column():
    library = FakeLibrary()
    engine = PipelineEngine(
        library=library,
        fundamental_availability_column="available_at",
        fundamental_lag_sessions=0,
        as_ex=True,
    )
    pipeline = Pipeline(
        columns={"roe_like": SimFinIncomeStatement.column("net_income") / SimFinBalanceSheet.column("total_equity")},
        screen=USEquityPricing.column("close").notnull(),
    )

    result = engine.run_pipeline(
        pipeline,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-06",
    )

    aapl = result[result["code"] == "AAPL"].set_index("date_")["roe_like"]
    assert aapl.loc[pd.Timestamp("2024-01-02")] == 0.2
    assert aapl.loc[pd.Timestamp("2024-01-04")] == 0.2
    assert aapl.loc[pd.Timestamp("2024-01-05")] == 0.25
