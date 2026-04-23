from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_algorithm.traditional_factors.portfolio import TraditionalPortfolioEngine
from tiger_factors.factor_algorithm.traditional_factors.portfolio import create_crsp_predictors
from tiger_factors.factor_algorithm.traditional_factors.portfolio import load_signal_doc


def _toy_engine() -> TraditionalPortfolioEngine:
    crspinfo = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "yyyymm": [202006, 202006, 202006, 202006],
            "exchcd": [1, 1, 1, 1],
            "prc": [10.0, 20.0, 30.0, 40.0],
            "me": [100.0, 200.0, 300.0, 400.0],
            "me_nyse20": [150.0, 150.0, 150.0, 150.0],
        }
    )
    crspret = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "yyyymm": [202007, 202007, 202007, 202007],
            "date": pd.to_datetime(["2020-07-31"] * 4),
            "ret": [-0.02, -0.01, 0.03, 0.05],
            "melag": [90.0, 190.0, 290.0, 390.0],
        }
    )
    return TraditionalPortfolioEngine(crspret=crspret, crspinfo=crspinfo, signal_doc=load_signal_doc())


def _ff93_engine() -> TraditionalPortfolioEngine:
    crspinfo = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4, 5, 6],
            "yyyymm": [202006] * 6,
            "exchcd": [1] * 6,
            "shrcd": [10] * 6,
            "me": [10.0, 100.0, 20.0, 120.0, 30.0, 130.0],
            "prc": [10.0] * 6,
        }
    )
    crspret = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "yyyymm": [202006] * 6 + [202007] * 6,
            "date": pd.to_datetime(["2020-06-30"] * 6 + ["2020-07-31"] * 6),
            "ret": [0.0] * 6 + [-0.10, -0.05, 0.00, 0.01, 0.07, 0.08],
            "melag": [10.0, 100.0, 20.0, 120.0, 30.0, 130.0] * 2,
        }
    )
    return TraditionalPortfolioEngine(crspret=crspret, crspinfo=crspinfo, signal_doc=load_signal_doc())


def test_create_crsp_predictors_returns_size_price_and_short_term_reversal():
    crspinfo = pd.DataFrame(
        {
            "permno": [1],
            "yyyymm": [202001],
            "prc": [10.0],
            "me": [100.0],
        }
    )
    crspret = pd.DataFrame(
        {
            "permno": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "ret": [0.05],
        }
    )

    predictors = create_crsp_predictors(crspret, crspinfo)

    assert set(predictors) == {"Price", "STreversal", "Size"}
    assert predictors["Size"]["Size"].iat[0] == np.log(100.0)
    assert predictors["Price"]["Price"].iat[0] == np.log(10.0)
    assert predictors["STreversal"]["STreversal"].iat[0] == 0.05


def test_build_signal_portfolio_matches_upstream_monthly_long_short_logic():
    engine = _toy_engine()
    signal = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "yyyymm": [202006, 202006, 202006, 202006],
            "ToySignal": [1.0, 2.0, 3.0, 4.0],
        }
    )

    port = engine.build_signal_portfolio(
        "Accruals",
        signal_frame=signal,
        cat_form="continuous",
        q_cut=0.5,
        sweight="EW",
        sign=1.0,
        startmonth=1,
        portperiod=1,
    )

    ls = port.loc[port["port"] == "LS"].iloc[0]
    low = port.loc[port["port"] == "01"].iloc[0]
    high = port.loc[port["port"] == "02"].iloc[0]

    assert low["ret"] == np.mean([-0.02, -0.01])
    assert high["ret"] == np.mean([0.03, 0.05])
    assert ls["ret"] == high["ret"] - low["ret"]
    assert ls["Nlong"] == 2
    assert ls["Nshort"] == 2


def test_summarize_portfolios_assigns_sample_buckets_and_statistics():
    engine = _toy_engine()
    port = pd.DataFrame(
        {
            "signalname": ["Accruals", "Accruals"],
            "port": ["LS", "LS"],
            "date": pd.to_datetime(["1980-01-31", "1980-02-29"]),
            "ret": [0.02, 0.01],
            "signallag": [np.nan, np.nan],
            "Nlong": [25, 25],
            "Nshort": [25, 25],
        }
    )

    summary = engine.summarize_portfolios(port, n_stocks_min=20)

    assert set(summary["samptype"]) == {"insamp"}
    assert summary["T"].iat[0] == 2
    assert summary["rbar"].iat[0] == 0.01


def test_build_ff93_signal_portfolio_matches_two_by_three_construction():
    engine = _ff93_engine()
    signal = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4, 5, 6],
            "yyyymm": [202006] * 6,
            "Accruals": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    port = engine.build_ff93_signal_portfolio("Accruals", signal_frame=signal, sign=1.0)

    assert set(port["port"].astype(str)) == {"SL", "SM", "SH", "BL", "BM", "BH", "LS"}
    ls = port.loc[port["port"].astype(str) == "LS"].iloc[0]
    assert np.isclose(ls["ret"], 0.15)
    assert ls["Nlong"] == 2
    assert ls["Nshort"] == 2


def test_alt_and_daily_helpers_return_expected_output_shapes():
    engine = _toy_engine()
    signal = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "yyyymm": [202006, 202006, 202006, 202006],
            "Accruals": [1.0, 2.0, 3.0, 4.0],
        }
    )
    signal_frames = {"Accruals": signal}

    alt = engine.build_alternative_portfolios(signal_frames=signal_frames, quickrun=["Accruals"])
    assert "PredictorAltPorts_Deciles.csv" in alt
    assert "PredictorAltPorts_HoldPer_12.csv" in alt
    assert set(alt["PredictorAltPorts_Quintiles.csv"]["signalname"]) == {"Accruals"}

    daily_summary = engine.summarize_daily_portfolios({"Predictor": alt["PredictorAltPorts_Quintiles.csv"]})
    assert not daily_summary.empty
    assert set(daily_summary["implementation"]) == {"Predictor"}


def test_compare_daily_monthly_timing_returns_regression_stats():
    daily_long = pd.DataFrame(
        {
            "signalname": ["Accruals"] * 12,
            "port": ["LS"] * 12,
            "date": pd.date_range("2020-01-20", periods=12, freq="ME"),
            "ret": np.linspace(0.01, 0.12, 12),
        }
    )
    monthly_long = pd.DataFrame(
        {
            "signalname": ["Accruals"] * 12,
            "port": ["LS"] * 12,
            "date": pd.date_range("2020-01-31", periods=12, freq="ME"),
            "ret": np.linspace(0.01, 0.12, 12),
        }
    )

    result = TraditionalPortfolioEngine.compare_daily_monthly_timing(daily_long, monthly_long)

    assert list(result["signalname"]) == ["Accruals"]
    assert list(result["port"]) == ["LS"]
    assert {"intercept", "slope", "rsq"} <= set(result.columns)


def test_predictor_report_includes_full_ports_wide_ls_and_summaries():
    engine = _toy_engine()
    signal = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "yyyymm": [202006, 202006, 202006, 202006],
            "Accruals": [1.0, 2.0, 3.0, 4.0],
        }
    )

    report = engine.build_predictor_report(signal_frames={"Accruals": signal}, quickrun=["Accruals"])

    assert {"PredictorPortsFull", "PredictorLSretWide", "PredictorSummaryFull", "PredictorSummaryLSInSample"} <= set(report)
    assert list(report["PredictorLSretWide"].columns) == ["date", "Accruals"]
