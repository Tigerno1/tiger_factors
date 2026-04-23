from __future__ import annotations

import pandas as pd

from tiger_factors.examples.practical_factor_10y_eval import (
    build_adjusted_panel,
    evaluate_practical_factors,
    select_universe,
)
from tiger_factors.factor_evaluation import create_native_full_tear_sheet, monthly_returns_heatmap


def _sample_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    for code, base in [("AAA", 100.0), ("BBB", 80.0), ("CCC", 120.0)]:
        for idx, date in enumerate(dates):
            close = base + 0.25 * idx
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "open": close - 0.3,
                    "high": close + 0.8,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + idx * 1_000,
                    "amount": (1_000_000 + idx * 1_000) * close,
                    "vwap": close - 0.05,
                    "TURN_RATE": 0.01 + 0.0002 * idx,
                    "FACTOR_ROCTTM": 0.02 + 0.0001 * idx,
                    "NET_MF_AMOUNT_V2": 5_000 + 12 * idx,
                    "FACTOR_VROC12D": 0.01 + 0.00015 * idx,
                    "REINSTATEMENT_CHG_60D": 0.015 + 0.0001 * idx,
                    "MAIN_IN_FLOW_20D_V2": 1_000 + 20 * idx,
                    "SLARGE_IN_FLOW_V2": 700 + 15 * idx,
                    "FACTOR_VOL60D": 0.02 + 0.00005 * idx,
                    "FACTOR_TVSD20D": 0.015 + 0.00003 * idx,
                    "FACTOR_CNE5_BETA": 0.8 + 0.001 * idx,
                    "FACTOR_CNE5_SIZE": 1.2 + 0.0007 * idx,
                    "MAIN_IN_FLOW_V2": 1_100 + 24 * idx,
                }
            )
    return pd.DataFrame(rows)


def test_select_universe_filters_full_coverage(tmp_path) -> None:
    csv_path = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "start_date": ["2010-01-01", "2014-01-01", "2010-01-01"],
            "end_date": ["2025-01-01", "2020-01-01", None],
        }
    ).to_csv(csv_path, index=False)

    codes = select_universe(csv_path, "2014-01-01", "2024-12-31")
    assert codes == ["AAA", "CCC"]


def test_evaluate_practical_factors_returns_a_summary_frame() -> None:
    summary = evaluate_practical_factors(_sample_panel())
    assert not summary.empty
    assert "factor_001_volume_flow_sine_skew" in set(summary["factor"])
    assert {"ic_mean", "rank_ic_mean", "sharpe", "fitness"}.issubset(summary.columns)


def test_build_adjusted_panel_merges_batches(monkeypatch) -> None:
    import tiger_factors.examples.practical_factor_10y_eval as module

    def fake_fetch_price_data(self, *, codes, start, end, provider, as_ex=None):  # noqa: ANN001
        records = []
        for code in codes:
            for date in pd.date_range("2024-01-01", periods=3, freq="D"):
                records.append(
                    {
                        "date": date,
                        "code": code,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "volume": 1.0,
                    }
                )
        return pd.DataFrame(records)

    def fake_export(raw_frame, *, code, start, end):  # noqa: ANN001
        dates = pd.date_range("2024-01-01", periods=len(raw_frame), freq="D")
        frame = pd.DataFrame(
            {
                "date_": dates,
                "code": raw_frame["code"].astype(str).iloc[0],
                "open": raw_frame["open"].to_numpy(),
                "high": raw_frame["high"].to_numpy(),
                "low": raw_frame["low"].to_numpy(),
                "close": raw_frame["close"].to_numpy(),
                "volume": raw_frame["volume"].to_numpy(),
                "vwap": 1.0,
            }
        )
        return frame, pd.DataFrame(), pd.DataFrame(), pd.DatetimeIndex([])

    monkeypatch.setattr(module.TigerFactorLibrary, "fetch_price_data", fake_fetch_price_data)
    monkeypatch.setattr(module, "build_yahoo_us_export_frame", fake_export)
    monkeypatch.setattr(module, "ensure_domain_registered", lambda *args, **kwargs: None)

    panel = build_adjusted_panel(
        ["AAA", "BBB", "CCC"],
        start="2024-01-01",
        end="2024-01-03",
        provider="yahoo",
        region="us",
        sec_type="stock",
        db_path=None,
        batch_size=2,
    )

    assert set(panel.columns) == {"date_", "code", "open", "high", "low", "close", "volume", "vwap"}
    assert len(panel) == 9


def test_create_native_full_tear_sheet_writes_charts(tmp_path) -> None:
    panel = _sample_panel()
    factor = panel.pivot(index="date_", columns="code", values="close").sort_index()
    forward_returns = factor.pct_change().shift(-1)
    benchmark_returns = pd.Series([0.01, 0.02, 0.015, 0.0, 0.01] + [0.005] * (len(factor.index) - 5), index=factor.index)
    group_labels = pd.Series({"AAA": "tech", "BBB": "tech", "CCC": "fin"})

    report = create_native_full_tear_sheet(
        "demo_factor",
        factor,
        forward_returns,
        output_dir=tmp_path / "report",
        benchmark_returns=benchmark_returns,
        group_labels=group_labels,
    )

    assert report.figure_paths
    assert report.table_paths
    assert (tmp_path / "report" / "ic_series.png").exists()
    assert (tmp_path / "report" / "quantile_returns.parquet").exists()
    assert (tmp_path / "report" / "factor_autocorr_series.parquet").exists()
    assert (tmp_path / "report" / "group_summary.parquet").exists()
    assert (tmp_path / "report" / "benchmark_regression.parquet").exists()
    assert monthly_returns_heatmap(report.long_short_returns).empty is False
