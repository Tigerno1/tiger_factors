from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tiger_factors.factor_maker.vectorization.indicators import Alpha101IndicatorTransformer


@dataclass
class FakeCalendar:
    trading_days: set[date]

    def sessions_in_range(self, start_day: date, end_day: date) -> pd.DatetimeIndex:
        sessions = [
            pd.Timestamp(day)
            for day in sorted(self.trading_days)
            if start_day <= day <= end_day
        ]
        return pd.DatetimeIndex(sessions)


class DummyAlpha101Engine:
    last_input: pd.DataFrame | None = None
    last_alpha_id: int | None = None

    def __init__(self, data: pd.DataFrame, **kwargs) -> None:
        self.data = data.copy()
        DummyAlpha101Engine.last_input = self.data.copy()

    def compute(self, alpha_id: int) -> pd.DataFrame:
        DummyAlpha101Engine.last_alpha_id = alpha_id
        return self.data[["date_", "code"]].assign(**{f"alpha_{alpha_id:03d}": 1.0})


def test_alpha101_indicator_pipeline_builds_input_and_factor_frame(monkeypatch, tmp_path):
    from tiger_factors.factor_maker.vectorization.indicators import alpha101_indicator as module

    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)})
    transformer = Alpha101IndicatorTransformer(calendar=calendar, start="2024-01-02", end="2024-01-04")

    monkeypatch.setattr(module, "fetch_codes", lambda **kwargs: ["aaa", "bbb"])

    def fake_fetch_data(**kwargs):
        if kwargs.get("name") == "companies":
            return pd.DataFrame(
                {
                    "code": ["AAA", "BBB"],
                    "industry_id": [1, 2],
                    "ticker": ["AAA", "BBB"],
                    "market": ["us", "us"],
                }
            )
        if kwargs.get("name") == "industry":
            return pd.DataFrame(
                {
                    "industry_id": [1, 2],
                    "industry": ["Tech Hardware", "Banks"],
                    "sector": ["Technology", "Financials"],
                }
            )
        return pd.DataFrame(
            {
                "date_": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ),
                "code": ["AAA", "AAA", "BBB", "BBB"],
                "open": [100.0, 110.0, 20.0, 21.0],
                "high": [102.0, 112.0, 22.0, 23.0],
                "low": [98.0, 108.0, 18.0, 19.0],
                "close": [100.0, 110.0, 20.0, 21.0],
                "adj_close": [50.0, 55.0, 10.0, 10.5],
                "dividend": [0.0, 0.0, 0.0, 0.0],
                "volume": [1000.0, 2000.0, 500.0, 700.0],
                "shares_outstanding": [10_000.0, 10_000.0, 5_000.0, 5_000.0],
            }
        )

    monkeypatch.setattr(module, "fetch_data", fake_fetch_data)
    monkeypatch.setattr(module, "Alpha101Engine", DummyAlpha101Engine)

    result = transformer.compute_alpha101(
        1,
        save_factors=True,
        save_adj_price=True,
        output_dir=tmp_path,
    )

    assert result.codes == ["AAA", "BBB"]
    assert list(result.calendar_frame["date_"]) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-03"),
    ]
    assert "exchange_date_" not in result.calendar_frame.columns
    assert result.adjusted_frame.loc[0, "close"] == pytest.approx(50.0)
    assert result.alpha_input_frame.loc[0, "vwap"] == pytest.approx(50.0)
    assert result.alpha_input_frame.loc[0, "market_value"] == pytest.approx(500_000.0)
    assert list(result.companies_frame["code"]) == ["AAA", "BBB"]
    assert list(result.industry_frame["industry"]) == ["Tech Hardware", "Banks"]
    assert "industry" in result.classification_frame.columns
    assert "sector" in result.classification_frame.columns
    assert "date_" not in result.classification_frame.columns
    assert result.alpha_input_frame.loc[0, "industry"] == "Tech Hardware"
    assert result.alpha_input_frame.loc[0, "sector"] == "Technology"
    assert DummyAlpha101Engine.last_alpha_id == 1
    assert result.factor_frame.columns.tolist() == ["date_", "code", "alpha_001"]
    assert result.factor_frame["alpha_001"].eq(1.0).all()
    assert result.saved_factor_paths is not None
    assert "alpha_001" in result.saved_factor_paths
    assert Path(result.saved_factor_paths["alpha_001"]).exists()
    assert Path(result.saved_factor_paths["alpha_001"]).name == "alpha_001.parquet"
    assert result.saved_adjusted_price_path is not None
    assert Path(result.saved_adjusted_price_path).exists()
    assert Path(result.saved_adjusted_price_path).name == "adj_price__fwd.parquet"


def test_alpha101_indicator_parallel_computes_and_saves_parquets(monkeypatch, tmp_path):
    from tiger_factors.factor_maker.vectorization.indicators import alpha101_indicator as module

    calendar = FakeCalendar(
        trading_days={
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 1, 10),
            date(2024, 1, 11),
            date(2024, 1, 12),
        }
    )
    transformer = Alpha101IndicatorTransformer(calendar=calendar, start="2024-01-02", end="2024-01-12")

    monkeypatch.setattr(module, "fetch_codes", lambda **kwargs: ["aaa", "bbb"])

    def fake_fetch_data(**kwargs):
        if kwargs.get("name") == "companies":
            return pd.DataFrame(
                {
                    "code": ["AAA", "BBB"],
                    "industry_id": [1, 2],
                    "ticker": ["AAA", "BBB"],
                    "market": ["us", "us"],
                }
            )
        if kwargs.get("name") == "industry":
            return pd.DataFrame(
                {
                    "industry_id": [1, 2],
                    "industry": ["Tech Hardware", "Banks"],
                    "sector": ["Technology", "Financials"],
                }
            )

        dates = pd.date_range("2024-01-02", periods=9, freq="D")
        rows = []
        for code, base in [("AAA", 100.0), ("BBB", 20.0)]:
            for idx, value_date in enumerate(dates):
                close = base + idx
                rows.append(
                    {
                        "date_": value_date,
                        "code": code,
                        "open": close - 0.5,
                        "high": close + 0.5,
                        "low": close - 1.0,
                        "close": close,
                        "adj_close": close / 2.0,
                        "dividend": 0.0,
                        "volume": 1000.0 + 10.0 * idx,
                        "shares_outstanding": 10_000.0,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(module, "fetch_data", fake_fetch_data)

    result = transformer.compute_all_alpha101_parallel(
        alpha_ids=[1, 2],
        codes=["AAA", "BBB"],
        start="2024-01-02",
        end="2024-01-12",
        compute_workers=2,
        save_workers=2,
        save_factors=True,
        output_dir=tmp_path,
    )

    assert result.alpha_ids == [1, 2]
    assert result.saved_factor_paths is not None
    assert "alpha_001" in result.saved_factor_paths
    assert "alpha_002" in result.saved_factor_paths
    assert result.factor_frame.columns.tolist() == ["date_", "code", "alpha_001", "alpha_002"]
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "alpha_001.parquet").exists()
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "alpha_002.parquet").exists()
