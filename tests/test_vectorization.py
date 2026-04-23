from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pytest

from tiger_factors.utils.calculation import Interval
from tiger_factors.factor_maker.vectorization import FactorVectorizationTransformer
from tiger_factors.factor_maker.vectorization import VectorDatasetSpec


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


@dataclass
class FakeIntradayCalendar(FakeCalendar):
    open_time: str = "09:30"
    close_time: str = "16:00"

    def session_open_close(self, session_label):
        day = pd.Timestamp(session_label).date()
        open_ts = pd.Timestamp(f"{day.isoformat()} {self.open_time}", tz="UTC")
        close_ts = pd.Timestamp(f"{day.isoformat()} {self.close_time}", tz="UTC")
        return open_ts, close_ts


def test_vectorization_merges_daily_frames_on_calendar_master():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    price = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 4)],
            "close": [10.0, 12.0],
        }
    )
    signal = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 3)],
            "score": [1.0, 2.0],
        }
    )

    result = engine.merge(
        [
            VectorDatasetSpec(name="price", frame=price, time_column="date_"),
            VectorDatasetSpec(name="signal", frame=signal, time_column="date_", forward_fill=True),
        ]
    )

    frame = result.frame
    assert list(frame["date_"]) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert list(frame["eff_at"]) == [date(2024, 1, 3), date(2024, 1, 4), None]
    assert frame["price__close"].iloc[0] == 10.0
    assert pd.isna(frame["price__close"].iloc[1])
    assert frame["price__close"].iloc[2] == 12.0
    assert list(frame["signal__score"]) == [1.0, 2.0, 2.0]
    assert result.key_column == "exchange_date_"
    assert result.eff_column == "eff_at"
    assert result.available_column == "eff_at"
    assert result.time_kind == "daily"


def test_vectorization_accepts_plain_dataframe_list():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    price = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2)],
            "close": [10.0],
        }
    )
    signal = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 3)],
            "score": [1.0, 2.0],
        }
    )

    result = engine.merge([price, signal], time_column="date_", forward_fill=[False, True])
    frame = result.frame

    assert list(frame["date_"]) == [date(2024, 1, 2), date(2024, 1, 3)]
    assert list(frame["eff_at"]) == [date(2024, 1, 3), None]
    assert frame["dataset_0__close"].iloc[0] == 10.0
    assert pd.isna(frame["dataset_0__close"].iloc[1])
    assert list(frame["dataset_1__score"]) == [1.0, 2.0]


def test_vectorization_requires_matching_forward_fill_length():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    with pytest.raises(ValueError, match="forward_fill sequence must have the same length as datasets"):
        engine.merge(
            [
                pd.DataFrame({"date_": [date(2024, 1, 2)], "close": [10.0]}),
                pd.DataFrame({"date_": [date(2024, 1, 2)], "score": [1.0]}),
            ],
            time_column="date_",
            forward_fill=[True],
        )


def test_vectorization_none_keeps_spec_default_forward_fill():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    signal = pd.DataFrame(
            {
                "date_": [date(2024, 1, 2), date(2024, 1, 3)],
                "score": [1.0, 2.0],
            }
        )

    result = engine.merge(
        [
            VectorDatasetSpec(
                name="signal",
                frame=signal,
                time_column="date_",
                forward_fill=True,
            )
        ],
        forward_fill=[None],
    )

    frame = result.frame
    assert list(frame["signal__score"]) == [1.0, 2.0]


def test_vectorization_long_to_wide_single_value_uses_code_columns():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    long_frame = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "code": ["AAA", "BBB", "AAA"],
            "close": [1.0, 2.0, 3.0],
            "open": [10.0, 20.0, 30.0],
        }
    )

    wide = engine.long_to_wide(long_frame, time_column="date_", code_column="code", value_columns=["close"])

    assert list(wide.columns) == ["AAA", "BBB"]
    assert float(wide.loc[pd.Timestamp("2024-01-02"), "AAA"]) == 1.0
    assert float(wide.loc[pd.Timestamp("2024-01-02"), "BBB"]) == 2.0
    assert float(wide.loc[pd.Timestamp("2024-01-03"), "AAA"]) == 3.0


def test_vectorization_long_to_wide_multiple_values_uses_code_field_columns():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    long_frame = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "code": ["AAA", "BBB", "AAA"],
            "open": [10.0, 20.0, 30.0],
            "close": [1.0, 2.0, 3.0],
        }
    )

    wide = engine.long_to_wide(
        long_frame,
        time_column="date_",
        code_column="code",
        value_columns=["open", "close"],
    )

    assert list(wide.columns) == ["AAA__open", "AAA__close", "BBB__open", "BBB__close"]
    assert float(wide.loc[pd.Timestamp("2024-01-02"), "AAA__open"]) == 10.0
    assert float(wide.loc[pd.Timestamp("2024-01-02"), "AAA__close"]) == 1.0
    assert float(wide.loc[pd.Timestamp("2024-01-02"), "BBB__open"]) == 20.0
    assert float(wide.loc[pd.Timestamp("2024-01-02"), "BBB__close"]) == 2.0


def test_vectorization_adjusts_raw_ohlcv_using_adj_close():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    raw = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 3)],
            "code": ["AAA", "AAA"],
            "open": [100.0, 110.0],
            "high": [102.0, 112.0],
            "low": [98.0, 108.0],
            "close": [100.0, 110.0],
            "adj_close": [50.0, 55.0],
            "dividend": [0.0, 1.0],
            "volume": [1000.0, 2000.0],
        }
    )

    adjusted = engine.adjust(raw, time_column="date_", code_column="code")

    assert list(adjusted["date_"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert adjusted.loc[0, "open"] == pytest.approx(50.0)
    assert adjusted.loc[0, "high"] == pytest.approx(51.0)
    assert adjusted.loc[0, "low"] == pytest.approx(49.0)
    assert adjusted.loc[0, "close"] == pytest.approx(50.0)
    assert adjusted.loc[0, "volume"] == 2000.0
    assert adjusted.loc[1, "close"] == 55.0


def test_vectorization_adjust_can_include_dividends_when_enabled():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 3)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=1)

    raw = pd.DataFrame(
        {
            "date_": [date(2024, 1, 2), date(2024, 1, 3)],
            "code": ["AAA", "AAA"],
            "open": [100.0, 110.0],
            "high": [102.0, 112.0],
            "low": [98.0, 108.0],
            "close": [100.0, 110.0],
            "adj_close": [50.0, 55.0],
            "dividend": [0.0, 1.0],
            "volume": [1000.0, 2000.0],
        }
    )

    adjusted_without_dividends = engine.adjust(raw, time_column="date_", code_column="code", dividends=False)
    adjusted_with_dividends = engine.adjust(raw, time_column="date_", code_column="code", dividends=True)

    assert adjusted_without_dividends.loc[0, "close"] == pytest.approx(50.0)
    assert adjusted_with_dividends.loc[0, "close"] == pytest.approx(50.4545454545)
    assert adjusted_with_dividends.loc[0, "close"] != pytest.approx(adjusted_without_dividends.loc[0, "close"])


def test_vectorization_supports_shifted_eff_at_lag():
    calendar = FakeCalendar(
        trading_days={date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)}
    )
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", interval=Interval(day=1), lag=2)

    result = engine.merge(
        [
            VectorDatasetSpec(
                name="factor",
                frame=pd.DataFrame(
                    {
                        "date_": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
                        "alpha": [1.0, 2.0, 3.0],
                    }
                ),
            )
        ]
    )

    frame = result.frame
    assert list(frame["date_"]) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    assert list(frame["eff_at"]) == [date(2024, 1, 4), date(2024, 1, 5), None, None]


def test_vectorization_merges_intraday_frames_with_datetime_calendar():
    calendar = FakeIntradayCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorVectorizationTransformer(calendar=calendar, start="2024-01-01", lag=1)
    engine.set_interval(min=30)

    price = pd.DataFrame(
        {
            "date_": [
                pd.Timestamp("2024-01-02 09:30:00+00:00"),
                pd.Timestamp("2024-01-02 10:30:00+00:00"),
            ],
            "close": [10.0, 11.0],
        }
    )

    result = engine.merge([VectorDatasetSpec(name="price", frame=price, time_column="date_")])
    frame = result.frame

    assert result.key_column == "exchange_date_"
    assert "exchange_date_" in frame.columns
    assert result.eff_column == "eff_at"
    assert result.available_column == "eff_at"
    assert result.time_kind == "intraday"
    assert frame.iloc[0]["date_"] == pd.Timestamp("2024-01-02 09:30:00+00:00")
    assert frame.iloc[0]["eff_at"] == pd.Timestamp("2024-01-02 10:00:00+00:00")
    assert frame.iloc[-1]["date_"] == pd.Timestamp("2024-01-02 16:00:00+00:00")
    assert frame.iloc[0]["price__close"] == 10.0
    assert pd.isna(frame.iloc[1]["price__close"])


def test_vectorization_merges_code_frames_with_code_aliases():
    engine = FactorVectorizationTransformer(calendar="XNYS", start="2024-01-01", interval=Interval(day=1), lag=1)

    companies = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "company_name": ["Alpha", "Beta"],
        }
    )
    industries = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "industry": ["Tech Hardware", "Banks"],
        }
    )

    result = engine.merge_code_list([companies, industries])
    frame = result.frame

    assert result.join_keys == ("code",)
    assert result.time_kind == "static"
    assert list(frame["code"]) == ["AAA", "BBB"]
    assert frame["dataset_0__company_name"].tolist() == ["Alpha", "Beta"]
    assert frame["dataset_1__industry"].tolist() == ["Tech Hardware", "Banks"]


def test_vectorization_merges_other_frames_on_custom_keys():
    engine = FactorVectorizationTransformer(calendar="XNYS", start="2024-01-01", interval=Interval(day=1), lag=1)

    left = pd.DataFrame(
        {
            "industry_id": [1, 2],
            "alpha": [10.0, 20.0],
        }
    )
    right = pd.DataFrame(
        {
            "industry_id": [1, 2],
            "sector": ["Technology", "Financials"],
        }
    )

    result = engine.merge_other_list([left, right], join_keys=["industry_id"])
    frame = result.frame

    assert result.join_keys == ("industry_id",)
    assert list(frame["industry_id"]) == [1, 2]
    assert frame["dataset_0__alpha"].tolist() == [10.0, 20.0]
    assert frame["dataset_1__sector"].tolist() == ["Technology", "Financials"]
