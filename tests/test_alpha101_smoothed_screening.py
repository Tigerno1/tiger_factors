from __future__ import annotations

import pandas as pd

from tiger_factors.factor_algorithm.alpha101.smoothed_engine import smooth_alpha101_factor_frame


def test_smooth_alpha101_factor_frame_rolling_mean() -> None:
    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]),
            "code": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "alpha_001": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
            "alpha_002": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    smoothed = smooth_alpha101_factor_frame(frame, method="rolling_mean", window=2, min_periods=1)

    aaa = smoothed[smoothed["code"] == "AAA"].sort_values("date_")
    assert list(aaa["alpha_001"].round(6)) == [1.0, 2.0, 4.0]
    assert list(aaa["alpha_002"].round(6)) == [10.0, 15.0, 25.0]


def test_smooth_alpha101_factor_frame_ema() -> None:
    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "code": ["AAA", "AAA", "AAA"],
            "alpha_001": [1.0, 3.0, 5.0],
        }
    )

    smoothed = smooth_alpha101_factor_frame(frame, method="ema", ewm_span=2, min_periods=1)
    aaa = smoothed.sort_values("date_")
    assert aaa["alpha_001"].iloc[0] == 1.0
    assert aaa["alpha_001"].iloc[-1] > 3.0
