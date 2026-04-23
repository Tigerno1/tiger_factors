from __future__ import annotations

import pandas as pd

from tiger_factors import StreamingFactorEngine


def test_streaming_factor_engine_snapshot():
    engine = StreamingFactorEngine(buffer_size=3, prefer_rust=False)
    engine.update_bar(
        datetime=pd.Timestamp("2024-01-01"),
        open_price=10,
        high_price=11,
        low_price=9,
        close_price=10.5,
        volume=1,
    )
    engine.update_bar(
        datetime=pd.Timestamp("2024-01-02"),
        open_price=11,
        high_price=12,
        low_price=10,
        close_price=11.5,
        volume=2,
    )
    engine.update_bar(
        datetime=pd.Timestamp("2024-01-03"),
        open_price=12,
        high_price=13,
        low_price=11,
        close_price=12.5,
        volume=3,
    )
    snapshot = engine.latest_snapshot()
    assert snapshot["close"] == 12.5
