from __future__ import annotations

import pandas as pd

from tiger_factors.utils.merge import merge_code_date_frames
from tiger_factors.utils.merge import merge_by_keys
from tiger_factors.utils.merge import merge_code_frames
from tiger_factors.utils.merge import merge_frames
from tiger_factors.utils.merge import merge_other_frames


def test_merge_code_frames_uses_aliases_and_prefixes():
    companies = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "sector": ["tech", "software"],
        }
    )
    industry = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "industry": ["hardware", "software"],
        }
    )

    out = merge_code_frames([companies, industry], names=["companies", "industry"])

    assert list(out.columns) == ["code", "companies__sector", "industry__industry"]
    assert out["code"].tolist() == ["AAPL", "MSFT"]


def test_merge_other_frames_joins_on_custom_key():
    companies = pd.DataFrame(
        {
            "industry_id": [10, 20],
            "code": ["AAPL", "MSFT"],
        }
    )
    lookup = pd.DataFrame(
        {
            "industry_id": [10, 20],
            "sector": ["tech", "software"],
        }
    )

    out = merge_other_frames([companies, lookup], join_keys=["industry_id"], names=["companies", "lookup"])

    assert list(out.columns) == ["industry_id", "companies__code", "lookup__sector"]
    assert out["lookup__sector"].tolist() == ["tech", "software"]


def test_merge_code_date_frames_keeps_both_keys():
    left = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-02"],
            "code": ["AAPL", "MSFT"],
            "x": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "MSFT"],
            "y": [3.0, 4.0],
        }
    )

    out = merge_code_date_frames([left, right], names=["left", "right"])

    assert list(out.columns) == ["date_", "code", "left__x", "right__y"]
    assert pd.api.types.is_datetime64_any_dtype(out["date_"])


def test_merge_frames_dispatches_by_mode():
    frame = pd.DataFrame({"code": ["AAPL"], "value": [1.0]})
    out = merge_frames([frame], mode="code", names=["values"])
    assert list(out.columns) == ["code", "values__value"]


def test_merge_by_keys_keeps_original_column_names():
    companies = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "industry_id": [10, 20],
        }
    )
    industry = pd.DataFrame(
        {
            "industry_id": [10, 20],
            "sector": ["tech", "software"],
            "industry": ["hardware", "software"],
        }
    )

    out = merge_by_keys([companies, industry], join_keys=["industry_id"])

    assert list(out.columns) == ["code", "industry_id", "sector", "industry"]
    assert out["sector"].tolist() == ["tech", "software"]
