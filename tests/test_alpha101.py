from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tiger_factors import Alpha101Engine, NeutralizationColumns, alpha101_factor_names, alpha101_descriptions
from tiger_factors.factor_algorithm.alpha101.engine import build_code_industry_frame


def _sample_ohlcv_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rows = []
    base_prices = {"AAA": 100.0, "BBB": 50.0, "CCC": 75.0}
    for code, base_price in base_prices.items():
        for idx, date in enumerate(dates):
            drift = 0.2 * idx
            noise = {"AAA": 1.0, "BBB": -0.5, "CCC": 0.3}[code]
            close = base_price + drift + noise * np.sin(idx / 5)
            open_ = close - 0.4
            high = close + 1.0
            low = close - 1.2
            volume = 1_000_000 + idx * 1000 + (hash(code) % 1000)
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(volume),
                    "industry": "tech" if code != "BBB" else "finance",
                }
            )
    return pd.DataFrame(rows)


def _sample_adjusted_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["AAA", "AAA"],
            "open": [50.0, 55.0],
            "high": [51.0, 56.0],
            "low": [49.0, 54.0],
            "close": [50.0, 55.0],
            "volume": [1000.0, 2000.0],
            "shares_outstanding": [10_000.0, 10_000.0],
        }
    )


def _reference_ohlcv_panel(*, collapsed_groups: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=320, freq="D")
    rows = []
    configs = [
        ("AAA", 100.0, "tech", "software", "app"),
        ("BBB", 60.0, "finance", "bank", "retail_bank"),
        ("CCC", 80.0, "energy", "oil", "upstream"),
        ("DDD", 120.0, "tech", "hardware", "devices"),
    ]
    for code, base, sector, industry, subindustry in configs:
        for idx, date in enumerate(dates):
            close = base + 0.08 * idx + np.sin(idx / 7.0) + np.cos(idx / 13.0)
            group_value = industry if collapsed_groups else None
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "open": close - 0.4,
                    "high": close + 1.2,
                    "low": close - 1.1,
                    "close": close,
                    "volume": float(1_000_000 + idx * 500 + (abs(hash(code)) % 997)),
                    "sector": group_value or sector,
                    "industry": industry,
                    "subindustry": group_value or subindustry,
                }
            )
    return pd.DataFrame(rows)


def test_alpha101_names_has_full_coverage():
    names = alpha101_factor_names()
    assert len(names) == 101
    assert names[0] == "alpha_001"
    assert names[-1] == "alpha_101"


def test_alpha101_engine_computes_factor_one():
    engine = Alpha101Engine(_sample_ohlcv_panel())
    result = engine.compute(1)
    assert list(result.columns) == ["date_", "code", "alpha_001"]
    assert not result.empty


def test_alpha101_engine_uses_standardized_price_columns():
    engine = Alpha101Engine(_sample_adjusted_panel())
    assert list(engine.data["symbol"].unique()) == ["AAA"]
    assert engine.data.loc[0, "close"] == 50.0
    assert engine.data.loc[0, "open"] == 50.0
    assert engine.data.loc[1, "return"] == pytest.approx(0.1)
    assert engine.data.loc[1, "market_value"] == 550_000.0


def test_build_code_industry_frame_joins_companies_and_industries():
    companies = pd.DataFrame(
        {
            "code": ["AAA", "BBB", "CCC"],
            "industry_id": [10, 20, None],
        }
    )
    industries = pd.DataFrame(
        {
            "industry_id": [10, 20],
            "industry": ["software", "bank"],
            "sector": ["tech", "finance"],
        }
    )

    result = build_code_industry_frame(companies, industries)

    assert result.loc[result["code"] == "AAA", "industry"].iloc[0] == "software"
    assert result.loc[result["code"] == "BBB", "sector"].iloc[0] == "finance"
    assert result.loc[result["code"] == "CCC", "industry"].isna().iloc[0]


def test_alpha101_engine_can_compute_all_columns():
    engine = Alpha101Engine(_sample_ohlcv_panel())
    result = engine.compute_all(alpha_ids=[1, 21, 48, 61, 101])
    assert {"date_", "code", "alpha_001", "alpha_021", "alpha_048", "alpha_061", "alpha_101"}.issubset(result.columns)


def test_alpha101_engine_full_smoke_run():
    engine = Alpha101Engine(_sample_ohlcv_panel())
    result = engine.compute_all()
    alpha_columns = [column for column in result.columns if column.startswith("alpha_")]
    assert len(alpha_columns) == 101
    assert result[alpha_columns].notna().sum().sum() > 0


def test_alpha101_engine_matches_public_reference_values_for_selected_factors():
    engine = Alpha101Engine(_reference_ohlcv_panel(collapsed_groups=True))
    expected = {
        1: {"AAA": 0.125, "BBB": 0.125, "CCC": 0.125, "DDD": 0.125},
        48: {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0},
        69: {"AAA": -0.9491175456, "BBB": -0.9491175456, "CCC": -0.9491175456, "DDD": -0.9491175456},
        80: {"AAA": -0.9246555971, "BBB": -0.9246555971, "CCC": -0.9246555971, "DDD": -0.9246555971},
        82: {"AAA": -0.5, "BBB": -0.5, "CCC": -0.5, "DDD": -1.0},
        89: {"AAA": -0.2833333333, "BBB": -0.2833333333, "CCC": -0.2833333333, "DDD": -0.2833333333},
        101: {"AAA": 0.173837462, "BBB": 0.173837462, "CCC": 0.173837462, "DDD": 0.173837462},
    }
    for alpha_id, expected_values in expected.items():
        frame = engine.compute(alpha_id)
        actual = frame.groupby("code").tail(1).set_index("code")[f"alpha_{alpha_id:03d}"].round(10).to_dict()
        assert actual == expected_values


def test_alpha101_engine_exposes_descriptions_for_all_factors():
    descriptions = alpha101_descriptions()
    assert len(descriptions) == 101
    assert descriptions[1]
    assert descriptions[101]

    engine = Alpha101Engine(_sample_ohlcv_panel())
    assert engine.alpha_description(1) == descriptions[1]
    assert engine.compute_series(1).attrs["description"] == descriptions[1]
    assert engine.compute(1).attrs["description"] == descriptions[1]


def test_alpha101_neutralization_levels_are_configurable():
    frame = _reference_ohlcv_panel(collapsed_groups=False).rename(
        columns={
            "sector": "gics_sector",
            "industry": "gics_industry",
            "subindustry": "gics_subindustry",
        }
    )
    engine = Alpha101Engine(
        frame,
        neutralization_columns=NeutralizationColumns(
            sector="gics_sector",
            industry="gics_industry",
            subindustry="gics_subindustry",
        ),
    )
    first_date = engine.data["date"].min()
    sample_index = engine.data.index[engine.data["date"] == first_date]
    sample = pd.Series([1.0, 3.0, 10.0, 14.0], index=sample_index)
    sector_neutral = engine._neutralize(sample, "sector")
    industry_neutral = engine._neutralize(sample, "industry")
    subindustry_neutral = engine._neutralize(sample, "subindustry")
    assert sector_neutral.groupby(engine.data.loc[sample.index, "sector"]).mean().abs().max() < 1e-12
    assert industry_neutral.groupby(engine.data.loc[sample.index, "industry"]).mean().abs().max() < 1e-12
    assert subindustry_neutral.groupby(engine.data.loc[sample.index, "subindustry"]).mean().abs().max() < 1e-12
    assert not sector_neutral.equals(industry_neutral)
