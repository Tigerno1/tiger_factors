from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import TigerFactorLibrary


def test_load_factor_frame_merges_multiple_factors(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec_kwargs = dict(provider="tiger", region="us", sec_type="stock", freq="1d", variant=None)
    factor_names = ["BM", "FSCORE", "BMFSCORE"]
    dates = pd.bdate_range("2024-01-02", periods=6)
    codes = ["AAPL", "MSFT", "NVDA"]

    for offset, factor_name in enumerate(factor_names):
        rows: list[dict[str, object]] = []
        for date_idx, date in enumerate(dates):
            for code_idx, code in enumerate(codes):
                rows.append(
                    {
                        "code": code,
                        "date_": date,
                        "value": float((offset + 1) * 10 + date_idx + code_idx),
                    }
                )
        store.save_factor(
            FactorSpec(table_name=factor_name, **spec_kwargs),
            pd.DataFrame(rows),
            force_update=True,
        )

    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)
    factor_frame = library.load_factor_frame(factor_names=factor_names, provider="tiger")
    panels = library.load_factor_panels(factor_names=factor_names, provider="tiger")

    assert set(panels) == set(factor_names)
    assert not factor_frame.empty
    assert {"date_", "code", *factor_names}.issubset(factor_frame.columns)
    assert factor_frame.groupby(["date_", "code"]).size().eq(1).all()
    assert factor_frame[factor_names].notna().all().all()


def test_load_factor_frame_applies_code_and_date_filters_in_query(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec_kwargs = dict(provider="tiger", region="us", sec_type="stock", freq="1d", variant=None)
    factor_names = ["BM", "FSCORE"]
    dates = pd.bdate_range("2024-01-02", periods=5)
    codes = ["AAPL", "MSFT", "NVDA"]

    for offset, factor_name in enumerate(factor_names):
        rows: list[dict[str, object]] = []
        for date_idx, date in enumerate(dates):
            for code_idx, code in enumerate(codes):
                rows.append(
                    {
                        "code": code,
                        "date_": date,
                        "value": float((offset + 1) * 100 + date_idx * 10 + code_idx),
                    }
                )
        store.save_factor(
            FactorSpec(table_name=factor_name, **spec_kwargs),
            pd.DataFrame(rows),
            force_update=True,
        )

    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)
    factor_frame = library.load_factor_frame(
        factor_names=factor_names,
        provider="tiger",
        codes=["MSFT", "AAPL"],
        start="2024-01-03",
        end="2024-01-05",
    )

    assert not factor_frame.empty
    assert factor_frame.equals(
        factor_frame.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)
    )
    assert set(factor_frame["code"]) == {"AAPL", "MSFT"}
    assert factor_frame["date_"].min() == pd.Timestamp("2024-01-03")
    assert factor_frame["date_"].max() == pd.Timestamp("2024-01-05")
    assert {"date_", "code", *factor_names}.issubset(factor_frame.columns)
