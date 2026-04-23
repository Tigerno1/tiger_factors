from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import AdjPriceData
from tiger_factors.factor_store import FactorData
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import MacroBatchResult
from tiger_factors.factor_store import MacroData
from tiger_factors.factor_store import MacroSpec
from tiger_factors.factor_store import OthersSpec
from tiger_factors.factor_store import FactorStore


def test_factor_monthly_split_and_cross_month_load(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="roe",
        variant=None,
        provider="simfin",
    )
    frame = pd.DataFrame(
        {
            "code": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date_": ["2024-01-05", "2024-02-02", "2024-02-10", "2024-03-01"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = store.save_factor(spec, frame)

    assert result.dataset_dir.exists()
    assert result.dataset_dir == tmp_path / "factor" / "simfin" / "us" / "stock" / "1d"
    assert (result.dataset_dir / "roe__2024-01.parquet").exists()
    assert (result.dataset_dir / "roe__2024-02.parquet").exists()
    assert (result.dataset_dir / "roe__2024-03.parquet").exists()
    assert result.manifest_path.exists()

    loaded = store.get_factor(spec)
    loaded_by_get = store.get_factor(spec)
    assert len(loaded) == 4
    assert loaded_by_get.equals(loaded)
    assert list(loaded.columns) == ["code", "date_", "value"]
    assert loaded.loc[loaded["code"] == "AAPL", "value"].tolist() == [1.0, 2.0]

    filtered = store.get_factor(spec, start="2024-02-01", end="2024-03-31")
    assert len(filtered) == 3
    assert filtered["date_"].min() == pd.Timestamp("2024-02-02")
    assert filtered["date_"].max() == pd.Timestamp("2024-03-01")

    ibis_expr = store.get_factor(spec, as_query=True)
    assert hasattr(ibis_expr, "to_pandas")
    ibis_frame = ibis_expr.to_pandas()
    assert len(ibis_frame) == 4

    duckdb_loaded = store.get_factor(spec, engine="duckdb")
    assert isinstance(duckdb_loaded, pd.DataFrame)
    assert duckdb_loaded.equals(loaded)

    overwritten = pd.DataFrame(
        {
            "code": ["AAPL"],
            "date_": ["2024-02-02"],
            "value": [9.5],
        }
    )
    with pytest.raises(FileExistsError, match="dataset already exists"):
        store.save_factor(spec, overwritten)
    store.save_factor(spec, overwritten, force_updated=True)
    loaded_after = store.get_factor(spec)
    feb_value = loaded_after.loc[
        (loaded_after["code"] == "AAPL") & (loaded_after["date_"] == pd.Timestamp("2024-02-02").normalize()),
        "value",
    ].iloc[0]
    assert feb_value == 9.5
    assert len(loaded_after) == 1

    with pytest.raises(ValueError, match="factor frame contains no valid rows after normalization"):
        store.save_factor(
            spec,
            pd.DataFrame(
                {
                    "code": ["AAPL"],
                    "date_": ["2024-02-02"],
                    "value": [pd.NA],
                }
            ),
        )
    loaded_after_nan = store.get_factor(spec)
    feb_value_after_nan = loaded_after_nan.loc[
        (loaded_after_nan["code"] == "AAPL") & (loaded_after_nan["date_"] == pd.Timestamp("2024-02-02").normalize()),
        "value",
    ].iloc[0]
    assert feb_value_after_nan == 9.5


def test_factor_store_defaults_to_factor_store_root() -> None:
    store = FactorStore()
    assert store.root_dir == Path("/Volumes/Quant_Disk/factor_store")


def test_get_factors_joins_multiple_specs_via_ibis(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec_a = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="roa",
        variant=None,
        provider="tiger",
    )
    spec_b = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="market_cap",
        variant=None,
        provider="tiger",
    )

    roa = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "date_": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "value": [0.10, 0.20, 0.11, 0.21],
        }
    )
    market_cap = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "date_": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "value": [100.0, 200.0, 110.0, 210.0],
        }
    )
    store.save_factor(spec_a, roa, force_updated=True)
    store.save_factor(spec_b, market_cap, force_updated=True)

    joined = store.get_factors([spec_a, spec_b], start="2024-01-02", end="2024-01-03")

    assert isinstance(joined, pd.DataFrame)
    assert list(joined.columns) == ["code", "date_", "roa", "market_cap"]
    assert joined.shape == (4, 4)
    assert joined.loc[(joined["code"] == "AAPL") & (joined["date_"] == pd.Timestamp("2024-01-02")), "roa"].iloc[0] == 0.10
    assert joined.loc[(joined["code"] == "MSFT") & (joined["date_"] == pd.Timestamp("2024-01-03")), "market_cap"].iloc[0] == 210.0


def test_get_factors_as_query_returns_ibis_relation(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec_a = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="bm",
        variant=None,
        provider="tiger",
    )
    spec_b = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="fscore",
        variant=None,
        provider="tiger",
    )
    frame = pd.DataFrame(
        {
            "code": ["AAPL", "MSFT"],
            "date_": ["2024-01-02", "2024-01-02"],
            "value": [1.0, 2.0],
        }
    )
    store.save_factor(spec_a, frame, force_updated=True)
    store.save_factor(spec_b, frame.assign(value=[10.0, 20.0]), force_updated=True)

    query = store.get_factors([spec_a, spec_b], as_query=True)

    assert hasattr(query, "to_pandas")
    joined = query.to_pandas()
    assert list(joined.columns) == ["code", "date_", "bm", "fscore"]
    assert joined.shape == (2, 4)


def test_factor_data_normalizes_to_standard_columns() -> None:
    schema = FactorData()
    frame = pd.DataFrame(
        {
            "code": ["AAPL", "AAPL", "MSFT"],
            "date_": ["2024-01-02", "2024-01-01", "2024-01-03"],
            "alpha_001": [2.0, 1.0, 3.0],
            "note": ["late", "early", "mid"],
        }
    )

    normalized = schema.normalize(frame, factor_name="alpha_001")

    assert list(normalized.columns) == ["code", "date_", "value"]
    assert normalized["code"].tolist() == ["AAPL", "AAPL", "MSFT"]
    assert normalized["date_"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert normalized["value"].tolist() == [1.0, 2.0, 3.0]


def test_factor_data_rejects_missing_value_column() -> None:
    schema = FactorData()
    frame = pd.DataFrame(
        {
            "code": ["AAPL"],
            "date_": ["2024-01-01"],
            "note": ["missing factor value"],
        }
    )

    with pytest.raises(ValueError, match="factor frame contains no valid rows after normalization"):
        schema.normalize(frame, factor_name="alpha_001")


def test_adj_price_data_preserves_extra_columns() -> None:
    schema = AdjPriceData()
    frame = pd.DataFrame(
        {
            "code": ["AAPL", "AAPL"],
            "date_": ["2024-01-02", "2024-01-01"],
            "open": [100.0, 99.0],
            "high": [101.0, 100.0],
            "low": [98.0, 97.0],
            "close": [100.5, 99.5],
            "volume": [10_000, 9_000],
            "dividend": [0.1, 0.0],
            "shares_outstanding": [1_000_000, 1_000_000],
        }
    )

    normalized = schema.normalize(frame)

    assert normalized.columns.tolist() == [
        "code",
        "date_",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividend",
        "shares_outstanding",
    ]
    assert normalized["code"].tolist() == ["AAPL", "AAPL"]
    assert normalized["date_"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]


def test_adj_price_data_rejects_missing_core_columns() -> None:
    schema = AdjPriceData()
    frame = pd.DataFrame(
        {
            "code": ["AAPL"],
            "date_": ["2024-01-01"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        }
    )

    try:
        schema.normalize(frame)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "volume" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid adj price frame")


def test_macro_data_normalizes_to_standard_columns() -> None:
    schema = MacroData()
    frame = pd.DataFrame(
        {
            "date_": ["2024-01-02", "2024-01-01"],
            "value": [2.0, 1.0],
            "note": ["b", "a"],
        }
    )

    normalized = schema.normalize(frame)

    assert normalized.columns.tolist() == ["date_", "value"]
    assert normalized["date_"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert normalized["value"].tolist() == [1.0, 2.0]


def test_macro_data_rejects_missing_value_column() -> None:
    schema = MacroData()
    frame = pd.DataFrame(
        {
            "date_": ["2024-01-01"],
            "note": ["missing macro value"],
        }
    )

    try:
        schema.normalize(frame)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "value" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid macro frame")


def test_adj_price_monthly_split_and_range_load(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = AdjPriceSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        variant="fwd",
        provider="simfin",
    )
    frame = pd.DataFrame(
        {
            "code": ["AAPL", "AAPL", "AAPL"],
            "date_": ["2024-01-05", "2024-01-10", "2024-02-02"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10_000, 11_000, 12_000],
            "adj_close": [100.4, 101.4, 102.4],
        }
    )

    result = store.save_adj_price(spec, frame)

    assert result.dataset_dir.exists()
    assert (result.dataset_dir / "adj_price__fwd__2024-01.parquet").exists()
    assert (result.dataset_dir / "adj_price__fwd__2024-02.parquet").exists()

    loaded = store.get_adj_price(spec, start="2024-01-01", end="2024-01-31")
    assert len(loaded) == 2
    assert list(loaded.columns) == [
        "code",
        "date_",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    ]

    with pytest.raises(FileExistsError, match="adj price dataset already exists"):
        store.save_adj_price(spec, frame)

    updated = frame.copy()
    updated.loc[updated["date_"] == "2024-01-05", "close"] = 999.0
    store.save_adj_price(spec, updated, force_updated=True)
    reloaded = store.get_adj_price(spec, start="2024-01-01", end="2024-01-31")
    assert reloaded.loc[reloaded["date_"] == pd.Timestamp("2024-01-05"), "close"].iloc[0] == 999.0
    feb_close = store.get_adj_price(spec, start="2024-02-01", end="2024-02-28").loc[
        lambda df: df["date_"] == pd.Timestamp("2024-02-02"),
        "close",
    ].iloc[0]
    assert feb_close == 102.5
    assert result.dataset_dir == tmp_path / "price" / "simfin" / "us" / "stock" / "1d"


def test_macro_and_polars_roundtrip(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = MacroSpec(
        region="US",
        freq="1M",
        table_name="FEDFUNDS",
        variant=None,
        provider="PANDAS_DATAREADER",
    )
    frame = pl.DataFrame(
        {
            "date_": ["2024-01-31", "2024-02-29"],
            "value": [5.33, 5.44],
        }
    )

    store.save_macro(spec, frame)
    dataset_dir = spec.dataset_dir(tmp_path)
    assert dataset_dir.exists()
    assert (dataset_dir / "fedfunds.parquet").exists()
    assert list(dataset_dir.glob("*.parquet")) == [dataset_dir / "fedfunds.parquet"]
    assert dataset_dir == tmp_path / "macro" / "pandas_datareader" / "us" / "1m"
    assert spec.region == "us"
    assert spec.freq == "1m"
    assert spec.table_name == "fedfunds"
    assert spec.variant is None
    assert spec.provider == "pandas_datareader"
    assert spec.series_name == "FEDFUNDS"

    loaded = store.get_macro(spec, engine="polars")

    assert isinstance(loaded, pl.DataFrame)
    assert loaded.shape == (2, 2)
    assert loaded.columns == ["date_", "value"]

    with pytest.raises(FileExistsError, match="dataset already exists"):
        store.save_macro(spec, frame)
    store.save_macro(spec, frame, force_updated=True)


def test_macro_infers_freq_and_normalizes_date_column(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = MacroSpec(
        region="US",
        freq="auto",
        table_name="DGS10",
        variant=None,
        provider="FRED",
    )
    frame = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "DGS10": [4.1, 4.2, 4.3],
        }
    )

    result = store.save_macro(spec, frame)

    assert result.dataset_dir == tmp_path / "macro" / "fred" / "us" / "1d"
    assert (result.dataset_dir / "dgs10.parquet").exists()
    loaded = store.get_macro(MacroSpec(region="US", freq="1d", table_name="DGS10", provider="FRED", variant=None))
    assert list(loaded.columns) == ["date_", "value"]
    assert loaded["value"].tolist() == [4.1, 4.2, 4.3]


def test_download_macro_uses_default_save(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = MacroSpec(
        region="US",
        freq="1M",
        table_name="FEDFUNDS",
        variant=None,
        provider="FRED",
    )
    raw = pd.DataFrame(
        {"FEDFUNDS": [5.33, 5.44]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    calls: list[tuple[object, ...]] = []

    def reader(name: object, source: object, **kwargs: object) -> pd.DataFrame:
        calls.append((name, source, kwargs.get("start"), kwargs.get("end"), kwargs.get("api_key")))
        return raw

    result = store.download_macro(
        spec,
        start="2024-01-01",
        end="2024-02-29",
        reader=reader,
    )

    assert calls == [("FEDFUNDS", "fred", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29"), None)]
    assert result.dataset_dir.exists()
    assert (result.dataset_dir / "fedfunds.parquet").exists()
    assert result.dataset_dir == tmp_path / "macro" / "fred" / "us" / "1m"
    loaded = store.get_macro(spec)
    assert len(loaded) == 2
    assert loaded.columns.tolist() == ["date_", "value"]


def test_download_macro_batch(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    specs = [
        MacroSpec(region="US", freq="1M", table_name="FEDFUNDS", variant=None, provider="FRED"),
        MacroSpec(region="US", freq="1D", table_name="DGS10", variant=None, provider="FRED"),
    ]
    raw_frames = {
        "FEDFUNDS": pd.DataFrame({"FEDFUNDS": [5.33]}, index=pd.to_datetime(["2024-01-31"])),
        "DGS10": pd.DataFrame({"DGS10": [4.10]}, index=pd.to_datetime(["2024-01-31"])),
    }
    calls: list[tuple[object, ...]] = []

    def reader(name: object, source: object, **kwargs: object) -> pd.DataFrame:
        calls.append((name, source))
        return raw_frames[str(name)]

    results = store.download_macro_batch(specs, reader=reader)

    assert set(results) == {"fred/us/1m/fedfunds", "fred/us/1d/dgs10"}
    assert calls == [("FEDFUNDS", "fred"), ("DGS10", "fred")]
    for spec in specs:
        loaded = store.get_macro(MacroSpec(region=spec.region, freq="auto", table_name=spec.table_name, variant=spec.variant, provider=spec.provider))
        assert len(loaded) == 1
        assert loaded.columns.tolist() == ["date_", "value"]


def test_download_fred_macro_series(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    raw_frames = {
        "FEDFUNDS": pd.DataFrame({"FEDFUNDS": [5.33]}, index=pd.to_datetime(["2024-01-31"])),
        "DGS10": pd.DataFrame({"DGS10": [4.10]}, index=pd.to_datetime(["2024-01-31"])),
    }
    calls: list[tuple[object, object]] = []

    def reader(name: object, source: object, **kwargs: object) -> pd.DataFrame:
        calls.append((name, source))
        return raw_frames[str(name)]

    batch = store.download_fred_macro_series(["FEDFUNDS", "DGS10"], reader=reader)

    assert isinstance(batch, MacroBatchResult)
    assert set(batch.results) == {"fred/us/auto/fedfunds", "fred/us/auto/dgs10"}
    assert calls == [("FEDFUNDS", "fred"), ("DGS10", "fred")]
    assert (tmp_path / "macro" / "fred" / "us" / "auto" / "fedfunds.parquet").exists()
    assert (tmp_path / "macro" / "fred" / "us" / "auto" / "dgs10.parquet").exists()
    assert batch.manifest_path.exists()


def test_download_fred_macro_series_from_file(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    list_file = tmp_path / "fred_series.txt"
    list_file.write_text("# comment\nFEDFUNDS\nDGS10\n", encoding="utf-8")
    raw_frames = {
        "FEDFUNDS": pd.DataFrame({"FEDFUNDS": [5.33]}, index=pd.to_datetime(["2024-01-31"])),
        "DGS10": pd.DataFrame({"DGS10": [4.10]}, index=pd.to_datetime(["2024-01-31"])),
    }
    calls: list[tuple[object, object]] = []

    def reader(name: object, source: object, **kwargs: object) -> pd.DataFrame:
        calls.append((name, source))
        return raw_frames[str(name)]

    batch = store.download_fred_macro_series_from_file(list_file, reader=reader)

    assert isinstance(batch, MacroBatchResult)
    assert set(batch.results) == {"fred/us/auto/fedfunds", "fred/us/auto/dgs10"}
    assert calls == [("FEDFUNDS", "fred"), ("DGS10", "fred")]
    assert batch.manifest_path.exists()


def test_macro_variant_none_omits_trailing_path(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = MacroSpec(
        region="US",
        freq="1M",
        table_name="FEDFUNDS",
        variant=None,
        provider="FRED",
    )
    frame = pd.DataFrame(
        {
            "date_": ["2024-01-31"],
            "value": [5.33],
        }
    )

    result = store.save_macro(spec, frame)

    assert result.dataset_dir == tmp_path / "macro" / "fred" / "us" / "1m"
    assert (result.dataset_dir / "fedfunds.parquet").exists()


def test_others_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = OthersSpec(
        provider="tiger",
        region="US",
        sec_type="stock",
        freq="1D",
        table_name="note_table",
        variant=None,
    )
    frame = pd.DataFrame(
        {
            "name": ["alpha", "beta"],
            "score": [1.2, 3.4],
        }
    )

    result = store.save_others(spec, frame)

    assert result.dataset_dir == tmp_path / "others" / "tiger" / "us" / "stock" / "1d" / "note_table"
    assert (result.dataset_dir / "note_table.parquet").exists()
    loaded = store.get_others(spec)
    assert loaded.equals(frame)

    with pytest.raises(FileExistsError, match="others dataset already exists"):
        store.save_others(spec, frame)

    updated = pd.DataFrame(
        {
            "name": ["alpha", "beta"],
            "score": [9.9, 3.4],
        }
    )
    store.save_others(spec, updated, force_updated=True)
    reloaded = store.get_others(spec)
    assert reloaded.loc[reloaded["name"] == "alpha", "score"].iloc[0] == 9.9


def test_others_spec_rejects_unknown_dimensions() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        OthersSpec(
            provider="local",
            region="us",
            sec_type="stock",
            freq="1d",
            table_name="note_table",
        )


def test_load_evaluation_summary_from_factor_store(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="alpha_test",
    )
    summary_dir = tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_test" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    expected = pd.DataFrame([{"ic_mean": 1.23, "sharpe": 4.56}])
    expected.to_parquet(summary_dir / "summary.parquet", index=False)

    loaded = store.evaluation.summary(spec).get_table()

    assert loaded.equals(expected)


def test_factor_store_evaluation_reader_and_store_are_separated(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="alpha_test",
    )
    expected = pd.DataFrame([{"bucket": "long", "return": 0.12}])

    result_path = store.evaluation_store.save_returns(
        expected,
        spec=spec,
        table_name="mean_return_by_quantile",
    )

    assert result_path == tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_test" / "returns" / "mean_return_by_quantile.parquet"
    loaded = store.evaluation.section(spec, "returns").get_table("mean_return_by_quantile")
    assert loaded.equals(expected)
    assert not hasattr(store.evaluation, "save_returns")
    assert hasattr(store, "evaluation_store")
