from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_store import TigerAPIAdapter, ProviderAdapter, TigerFactorLibrary, normalize_dates, to_long_factor


def _sample_price_rows() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=320, freq="D")
    rows = []
    for code, base in {"AAA": 100.0, "BBB": 60.0, "CCC": 80.0, "DDD": 120.0}.items():
        for idx, date in enumerate(dates):
            close = base + 0.08 * idx + np.sin(idx / 7.0) + np.cos(idx / 13.0)
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "open": close - 0.4,
                    "high": close + 1.2,
                    "low": close - 1.1,
                    "close": close,
                    "adj_close": close / 2.0,
                    "volume": float(1_000_000 + idx * 500 + (abs(hash(code)) % 997)),
                    "shares_outstanding": 10_000_000,
                }
            )
    return pd.DataFrame(rows)


def _sample_company_rows() -> pd.DataFrame:
    rows = []
    for code, sector, industry, subindustry, shares in [
        ("AAA", "tech", "software", "app", 10_000_000),
        ("BBB", "finance", "bank", "retail_bank", 12_000_000),
        ("CCC", "energy", "oil", "upstream", 8_000_000),
        ("DDD", "tech", "hardware", "devices", 9_000_000),
    ]:
        rows.append(
            {
                "date_": pd.Timestamp("2024-01-01"),
                "code": code,
                "sector": sector,
                "industry": industry,
                "subindustry": subindustry,
                "shares_basic": shares,
            }
        )
    return pd.DataFrame(rows)


def _sample_balance_rows() -> pd.DataFrame:
    rows = []
    for code, equity in [
        ("AAA", 50_000_000.0),
        ("BBB", 70_000_000.0),
        ("CCC", 40_000_000.0),
        ("DDD", 60_000_000.0),
    ]:
        rows.append(
            {
                "date_": pd.Timestamp("2023-12-31"),
                "code": code,
                "total_equity": equity,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_dates_returns_naive_datetime():
    values = ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
    normalized = normalize_dates(values)
    assert str(pd.DatetimeIndex(normalized).dtype) == "datetime64[ns]"


def test_to_long_factor_builds_expected_columns():
    frame = pd.DataFrame(
        {"AAPL": [1.0, 2.0], "MSFT": [3.0, 4.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    result = to_long_factor(frame, "demo_factor")
    assert list(result.columns) == ["date_", "code", "demo_factor"]
    assert len(result) == 4


def test_library_alpha101_saves_parquet_factor(tmp_path, monkeypatch):
    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)

    monkeypatch.setattr(
        library,
        "fetch_price_data",
        lambda **kwargs: _sample_price_rows(),
    )
    monkeypatch.setattr(
        library,
        "fetch_fundamental_data",
        lambda **kwargs: _sample_company_rows(),
    )

    result = library.alpha101(
        alpha_id=1,
        codes=["AAA", "BBB", "CCC", "DDD"],
        start="2024-02-01",
        end="2024-12-01",
    )

    assert result.name == "alpha_001"
    assert result.parquet_path.exists()
    assert result.metadata_path.exists()
    assert not result.data.empty


def test_library_build_alpha101_input_uses_adj_close_and_code_alignment(monkeypatch, tmp_path):
    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)

    def fake_fetch_price_data(**kwargs):
        return pd.DataFrame(
            {
                "date_": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "code": ["AAPL", "AAPL"],
                "open": [100.0, 110.0],
                "high": [102.0, 112.0],
                "low": [99.0, 109.0],
                "close": [101.0, 111.0],
                "adj_close": [50.5, 55.5],
                "volume": [1000.0, 2000.0],
                "dividend": [0.0, 0.1],
                "shares_outstanding": [10_000.0, 10_000.0],
            }
        )

    def fake_fetch_fundamental_data(**kwargs):
        return pd.DataFrame(
            {
                "date_": pd.to_datetime(["2024-01-01"]),
                "code": ["AAPL"],
                "sector": ["technology"],
                "industry": ["hardware"],
                "subindustry": ["devices"],
                "shares_basic": [9_000.0],
            }
        )

    monkeypatch.setattr(library, "fetch_price_data", fake_fetch_price_data)
    monkeypatch.setattr(library, "fetch_fundamental_data", fake_fetch_fundamental_data)

    alpha_input = library.build_alpha101_input(
        codes=["AAPL"],
        start="2024-01-02",
        end="2024-01-03",
        price_provider="simfin",
        classification_provider="simfin",
        classification_dataset="companies",
    )

    assert "adj_close" not in alpha_input.columns
    assert list(alpha_input["code"].unique()) == ["AAPL"]
    assert alpha_input.loc[alpha_input["date_"] == pd.Timestamp("2024-01-02"), "open"].iloc[0] == 50.0
    assert alpha_input.loc[alpha_input["date_"] == pd.Timestamp("2024-01-02"), "close"].iloc[0] == 50.5
    assert alpha_input.loc[alpha_input["date_"] == pd.Timestamp("2024-01-02"), "volume"].iloc[0] == 2000.0
    assert alpha_input.loc[alpha_input["date_"] == pd.Timestamp("2024-01-03"), "market_value"].iloc[0] == 555_000.0
    assert alpha_input.loc[alpha_input["date_"] == pd.Timestamp("2024-01-03"), "shares_outstanding"].iloc[0] == 10_000.0


def test_library_build_gtja191_input_can_include_fama3(monkeypatch, tmp_path):
    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)

    def fake_fetch_price_data(**kwargs):
        return _sample_price_rows()

    def fake_fetch_fundamental_data(**kwargs):
        name = kwargs.get("name")
        if name == "balance_sheet":
            return _sample_balance_rows()
        if name == "companies":
            return _sample_company_rows()
        return pd.DataFrame()

    monkeypatch.setattr(library, "fetch_price_data", fake_fetch_price_data)
    monkeypatch.setattr(library, "fetch_fundamental_data", fake_fetch_fundamental_data)

    gtja_input = library.build_gtja191_input(
        codes=["AAA", "BBB", "CCC", "DDD"],
        start="2024-02-01",
        end="2024-12-01",
        include_fama3=True,
        price_provider="simfin",
        fama_provider="simfin",
    )

    assert {"mkt", "smb", "hml"}.issubset(gtja_input.columns)
    assert gtja_input["mkt"].notna().any()
    assert gtja_input["smb"].notna().any()
    assert gtja_input["hml"].notna().any()


def test_library_build_sunday100plus_input_reuses_alpha101_layout(monkeypatch, tmp_path):
    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)

    monkeypatch.setattr(
        library,
        "fetch_price_data",
        lambda **kwargs: _sample_price_rows(),
    )
    monkeypatch.setattr(
        library,
        "fetch_fundamental_data",
        lambda **kwargs: _sample_company_rows(),
    )

    sunday_input = library.build_sunday100plus_input(
        codes=["AAA", "BBB", "CCC", "DDD"],
        start="2024-02-01",
        end="2024-12-01",
        price_provider="simfin",
        classification_provider="simfin",
        classification_dataset="companies",
    )

    assert {"date_", "code", "open", "high", "low", "close", "volume", "market_value"}.issubset(
        sunday_input.columns
    )
    assert sunday_input["code"].nunique() == 4


def test_library_save_and_load_fama3_panel(monkeypatch, tmp_path):
    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)

    monkeypatch.setattr(
        library,
        "fetch_price_data",
        lambda **kwargs: _sample_price_rows(),
    )
    monkeypatch.setattr(
        library,
        "fetch_fundamental_data",
        lambda **kwargs: _sample_company_rows() if kwargs.get("name") == "companies" else _sample_balance_rows(),
    )

    result = library.save_fama3(
        codes=["AAA", "BBB", "CCC", "DDD"],
        start="2024-02-01",
        end="2024-12-01",
        price_provider="simfin",
        fama_provider="simfin",
        force_updated=True,
    )

    assert result.dataset_dir.exists()
    assert result.manifest_path.exists()
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "mkt__fama.parquet").exists()
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "smb__fama.parquet").exists()
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "hml__fama.parquet").exists()
    loaded = library.load_fama3()
    assert {"date_", "mkt", "smb", "hml"}.issubset(loaded.columns)
    assert loaded["mkt"].notna().any()


def test_library_fama_factor_names_supports_short_and_long():
    library = TigerFactorLibrary(output_dir="/tmp/tiger_factors_test_output", verbose=False)

    assert library.fama_factor_names(3) == ("mkt", "smb", "hml")
    assert library.fama_factor_names("fama5", style="long") == (
        "market",
        "size",
        "value",
        "profitability",
        "investment",
    )
    assert library.fama_factor_names(6) == ("mkt", "smb", "hml", "rmw", "cma", "umd")
    assert library.fama_factor_names("6", style="long") == (
        "market",
        "size",
        "value",
        "profitability",
        "investment",
        "momentum",
    )


def test_library_gtja191_uses_cached_fama3_when_available(monkeypatch, tmp_path):
    library = TigerFactorLibrary(output_dir=tmp_path, verbose=False)
    cached_fama3 = pd.DataFrame(
        {
            "date_": pd.date_range("2024-01-01", periods=80, freq="D"),
            "mkt": np.linspace(0.001, 0.002, 80),
            "smb": np.linspace(0.0001, 0.0002, 80),
            "hml": np.linspace(-0.0002, 0.0001, 80),
        }
    )
    cached_fama3["date_"] = pd.to_datetime(cached_fama3["date_"])

    monkeypatch.setattr(library, "fetch_price_data", lambda **kwargs: _sample_price_rows())
    monkeypatch.setattr(library, "load_fama3", lambda **kwargs: cached_fama3)
    monkeypatch.setattr(
        library,
        "fetch_fundamental_data",
        lambda **kwargs: _sample_company_rows(),
    )

    result = library.gtja191(
        alpha_id=30,
        codes=["AAA", "BBB", "CCC", "DDD"],
        start="2024-02-01",
        end="2024-12-01",
        price_provider="simfin",
        use_cached_fama3=True,
    )

    assert result.name == "alpha_030"
    assert result.parquet_path.exists()
    assert not result.data.empty


class DemoAdapter(ProviderAdapter):
    def __init__(self) -> None:
        self.price_calls = 0
        self.fundamental_calls = 0
        self.dataset_calls = 0

    def fetch_price_data(self, *, provider, region, sec_type, freq, codes, start, end, as_ex=None):
        self.price_calls += 1
        return pd.DataFrame(
            {
                "date_": [pd.Timestamp("2024-01-02")],
                "code": [codes[0]],
                "close": [123.0],
            }
        )

    def fetch_fundamental_data(self, *, provider, name, region, sec_type, freq, variant=None, codes, start, end, as_ex=None):
        self.fundamental_calls += 1
        return pd.DataFrame(
            {
                "date_": [pd.Timestamp("2024-01-02")],
                "code": [codes[0]],
                "total_assets": [456.0],
            }
        )

    def fetch_dataset(self, *, provider, name, region, sec_type, freq, filters, as_ex=None):
        self.dataset_calls += 1
        return pd.DataFrame(
            {
                "symbol": ["aaa", "bbb", "aaa", None],
                "rank": [2, 1, 3, 4],
            }
        )


def test_library_can_use_custom_provider_adapter():
    adapter = DemoAdapter()
    library = TigerFactorLibrary(output_dir="/tmp/tiger_factors_test_output", verbose=False, provider_adapters={"demo_api": adapter})

    prices = library.fetch_price_data(
        codes=["AAA"],
        start="2024-01-01",
        end="2024-01-03",
        provider="demo_api",
    )
    fundamentals = library.fetch_fundamental_data(
        provider="demo_api",
        name="balance_sheet",
        freq="1q",
        codes=["AAA"],
        start="2024-01-01",
        end="2024-01-03",
    )

    assert adapter.price_calls == 1
    assert adapter.fundamental_calls == 1
    assert prices.loc[0, "close"] == 123.0
    assert fundamentals.loc[0, "total_assets"] == 456.0


def test_library_can_resolve_universe_codes_from_dataset():
    adapter = DemoAdapter()
    library = TigerFactorLibrary(output_dir="/tmp/tiger_factors_test_output", verbose=False, provider_adapters={"demo_api": adapter})

    codes = library.resolve_universe_codes(
        provider="demo_api",
        dataset="companies",
        code_column="symbol",
        sort_by="rank",
        ascending=True,
        limit=2,
    )

    assert adapter.dataset_calls == 1
    assert codes == ["BBB", "AAA"]


def test_library_aligns_fundamentals_one_session_late_without_point_in_time():
    library = TigerFactorLibrary(output_dir="/tmp/tiger_factors_test_output", verbose=False)
    fundamentals = pd.DataFrame(
        {
            "date_": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "code": ["AAA", "AAA"],
            "metric": [10.0, 20.0],
        }
    )
    trading_dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]))

    aligned = library.align_fundamental_to_trading_dates(
        fundamentals,
        trading_dates,
        value_columns=["metric"],
        use_point_in_time=False,
        lag_sessions=1,
    )["metric"]

    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-01"), "AAA"])
    assert aligned.loc[pd.Timestamp("2024-01-02"), "AAA"] == 10.0
    assert aligned.loc[pd.Timestamp("2024-01-03"), "AAA"] == 10.0
    assert aligned.loc[pd.Timestamp("2024-01-04"), "AAA"] == 20.0


def test_tiger_api_adapter_uses_simfin_eod_price_dataset(monkeypatch):
    adapter = TigerAPIAdapter()
    calls = []

    def fake_fetch(*args, **kwargs):
        calls.append(args)
        return pd.DataFrame()

    monkeypatch.setattr(adapter, "_fetch", fake_fetch)

    adapter.fetch_price_data(
        provider="simfin",
        region="us",
        sec_type="stock",
        freq="1d",
        codes=["AAPL"],
        start="2024-01-01",
        end="2024-01-02",
    )

    assert calls
    assert calls[0][1] == "eod_price"
