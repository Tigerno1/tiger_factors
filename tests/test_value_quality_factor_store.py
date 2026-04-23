from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.examples.value_quality_reporting import save_value_quality_backtest_plot
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation import run_value_quality_long_short_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


class _StaticPriceAdapter:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.copy()

    def fetch_price_data(
        self,
        *,
        provider: str,
        region: str,
        sec_type: str,
        freq: str,
        codes: list[str],
        start: str,
        end: str,
        as_ex: bool | None = None,
    ) -> pd.DataFrame:
        frame = self.frame.copy()
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame["code"] = frame["code"].astype(str)
        frame = frame.loc[frame["code"].isin([str(code) for code in codes])]
        frame = frame.loc[(frame["date_"] >= pd.Timestamp(start)) & (frame["date_"] <= pd.Timestamp(end))]
        return frame.sort_values(["date_", "code"], kind="stable").reset_index(drop=True)

    def fetch_fundamental_data(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_dataset(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


def test_value_quality_factor_store_roundtrip_backtest(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec_kwargs = dict(provider="tiger", region="us", sec_type="stock", freq="1d", variant=None)
    bm_spec = FactorSpec(table_name="bm", **spec_kwargs)
    fscore_spec = FactorSpec(table_name="fscore", **spec_kwargs)

    dates = pd.bdate_range("2024-01-02", periods=40)
    bm_rows: list[dict[str, object]] = []
    fscore_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        bm_rows.extend(
            [
                {"code": "AAPL", "date_": date, "value": 1.8},
                {"code": "MSFT", "date_": date, "value": 0.4},
            ]
        )
        fscore_rows.extend(
            [
                {"code": "AAPL", "date_": date, "value": 8.0},
                {"code": "MSFT", "date_": date, "value": 1.0},
            ]
        )
        price_rows.extend(
            [
                {"code": "AAPL", "date_": date, "close": 100.0 * (1.01**date_idx)},
                {"code": "MSFT", "date_": date, "close": 100.0 * (1.002**date_idx)},
            ]
        )

    store.save_factor(bm_spec, pd.DataFrame(bm_rows), force_updated=True)
    store.save_factor(fscore_spec, pd.DataFrame(fscore_rows), force_updated=True)

    library = TigerFactorLibrary(
        output_dir=tmp_path,
        verbose=False,
        provider_adapters={"yahoo": _StaticPriceAdapter(pd.DataFrame(price_rows))},
    )

    bm_panel = library.load_factor_panel(factor_name="BM", provider="tiger")
    fscore_panel = library.load_factor_panel(factor_name="FSCORE", provider="tiger")
    factor_frame = library.load_factor_frame(
        factor_names=["BM", "FSCORE"],
        provider="tiger",
    )

    assert bm_panel.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 1.8
    assert fscore_panel.loc[pd.Timestamp("2024-01-02"), "MSFT"] == 1.0
    assert {"date_", "code", "BM", "FSCORE"}.issubset(factor_frame.columns)

    close_panel = library.price_panel(
        codes=["AAPL", "MSFT"],
        start="2024-01-02",
        end=str(dates[-1].date()),
        provider="yahoo",
        field="close",
    )
    assert not close_panel.empty

    result = run_value_quality_long_short_backtest(
        factor_frame,
        close_panel,
        bm_column="BM",
        fscore_column="FSCORE",
        high_quantile=0.5,
        long_pct=0.5,
        rebalance_freq="ME",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert not result["combo_frame"].empty
    assert {"HH", "LL"}.issubset(set(result["combo_frame"]["value_quality_bucket"].dropna().unique()))
    assert not result["backtest"].empty
    assert result["backtest"]["portfolio"].notna().any()

    report = run_portfolio_from_backtest(
        result["backtest"],
        output_dir=tmp_path / "report",
        report_name="value_quality_factor_store",
    )
    assert report is not None

    figure_path = save_value_quality_backtest_plot(
        result["backtest"],
        output_dir=tmp_path / "report",
        report_name="value_quality_factor_store",
    )
    assert figure_path is not None
    assert figure_path.exists()
