from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.examples.factor_store_multi_factor_reporting import save_factor_backtest_plot
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest_from_store
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


def test_factor_store_multi_factor_backtest_demo(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    spec_kwargs = dict(provider="tiger", region="us", sec_type="stock", freq="1d", variant=None)
    factor_names = ["BM", "FSCORE", "BMFSCORE"]
    dates = pd.bdate_range("2024-01-02", periods=36)
    codes = ["AAPL", "MSFT", "NVDA"]

    for offset, factor_name in enumerate(factor_names):
        rows: list[dict[str, object]] = []
        for date_idx, date in enumerate(dates):
            for code_idx, code in enumerate(codes):
                rows.append(
                    {
                        "code": code,
                        "date_": date,
                        "value": float((offset + 1) * 5 + date_idx + code_idx),
                    }
                )
        store.save_factor(
            FactorSpec(table_name=factor_name, **spec_kwargs),
            pd.DataFrame(rows),
            force_update=True,
        )

    price_rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            price_rows.append(
                {
                    "code": code,
                    "date_": date,
                    "close": 100.0 * (1.004 + 0.001 * code_idx) ** date_idx,
                }
            )

    library = TigerFactorLibrary(
        output_dir=tmp_path,
        verbose=False,
        provider_adapters={"yahoo": _StaticPriceAdapter(pd.DataFrame(price_rows))},
    )

    factor_frame = library.load_factor_frame(factor_names=factor_names, provider="tiger")
    panels = library.load_factor_panels(factor_names=factor_names, provider="tiger")
    close_panel = library.price_panel(
        codes=codes,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        provider="yahoo",
        field="close",
    )

    assert not factor_frame.empty
    assert set(panels) == set(factor_names)
    assert not close_panel.empty

    result = multi_factor_backtest(
        panels,
        close_panel,
        weights={"BM": 0.5, "FSCORE": 0.3, "BMFSCORE": 0.2},
        standardize=True,
        rebalance_freq="ME",
        long_pct=0.5,
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert not result["composite_factor"].empty
    assert result["weights"] == {"BM": 0.5, "FSCORE": 0.3, "BMFSCORE": 0.2}
    assert not result["backtest"].empty
    assert result["backtest"]["portfolio"].notna().any()

    store_result = multi_factor_backtest_from_store(
        library,
        factor_names,
        provider="tiger",
        price_provider="yahoo",
        weights={"BM": 0.5, "FSCORE": 0.3, "BMFSCORE": 0.2},
        rebalance_freq="ME",
        long_pct=0.5,
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    assert not store_result["factor_frame"].empty
    assert set(store_result["factor_panels"]) == set(factor_names)
    assert not store_result["backtest"].empty

    report = run_portfolio_from_backtest(
        result["backtest"],
        output_dir=tmp_path / "report",
        report_name="factor_store_multi_factor_backtest",
    )
    assert report is not None

    figure_path = save_factor_backtest_plot(
        result["backtest"],
        output_dir=tmp_path / "report",
        report_name="factor_store_multi_factor_backtest",
    )
    assert figure_path is not None
    assert figure_path.exists()
