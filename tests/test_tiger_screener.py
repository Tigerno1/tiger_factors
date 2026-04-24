from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_screener import FactorScreener
from tiger_factors.factor_screener import CorrelationScreenerSpec
from tiger_factors.factor_screener import BacktestMarginalScreener
from tiger_factors.factor_screener import BacktestMarginalScreenerSpec
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener import ReturnAdapter
from tiger_factors.factor_screener import ReturnAdapterSpec
from tiger_factors.factor_screener import MarginalScreener
from tiger_factors.factor_screener import MarginalScreenerSpec
from tiger_factors.factor_screener import Screener


def _save_factor_artifacts(
    store: FactorStore,
    *,
    factor_name: str,
    group: str | None = None,
    summary_row: dict[str, object],
    factor_rows: list[dict[str, object]],
) -> None:
    spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name=factor_name,
        group=group,
    )
    store.save_factor(spec, pd.DataFrame(factor_rows), force_update=True)
    store.evaluation_store.save_evaluation(pd.DataFrame([summary_row]), spec=spec, force_updated=True)


def _save_return_artifacts(
    store: FactorStore,
    *,
    factor_name: str,
    group: str | None,
    dates: pd.DatetimeIndex,
    returns: list[float],
    long_only: list[float] | None = None,
) -> None:
    factor_spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name=factor_name,
        group=group,
    )
    store.evaluation_store.save_returns(
        pd.DataFrame(
            {
                "date_": dates,
                "long_short": returns,
                "long_only": long_only if long_only is not None else returns,
            }
        ),
        spec=factor_spec,
        table_name="factor_portfolio_returns",
        force_updated=True,
    )


def _save_ic_artifacts(
    store: FactorStore,
    *,
    factor_name: str,
    group: str | None,
    dates: pd.DatetimeIndex,
    ic_values: list[float],
) -> None:
    factor_spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name=factor_name,
        group=group,
    )
    store.evaluation_store.save_information(
        pd.DataFrame(
            {
                "date_": dates,
                "1D": ic_values,
            }
        ),
        spec=factor_spec,
        table_name="information_coefficient",
        force_updated=True,
    )


def test_tiger_screener_reads_summary_and_return_panel(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=3)
    codes = ["AAPL", "MSFT", "NVDA"]
    _save_factor_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        summary_row={
            "factor_name": "alpha_good",
            "ic_mean": 0.05,
            "rank_ic_mean": 0.04,
            "sharpe": 0.90,
            "turnover": 0.20,
            "fitness": 0.55,
            "ic_ir": 1.20,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        dates=dates,
        returns=[0.01, 0.02, 0.03],
    )
    _save_factor_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        summary_row={
            "factor_name": "alpha_bad",
            "ic_mean": 0.001,
            "rank_ic_mean": 0.001,
            "sharpe": 0.10,
            "turnover": 0.95,
            "fitness": 0.02,
            "ic_ir": 0.05,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(5 + date_idx - code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        dates=dates,
        returns=[-0.02, 0.00, 0.01],
    )
    _save_factor_artifacts(
        store,
        factor_name="alpha_sparse",
        group="core",
        summary_row={
            "factor_name": "alpha_sparse",
            "ic_mean": 0.04,
            "rank_ic_mean": 0.03,
            "sharpe": 0.80,
            "turnover": 0.25,
            "fitness": 0.50,
            "ic_ir": 1.10,
        },
        factor_rows=[
            {"date_": dates[0], "code": code, "value": float(2 + idx)}
            for idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_sparse",
        group="core",
        dates=dates[:1],
        returns=[0.05],
    )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_good", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_bad", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_sparse", group="core"),
    )
    screener = FactorScreener(
        FactorScreenerSpec(
            screening_config=FactorMetricFilterConfig(
                min_ic_mean=0.01,
                min_rank_ic_mean=0.01,
                min_sharpe=0.40,
                max_turnover=0.50,
                min_fitness=0.10,
            ),
        ),
        factor_specs=factor_specs,
        store=store,
    )

    result = screener.run()

    assert result.selected_factor_names == ["alpha_good"]
    assert set(result.rejected_factor_names) == {"alpha_bad", "alpha_sparse"}
    assert not result.return_panel.empty
    assert not result.return_long.empty
    assert "screened_at" in result.summary.columns
    assert "data_usable" in result.summary.columns
    assert "data_failed_rules" in result.summary.columns
    assert bool(result.summary.loc[result.summary["factor_name"] == "alpha_sparse", "data_usable"].iloc[0]) is False
    assert result.missing_return_factors == ()


def test_return_adapter_uses_evaluation_return_tables(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=3)
    _save_factor_artifacts(
        store,
        factor_name="alpha_one",
        group="core",
        summary_row={"factor_name": "alpha_one", "fitness": 0.50, "ic_mean": 0.03, "rank_ic_mean": 0.02, "sharpe": 0.80, "turnover": 0.20, "ic_ir": 1.0},
        factor_rows=[{"date_": date, "code": "AAPL", "value": float(i + 1)} for i, date in enumerate(dates)],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_one",
        group="core",
        dates=dates,
        returns=[0.10, 0.20, 0.30],
        long_only=[0.01, 0.02, 0.03],
    )
    _save_factor_artifacts(
        store,
        factor_name="alpha_two",
        group="core",
        summary_row={"factor_name": "alpha_two", "fitness": 0.40, "ic_mean": 0.02, "rank_ic_mean": 0.01, "sharpe": 0.60, "turnover": 0.30, "ic_ir": 0.8},
        factor_rows=[{"date_": date, "code": "MSFT", "value": float(i + 2)} for i, date in enumerate(dates)],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_two",
        group="core",
        dates=dates,
        returns=[0.05, 0.06, 0.07],
        long_only=[0.005, 0.006, 0.007],
    )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_one", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_two", group="core"),
    )
    result = ReturnAdapter(
        ReturnAdapterSpec(return_mode="long_only"),
        factor_specs=factor_specs,
        store=store,
    ).run()

    assert result.missing_return_factors == ()
    assert list(result.return_panel.columns) == ["alpha_one", "alpha_two"]
    assert result.return_panel.iloc[0, 0] == 0.01
    assert result.return_panel.iloc[0, 1] == 0.005
    assert not result.return_long.empty
    assert result.return_long["return_mode"].nunique() == 1
    assert result.return_long["return_mode"].iloc[0] == "long_only"


def test_tiger_screener_flow_summary_includes_group(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=3)
    codes = ["AAPL", "MSFT", "NVDA"]
    _save_factor_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        summary_row={
            "factor_name": "alpha_good",
            "ic_mean": 0.05,
            "rank_ic_mean": 0.04,
            "sharpe": 0.90,
            "turnover": 0.20,
            "fitness": 0.55,
            "ic_ir": 1.20,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        dates=dates,
        returns=[0.01, 0.02, 0.03],
    )

    result = FactorScreener(
        FactorScreenerSpec(
            screening_config=FactorMetricFilterConfig(
                min_ic_mean=0.01,
                min_rank_ic_mean=0.01,
                min_sharpe=0.40,
                max_turnover=0.50,
                min_fitness=0.10,
            ),
        ),
        factor_specs=(
            FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_good", group="core"),
        ),
        store=store,
    ).run()

    summary = result.to_summary()

    assert result.selected_factor_specs[0].group == "core"
    assert list(result.return_panel.columns) == ["alpha_good"]
    assert not result.return_long.empty
    assert summary["return_start"] is not None
    assert summary["return_end"] is not None


def test_total_screener_dispatches_factor_and_correlation_screeners(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=4)
    codes = ["AAPL", "MSFT", "NVDA"]
    for factor_name, fitness in (("alpha_a", 0.60), ("alpha_b", 0.50)):
        _save_factor_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            summary_row={
                "factor_name": factor_name,
                "ic_mean": 0.05,
                "rank_ic_mean": 0.04,
                "sharpe": 0.90,
                "turnover": 0.20,
                "fitness": fitness,
                "ic_ir": 1.20,
            },
            factor_rows=[
                {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
                for date_idx, date in enumerate(dates)
                for code_idx, code in enumerate(codes)
            ],
        )
        _save_return_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            dates=dates,
            returns=[0.01, 0.02, 0.03, 0.04],
        )
        _save_ic_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            dates=dates,
            ic_values=[0.02, 0.01, -0.01, 0.03],
        )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_a", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_b", group="core"),
    )
    result = Screener(
        FactorScreenerSpec(
            screening_config=FactorMetricFilterConfig(
                min_ic_mean=0.01,
                min_rank_ic_mean=0.01,
                min_sharpe=0.40,
                max_turnover=0.50,
                min_fitness=0.10,
            ),
        ),
        (
            CorrelationScreenerSpec(
                evaluation_source="factor",
                method="greedy",
                threshold=0.75,
                score_field="fitness",
            ),
            CorrelationScreenerSpec(
                evaluation_source="ic",
                method="greedy",
                threshold=0.75,
                score_field="fitness",
            ),
        ),
        factor_specs=factor_specs,
        store=store,
    ).run()

    assert result.factor_selected_factor_names == ["alpha_a", "alpha_b"]
    assert result.selected_factor_names == ["alpha_a"]
    assert len(result.correlation_results) == 2
    assert [spec.table_name for spec in result.selected_factor_specs] == ["alpha_a"]
    assert list(result.return_panel.columns) == ["alpha_a"]

    adapter_result = result.build_return_adapter(
        store=store,
        spec=ReturnAdapterSpec(return_mode="long_short"),
    ).run()
    assert list(adapter_result.return_panel.columns) == ["alpha_a"]
    assert not adapter_result.return_long.empty


def test_marginal_screener_keeps_only_improving_factors(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=4)
    codes = ["AAPL", "MSFT", "NVDA"]
    _save_factor_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        summary_row={
            "factor_name": "alpha_good",
            "ic_mean": 0.08,
            "rank_ic_mean": 0.06,
            "ic_ir": 1.40,
            "sharpe": 1.10,
            "turnover": 0.18,
            "fitness": 0.70,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        dates=dates,
        returns=[0.02, 0.03, 0.01, 0.04],
    )
    _save_ic_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        dates=dates,
        ic_values=[0.04, 0.03, 0.05, 0.02],
    )
    _save_factor_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        summary_row={
            "factor_name": "alpha_bad",
            "ic_mean": -0.02,
            "rank_ic_mean": -0.01,
            "ic_ir": -0.30,
            "sharpe": -0.20,
            "turnover": 0.92,
            "fitness": -0.10,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(5 + date_idx - code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        dates=dates,
        returns=[-0.03, -0.02, 0.00, -0.01],
    )
    _save_ic_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        dates=dates,
        ic_values=[-0.03, -0.02, 0.00, -0.01],
    )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_good", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_bad", group="core"),
    )
    result = MarginalScreener(
        MarginalScreenerSpec(),
        factor_specs=factor_specs,
        store=store,
    ).run()

    assert result.selected_factor_names == ["alpha_good"]
    assert set(result.rejected_factor_names) == {"alpha_bad"}
    assert list(result.return_panel.columns) == ["alpha_good"]
    assert not result.portfolio_returns.empty
    assert not result.return_long.empty


def test_total_screener_dispatches_marginal_mode(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=4)
    codes = ["AAPL", "MSFT", "NVDA"]
    for factor_name, summary_row, returns in (
        (
            "alpha_good",
            {
                "factor_name": "alpha_good",
                "ic_mean": 0.08,
                "rank_ic_mean": 0.06,
                "ic_ir": 1.40,
                "sharpe": 1.10,
                "turnover": 0.18,
                "fitness": 0.70,
            },
            [0.02, 0.03, 0.01, 0.04],
        ),
        (
            "alpha_bad",
            {
                "factor_name": "alpha_bad",
                "ic_mean": -0.02,
                "rank_ic_mean": -0.01,
                "ic_ir": -0.30,
                "sharpe": -0.20,
                "turnover": 0.92,
                "fitness": -0.10,
            },
            [-0.03, -0.02, 0.00, -0.01],
        ),
    ):
        _save_factor_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            summary_row=summary_row,
            factor_rows=[
                {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
                for date_idx, date in enumerate(dates)
                for code_idx, code in enumerate(codes)
            ],
        )
        _save_return_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            dates=dates,
            returns=returns,
        )
        _save_ic_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            dates=dates,
            ic_values=returns,
        )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_good", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_bad", group="core"),
    )
    result = Screener(
        FactorScreenerSpec(
            screening_config=FactorMetricFilterConfig(
                min_ic_mean=-1.0,
                min_rank_ic_mean=-1.0,
                min_sharpe=-1.0,
                max_turnover=1.0,
                min_fitness=-1.0,
            ),
        ),
        (),
        factor_specs=factor_specs,
        store=store,
        mode="marginal",
        marginal_spec=MarginalScreenerSpec(),
    ).run()

    assert result.factor_selected_factor_names == ["alpha_good", "alpha_bad"]
    assert result.selected_factor_names == ["alpha_good"]
    assert result.marginal_screener.selected_factor_names == ["alpha_good"]
    assert list(result.return_panel.columns) == ["alpha_good"]


def test_backtest_marginal_screener_keeps_only_improving_factors(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=4)
    codes = ["AAPL", "MSFT", "NVDA"]
    _save_factor_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        summary_row={
            "factor_name": "alpha_good",
            "ic_mean": 0.08,
            "rank_ic_mean": 0.06,
            "ic_ir": 1.40,
            "sharpe": 1.10,
            "turnover": 0.18,
            "fitness": 0.70,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        dates=dates,
        returns=[0.02, 0.03, 0.01, 0.04],
    )
    _save_ic_artifacts(
        store,
        factor_name="alpha_good",
        group="core",
        dates=dates,
        ic_values=[0.04, 0.03, 0.05, 0.02],
    )
    _save_factor_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        summary_row={
            "factor_name": "alpha_bad",
            "ic_mean": -0.02,
            "rank_ic_mean": -0.01,
            "ic_ir": -0.30,
            "sharpe": -0.20,
            "turnover": 0.92,
            "fitness": -0.10,
        },
        factor_rows=[
            {"date_": date, "code": code, "value": float(5 + date_idx - code_idx)}
            for date_idx, date in enumerate(dates)
            for code_idx, code in enumerate(codes)
        ],
    )
    _save_return_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        dates=dates,
        returns=[-0.03, -0.02, 0.00, -0.01],
    )
    _save_ic_artifacts(
        store,
        factor_name="alpha_bad",
        group="core",
        dates=dates,
        ic_values=[-0.03, -0.02, 0.00, -0.01],
    )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_good", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_bad", group="core"),
    )
    result = BacktestMarginalScreener(
        BacktestMarginalScreenerSpec(),
        factor_specs=factor_specs,
        store=store,
    ).run()

    assert result.selected_factor_names == ["alpha_good"]
    assert set(result.rejected_factor_names) == {"alpha_bad"}
    assert list(result.return_panel.columns) == ["alpha_good"]
    assert not result.portfolio_returns.empty
    assert "portfolio" in result.stats


def test_total_screener_dispatches_backtest_marginal_mode(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=4)
    codes = ["AAPL", "MSFT", "NVDA"]
    for factor_name, summary_row, returns in (
        (
            "alpha_good",
            {
                "factor_name": "alpha_good",
                "ic_mean": 0.08,
                "rank_ic_mean": 0.06,
                "ic_ir": 1.40,
                "sharpe": 1.10,
                "turnover": 0.18,
                "fitness": 0.70,
            },
            [0.02, 0.03, 0.01, 0.04],
        ),
        (
            "alpha_bad",
            {
                "factor_name": "alpha_bad",
                "ic_mean": -0.02,
                "rank_ic_mean": -0.01,
                "ic_ir": -0.30,
                "sharpe": -0.20,
                "turnover": 0.92,
                "fitness": -0.10,
            },
            [-0.03, -0.02, 0.00, -0.01],
        ),
    ):
        _save_factor_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            summary_row=summary_row,
            factor_rows=[
                {"date_": date, "code": code, "value": float(10 + date_idx + code_idx)}
                for date_idx, date in enumerate(dates)
                for code_idx, code in enumerate(codes)
            ],
        )
        _save_return_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            dates=dates,
            returns=returns,
        )
        _save_ic_artifacts(
            store,
            factor_name=factor_name,
            group="core",
            dates=dates,
            ic_values=returns,
        )

    factor_specs = (
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_good", group="core"),
        FactorSpec(provider="tiger", region="us", sec_type="stock", freq="1d", table_name="alpha_bad", group="core"),
    )
    result = Screener(
        FactorScreenerSpec(
            screening_config=FactorMetricFilterConfig(
                min_ic_mean=-1.0,
                min_rank_ic_mean=-1.0,
                min_sharpe=-1.0,
                max_turnover=1.0,
                min_fitness=-1.0,
            ),
        ),
        (),
        factor_specs=factor_specs,
        store=store,
        mode="backtest_marginal",
        backtest_marginal_spec=BacktestMarginalScreenerSpec(),
    ).run()

    assert result.factor_selected_factor_names == ["alpha_good", "alpha_bad"]
    assert result.selected_factor_names == ["alpha_good"]
    assert result.backtest_marginal_screener.selected_factor_names == ["alpha_good"]
    assert list(result.return_panel.columns) == ["alpha_good"]
