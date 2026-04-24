from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_screener import FactorScreener
from tiger_factors.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener import FactorMetricFilterConfig


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
            }
        ),
        spec=factor_spec,
        table_name="factor_portfolio_returns",
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
            selection_threshold=0.75,
            selection_score_field="fitness",
            correlation_method="graph",
            ic_correlation_method="average",
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

    assert summary["correlation_method"] == "graph"
    assert summary["ic_correlation_method"] == "average"
    assert result.selected_factor_specs[0].group == "core"
    assert list(result.return_panel.columns) == ["alpha_good"]
    assert not result.return_long.empty
    assert summary["return_start"] is not None
    assert summary["return_end"] is not None
