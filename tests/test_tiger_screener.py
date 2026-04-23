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
    summary_row: dict[str, object],
    factor_rows: list[dict[str, object]],
) -> None:
    spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name=factor_name,
    )
    store.save_factor(spec, pd.DataFrame(factor_rows), force_update=True)
    store.evaluation_store.save_evaluation(pd.DataFrame([summary_row]), spec=spec, force_updated=True)


def test_tiger_screener_reads_summary_and_return_panel(tmp_path: Path) -> None:
    store = FactorStore(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=3)
    codes = ["AAPL", "MSFT", "NVDA"]
    price_panel = pd.DataFrame(
        {
            code: [100.0 + idx * 2 + offset for offset in range(len(dates))]
            for idx, code in enumerate(codes)
        },
        index=dates,
    )

    _save_factor_artifacts(
        store,
        factor_name="alpha_good",
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
    _save_factor_artifacts(
        store,
        factor_name="alpha_bad",
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

    screener = FactorScreener(
        FactorScreenerSpec(
            factor_names=("alpha_good", "alpha_bad"),
            price_panel=price_panel,
            screening_config=FactorMetricFilterConfig(
                min_ic_mean=0.01,
                min_rank_ic_mean=0.01,
                min_sharpe=0.40,
                max_turnover=0.50,
                min_fitness=0.10,
            ),
        ),
        store=store,
    )

    result = screener.run()

    assert result.selected_factor_names == ["alpha_good"]
    assert result.rejected_factor_names == ["alpha_bad"]
    assert list(result.return_panel.columns) == ["alpha_good"]
    assert result.return_panel.index.min() == pd.Timestamp("2024-01-02")
    assert result.return_panel.index.max() == pd.Timestamp("2024-01-04")
    assert set(result.return_long["return_mode"].unique()) == {"long_short", "long_only"}
    assert list(result.return_long.columns) == ["date_", "factor", "return", "return_mode"]
    assert set(result.return_long["factor"].unique()) == {"alpha_good"}
    assert "screened_at" in result.summary.columns
    assert "return_start" in result.summary.columns
    assert "return_end" in result.summary.columns
    assert result.missing_return_factors == ()
