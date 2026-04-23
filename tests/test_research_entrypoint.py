from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.evaluation import SingleFactorEvaluation
from tiger_factors.factor_store import FactorSpec
import tiger_factors.factor_store.evaluation_store as evaluation_store_module


def _make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=90)
    codes = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META"]
    rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    base_levels = dict(zip(codes, np.linspace(80.0, 160.0, len(codes))))

    for code in codes:
        level = base_levels[code]
        shocks = rng.normal(0.0008, 0.02, size=len(dates))
        closes = level * np.cumprod(1.0 + shocks)
        factor_vals = pd.Series(closes).pct_change(5).rolling(3, min_periods=1).mean().to_numpy()
        for date, close, factor in zip(dates, closes, factor_vals, strict=False):
            price_rows.append({"date_": date, "code": code, "close": float(close)})
            rows.append({"date_": date, "code": code, "alpha_test": float(factor) if np.isfinite(factor) else np.nan})

    factor_frame = pd.DataFrame(rows)
    price_frame = pd.DataFrame(price_rows)
    return factor_frame, price_frame


def _use_temp_evaluation_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluation_store_module, "DEFAULT_EVALUATION_ROOT", tmp_path)


def test_single_factor_research_entrypoint(tmp_path, monkeypatch) -> None:
    _use_temp_evaluation_root(monkeypatch, tmp_path)
    factor_frame, price_frame = _make_frames()
    research = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_test",
        include_horizon=False,
    )

    evaluation = research.evaluate()
    assert evaluation is not None
    assert np.isfinite(evaluation.ic_mean)

    result = research.run(include_native_report=False)
    assert result.report is not None
    assert result.report_bundle is not None
    assert result.report.output_dir.exists()
    assert result.output_dir is not None


def test_single_factor_evaluation_open_and_summary_accessor(tmp_path, monkeypatch) -> None:
    _use_temp_evaluation_root(monkeypatch, tmp_path)
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

    evaluation = SingleFactorEvaluation.open(spec=spec)
    loaded = evaluation.summary().get_table()

    assert loaded.equals(expected)


def test_single_factor_evaluation_extensions(tmp_path, monkeypatch) -> None:
    _use_temp_evaluation_root(monkeypatch, tmp_path)
    factor_frame, price_frame = _make_frames()
    evaluation = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_test",
        include_horizon=False,
    )

    exposure = evaluation.exposure()
    assert "latest_date" in exposure
    assert "latest_cross_section" in exposure

    attribution = evaluation.attribution()
    assert "factor_contribution" in attribution
    assert "return_decomposition" in attribution

    monitoring = evaluation.monitoring()
    assert not monitoring.empty
    assert "rolling_ic_mean" in monitoring.columns

    effectiveness = evaluation.effectiveness()
    assert "passed" in effectiveness
    assert "checks" in effectiveness


def test_single_factor_section_formats_follow_section_rule(tmp_path, monkeypatch) -> None:
    _use_temp_evaluation_root(monkeypatch, tmp_path)
    factor_frame, price_frame = _make_frames()
    spec = FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="alpha_test",
    )
    evaluation = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_test",
        spec=spec,
    )

    root = tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_test"
    evaluation.summary(format="report")
    assert not (root / "summary" / "report.html").exists()
    assert (root / "summary" / "summary.parquet").exists()
    assert not any((root / "summary").glob("*.png"))

    evaluation.horizon(format="report")
    assert (root / "horizon" / "report.html").exists()
    assert (root / "horizon" / "horizon_result.png").exists()
    assert not (root / "horizon" / "horizon_result.parquet").exists()
    assert not (root / "horizon" / "horizon_summary.parquet").exists()

    evaluation.full(format="img")
    assert any(path.suffix == ".png" for path in (root / "returns").iterdir())
    assert any(path.suffix == ".png" for path in (root / "information").iterdir())
    assert any(path.suffix == ".png" for path in (root / "turnover").iterdir())
    assert any(path.suffix == ".png" for path in (root / "event_returns").iterdir())
    assert not (root / "report.html").exists()

    evaluation.full(format="all", force_updated=True)
    assert (root / "report.html").exists()
    assert (root / "summary" / "summary.parquet").exists()
    assert any(path.suffix == ".parquet" for path in (root / "returns").iterdir())
    assert any(path.suffix == ".parquet" for path in (root / "information").iterdir())
    assert any(path.suffix == ".parquet" for path in (root / "turnover").iterdir())
    assert any(path.suffix == ".parquet" for path in (root / "event_returns").iterdir())
    assert (root / "returns" / "report.html").exists()
    assert (root / "information" / "report.html").exists()
    assert (root / "turnover" / "report.html").exists()
    assert (root / "event_returns" / "report.html").exists()
    report_html = (root / "report.html").read_text(encoding="utf-8")
    assert "./returns/" in report_html
