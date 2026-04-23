from __future__ import annotations

from pathlib import Path

import pandas as pd

from tiger_factors.multifactor_evaluation.evaluation import MultifactorEvaluation
from tiger_factors.multifactor_evaluation.reporting.summary_html import render_summary_html


class _FakeResult:
    def __init__(self, output_dir: Path, report_name: str, table_order: list[str]) -> None:
        self.output_dir = output_dir
        self.report_name = report_name
        self.figure_paths: list[Path] = []
        self._table_order = table_order
        self._tables = {name: pd.DataFrame({"value": [name]}) for name in table_order}

    def get_report(self, *, open_browser: bool = True) -> Path:
        path = self.output_dir / f"{self.report_name}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>", encoding="utf-8")
        return path

    def tables(self) -> list[str]:
        return sorted(self._tables)

    def ordered_tables(self) -> list[str]:
        return list(self._table_order)

    def get_table(self, table_name: str) -> pd.DataFrame:
        return self._tables[table_name]

    def to_summary(self) -> dict[str, str]:
        return {"report_name": self.report_name}


def test_portfolio_respects_column_overrides(monkeypatch, tmp_path) -> None:
    captured: dict[str, pd.Series | None] = {}

    def _fake_create_portfolio_tear_sheet(
        portfolio_returns,
        *,
        benchmark_returns=None,
        **kwargs,
    ):
        captured["portfolio_returns"] = portfolio_returns
        captured["benchmark_returns"] = benchmark_returns
        return _FakeResult(kwargs["output_dir"], kwargs["report_name"], ["summary"])

    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.evaluation._create_portfolio_tear_sheet",
        _fake_create_portfolio_tear_sheet,
    )

    index = pd.date_range("2024-01-01", periods=3, freq="D")
    backtest = pd.DataFrame(
        {
            "custom_portfolio": [0.01, 0.02, -0.01],
            "custom_benchmark": [0.0, 0.01, -0.005],
        },
        index=index,
    )
    evaluation = MultifactorEvaluation(output_dir=tmp_path)

    evaluation.portfolio(
        backtest,
        portfolio_column="custom_portfolio",
        benchmark_column="custom_benchmark",
    )

    assert captured["portfolio_returns"] is not None
    assert captured["benchmark_returns"] is not None
    pd.testing.assert_series_equal(captured["portfolio_returns"], backtest["custom_portfolio"], check_names=True)
    pd.testing.assert_series_equal(captured["benchmark_returns"], backtest["custom_benchmark"], check_names=True)


def test_full_uses_backtest_for_trades_and_preserves_ordered_tables(monkeypatch, tmp_path) -> None:
    trade_calls: list[pd.Series] = []
    captured_modules: list[dict[str, object]] = []

    def _fake_summary(backtest, **kwargs):
        return _FakeResult(kwargs["output_dir"], kwargs["report_name"], ["summary", "drawdown"])

    def _fake_trades(returns, **kwargs):
        trade_calls.append(returns)
        return _FakeResult(
            kwargs["output_dir"],
            kwargs["report_name"],
            ["summary", "transaction_summary", "transactions"],
        )

    def _fake_portfolio(portfolio_returns, **kwargs):
        return _FakeResult(
            kwargs["output_dir"],
            kwargs["report_name"],
            ["summary", "portfolio_returns", "transactions"],
        )

    def _fake_render_multifactor_index_report(*, modules, **kwargs):
        captured_modules.extend(modules)
        path = kwargs["output_dir"] / f"{kwargs['report_name']}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>", encoding="utf-8")
        return path

    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.evaluation._create_summary_tear_sheet",
        _fake_summary,
    )
    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.evaluation._create_trade_tear_sheet",
        _fake_trades,
    )
    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.evaluation._create_portfolio_tear_sheet",
        _fake_portfolio,
    )
    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.evaluation.render_multifactor_index_report",
        _fake_render_multifactor_index_report,
    )

    index = pd.date_range("2024-01-01", periods=3, freq="D")
    backtest = pd.DataFrame(
        {
            "portfolio": [0.01, -0.02, 0.03],
            "benchmark": [0.005, -0.01, 0.01],
        },
        index=index,
    )
    evaluation = MultifactorEvaluation(output_dir=tmp_path, backtest=None)

    bundle = evaluation.full(backtest=backtest, report_name="demo")

    assert bundle.trade_result is not None
    assert len(trade_calls) == 1
    pd.testing.assert_series_equal(trade_calls[0], backtest["portfolio"], check_names=True)

    trade_module = next(module for module in captured_modules if module["name"] == "trades")
    assert list(trade_module["tables"].keys()) == ["summary", "transaction_summary", "transactions"]


def test_summary_html_wrapper_still_renders_from_shared_renderer(tmp_path) -> None:
    summary_table = pd.DataFrame({"value": [1.23]}, index=["sharpe"])
    figure_path = tmp_path / "demo_snapshot.png"
    figure_path.write_bytes(b"png")

    html = render_summary_html(
        summary_table,
        [figure_path],
        "demo",
        title="Multifactor Tearsheet",
        date_range="2024-01-01 - 2024-01-31",
    )

    assert "Multifactor Tearsheet" in html
    assert "Key Performance Metrics" in html
