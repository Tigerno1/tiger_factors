from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_evaluation import get_clean_factor_and_forward_returns
from tiger_factors.factor_evaluation.engine import ReportBundleSummary
from tiger_factors.factor_evaluation import select_best_holding_period_from_event_returns
from tiger_factors.factor_evaluation import summarize_best_horizon
from tiger_factors.factor_evaluation.input import clear_group_labels_cache
from tiger_factors.factor_evaluation.input import load_group_labels
from tiger_factors.factor_evaluation.input import prewarm_group_labels_cache
import tiger_factors.factor_evaluation.input as input_module
import tiger_factors.factor_store.store as factor_store_store
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import FactorSpec
from tiger_factors.report_paths import report_output_root_for
from tiger_factors.factor_evaluation.plotting import plot_top_bottom_quantile_turnover
from tiger_factors.factor_evaluation.utils import quantize_factor


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    factor = pd.DataFrame(
        {
            "date_": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "code": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "alpha_001": [1.0, 2.0, 1.2, 1.8, 1.3, 1.7, 1.4, 1.6],
            "alpha_002": [2.0, 1.0, 1.8, 1.2, 1.7, 1.3, 1.6, 1.4],
        }
    )
    price = pd.DataFrame(
        {
            "date_": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
            ],
            "code": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [10.0, 20.0, 10.5, 20.5, 11.0, 21.0, 11.2, 21.5, 11.5, 22.0],
        }
    )
    return factor, price


def _sample_frames_many_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    codes = [f"C{i:02d}" for i in range(12)]

    factor_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []

    ranks = {code: i + 1 for i, code in enumerate(codes)}
    base_prices = {code: 10.0 + i for i, code in enumerate(codes)}

    price_state = base_prices.copy()
    for date_idx, dt in enumerate(dates):
        for code in codes:
            rank = ranks[code]
            factor_value = float(rank + date_idx * 0.05)
            factor_rows.append({"date_": dt, "code": code, "alpha_001": factor_value})

            daily_return = 0.001 + 0.0007 * rank
            price_rows.append({"date_": dt, "code": code, "close": price_state[code]})
            price_state[code] = price_state[code] * (1.0 + daily_return)

    return pd.DataFrame(factor_rows), pd.DataFrame(price_rows)


def test_get_clean_factor_and_forward_returns_works() -> None:
    factor, price = _sample_frames()
    result = get_clean_factor_and_forward_returns(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        periods=(1,),
        quantiles=2,
    )

    assert not result.factor_data.empty
    assert "1D" in result.factor_data.columns
    assert "factor_quantile" in result.factor_data.columns


def test_tiger_factor_evaluation_creates_reports(tmp_path) -> None:
    factor, price = _sample_frames()
    groups = pd.Series({"AAA": "tech", "BBB": "fin"})
    engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        group_labels=groups,
        factor_store=FactorStore(root_dir=tmp_path),
    )

    summary = engine.evaluate()
    assert summary.ic_mean == summary.ic_mean

    report = engine.full(periods=(1,), quantiles=2)
    assert report.output_dir.exists()
    assert report.figure_paths
    assert report.table_paths
    assert (report.output_dir / "manifest.json").exists()
    assert report.figure_output_dir is not None
    assert "signal_overview" not in report.figure_paths
    assert any(path.suffix == ".png" for path in report.figure_output_dir.glob("*.png"))
    assert any("ic_histogram" in key for key in report.figure_paths)
    assert sorted(path.name for path in (report.output_dir / "summary").glob("*.parquet")) == ["summary.parquet"]
    assert "horizon_result" in report.table_paths
    assert "horizon_summary" in report.table_paths
    assert "horizon_manifest" in report.table_paths

    full_manifest = json.loads((report.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert full_manifest["factor_column"] == "alpha_001"
    assert full_manifest["includes_horizon"] is True
    assert full_manifest["horizon_config"]["horizons"] == [1, 3, 5, 10, 20]
    assert full_manifest["factor_rows"] == len(factor)

    bundle_summary = engine.create_report_bundle_summary(report)
    assert isinstance(bundle_summary, ReportBundleSummary)
    assert bundle_summary.factor_column == "alpha_001"
    assert bundle_summary.figure_count >= 1
    assert bundle_summary.table_count >= 1
    assert bundle_summary.horizon_summary is not None

    bundle_summary_from_disk = engine.create_report_bundle_summary(output_dir=report.output_dir)
    assert isinstance(bundle_summary_from_disk, ReportBundleSummary)
    assert bundle_summary_from_disk.factor_column == "alpha_001"
    assert bundle_summary_from_disk.full_manifest.endswith("manifest.json")

    event_report = engine.event_returns(periods=(1,), quantiles=2)
    assert event_report.output_dir.exists()
    assert event_report.table_paths
    assert not event_report.figure_paths
    assert "best_holding_period" in event_report.table_paths
    assert "best_holding_period" in event_report.payload
    assert event_report.payload["best_holding_period"]["direction"] in {"use_as_is", "reverse_factor", "unknown"}
    assert (report.figure_output_dir / "returns_overview.png").exists()

    summary_frame = engine.summary(save=False)
    assert isinstance(summary_frame, pd.DataFrame)
    assert len(summary_frame) == 1

    summary_saved = engine.summary(save=True)
    assert isinstance(summary_saved, pd.DataFrame)
    assert (report_output_root_for("factor_evaluation", "alpha_001") / "summary" / "summary.parquet").exists()

    returns_report = engine.returns(periods=(1,), quantiles=2)
    assert returns_report.output_dir.name == "returns"
    assert any("long_only" in key for key in returns_report.table_paths)
    assert any("long_short" in key for key in returns_report.table_paths)
    assert "factor_portfolio_returns" in returns_report.table_paths
    returns_table = pd.read_parquet(returns_report.table_paths["factor_portfolio_returns"])
    assert list(returns_table.columns) == ["long_short", "long_only"] or list(returns_table.columns) == ["long_only", "long_short"]
    assert returns_table.index.name == "date_"

    full_alias = engine.full()
    assert full_alias.output_dir.exists()
    assert "signal_overview" not in full_alias.figure_paths
    assert any("long_only" in key for key in full_alias.table_paths)
    assert any("long_short" in key for key in full_alias.table_paths)
    assert any(key.endswith("factor_portfolio_returns") for key in full_alias.table_paths)


def test_tiger_factor_evaluation_supports_alpha002(tmp_path) -> None:
    factor, price = _sample_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_002",
        factor_store=FactorStore(root_dir=tmp_path),
    )

    summary = engine.evaluate()
    assert summary.ic_mean == summary.ic_mean

    report = engine.full(periods=(1,), quantiles=2)
    assert report.output_dir.exists()
    assert (report.output_dir / "manifest.json").exists()

    manifest = json.loads((report.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["factor_column"] == "alpha_002"


def test_tiger_factor_evaluation_uses_spec_for_default_root(tmp_path, monkeypatch) -> None:
    factor, price = _sample_frames()
    spec = FactorSpec(
        region="US",
        sec_type="STOCK",
        freq="1d",
        table_name="alpha_001",
        variant=None,
        provider="simfin",
    )
    engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        spec=spec,
    )

    assert engine._default_report_root() == report_output_root_for(
        "factor_evaluation",
        "simfin",
        "us",
        "stock",
        "1d",
        "alpha_001",
    )

    report = engine.summary(periods=(1,), quantiles=2)
    loaded = engine.load_summary()
    assert not loaded.empty
    assert list(loaded.columns) == list(report.columns)

    # the section reader should be able to pick up the generated summary parquet
    summary_table = engine.load_section_table("summary")
    assert not summary_table.empty


def test_select_best_holding_period_from_event_returns() -> None:
    average_cumulative = pd.DataFrame(
        {
            1: [0.00, 0.01, 0.02],
            2: [0.00, 0.02, 0.05],
            3: [0.00, 0.03, 0.08],
        },
        index=[0, 1, 2],
    )

    summary = select_best_holding_period_from_event_returns(average_cumulative)

    assert summary["best_holding_period"] == 2
    assert summary["best_cumulative_spread"] == 0.06
    assert summary["direction"] == "use_as_is"


def test_tiger_factor_evaluation_horizon_result(tmp_path) -> None:
    factor, price = _sample_frames_many_assets()
    engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        factor_store=FactorStore(root_dir=tmp_path),
    )

    result = engine.analyze_horizons([1, 3, 5], quantiles=5)
    assert list(result["horizon"]) == [1, 3, 5]
    assert "mean_ic" in result.columns
    assert "sharpe" in result.columns

    summary = engine.summarize_best_horizon([1, 3, 5], quantiles=5)
    assert "suggested_direction" in summary

    direct_summary = summarize_best_horizon(result)
    assert direct_summary["suggested_direction"] in {"use_as_is", "reverse_factor"}

    output_path = tmp_path / "horizon_result.png"
    saved_path = engine.plot_horizon_result([1, 3, 5], quantiles=5, output_path=output_path)
    assert saved_path == output_path
    assert output_path.exists()

    default_path = engine.plot_horizon_result([1, 3, 5], quantiles=5)
    assert default_path == report_output_root_for("factor_evaluation", "alpha_001") / "horizon" / "horizon_result.png"
    assert default_path.exists()
    assert default_path.parent == report_output_root_for("factor_evaluation", "alpha_001") / "horizon"


def test_plot_top_bottom_quantile_turnover_uses_subplots() -> None:
    turnover = pd.DataFrame(
        {
            "top_1D": [0.2, 0.3],
            "bottom_1D": [0.8, 0.7],
            "top_5D": [0.25, 0.35],
            "bottom_5D": [0.75, 0.65],
            "top_10D": [0.28, 0.38],
            "bottom_10D": [0.72, 0.62],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    axes = plot_top_bottom_quantile_turnover(turnover)
    assert len(list(axes)) == 3
    plt.close("all")


def test_load_group_labels_long_frame_uses_wide_layout() -> None:
    labels = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "group": ["tech", "fin", "tech", "fin"],
        }
    )

    wide = load_group_labels(labels)

    assert list(wide.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert list(wide.columns) == ["AAA", "BBB"]
    assert wide.loc[pd.Timestamp("2024-01-02"), "AAA"] == "tech"


def test_load_group_labels_path_uses_cache(tmp_path, monkeypatch) -> None:
    labels = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "group": ["tech", "fin", "tech", "fin"],
        }
    )
    path = tmp_path / "groups.csv"
    labels.to_csv(path, index=False)

    calls = {"count": 0}
    original_read_csv = input_module.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        calls["count"] += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(input_module.pd, "read_csv", counting_read_csv)

    first = load_group_labels(path)
    second = load_group_labels(path)

    assert calls["count"] == 1
    assert first.equals(second)


def test_load_group_labels_cache_can_be_cleared_and_preheated(tmp_path, monkeypatch) -> None:
    labels = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "group": ["tech", "fin", "tech", "fin"],
        }
    )
    path = tmp_path / "groups.csv"
    labels.to_csv(path, index=False)

    calls = {"count": 0}
    original_read_csv = input_module.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        calls["count"] += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(input_module.pd, "read_csv", counting_read_csv)

    clear_group_labels_cache()
    prewarm_group_labels_cache([path])
    assert calls["count"] == 1

    loaded = load_group_labels(path)
    assert calls["count"] == 1
    assert loaded.loc[pd.Timestamp("2024-01-02"), "BBB"] == "fin"

    clear_group_labels_cache()
    load_group_labels(path)
    assert calls["count"] == 2


def test_engine_group_labels_cache_toggle(tmp_path, monkeypatch) -> None:
    factor, price = _sample_frames()
    labels = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "group": ["tech", "fin", "tech", "fin"],
        }
    )
    path = tmp_path / "groups.csv"
    labels.to_csv(path, index=False)

    calls = {"count": 0}
    original_read_csv = input_module.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        calls["count"] += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(input_module.pd, "read_csv", counting_read_csv)
    clear_group_labels_cache()

    cached_engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        group_labels=path,
        group_labels_cache=True,
        factor_store=FactorStore(root_dir=tmp_path),
    )

    assert calls["count"] == 1
    group_frame = cached_engine.group_frame()
    assert calls["count"] == 1
    assert group_frame is not None
    assert group_frame.loc[pd.Timestamp("2024-01-02"), "BBB"] == "fin"

    uncached_engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        group_labels=path,
        group_labels_cache=False,
        factor_store=FactorStore(root_dir=tmp_path),
    )
    assert calls["count"] == 1

    uncached_group_frame = uncached_engine.group_frame()
    assert calls["count"] == 2
    assert uncached_group_frame is not None
    assert uncached_group_frame.loc[pd.Timestamp("2024-01-02"), "BBB"] == "fin"

    second_uncached_engine = FactorEvaluationEngine(
        factor_frame=factor,
        price_frame=price,
        factor_column="alpha_001",
        group_labels=path,
        group_labels_cache=False,
        factor_store=FactorStore(root_dir=tmp_path),
    )
    second_uncached_engine.group_frame()
    assert calls["count"] == 3


def test_quantize_factor_by_group_keeps_group_separation() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2, freq="D"), ["AAA", "BBB", "CCC", "DDD"]],
        names=["date_", "code"],
    )
    factor_data = pd.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
            "group": ["tech", "tech", "fin", "fin", "tech", "tech", "fin", "fin"],
        },
        index=index,
    )

    quantiles = quantize_factor(factor_data, quantiles=2, by_group=True)

    assert quantiles.loc[(pd.Timestamp("2024-01-01"), "AAA")] == 1.0
    assert quantiles.loc[(pd.Timestamp("2024-01-01"), "BBB")] == 2.0
    assert quantiles.loc[(pd.Timestamp("2024-01-01"), "CCC")] == 1.0
    assert quantiles.loc[(pd.Timestamp("2024-01-01"), "DDD")] == 2.0


def test_clean_factor_supports_long_groupby_dataframe() -> None:
    factor = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=2, freq="D"), ["AAA", "BBB"]],
            names=["date_", "code"],
        ),
        name="alpha_001",
    )
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 10.5],
            "BBB": [20.0, 20.5],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    groupby = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "group": ["tech", "tech", "fin", "fin"],
        }
    )

    result = get_clean_factor_and_forward_returns(factor=factor, prices=prices, groupby=groupby, quantiles=2)

    assert "group" in result.factor_data.columns
    assert result.factor_data["factor_quantile"].notna().all()
