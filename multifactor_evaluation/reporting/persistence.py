from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from tiger_factors.factor_evaluation import evaluate_factor_panel
from tiger_factors.multifactor_evaluation.reporting.summary import create_summary_tear_sheet as create_multifactor_summary_report


def build_penetration_analysis(
    prices_long: pd.DataFrame,
    combined_factor: pd.DataFrame,
    *,
    long_pct: float,
) -> dict[str, object]:
    latest_date = combined_factor.index[-1]
    latest_scores = combined_factor.loc[latest_date].dropna().sort_values(ascending=False)
    n_long = max(1, int(len(latest_scores) * long_pct))
    n_short = n_long
    long_basket = latest_scores.nlargest(n_long)
    short_basket = latest_scores.nsmallest(n_short)
    long_equal_weights = pd.Series(1.0 / n_long, index=long_basket.index)
    short_equal_weights = pd.Series(1.0 / n_short, index=short_basket.index)
    long_sector = pd.Series(
        {
            code: prices_long.loc[prices_long["code"] == code, "sector"].dropna().iloc[0]
            if not prices_long.loc[prices_long["code"] == code, "sector"].dropna().empty
            else "Unknown"
            for code in long_basket.index
        }
    )
    short_sector = pd.Series(
        {
            code: prices_long.loc[prices_long["code"] == code, "sector"].dropna().iloc[0]
            if not prices_long.loc[prices_long["code"] == code, "sector"].dropna().empty
            else "Unknown"
            for code in short_basket.index
        }
    )
    return {
        "latest_date": str(latest_date),
        "long_pct": float(long_pct),
        "long_basket": [
            {
                "code": code,
                "score": float(long_basket[code]),
                "position_weight": float(long_equal_weights[code]),
                "sector": str(long_sector.get(code, "Unknown")),
            }
            for code in long_basket.index
        ],
        "short_basket": [
            {
                "code": code,
                "score": float(short_basket[code]),
                "position_weight": float(short_equal_weights[code]),
                "sector": str(short_sector.get(code, "Unknown")),
            }
            for code in short_basket.index
        ],
    }


def build_selected_factor_evaluations(
    factors_wide: dict[str, pd.DataFrame],
    selected_factors: list[str],
    forward_returns: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    evaluation_by_factor: dict[str, dict[str, float]] = {}
    for factor_name in selected_factors:
        panel = factors_wide[factor_name]
        aligned_index = panel.index.intersection(forward_returns.index)
        evaluation = evaluate_factor_panel(panel.loc[aligned_index], forward_returns.loc[aligned_index])
        evaluation_by_factor[factor_name] = {k: float(v) for k, v in asdict(evaluation).items()}
    return evaluation_by_factor


def persist_multifactors_outputs(
    *,
    output_dir: Path,
    pipeline_summary: pd.DataFrame,
    factor_score_table: pd.DataFrame | None = None,
    correlation_matrix: pd.DataFrame,
    prices_long: pd.DataFrame,
    factors_wide: dict[str, pd.DataFrame],
    selected_factors: list[str],
    chosen_weights: dict[str, float],
    combined_factor: pd.DataFrame,
    backtest_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    factor_cols: list[str],
    params: dict[str, Any],
    long_pct: float,
    weight_method: str,
    forward_returns: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    penetration = build_penetration_analysis(prices_long, combined_factor, long_pct=long_pct)
    (output_dir / "multifactors_penetration_analysis.json").write_text(json.dumps(penetration, indent=2), encoding="utf-8")

    evaluation_by_factor = build_selected_factor_evaluations(factors_wide, selected_factors, forward_returns)
    report = {
        "params": params,
        "universe_size": int(prices_long["code"].nunique()),
        "factor_count": len(factor_cols),
        "selected_factor_count": len(selected_factors),
        "selected_factors": selected_factors,
        "weights": {name: float(weight) for name, weight in chosen_weights.items()},
        "weights_all_methods": {
            weight_method: {name: float(weight) for name, weight in chosen_weights.items()}
        },
        "portfolio_stats": stats["portfolio"],
        "benchmark_stats": stats["benchmark"],
        "output_dir": str(output_dir),
    }

    pipeline_summary.to_parquet(output_dir / "multifactors_factor_evaluation_summary.parquet")
    pipeline_summary.to_parquet(output_dir / "multifactors_pipeline_summary.parquet")
    if factor_score_table is not None:
        factor_score_table.to_csv(output_dir / "multifactors_factor_score_table.csv", index=False)
        factor_score_table.to_parquet(output_dir / "multifactors_factor_score_table.parquet")
    correlation_matrix.to_parquet(output_dir / "multifactors_factor_correlation_matrix.parquet")
    pd.Series(selected_factors, name="factor").to_parquet(output_dir / "multifactors_selected_factors.parquet")
    pd.Series(chosen_weights, name="weight").to_parquet(output_dir / "multifactors_factor_weights.parquet")
    combined_factor.to_parquet(output_dir / "multifactors_combined_factor.parquet")
    backtest_df.to_parquet(output_dir / "multifactors_backtest_daily.parquet")
    pd.DataFrame(stats).T.to_parquet(output_dir / "multifactors_backtest_stats.parquet")
    (output_dir / "multifactors_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "multifactors_selected_factor_evaluation.json").write_text(
        json.dumps(evaluation_by_factor, indent=2), encoding="utf-8"
    )
    summary_report = create_multifactor_summary_report(
        backtest_df,
        output_dir=output_dir / "multifactors_summary",
        report_name="multifactors_summary",
    )
    if summary_report is not None:
        report["summary_report_dir"] = str(summary_report.output_dir)
        report["summary_report_html"] = str(summary_report.html_path)
        report["summary_report_table"] = str(summary_report.summary_table_path)
        report["summary_report_figures"] = [str(path) for path in summary_report.figure_paths]
        if summary_report.comparison_table_path is not None:
            report["summary_report_comparison_table"] = str(summary_report.comparison_table_path)
        if summary_report.drawdown_table_path is not None:
            report["summary_report_drawdown_table"] = str(summary_report.drawdown_table_path)
        if summary_report.monthly_returns_table_path is not None:
            report["summary_report_monthly_returns_table"] = str(summary_report.monthly_returns_table_path)
        if summary_report.montecarlo_summary_path is not None:
            report["summary_report_montecarlo_summary"] = str(summary_report.montecarlo_summary_path)
        if summary_report.montecarlo_plot_path is not None:
            report["summary_report_montecarlo_plot"] = str(summary_report.montecarlo_plot_path)
        if summary_report.compare_table_paths is not None:
            report["summary_report_compare_table_paths"] = {
                name: str(path) for name, path in summary_report.compare_table_paths.items()
            }
        (output_dir / "multifactors_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = [
    "build_penetration_analysis",
    "build_selected_factor_evaluations",
    "persist_multifactors_outputs",
]
