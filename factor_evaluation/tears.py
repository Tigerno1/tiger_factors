from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tiger_factors.factor_evaluation.core import FactorEvaluation
from tiger_factors.factor_evaluation.core import evaluate_factor_panel
from tiger_factors.factor_evaluation.performance import average_cumulative_return_by_quantile
from tiger_factors.factor_evaluation.performance import compute_mean_returns_spread
from tiger_factors.factor_evaluation.performance import factor_alpha_beta
from tiger_factors.factor_evaluation.performance import factor_information_coefficient
from tiger_factors.factor_evaluation.performance import factor_rank_autocorrelation
from tiger_factors.factor_evaluation.performance import factor_returns
from tiger_factors.factor_evaluation.performance import mean_information_coefficient
from tiger_factors.factor_evaluation.performance import mean_return_by_quantile
from tiger_factors.factor_evaluation.performance import quantile_turnover
from tiger_factors.factor_evaluation.plotting import plot_cumulative_returns
from tiger_factors.factor_evaluation.plotting import plot_cumulative_returns_by_quantile
from tiger_factors.factor_evaluation.plotting import plot_events_distribution
from tiger_factors.factor_evaluation.plotting import plot_factor_rank_auto_correlation
from tiger_factors.factor_evaluation.plotting import plot_ic_by_group
from tiger_factors.factor_evaluation.plotting import plot_ic_hist
from tiger_factors.factor_evaluation.plotting import plot_ic_missingness
from tiger_factors.factor_evaluation.plotting import plot_ic_qq
from tiger_factors.factor_evaluation.plotting import plot_ic_rolling
from tiger_factors.factor_evaluation.plotting import plot_ic_ts
from tiger_factors.factor_evaluation.plotting import plot_mean_quantile_returns_spread_time_series
from tiger_factors.factor_evaluation.plotting import plot_monthly_ic_heatmap
from tiger_factors.factor_evaluation.plotting import plot_quantile_average_cumulative_return
from tiger_factors.factor_evaluation.plotting import plot_quantile_returns_bar
from tiger_factors.factor_evaluation.plotting import plot_quantile_returns_violin
from tiger_factors.factor_evaluation.plotting import plot_quantile_statistics_table
from tiger_factors.factor_evaluation.plotting import plot_top_bottom_quantile_turnover
from tiger_factors.factor_evaluation.plotting import plot_turnover_table
from tiger_factors.factor_evaluation.plotting import save_figure
from tiger_factors.factor_evaluation.plotting import save_stability_overview
from tiger_factors.report_paths import figure_output_dir_for
from tiger_factors.factor_evaluation.horizon import select_best_holding_period_from_event_returns
from tiger_factors.factor_evaluation.utils import TigerFactorData
from tiger_factors.factor_evaluation.utils import get_forward_returns_columns
from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_evaluation.utils import rate_of_return
from tiger_factors.factor_evaluation.utils import std_conversion
from tiger_factors.factor_evaluation.utils import timedelta_strings_to_integers


@dataclass(frozen=True)
class TigerEvaluationSheet:
    name: str
    output_dir: Path
    figure_paths: dict[str, Path]
    table_paths: dict[str, Path]
    figure_output_dir: Path | None = None
    evaluation: FactorEvaluation | None = None
    payload: dict[str, Any] | None = None


def _resolve_return_mode(
    *,
    long_short: bool,
    return_mode: str | None,
) -> tuple[str, bool]:
    if return_mode is None:
        mode = "long_short" if long_short else "long_only"
        return mode, long_short

    normalized = str(return_mode).strip().lower().replace("-", "_")
    if normalized not in {"long_short", "long_only"}:
        raise ValueError("return_mode must be either 'long_short' or 'long_only'")
    return normalized, normalized == "long_short"


def _write_table(table: pd.DataFrame | pd.Series, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".parquet":
        path = path.with_suffix(".parquet")
    if isinstance(table, pd.Series):
        table.to_frame().to_parquet(path)
    else:
        table.to_parquet(path)
    return path


def _save_named_figure(ax, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.png"
    save_figure(ax, path)
    return path


def _write_section_tables(
    section_name: str,
    section_dir: Path,
    tables: dict[str, pd.DataFrame | pd.Series | None],
) -> dict[str, Path]:
    table_paths: dict[str, Path] = {}
    for filename, table_obj in tables.items():
        if isinstance(table_obj, pd.DataFrame) or isinstance(table_obj, pd.Series):
            table_paths[f"{section_name}_{Path(filename).stem}"] = _write_table(table_obj, section_dir / filename)
    return table_paths


def _default_turnover_periods(factor_frame: pd.DataFrame, fallback: tuple[int | str | pd.Timedelta, ...]) -> tuple[int | str | pd.Timedelta, ...]:
    periods = tuple(get_forward_returns_columns(factor_frame.columns, require_exact_day_multiple=True))
    return periods or fallback


def _turnover_inputs(periods: tuple[int | str | pd.Timedelta, ...], quantile_factor: pd.Series, factor_frame: pd.DataFrame):
    quantile_values = sorted(pd.unique(quantile_factor.dropna().astype(int)))
    period_steps = [
        timedelta_strings_to_integers([period_to_label(period)])[0]
        for period in periods
    ]
    quantile_turnover_data: dict[object, dict[int, pd.Series]] = {}
    autocorrelation: dict[object, pd.Series] = {}
    for raw_period, period_step in zip(periods, period_steps):
        label = period_to_label(raw_period)
        quantile_turnover_data[label] = {
            int(quantile): quantile_turnover(quantile_factor, int(quantile), period_step)
            for quantile in quantile_values
        }
        autocorrelation[label] = factor_rank_autocorrelation(factor_frame, period_step)
    return quantile_turnover_data, autocorrelation


def prepare_shared_artifacts(
    factor_data: TigerFactorData,
    *,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
    turnover_periods: tuple[int | str | pd.Timedelta, ...] | None,
    avgretplot: tuple[int, int],
) -> dict[str, Any]:
    evaluation = evaluate_factor_panel(
        factor_data.factor_panel,
        factor_data.forward_returns,
    )
    mean_quant_ret, std_quantile = mean_return_by_quantile(
        factor_data.factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )
    mean_quant_ret_bydate, std_quant_daily = mean_return_by_quantile(
        factor_data.factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )
    mean_quant_ror = mean_quant_ret.apply(rate_of_return, axis=0, base_period=mean_quant_ret.columns[0])
    mean_quant_ror_bydate = mean_quant_ret_bydate.apply(
        rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )
    compstd_quant_daily = std_quant_daily.apply(
        std_conversion,
        axis=0,
        base_period=std_quant_daily.columns[0],
    )
    primary_period = str(mean_quant_ror_bydate.columns[0])
    primary_quantile_returns = mean_quant_ror_bydate[primary_period].unstack("factor_quantile").sort_index()
    primary_quantile_std = compstd_quant_daily[primary_period].unstack("factor_quantile").sort_index()
    spread, spread_std = compute_mean_returns_spread(
        primary_quantile_returns,
        int(factor_data.factor_data["factor_quantile"].max()),
        int(factor_data.factor_data["factor_quantile"].min()),
        std_err=primary_quantile_std,
    )
    ic = factor_information_coefficient(factor_data.factor_data, group_adjust=group_neutral)
    mean_ic = _mean_ic_from_ic(ic)
    monthly_ic = _monthly_ic_from_ic(ic)
    resolved_turnover_periods: tuple[int | str | pd.Timedelta, ...] = (
        turnover_periods or _default_turnover_periods(factor_data.factor_data, factor_data.periods)
    )
    quantile_factor = factor_data.factor_data["factor_quantile"]
    quantile_turnover_data, autocorrelation_data = _turnover_inputs(
        resolved_turnover_periods,
        quantile_factor,
        factor_data.factor_data,
    )
    turnover_table = plot_turnover_table(autocorrelation_data, quantile_turnover_data, return_df=True)
    turnover_summary: pd.DataFrame = _coerce_dataframe(turnover_table)
    factor_portfolio_returns = factor_returns(
        factor_data.factor_data,
        demeaned=long_short,
        group_adjust=group_neutral,
    )
    daily_returns = factor_data.prices.pct_change(fill_method=None)
    average_cumulative = average_cumulative_return_by_quantile(
        factor_data.factor_data,
        daily_returns,
        periods_before=avgretplot[0],
        periods_after=avgretplot[1],
        demeaned=long_short,
        group_adjust=group_neutral,
        by_group=False,
    )
    best_holding_period = select_best_holding_period_from_event_returns(average_cumulative)
    average_cumulative_by_group = None
    ic_group = None
    mean_ic_by_group = None
    mean_quant_group = None
    if by_group and factor_data.group_column and factor_data.group_column in factor_data.factor_data.columns:
        mean_quant_group, _ = mean_return_by_quantile(
            factor_data.factor_data,
            by_date=False,
            by_group=True,
            demeaned=long_short,
            group_adjust=group_neutral,
        )
        ic_group = factor_information_coefficient(
            factor_data.factor_data,
            group_adjust=group_neutral,
            by_group=True,
        )
        mean_ic_by_group = mean_information_coefficient(
            factor_data.factor_data,
            group_adjust=group_neutral,
            by_group=True,
        )
        average_cumulative_by_group = average_cumulative_return_by_quantile(
            factor_data.factor_data,
            daily_returns,
            periods_before=avgretplot[0],
            periods_after=avgretplot[1],
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=True,
        )
    return {
        "evaluation": evaluation,
        "mean_quant_ret": mean_quant_ret,
        "std_quantile": std_quantile,
        "mean_quant_ret_bydate": mean_quant_ret_bydate,
        "std_quant_daily": std_quant_daily,
        "mean_quant_ror": mean_quant_ror,
        "mean_quant_ror_bydate": mean_quant_ror_bydate,
        "compstd_quant_daily": compstd_quant_daily,
        "primary_quantile_returns": primary_quantile_returns,
        "primary_quantile_std": primary_quantile_std,
        "spread": spread,
        "spread_std": spread_std,
        "ic": ic,
        "mean_ic": mean_ic,
        "monthly_ic": monthly_ic,
        "turnover_periods": resolved_turnover_periods,
        "quantile_turnover_data": quantile_turnover_data,
        "autocorrelation_data": autocorrelation_data,
        "turnover_summary": turnover_summary,
        "factor_portfolio_returns": factor_portfolio_returns,
        "average_cumulative": average_cumulative,
        "best_holding_period": best_holding_period,
        "average_cumulative_by_group": average_cumulative_by_group,
        "mean_quant_group": mean_quant_group,
        "ic_group": ic_group,
            "mean_ic_by_group": mean_ic_by_group,
        }


def _mean_return_spread_by_period(
    mean_quant_ror_bydate: pd.DataFrame,
    compstd_quant_daily: pd.DataFrame,
    factor_data: TigerFactorData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spread_frames: dict[str, pd.Series] = {}
    spread_std_frames: dict[str, pd.Series] = {}
    top = int(factor_data.factor_data["factor_quantile"].max())
    bottom = int(factor_data.factor_data["factor_quantile"].min())
    for period in mean_quant_ror_bydate.columns:
        period_key = str(period)
        period_returns = mean_quant_ror_bydate[period].unstack("factor_quantile").sort_index()
        period_std = compstd_quant_daily[period].unstack("factor_quantile").sort_index()
        spread, spread_std = compute_mean_returns_spread(
            period_returns,
            top,
            bottom,
            std_err=period_std,
        )
        spread_frames[period_key] = spread
        if spread_std is not None:
            spread_std_frames[period_key] = spread_std
    spread_frame = pd.DataFrame(spread_frames).sort_index()
    spread_std_frame = pd.DataFrame(spread_std_frames).sort_index() if spread_std_frames else pd.DataFrame(index=spread_frame.index)
    return spread_frame, spread_std_frame


def _mean_ic_from_ic(ic: pd.DataFrame) -> pd.Series:
    if ic.empty:
        return pd.Series(dtype=float)
    return ic.mean()


def _monthly_ic_from_ic(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return pd.DataFrame()
    index = pd.to_datetime(ic.index, errors="coerce")
    frame = ic.copy()
    frame.index = index
    valid_index = pd.notna(frame.index)
    frame = frame.loc[valid_index]
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby(pd.Grouper(freq="ME")).mean()


def _cached_value(precomputed: dict[str, Any] | None, key: str, compute):
    if precomputed is not None and key in precomputed:
        return precomputed[key]
    return compute()


def _coerce_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, tuple):
        return pd.concat(value, axis=0)
    return pd.DataFrame(value)


def _build_return_mode_portfolio_returns_table(returns_payload: dict[str, Any]) -> pd.DataFrame:
    combined_frames: list[pd.DataFrame] = []
    for mode in ("long_short", "long_only"):
        payload = returns_payload.get(mode)
        if not isinstance(payload, dict):
            continue
        mode_returns = payload.get("factor_portfolio_returns")
        if isinstance(mode_returns, pd.Series):
            mode_returns = mode_returns.to_frame(name=mode)
        elif isinstance(mode_returns, pd.DataFrame) and not mode_returns.empty:
            primary_column = mode_returns.columns[0]
            mode_returns = mode_returns[[primary_column]].rename(columns={primary_column: mode})
        else:
            continue
        combined_frames.append(mode_returns)
    if not combined_frames:
        return pd.DataFrame()
    returns_table = pd.concat(combined_frames, axis=1).sort_index()
    returns_table.index.name = "date_"
    return returns_table


def _summary_row_from_evaluation(
    evaluation: FactorEvaluation,
    *,
    highlights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(evaluation.__dict__)
    if highlights:
        row.update(highlights)
    return row


def _summary_highlights_from_spread(spread: pd.Series | pd.DataFrame) -> dict[str, Any]:
    highlights: dict[str, Any] = {}
    if isinstance(spread, pd.Series):
        clean_spread = pd.to_numeric(spread, errors="coerce").dropna()
        if not clean_spread.empty:
            highlights["mean_return_spread_mean"] = float(clean_spread.mean())
            highlights["mean_return_spread_latest"] = float(clean_spread.iloc[-1])
        return highlights
    for period, series in spread.items():
        clean_spread = pd.to_numeric(series, errors="coerce").dropna()
        if clean_spread.empty:
            continue
        highlights[f"mean_return_spread_{period}_mean"] = float(clean_spread.mean())
        highlights[f"mean_return_spread_{period}_latest"] = float(clean_spread.iloc[-1])
    return highlights


def _section_figure_output_dir(factor_data: TigerFactorData, section: str, figure_output_dir: str | Path | None) -> Path:
    if figure_output_dir is not None:
        return Path(figure_output_dir)
    return figure_output_dir_for(factor_data.factor_column) / section


def create_summary_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    long_short: bool = True,
    group_neutral: bool = False,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
    highlights: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = _cached_value(precomputed, "evaluation", lambda: evaluate_factor_panel(
        factor_data.factor_panel,
        factor_data.forward_returns,
    ))
    summary_row = _summary_row_from_evaluation(evaluation, highlights=highlights)
    if figure:
        _cached_value(precomputed, "ic", lambda: factor_information_coefficient(
            factor_data.factor_data,
            group_adjust=group_neutral,
        ))
        _cached_value(precomputed, "monthly_ic", lambda: mean_information_coefficient(
            factor_data.factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="ME",
        ))
    figure_paths: dict[str, Path] = {}
    table_paths = {}
    if table:
        table_paths = {
            "summary": _write_table(pd.DataFrame([summary_row]), output / "summary.parquet"),
        }
    return TigerEvaluationSheet(
        name="summary",
        output_dir=output,
        figure_paths=figure_paths,
        table_paths=table_paths,
        evaluation=evaluation,
        payload={"evaluation": evaluation, "summary": summary_row},
    )


def _create_returns_tear_sheet_single(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    long_short: bool = True,
    return_mode_label: str,
    group_neutral: bool = False,
    by_group: bool = False,
    rate_of_ret: bool = True,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    mode_suffix = f"_{return_mode_label}"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = _section_figure_output_dir(factor_data, "returns", figure_output_dir)
    figure_output.mkdir(parents=True, exist_ok=True)
    factor_portfolio_returns = _cached_value(precomputed, "factor_portfolio_returns", lambda: factor_returns(
        factor_data.factor_data,
        demeaned=long_short,
        group_adjust=group_neutral,
    ))
    mean_quant_ret, std_quantile = _cached_value(precomputed, "mean_quant_ret_pair", lambda: mean_return_by_quantile(
        factor_data.factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    ))
    mean_quant_ret_bydate, std_quant_daily = _cached_value(precomputed, "mean_quant_ret_bydate_pair", lambda: mean_return_by_quantile(
        factor_data.factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    ))
    if rate_of_ret:
        mean_quant_ror = _cached_value(precomputed, "mean_quant_ror", lambda: mean_quant_ret.apply(rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]))
        mean_quant_ror_bydate = _cached_value(precomputed, "mean_quant_ror_bydate", lambda: mean_quant_ret_bydate.apply(
            rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        ))
        compstd_quant_daily = _cached_value(precomputed, "compstd_quant_daily", lambda: std_quant_daily.apply(
            std_conversion,
            axis=0,
            base_period=std_quant_daily.columns[0],
        ))
    else:
        mean_quant_ror = mean_quant_ret
        mean_quant_ror_bydate = mean_quant_ret_bydate
        compstd_quant_daily = std_quant_daily
    quantile_stats: pd.DataFrame = _coerce_dataframe(
        _cached_value(
            precomputed,
            "quantile_stats",
            lambda: plot_quantile_statistics_table(
                factor_data.factor_data,
                return_df=True,
            ),
        )
    )
    alpha_beta: pd.DataFrame = _coerce_dataframe(
        _cached_value(precomputed, "alpha_beta", lambda: factor_alpha_beta(
            factor_data.factor_data,
            demeaned=long_short,
            group_adjust=group_neutral,
        ))
    )
    primary_period = str(mean_quant_ror_bydate.columns[0])
    primary_quantile_returns = _cached_value(precomputed, "primary_quantile_returns", lambda: mean_quant_ror_bydate[primary_period].unstack("factor_quantile").sort_index())
    factor_portfolio_return_series = factor_portfolio_returns[mean_quant_ror.columns[0]]
    spread, spread_std = _cached_value(
        precomputed,
        "spread_by_period",
        lambda: _mean_return_spread_by_period(mean_quant_ror_bydate, compstd_quant_daily, factor_data),
    )

    if figure:
        figure_paths = {
            f"quantile_returns_bar_{return_mode_label}": _save_named_figure(plot_quantile_returns_bar(mean_quant_ror), figure_output, f"quantile_returns_bar_{return_mode_label}"),
            f"quantile_returns_violin_{return_mode_label}": _save_named_figure(
                plot_quantile_returns_violin(mean_quant_ror_bydate, ylim_percentiles=(1, 99)),
                figure_output,
                f"quantile_returns_violin_{return_mode_label}",
            ),
        }
        cumulative_periods = list(mean_quant_ror.columns)
        if cumulative_periods:
            first_period = cumulative_periods[0]
            first_period_label = period_to_label(first_period)
            first_quantile_frame = primary_quantile_returns
            figure_paths[f"cumulative_returns_{first_period_label}{mode_suffix}"] = _save_named_figure(
                plot_cumulative_returns(
                    factor_portfolio_return_series,
                    period=first_period,
                ),
                figure_output,
                f"cumulative_returns_{first_period_label}{mode_suffix}",
            )
            figure_paths[f"cumulative_returns_by_quantile_{first_period_label}{mode_suffix}"] = _save_named_figure(
                plot_cumulative_returns_by_quantile(
                    first_quantile_frame,
                    period=primary_period,
                ),
                figure_output,
                f"cumulative_returns_by_quantile_{first_period_label}{mode_suffix}",
            )
            for period in cumulative_periods:
                if period == first_period:
                    continue
                period_label = period_to_label(period)
                period_returns = factor_portfolio_returns[period]
                period_key = period if period in mean_quant_ror_bydate.columns else str(period)
                period_quantile_returns = mean_quant_ror_bydate[period_key].unstack("factor_quantile").sort_index()
                figure_paths[f"cumulative_returns_{period_label}{mode_suffix}"] = _save_named_figure(
                    plot_cumulative_returns(
                        period_returns,
                        period=period,
                    ),
                    figure_output,
                    f"cumulative_returns_{period_label}{mode_suffix}",
                )
                figure_paths[f"cumulative_returns_by_quantile_{period_label}{mode_suffix}"] = _save_named_figure(
                    plot_cumulative_returns_by_quantile(
                        period_quantile_returns,
                        period=period,
                    ),
                    figure_output,
                    f"cumulative_returns_by_quantile_{period_label}{mode_suffix}",
                )
        if not spread.empty:
            for period in spread.columns:
                period_axis = plot_mean_quantile_returns_spread_time_series(spread[period])
                figure_paths[f"mean_return_spread_{period}{mode_suffix}"] = _save_named_figure(
                    period_axis,
                    figure_output,
                    f"mean_return_spread_{period}{mode_suffix}",
                )
    else:
        figure_paths = {}
    table_paths = {}
    if table:
        table_paths = {
            f"mean_return_by_quantile_{return_mode_label}": _write_table(mean_quant_ror, output / f"mean_return_by_quantile_{return_mode_label}.parquet"),
            f"mean_return_by_quantile_std_{return_mode_label}": _write_table(
                std_quantile,
                output / f"mean_return_by_quantile_std_{return_mode_label}.parquet",
            ),
            f"mean_return_by_quantile_by_date_{return_mode_label}": _write_table(mean_quant_ror_bydate, output / f"mean_return_by_quantile_by_date_{return_mode_label}.parquet"),
            f"primary_quantile_returns_by_date_{return_mode_label}": _write_table(primary_quantile_returns, output / f"primary_quantile_returns_by_date_{return_mode_label}.parquet"),
            f"alpha_beta_{return_mode_label}": _write_table(alpha_beta, output / f"alpha_beta_{return_mode_label}.parquet"),
            f"mean_return_spread_{return_mode_label}": _write_table(spread, output / f"mean_return_spread_{return_mode_label}.parquet"),
            f"quantile_statistics_{return_mode_label}": _write_table(quantile_stats, output / f"quantile_statistics_{return_mode_label}.parquet"),
        }
        if spread_std is not None:
            table_paths[f"mean_return_spread_std_{return_mode_label}"] = _write_table(spread_std, output / f"mean_return_spread_std_{return_mode_label}.parquet")
    payload: dict[str, Any] = {
        "alpha_beta": alpha_beta,
        "factor_portfolio_returns": factor_portfolio_returns,
        "return_mode": return_mode_label,
        "long_short": long_short,
        "mean_return_by_quantile": mean_quant_ror,
        "mean_return_by_quantile_std": std_quantile,
        "mean_return_by_quantile_by_date": mean_quant_ror_bydate,
        "mean_return_spread": spread,
        "mean_return_spread_std": spread_std,
        "primary_quantile_returns_by_date": primary_quantile_returns,
        "quantile_statistics": quantile_stats,
    }
    if by_group and factor_data.group_column and factor_data.group_column in factor_data.factor_data.columns:
        mean_quant_group = _cached_value(precomputed, "mean_quant_group", lambda: mean_return_by_quantile(
            factor_data.factor_data,
            by_date=False,
            by_group=True,
            demeaned=long_short,
            group_adjust=group_neutral,
        )[0])
        if table:
            table_paths[f"mean_return_by_quantile_by_group_{return_mode_label}"] = _write_table(
                mean_quant_group,
                output / f"mean_return_by_quantile_by_group_{return_mode_label}.parquet",
            )
        if figure:
            figure_paths[f"quantile_returns_bar_by_group_{return_mode_label}"] = _save_named_figure(
                plot_quantile_returns_bar(mean_quant_group, by_group=True),
                figure_output,
                f"quantile_returns_bar_by_group_{return_mode_label}",
            )
        payload["mean_return_by_quantile_by_group"] = mean_quant_group
    return TigerEvaluationSheet(
        name="returns",
        output_dir=output,
        figure_paths=figure_paths,
        table_paths=table_paths,
        payload=payload,
    )


def create_returns_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    group_neutral: bool = False,
    by_group: bool = False,
    rate_of_ret: bool = True,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = _section_figure_output_dir(factor_data, "returns", figure_output_dir)
    figure_output.mkdir(parents=True, exist_ok=True)

    long_short_sheet = _create_returns_tear_sheet_single(
        factor_data,
        output_dir=output,
        figure_output_dir=figure_output,
        long_short=True,
        return_mode_label="long_short",
        group_neutral=group_neutral,
        by_group=by_group,
        rate_of_ret=rate_of_ret,
        table=table,
        figure=figure,
        precomputed=precomputed,
    )
    long_only_sheet = _create_returns_tear_sheet_single(
        factor_data,
        output_dir=output,
        figure_output_dir=figure_output,
        long_short=False,
        return_mode_label="long_only",
        group_neutral=group_neutral,
        by_group=by_group,
        rate_of_ret=rate_of_ret,
        table=table,
        figure=figure,
        precomputed=precomputed,
    )
    combined_table_paths = {**long_short_sheet.table_paths, **long_only_sheet.table_paths}
    returns_table = _build_return_mode_portfolio_returns_table(
        {
            "long_short": long_short_sheet.payload or {},
            "long_only": long_only_sheet.payload or {},
        }
    )
    if not returns_table.empty:
        returns_path = output / "factor_portfolio_returns.parquet"
        returns_table.to_parquet(returns_path)
        combined_table_paths["factor_portfolio_returns"] = returns_path
    return TigerEvaluationSheet(
        name="returns",
        output_dir=output,
        figure_paths={**long_short_sheet.figure_paths, **long_only_sheet.figure_paths},
        table_paths=combined_table_paths,
        payload={
            "return_modes": ["long_short", "long_only"],
            "long_short": long_short_sheet.payload or {},
            "long_only": long_only_sheet.payload or {},
            "factor_portfolio_returns": returns_table,
        },
    )


def create_information_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    group_neutral: bool = False,
    by_group: bool = False,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = _section_figure_output_dir(factor_data, "information", figure_output_dir)
    figure_output.mkdir(parents=True, exist_ok=True)
    ic = _cached_value(precomputed, "ic", lambda: factor_information_coefficient(
        factor_data.factor_data,
        group_adjust=group_neutral,
        by_group=False,
    ))
    mean_ic = _cached_value(precomputed, "mean_ic", lambda: _mean_ic_from_ic(ic))
    monthly_ic = _cached_value(precomputed, "monthly_ic", lambda: _monthly_ic_from_ic(ic))

    if figure and not ic.empty:
        figure_paths = {
            "ic_series": _save_named_figure(plot_ic_ts(ic), figure_output, "ic_series"),
            "ic_histogram": _save_named_figure(plot_ic_hist(ic), figure_output, "ic_histogram"),
            "ic_missingness": _save_named_figure(plot_ic_missingness(ic), figure_output, "ic_missingness"),
            "ic_qq": _save_named_figure(plot_ic_qq(ic), figure_output, "ic_qq"),
            "ic_rolling": _save_named_figure(plot_ic_rolling(ic), figure_output, "ic_rolling"),
        }
        if not by_group and monthly_ic is not None:
            figure_paths["monthly_ic_heatmap"] = _save_named_figure(
                plot_monthly_ic_heatmap(monthly_ic),
                figure_output,
                "monthly_ic_heatmap",
            )
    else:
        figure_paths = {}
    table_paths = {}
    if table:
        table_paths = {
            "information_coefficient": _write_table(ic, output / "information_coefficient.parquet"),
            "mean_information_coefficient": _write_table(mean_ic.to_frame(name="mean_ic"), output / "mean_information_coefficient.parquet"),
            "monthly_information_coefficient": _write_table(monthly_ic, output / "monthly_information_coefficient.parquet"),
        }
    payload: dict[str, Any] = {"ic": ic, "mean_ic": mean_ic, "monthly_ic": monthly_ic}
    if by_group and factor_data.group_column and factor_data.group_column in factor_data.factor_data.columns:
        ic_group = _cached_value(precomputed, "ic_group", lambda: factor_information_coefficient(
            factor_data.factor_data,
            group_adjust=group_neutral,
            by_group=True,
        ))
        group_mean: pd.DataFrame = _coerce_dataframe(
            _cached_value(precomputed, "mean_ic_by_group", lambda: mean_information_coefficient(
                factor_data.factor_data,
                group_adjust=group_neutral,
                by_group=True,
            ))
        )
        if figure:
            figure_paths["ic_by_group"] = _save_named_figure(plot_ic_by_group(group_mean), figure_output, "ic_by_group")
        if table:
            table_paths["information_coefficient_by_group"] = _write_table(ic_group, output / "information_coefficient_by_group.parquet")
            table_paths["mean_information_coefficient_by_group"] = _write_table(group_mean, output / "mean_information_coefficient_by_group.parquet")
        payload["ic_group"] = ic_group
        payload["mean_ic_by_group"] = group_mean
    return TigerEvaluationSheet(
        name="information",
        output_dir=output,
        figure_paths=figure_paths,
        table_paths=table_paths,
        payload=payload,
    )


def create_turnover_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    turnover_periods: tuple[int | str | pd.Timedelta, ...] | None = None,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = _section_figure_output_dir(factor_data, "turnover", figure_output_dir)
    figure_output.mkdir(parents=True, exist_ok=True)
    resolved_turnover_periods: tuple[int | str | pd.Timedelta, ...] = (
        turnover_periods
        or (precomputed.get("turnover_periods") if precomputed is not None else None)
        or _default_turnover_periods(factor_data.factor_data, factor_data.periods)
    )
    quantile_factor = factor_data.factor_data["factor_quantile"]
    turnover_stats: dict[str, pd.Series] = {}
    turnover_inputs = _cached_value(precomputed, "turnover_inputs", lambda: _turnover_inputs(
        resolved_turnover_periods,
        quantile_factor,
        factor_data.factor_data,
    ))
    quantile_turnover_data, autocorrelation_data = turnover_inputs
    for label, quantile_map in quantile_turnover_data.items():
        for quantile, values in quantile_map.items():
            suffix = "top" if quantile == int(quantile_factor.max()) else "bottom"
            turnover_stats[f"{suffix}_{label}"] = values
    turnover_frame = _cached_value(precomputed, "turnover_frame", lambda: pd.DataFrame(turnover_stats))
    rank_autocorr = _cached_value(precomputed, "rank_autocorr", lambda: pd.concat(
        [autocorrelation_data[period_to_label(period)] for period in resolved_turnover_periods],
        axis=1,
    ))
    if not rank_autocorr.empty:
        rank_autocorr.columns = [period_to_label(period) for period in resolved_turnover_periods]
    turnover_summary: pd.DataFrame = _coerce_dataframe(
        _cached_value(
            precomputed,
            "turnover_summary",
            lambda: plot_turnover_table(autocorrelation_data, quantile_turnover_data, return_df=True),
        )
    )
    if figure:
        figure_paths = {
            "turnover": _save_named_figure(plot_top_bottom_quantile_turnover(turnover_frame), figure_output, "turnover"),
        }
        if not rank_autocorr.empty:
            for column in rank_autocorr.columns:
                period_label = period_to_label(str(column))
                rank_key = f"rank_autocorrelation_{period_label}"
                figure_paths[rank_key] = _save_named_figure(
                    plot_factor_rank_auto_correlation(rank_autocorr[column], period=period_label),
                    figure_output,
                    rank_key,
                )
    else:
        figure_paths = {}
    table_paths = {}
    if table:
        table_paths = {
            "turnover": _write_table(turnover_frame, output / "turnover.parquet"),
            "rank_autocorrelation": _write_table(rank_autocorr, output / "rank_autocorrelation.parquet"),
            "turnover_summary": _write_table(turnover_summary, output / "turnover_summary.parquet"),
        }
    return TigerEvaluationSheet(
        name="turnover",
        output_dir=output,
        figure_paths=figure_paths,
        table_paths=table_paths,
        payload={"turnover": turnover_frame, "rank_autocorrelation": rank_autocorr, "turnover_summary": turnover_summary},
    )


def create_event_returns_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    returns: pd.DataFrame | None = None,
    avgretplot: tuple[int, int] = (5, 15),
    long_short: bool = True,
    group_neutral: bool = False,
    std_bar: bool = True,
    by_group: bool = False,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = _section_figure_output_dir(factor_data, "event_returns", figure_output_dir)
    figure_output.mkdir(parents=True, exist_ok=True)
    daily_returns = returns if returns is not None else factor_data.prices.pct_change(fill_method=None)
    average_cumulative = _cached_value(precomputed, "average_cumulative", lambda: average_cumulative_return_by_quantile(
        factor_data.factor_data,
        daily_returns,
        periods_before=avgretplot[0],
        periods_after=avgretplot[1],
        demeaned=long_short,
        group_adjust=group_neutral,
        by_group=False,
    ))
    best_holding_period = _cached_value(precomputed, "best_holding_period", lambda: select_best_holding_period_from_event_returns(average_cumulative))
    if figure:
        figure_paths = {
            "average_cumulative_return_by_quantile": _save_named_figure(
                plot_quantile_average_cumulative_return(
                    average_cumulative,
                    by_quantile=False,
                    std_bar=False,
                ),
                figure_output,
                "average_cumulative_return_by_quantile",
            ),
        }
        if std_bar:
            figure_paths["average_cumulative_return_by_quantile_std"] = _save_named_figure(
                plot_quantile_average_cumulative_return(
                    average_cumulative,
                    by_quantile=True,
                    std_bar=True,
                ),
                figure_output,
                "average_cumulative_return_by_quantile_std",
            )
    else:
        figure_paths = {}
    average_cumulative_by_group: pd.DataFrame | None = None
    if by_group and factor_data.group_column and factor_data.group_column in factor_data.factor_data.columns:
        average_cumulative_by_group = pd.DataFrame(
            _cached_value(
                precomputed,
                "average_cumulative_by_group",
                lambda: average_cumulative_return_by_quantile(
                    factor_data.factor_data,
                    daily_returns,
                    periods_before=avgretplot[0],
                    periods_after=avgretplot[1],
                    demeaned=long_short,
                    group_adjust=group_neutral,
                    by_group=True,
                ),
            )
    )
    if average_cumulative_by_group is not None and isinstance(average_cumulative_by_group.index, pd.MultiIndex) and "group" in average_cumulative_by_group.index.names:
        for group in average_cumulative_by_group.index.get_level_values("group").unique():
            group_key = group
            group_label = str(group)
            if figure:
                figure_paths[f"average_cumulative_return_by_quantile_{group_label}"] = _save_named_figure(
                    plot_quantile_average_cumulative_return(
                        average_cumulative_by_group.xs(group_key, level="group"),
                        by_quantile=False,
                        std_bar=False,
                        title=f"Average Cumulative Return By Quantile - {group_label}",
                    ),
                    figure_output,
                    f"average_cumulative_return_by_quantile_{group_label}",
                )
    if table:
        table_paths = {
            "average_cumulative_return_by_quantile": _write_table(
                average_cumulative,
                output / "average_cumulative_return_by_quantile.parquet",
            ),
            "best_holding_period": _write_table(
                pd.DataFrame([best_holding_period]),
                output / "best_holding_period.parquet",
            ),
        }
    else:
        table_paths = {}
    return TigerEvaluationSheet(
        name="event_returns",
        output_dir=output,
        figure_paths=figure_paths,
        table_paths=table_paths,
        payload={
            "average_cumulative_return_by_quantile": average_cumulative,
            "average_cumulative_return_by_group": average_cumulative_by_group,
            "best_holding_period": best_holding_period,
        },
    )


def create_event_study_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    returns: pd.DataFrame | None = None,
    avgretplot: tuple[int, int] | None = (5, 15),
    rate_of_ret: bool = True,
    n_bars: int = 50,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    turnover_periods: tuple[int | str | pd.Timedelta, ...] | None = None,
    table: bool = True,
    figure: bool = False,
    precomputed: dict[str, Any] | None = None,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = _section_figure_output_dir(factor_data, "event_study", figure_output_dir)
    figure_output.mkdir(parents=True, exist_ok=True)
    quantile_stats: pd.DataFrame = _coerce_dataframe(
        plot_quantile_statistics_table(factor_data.factor_data, return_df=True)
    )
    event_distribution_path = None
    if figure:
        event_distribution_path = _save_named_figure(
            plot_events_distribution(factor_data.factor_data["factor"], num_bars=n_bars),
            figure_output,
            "event_distribution",
        )
    event_returns = None
    if avgretplot is not None:
        if returns is None:
            raise ValueError("returns must be provided when avgretplot is not None")
        event_returns = create_event_returns_tear_sheet(
            factor_data,
            output_dir=output / "event_returns",
            figure_output_dir=figure_output / "event_returns",
            returns=returns,
            avgretplot=avgretplot,
            long_short=False,
            group_neutral=False,
            std_bar=True,
            by_group=False,
            table=table,
            figure=figure,
            precomputed=precomputed,
        )

    mean_quant_ret, _ = mean_return_by_quantile(
        factor_data.factor_data,
        by_group=False,
        demeaned=False,
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(rate_of_return, axis=0, base_period=mean_quant_ret.columns[0])

    mean_quant_ret_bydate, _ = mean_return_by_quantile(
        factor_data.factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )
    returns_bar_path = None
    returns_violin_path = None
    if figure:
        returns_bar_path = _save_named_figure(
            plot_quantile_returns_bar(mean_quant_ret, by_group=False),
            figure_output,
            "quantile_returns_bar",
        )
        returns_violin_path = _save_named_figure(
            plot_quantile_returns_violin(mean_quant_ret_bydate, ylim_percentiles=(1, 99)),
            figure_output,
            "quantile_returns_violin",
        )
    table_paths = {}
    if table:
        table_paths = {
            "quantile_statistics": _write_table(quantile_stats, output / "quantile_statistics.parquet"),
            "mean_return_by_quantile": _write_table(mean_quant_ret, output / "mean_return_by_quantile.parquet"),
            "mean_return_by_quantile_by_date": _write_table(
                mean_quant_ret_bydate,
                output / "mean_return_by_quantile_by_date.parquet",
            ),
        }
    if event_returns is not None:
        table_paths.update(event_returns.table_paths)
    return TigerEvaluationSheet(
        name="event_study",
        output_dir=output,
        figure_paths={
            **({"event_distribution": event_distribution_path} if event_distribution_path is not None else {}),
            **({"quantile_returns_bar": returns_bar_path} if returns_bar_path is not None else {}),
            **({"quantile_returns_violin": returns_violin_path} if returns_violin_path is not None else {}),
            **({} if event_returns is None else event_returns.figure_paths),
        },
        table_paths=table_paths,
        payload={
            "event_returns": event_returns,
            "mean_return_by_quantile": mean_quant_ret,
            "mean_return_by_quantile_by_date": mean_quant_ret_bydate,
        },
    )


def create_full_tear_sheet(
    factor_data: TigerFactorData,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    turnover_periods: tuple[int, ...] | None = None,
    avgretplot: tuple[int, int] = (5, 15),
    horizon_result: pd.DataFrame | None = None,
    horizon_summary: dict[str, Any] | None = None,
    figure: bool = True,
) -> TigerEvaluationSheet:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = Path(figure_output_dir) if figure_output_dir is not None else None
    if figure_output is not None:
        figure_output.mkdir(parents=True, exist_ok=True)
    summary_dir = output / "summary"
    returns_dir = output / "returns"
    information_dir = output / "information"
    turnover_dir = output / "turnover"
    event_returns_dir = output / "event_returns"
    horizon_dir = output / "horizon"
    artifacts = prepare_shared_artifacts(
        factor_data,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        turnover_periods=turnover_periods,
        avgretplot=avgretplot,
    )
    ic = artifacts.get("ic", pd.DataFrame())
    monthly_ic = artifacts.get("monthly_ic", pd.DataFrame())
    mean_quant_ret = artifacts.get("mean_quant_ret", pd.DataFrame())
    mean_quant_ret_bydate = artifacts.get("mean_quant_ret_bydate", pd.DataFrame())
    spread = artifacts.get("spread", pd.DataFrame())
    factor_portfolio_returns = artifacts.get("factor_portfolio_returns", pd.DataFrame())
    average_cumulative = artifacts.get("average_cumulative", pd.DataFrame())
    best_holding_period = artifacts.get("best_holding_period")
    turnover_summary = artifacts.get("turnover_summary", pd.DataFrame())
    summary_highlights: dict[str, Any] = _summary_highlights_from_spread(spread)
    if isinstance(best_holding_period, dict):
        summary_highlights["best_return_holding_period"] = best_holding_period.get("best_holding_period")
        summary_highlights["best_return_cumulative_spread"] = best_holding_period.get("best_cumulative_spread")
        summary_highlights["best_return_direction"] = best_holding_period.get("direction")
    if horizon_summary is not None:
        summary_highlights.update(horizon_summary)
    summary = create_summary_tear_sheet(
        factor_data,
        output_dir=summary_dir,
        long_short=long_short,
        group_neutral=group_neutral,
        table=False,
        figure=False,
        precomputed=artifacts,
        highlights=summary_highlights or None,
    )
    returns = create_returns_tear_sheet(
        factor_data,
        output_dir=returns_dir,
        figure_output_dir=returns_dir,
        group_neutral=group_neutral,
        by_group=by_group,
        table=False,
        figure=figure,
        precomputed=artifacts,
    )
    information = create_information_tear_sheet(
        factor_data,
        output_dir=information_dir,
        figure_output_dir=information_dir,
        group_neutral=group_neutral,
        by_group=by_group,
        table=False,
        figure=figure,
        precomputed=artifacts,
    )
    turnover = create_turnover_tear_sheet(
        factor_data,
        output_dir=turnover_dir,
        figure_output_dir=turnover_dir,
        turnover_periods=turnover_periods,
        table=False,
        figure=figure,
        precomputed=artifacts,
    )
    event_returns = create_event_returns_tear_sheet(
        factor_data,
        output_dir=event_returns_dir,
        figure_output_dir=event_returns_dir,
        avgretplot=avgretplot,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        table=False,
        figure=figure,
        precomputed=artifacts,
    )

    summary_payload = summary.payload or {}
    returns_payload = returns.payload or {}
    information_payload = information.payload or {}
    turnover_payload = turnover.payload or {}
    event_returns_payload = event_returns.payload or {}

    summary_eval = summary_payload.get("evaluation")
    summary_row = summary_payload.get("summary")
    summary_row_df = pd.DataFrame([summary_row]) if isinstance(summary_row, dict) else (
        pd.DataFrame([summary_eval.__dict__]) if summary_eval is not None else pd.DataFrame()
    )

    section_tables = {
        "summary": {
            "summary.parquet": summary_row_df,
        },
        "returns": {
            "mean_return_by_quantile_long_short.parquet": returns_payload.get("long_short", {}).get("mean_return_by_quantile", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "mean_return_by_quantile_std_long_short.parquet": returns_payload.get("long_short", {}).get("mean_return_by_quantile_std", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "mean_return_by_quantile_by_date_long_short.parquet": returns_payload.get("long_short", {}).get("mean_return_by_quantile_by_date", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "primary_quantile_returns_by_date_long_short.parquet": returns_payload.get("long_short", {}).get("primary_quantile_returns_by_date", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "alpha_beta_long_short.parquet": returns_payload.get("long_short", {}).get("alpha_beta", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "mean_return_spread_long_short.parquet": returns_payload.get("long_short", {}).get("mean_return_spread", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "mean_return_spread_std_long_short.parquet": returns_payload.get("long_short", {}).get("mean_return_spread_std") if isinstance(returns_payload.get("long_short"), dict) else None,
            "quantile_statistics_long_short.parquet": returns_payload.get("long_short", {}).get("quantile_statistics", pd.DataFrame()) if isinstance(returns_payload.get("long_short"), dict) else pd.DataFrame(),
            "mean_return_by_quantile_by_group_long_short.parquet": returns_payload.get("long_short", {}).get("mean_return_by_quantile_by_group") if isinstance(returns_payload.get("long_short"), dict) and "mean_return_by_quantile_by_group" in returns_payload.get("long_short", {}) else None,
            "mean_return_by_quantile_long_only.parquet": returns_payload.get("long_only", {}).get("mean_return_by_quantile", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "mean_return_by_quantile_std_long_only.parquet": returns_payload.get("long_only", {}).get("mean_return_by_quantile_std", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "mean_return_by_quantile_by_date_long_only.parquet": returns_payload.get("long_only", {}).get("mean_return_by_quantile_by_date", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "primary_quantile_returns_by_date_long_only.parquet": returns_payload.get("long_only", {}).get("primary_quantile_returns_by_date", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "alpha_beta_long_only.parquet": returns_payload.get("long_only", {}).get("alpha_beta", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "mean_return_spread_long_only.parquet": returns_payload.get("long_only", {}).get("mean_return_spread", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "mean_return_spread_std_long_only.parquet": returns_payload.get("long_only", {}).get("mean_return_spread_std") if isinstance(returns_payload.get("long_only"), dict) else None,
            "quantile_statistics_long_only.parquet": returns_payload.get("long_only", {}).get("quantile_statistics", pd.DataFrame()) if isinstance(returns_payload.get("long_only"), dict) else pd.DataFrame(),
            "mean_return_by_quantile_by_group_long_only.parquet": returns_payload.get("long_only", {}).get("mean_return_by_quantile_by_group") if isinstance(returns_payload.get("long_only"), dict) and "mean_return_by_quantile_by_group" in returns_payload.get("long_only", {}) else None,
            "factor_portfolio_returns.parquet": returns_payload.get("factor_portfolio_returns", pd.DataFrame()),
        },
        "information": {
            "information_coefficient.parquet": information_payload.get("ic", pd.DataFrame()),
            "mean_information_coefficient.parquet": information_payload.get("mean_ic", pd.DataFrame()),
            "monthly_information_coefficient.parquet": information_payload.get("monthly_ic", pd.DataFrame()),
            "information_coefficient_by_group.parquet": information_payload.get("ic_group") if "ic_group" in information_payload else None,
            "mean_information_coefficient_by_group.parquet": information_payload.get("mean_ic_by_group") if "mean_ic_by_group" in information_payload else None,
        },
        "turnover": {
            "turnover.parquet": turnover_payload.get("turnover", pd.DataFrame()),
            "rank_autocorrelation.parquet": turnover_payload.get("rank_autocorr", turnover_payload.get("rank_autocorrelation", pd.DataFrame())),
            "turnover_summary.parquet": turnover_payload.get("turnover_summary", pd.DataFrame()),
        },
        "event_returns": {
            "average_cumulative_return_by_quantile.parquet": event_returns_payload.get("average_cumulative_return_by_quantile", pd.DataFrame()),
            "best_holding_period.parquet": pd.DataFrame([event_returns_payload.get("best_holding_period", {})]) if event_returns_payload.get("best_holding_period") is not None else None,
        },
    }

    section_table_dirs = {
        "summary": summary_dir,
        "returns": returns_dir,
        "information": information_dir,
        "turnover": turnover_dir,
        "event_returns": event_returns_dir,
    }
    table_paths: dict[str, Path] = {}
    for section_name, tables in section_tables.items():
        table_paths.update(_write_section_tables(section_name, section_table_dirs[section_name], tables))

    horizon_frame = horizon_result if horizon_result is not None else pd.DataFrame()
    if horizon_result is not None and not horizon_frame.empty:
        table_paths["horizon_result"] = _write_table(horizon_frame, horizon_dir / "horizon_result.parquet")
        if figure:
            horizon_figure = save_stability_overview(
                pd.DataFrame(),
                pd.DataFrame(),
                horizon_frame,
                horizon_dir / "horizon_result.png",
                title="Horizon Result",
            )
        else:
            horizon_figure = None
    if horizon_summary is not None:
        horizon_summary_df = pd.DataFrame([horizon_summary])
        table_paths["horizon_summary"] = _write_table(horizon_summary_df, horizon_dir / "horizon_summary.parquet")
        horizon_manifest = {
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "factor_column": factor_data.factor_column,
            "date_column": factor_data.date_column,
            "code_column": factor_data.code_column,
            "horizons": list(horizon_frame["horizon"]) if not horizon_frame.empty and "horizon" in horizon_frame.columns else None,
            "analysis_rows": int(len(horizon_frame)),
            "summary": horizon_summary,
        }
        horizon_manifest_path = horizon_dir / "manifest.json"
        horizon_manifest_path.write_text(json.dumps(horizon_manifest, indent=2, default=str), encoding="utf-8")
        table_paths["horizon_manifest"] = horizon_manifest_path

    figure_paths = {}
    if figure:
        if horizon_result is not None and not horizon_frame.empty:
            figure_paths["horizon_result"] = horizon_figure
        for section_name, report in (
            ("returns", returns),
            ("information", information),
            ("turnover", turnover),
            ("event_returns", event_returns),
        ):
            for name, path in report.figure_paths.items():
                if path is not None:
                    figure_paths[f"{section_name}_{name}"] = path
        overview_source = returns.figure_paths.get("quantile_returns_bar")
        if overview_source is None and returns.figure_paths:
            overview_source = next(iter(returns.figure_paths.values()))
        if overview_source is not None and Path(overview_source).exists():
            overview_path = output / "returns_overview.png"
            shutil.copyfile(overview_source, overview_path)
            figure_paths["returns_overview"] = overview_path

    payload = {
        "summary": summary,
        "returns": returns,
        "information": information,
        "turnover": turnover,
        "event_returns": event_returns,
        "horizon_result": horizon_frame if horizon_result is not None else None,
        "horizon_summary": horizon_summary,
    }
    return TigerEvaluationSheet(
        name="full",
        output_dir=output,
        figure_output_dir=output,
        figure_paths={k: v for k, v in figure_paths.items() if v is not None},
        table_paths=table_paths,
        evaluation=summary_eval,
        payload=payload,
    )


__all__ = [
    "TigerEvaluationSheet",
    "create_event_returns_tear_sheet",
    "create_event_study_tear_sheet",
    "create_full_tear_sheet",
    "create_information_tear_sheet",
    "create_returns_tear_sheet",
    "create_summary_tear_sheet",
    "create_turnover_tear_sheet",
    "prepare_shared_artifacts",
]
