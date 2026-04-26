from __future__ import annotations

import pandas as pd

from tiger_factors.factor_screener.batch_selection import FactorSelectionMode
from tiger_factors.factor_screener.batch_selection import collect_candidate_panels
from tiger_factors.factor_screener.batch_selection import collect_candidate_returns
from tiger_factors.factor_screener.batch_selection import greedy_select_by_return_gain
from tiger_factors.factor_screener.factor_screener import FactorScreener
from tiger_factors.factor_screener.factor_screener import FactorScreenerResult
from tiger_factors.factor_screener.selection import FactorMarginalSelectionConfig
from tiger_factors.factor_screener.selection import select_factors_by_marginal_gain
from tiger_factors.factor_screener.selection import select_non_redundant_factors


def select_global_factor_keys(
    screeners: list[tuple[str, FactorScreener]],
    results: list[tuple[str, FactorScreenerResult]],
    selection_summary: pd.DataFrame,
    *,
    selection_mode: str,
    cross_spec_selection_threshold: float | None,
    cross_spec_selection_score_field: str,
    marginal_selection_config: FactorMarginalSelectionConfig,
    return_gain_config,
) -> tuple[tuple[str, ...], tuple[str, ...], pd.DataFrame]:
    global_selected_keys: tuple[str, ...] = ()
    global_selected_names: tuple[str, ...] = ()
    if cross_spec_selection_threshold is None or selection_summary.empty:
        return global_selected_keys, global_selected_names, selection_summary

    candidate_panels, candidate_frame, candidate_scores = collect_candidate_panels(
        screeners,
        selection_summary,
        score_field=cross_spec_selection_score_field,
    )
    candidate_returns, _ = collect_candidate_returns(screeners, results, selection_summary)

    if candidate_panels and not candidate_frame.empty:
        if selection_mode == FactorSelectionMode.CONDITIONAL:
            selected_keys = select_factors_by_marginal_gain(
                candidate_panels,
                candidate_frame,
                config=marginal_selection_config,
                key_column="batch_factor_key",
            )
        elif selection_mode == FactorSelectionMode.RETURN_GAIN:
            selected_keys = greedy_select_by_return_gain(
                candidate_returns,
                candidate_scores,
                candidate_frame,
                config=return_gain_config,
            )
        else:
            selected_keys = select_non_redundant_factors(
                candidate_panels,
                candidate_scores,
                threshold=float(cross_spec_selection_threshold),
            )
        global_selected_keys = tuple(selected_keys)
        global_selected_names = tuple(key.split("::", 1)[-1] for key in selected_keys)

        if not selection_summary.empty and "batch_factor_key" in selection_summary.columns:
            selection_summary = selection_summary.copy()
            selection_summary["global_selected"] = selection_summary["batch_factor_key"].astype(str).isin(selected_keys)

    return global_selected_keys, global_selected_names, selection_summary


__all__ = ["select_global_factor_keys"]
