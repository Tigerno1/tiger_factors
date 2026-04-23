from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from tiger_factors.factor_evaluation import evaluate_factor_panel


def summarize_factor_evaluation(panel: pd.DataFrame, forward_returns: pd.DataFrame) -> dict[str, float]:
    evaluation = evaluate_factor_panel(panel, forward_returns)
    payload = asdict(evaluation)
    return {key: float(value) for key, value in payload.items()}


def score_factor_panels(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    *,
    score_field: str = "fitness",
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}

    for name, panel in factors.items():
        summary = summarize_factor_evaluation(panel, forward_returns)
        summaries[name] = summary
        scores[name] = float(summary.get(score_field, 0.0))

    return scores, summaries
