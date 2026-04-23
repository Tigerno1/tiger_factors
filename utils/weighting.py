from __future__ import annotations

import numpy as np


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    total = float(sum(weights.values()))
    if total == 0:
        equal = 1.0 / len(weights)
        return {key: equal for key in weights}
    return {key: float(value) / total for key, value in weights.items()}


def score_to_weights(
    scores: dict[str, float],
    *,
    selected: list[str] | None = None,
    method: str = "positive",
    temperature: float = 1.0,
) -> dict[str, float]:
    names = selected or list(scores.keys())
    if not names:
        return {}

    values = np.array([float(scores.get(name, 0.0)) for name in names], dtype=float)

    if method == "equal":
        raw = {name: 1.0 for name in names}
        return normalize_weights(raw)

    if method == "softmax":
        temp = max(float(temperature), 1e-6)
        logits = values / temp
        logits = logits - np.nanmax(logits)
        expv = np.exp(logits)
        raw = {name: float(value) for name, value in zip(names, expv)}
        return normalize_weights(raw)

    if method == "positive":
        shifted = values - np.nanmin(values)
        shifted = shifted + 1e-12
        raw = {name: float(value) for name, value in zip(names, shifted)}
        return normalize_weights(raw)

    raise ValueError("method must be one of: 'equal', 'positive', 'softmax'.")


def apply_weight_bounds(
    weights: dict[str, float],
    *,
    min_weight: float | None = None,
    max_weight: float | None = None,
    total: float = 1.0,
    max_iter: int = 128,
    tol: float = 1e-9,
) -> dict[str, float]:
    if not weights:
        return {}

    names = list(weights.keys())
    base = np.array([float(weights[name]) for name in names], dtype=float)
    base = np.where(np.isfinite(base), np.maximum(base, 0.0), 0.0)

    target_total = float(total)
    if target_total <= 0:
        raise ValueError("total must be positive.")

    if base.sum() <= tol:
        base[:] = 1.0 / len(base)
    else:
        base = base / base.sum()

    current = base * target_total
    min_w = 0.0 if min_weight is None else float(min_weight)
    max_w = None if max_weight is None else float(max_weight)

    if min_w < 0:
        raise ValueError("min_weight must be non-negative.")
    if max_w is not None and max_w <= 0:
        raise ValueError("max_weight must be positive.")
    if max_w is not None and max_w + tol < min_w:
        raise ValueError("max_weight must be >= min_weight.")
    if min_w * len(names) > target_total + tol:
        raise ValueError("min_weight is infeasible for the number of selected weights.")
    if max_w is not None and max_w * len(names) < target_total - tol:
        raise ValueError("max_weight is infeasible for the number of selected weights.")

    locked = np.zeros(len(names), dtype=bool)
    for _ in range(max_iter):
        changed = False

        if min_weight is not None:
            low_mask = (~locked) & (current < min_w - tol)
            if low_mask.any():
                current[low_mask] = min_w
                locked[low_mask] = True
                changed = True

        if max_w is not None:
            high_mask = (~locked) & (current > max_w + tol)
            if high_mask.any():
                current[high_mask] = max_w
                locked[high_mask] = True
                changed = True

        free = ~locked
        if not free.any():
            break

        residual = target_total - float(current[locked].sum())
        if residual < -tol:
            raise ValueError("Weight bounds are infeasible under the requested total.")

        free_base = base[free]
        if free_base.sum() <= tol:
            current[free] = residual / float(free.sum())
        else:
            current[free] = residual * free_base / free_base.sum()

        if not changed and abs(float(current.sum()) - target_total) <= tol:
            break

    return {name: float(value) for name, value in zip(names, current)}
