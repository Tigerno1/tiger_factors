from __future__ import annotations

from collections.abc import Callable

import pandas as pd


FEATURE_TEMPLATE_COUNT = 318


def _template_not_implemented(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    def _template(data: pd.DataFrame, **kwargs: object) -> pd.Series:
        raise NotImplementedError(
            f"{name} is a placeholder template and does not yet have an implemented formula."
        )

    _template.__name__ = name
    _template.__qualname__ = name
    _template.__doc__ = (
        f"Placeholder for {name}. "
        "The upstream OpenAssetPricing formula has not been ported into Tiger yet."
    )
    return _template


factor_template_names: tuple[str, ...] = tuple(
    f"factor_{index:03d}" for index in range(1, FEATURE_TEMPLATE_COUNT + 1)
)

for _template_name in factor_template_names:
    globals()[_template_name] = _template_not_implemented(_template_name)


def available_factor_templates() -> tuple[str, ...]:
    return factor_template_names


def get_factor_template(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    try:
        return globals()[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown factor template: {name!r}") from exc


__all__ = [
    "FEATURE_TEMPLATE_COUNT",
    "available_factor_templates",
    "factor_template_names",
    "get_factor_template",
    *factor_template_names,
]
