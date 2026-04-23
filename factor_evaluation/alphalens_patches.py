from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchResult:
    name: str
    applied: bool
    reason: str = ""


_FORWARD_RETURNS_PATCHED = False
_ALPHA_BETA_PATCHED = False


def _get_alphalens_module(al: Any | None = None) -> Any:
    if al is not None:
        return al
    try:
        import alphalens as imported_al  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("alphalens is not installed in this environment.") from e
    return imported_al


def patch_forward_returns_freq(
    al: Any | None = None,
    *,
    force: bool = False,
) -> PatchResult:
    """
    Patch alphalens.utils.compute_forward_returns for pandas compatibility.

    What this patch does:
    1. Removes direct assignment to df.index.levels[0].freq
    2. Rewrites pct_change(...) to use fill_method=None
    """
    global _FORWARD_RETURNS_PATCHED

    if _FORWARD_RETURNS_PATCHED and not force:
        return PatchResult(
            name="forward_returns_freq",
            applied=False,
            reason="already patched",
        )

    al = _get_alphalens_module(al)
    from alphalens import utils as al_utils  # type: ignore

    original = al_utils.compute_forward_returns

    try:
        source = inspect.getsource(original)
    except OSError as e:
        raise RuntimeError("Unable to read source for alphalens.utils.compute_forward_returns") from e

    patched_source = source
    patched_source = patched_source.replace(
        "    df.index.levels[0].freq = freq\n",
        "    # Tiger compatibility: skip freq pinning for pandas 3.\n",
    )
    patched_source = patched_source.replace(
        "prices.pct_change(period)",
        "prices.pct_change(period, fill_method=None)",
    )
    patched_source = patched_source.replace(
        "prices.pct_change()",
        "prices.pct_change(fill_method=None)",
    )

    if patched_source == source and not force:
        return PatchResult(
            name="forward_returns_freq",
            applied=False,
            reason="no matching source pattern found",
        )

    namespace: dict[str, object] = {}
    exec(patched_source, al_utils.__dict__, namespace)
    patched_fn = namespace["compute_forward_returns"]

    setattr(patched_fn, "_tiger_patch_name", "forward_returns_freq")
    setattr(patched_fn, "_tiger_original", original)

    al_utils.compute_forward_returns = patched_fn  # type: ignore[assignment]
    _FORWARD_RETURNS_PATCHED = True

    return PatchResult(
        name="forward_returns_freq",
        applied=True,
        reason="patched successfully",
    )


def patch_alpha_beta(
    al: Any | None = None,
    *,
    force: bool = False,
) -> PatchResult:
    """
    Patch alphalens.performance.factor_alpha_beta with a safer implementation.

    What this patch does:
    1. Filters non-finite x/y values before OLS
    2. Returns NaN alpha/beta when valid samples are insufficient
    """
    global _ALPHA_BETA_PATCHED

    if _ALPHA_BETA_PATCHED and not force:
        return PatchResult(
            name="alpha_beta",
            applied=False,
            reason="already patched",
        )

    al = _get_alphalens_module(al)

    import numpy as np
    import pandas as pd
    from alphalens import performance as al_performance  # type: ignore
    from alphalens import utils as al_utils  # type: ignore
    from statsmodels.api import OLS, add_constant

    original = al_performance.factor_alpha_beta

    def _safe_factor_alpha_beta(
        factor_data,
        returns=None,
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
    ):
        if returns is None:
            returns = al_performance.factor_returns(
                factor_data,
                demeaned=demeaned,
                group_adjust=group_adjust,
                equal_weight=equal_weight,
            )

        universe_ret = (
            factor_data.groupby(level="date")[al_utils.get_forward_returns_columns(factor_data.columns)]
            .mean()
            .reindex(returns.index, axis=0)
        )

        if isinstance(returns, pd.Series):
            returns.name = universe_ret.columns.values[0]
            returns = pd.DataFrame(returns)

        alpha_beta = pd.DataFrame()

        for period in returns.columns.values:
            x = universe_ret[period].values
            y = returns[period].values

            valid_mask = np.isfinite(x) & np.isfinite(y)
            if valid_mask.sum() < 2:
                alpha_beta.loc["Ann. alpha", period] = np.nan
                alpha_beta.loc["beta", period] = np.nan
                continue

            x = add_constant(x[valid_mask])
            y = y[valid_mask]
            reg_fit = OLS(y, x, missing="drop").fit()

            try:
                alpha, beta = reg_fit.params
            except ValueError:
                alpha_beta.loc["Ann. alpha", period] = np.nan
                alpha_beta.loc["beta", period] = np.nan
            else:
                freq_adjust = pd.Timedelta("252Days") / pd.Timedelta(period)
                alpha_beta.loc["Ann. alpha", period] = (1 + alpha) ** freq_adjust - 1
                alpha_beta.loc["beta", period] = beta

        return alpha_beta

    setattr(_safe_factor_alpha_beta, "_tiger_patch_name", "alpha_beta")
    setattr(_safe_factor_alpha_beta, "_tiger_original", original)

    al_performance.factor_alpha_beta = _safe_factor_alpha_beta  # type: ignore[assignment]
    _ALPHA_BETA_PATCHED = True

    return PatchResult(
        name="alpha_beta",
        applied=True,
        reason="patched successfully",
    )


def apply_default_patches(
    al: Any | None = None,
    *,
    force: bool = False,
) -> list[PatchResult]:
    """
    Apply the default set of tiger compatibility patches for alphalens.
    """
    al = _get_alphalens_module(al)

    results = [
        patch_forward_returns_freq(al, force=force),
        patch_alpha_beta(al, force=force),
    ]
    return results