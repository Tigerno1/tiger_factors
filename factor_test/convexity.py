from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorConvexityResult:
    method: str
    alpha: float
    table: pd.DataFrame
    n_obs: int
    alpha_hat: float
    beta_hat: float
    gamma_hat: float
    alpha_tstat: float
    beta_tstat: float
    gamma_tstat: float
    alpha_pvalue: float
    beta_pvalue: float
    gamma_pvalue: float
    r_squared: float
    adj_r_squared: float
    positive_convexity: bool
    negative_convexity: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_obs": self.n_obs,
            "alpha_hat": self.alpha_hat,
            "beta_hat": self.beta_hat,
            "gamma_hat": self.gamma_hat,
            "alpha_tstat": self.alpha_tstat,
            "beta_tstat": self.beta_tstat,
            "gamma_tstat": self.gamma_tstat,
            "alpha_pvalue": self.alpha_pvalue,
            "beta_pvalue": self.beta_pvalue,
            "gamma_pvalue": self.gamma_pvalue,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "positive_convexity": self.positive_convexity,
            "negative_convexity": self.negative_convexity,
            "table": self.table.to_dict(orient="records"),
        }


def _coerce_series(values: pd.Series | pd.DataFrame | list[float] | np.ndarray, *, column: str | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if column is not None and column in values.columns:
            series = values[column]
        else:
            numeric = values.select_dtypes(include=[np.number])
            if numeric.empty:
                raise KeyError("DataFrame input must contain at least one numeric column.")
            series = numeric.iloc[:, 0]
    else:
        series = pd.Series(values)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _align_factor_and_market(
    factor_returns: pd.Series | pd.DataFrame | list[float] | np.ndarray,
    market_returns: pd.Series | pd.DataFrame | list[float] | np.ndarray | None,
    *,
    factor_column: str | None = None,
    market_column: str | None = None,
) -> pd.DataFrame:
    if market_returns is None and isinstance(factor_returns, pd.DataFrame):
        frame = factor_returns.copy()
        factor_series = _coerce_series(frame, column=factor_column)
        if market_column is not None and market_column in frame.columns:
            market_series = _coerce_series(frame, column=market_column)
        else:
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                raise ValueError("DataFrame input must contain factor and market return columns.")
            market_series = _coerce_series(numeric.iloc[:, 1])
    else:
        factor_series = _coerce_series(factor_returns, column=factor_column)
        if market_returns is None:
            raise ValueError("market_returns is required unless a two-column DataFrame is supplied.")
        market_series = _coerce_series(market_returns, column=market_column)

    aligned = pd.concat([factor_series.rename("factor"), market_series.rename("market")], axis=1).dropna()
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    return aligned


def factor_convexity_test(
    factor_returns: pd.Series | pd.DataFrame | list[float] | np.ndarray,
    market_returns: pd.Series | pd.DataFrame | list[float] | np.ndarray | None = None,
    *,
    factor_column: str | None = None,
    market_column: str | None = None,
    alpha: float = 0.05,
    maxlags: int = 5,
    use_hac: bool = True,
) -> FactorConvexityResult:
    aligned = _align_factor_and_market(
        factor_returns,
        market_returns,
        factor_column=factor_column,
        market_column=market_column,
    )
    if aligned.empty or len(aligned) < 5:
        empty = pd.DataFrame(
            columns=["term", "estimate", "std_error", "t_stat", "p_value", "ci_low", "ci_high"]
        )
        return FactorConvexityResult(
            method="quadratic_regression",
            alpha=float(alpha),
            table=empty,
            n_obs=int(len(aligned)),
            alpha_hat=0.0,
            beta_hat=0.0,
            gamma_hat=0.0,
            alpha_tstat=0.0,
            beta_tstat=0.0,
            gamma_tstat=0.0,
            alpha_pvalue=1.0,
            beta_pvalue=1.0,
            gamma_pvalue=1.0,
            r_squared=0.0,
            adj_r_squared=0.0,
            positive_convexity=False,
            negative_convexity=False,
        )

    import statsmodels.api as sm

    design = pd.DataFrame(
        {
            "market": aligned["market"],
            "market_sq": aligned["market"] ** 2,
        },
        index=aligned.index,
    )
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(aligned["factor"], design)
    fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)}) if use_hac else model.fit()

    params = fitted.params
    bse = fitted.bse
    tvalues = fitted.tvalues
    pvalues = fitted.pvalues
    conf_int = fitted.conf_int(alpha=float(alpha))
    conf_int.index = params.index
    result_table = pd.DataFrame(
        {
            "term": params.index.astype(str),
            "estimate": params.to_numpy(dtype=float),
            "std_error": bse.reindex(params.index).to_numpy(dtype=float),
            "t_stat": tvalues.reindex(params.index).to_numpy(dtype=float),
            "p_value": pvalues.reindex(params.index).to_numpy(dtype=float),
            "ci_low": conf_int.iloc[:, 0].to_numpy(dtype=float),
            "ci_high": conf_int.iloc[:, 1].to_numpy(dtype=float),
        }
    )

    alpha_hat = float(params.get("const", 0.0))
    beta_hat = float(params.get("market", 0.0))
    gamma_hat = float(params.get("market_sq", 0.0))
    alpha_tstat = float(tvalues.get("const", 0.0))
    beta_tstat = float(tvalues.get("market", 0.0))
    gamma_tstat = float(tvalues.get("market_sq", 0.0))
    alpha_pvalue = float(pvalues.get("const", 1.0))
    beta_pvalue = float(pvalues.get("market", 1.0))
    gamma_pvalue = float(pvalues.get("market_sq", 1.0))

    return FactorConvexityResult(
        method="quadratic_regression",
        alpha=float(alpha),
        table=result_table,
        n_obs=int(fitted.nobs),
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        alpha_tstat=alpha_tstat,
        beta_tstat=beta_tstat,
        gamma_tstat=gamma_tstat,
        alpha_pvalue=alpha_pvalue,
        beta_pvalue=beta_pvalue,
        gamma_pvalue=gamma_pvalue,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        positive_convexity=bool(gamma_hat > 0.0),
        negative_convexity=bool(gamma_hat < 0.0),
    )


__all__ = [
    "FactorConvexityResult",
    "factor_convexity_test",
]
