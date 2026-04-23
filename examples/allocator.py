# allocator.py  (PRODUCTION DEFAULTS, STRATEGY-LEVEL FoF)
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from scipy.optimize import minimize

# Optional (recommended) shrinkage covariance
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ============================================================
# Dynamic lambda (corr spike + corr crash + resonance)
# ============================================================
def rolling_avg_correlation(R: pd.DataFrame, window=12) -> pd.Series:
    avg_corr, idx = [], []
    for i in range(window, len(R)):
        corr = R.iloc[i-window:i].corr()
        off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_corr.append(np.nanmean(off_diag))
        idx.append(R.index[i])
    return pd.Series(avg_corr, index=idx, name="avg_corr")


def rolling_concentration_g(R: pd.DataFrame, window=12, eps=1e-12) -> pd.Series:
    g_list, idx = [], []
    for i in range(window, len(R)):
        X = R.iloc[i-window:i].dropna(how="any", axis=1)
        if X.shape[1] < 2:
            g_list.append(np.nan); idx.append(R.index[i]); continue
        Sigma = X.cov().values
        tr = float(np.trace(Sigma))
        if tr <= eps:
            g = np.nan
        else:
            eigs = np.linalg.eigvalsh(Sigma)
            g = float(np.max(eigs) / tr)
        g_list.append(g); idx.append(R.index[i])
    return pd.Series(g_list, index=idx, name="g_concentration")


def _norm_01_by_rolling_quantiles(x: pd.Series, q_window=60, q_lo=0.2, q_hi=0.8, eps=1e-12) -> pd.Series:
    lo = x.rolling(q_window).quantile(q_lo)
    hi = x.rolling(q_window).quantile(q_hi)
    z = (x - lo) / (hi - lo + eps)
    return z.clip(0.0, 1.0)


def build_lambda_series_corr_crash_and_spike(
    R: pd.DataFrame,
    corr_window=12,
    g_window=12,
    q_window=60,
    lam_min=0.0,
    lam_max=4.0,
    alpha=0.45,
    beta=0.35,
    gamma=0.15,
    delta=0.05,
    eps=1e-12
) -> Tuple[pd.Series, pd.DataFrame]:
    c = rolling_avg_correlation(R, window=corr_window)
    g = rolling_concentration_g(R, window=g_window, eps=eps)

    d_c_abs = c.diff().abs()
    d_g_abs = g.diff().abs()

    z_c  = _norm_01_by_rolling_quantiles(c,        q_window=q_window, eps=eps)
    z_dc = _norm_01_by_rolling_quantiles(d_c_abs,  q_window=q_window, eps=eps)
    z_g  = _norm_01_by_rolling_quantiles(g,        q_window=q_window, eps=eps)
    z_dg = _norm_01_by_rolling_quantiles(d_g_abs,  q_window=q_window, eps=eps)

    score = (alpha * z_c + beta * z_dc + gamma * z_g + delta * z_dg).clip(0.0, 1.0)
    lam_raw = (lam_min + (lam_max - lam_min) * score).clip(lam_min, lam_max)

    lam_raw = lam_raw.reindex(R.index).ffill().fillna(lam_min)
    lam_use = lam_raw.shift(1).fillna(lam_min)  # no look-ahead

    diag = pd.DataFrame({
        "avg_corr": c.reindex(R.index).ffill(),
        "d_corr_abs": d_c_abs.reindex(R.index).ffill(),
        "g_conc": g.reindex(R.index).ffill(),
        "d_g_abs": d_g_abs.reindex(R.index).ffill(),
        "score": score.reindex(R.index).ffill(),
        "lam_raw": lam_raw,
        "lam_use": lam_use,
    }, index=R.index)

    return lam_use, diag


# ============================================================
# VIX -> rho(t) fear score (no look-ahead)
# ============================================================
def build_vix_fear_score(
    vix_m: pd.Series,
    q_window=60,
    q_lo=0.2,
    q_hi=0.8,
    eps=1e-12
) -> pd.Series:
    lo = vix_m.rolling(q_window).quantile(q_lo)
    hi = vix_m.rolling(q_window).quantile(q_hi)
    fear_raw = ((vix_m - lo) / (hi - lo + eps)).clip(0.0, 1.0)
    fear_use = fear_raw.shift(1).fillna(0.0)
    return fear_use


# ============================================================
# Robust covariance (production default)
# ============================================================
def estimate_covariance(R_win: pd.DataFrame, method: str = "lw") -> np.ndarray:
    """
    method:
      - "sample": sample covariance
      - "lw": LedoitWolf shrinkage (recommended)
    """
    X = R_win.values
    if method == "sample" or (method == "lw" and not _HAS_SKLEARN):
        return np.cov(X, rowvar=False, ddof=1)
    lw = LedoitWolf().fit(X)
    return lw.covariance_


def portfolio_var(theta: np.ndarray, Sigma: np.ndarray) -> float:
    th = theta.reshape(-1, 1)
    return float(th.T @ Sigma @ th)

def portfolio_sig(theta: np.ndarray, Sigma: np.ndarray) -> float:
    return float(np.sqrt(portfolio_var(theta, Sigma) + 1e-18))


# ============================================================
# Production config
# ============================================================
@dataclass
class StrategyLevelConfig:
    # rolling window
    L: int = 36

    # dynamic lambda params
    lam_max: float = 4.0
    corr_window: int = 12
    g_window: int = 12
    q_window: int = 60

    # VIX->rho
    rho_base: float = 0.0
    rho_max: float = 0.5

    # covariance estimate
    cov_method: str = "lw"   # "lw" or "sample"

    # production guardrails
    cap_default: float = 0.50
    cap_overrides: Optional[Dict[str, float]] = None  # {"Momentum":0.45,...}

    # turnover penalty (L1)
    kappa_turnover: float = 0.05

    # volatility target penalty (soft)
    use_vol_target: bool = True
    vol_target_annual: float = 0.12
    vol_penalty_eta: float = 10.0

    # optional group constraints
    group_min: Optional[Dict[str, float]] = None  # {"def":0.2}
    group_max: Optional[Dict[str, float]] = None  # {"risky":0.75}
    group_map: Optional[Dict[str, str]] = None    # strategy_name -> group


def _build_caps(strategies: List[str], cfg: StrategyLevelConfig) -> np.ndarray:
    caps = np.array([cfg.cap_default] * len(strategies), dtype=float)
    if cfg.cap_overrides:
        for i, s in enumerate(strategies):
            if s in cfg.cap_overrides:
                caps[i] = float(cfg.cap_overrides[s])
    caps = np.clip(caps, 0.0, 1.0)

    # feasibility guard: if sum(caps)<1, auto-loosen proportionally
    if caps.sum() < 1.0 - 1e-9:
        caps = caps / max(caps.sum(), 1e-12)
    return caps


def _group_constraints(strategies: List[str], cfg: StrategyLevelConfig):
    cons = []
    if not cfg.group_map:
        return cons

    groups = set(cfg.group_map.values())
    for g in groups:
        idx = [i for i, s in enumerate(strategies) if cfg.group_map.get(s) == g]
        if not idx:
            continue

        if cfg.group_min and g in cfg.group_min:
            gmin = float(cfg.group_min[g])
            cons.append({"type": "ineq", "fun": (lambda x, idx=idx, gmin=gmin: np.sum(x[idx]) - gmin)})

        if cfg.group_max and g in cfg.group_max:
            gmax = float(cfg.group_max[g])
            cons.append({"type": "ineq", "fun": (lambda x, idx=idx, gmax=gmax: gmax - np.sum(x[idx]))})

    return cons


# ============================================================
# Production theta solver (strategy-level)
# ============================================================
def solve_theta_over_time_strategy_level(
    R: pd.DataFrame,
    lam_series: Optional[pd.Series],
    rho_series: Optional[pd.Series],
    cfg: StrategyLevelConfig,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Production objective:
      J = θᵀμ - 0.5*(1+λ_t)*θᵀΣθ - ρ_t*sqrt(θᵀΣθ)
          - κ * ||θ-θ_prev||_1
          - η * max(0, σ(θ) - σ_target)^2
    Constraints:
      sum θ = 1
      0 <= θ_i <= cap_i
      + optional group min/max
    """
    strategies = list(R.columns)
    K = len(strategies)
    dates = list(R.index)

    caps = _build_caps(strategies, cfg)

    theta_by_date = {}
    diag_by_date = {}

    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    cons += _group_constraints(strategies, cfg)

    bounds = [(0.0, float(caps[i])) for i in range(K)]

    theta_prev = np.ones(K) / K

    sig_target_m = cfg.vol_target_annual / np.sqrt(12.0)  # annual -> monthly

    for i, dt in enumerate(dates):
        if i < cfg.L:
            continue

        dt_start = dates[i - cfg.L]
        dt_end = dates[i - 1]

        R_win = R.loc[dt_start:dt_end]
        mu = R_win.mean().values.reshape(-1, 1)
        Sigma = estimate_covariance(R_win, method=cfg.cov_method)

        lam_t = float(lam_series.loc[dt]) if (lam_series is not None and dt in lam_series.index) else 0.0
        rho_t = float(rho_series.loc[dt]) if (rho_series is not None and dt in rho_series.index) else cfg.rho_base

        def J(theta):
            th = theta.reshape(-1, 1)
            var = float(th.T @ Sigma @ th)
            sig = float(np.sqrt(var + 1e-18))

            attack = float(th.T @ mu)
            defense = 0.5 * (1.0 + lam_t) * var
            audit = rho_t * sig

            turn = cfg.kappa_turnover * float(np.sum(np.abs(theta - theta_prev)))

            vol_pen = 0.0
            if cfg.use_vol_target:
                excess = max(0.0, sig - sig_target_m)
                vol_pen = cfg.vol_penalty_eta * (excess ** 2)

            return attack - defense - audit - turn - vol_pen

        res = minimize(
            fun=lambda x: -J(x),
            x0=theta_prev.copy(),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12}
        )

        theta_star = res.x
        theta_by_date[dt] = theta_star

        var = portfolio_var(theta_star, Sigma)
        sig = float(np.sqrt(var + 1e-18))

        diag = {
            "ok": bool(res.success),
            "status": int(getattr(res, "status", -1)),
            "message": str(getattr(res, "message", "")),
            "obj": float(J(theta_star)),
            "sig_model": sig,
            "var_model": float(var),
            "lam_t": float(lam_t),
            "rho_t": float(rho_t),
            "turnover_l1": float(np.sum(np.abs(theta_star - theta_prev))),
            "sig_target_m": float(sig_target_m) if cfg.use_vol_target else np.nan,
        }
        for j, s in enumerate(strategies):
            diag[f"theta_{s}"] = float(theta_star[j])
        diag_by_date[dt] = diag

        if verbose and not res.success:
            print("FAIL @", dt, res.message)

        theta_prev = theta_star.copy()

    theta_df = pd.DataFrame(theta_by_date, index=strategies).T.sort_index()
    diag_df = pd.DataFrame(diag_by_date).T.sort_index()
    return theta_df, diag_df


def compute_portfolio_returns_strategy_level(theta_df: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    R_aligned = R.reindex(theta_df.index)
    return (theta_df * R_aligned).sum(axis=1)


# ============================================================
# Metrics
# ============================================================
def compute_cagr(R_port: pd.Series) -> Tuple[float, float, float]:
    eq = (1.0 + R_port.fillna(0.0)).cumprod()
    final_equity = float(eq.iloc[-1])
    total_return = final_equity - 1.0
    years = len(R_port.dropna()) / 12.0 if len(R_port.dropna()) > 0 else 0.0
    cagr = final_equity ** (1 / years) - 1.0 if years > 0 else np.nan
    return final_equity, total_return, cagr


def max_drawdown(R: pd.Series) -> float:
    eq = (1.0 + R.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def sharpe_monthly(R: pd.Series, rf: float = 0.0) -> float:
    x = R.dropna() - rf
    if len(x) < 3:
        return np.nan
    sd = x.std(ddof=1)
    if sd <= 0:
        return np.nan
    return float(x.mean() / sd * np.sqrt(12.0))


# ============================================================
# High-level runner (strategy-level)
# ============================================================
def run_allocator_strategy_level(
    R: pd.DataFrame,
    vix_m: Optional[pd.Series],
    cfg: StrategyLevelConfig
) -> Dict[str, object]:
    # lambda
    lam_series, lam_diag = build_lambda_series_corr_crash_and_spike(
        R=R,
        corr_window=cfg.corr_window,
        g_window=cfg.g_window,
        q_window=cfg.q_window,
        lam_min=0.0,
        lam_max=cfg.lam_max,
    )

    # rho from VIX
    if vix_m is not None:
        vix_m = vix_m.reindex(R.index).ffill()
        vix_fear = build_vix_fear_score(vix_m, q_window=cfg.q_window)
        rho_series = (cfg.rho_base + (cfg.rho_max - cfg.rho_base) * vix_fear).reindex(R.index).ffill().fillna(cfg.rho_base)
    else:
        vix_fear = None
        rho_series = None

    theta_df, diag_df = solve_theta_over_time_strategy_level(
        R=R,
        lam_series=lam_series,
        rho_series=rho_series,
        cfg=cfg,
        verbose=False
    )

    R_port = compute_portfolio_returns_strategy_level(theta_df, R).rename("R_port")

    final_eq, total_ret, cagr = compute_cagr(R_port)
    mdd = max_drawdown(R_port)
    shp = sharpe_monthly(R_port)

    return {
        "lam_series": lam_series,
        "lam_diag": lam_diag,
        "vix_fear": vix_fear,
        "rho_series": rho_series,
        "theta_df": theta_df,
        "diag_df": diag_df,
        "R_port": R_port,
        "summary": {
            "final_equity": final_eq,
            "total_return": total_ret,
            "cagr": cagr,
            "max_drawdown": mdd,
            "sharpe": shp,
        }
    }
