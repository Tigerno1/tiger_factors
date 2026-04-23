# main.py  (STRATEGY-LEVEL PRODUCTION DEFAULTS)
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from tiger_factors.examples.strategies import (
    StrategyBundle,
    ETFMomBlendStrategy,
    StockReversalStrategy,
    SingleAssetStrategy,
)
from tiger_factors.examples.allocator import (
    StrategyLevelConfig,
    run_allocator_strategy_level,
)


# -----------------------------
# Data helpers
# -----------------------------
def download_prices_close(tickers, start, end, auto_adjust=True):
    px = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")


def download_ohlc(tickers, start, end, auto_adjust=True):
    df = yf.download(
        tickers, start=start, end=end,
        auto_adjust=auto_adjust, progress=False, group_by="ticker"
    )
    if isinstance(tickers, str):
        tickers = [tickers]
    out = {}
    for t in tickers:
        d = df[t].copy() if isinstance(df.columns, pd.MultiIndex) else df.copy()
        d = d.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        out[t] = d[["open","high","low","close"]].dropna(how="all").sort_index()
    return out


def to_monthly_returns(px_daily_close: pd.DataFrame) -> pd.DataFrame:
    px_m = px_daily_close.resample("M").last()
    rets_m = px_m.pct_change()
    return rets_m.dropna(how="all")


def download_vix_monthly(start, end):
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"].dropna()
    vix_m = vix.resample("M").last()
    vix_m.name = "VIX"
    return vix_m


# -----------------------------
# Plot helpers
# -----------------------------
def plot_equity_curve(R_port: pd.Series, title="Equity Curve"):
    eq = (1 + R_port.fillna(0.0)).cumprod()
    plt.figure(figsize=(11, 4))
    eq.plot(linewidth=2.5)
    plt.grid(True)
    plt.title(title)
    plt.show()


def plot_theta(theta_df: pd.DataFrame, title="theta over time"):
    plt.figure(figsize=(11, 4))
    theta_df.plot(ax=plt.gca())
    plt.grid(True)
    plt.title(title)
    plt.show()


def plot_risk_terms(diag_df: pd.DataFrame):
    cols = [c for c in ["sig_model", "lam_t", "rho_t", "turnover_l1", "sig_target_m"] if c in diag_df.columns]
    if not cols:
        return
    plt.figure(figsize=(11, 4))
    diag_df[cols].plot(ax=plt.gca())
    plt.grid(True)
    plt.title("Risk/Control Terms")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) Universe
    stock_tickers = [
        "AXP", "BA", "CAT", "CVS", "DIS",
        "F", "GE", "GM", "GS", "HON",
        "IBM", "JCI", "KHC", "MCD", "MMM",
        "NKE", "PSX", "TRV", "USB", "ZTS"
    ]
    etf_tickers = ["SPY", "QQQ", "TLT", "SHY", "GLD"]

    start = "2016-01-01"
    end   = "2024-01-01"

    # 2) Download data
    px_stock = download_prices_close(stock_tickers, start, end, auto_adjust=True)
    ohlc_etf = download_ohlc(etf_tickers, start, end, auto_adjust=True)
    px_etf = pd.DataFrame({e: ohlc_etf[e]["close"] for e in etf_tickers}).dropna(how="all")

    px_all = pd.concat([px_stock, px_etf], axis=1).dropna(how="any")
    assets_all = list(px_all.columns)

    returns_m_all = to_monthly_returns(px_all)

    print("Assets:", len(assets_all), "months:", len(returns_m_all))
    print("Date range:", px_all.index.min().date(), "->", px_all.index.max().date())

    # 3) Build strategy pool (StrategyBundle outputs R)
    strategies = [
        ETFMomBlendStrategy(etf_list=etf_tickers, name="Momentum", threshold=1.5),
        StockReversalStrategy(stock_universe=stock_tickers, name="Reversal", bottom_q=0.2),
        *[SingleAssetStrategy(e) for e in etf_tickers],
    ]

    bundle = StrategyBundle(strategies=strategies, align_mode="intersection")
    out = bundle.build(
        prices_daily_close=px_all,
        ohlc_daily=ohlc_etf,
        returns_m_all=returns_m_all,
        assets_all=assets_all,
        start=start,
        end=end
    )

    R = out.R
    print("Strategy universe:", out.strategies)
    print("R shape:", R.shape)

    # 4) VIX
    vix_m = download_vix_monthly(start, end)

    # 5) Production default config (tweak here)
    cfg = StrategyLevelConfig(
        L=36,
        lam_max=4.0,
        rho_base=0.0,
        rho_max=0.5,
        cov_method="lw",                 # shrinkage covariance
        cap_default=0.50,                # per-strategy cap
        cap_overrides={                  # optional
            "Momentum": 0.45,
            "Reversal": 0.45,
        },
        kappa_turnover=0.05,             # turnover penalty
        use_vol_target=True,
        vol_target_annual=0.12,
        vol_penalty_eta=10.0,

        # optional grouping example (uncomment)
        # group_map={"Momentum":"risky","Reversal":"risky","SPY":"risky","QQQ":"risky",
        #            "TLT":"def","SHY":"def","GLD":"def"},
        # group_max={"risky":0.75},
        # group_min={"def":0.15},
    )

    # 6) Run allocator (strategy-level)
    res = run_allocator_strategy_level(R=R, vix_m=vix_m, cfg=cfg)

    theta_df = res["theta_df"]
    diag_df  = res["diag_df"]
    R_port   = res["R_port"]
    summary  = res["summary"]

    print("\n=== Summary ===")
    print(f"final_equity : {summary['final_equity']:.3f}x")
    print(f"total_return : {summary['total_return']:.2%}")
    print(f"cagr         : {summary['cagr']:.2%}")
    print(f"max_drawdown : {summary['max_drawdown']:.2%}")
    print(f"sharpe       : {summary['sharpe']:.3f}")

    # 7) Plots
    plot_equity_curve(R_port, title="Strategy-Level Theta Portfolio (Production Defaults)")
    plot_theta(theta_df, title="theta(t) over strategies")
    plot_risk_terms(diag_df)

    # Compare with underlying strategies
    eq_strat = (1 + R.reindex(R_port.index)).cumprod()
    plt.figure(figsize=(11, 5))
    ((1 + R_port.fillna(0.0)).cumprod()).plot(label="Theta", linewidth=3)
    for c in eq_strat.columns:
        eq_strat[c].plot(label=c, alpha=0.6)
    plt.legend()
    plt.grid(True)
    plt.title("Theta vs Underlying Strategies")
    plt.show()
