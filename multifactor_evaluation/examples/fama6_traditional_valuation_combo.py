from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.utils.combine import blend_factor_panels
from tiger_factors.utils.combine import factor_correlation_matrix
from tiger_factors.utils.scoring import score_factor_panels
from tiger_factors.utils.weighting import score_to_weights
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest


PRICE_PATH = Path("/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet")
FINANCIAL_MANIFEST = Path(
    "/Volumes/Quant_Disk/evaluation/financial_factors/ttm/base/ttm_base_financial_factors/ttm_base_financial_factors_manifest.json"
)
VALUATION_MANIFEST = Path(
    "/Volumes/Quant_Disk/factor/valuation/ttm/base/ttm_base_valuation_factors/ttm_base_valuation_factors_manifest.json"
)
OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/fama6_traditional_valuation_combo")

DEFAULT_FORWARD_DAYS = 21
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_LONG_PCT = 0.20
DEFAULT_TRANSACTION_COST_BPS = 8.0
DEFAULT_SLIPPAGE_BPS = 4.0


@dataclass(frozen=True)
class LoadedFactor:
    label: str
    family: str
    factor_key: str
    direction: float
    panel: pd.DataFrame


def _load_manifest_map(manifest_path: Path) -> dict[str, Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    factor_map: dict[str, Path] = {}
    for item in manifest.get("files", []):
        factor_key = str(item["factor"])
        factor_map[factor_key] = Path(str(item["parquet_path"]))
    if not factor_map:
        raise ValueError(f"No factor files found in manifest: {manifest_path}")
    return factor_map


def _load_wide_factor_panel(path: Path, factor_key: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if {"date_", "code", factor_key}.issubset(frame.columns):
        subset = frame[["date_", "code", factor_key]].copy()
        subset = subset.rename(columns={factor_key: "value"})
    else:
        value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
        if len(value_columns) != 1:
            raise ValueError(f"Unexpected factor schema in {path}: {frame.columns.tolist()}")
        subset = frame[["date_", "code", value_columns[0]]].copy()
        subset = subset.rename(columns={value_columns[0]: "value"})

    subset["date_"] = pd.to_datetime(subset["date_"], errors="coerce")
    subset["code"] = subset["code"].astype(str)
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.dropna(subset=["date_"])
    wide = (
        subset.set_index(["date_", "code"])["value"]
        .sort_index()
        .unstack("code")
        .sort_index()
    )
    wide.index = pd.DatetimeIndex(wide.index)
    wide.index.name = "date_"
    return wide


def _trim_monthly(panel: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    monthly = panel.sort_index().resample("ME").last()
    monthly = monthly.loc[(monthly.index >= start) & (monthly.index <= end)]
    monthly.index = pd.DatetimeIndex(monthly.index)
    monthly.index.name = "date_"
    return monthly


def _align_panel(panel: pd.DataFrame, *, codes: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date_"), columns=codes, dtype=float)
    aligned = panel.reindex(columns=codes)
    aligned = aligned.loc[(aligned.index >= start) & (aligned.index <= end)]
    aligned.index = pd.DatetimeIndex(aligned.index)
    aligned.index.name = "date_"
    return aligned.sort_index()


def _load_requested_factor(
    factor_map: dict[str, Path],
    *,
    family: str,
    label: str,
    factor_key: str,
    direction: float,
    codes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    monthly: bool = True,
) -> LoadedFactor:
    if factor_key not in factor_map:
        raise KeyError(f"Factor {factor_key!r} not found for family={family}.")
    panel = _load_wide_factor_panel(factor_map[factor_key], factor_key)
    if monthly:
        panel = _trim_monthly(panel, start=start, end=end)
    else:
        panel = panel.loc[(panel.index >= start) & (panel.index <= end)]
    panel = _align_panel(panel, codes=codes, start=start, end=end)
    if direction < 0:
        panel = -panel
    return LoadedFactor(
        label=label,
        family=family,
        factor_key=factor_key,
        direction=float(direction),
        panel=panel,
    )


def _build_fama6_component(
    *,
    library: TigerFactorLibrary,
    close_panel: pd.DataFrame,
    codes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float], dict[str, dict[str, float]], pd.DataFrame, list[str], dict[str, float]]:
    daily_close = close_panel.sort_index().ffill()
    trading_dates = pd.DatetimeIndex(daily_close.index)
    buffer_start = str((start - pd.Timedelta(days=750)).date())
    end_str = str(end.date())

    balance = library.fetch_fundamental_data(
        name="balance_sheet",
        freq="1q",
        start=buffer_start,
        end=end_str,
        codes=codes,
    )
    income = library.fetch_fundamental_data(
        name="income_statement",
        freq="1q",
        start=buffer_start,
        end=end_str,
        codes=codes,
    )

    bal = library.align_fundamental_to_trading_dates(
        balance,
        trading_dates,
        value_columns=["shares_basic", "total_assets", "total_equity"],
        use_point_in_time=True,
        lag_sessions=1,
    )
    inc = library.align_fundamental_to_trading_dates(
        income,
        trading_dates,
        value_columns=["net_income"],
        use_point_in_time=True,
        lag_sessions=1,
    )

    shares = bal["shares_basic"].reindex(columns=codes)
    total_assets = bal["total_assets"].reindex(columns=codes)
    total_equity = bal["total_equity"].reindex(columns=codes)
    net_income = inc["net_income"].reindex(columns=codes)

    market_cap = daily_close * shares
    log_market_cap = np.log(market_cap.replace(0, np.nan))
    returns = daily_close.pct_change(fill_method=None)
    market_return = returns.mean(axis=1)

    rolling_cov = returns.apply(lambda s: s.rolling(126, min_periods=63).cov(market_return), axis=0)
    rolling_var = market_return.rolling(126, min_periods=63).var()
    fama6_beta = rolling_cov.div(rolling_var, axis=0)
    fama6_size = -log_market_cap
    fama6_value = total_equity.div(market_cap.replace(0, np.nan))
    fama6_profitability = net_income.div(total_equity.replace(0, np.nan))
    fama6_investment = -total_assets.pct_change(252, fill_method=None)
    fama6_momentum = daily_close.pct_change(252, fill_method=None) - daily_close.pct_change(21, fill_method=None)

    raw_daily = {
        "fama6_beta": fama6_beta,
        "fama6_size": fama6_size,
        "fama6_value": fama6_value,
        "fama6_profitability": fama6_profitability,
        "fama6_investment": fama6_investment,
        "fama6_momentum": fama6_momentum,
    }
    raw_monthly = {
        name: _trim_monthly(panel, start=start, end=end)
        for name, panel in raw_daily.items()
    }

    forward_returns = daily_close.resample("ME").last().pct_change(fill_method=None).shift(-1)
    forward_returns = forward_returns.loc[(forward_returns.index >= start) & (forward_returns.index <= end)]

    factor_scores, summaries = score_factor_panels(raw_monthly, forward_returns, score_field="fitness")
    corr = factor_correlation_matrix(raw_monthly, standardize=True)
    selected = [name for name in factor_scores.keys()]
    selected = [name for name in selected if name in raw_monthly]
    selected = [name for name in selected if not pd.isna(factor_scores.get(name, np.nan))]
    # Mimic the existing Fama6 loop: keep the positively scored factors after correlation screening.
    fama6_selected = []
    ordered = sorted(selected, key=lambda name: factor_scores.get(name, float("-inf")), reverse=True)
    for name in ordered:
        if not fama6_selected:
            fama6_selected.append(name)
            continue
        too_correlated = any(abs(float(corr.loc[name, picked])) >= 0.75 for picked in fama6_selected)
        if not too_correlated:
            fama6_selected.append(name)

    fama6_weights = score_to_weights(factor_scores, selected=fama6_selected, method="positive")
    fama6_weights = {name: float(value) for name, value in fama6_weights.items()}
    fama6_component = blend_factor_panels(
        {name: raw_monthly[name] for name in fama6_selected},
        fama6_weights,
        standardize=True,
    )
    fama6_component = fama6_component.loc[(fama6_component.index >= start) & (fama6_component.index <= end)]
    fama6_component.index = pd.DatetimeIndex(fama6_component.index)
    fama6_component.index.name = "date_"
    return fama6_component, raw_monthly, factor_scores, summaries, corr, fama6_selected, fama6_weights


def main() -> None:
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_PATH}")
    if not FINANCIAL_MANIFEST.exists():
        raise FileNotFoundError(f"Financial manifest not found: {FINANCIAL_MANIFEST}")
    if not VALUATION_MANIFEST.exists():
        raise FileNotFoundError(f"Valuation manifest not found: {VALUATION_MANIFEST}")

    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / pd.Timestamp.now(tz="UTC").strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    price_df = pd.read_parquet(PRICE_PATH, columns=["date_", "code", "close"])
    price_df["date_"] = pd.to_datetime(price_df["date_"], errors="coerce")
    price_df["code"] = price_df["code"].astype(str)
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_panel = coerce_price_panel(price_df)
    price_panel = price_panel.sort_index().ffill()
    start = pd.Timestamp(price_panel.index.min())
    end = pd.Timestamp(price_panel.index.max())
    codes = [str(code) for code in price_panel.columns]

    financial_map = _load_manifest_map(FINANCIAL_MANIFEST)
    valuation_map = _load_manifest_map(VALUATION_MANIFEST)

    library = TigerFactorLibrary(output_dir=run_dir / "fama6", price_provider="simfin", verbose=False)
    fama6_component, fama6_raw_panels, fama6_scores, fama6_summaries, fama6_corr, fama6_selected, fama6_weights = _build_fama6_component(
        library=library,
        close_panel=price_panel,
        codes=codes,
        start=start,
        end=end,
    )

    requested_factors = [
        ("fama6", "fama6_multi_factor", "fama6_multi_factor", 1.0),
        ("financial", "roe", "roe__raw", 1.0),
        ("financial", "roa", "roa__raw", 1.0),
        ("financial", "gross_profit_margin", "gross_profit_margin__raw", 1.0),
        ("financial", "operating_margin", "operating_margin__raw", 1.0),
        ("financial", "net_profit_margin", "net_profit_margin__raw", 1.0),
        ("financial", "current_ratio", "current_ratio__raw", 1.0),
        ("financial", "debt_ratio", "debt_ratio__raw", -1.0),
        ("financial", "cash_ratio", "cash_ratio__raw", 1.0),
        ("financial", "cfo_to_assets", "cfo_to_assets__raw", 1.0),
        ("financial", "fcf_to_assets", "fcf_to_assets__raw", 1.0),
        ("financial", "piotroski_f_score", "piotroski_f_score__raw", 1.0),
        ("valuation", "pe", "price_to_earnings__raw", -1.0),
        ("valuation", "pb", "price_to_book__raw", -1.0),
        ("valuation", "ps", "price_to_sales__raw", -1.0),
    ]

    loaded: list[LoadedFactor] = [LoadedFactor(
        label="fama6_multi_factor",
        family="fama6",
        factor_key="fama6_multi_factor",
        direction=1.0,
        panel=_trim_monthly(fama6_component, start=start, end=end),
    )]

    for family, label, factor_key, direction in requested_factors[1:]:
        factor_map = financial_map if family == "financial" else valuation_map
        loaded.append(
            _load_requested_factor(
                factor_map,
                family=family,
                label=label,
                factor_key=factor_key,
                direction=direction,
                codes=codes,
                start=start,
                end=end,
                monthly=True,
            )
        )

    factor_panels = {item.label: item.panel for item in loaded}
    monthly_close = price_panel.resample("ME").last()
    monthly_close = monthly_close.loc[(monthly_close.index >= start) & (monthly_close.index <= end)]
    forward_returns = monthly_close.pct_change(fill_method=None).shift(-1)
    factor_scores, factor_summaries = score_factor_panels(factor_panels, forward_returns, score_field="fitness")
    usable_factor_names = [name for name, score in factor_scores.items() if np.isfinite(score) and name in factor_panels]
    dropped_factors = [name for name in factor_panels.keys() if name not in usable_factor_names]
    if not usable_factor_names:
        raise RuntimeError("No factors with finite evaluation scores survived weighting.")

    factor_panels = {name: factor_panels[name] for name in usable_factor_names}
    factor_summaries = {name: factor_summaries[name] for name in usable_factor_names}
    factor_scores = {name: float(factor_scores[name]) for name in usable_factor_names}
    factor_corr = factor_correlation_matrix(factor_panels, standardize=True)
    factor_weights = score_to_weights(factor_scores, selected=list(factor_panels.keys()), method="positive")
    factor_weights = {name: float(value) for name, value in factor_weights.items()}

    combined_factor = blend_factor_panels(factor_panels, factor_weights, standardize=True)
    combined_factor = combined_factor.loc[(combined_factor.index >= start) & (combined_factor.index <= end)]
    lagged_combined = combined_factor.shift(1)

    backtest, stats = run_factor_backtest(
        lagged_combined,
        price_panel,
        long_pct=DEFAULT_LONG_PCT,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        start=start,
        end=end,
    )

    factor_summary_df = pd.DataFrame(factor_summaries).T
    factor_summary_df.index.name = "factor"
    factor_summary_df = factor_summary_df.reset_index()
    factor_weights_df = pd.DataFrame(
        [{"factor": name, "weight": float(weight), "fitness": float(factor_scores.get(name, np.nan))} for name, weight in factor_weights.items()]
    ).sort_values("weight", ascending=False)
    factor_corr_pairs = []
    names = list(factor_corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = float(factor_corr.loc[left, right])
            factor_corr_pairs.append({"left": left, "right": right, "corr": value, "abs_corr": abs(value)})
    factor_corr_pairs_df = pd.DataFrame(factor_corr_pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)

    factor_summary_df.to_parquet(run_dir / "factor_summary.parquet", index=False)
    factor_weights_df.to_parquet(run_dir / "factor_weights.parquet", index=False)
    factor_corr.to_parquet(run_dir / "factor_correlation_matrix.parquet")
    factor_corr_pairs_df.to_parquet(run_dir / "factor_correlation_pairs.parquet", index=False)
    combined_factor.to_parquet(run_dir / "combined_factor.parquet")
    lagged_combined.to_parquet(run_dir / "combined_factor_lagged.parquet")
    backtest.to_parquet(run_dir / "backtest_daily.parquet")
    pd.DataFrame(stats).T.to_parquet(run_dir / "backtest_stats.parquet")

    (run_dir / "fama6_component_weights.json").write_text(
        json.dumps(
            {
                "selected": fama6_selected,
                "weights": fama6_weights,
                "scores": fama6_scores,
                "summaries": fama6_summaries,
                "correlation": fama6_corr.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "price_path": str(PRICE_PATH),
        "financial_manifest": str(FINANCIAL_MANIFEST),
        "valuation_manifest": str(VALUATION_MANIFEST),
        "start": str(start.date()),
        "end": str(end.date()),
        "forward_days": DEFAULT_FORWARD_DAYS,
        "rebalance_freq": DEFAULT_REBALANCE_FREQ,
        "long_pct": DEFAULT_LONG_PCT,
        "transaction_cost_bps": DEFAULT_TRANSACTION_COST_BPS,
        "slippage_bps": DEFAULT_SLIPPAGE_BPS,
        "factor_count": int(len(factor_panels)),
        "dropped_factors": dropped_factors,
        "fama6_selected": fama6_selected,
        "factor_weights": factor_weights,
        "factor_scores": {name: float(score) for name, score in factor_scores.items()},
        "backtest_stats": stats,
        "output_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("fama6 + traditional + valuation combo complete")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"saved outputs to: {run_dir}")
    print("\nselected factors and weights:")
    for row in factor_weights_df.to_dict(orient="records"):
        print(f" - {row['factor']}: weight={row['weight']:.4f}, fitness={row['fitness']:.6f}")
    print("\nportfolio stats:")
    print(json.dumps(stats["portfolio"], indent=2, ensure_ascii=False, default=str))
    print("\nbenchmark stats:")
    print(json.dumps(stats["benchmark"], indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
