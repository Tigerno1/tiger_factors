"""Screen stored Alpha101 factors, then inspect their backing asset coverage.

This demo is intentionally constant-driven and does not use argparse. It expects
the Alpha101 factor values and ``alpha101_summary`` table to already exist in
the local ``FactorStore``; see ``sp500_alpha101_researchengine_demo.py`` for the
generation/evaluation flow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import OthersSpec
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics


STORE_ROOT = DEFAULT_FACTOR_STORE_ROOT
OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "alpha101_screened_asset_coverage_demo"

FACTOR_PROVIDER = "tiger"
REGION = "us"
SEC_TYPE = "stock"
FREQ = "1d"
FACTOR_VARIANT = None
FACTOR_GROUP = None
SUMMARY_TABLE_NAME = "alpha101_summary"
SUMMARY_VARIANT = "summary"

START = "2020-01-01"
END = "2024-12-31"
CODES: list[str] | None = None

TOP_BOTTOM_PCT = 0.20
SAVE_CSV = True
SAVE_PARQUET = True

SCREENING_CONFIG = FactorMetricFilterConfig(
    min_fitness=0.10,
    min_ic_mean=0.01,
    min_rank_ic_mean=0.01,
    min_sharpe=0.40,
    max_turnover=0.50,
    min_decay_score=0.20,
    min_capacity_score=0.20,
    max_correlation_penalty=0.60,
    min_regime_robustness=0.60,
    min_out_of_sample_stability=0.60,
    sort_field="fitness",
    tie_breaker_field="ic_ir",
)


def _alpha101_summary_spec() -> OthersSpec:
    return OthersSpec(
        provider=FACTOR_PROVIDER,
        region=REGION,
        sec_type=SEC_TYPE,
        freq=FREQ,
        table_name=SUMMARY_TABLE_NAME,
        variant=SUMMARY_VARIANT,
    )


def _normalize_factor_names(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "factor_name" not in out.columns:
        raise KeyError("alpha101 summary must contain a factor_name column.")
    out["factor_name"] = out["factor_name"].astype(str).str.strip().str.lower()
    return out[out["factor_name"].str.match(r"^alpha_\d{3}$", na=False)].reset_index(drop=True)


def _asset_coverage_row(factor_name: str, panel: pd.DataFrame, *, universe_asset_count: int) -> dict[str, object]:
    if panel.empty:
        return {
            "factor_name": factor_name,
            "date_count": 0,
            "first_date": pd.NaT,
            "last_date": pd.NaT,
            "unique_assets": 0,
            "universe_asset_count": int(universe_asset_count),
            "asset_coverage_ratio": 0.0,
            "total_observations": 0,
            "avg_daily_available_assets": 0.0,
            "median_daily_available_assets": 0.0,
            "avg_daily_available_ratio": 0.0,
            "top_bottom_pct": float(TOP_BOTTOM_PCT),
            "avg_top_assets": 0.0,
            "avg_bottom_assets": 0.0,
            "avg_top_ratio": 0.0,
            "avg_bottom_ratio": 0.0,
        }

    clean = panel.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    available = clean.notna()
    daily_available = available.sum(axis=1)
    valid_dates = daily_available[daily_available > 0].index
    unique_assets = int(available.any(axis=0).sum())
    total_observations = int(available.sum().sum())

    ranks = clean.rank(axis=1, method="first")
    leg_counts = np.ceil(daily_available * float(TOP_BOTTOM_PCT)).clip(lower=1)
    leg_counts = leg_counts.where(daily_available > 0, 0)
    top_thresholds = daily_available - leg_counts + 1
    top_mask = ranks.ge(top_thresholds, axis=0)
    bottom_mask = ranks.le(leg_counts, axis=0)
    top_counts = top_mask.where(available, False).sum(axis=1)
    bottom_counts = bottom_mask.where(available, False).sum(axis=1)

    denominator = daily_available.replace(0, np.nan)
    return {
        "factor_name": factor_name,
        "date_count": int(len(valid_dates)),
        "first_date": pd.Timestamp(valid_dates.min()) if len(valid_dates) else pd.NaT,
        "last_date": pd.Timestamp(valid_dates.max()) if len(valid_dates) else pd.NaT,
        "unique_assets": unique_assets,
        "universe_asset_count": int(universe_asset_count),
        "asset_coverage_ratio": float(unique_assets / universe_asset_count) if universe_asset_count else 0.0,
        "total_observations": total_observations,
        "avg_daily_available_assets": float(daily_available.mean()) if len(daily_available) else 0.0,
        "median_daily_available_assets": float(daily_available.median()) if len(daily_available) else 0.0,
        "avg_daily_available_ratio": float((daily_available / universe_asset_count).mean()) if universe_asset_count else 0.0,
        "top_bottom_pct": float(TOP_BOTTOM_PCT),
        "avg_top_assets": float(top_counts.mean()) if len(top_counts) else 0.0,
        "avg_bottom_assets": float(bottom_counts.mean()) if len(bottom_counts) else 0.0,
        "avg_top_ratio": float((top_counts / denominator).mean(skipna=True)) if len(top_counts) else 0.0,
        "avg_bottom_ratio": float((bottom_counts / denominator).mean(skipna=True)) if len(bottom_counts) else 0.0,
    }


def _build_asset_coverage(
    library: TigerFactorLibrary,
    factor_names: list[str],
) -> pd.DataFrame:
    panels = library.load_factor_panels(
        factor_names=factor_names,
        provider=FACTOR_PROVIDER,
        freq=FREQ,
        variant=FACTOR_VARIANT,
        group=FACTOR_GROUP,
        codes=CODES,
        start=START,
        end=END,
    )
    universe_assets = sorted({str(column) for panel in panels.values() for column in panel.columns})
    universe_asset_count = len(universe_assets)

    rows = [
        _asset_coverage_row(name, panels.get(name, pd.DataFrame()), universe_asset_count=universe_asset_count)
        for name in factor_names
    ]
    coverage = pd.DataFrame(rows)
    if coverage.empty:
        return coverage
    return coverage.sort_values(
        ["asset_coverage_ratio", "avg_daily_available_assets", "factor_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _save_outputs(screened: pd.DataFrame, coverage: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PARQUET:
        screened.to_parquet(OUTPUT_DIR / "alpha101_screened_factors.parquet", index=False)
        coverage.to_parquet(OUTPUT_DIR / "alpha101_screened_asset_coverage.parquet", index=False)
    if SAVE_CSV:
        screened.to_csv(OUTPUT_DIR / "alpha101_screened_factors.csv", index=False)
        coverage.to_csv(OUTPUT_DIR / "alpha101_screened_asset_coverage.csv", index=False)


def main() -> None:
    store = FactorStore(STORE_ROOT)
    library = TigerFactorLibrary(
        store=store,
        region=REGION,
        sec_type=SEC_TYPE,
        price_provider="yahoo",
        verbose=False,
    )

    summary = _normalize_factor_names(store.get_others(_alpha101_summary_spec()))
    if summary.empty:
        raise RuntimeError(
            "No alpha101_summary rows were found in FactorStore. Run the Alpha101 evaluation demo first."
        )

    screened = screen_factor_metrics(summary, config=SCREENING_CONFIG)
    selected = screened.loc[screened["usable"].fillna(False), "factor_name"].astype(str).tolist()
    if not selected:
        print("No Alpha101 factors passed the screening config.")
        print(screened[["factor_name", "usable", "failed_rules"]].head(20).to_string(index=False))
        return

    coverage = _build_asset_coverage(library, selected)
    _save_outputs(screened.loc[screened["factor_name"].isin(selected)].reset_index(drop=True), coverage)

    print(f"Alpha101 summary rows: {len(summary)}")
    print(f"Screened Alpha101 factors: {len(selected)}")
    print("\nSelected factors:")
    selected_columns = ["factor_name", "direction_hint", "fitness", "ic_mean", "rank_ic_mean", "sharpe"]
    print(screened.loc[screened["factor_name"].isin(selected), selected_columns].to_string(index=False))

    print("\nBacking asset count/ratio:")
    display_columns = [
        "factor_name",
        "unique_assets",
        "universe_asset_count",
        "asset_coverage_ratio",
        "avg_daily_available_assets",
        "avg_daily_available_ratio",
        "avg_top_assets",
        "avg_top_ratio",
        "avg_bottom_assets",
        "avg_bottom_ratio",
    ]
    print(coverage[display_columns].to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
