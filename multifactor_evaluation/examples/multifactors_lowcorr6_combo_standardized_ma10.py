from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk/factor")
DEFAULT_PRICE_PATH = DEFAULT_FACTOR_ROOT / "price" / "tiger" / "us" / "stock" / "1d" / "adj_price.parquet"
DEFAULT_OUTPUT_DIR = Path("/Volumes/Quant_Disk/evaluation/multifactors_lowcorr6_combo_standardized_ma10")
DEFAULT_FACTORS = (
    "alpha_047",
    "alpha_083",
    "alpha_042",
    "alpha_036",
    "alpha_060",
    "alpha_068",
)
DEFAULT_THRESHOLD = 1.0
DEFAULT_MA_WINDOW = 10
DEFAULT_MIN_PERIODS = 5


def _load_factor_frame(factor_root: Path, factor_name: str) -> pd.DataFrame:
    path = factor_root / factor_name / f"{factor_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")
    frame = pd.read_parquet(path)
    if {"date_", "code", factor_name}.issubset(frame.columns):
        return frame[["date_", "code", factor_name]].copy()
    if "factor" in frame.columns:
        return frame[["date_", "code", "factor"]].rename(columns={"factor": factor_name})
    value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
    if len(value_columns) != 1:
        raise ValueError(f"Unexpected factor schema in: {path}")
    return frame[["date_", "code", value_columns[0]]].rename(columns={value_columns[0]: factor_name})


def _load_factor_panel(factor_root: Path, factor_name: str) -> pd.DataFrame:
    frame = _load_factor_frame(factor_root, factor_name)
    factor_series = coerce_factor_series(frame)
    wide = factor_series.unstack("code").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    wide.index.name = "date_"
    return wide


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _build_combined_panel(factor_panels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    common_index = None
    common_columns = None
    for panel in factor_panels.values():
        common_index = panel.index if common_index is None else common_index.intersection(panel.index)
        common_columns = panel.columns if common_columns is None else common_columns.intersection(panel.columns)
    if common_index is None or common_columns is None or len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("No overlapping dates or codes across selected factor panels.")

    aligned = {name: panel.reindex(index=common_index, columns=common_columns) for name, panel in factor_panels.items()}
    filtered_parts: list[pd.DataFrame] = []
    contribution_count = np.zeros((len(common_index), len(common_columns)), dtype=float)

    for panel in aligned.values():
        zscore = _cross_sectional_zscore(panel)
        signal = zscore.where(zscore.abs() >= DEFAULT_THRESHOLD)
        filtered_parts.append(signal)
        contribution_count += np.isfinite(signal.to_numpy(dtype=float)).astype(float)

    stacked = np.stack([part.to_numpy(dtype=float) for part in filtered_parts], axis=0)
    summed = np.nansum(stacked, axis=0)
    combined = np.full_like(summed, np.nan, dtype=float)
    np.divide(summed, contribution_count, out=combined, where=contribution_count > 0)
    combined_panel = pd.DataFrame(combined, index=common_index, columns=common_columns)
    contribution_panel = pd.DataFrame(contribution_count, index=common_index, columns=common_columns)
    return combined_panel, list(aligned.keys()), contribution_panel


def main() -> None:
    factor_root = DEFAULT_FACTOR_ROOT
    price_path = DEFAULT_PRICE_PATH
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_panels = {name: _load_factor_panel(factor_root, name) for name in DEFAULT_FACTORS}
    prices_panel = coerce_price_panel(pd.read_parquet(price_path))
    combined_factor, selected_order, contribution_panel = _build_combined_panel(factor_panels)
    smoothed_factor = combined_factor.rolling(DEFAULT_MA_WINDOW, min_periods=DEFAULT_MIN_PERIODS).mean()
    lagged_smoothed = smoothed_factor.shift(1)

    backtest, stats = run_factor_backtest(
        lagged_smoothed,
        prices_panel,
        long_pct=0.20,
        rebalance_freq="ME",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=8.0,
        slippage_bps=4.0,
    )

    summary = {
        "selected_factors": selected_order,
        "factor_threshold": DEFAULT_THRESHOLD,
        "ma_window": DEFAULT_MA_WINDOW,
        "ma_min_periods": DEFAULT_MIN_PERIODS,
        "signal_lag_days": 1,
        "selected_factor_count": len(selected_order),
        "combined_rows": int(combined_factor.notna().sum().sum()),
        "combined_coverage": float(combined_factor.notna().mean().mean()),
        "smoothed_rows": int(smoothed_factor.notna().sum().sum()),
        "smoothed_coverage": float(smoothed_factor.notna().mean().mean()),
        "contribution_mean": float(contribution_panel.stack().mean()),
        "contribution_median": float(contribution_panel.stack().median()),
        "backtest_stats": stats,
        "output_dir": str(output_dir),
    }

    combined_factor.to_parquet(output_dir / "lowcorr6_combo_standardized_factor.parquet")
    smoothed_factor.to_parquet(output_dir / "lowcorr6_combo_standardized_factor_ma10.parquet")
    lagged_smoothed.to_parquet(output_dir / "lowcorr6_combo_standardized_factor_ma10_lagged.parquet")
    contribution_panel.to_parquet(output_dir / "lowcorr6_combo_standardized_contribution_count.parquet")
    backtest.to_parquet(output_dir / "lowcorr6_combo_standardized_backtest_daily.parquet")
    pd.DataFrame(stats).T.to_parquet(output_dir / "lowcorr6_combo_standardized_backtest_stats.parquet")
    (output_dir / "lowcorr6_combo_standardized_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("Low-correlation 6-factor standardized MA10 combo strategy complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Saved outputs to: {output_dir}")
    print("\nPortfolio stats:")
    print(json.dumps(stats["portfolio"], indent=2))


if __name__ == "__main__":
    main()
