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

from tiger_factors.multifactor_evaluation import FactorPipelineConfig
from tiger_factors.multifactor_evaluation import run_factor_backtest
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk/factor")
DEFAULT_PRICE_PATH = DEFAULT_FACTOR_ROOT / "price" / "tiger" / "us" / "stock" / "1d" / "adj_price.parquet"
DEFAULT_OUTPUT_DIR = Path("/Volumes/Quant_Disk/evaluation/multifactors_resonance")
DEFAULT_RESONANCE_FACTORS = (
    "alpha_005",
    "alpha_010",
    "alpha_017",
    "alpha_018",
    "alpha_025",
    "alpha_033",
    "alpha_034",
    "alpha_038",
    "alpha_047",
)
DEFAULT_MIN_AGREEMENT = 3
DEFAULT_ABS_THRESHOLD = 0.50


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
    if not isinstance(factor_series.index, pd.MultiIndex):
        raise ValueError(f"Factor {factor_name} could not be coerced into a MultiIndex series.")
    wide = factor_series.unstack("code").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    wide.index.name = "date_"
    return wide


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _build_resonance_panel(
    factor_panels: dict[str, pd.DataFrame],
    *,
    min_agreement: int = 3,
    abs_threshold: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = None
    common_columns = None
    for panel in factor_panels.values():
        common_index = panel.index if common_index is None else common_index.intersection(panel.index)
        common_columns = panel.columns if common_columns is None else common_columns.intersection(panel.columns)
    if common_index is None or common_columns is None or len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("No overlapping dates or codes across resonance factors.")

    aligned = {name: panel.reindex(index=common_index, columns=common_columns) for name, panel in factor_panels.items()}
    standardized = {name: _cross_sectional_zscore(panel) for name, panel in aligned.items()}

    stacked = np.stack([panel.to_numpy(dtype=float) for panel in standardized.values()], axis=0)
    sign_stack = np.sign(stacked)
    agreement_count = np.sum(sign_stack > 0, axis=0)
    disagreement_count = np.sum(sign_stack < 0, axis=0)
    valid_count = np.sum(np.isfinite(stacked), axis=0)
    directional_count = np.maximum(agreement_count, disagreement_count)
    with np.errstate(all="ignore"):
        resonance_score = np.nanmean(stacked, axis=0)
    resonance_score = np.where(valid_count > 0, resonance_score, np.nan)
    resonance_score = np.where(directional_count >= min_agreement, resonance_score, np.nan)
    resonance_score = np.where(np.abs(resonance_score) >= abs_threshold, resonance_score, np.nan)

    resonance_panel = pd.DataFrame(resonance_score, index=common_index, columns=common_columns)
    agreement_panel = pd.DataFrame(directional_count, index=common_index, columns=common_columns)
    return resonance_panel, agreement_panel


def main() -> None:
    factor_root = DEFAULT_FACTOR_ROOT
    price_path = DEFAULT_PRICE_PATH
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_names = list(DEFAULT_RESONANCE_FACTORS)
    factor_panels = {name: _load_factor_panel(factor_root, name) for name in factor_names}
    prices = pd.read_parquet(price_path)
    prices_panel = coerce_price_panel(prices)

    resonance_panel, agreement_panel = _build_resonance_panel(
        factor_panels,
        min_agreement=DEFAULT_MIN_AGREEMENT,
        abs_threshold=DEFAULT_ABS_THRESHOLD,
    )
    lagged_resonance_panel = resonance_panel.shift(1)

    backtest, stats = run_factor_backtest(
        lagged_resonance_panel,
        prices_panel,
        long_pct=0.20,
        rebalance_freq="ME",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=8.0,
        slippage_bps=4.0,
    )

    summary = {
        "factor_names": factor_names,
        "min_agreement": DEFAULT_MIN_AGREEMENT,
        "abs_threshold": DEFAULT_ABS_THRESHOLD,
        "signal_lag_days": 1,
        "resonance_rows": int(resonance_panel.notna().sum().sum()),
        "resonance_coverage": float(resonance_panel.notna().mean().mean()),
        "agreement_mean": float(agreement_panel.stack().mean()),
        "agreement_median": float(agreement_panel.stack().median()),
        "backtest_stats": stats,
        "output_dir": str(output_dir),
    }

    resonance_panel.to_parquet(output_dir / "resonance_factor.parquet")
    agreement_panel.to_parquet(output_dir / "resonance_agreement_count.parquet")
    lagged_resonance_panel.to_parquet(output_dir / "resonance_factor_lagged.parquet")
    backtest.to_parquet(output_dir / "resonance_backtest_daily.parquet")
    pd.DataFrame(stats).T.to_parquet(output_dir / "resonance_backtest_stats.parquet")
    (output_dir / "resonance_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("Resonance strategy complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Saved outputs to: {output_dir}")
    print("\nPortfolio stats:")
    print(json.dumps(stats["portfolio"], indent=2))


if __name__ == "__main__":
    main()
