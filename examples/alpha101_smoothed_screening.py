from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_algorithm.alpha101.smoothed_engine import Alpha101SmoothedScreeningEngine


OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/alpha101_smoothed_screening_backtest")
START = "2020-06-01"
END = "2024-06-01"


def main() -> None:
    engine = Alpha101SmoothedScreeningEngine(
        output_root=OUTPUT_ROOT,
        start=START,
        end=END,
        smoothing_method="rolling_mean",
        smoothing_window=5,
        smoothing_min_periods=3,
    )
    result = engine.run(
        save_adjusted_price=False,
        compute_workers=1,
        save_workers=1,
    )
    summary = {
        "output_root": result.output_root,
        "manifest_path": result.manifest_path,
        "backtest_manifest_path": result.backtest_manifest_path,
        "codes": len(result.codes),
        "raw_factor_count": int(len([column for column in result.factor_frame.columns if column not in {"date_", "code"}])),
        "screened_factor_count": int(len(result.screened_registry)),
        "selected_factor_count": int(len(result.selected_factors)),
        "combined_factor_rows": int(result.combined_factor.notna().sum().sum()) if not result.combined_factor.empty else 0,
        "combined_factor_coverage": float(result.combined_factor.notna().mean().mean()) if not result.combined_factor.empty else 0.0,
        "evaluation_registry_path": str(Path(result.output_root) / "alpha101_smoothed_evaluation_registry.parquet"),
        "screened_registry_path": str(Path(result.output_root) / "alpha101_smoothed_screened_registry.parquet"),
        "backtest_returns_path": str(Path(result.output_root) / "backtest" / "alpha101_smoothed_backtest_returns.parquet"),
        "backtest_stats_path": str(Path(result.output_root) / "backtest" / "alpha101_smoothed_backtest_stats.parquet"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\nSelected factors:")
    print(result.selected_factors)
    print("\nPortfolio stats:")
    print(json.dumps(result.backtest_stats.get("portfolio", {}), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
