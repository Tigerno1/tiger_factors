from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_store import FactorStore

DEFAULT_FACTOR_PATH = "/Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_101.parquet"
DEFAULT_PRICE_PATH = "/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet"
DEFAULT_FACTOR_COLUMN = "alpha_101"
DEFAULT_OUTPUT_ROOT = "/Volumes/Quant_Disk/evaluation/summary/alpha_101"


def _load_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"Unsupported file format: {resolved.suffix}")


def main() -> None:
    factor_df = _load_table(DEFAULT_FACTOR_PATH)
    price_df = _load_table(DEFAULT_PRICE_PATH)

    engine = FactorEvaluationEngine(
        factor_frame=factor_df,
        price_frame=price_df,
        factor_column=DEFAULT_FACTOR_COLUMN,
        factor_store=FactorStore(root_dir=DEFAULT_OUTPUT_ROOT),
    )

    summary = engine.summary()

    print("summary evaluation:")
    print(summary.evaluation)
    print("\nsummary output dir:")
    print(summary.output_dir)
    print("\nsummary figures:")
    if summary.figure_paths:
        for name, path in summary.figure_paths.items():
            print(f"{name}: {path}")
    else:
        print("(none for summary-only sheet)")
    print("\nsummary tables:")
    for name, path in summary.table_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
