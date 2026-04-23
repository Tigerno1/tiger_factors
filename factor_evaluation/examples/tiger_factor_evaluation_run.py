from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine


def main() -> None:
    factor_frame = pd.read_parquet("/Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_001.parquet")
    price_frame = pd.read_parquet("/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet")
    evaluation = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
    )

    summary = evaluation.evaluate()
    print(summary)

    report = evaluation.full()
    print(report.output_dir)
    for name, path in report.figure_paths.items():
        print(name, path)


if __name__ == "__main__":
    main()
