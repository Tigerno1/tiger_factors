from __future__ import annotations

import json
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

DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk/factor")
DEFAULT_PRICE_PATH = Path("/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet")
DEFAULT_OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/summary")


def _discover_factor_paths(root: Path) -> list[Path]:
    candidates = sorted(root.glob("alpha_*/alpha_*.parquet"))
    factor_paths = [path for path in candidates if path.parent.name == path.stem]

    def _factor_sort_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.stem.split("_")[-1]), path.stem)
        except ValueError:
            return (10**9, path.stem)

    return sorted(factor_paths, key=_factor_sort_key)


def main() -> None:
    factor_paths = _discover_factor_paths(DEFAULT_FACTOR_ROOT)
    if not factor_paths:
        raise RuntimeError(f"No alpha factor parquet files found under {DEFAULT_FACTOR_ROOT}")

    price_df = pd.read_parquet(DEFAULT_PRICE_PATH)
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    registry_rows: list[dict[str, object]] = []
    for idx, factor_path in enumerate(factor_paths, start=1):
        factor_column = factor_path.stem
        factor_df = pd.read_parquet(factor_path)
        factor_output_root = DEFAULT_OUTPUT_ROOT / factor_column

        print(f"[{idx}/{len(factor_paths)}] building summary for {factor_column}")
        engine = FactorEvaluationEngine(
            factor_frame=factor_df,
            price_frame=price_df,
            factor_column=factor_column,
            factor_store=FactorStore(root_dir=factor_output_root),
        )
        summary = engine.summary()

        summary_dir = summary.output_dir
        evaluation = summary.evaluation
        registry_rows.append(
            {
                "factor_name": factor_column,
                "factor_path": str(factor_path),
                "summary_dir": str(summary_dir),
                "evaluation_path": str(summary.table_paths.get("evaluation", "")),
                "table_count": len(summary.table_paths),
                "figure_count": len(summary.figure_paths),
                "factor_rows": int(len(factor_df)),
                "price_rows": int(len(price_df)),
                **(evaluation.__dict__ if evaluation is not None else {}),
            }
        )
        print(f"  summary dir: {summary_dir}")

    registry = pd.DataFrame(registry_rows).sort_values("factor_name").reset_index(drop=True)
    registry_path = DEFAULT_OUTPUT_ROOT / "summary_registry.parquet"
    registry.to_parquet(registry_path, index=False)

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "factor_root": str(DEFAULT_FACTOR_ROOT),
        "price_path": str(DEFAULT_PRICE_PATH),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "factor_count": int(len(registry)),
        "registry_path": str(registry_path),
        "summary_dirs": registry["summary_dir"].tolist(),
    }
    manifest_path = DEFAULT_OUTPUT_ROOT / "summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\nregistry:")
    print(registry[["factor_name", "ic_mean", "rank_ic_mean", "sharpe", "turnover", "fitness"]].head(10))
    print(f"\nregistry saved to: {registry_path}")
    print(f"manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
