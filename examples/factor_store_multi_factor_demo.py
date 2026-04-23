"""Load an arbitrary basket of stored factors into one research frame.

This demo is intentionally generic: it does not assume any particular factor
family. It shows how ``TigerFactorLibrary.load_factor_frame(...)`` can be used
to assemble a multi-factor Tiger-style long frame from the factor store, and
how the same stored factors can also be inspected as individual wide panels.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import TigerFactorLibrary


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "factor_store_multi_factor_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load arbitrary stored factors into one research frame.")
    parser.add_argument(
        "--store-root",
        default=str(DEFAULT_FACTOR_STORE_ROOT),
        help="Root directory of the Tiger factor store.",
    )
    parser.add_argument("--factor-provider", default="tiger", help="Provider namespace used when the factors were saved.")
    parser.add_argument("--factor-variant", default=None, help="Optional factor variant used when the factors were saved.")
    parser.add_argument(
        "--factor-names",
        nargs="+",
        default=["BM", "FSCORE", "BMFSCORE"],
        help="Stored factor names to load and merge.",
    )
    parser.add_argument("--price-provider", default="yahoo", help="Price provider used when a matching close panel is needed.")
    parser.add_argument("--codes", nargs="*", default=None, help="Optional explicit universe to keep.")
    parser.add_argument("--start", default=None, help="Optional start date.")
    parser.add_argument("--end", default=None, help="Optional end date.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="factor_store_multi_factor")
    parser.add_argument("--save-csv", action="store_true", help="Write the merged research frame to CSV.")
    return parser.parse_args()


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def _coverage_summary(frame: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in factor_names:
        if name not in frame.columns:
            rows.append({"factor": name, "non_null": 0, "coverage": 0.0})
            continue
        series = pd.to_numeric(frame[name], errors="coerce")
        rows.append(
            {
                "factor": name,
                "non_null": int(series.notna().sum()),
                "coverage": float(series.notna().mean()) if len(series) else 0.0,
                "mean": float(series.mean(skipna=True)) if series.notna().any() else float("nan"),
                "std": float(series.std(skipna=True)) if series.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variant = _normalize_variant(args.factor_variant)
    factor_names = [str(name) for name in args.factor_names]

    library = TigerFactorLibrary(output_dir=args.store_root, price_provider=args.price_provider, verbose=True)

    wide_panels = library.load_factor_panels(
        factor_names=factor_names,
        provider=args.factor_provider,
        variant=variant,
        codes=args.codes,
        start=args.start,
        end=args.end,
    )
    factor_frame = library.load_factor_frame(
        factor_names=factor_names,
        provider=args.factor_provider,
        variant=variant,
        codes=args.codes,
        start=args.start,
        end=args.end,
    )

    print("loaded factor panels:")
    for name, panel in wide_panels.items():
        print(f"  {name}: shape={panel.shape}")
    print("\nmerged research frame:")
    print(f"  shape={factor_frame.shape}")
    print(f"  columns={list(factor_frame.columns)}")
    print("\ncoverage summary:")
    print(_coverage_summary(factor_frame, factor_names).to_string(index=False))

    if not factor_frame.empty:
        print("\nmerged frame head:")
        print(factor_frame.head().to_string(index=False))

    if args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{args.report_name}_factor_frame.csv"
        factor_frame.to_csv(csv_path, index=False)
        print(f"\nmerged research frame saved to: {csv_path}")


if __name__ == "__main__":
    main()
