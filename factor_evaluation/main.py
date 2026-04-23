from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tiger_factors.factor_evaluation.native_tears import create_native_full_tear_sheet


def _load_panel(path: str | Path, *, date_col: str = "date_", code_col: str = "code", value_col: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if date_col not in frame.columns:
        raise ValueError(f"Missing required date column: {date_col}")
    if code_col not in frame.columns:
        raise ValueError(f"Missing required code column: {code_col}")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, code_col]).copy()
    frame[code_col] = frame[code_col].astype(str)
    if value_col is not None:
        if value_col not in frame.columns:
            raise ValueError(f"Missing required value column: {value_col}")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        return frame[[date_col, code_col, value_col]].dropna(subset=[value_col]).sort_values([date_col, code_col]).reset_index(drop=True)
    return frame.sort_values([date_col, code_col]).reset_index(drop=True)


def _to_wide(frame: pd.DataFrame, *, date_col: str, code_col: str, value_col: str) -> pd.DataFrame:
    wide = frame.pivot(index=date_col, columns=code_col, values=value_col).sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Tiger-native factor tear sheets.")
    parser.add_argument("--factor-csv", required=True, help="Factor CSV in long format: date_, code, factor column.")
    parser.add_argument("--forward-returns-csv", required=True, help="Forward returns CSV in long or wide format.")
    parser.add_argument("--factor-name", required=True, help="Name of the factor column inside factor CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to save charts and tables.")
    parser.add_argument("--factor-format", choices=["long", "wide"], default="long")
    parser.add_argument("--forward-format", choices=["long", "wide"], default="wide")
    parser.add_argument("--date-col", default="date_")
    parser.add_argument("--code-col", default="code")
    parser.add_argument("--forward-value-col", default="forward_return")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--portfolio-returns-csv", default="", help="Optional CSV with date_/returns columns.")
    parser.add_argument("--benchmark-returns-csv", default="", help="Optional CSV with date_/returns columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    factor_frame = _load_panel(args.factor_csv, date_col=args.date_col, code_col=args.code_col, value_col=args.factor_name)
    factor_wide = _to_wide(factor_frame, date_col=args.date_col, code_col=args.code_col, value_col=args.factor_name) if args.factor_format == "long" else factor_frame.set_index(args.date_col)

    forward_frame = pd.read_csv(args.forward_returns_csv)
    if args.forward_format == "long":
        forward_frame = _load_panel(
            args.forward_returns_csv,
            date_col=args.date_col,
            code_col=args.code_col,
            value_col=args.forward_value_col,
        )
        forward_wide = _to_wide(forward_frame, date_col=args.date_col, code_col=args.code_col, value_col=args.forward_value_col)
    else:
        if args.date_col not in forward_frame.columns:
            raise ValueError(f"Missing required date column in forward returns CSV: {args.date_col}")
        forward_frame[args.date_col] = pd.to_datetime(forward_frame[args.date_col], errors="coerce")
        forward_wide = forward_frame.set_index(args.date_col).sort_index()

    portfolio_returns = None
    benchmark_returns = None
    if args.portfolio_returns_csv:
        portfolio_df = pd.read_csv(args.portfolio_returns_csv)
        if "returns" not in portfolio_df.columns:
            raise ValueError("portfolio returns CSV must contain a 'returns' column")
        date_col = args.date_col if args.date_col in portfolio_df.columns else "date"
        portfolio_df[date_col] = pd.to_datetime(portfolio_df[date_col], errors="coerce")
        portfolio_returns = portfolio_df.set_index(date_col)["returns"].sort_index()
    if args.benchmark_returns_csv:
        benchmark_df = pd.read_csv(args.benchmark_returns_csv)
        if "returns" not in benchmark_df.columns:
            raise ValueError("benchmark returns CSV must contain a 'returns' column")
        date_col = args.date_col if args.date_col in benchmark_df.columns else "date"
        benchmark_df[date_col] = pd.to_datetime(benchmark_df[date_col], errors="coerce")
        benchmark_returns = benchmark_df.set_index(date_col)["returns"].sort_index()

    report = create_native_full_tear_sheet(
        args.factor_name,
        factor_wide,
        forward_wide,
        output_dir=args.output_dir,
        quantiles=args.quantiles,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report.to_summary(), indent=2, default=str), encoding="utf-8")
    print(f"Saved report to {args.output_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
