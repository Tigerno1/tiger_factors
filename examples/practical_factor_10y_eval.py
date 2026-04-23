from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_api.core.domain_facade import ensure_domain_registered
from tiger_factors.factor_evaluation import evaluate_factor_panel
from tiger_factors.factor_algorithm.data_mining.practical_factors import available_practical_factors
from tiger_factors.factor_algorithm.data_mining.practical_factors import PracticalFactorEngine
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_evaluation import create_native_full_tear_sheet
from tiger_qlib.yahoo_us import build_yahoo_us_export_frame


DEFAULT_DATA_ROOT = Path("/Volumes/Quant_Disk/data")
DEFAULT_DB_PATH = DEFAULT_DATA_ROOT / "yahoo_us_stock.db"
DEFAULT_UNIVERSE_CSV = DEFAULT_DATA_ROOT / "sp500_ticker_start_end.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "out" / "practical_factor_10y_eval.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate practical factors on 10-year Yahoo US data.")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--universe-csv", default=str(DEFAULT_UNIVERSE_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--provider", default="yahoo")
    parser.add_argument("--region", default="us")
    parser.add_argument("--sec-type", default="stock")
    parser.add_argument("--as-ex", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--codes", default="", help="Optional comma-separated code list override.")
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "out" / "practical_factor_10y_report"),
        help="Directory to save the analysis charts for the top factor.",
    )
    return parser.parse_args()


def select_universe(universe_csv: str | Path, start: str, end: str) -> list[str]:
    table = pd.read_csv(universe_csv)
    if "ticker" not in table.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column.")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_col = pd.to_datetime(table["start_date"], errors="coerce")
    end_col = pd.to_datetime(table["end_date"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))
    mask = (start_col <= start_ts) & (end_col >= end_ts)
    codes = (
        table.loc[mask, "ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    if not codes:
        raise RuntimeError("No codes matched the requested date range.")
    return codes


def build_adjusted_panel(
    codes: list[str],
    *,
    start: str,
    end: str,
    provider: str,
    region: str,
    sec_type: str,
    db_path: str | Path | None = None,
    batch_size: int = 80,
    as_ex: bool | None = None,
) -> pd.DataFrame:
    if db_path is not None:
        os.environ["TIGER_DB_URL_YAHOO_US_STOCK"] = f"sqlite:///{Path(db_path).resolve()}"

    ensure_domain_registered(provider="yahoo", region=region, sec_type=sec_type)
    library = TigerFactorLibrary(verbose=False)

    frames: list[pd.DataFrame] = []
    total_batches = (len(codes) + batch_size - 1) // batch_size
    for batch_index in range(0, len(codes), batch_size):
        batch = codes[batch_index : batch_index + batch_size]
        print(f"fetch batch {batch_index // batch_size + 1}/{total_batches}: {len(batch)} codes")
        raw = library.fetch_price_data(codes=batch, start=start, end=end, provider=provider, as_ex=as_ex)
        if raw.empty:
            continue
        if "date" in raw.columns:
            raw = raw.loc[:, [column for column in raw.columns if column != "date"]].copy()

        for code in batch:
            code_raw = raw.loc[raw["code"].astype(str).str.upper() == code].copy()
            if code_raw.empty:
                continue
            export, _, _, _ = build_yahoo_us_export_frame(code_raw, code=code, start=start, end=end)
            if export.empty:
                continue
            export = export.rename(columns={"date": "date_"})
            export["date_"] = pd.to_datetime(export["date_"], errors="coerce")
            frames.append(export[["date_", "code", "open", "high", "low", "close", "volume", "vwap"]])

    if not frames:
        raise RuntimeError("No adjusted panels were built from the requested codes.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date_", "code"]).reset_index(drop=True)
    return panel


def evaluate_practical_factors(panel: pd.DataFrame) -> pd.DataFrame:
    close_wide = panel.pivot(index="date_", columns="code", values="close").sort_index()
    forward_returns = close_wide.pct_change().shift(-1)

    engine = PracticalFactorEngine(panel)
    summaries: list[dict[str, float | str | int]] = []
    for name in available_practical_factors():
        factor_df = engine.compute(name)
        factor_panel = factor_df.pivot(index="date_", columns="code", values=name).sort_index()
        aligned_index = factor_panel.index.intersection(forward_returns.index)
        aligned_cols = factor_panel.columns.intersection(forward_returns.columns)
        factor_panel = factor_panel.loc[aligned_index, aligned_cols]
        fwd = forward_returns.loc[aligned_index, aligned_cols]
        summary = evaluate_factor_panel(factor_panel, fwd)
        summaries.append(
            {
                "factor": name,
                "ic_mean": summary.ic_mean,
                "ic_ir": summary.ic_ir,
                "rank_ic_mean": summary.rank_ic_mean,
                "sharpe": summary.sharpe,
                "turnover": summary.turnover,
                "decay_score": summary.decay_score,
                "capacity_score": summary.capacity_score,
                "correlation_penalty": summary.correlation_penalty,
                "regime_robustness": summary.regime_robustness,
                "out_of_sample_stability": summary.out_of_sample_stability,
                "fitness": summary.fitness,
                "coverage_dates": int(factor_panel.notna().any(axis=1).sum()),
                "coverage_cells": int(factor_panel.notna().sum().sum()),
            }
        )
    return pd.DataFrame(summaries).sort_values(["fitness", "ic_mean"], ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.codes:
        codes = [code.strip().upper() for code in args.codes.split(",") if code.strip()]
    else:
        codes = select_universe(args.universe_csv, args.start, args.end)

    print(f"universe_size={len(codes)}")
    panel = build_adjusted_panel(
        codes,
        start=args.start,
        end=args.end,
        provider=args.provider,
        region=args.region,
        sec_type=args.sec_type,
        db_path=args.db_path,
        batch_size=args.batch_size,
        as_ex=args.as_ex,
    )
    print(f"panel_rows={len(panel):,}, codes={panel['code'].nunique()}, dates={panel['date_'].nunique()}")

    summary = evaluate_practical_factors(panel)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    report_dir = Path(args.report_dir)
    if not summary.empty and str(args.report_dir).strip():
        top_factor = str(summary.iloc[0]["factor"])
        print(f"\nGenerating factor tear sheet for top factor: {top_factor}")
        close_wide = panel.pivot(index="date_", columns="code", values="close").sort_index()
        forward_returns = close_wide.pct_change(fill_method=None).shift(-1)
        engine = PracticalFactorEngine(panel)
        factor_df = engine.compute(top_factor)
        factor_panel = factor_df.pivot(index="date_", columns="code", values=top_factor).sort_index()
        aligned_index = factor_panel.index.intersection(forward_returns.index)
        aligned_cols = factor_panel.columns.intersection(forward_returns.columns)
        factor_panel = factor_panel.loc[aligned_index, aligned_cols]
        fwd = forward_returns.loc[aligned_index, aligned_cols]
        create_native_full_tear_sheet(
            top_factor,
            factor_panel,
            fwd,
            output_dir=report_dir / top_factor,
        )

    print("\nTOP FACTORS BY FITNESS")
    print(summary.to_string(index=False))
    print(f"\nSaved to {output}")
    if str(args.report_dir).strip():
        print(f"Charts saved to {report_dir}")


if __name__ == "__main__":
    main()
