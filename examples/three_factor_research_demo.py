"""End-to-end three-factor research demo.

This example walks through the full Tiger Factors workflow with only three
toy factors:

1. resolve a demo universe and fetch price data
2. build three factor panels from the close prices
3. calculate and save each factor in long-table form
4. evaluate each factor individually and write tear sheets
5. screen the three factors, pick a low-correlation subset, and blend them
6. backtest the blended factor and write a final report

The script is intentionally small and self-contained. It is meant as a
research demo, not as a production factor library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Optional fallback if tiger_api data access is not available locally.
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency fallback
    yf = None  # type: ignore[assignment]

from tiger_factors.factor_evaluation import create_native_full_tear_sheet
from tiger_factors.factor_evaluation import evaluate_factor_panel
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_store import to_long_factor
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import factor_correlation_matrix
from tiger_factors.factor_screener import screen_factor_metrics
from tiger_factors.multifactor_evaluation.pipeline import (
    blend_factor_panels,
    greedy_select_by_correlation,
    score_to_weights,
    run_factor_backtest,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "three_factor_research_demo"
DEFAULT_UNIVERSE_PROVIDER = "github"
DEFAULT_UNIVERSE_DATASET = "sp500_constituents"
DEFAULT_CODE_COLUMN = "code"
DEFAULT_LIMIT = 80
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_FORWARD_DAYS = 20
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_LONG_PCT = 0.20
DEFAULT_CORR_THRESHOLD = 0.80
DEFAULT_FALLBACK_CODES = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "JPM",
    "XOM",
    "UNH",
    "COST",
    "V",
    "MA",
    "HD",
    "PG",
    "LLY",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a three-factor end-to-end research demo.",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--price-provider", default="yahoo")
    parser.add_argument("--universe-provider", default=DEFAULT_UNIVERSE_PROVIDER)
    parser.add_argument("--universe-dataset", default=DEFAULT_UNIVERSE_DATASET)
    parser.add_argument("--code-column", default=DEFAULT_CODE_COLUMN)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--codes", nargs="*", default=None, help="Optional code override.")
    parser.add_argument("--as-ex", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    parser.add_argument("--rebalance-freq", default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--long-pct", type=float, default=DEFAULT_LONG_PCT)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument(
        "--skip-tearsheets",
        action="store_true",
        help="Skip writing single-factor and combined-factor tear sheets.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not save factor parquet/metadata outputs.",
    )
    return parser.parse_args()


def _download_close_panel_with_yfinance(codes: list[str], *, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available, and tiger_api price fetch failed.")

    raw = yf.download(codes, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0).unique())
        price_fields = {"Open", "High", "Low", "Close", "Volume"}
        slice_level = 1 if price_fields.issubset(level0) else 0
        for code in codes:
            try:
                sub = raw.xs(code, axis=1, level=slice_level)
            except Exception:
                continue
            if "Close" not in sub.columns:
                continue
            frame = sub[["Close"]].copy().rename(columns={"Close": code})
            frames.append(frame)
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        code = codes[0]
        frames.append(raw[["Close"]].copy().rename(columns={"Close": code}))

    if not frames:
        return pd.DataFrame()
    close_panel = pd.concat(frames, axis=1).sort_index()
    close_index = pd.DatetimeIndex(close_panel.index)
    if close_index.tz is not None:
        close_index = close_index.tz_localize(None)
    close_panel.index = close_index
    close_panel = close_panel.apply(pd.to_numeric, errors="coerce")
    return close_panel.dropna(how="all")


def resolve_universe(library: TigerFactorLibrary, args: argparse.Namespace) -> tuple[list[str], str]:
    if args.codes:
        codes = [str(code).strip().upper() for code in args.codes if str(code).strip()]
        codes = list(dict.fromkeys(codes))
        return codes, "cli"

    try:
        codes = library.resolve_universe_codes(
            provider=args.universe_provider,
            dataset=args.universe_dataset,
            code_column=args.code_column,
            limit=args.limit,
            as_ex=args.as_ex,
        )
        if codes:
            return codes, f"{args.universe_provider}:{args.universe_dataset}"
    except Exception as exc:  # pragma: no cover - demo fallback
        print(f"Universe lookup failed, using fallback basket: {exc}")

    codes = DEFAULT_FALLBACK_CODES[: max(int(args.limit), 1)] if args.limit else list(DEFAULT_FALLBACK_CODES)
    return codes, "fallback"


def fetch_close_panel(
    library: TigerFactorLibrary,
    codes: list[str],
    *,
    start: str,
    end: str,
    price_provider: str,
    as_ex: bool | None,
) -> tuple[pd.DataFrame, str]:
    source = f"tiger_api:{price_provider}"
    try:
        close_panel = library.price_panel(
            codes=codes,
            start=start,
            end=end,
            provider=price_provider,
            field="close",
            as_ex=as_ex,
        )
        if not close_panel.empty:
            return close_panel, source
    except Exception as exc:  # pragma: no cover - demo fallback
        print(f"Price fetch via tiger_api failed, falling back to yfinance: {exc}")

    close_panel = _download_close_panel_with_yfinance(codes, start=start, end=end)
    if close_panel.empty:
        raise RuntimeError("Could not build a close panel from tiger_api or yfinance.")
    return close_panel, "yfinance"


def build_demo_factor_panels(close_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = close_panel.pct_change(fill_method=None)
    factor_panels = {
        "demo_momentum_20": close_panel.pct_change(20),
        "demo_reversal_5": -close_panel.pct_change(5),
        "demo_low_vol_20": -returns.rolling(20, min_periods=20).std(ddof=0),
    }
    return {name: panel.sort_index() for name, panel in factor_panels.items()}


def save_factor_panels(
    library: TigerFactorLibrary,
    factor_panels: dict[str, pd.DataFrame],
    *,
    persist_outputs: bool,
) -> dict[str, dict[str, str]]:
    saved: dict[str, dict[str, str]] = {}
    if not persist_outputs:
        return saved

    for factor_name, panel in factor_panels.items():
        long_df = to_long_factor(panel, factor_name)
        result = library.save_factor(
            factor_name=factor_name,
            factor_df=long_df,
            metadata={
                "provider": "demo",
                "family": "price_based",
                "construction": factor_name,
            },
        )
        saved[factor_name] = {
            "parquet_path": str(result.parquet_path),
            "metadata_path": str(result.metadata_path),
        }
    return saved


def evaluate_single_factors(
    library: TigerFactorLibrary,
    factor_panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    output_dir: Path,
    write_reports: bool,
    persist_outputs: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    report_root = output_dir / "single_factor_reports"
    report_root.mkdir(parents=True, exist_ok=True)

    saved_factor_paths = save_factor_panels(library, factor_panels, persist_outputs=persist_outputs)
    rows: list[dict[str, object]] = []
    report_dirs: dict[str, dict[str, str]] = {}

    for factor_name, factor_panel in factor_panels.items():
        evaluation = evaluate_factor_panel(
            factor_panel,
            forward_returns,
            benchmark_returns=benchmark_returns,
        )
        factor_rows = int(factor_panel.notna().sum().sum())
        coverage = float(factor_panel.notna().mean().mean()) if not factor_panel.empty else 0.0
        row: dict[str, object] = {
            "factor_name": factor_name,
            "coverage_cells": factor_rows,
            "coverage_ratio": coverage,
            **evaluation.__dict__,
        }

        if write_reports:
            factor_report_dir = report_root / factor_name
            tear_sheet = create_native_full_tear_sheet(
                factor_name,
                factor_panel,
                forward_returns,
                output_dir=factor_report_dir,
                benchmark_returns=benchmark_returns,
            )
            row["tear_sheet_dir"] = str(tear_sheet.output_dir)
            report_dirs[factor_name] = {
                "tear_sheet_dir": str(tear_sheet.output_dir),
            }
        else:
            row["tear_sheet_dir"] = None

        if factor_name in saved_factor_paths:
            row.update(saved_factor_paths[factor_name])
        rows.append(row)

    return pd.DataFrame(rows), report_dirs


def build_combined_factor(
    factor_panels: dict[str, pd.DataFrame],
    screened: pd.DataFrame,
    *,
    corr_threshold: float,
) -> tuple[list[str], dict[str, float], pd.DataFrame, pd.DataFrame]:
    if screened.empty:
        raise ValueError("No factor metrics were available for screening.")

    metrics = screened.copy()
    if "factor_name" not in metrics.columns:
        metrics = metrics.reset_index()
    metrics = metrics.set_index("factor_name", drop=False)

    candidate_names = [str(name) for name in metrics["factor_name"].tolist() if name in factor_panels]
    if not candidate_names:
        raise ValueError("No candidate factor exists in factor_panels.")

    candidate_panels = {name: factor_panels[name] for name in candidate_names}
    corr = factor_correlation_matrix(candidate_panels, standardize=True)

    score_field = "fitness" if "fitness" in metrics.columns else "ic_ir"
    scores = {
        name: float(metrics.loc[name, score_field])
        for name in candidate_names
        if name in metrics.index
    }
    selected = greedy_select_by_correlation(scores, corr, corr_threshold)
    if not selected:
        selected = candidate_names[:1]

    selected_scores = {name: float(scores.get(name, 0.0)) for name in selected}
    weights = score_to_weights(selected_scores, selected=selected, method="softmax", temperature=1.0)
    combined = blend_factor_panels({name: factor_panels[name] for name in selected}, weights, standardize=True)
    return selected, weights, corr, combined


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    library = TigerFactorLibrary(
        output_dir=output_dir / "factor_store",
        region="us",
        sec_type="stock",
        price_provider=args.price_provider,
        verbose=True,
    )

    codes, universe_source = resolve_universe(library, args)
    close_panel, price_source = fetch_close_panel(
        library,
        codes,
        start=args.start,
        end=args.end,
        price_provider=args.price_provider,
        as_ex=args.as_ex,
    )
    if close_panel.empty:
        raise RuntimeError("Close panel is empty after data fetch.")

    close_panel = close_panel.sort_index().ffill()
    factor_panels = build_demo_factor_panels(close_panel)
    forward_returns = close_panel.pct_change(args.forward_days, fill_method=None).shift(-args.forward_days)
    benchmark_returns = forward_returns.mean(axis=1)

    report_root = output_dir / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    single_factor_df, single_factor_reports = evaluate_single_factors(
        library,
        factor_panels,
        forward_returns,
        benchmark_returns,
        output_dir=report_root,
        write_reports=not args.skip_tearsheets,
        persist_outputs=not args.no_persist,
    )

    screen_config = FactorMetricFilterConfig(
        min_fitness=None,
        min_ic_mean=None,
        min_rank_ic_mean=None,
        min_sharpe=None,
        max_turnover=None,
        min_decay_score=None,
        min_capacity_score=None,
        max_correlation_penalty=None,
        min_regime_robustness=None,
        min_out_of_sample_stability=None,
        sort_field="fitness",
        tie_breaker_field="ic_ir",
    )
    screened = screen_factor_metrics(single_factor_df, config=screen_config)
    selected_factors, factor_weights, corr, combined_factor = build_combined_factor(
        factor_panels,
        screened,
        corr_threshold=float(args.corr_threshold),
    )

    combined_backtest, combined_stats = run_factor_backtest(
        combined_factor,
        close_panel,
        long_pct=float(args.long_pct),
        rebalance_freq=str(args.rebalance_freq),
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    combined_report_dir = report_root / "combined_factor"
    if not args.skip_tearsheets:
        create_native_full_tear_sheet(
            "combined_demo_factor",
            combined_factor,
            forward_returns.loc[combined_factor.index.intersection(forward_returns.index)],
            output_dir=combined_report_dir,
            portfolio_returns=combined_backtest["portfolio"] if "portfolio" in combined_backtest else None,
            benchmark_returns=combined_backtest["benchmark"] if "benchmark" in combined_backtest else None,
        )

    if not args.no_persist:
        single_factor_df.to_parquet(output_dir / "single_factor_evaluations.parquet", index=False)
        screened.to_parquet(output_dir / "screened_factors.parquet", index=False)
        corr.to_csv(output_dir / "factor_correlation_matrix.csv")
        combined_factor.to_parquet(output_dir / "combined_factor.parquet")
        combined_backtest.to_parquet(output_dir / "combined_backtest.parquet")
        pd.DataFrame(combined_stats).T.to_csv(output_dir / "combined_backtest_stats.csv")
        pd.Series(factor_weights, name="weight").to_frame().to_csv(output_dir / "factor_weights.csv")

    summary = {
        "universe_source": universe_source,
        "price_source": price_source,
        "universe_size": len(codes),
        "start": args.start,
        "end": args.end,
        "forward_days": args.forward_days,
        "rebalance_freq": args.rebalance_freq,
        "long_pct": args.long_pct,
        "codes_sample": codes[:10],
        "factor_names": list(factor_panels.keys()),
        "single_factor_metrics": single_factor_df[
            [
                "factor_name",
                "ic_mean",
                "ic_ir",
                "rank_ic_mean",
                "sharpe",
                "turnover",
                "fitness",
                "coverage_cells",
                "coverage_ratio",
            ]
        ].to_dict(orient="records"),
        "screened_factors": screened[
            [column for column in ["factor_name", "usable", "failed_rules", "ic_mean", "ic_ir", "sharpe", "turnover", "fitness"] if column in screened.columns]
        ].to_dict(orient="records"),
        "selected_factors": selected_factors,
        "factor_weights": factor_weights,
        "backtest_stats": combined_stats,
        "output_dir": str(output_dir),
        "combined_report_dir": str(combined_report_dir),
        "single_factor_reports": single_factor_reports,
    }
    _write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
