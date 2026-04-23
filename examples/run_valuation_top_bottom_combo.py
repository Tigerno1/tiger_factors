from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_algorithm.valuation_factors.valuation_factor_recorder import record_valuation_factors
from tiger_factors.factor_evaluation.factor_screening import evaluate_and_screen_factor_root
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.pipeline import blend_factor_panels
from tiger_factors.multifactor_evaluation.pipeline import greedy_select_by_correlation
from tiger_factors.multifactor_evaluation.pipeline import factor_correlation_matrix
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest
from tiger_factors.multifactor_evaluation.pipeline import score_to_weights


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk/factor/valuation")
DEFAULT_EVALUATION_ROOT = Path("/Volumes/Quant_Disk")
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_UNIVERSE_PROVIDER = "github"
DEFAULT_UNIVERSE_NAME = "sp500_constituents"
DEFAULT_UNIVERSE_CODE_COLUMN = "code"
DEFAULT_PRICE_PROVIDER = "simfin"
DEFAULT_TOP_N = 10
DEFAULT_BOTTOM_N = 10
DEFAULT_CORR_THRESHOLD = 0.75
DEFAULT_MIN_FACTOR_ROWS = 1000
DEFAULT_MIN_FACTOR_CODES = 20
DEFAULT_MIN_FACTOR_DAYS = 365


def _factor_file_to_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    series = coerce_factor_series(frame)
    wide = series.unstack("code").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    wide.index.name = "date_"
    return wide


def _filter_by_data_support(
    screened: pd.DataFrame,
    *,
    min_factor_rows: int,
    min_factor_codes: int,
    min_factor_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = screened.copy()

    if "factor_rows" in working.columns:
        working["factor_rows"] = pd.to_numeric(working["factor_rows"], errors="coerce")
    if "factor_codes" in working.columns:
        working["factor_codes"] = pd.to_numeric(working["factor_codes"], errors="coerce")
    if "factor_date_min" in working.columns and "factor_date_max" in working.columns:
        date_min = pd.to_datetime(working["factor_date_min"], errors="coerce")
        date_max = pd.to_datetime(working["factor_date_max"], errors="coerce")
        working["support_days"] = (date_max - date_min).dt.days.add(1)
    else:
        working["support_days"] = pd.NA

    support_mask = pd.Series(True, index=working.index)
    if "factor_rows" in working.columns:
        support_mask &= working["factor_rows"].fillna(-1) >= min_factor_rows
    if "factor_codes" in working.columns:
        support_mask &= working["factor_codes"].fillna(-1) >= min_factor_codes
    if "support_days" in working.columns:
        support_mask &= working["support_days"].fillna(-1) >= min_factor_days

    supported = working[support_mask].copy().drop(columns=["support_days"], errors="ignore")
    rejected = working[~support_mask].copy().drop(columns=["support_days"], errors="ignore")
    return supported, rejected


def _select_factor_rows(screened: pd.DataFrame, top_n: int, bottom_n: int) -> pd.DataFrame:
    if screened.empty:
        raise RuntimeError("No screened valuation factors were produced.")

    working = screened.copy()
    if "usable" in working.columns and working["usable"].notna().any():
        usable = working[working["usable"].fillna(False)].copy()
        if not usable.empty:
            working = usable

    working["fitness"] = pd.to_numeric(working["fitness"], errors="coerce")
    working["directional_fitness"] = pd.to_numeric(
        working.get("directional_fitness", working["fitness"].abs()), errors="coerce"
    )
    working["direction"] = np.where(
        working.get("direction_hint", pd.Series(index=working.index, dtype=object)).astype(str).eq("reverse_factor"),
        -1.0,
        1.0,
    )
    ic_mean = pd.to_numeric(working.get("ic_mean"), errors="coerce")
    working.loc[ic_mean < 0, "direction"] = -1.0
    working["selection_score"] = working["directional_fitness"].abs()
    working = working.dropna(subset=["fitness"]).sort_values(
        ["selection_score", "ic_mean", "rank_ic_mean", "factor_name"],
        ascending=[False, False, False, True],
    )

    best = working.head(top_n).copy()
    worst = working.tail(bottom_n).copy()
    best["direction"] = best["direction"].fillna(1.0)
    worst["direction"] = -worst["direction"].fillna(1.0)
    best["selection_score"] = best["selection_score"].clip(lower=1e-12)
    worst["selection_score"] = worst["selection_score"].clip(lower=1e-12)
    best["bucket"] = "best"
    worst["bucket"] = "worst"
    chosen = pd.concat([best, worst], ignore_index=True)
    chosen = chosen.drop_duplicates(subset=["factor_name", "direction"], keep="first").reset_index(drop=True)
    chosen["adjusted_factor_name"] = chosen["factor_name"].astype(str) + chosen["direction"].map({1.0: "", -1.0: "_rev"})
    return chosen.reset_index(drop=True)


def _build_selected_panels(selection: pd.DataFrame) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for _, row in selection.iterrows():
        factor_path = Path(row["factor_path"])
        panel = _factor_file_to_panel(factor_path)
        if float(row["direction"]) < 0:
            panel = -panel
        panels[str(row["adjusted_factor_name"])] = panel
    return panels


def _price_panel_for_codes(
    *,
    library: TigerFactorLibrary,
    codes: list[str],
    start: str,
    end: str,
    price_provider: str,
    as_ex: bool | None = None,
) -> pd.DataFrame:
    price = library.fetch_price_data(
        provider=price_provider,
        region=library.region,
        sec_type=library.sec_type,
        freq="1d",
        codes=codes,
        start=start,
        end=end,
        as_ex=as_ex,
    )
    return coerce_price_panel(price)


def run_valuation_top_bottom_combo(
    *,
    factor_root: str | Path = DEFAULT_FACTOR_ROOT,
    evaluation_root: str | Path = DEFAULT_EVALUATION_ROOT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    region: str = "us",
    sec_type: str = "stock",
    price_provider: str = DEFAULT_PRICE_PROVIDER,
    universe_provider: str = DEFAULT_UNIVERSE_PROVIDER,
    universe_name: str = DEFAULT_UNIVERSE_NAME,
    universe_code_column: str = DEFAULT_UNIVERSE_CODE_COLUMN,
    universe_limit: int | None = None,
    max_factors: int | None = None,
    top_n: int = DEFAULT_TOP_N,
    bottom_n: int = DEFAULT_BOTTOM_N,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    price_start_buffer_days: int = 20,
    min_factor_rows: int = DEFAULT_MIN_FACTOR_ROWS,
    min_factor_codes: int = DEFAULT_MIN_FACTOR_CODES,
    min_factor_days: int = DEFAULT_MIN_FACTOR_DAYS,
    record: bool = True,
    verbose: bool = False,
    as_ex: bool | None = None,
) -> dict[str, object]:
    factor_root = Path(factor_root)
    evaluation_root = Path(evaluation_root)
    screening_root = evaluation_root / "valuation_top_bottom_combo" / "screening"
    run_dir = evaluation_root / "valuation_top_bottom_combo" / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    library = TigerFactorLibrary(
        output_dir=factor_root,
        region=region,
        sec_type=sec_type,
        price_provider=price_provider,
        verbose=verbose,
    )

    if record:
        record_valuation_factors(
            library=library,
            start=start,
            end=end,
            output_root=factor_root,
            region=region,
            sec_type=sec_type,
            price_provider=price_provider,
            universe_provider=universe_provider,
            universe_name=universe_name,
            universe_code_column=universe_code_column,
            universe_limit=universe_limit,
            monthly_output=True,
            max_factors=max_factors,
            verbose=verbose,
            ensure_domain=True,
        )

    registry = evaluate_and_screen_factor_root(
        input_root=factor_root,
        output_root=screening_root,
        price_provider=price_provider,
        library=library,
    )
    screened_registry_path = screening_root / "single_factor_screened_registry.parquet"
    if screened_registry_path.exists():
        screened = pd.read_parquet(screened_registry_path)
    else:
        screened = registry.copy()

    supported, rejected = _filter_by_data_support(
        screened,
        min_factor_rows=min_factor_rows,
        min_factor_codes=min_factor_codes,
        min_factor_days=min_factor_days,
    )
    chosen = _select_factor_rows(supported, top_n=top_n, bottom_n=bottom_n)
    selected_panels = _build_selected_panels(chosen)
    correlation_matrix = factor_correlation_matrix(selected_panels, standardize=True)

    selection_scores = {
        str(row["adjusted_factor_name"]): float(row["selection_score"])
        for _, row in chosen.iterrows()
    }
    selected_factors = greedy_select_by_correlation(selection_scores, correlation_matrix, corr_threshold)
    if not selected_factors:
        selected_factors = list(selection_scores.keys())[:1]

    selected_scores = {
        name: float(selection_scores.get(name, np.nan))
        for name in selected_factors
        if pd.notna(selection_scores.get(name, np.nan))
    }
    factor_weights = score_to_weights(selected_scores, selected=selected_factors, method="positive")
    if not factor_weights:
        equal_weight = 1.0 / len(selected_factors)
        factor_weights = {name: equal_weight for name in selected_factors}
    selected_panel_map = {name: selected_panels[name] for name in selected_factors}
    combined_factor = blend_factor_panels(selected_panel_map, factor_weights, standardize=True)
    lagged_combined_factor = combined_factor.shift(1)

    price_codes = list(combined_factor.columns)
    price_buffer_start = str((pd.Timestamp(start) - pd.Timedelta(days=int(price_start_buffer_days))).date())
    price_panel = _price_panel_for_codes(
        library=library,
        codes=price_codes,
        start=price_buffer_start,
        end=end,
        price_provider=price_provider,
        as_ex=as_ex,
    )

    backtest, stats = run_factor_backtest(
        lagged_combined_factor,
        price_panel,
        long_pct=0.20,
        rebalance_freq="ME",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=8.0,
        slippage_bps=4.0,
        start=start,
        end=end,
    )

    selected_detail = chosen[chosen["adjusted_factor_name"].isin(selected_factors)].copy()
    selected_detail = selected_detail[
        [
            "bucket",
            "adjusted_factor_name",
            "factor_name",
            "direction",
            "selection_score",
            "fitness",
            "ic_mean",
            "rank_ic_mean",
            "sharpe",
            "turnover",
            "factor_path",
        ]
    ].sort_values("selection_score", ascending=False)

    screened.to_parquet(run_dir / "screened_registry.parquet", index=False)
    supported.to_parquet(run_dir / "supported_registry.parquet", index=False)
    rejected.to_parquet(run_dir / "rejected_for_insufficient_support.parquet", index=False)
    chosen[chosen["bucket"] == "best"].to_parquet(run_dir / "best_factors.parquet", index=False)
    chosen[chosen["bucket"] == "worst"].to_parquet(run_dir / "worst_factors.parquet", index=False)
    chosen.to_parquet(run_dir / "top_bottom_candidates.parquet", index=False)
    selected_detail.to_parquet(run_dir / "selected_factors.parquet", index=False)
    correlation_matrix.to_parquet(run_dir / "selected_factor_correlation_matrix.parquet")
    combined_factor.to_parquet(run_dir / "combined_factor.parquet")
    lagged_combined_factor.to_parquet(run_dir / "combined_factor_lagged.parquet")
    backtest.to_parquet(run_dir / "backtest_daily.parquet")
    pd.DataFrame(stats).T.to_parquet(run_dir / "backtest_stats.parquet")
    pd.DataFrame(list(factor_weights.items()), columns=["factor_name", "weight"]).to_parquet(
        run_dir / "factor_weights.parquet",
        index=False,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "factor_root": str(factor_root),
        "screening_root": str(screening_root),
        "evaluation_root": str(evaluation_root),
        "run_dir": str(run_dir),
        "start": start,
        "end": end,
        "region": region,
        "sec_type": sec_type,
        "price_provider": price_provider,
        "universe_provider": universe_provider,
        "universe_name": universe_name,
        "universe_code_column": universe_code_column,
        "universe_limit": universe_limit,
        "max_factors": max_factors,
        "top_n": top_n,
        "bottom_n": bottom_n,
        "corr_threshold": corr_threshold,
        "min_factor_rows": min_factor_rows,
        "min_factor_codes": min_factor_codes,
        "min_factor_days": min_factor_days,
        "selected_factors": selected_factors,
        "factor_weights": factor_weights,
        "portfolio_stats": stats["portfolio"],
        "benchmark_stats": stats["benchmark"],
        "factor_count": int(len(screened)) if not screened.empty else 0,
        "selected_count": int(len(selected_factors)),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Record, screen, and backtest valuation top/bottom factors.")
    parser.add_argument("--factor-root", default=str(DEFAULT_FACTOR_ROOT))
    parser.add_argument("--evaluation-root", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--region", default="us")
    parser.add_argument("--sec-type", default="stock")
    parser.add_argument("--price-provider", default=DEFAULT_PRICE_PROVIDER)
    parser.add_argument("--universe-provider", default=DEFAULT_UNIVERSE_PROVIDER)
    parser.add_argument("--universe-name", default=DEFAULT_UNIVERSE_NAME)
    parser.add_argument("--universe-code-column", default=DEFAULT_UNIVERSE_CODE_COLUMN)
    parser.add_argument("--universe-limit", type=int, default=None)
    parser.add_argument("--max-factors", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--bottom-n", type=int, default=DEFAULT_BOTTOM_N)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument("--min-factor-rows", type=int, default=DEFAULT_MIN_FACTOR_ROWS)
    parser.add_argument("--min-factor-codes", type=int, default=DEFAULT_MIN_FACTOR_CODES)
    parser.add_argument("--min-factor-days", type=int, default=DEFAULT_MIN_FACTOR_DAYS)
    parser.add_argument("--price-start-buffer-days", type=int, default=20)
    parser.add_argument("--no-record", action="store_true", help="Skip recording valuation factors and use existing files.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_valuation_top_bottom_combo(
        factor_root=args.factor_root,
        evaluation_root=args.evaluation_root,
        start=args.start,
        end=args.end,
        region=args.region,
        sec_type=args.sec_type,
        price_provider=args.price_provider,
        universe_provider=args.universe_provider,
        universe_name=args.universe_name,
        universe_code_column=args.universe_code_column,
        universe_limit=args.universe_limit,
        max_factors=args.max_factors,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        corr_threshold=args.corr_threshold,
        min_factor_rows=args.min_factor_rows,
        min_factor_codes=args.min_factor_codes,
        min_factor_days=args.min_factor_days,
        price_start_buffer_days=args.price_start_buffer_days,
        record=not args.no_record,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
