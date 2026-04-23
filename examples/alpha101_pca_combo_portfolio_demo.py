"""Alpha101 PCA composite-factor demo with backtest and local portfolio reports.

This demo keeps the same 8 Alpha101 factors as the candidate universe,
builds a PCA-based composite factor from them, backtests the composite signal,
and optionally opens the local portfolio reports for the resulting portfolio returns.

The flow is:

1. load the eight factor panels from the Tiger factor store
2. align and standardize the factor panels
3. stack them into a PCA matrix
4. fit PCA and build a composite factor from the leading principal components
5. backtest the composite factor against the close panel
6. print the key statistics
7. optionally run local positions, trade, and combined portfolio reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest
from tiger_factors.multifactor_evaluation.common.parquet_utils import to_parquet_clean
from tiger_factors.multifactor_evaluation.common.matplotlib_config import configure_matplotlib
from tiger_factors.multifactor_evaluation.reporting.portfolio import create_position_report
from tiger_factors.multifactor_evaluation.reporting.trades import create_trade_report
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest
from tiger_factors.utils import normalize_cross_section

configure_matplotlib()


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "alpha101_pca_combo_demo"
DEFAULT_PROVIDER = "tiger"
DEFAULT_REGION = "us"
DEFAULT_SEC_TYPE = "stock"
DEFAULT_FREQ = "1d"
DEFAULT_VARIANT = None
DEFAULT_FACTOR_NAMES = (
    "alpha_021",
    "alpha_030",
    "alpha_047",
    "alpha_005",
    "alpha_076",
    "alpha_068",
    "alpha_066",
    "alpha_092",
)
DEFAULT_EXPLAINED_THRESHOLD = 0.80
DEFAULT_LONG_PCT = 0.20
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_ANNUAL_TRADING_DAYS = 252
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PCA composite-factor demo on the balanced Alpha101 shortlist.")
    parser.add_argument("--factor-root", default=str(DEFAULT_FACTOR_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--sec-type", default=DEFAULT_SEC_TYPE)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--factor-names", nargs="*", default=list(DEFAULT_FACTOR_NAMES))
    parser.add_argument(
        "--explained-threshold",
        type=float,
        default=DEFAULT_EXPLAINED_THRESHOLD,
        help="Minimum cumulative explained variance to keep for the composite PCA factor.",
    )
    parser.add_argument(
        "--pc-count",
        type=int,
        default=None,
        help="Optional hard cap on the number of principal components to use.",
    )
    parser.add_argument("--long-pct", type=float, default=DEFAULT_LONG_PCT)
    parser.add_argument("--rebalance-freq", default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--annual-trading-days", type=int, default=DEFAULT_ANNUAL_TRADING_DAYS)
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--skip-report", action="store_true", help="Skip generating local portfolio reports.")
    parser.add_argument("--no-persist", action="store_true", help="Do not write any output files.")
    return parser.parse_args()


def _load_factor_panel(
    store: FactorStore,
    *,
    factor_name: str,
    provider: str,
    region: str,
    sec_type: str,
    freq: str,
    variant: str | None,
) -> pd.DataFrame:
    spec = FactorSpec(
        provider=provider,
        region=region,
        sec_type=sec_type,
        freq=freq,
        table_name=factor_name,
        variant=variant,
    )
    frame = store.get_factor(spec)
    if frame.empty:
        raise FileNotFoundError(
            f"Factor panel not found for {factor_name!r} with spec {spec}. "
            "Use the current factor store layout to persist the factor first."
        )
    factor_series = coerce_factor_series(frame)
    panel = factor_series.unstack("code").sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = "date_"
    return panel


def _load_close_panel(
    store: FactorStore,
    *,
    provider: str,
    region: str,
    sec_type: str,
    freq: str,
) -> pd.DataFrame:
    price_spec = AdjPriceSpec(provider=provider, region=region, sec_type=sec_type, freq=freq)
    price_frame = store.get_adj_price(price_spec)
    close_panel = coerce_price_panel(price_frame)
    if close_panel.empty:
        raise RuntimeError(f"Empty close panel loaded from store using spec: {price_spec}")
    return close_panel


def _align_panels(factor_panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common_index = None
    common_columns = None
    for panel in factor_panels.values():
        common_index = panel.index if common_index is None else common_index.intersection(panel.index)
        common_columns = panel.columns if common_columns is None else common_columns.intersection(panel.columns)
    if common_index is None or common_columns is None or len(common_index) == 0 or len(common_columns) == 0:
        details = {
            name: {
                "shape": tuple(panel.shape),
                "start": None if panel.empty else str(panel.index.min()),
                "end": None if panel.empty else str(panel.index.max()),
                "columns": [] if panel.empty else list(map(str, panel.columns[:5])),
            }
            for name, panel in factor_panels.items()
        }
        raise ValueError(f"No overlapping dates or codes across selected factor panels: {details}")
    return {
        name: panel.reindex(index=common_index, columns=common_columns).sort_index()
        for name, panel in factor_panels.items()
    }


def _stack_panel(panel: pd.DataFrame, name: str) -> pd.Series:
    try:
        stacked = panel.stack(future_stack=True)
    except TypeError:  # pragma: no cover - older pandas fallback
        stacked = panel.stack(dropna=False)
    return stacked.rename(name)


def _build_pca_matrix(factor_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    standardized = {
        name: normalize_cross_section(panel, method="zscore", axis=1)
        for name, panel in factor_panels.items()
    }
    stacked_frames = [_stack_panel(panel, name) for name, panel in standardized.items()]
    matrix = pd.concat(stacked_frames, axis=1).dropna(how="any")
    if matrix.empty:
        raise ValueError("PCA matrix is empty after alignment and standardization.")
    return matrix


def _fit_pca(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    x = matrix.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    explained_variance = (s**2) / max(len(x) - 1, 1)
    total_variance = float(explained_variance.sum())
    explained_ratio = explained_variance / total_variance if total_variance > 1e-12 else np.zeros_like(explained_variance)

    pc_scores = u * s
    pc_columns = [f"PC{i + 1}" for i in range(pc_scores.shape[1])]
    scores = pd.DataFrame(pc_scores, index=matrix.index, columns=pc_columns)
    loadings = pd.DataFrame(vt.T, index=matrix.columns, columns=pc_columns)
    return scores, explained_ratio, loadings


def _choose_pc_count(explained_ratio: np.ndarray, threshold: float, hard_cap: int | None) -> int:
    cumulative = np.cumsum(explained_ratio)
    hit = np.where(cumulative >= threshold)[0]
    count = int(hit[0] + 1) if len(hit) else int(len(explained_ratio))
    if hard_cap is not None:
        count = min(count, max(int(hard_cap), 1))
    return max(count, 1)


def _orient_components(scores: pd.DataFrame, loadings: pd.DataFrame, pc_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    oriented_scores = scores.copy()
    oriented_loadings = loadings.copy()
    for column in list(scores.columns[:pc_count]):
        loading = oriented_loadings[column]
        if float(loading.sum()) < 0:
            oriented_scores[column] = -oriented_scores[column]
            oriented_loadings[column] = -oriented_loadings[column]
    return oriented_scores, oriented_loadings


def _build_composite_factor(
    matrix: pd.DataFrame,
    *,
    explained_threshold: float,
    pc_count: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    scores, explained_ratio, loadings = _fit_pca(matrix)
    use_count = _choose_pc_count(explained_ratio, explained_threshold, pc_count)
    scores, loadings = _orient_components(scores, loadings, use_count)

    component_weights = explained_ratio[:use_count]
    weight_total = float(component_weights.sum())
    if weight_total <= 1e-12:
        component_weights = np.full(use_count, 1.0 / use_count, dtype=float)
    else:
        component_weights = component_weights / weight_total

    composite_score = scores.iloc[:, :use_count].to_numpy(dtype=float) @ component_weights
    composite = pd.Series(composite_score, index=matrix.index, name="pca_composite")
    composite_factor = composite.unstack("code").sort_index()

    explained = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(explained_ratio))],
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance": np.cumsum(explained_ratio),
        }
    )
    meta = {
        "pc_count": use_count,
        "component_weights": {f"PC{i + 1}": float(value) for i, value in enumerate(component_weights)},
        "explained": explained,
        "scores": scores,
        "loadings": loadings,
    }
    return composite_factor, meta


def _run_reports(backtest: pd.DataFrame, *, output_dir: Path) -> None:
    positions = backtest.attrs.get("positions")
    close_panel = backtest.attrs.get("close_panel")
    create_position_report(
        positions,
        output_dir=output_dir / "positions",
        report_name="alpha101_pca_combo",
    )
    create_trade_report(
        backtest["portfolio"],
        positions=positions,
        close_panel=close_panel,
        output_dir=output_dir / "trades",
        report_name="alpha101_pca_combo",
    )
    run_portfolio_from_backtest(
        backtest,
        output_dir=output_dir / "portfolio",
        report_name="alpha101_pca_combo",
    )


def main() -> None:
    args = parse_args()
    factor_root = Path(args.factor_root)
    output_dir = Path(args.output_dir)
    factor_names = [str(name).strip() for name in args.factor_names if str(name).strip()]
    if not factor_names:
        raise RuntimeError("No factor names were provided.")

    store = FactorStore(root_dir=factor_root)
    close_panel = _load_close_panel(
        store,
        provider=args.provider,
        region=args.region,
        sec_type=args.sec_type,
        freq=args.freq,
    )

    factor_panels: dict[str, pd.DataFrame] = {}
    for factor_name in factor_names:
        factor_panels[factor_name] = _load_factor_panel(
            store,
            factor_name=factor_name,
            provider=args.provider,
            region=args.region,
            sec_type=args.sec_type,
            freq=args.freq,
            variant=args.variant,
        )

    aligned = _align_panels(factor_panels)
    matrix = _build_pca_matrix(aligned)
    composite_factor, meta = _build_composite_factor(
        matrix,
        explained_threshold=float(args.explained_threshold),
        pc_count=args.pc_count,
    )

    result = run_factor_backtest(
        composite_factor,
        close_panel,
        long_pct=args.long_pct,
        rebalance_freq=args.rebalance_freq,
        long_short=True,
        annual_trading_days=args.annual_trading_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    backtest, stats = result

    explained = meta["explained"]
    assert isinstance(explained, pd.DataFrame)
    pc_count = int(meta["pc_count"])
    component_weights = meta["component_weights"]
    loadings = meta["loadings"]
    assert isinstance(loadings, pd.DataFrame)

    print("Alpha101 PCA composite-factor demo")
    print("Candidate factors:")
    print(factor_names)
    print(f"\nAligned PCA samples: {len(matrix):,}")
    print(f"Selected principal components used in the composite factor: {pc_count}")

    print("\nExplained variance ratio by component:")
    for _, row in explained.iterrows():
        print(f"  {row['component']}: {float(row['explained_variance_ratio']):.6f}")

    print("\nCumulative explained variance:")
    for _, row in explained.iterrows():
        print(f"  {row['component']}: {float(row['cumulative_explained_variance']):.6f}")

    print("\nComponent weights used in the composite factor:")
    print(json.dumps(component_weights, indent=2, ensure_ascii=False, default=str))

    print("\nTop loadings for the selected principal components:")
    for component in list(loadings.columns[:pc_count]):
        top = loadings[component].abs().sort_values(ascending=False).head(5).index.tolist()
        print(f"  {component}: {top}")

    print("\nBacktest stats:")
    print(json.dumps(stats["portfolio"], indent=2, ensure_ascii=False, default=str))
    print("\nBenchmark stats:")
    print(json.dumps(stats["benchmark"], indent=2, ensure_ascii=False, default=str))

    if not args.no_persist:
        output_dir.mkdir(parents=True, exist_ok=True)
        backtest_copy = backtest.copy()
        backtest_copy.attrs = {}
        to_parquet_clean(composite_factor, output_dir / "alpha101_pca_composite_factor.parquet")
        to_parquet_clean(matrix, output_dir / "alpha101_pca_matrix.parquet")
        to_parquet_clean(explained, output_dir / "alpha101_pca_explained_variance.parquet", index=False)
        to_parquet_clean(loadings.iloc[:, :pc_count], output_dir / "alpha101_pca_loadings.parquet")
        to_parquet_clean(pd.DataFrame(stats).T, output_dir / "alpha101_pca_backtest_stats.parquet")
        to_parquet_clean(backtest_copy, output_dir / "alpha101_pca_backtest_daily.parquet")
        (output_dir / "alpha101_pca_summary.json").write_text(
            json.dumps(
                {
                    "factor_names": factor_names,
                    "selected_pc_count": pc_count,
                    "component_weights": component_weights,
                    "backtest_stats": stats,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved outputs to: {output_dir}")

    if not args.skip_report:
        _run_reports(backtest, output_dir=output_dir)


if __name__ == "__main__":
    main()
