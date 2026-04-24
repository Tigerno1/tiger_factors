"""Run screener first, then feed the selected factors into multifactor evaluation.

This demo shows the intended boundary between the two modules:

1. ``factor_screener`` filters a candidate factor basket
2. ``multifactor_evaluation`` takes the selected factors and builds the
   composite backtest plus report

The script expects the factors and prices to already exist in the local Tiger
factor store. It is intentionally explicit so you can swap in your own factor
names, providers, and universe controls.
"""

from __future__ import annotations

import argparse
import json
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
from tiger_factors.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener import run_factor_screener_flow
from tiger_factors.multifactor_evaluation import MultifactorEvaluation
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "factor_screener_multifactor_demo"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen factors first, then run a multifactor backtest and report."
    )
    parser.add_argument("--factor-root", default=str(DEFAULT_FACTOR_STORE_ROOT), help="Root directory of the Tiger factor store.")
    parser.add_argument("--factor-provider", default="tiger", help="Provider namespace used when the factors were saved.")
    parser.add_argument("--price-provider", default="yahoo", help="Price provider used for the matching close panel.")
    parser.add_argument("--factor-variant", default=None, help="Optional factor variant used when the factors were saved.")
    parser.add_argument("--region", default="us")
    parser.add_argument("--sec-type", default="stock")
    parser.add_argument("--freq", default="1d")
    parser.add_argument("--factor-names", nargs="+", default=list(DEFAULT_FACTOR_NAMES), help="Candidate factors to screen.")
    parser.add_argument("--codes", nargs="*", default=None, help="Optional explicit universe to keep.")
    parser.add_argument("--start", default=None, help="Optional start date.")
    parser.add_argument("--end", default=None, help="Optional end date.")
    parser.add_argument(
        "--selection-mode",
        default="return_gain",
        choices=("correlation", "conditional", "return_gain"),
        help="How to do the final cross-factor selection.",
    )
    parser.add_argument(
        "--return-gain-preset",
        default="balanced",
        choices=("balanced", "metric_focused", "return_focused", "robust"),
        help="Preset used when selection_mode=return_gain.",
    )
    parser.add_argument("--long-pct", type=float, default=0.20)
    parser.add_argument("--rebalance-freq", default="ME")
    parser.add_argument("--annual-trading-days", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="factor_screener_multifactor_demo")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def _infer_common_universe(
    factor_panels: dict[str, pd.DataFrame],
    *,
    codes: list[str] | None = None,
) -> tuple[list[str], str | None, str | None]:
    if not factor_panels:
        raise ValueError("factor_panels must not be empty")

    common_dates: pd.Index | None = None
    common_codes: set[str] | None = None
    for panel in factor_panels.values():
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            continue
        dates = pd.Index(pd.to_datetime(panel.index, errors="coerce")).dropna()
        panel_codes = {str(code) for code in panel.columns}
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
        common_codes = panel_codes if common_codes is None else common_codes.intersection(panel_codes)

    if common_dates is None or common_dates.empty:
        raise ValueError("Could not infer any overlapping dates from the candidate factor panels.")
    if common_codes is None or not common_codes:
        raise ValueError("Could not infer any overlapping codes from the candidate factor panels.")

    selected_codes = sorted(common_codes)
    if codes is not None:
        requested_codes = [str(code) for code in codes]
        selected_codes = [code for code in requested_codes if code in common_codes]
    if not selected_codes:
        raise ValueError("No overlapping codes were found after applying the requested universe filter.")

    start = str(pd.Timestamp(common_dates.min()).date())
    end = str(pd.Timestamp(common_dates.max()).date())
    return selected_codes, start, end


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variant = _normalize_variant(args.factor_variant)
    factor_names = [str(name) for name in args.factor_names]

    library = TigerFactorLibrary(
        output_dir=args.factor_root,
        price_provider=args.price_provider,
        verbose=True,
    )

    factor_panels = library.load_factor_panels(
        factor_names=factor_names,
        provider=args.factor_provider,
        freq=args.freq,
        variant=variant,
        codes=args.codes,
        start=args.start,
        end=args.end,
    )
    if not factor_panels:
        raise ValueError("No factor panels could be loaded from the store.")

    selected_codes, inferred_start, inferred_end = _infer_common_universe(factor_panels, codes=args.codes)
    start = args.start or inferred_start
    end = args.end or inferred_end

    close_panel = library.price_panel(
        codes=selected_codes,
        start=start,
        end=end,
        provider=args.price_provider,
        field="close",
    )
    if close_panel.empty:
        raise ValueError("Could not load a matching close panel for the selected universe.")

    close_panel = close_panel.reindex(index=pd.to_datetime(close_panel.index, errors="coerce")).sort_index()
    close_panel = close_panel.reindex(columns=selected_codes)

    screener = FactorScreenerSpec(
        factor_names=tuple(factor_names),
        provider=args.factor_provider,
        region=args.region,
        sec_type=args.sec_type,
        freq=args.freq,
        variant=variant,
        price_panel=close_panel,
        preferred_return_period="1D",
        return_modes=("long_short", "long_only"),
        selection_threshold=0.75,
        selection_score_field="fitness",
    )

    screener_result = run_factor_screener_flow(
        [screener],
        selection_mode=args.selection_mode,
        return_gain_preset=args.return_gain_preset,
        save_dir=output_dir / "screener",
    )

    selected_factor_names = screener_result.global_selected_factor_names or screener_result.selected_factor_names
    if not selected_factor_names:
        raise ValueError("Screener did not select any factors.")

    selected_factor_panels = {
        name: factor_panels[name].reindex(index=close_panel.index, columns=close_panel.columns)
        for name in selected_factor_names
        if name in factor_panels
    }
    if not selected_factor_panels:
        raise ValueError("None of the selected factors were available in the loaded factor panels.")

    backtest_result = multi_factor_backtest(
        selected_factor_panels,
        close_panel,
        standardize=True,
        long_pct=args.long_pct,
        rebalance_freq=args.rebalance_freq,
        long_short=True,
        annual_trading_days=args.annual_trading_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    factor_data = {name: selected_factor_panels[name] for name in selected_factor_panels}
    evaluation = MultifactorEvaluation(
        backtest=backtest_result["backtest"],
        positions_frame=backtest_result["backtest"].attrs.get("positions"),
        close_panel_frame=close_panel,
        factor_data=factor_data,
        output_dir=output_dir / "multifactor",
        report_name=args.report_name,
        capital_base=1_000_000.0,
    )
    bundle = evaluation.full(
        backtest_result["backtest"],
        output_dir=output_dir / "multifactor",
        report_name=args.report_name,
        open_browser=args.open_browser,
    )

    manifest = {
        "factor_root": str(args.factor_root),
        "factor_names": factor_names,
        "selected_factor_names": selected_factor_names,
        "selection_mode": args.selection_mode,
        "return_gain_preset": args.return_gain_preset,
        "start": start,
        "end": end,
        "output_dir": str(output_dir),
        "screener_detail_manifest": None if screener_result.detail_manifest is None else screener_result.detail_manifest.to_dict(),
        "multifactor_report_path": None if bundle.report_path is None else str(bundle.report_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("screened factors:")
    print(f"  candidates: {factor_names}")
    print(f"  selected: {selected_factor_names}")
    print("\nbacktest stats:")
    print(pd.DataFrame(backtest_result["stats"]).T.to_string())
    print("\noutputs:")
    print(f"  screener detail: {output_dir / 'screener'}")
    print(f"  multifactor report: {bundle.report_path}")
    print(f"  workflow manifest: {output_dir / 'workflow_manifest.json'}")


if __name__ == "__main__":
    main()
