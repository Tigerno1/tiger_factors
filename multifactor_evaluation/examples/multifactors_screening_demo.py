"""Diagnose missing single-factor reports and demo multifactors screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_evaluation.utils import MaxLossExceededError
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import build_factor_registry_from_root
from tiger_factors.factor_screener import screen_factor_metrics
from tiger_factors.factor_screener import screen_factor_registry


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk/factor")
DEFAULT_PRICE_PATH = DEFAULT_FACTOR_ROOT / "price" / "tiger" / "us" / "stock" / "1d" / "adj_price.parquet"
DEFAULT_DIAGNOSTIC_FACTORS = ("alpha_007", "alpha_021", "alpha_046", "alpha_049")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Quant_Disk/evaluation/multifactors_screening_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose missing single-factor reports and screen factors with Tiger multifactors.",
    )
    parser.add_argument(
        "--factor-root",
        default=str(DEFAULT_FACTOR_ROOT),
        help="Root directory containing alpha_*/alpha_*.parquet factor files.",
    )
    parser.add_argument(
        "--price-path",
        default=str(DEFAULT_PRICE_PATH),
        help="Adjusted price parquet used for factor evaluation.",
    )
    parser.add_argument(
        "--factor-names",
        nargs="*",
        default=list(DEFAULT_DIAGNOSTIC_FACTORS),
        help="Specific factor names to diagnose. Ignored when --all-factors is set.",
    )
    parser.add_argument(
        "--all-factors",
        action="store_true",
        help="Evaluate every alpha_* factor found under --factor-root.",
    )
    parser.add_argument(
        "--summary-root",
        default="",
        help="Optional root containing strategy/summary/evaluation.parquet files for registry screening.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for persisting diagnostics and screening results.",
    )
    parser.add_argument(
        "--persist-outputs",
        action="store_true",
        help="Write the diagnostics and screening outputs to --output-dir.",
    )
    parser.add_argument("--min-ic-mean", type=float, default=0.01)
    parser.add_argument("--min-rank-ic-mean", type=float, default=0.01)
    parser.add_argument("--min-sharpe", type=float, default=0.40)
    parser.add_argument("--max-turnover", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def _resolve_factor_names(factor_root: Path, factor_names: list[str], all_factors: bool) -> list[str]:
    if all_factors:
        names = sorted(path.name for path in factor_root.glob("alpha_*") if path.is_dir())
        return [name for name in names if (factor_root / name / f"{name}.parquet").exists()]
    return [name for name in factor_names if (factor_root / name / f"{name}.parquet").exists()]


def _load_factor_frame(factor_root: Path, factor_name: str) -> pd.DataFrame:
    path = factor_root / factor_name / f"{factor_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")
    return pd.read_parquet(path)


def _load_price_frame(price_path: Path) -> pd.DataFrame:
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")
    return pd.read_parquet(price_path)


def _guess_report_issue(factor_frame: pd.DataFrame, factor_column: str, clean_status: str, clean_error: str | None) -> str:
    factor_series = pd.to_numeric(factor_frame[factor_column], errors="coerce")
    issues: list[str] = []
    if factor_series.dropna().empty:
        issues.append("all factor values are missing")
    unique_values = int(factor_series.nunique(dropna=True))
    if unique_values <= 1:
        issues.append("factor is constant")
    elif unique_values <= 2:
        issues.append("factor is almost binary")
    if factor_frame["date_"].nunique(dropna=True) < 20:
        issues.append("too few dates")
    if factor_frame["code"].nunique(dropna=True) < 10:
        issues.append("too few assets")
    if clean_status != "ok":
        issues.append(clean_status)
    if clean_error:
        issues.append(clean_error)
    return "; ".join(issues) if issues else "no obvious blocker"


def _diagnose_factor(
    factor_root: Path,
    price_frame: pd.DataFrame,
    factor_name: str,
) -> dict[str, object]:
    factor_frame = _load_factor_frame(factor_root, factor_name)
    factor_series = pd.to_numeric(factor_frame[factor_name], errors="coerce")
    dominant_share = float(factor_series.value_counts(normalize=True, dropna=True).iloc[0]) if not factor_series.dropna().empty else float("nan")

    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=factor_name,
    )
    evaluation = engine.evaluate()

    clean_status = "ok"
    clean_error: str | None = None
    clean_rows = None
    clean_quantiles = None
    try:
        cleaned = engine.get_clean_factor_and_forward_returns(periods=(1, 5, 10), quantiles=5, max_loss=0.35)
        clean_rows = int(len(cleaned.factor_data))
        clean_quantiles = int(cleaned.factor_data["factor_quantile"].nunique(dropna=True))
    except MaxLossExceededError as exc:
        clean_status = "max_loss_exceeded"
        clean_error = str(exc)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        clean_status = type(exc).__name__
        clean_error = str(exc)

    return {
        "factor_name": factor_name,
        "rows": int(len(factor_frame)),
        "dates": int(factor_frame["date_"].nunique(dropna=True)),
        "codes": int(factor_frame["code"].nunique(dropna=True)),
        "non_na_rows": int(factor_series.notna().sum()),
        "unique_values": int(factor_series.nunique(dropna=True)),
        "dominant_value_share": dominant_share,
        "factor_min": float(factor_series.min()) if factor_series.notna().any() else float("nan"),
        "factor_max": float(factor_series.max()) if factor_series.notna().any() else float("nan"),
        "clean_status": clean_status,
        "clean_error": clean_error,
        "clean_rows": clean_rows,
        "clean_quantiles": clean_quantiles,
        "report_issue_guess": _guess_report_issue(factor_frame, factor_name, clean_status, clean_error),
        **evaluation.__dict__,
    }


def main() -> None:
    args = parse_args()
    factor_root = Path(args.factor_root)
    price_path = Path(args.price_path)
    price_frame = _load_price_frame(price_path)

    factor_names = _resolve_factor_names(factor_root, list(args.factor_names), args.all_factors)
    if not factor_names:
        raise RuntimeError(f"No factor parquet files found under {factor_root}")

    diagnostics = [_diagnose_factor(factor_root, price_frame, name) for name in factor_names]
    diagnostics_df = pd.DataFrame(diagnostics)

    screen_config = FactorMetricFilterConfig(
        min_ic_mean=args.min_ic_mean,
        min_rank_ic_mean=args.min_rank_ic_mean,
        min_sharpe=args.min_sharpe,
        max_turnover=args.max_turnover,
        sort_field="fitness",
        tie_breaker_field="ic_ir",
    )
    screened = screen_factor_metrics(diagnostics_df, config=screen_config)

    print("\nSingle-factor diagnostics")
    print(
        diagnostics_df[
            [
                "factor_name",
                "rows",
                "dates",
                "codes",
                "unique_values",
                "dominant_value_share",
                "clean_status",
                "report_issue_guess",
                "ic_mean",
                "ic_ir",
                "sharpe",
                "turnover",
                "fitness",
            ]
        ].to_string(index=False)
    )

    print("\nMultifactor screening result")
    if screened.empty:
        print("No factors survived the screen.")
    else:
        cols = [col for col in ["factor_name", "usable", "failed_rules", "ic_mean", "ic_ir", "sharpe", "turnover", "fitness"] if col in screened.columns]
        print(screened[cols].head(max(args.top_n, 1)).to_string(index=False))

    summary_root = Path(args.summary_root) if args.summary_root else None
    if summary_root is not None:
        summary_registry = build_factor_registry_from_root(summary_root)
        if summary_registry.empty:
            print("\nNo summary/evaluation.parquet files were found under --summary-root.")
            print("That means there is no persisted report bundle to screen yet.")
        else:
            summary_screened = screen_factor_registry(summary_registry, config=screen_config)
            print("\nSummary-registry screening result")
            print(summary_screened.head(max(args.top_n, 1)).to_string(index=False))
    if args.persist_outputs:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_parquet(output_dir / "multifactors_diagnostic_registry.parquet", index=False)
        screened.to_parquet(output_dir / "multifactors_screened.parquet", index=False)
        if summary_root is not None:
            summary_registry = build_factor_registry_from_root(summary_root)
            if not summary_registry.empty:
                summary_registry.to_parquet(output_dir / "multifactors_summary_registry.parquet", index=False)
                screen_factor_registry(summary_registry, config=screen_config).to_parquet(
                    output_dir / "multifactors_summary_screened.parquet",
                    index=False,
                )
        (output_dir / "multifactors_screening_demo.json").write_text(
            json.dumps(
                {
                    "factor_root": str(factor_root),
                    "price_path": str(price_path),
                    "factor_names": factor_names,
                    "screening_config": screen_config.__dict__,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nPersisted diagnostics to: {output_dir}")
    else:
        print("\nNo files were written; the demo only returned in-memory diagnostics.")


if __name__ == "__main__":
    main()
