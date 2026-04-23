"""Alpha101 PCA demo.

This demo loads the balanced Alpha101 shortlist, standardizes the factor panels,
and runs PCA to show:

- how many principal components are needed to explain most variance
- the explained variance ratio of each component
- the cumulative explained variance
- the leading factor loadings of the first few components

The demo is intentionally simple and uses the local Tiger factor store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.utils import normalize_cross_section


DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk")
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
DEFAULT_CUMULATIVE_THRESHOLDS = (0.70, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA on the selected Alpha101 factor shortlist.")
    parser.add_argument("--factor-root", default=str(DEFAULT_FACTOR_ROOT))
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--sec-type", default=DEFAULT_SEC_TYPE)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--factor-names", nargs="*", default=list(DEFAULT_FACTOR_NAMES))
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=list(DEFAULT_CUMULATIVE_THRESHOLDS),
        help="Cumulative explained variance thresholds to report.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write PCA outputs.",
    )
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


def _align_panels(factor_panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common_index = None
    common_columns = None
    for panel in factor_panels.values():
        common_index = panel.index if common_index is None else common_index.intersection(panel.index)
        common_columns = panel.columns if common_columns is None else common_columns.intersection(panel.columns)
    if common_index is None or common_columns is None or len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("No overlapping dates or codes across selected factor panels.")
    return {
        name: panel.reindex(index=common_index, columns=common_columns).sort_index()
        for name, panel in factor_panels.items()
    }


def _build_pca_matrix(factor_panels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardized = {
        name: normalize_cross_section(panel, method="zscore", axis=1)
        for name, panel in factor_panels.items()
    }
    stacked_frames: list[pd.DataFrame] = []
    for name, panel in standardized.items():
        try:
            stacked = panel.stack(future_stack=True).rename(name)
        except TypeError:  # pragma: no cover - older pandas fallback
            stacked = panel.stack(dropna=False).rename(name)
        stacked_frames.append(stacked)
    matrix = pd.concat(stacked_frames, axis=1).dropna(how="any")
    if matrix.empty:
        raise ValueError("PCA matrix is empty after alignment and standardization.")
    return matrix


def _run_pca(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = matrix.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    # SVD is numerically stable and avoids depending on sklearn.
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    explained_variance = (s**2) / max(len(x) - 1, 1)
    total_variance = float(explained_variance.sum())
    if total_variance <= 1e-12:
        explained_ratio = np.zeros_like(explained_variance)
    else:
        explained_ratio = explained_variance / total_variance

    pc_scores = u * s
    pc_columns = [f"PC{i + 1}" for i in range(pc_scores.shape[1])]
    pc_frame = pd.DataFrame(pc_scores, index=matrix.index, columns=pc_columns)
    loadings = pd.DataFrame(vt.T, index=matrix.columns, columns=pc_columns)
    return pc_frame, explained_ratio, loadings


def _summarize_explained_variance(explained_ratio: np.ndarray, thresholds: list[float]) -> dict[str, int | float | None]:
    cumulative = np.cumsum(explained_ratio)
    summary: dict[str, int | float | None] = {}
    for threshold in thresholds:
        hit = np.where(cumulative >= threshold)[0]
        summary[f"pcs_to_reach_{int(threshold * 100)}pct"] = int(hit[0] + 1) if len(hit) else None
    summary["total_components"] = int(len(explained_ratio))
    return summary


def main() -> None:
    args = parse_args()
    factor_root = Path(args.factor_root)
    factor_names = [str(name).strip() for name in args.factor_names if str(name).strip()]
    if not factor_names:
        raise RuntimeError("No factor names were provided.")

    store = FactorStore(root_dir=factor_root)
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
    pc_frame, explained_ratio, loadings = _run_pca(matrix)
    cumulative = np.cumsum(explained_ratio)
    summary = _summarize_explained_variance(explained_ratio, list(args.thresholds))

    print("Alpha101 PCA demo")
    print("Candidate factors:")
    print(factor_names)
    print(f"Aligned PCA samples: {len(matrix):,}")
    print("\nExplained variance ratio by component:")
    for i, ratio in enumerate(explained_ratio, start=1):
        print(f"  PC{i}: {ratio:.6f}")

    print("\nCumulative explained variance:")
    for i, ratio in enumerate(cumulative, start=1):
        print(f"  PC{i}: {ratio:.6f}")

    print("\nPrincipal components needed to reach thresholds:")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    print("\nTop loadings for the first 3 principal components:")
    for component in loadings.columns[:3]:
        top = loadings[component].abs().sort_values(ascending=False).head(5).index.tolist()
        print(f"  {component}: {top}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(output_dir / "alpha101_pca_matrix.parquet")
        pc_frame.to_parquet(output_dir / "alpha101_pca_scores.parquet")
        loadings.to_parquet(output_dir / "alpha101_pca_loadings.parquet")
        pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(len(explained_ratio))],
                "explained_variance_ratio": explained_ratio,
                "cumulative_explained_variance": cumulative,
            }
        ).to_parquet(output_dir / "alpha101_pca_explained_variance.parquet", index=False)
        (output_dir / "alpha101_pca_summary.json").write_text(
            json.dumps(
                {
                    "factor_names": factor_names,
                    "samples": int(len(matrix)),
                    "summary": summary,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved PCA outputs to: {output_dir}")


if __name__ == "__main__":
    main()
