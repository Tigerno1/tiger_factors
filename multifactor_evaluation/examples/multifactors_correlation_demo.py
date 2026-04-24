"""Build a multifactors correlation matrix and screen high/low correlated factors.

This example expects the batch Alpha101 summary registry to be available at:

    /Volumes/Quant_Disk/evaluation/summary/summary_registry.parquet

It screens the usable factors, loads their factor parquet files, computes a
correlation matrix, and writes:

- the full correlation matrix
- high-correlation factor pairs
- low/high average-correlation factor tables
- cluster representatives for redundant groups
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import build_factor_registry_from_root
from tiger_factors.factor_screener import cluster_factors
from tiger_factors.factor_screener import factor_correlation_matrix
from tiger_factors.factor_screener import screen_factor_registry


DEFAULT_SUMMARY_REGISTRY = Path("/Volumes/Quant_Disk/evaluation/summary/summary_registry.parquet")
DEFAULT_FACTOR_ROOT = Path("/Volumes/Quant_Disk/factor")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Quant_Disk/evaluation/multifactors_correlation")
DEFAULT_HIGH_PAIR_THRESHOLD = 0.60
DEFAULT_LOW_MEAN_ABS_THRESHOLD = 0.20
DEFAULT_HIGH_MEAN_ABS_THRESHOLD = 0.50


def _load_registry() -> pd.DataFrame:
    if DEFAULT_SUMMARY_REGISTRY.exists():
        return pd.read_parquet(DEFAULT_SUMMARY_REGISTRY)
    return build_factor_registry_from_root("/Volumes/Quant_Disk/evaluation/summary")


def _factor_path_from_name(factor_name: str) -> Path:
    return DEFAULT_FACTOR_ROOT / factor_name / f"{factor_name}.parquet"


def _load_factor_frame(factor_name: str) -> pd.DataFrame:
    path = _factor_path_from_name(factor_name)
    if not path.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")
    frame = pd.read_parquet(path)
    if {"date_", "code", factor_name}.issubset(frame.columns):
        return frame[["date_", "code", factor_name]].copy()
    if "factor" in frame.columns:
        renamed = frame[["date_", "code", "factor"]].copy()
        renamed = renamed.rename(columns={"factor": factor_name})
        return renamed
    value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
    if len(value_columns) != 1:
        raise ValueError(f"Unexpected factor schema in: {path}")
    return frame[["date_", "code", value_columns[0]]].rename(columns={value_columns[0]: factor_name})


def _pair_table(corr: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = float(corr.loc[left, right])
            records.append(
                {
                    "left": left,
                    "right": right,
                    "corr": value,
                    "abs_corr": abs(value),
                }
            )
    return pd.DataFrame(records).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _cluster_representatives(
    corr: pd.DataFrame,
    screened: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    cluster_map = cluster_factors(corr, threshold=threshold)
    if not cluster_map:
        return pd.DataFrame(columns=["cluster", "factor_name", "fitness", "ic_mean", "rank_ic_mean", "sharpe", "turnover"])

    usable = screened.copy()
    if "factor_name" in usable.columns:
        usable = usable.set_index("factor_name", drop=False)
    rows: list[dict[str, object]] = []
    for factor_name, cluster in cluster_map.items():
        if factor_name not in usable.index:
            continue
        rows.append(
            {
                "cluster": int(cluster),
                "factor_name": factor_name,
                "fitness": float(usable.loc[factor_name, "directional_fitness"])
                if "directional_fitness" in usable.columns
                else float(usable.loc[factor_name, "fitness"]),
                "ic_mean": float(usable.loc[factor_name, "ic_mean"]),
                "rank_ic_mean": float(usable.loc[factor_name, "rank_ic_mean"]),
                "sharpe": float(usable.loc[factor_name, "sharpe"]),
                "turnover": float(usable.loc[factor_name, "turnover"]),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["cluster", "fitness"], ascending=[True, False]).reset_index(drop=True)
    representatives = frame.groupby("cluster", as_index=False).head(1).reset_index(drop=True)
    return representatives


def main() -> None:
    registry = _load_registry()
    if registry.empty:
        raise RuntimeError("No registry rows found. Run the alpha101 summary batch first.")

    screen_config = FactorMetricFilterConfig(
        min_ic_mean=0.01,
        min_rank_ic_mean=0.01,
        min_sharpe=0.40,
        max_turnover=0.50,
        sort_field="fitness",
        tie_breaker_field="ic_ir",
    )
    screened = screen_factor_registry(registry, config=screen_config)
    usable = screened[screened["usable"].fillna(False)].copy()
    if usable.empty:
        raise RuntimeError("No usable factors survived screening.")

    factor_names = [str(name) for name in usable["factor_name"].tolist()]
    factor_dict = {name: _load_factor_frame(name) for name in factor_names}
    corr = factor_correlation_matrix(factor_dict)
    pair_table = _pair_table(corr)

    mean_abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool)).mean(axis=1).sort_values()
    corr_summary = pd.DataFrame(
        {
            "factor_name": mean_abs_corr.index,
            "mean_abs_corr": mean_abs_corr.values,
            "fitness": [
                float(usable.set_index("factor_name").loc[name, "directional_fitness"])
                if name in usable["factor_name"].values and "directional_fitness" in usable.columns
                else (float(usable.set_index("factor_name").loc[name, "fitness"]) if name in usable["factor_name"].values else np.nan)
                for name in mean_abs_corr.index
            ],
        }
    )
    corr_summary["high_corr"] = corr_summary["mean_abs_corr"] >= DEFAULT_HIGH_MEAN_ABS_THRESHOLD
    corr_summary["low_corr"] = corr_summary["mean_abs_corr"] <= DEFAULT_LOW_MEAN_ABS_THRESHOLD
    high_corr_factors = corr_summary[corr_summary["high_corr"]].sort_values("mean_abs_corr", ascending=False).reset_index(drop=True)
    low_corr_factors = corr_summary[corr_summary["low_corr"]].sort_values("mean_abs_corr", ascending=True).reset_index(drop=True)
    high_corr_pairs = pair_table[pair_table["abs_corr"] >= DEFAULT_HIGH_PAIR_THRESHOLD].reset_index(drop=True)
    cluster_representatives = _cluster_representatives(corr, usable, threshold=0.70)

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    corr.to_parquet(output_dir / "multifactors_factor_correlation_matrix.parquet")
    pair_table.to_parquet(output_dir / "multifactors_factor_correlation_pairs.parquet", index=False)
    high_corr_pairs.to_parquet(output_dir / "multifactors_high_corr_pairs.parquet", index=False)
    corr_summary.to_parquet(output_dir / "multifactors_factor_correlation_summary.parquet", index=False)
    high_corr_factors.to_parquet(output_dir / "multifactors_high_corr_factors.parquet", index=False)
    low_corr_factors.to_parquet(output_dir / "multifactors_low_corr_factors.parquet", index=False)
    cluster_representatives.to_parquet(output_dir / "multifactors_cluster_representatives.parquet", index=False)

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary_registry": str(DEFAULT_SUMMARY_REGISTRY),
        "registry_rows": int(len(registry)),
        "usable_rows": int(len(usable)),
        "pair_threshold": DEFAULT_HIGH_PAIR_THRESHOLD,
        "low_mean_abs_threshold": DEFAULT_LOW_MEAN_ABS_THRESHOLD,
        "high_mean_abs_threshold": DEFAULT_HIGH_MEAN_ABS_THRESHOLD,
        "output_dir": str(output_dir),
    }
    (output_dir / "multifactors_correlation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("usable factors:", len(usable))
    print("high-correlation pairs:", len(high_corr_pairs))
    print("high-correlation factors:", len(high_corr_factors))
    print("low-correlation factors:", len(low_corr_factors))
    print("cluster representatives:", len(cluster_representatives))
    print(f"saved correlation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
