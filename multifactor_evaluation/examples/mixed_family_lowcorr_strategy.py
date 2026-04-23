"""Select alpha101, financial, and valuation factors, then build a low-corr strategy.

This example:
- keeps only base financial/valuation bundles (drops bank/insurance)
- drops the top 5 candidates from each family by score, then keeps the next 15
- computes a cross-family correlation matrix
- greedily selects up to 10 factors with abs(corr) < threshold
- builds a combined factor and backtests it against adjusted prices

Outputs are written under the Quant Disk evaluation root.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.pipeline import blend_factor_panels
from tiger_factors.multifactor_evaluation.pipeline import greedy_select_by_correlation
from tiger_factors.multifactor_evaluation.pipeline import normalize_weights
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest


DEFAULT_ALPHA_REGISTRY = Path("/Volumes/Quant_Disk/evaluation/alpha101_smoothed_screening/alpha101_smoothed_evaluation_registry.parquet")
DEFAULT_FINANCIAL_REGISTRY = Path("/Volumes/Quant_Disk/evaluation/financial_factors/screening/single_factor_evaluation_registry.parquet")
DEFAULT_VALUATION_REGISTRY = Path("/Volumes/Quant_Disk/evaluation/valuation_top_bottom_combo/screening/single_factor_evaluation_registry.parquet")
DEFAULT_PRICE_PATH = Path("/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet")
DEFAULT_OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/mixed_family_lowcorr_strategy")

DEFAULT_TOP_N_PER_FAMILY = 15
DEFAULT_DROP_TOP_N_PER_FAMILY = 5
DEFAULT_SELECTED_N = 10
DEFAULT_CORR_THRESHOLD = 0.20
DEFAULT_FAMILY_QUOTAS = {"alpha101": 3, "financial": 3, "valuation": 3}
DEFAULT_MIN_FACTOR_ROWS = 1000
DEFAULT_MIN_FACTOR_CODES = 20
DEFAULT_MIN_SUPPORT_DAYS = 365
DEFAULT_LONG_PCT = 0.20
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_TRANSACTION_COST_BPS = 8.0
DEFAULT_SLIPPAGE_BPS = 4.0


@dataclass(frozen=True)
class Candidate:
    family: str
    candidate_name: str
    factor_name: str
    factor_path: Path
    score_raw: float
    selection_score: float
    direction: float
    factor_rows: int
    factor_codes: int
    factor_date_min: pd.Timestamp
    factor_date_max: pd.Timestamp


def _load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    return pd.read_parquet(path)


def _base_only_registry(registry: pd.DataFrame) -> pd.DataFrame:
    if "bundle_dir" not in registry.columns:
        return registry.copy()
    bundle_dir = registry["bundle_dir"].astype(str)
    mask = bundle_dir.str.contains(r"/base/", regex=True, na=False)
    filtered = registry.loc[mask].copy()
    if filtered.empty:
        raise RuntimeError("No base bundles found in registry.")
    return filtered


def _support_days(row: pd.Series) -> int:
    date_min = pd.to_datetime(row.get("factor_date_min"), errors="coerce")
    date_max = pd.to_datetime(row.get("factor_date_max"), errors="coerce")
    if pd.isna(date_min) or pd.isna(date_max):
        return 0
    return int((date_max - date_min).days) + 1


def _supports_enough_data(row: pd.Series) -> bool:
    rows = int(row.get("factor_rows", 0) or 0)
    codes = int(row.get("factor_codes", 0) or 0)
    days = _support_days(row)
    return rows >= DEFAULT_MIN_FACTOR_ROWS and codes >= DEFAULT_MIN_FACTOR_CODES and days >= DEFAULT_MIN_SUPPORT_DAYS


def _score_from_row(row: pd.Series, *, family: str) -> float:
    if "directional_fitness" in row and pd.notna(row.get("directional_fitness")):
        return float(row["directional_fitness"])
    if family == "alpha101":
        score = row.get("fitness", np.nan)
        if pd.isna(score):
            score = row.get("ic_mean", np.nan)
    else:
        score = row.get("ic_mean", np.nan)
        if pd.isna(score):
            score = row.get("rank_ic_mean", np.nan)
    if pd.isna(score):
        return float("nan")
    return float(score)


def _direction_from_row(row: pd.Series) -> float:
    direction_hint = row.get("direction_hint", None)
    if isinstance(direction_hint, str):
        if direction_hint == "reverse_factor":
            return -1.0
        if direction_hint == "use_as_is":
            return 1.0
    ic_mean = pd.to_numeric(row.get("ic_mean", np.nan), errors="coerce")
    if pd.notna(ic_mean) and float(ic_mean) < 0:
        return -1.0
    return 1.0


def _family_label(row: pd.Series, *, family: str) -> str:
    if family == "alpha101":
        return f"alpha101::{row['factor_name']}"
    bundle = Path(str(row["bundle_dir"])).name if "bundle_dir" in row and pd.notna(row["bundle_dir"]) else "bundle"
    return f"{family}::{bundle}::{row['factor_name']}"


def _build_candidates(
    registry: pd.DataFrame,
    *,
    family: str,
    top_n: int,
    drop_top_n: int,
) -> pd.DataFrame:
    usable = registry.copy()
    usable = usable[usable.apply(_supports_enough_data, axis=1)].copy()
    if usable.empty:
        raise RuntimeError(f"No supported factors found for family={family}")

    usable["score_raw"] = usable.apply(lambda row: _score_from_row(row, family=family), axis=1)
    usable = usable.replace([np.inf, -np.inf], np.nan).dropna(subset=["score_raw"]).copy()
    usable["selection_score"] = usable["score_raw"].abs()
    usable["direction"] = usable.apply(_direction_from_row, axis=1)
    usable["candidate_name"] = usable.apply(lambda row: _family_label(row, family=family), axis=1)
    usable = usable.sort_values("selection_score", ascending=False).reset_index(drop=True)
    drop_top_n = max(int(drop_top_n), 0)
    top_n = max(int(top_n), 0)
    usable = usable.iloc[drop_top_n : drop_top_n + top_n].copy()

    cols = [
        "candidate_name",
        "factor_name",
        "factor_path" if "factor_path" in usable.columns else None,
        "score_raw",
        "selection_score",
        "direction",
        "factor_rows",
        "factor_codes",
        "factor_date_min",
        "factor_date_max",
    ]
    cols = [col for col in cols if col is not None and col in usable.columns]
    usable["family"] = family
    return usable[["family"] + cols].reset_index(drop=True)


def _factor_path_from_alpha_name(name: str) -> Path:
    return Path("/Volumes/Quant_Disk/factor") / name / f"{name}.parquet"


def _load_factor_frame(path: Path, factor_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Factor parquet not found: {path}")
    frame = pd.read_parquet(path)
    if {"date_", "code", factor_column}.issubset(frame.columns):
        subset = frame[["date_", "code", factor_column]].copy()
    else:
        value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
        if len(value_columns) != 1:
            raise ValueError(f"Unexpected factor schema in {path}: {frame.columns.tolist()}")
        subset = frame[["date_", "code", value_columns[0]]].rename(columns={value_columns[0]: factor_column})
    return subset


def _load_wide_panel(path: Path, factor_column: str) -> pd.DataFrame:
    subset = _load_factor_frame(path, factor_column)
    series = coerce_factor_series(subset)
    wide = series.unstack("code").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    wide.index.name = "date_"
    return wide


def _pair_table(corr: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = float(corr.loc[left, right])
            records.append({"left": left, "right": right, "corr": value, "abs_corr": abs(value)})
    return pd.DataFrame(records).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _factor_correlation_matrix_fast(
    factors: dict[str, pd.DataFrame],
    *,
    standardize: bool = True,
) -> pd.DataFrame:
    stacked: dict[str, pd.Series] = {}
    for name, panel in factors.items():
        transformed = panel
        if standardize:
            transformed = transformed.copy()
            mean = transformed.mean(axis=1)
            std = transformed.std(axis=1, ddof=0).replace(0, np.nan)
            transformed = transformed.sub(mean, axis=0).div(std, axis=0)
        stacked[name] = transformed.stack(dropna=True)

    aligned = pd.concat(stacked, axis=1)
    corr = aligned.corr()
    np.fill_diagonal(corr.values, 1.0)
    corr.index.name = "factor_name"
    corr.columns.name = "factor_name"
    return corr


def _select_with_family_quotas(
    candidate_registry: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    quotas: dict[str, int],
    threshold: float,
) -> list[str]:
    ordered = candidate_registry.sort_values(
        ["family", "selection_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    selected: list[str] = []
    selected_by_family = {family: 0 for family in quotas}
    used: set[str] = set()
    family_order = list(quotas.keys())

    def can_add(name: str) -> bool:
        if not selected:
            return True
        pair = corr.loc[name, selected].abs()
        return bool((pair < threshold).all())

    progress = True
    while progress and len(selected) < sum(int(v) for v in quotas.values()):
        progress = False
        for family in family_order:
            quota = int(quotas.get(family, 0))
            if selected_by_family.get(family, 0) >= quota:
                continue
            family_rows = ordered[(ordered["family"] == family) & (~ordered["candidate_name"].isin(used))]
            for _, row in family_rows.iterrows():
                name = str(row["candidate_name"])
                if name in used:
                    continue
                if can_add(name):
                    selected.append(name)
                    used.add(name)
                    selected_by_family[family] = selected_by_family.get(family, 0) + 1
                    progress = True
                    break

    return selected


def _fill_spillover_selection(
    candidate_registry: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    selected_names: list[str],
    threshold: float,
    target_n: int,
) -> list[str]:
    if len(selected_names) >= target_n:
        return selected_names[:target_n]

    selected = list(selected_names)
    used = set(selected)
    ordered = candidate_registry.sort_values("selection_score", ascending=False).reset_index(drop=True)
    for _, row in ordered.iterrows():
        if len(selected) >= target_n:
            break
        name = str(row["candidate_name"])
        if name in used:
            continue
        if selected:
            pair = corr.loc[name, selected].abs()
            if not bool((pair < threshold).all()):
                continue
        selected.append(name)
        used.add(name)
    return selected[:target_n]


def _resolve_path(row: pd.Series, family: str) -> Path:
    if family == "alpha101":
        return _factor_path_from_alpha_name(str(row["factor_name"]))
    factor_path = row.get("factor_path")
    if pd.isna(factor_path) or not factor_path:
        raise ValueError(f"Missing factor_path for family={family}, factor={row['factor_name']}")
    return Path(str(factor_path))


def _materialize_panels(selected: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        family = str(row["family"])
        path = _resolve_path(row, family)
        factor_column = str(row["factor_name"])
        label = str(row["candidate_name"])
        direction = float(row["direction"])
        panel = _load_wide_panel(path, factor_column)
        if direction < 0:
            panel = -panel
        panels[label] = panel
        rows.append(
            {
                "family": family,
                "candidate_name": label,
                "factor_name": factor_column,
                "factor_path": str(path),
                "score_raw": float(row["score_raw"]),
                "selection_score": float(row["selection_score"]),
                "direction": direction,
                "factor_rows": int(row["factor_rows"]),
                "factor_codes": int(row["factor_codes"]),
                "factor_date_min": str(pd.to_datetime(row["factor_date_min"])),
                "factor_date_max": str(pd.to_datetime(row["factor_date_max"])),
            }
        )
    return panels, pd.DataFrame(rows)


def main() -> None:
    alpha_registry = _load_registry(DEFAULT_ALPHA_REGISTRY)
    financial_registry = _base_only_registry(_load_registry(DEFAULT_FINANCIAL_REGISTRY))
    valuation_registry = _base_only_registry(_load_registry(DEFAULT_VALUATION_REGISTRY))

    alpha_candidates = _build_candidates(
        alpha_registry,
        family="alpha101",
        top_n=DEFAULT_TOP_N_PER_FAMILY,
        drop_top_n=DEFAULT_DROP_TOP_N_PER_FAMILY,
    )
    financial_candidates = _build_candidates(
        financial_registry,
        family="financial",
        top_n=DEFAULT_TOP_N_PER_FAMILY,
        drop_top_n=DEFAULT_DROP_TOP_N_PER_FAMILY,
    )
    valuation_candidates = _build_candidates(
        valuation_registry,
        family="valuation",
        top_n=DEFAULT_TOP_N_PER_FAMILY,
        drop_top_n=DEFAULT_DROP_TOP_N_PER_FAMILY,
    )

    candidate_registry = pd.concat(
        [alpha_candidates, financial_candidates, valuation_candidates],
        ignore_index=True,
    )
    candidate_registry = candidate_registry.sort_values(
        ["family", "selection_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    selected_panels, selected_registry = _materialize_panels(candidate_registry)
    scores = {
        name: float(score)
        for name, score in zip(selected_registry["candidate_name"], selected_registry["selection_score"])
    }
    corr = _factor_correlation_matrix_fast(selected_panels, standardize=True)
    pair_table = _pair_table(corr)
    selected_names = _select_with_family_quotas(
        selected_registry,
        corr,
        quotas=DEFAULT_FAMILY_QUOTAS,
        threshold=DEFAULT_CORR_THRESHOLD,
    )
    selected_names = [name for name in selected_names if name in selected_panels][:DEFAULT_SELECTED_N]
    if len(selected_names) < DEFAULT_SELECTED_N:
        selected_names = _fill_spillover_selection(
            selected_registry,
            corr,
            selected_names=selected_names,
            threshold=DEFAULT_CORR_THRESHOLD,
            target_n=DEFAULT_SELECTED_N,
        )
    if len(selected_names) < DEFAULT_SELECTED_N:
        selected_names = _fill_spillover_selection(
            selected_registry,
            corr,
            selected_names=selected_names,
            threshold=0.30,
            target_n=DEFAULT_SELECTED_N,
        )
    if len(selected_names) < DEFAULT_SELECTED_N:
        raise RuntimeError(
            f"Could not fill target_n={DEFAULT_SELECTED_N}; only selected {len(selected_names)} factors "
            f"under corr threshold {DEFAULT_CORR_THRESHOLD}."
        )

    selected_panels = {name: selected_panels[name] for name in selected_names}
    selected_weights = normalize_weights({name: 1.0 for name in selected_names})
    combined_factor = blend_factor_panels(selected_panels, selected_weights, standardize=True)
    lagged_combined = combined_factor.shift(1)

    price_panel = coerce_price_panel(pd.read_parquet(DEFAULT_PRICE_PATH))
    backtest, stats = run_factor_backtest(
        lagged_combined,
        price_panel,
        long_pct=DEFAULT_LONG_PCT,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )

    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / pd.Timestamp.now(tz="UTC").strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    candidate_registry.to_parquet(run_dir / "top50_candidates.parquet", index=False)
    selected_registry.to_parquet(run_dir / "selected_candidates_before_corr.parquet", index=False)
    corr.to_parquet(run_dir / "candidate_correlation_matrix.parquet")
    pair_table.to_parquet(run_dir / "candidate_correlation_pairs.parquet", index=False)
    pd.DataFrame(
        [{"factor_name": name, "weight": float(weight)} for name, weight in selected_weights.items()]
    ).to_parquet(run_dir / "selected_factor_weights.parquet", index=False)
    pd.DataFrame({"selected_factor": selected_names}).to_parquet(run_dir / "selected_factors.parquet", index=False)
    combined_factor.to_parquet(run_dir / "combined_factor.parquet")
    lagged_combined.to_parquet(run_dir / "combined_factor_lagged.parquet")
    backtest.to_parquet(run_dir / "backtest_daily.parquet")
    pd.DataFrame(stats).T.to_parquet(run_dir / "backtest_stats.parquet")

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "alpha_registry": str(DEFAULT_ALPHA_REGISTRY),
        "financial_registry": str(DEFAULT_FINANCIAL_REGISTRY),
        "valuation_registry": str(DEFAULT_VALUATION_REGISTRY),
        "price_path": str(DEFAULT_PRICE_PATH),
        "top_n_per_family": DEFAULT_TOP_N_PER_FAMILY,
        "drop_top_n_per_family": DEFAULT_DROP_TOP_N_PER_FAMILY,
        "selected_n": DEFAULT_SELECTED_N,
        "corr_threshold": DEFAULT_CORR_THRESHOLD,
        "min_factor_rows": DEFAULT_MIN_FACTOR_ROWS,
        "min_factor_codes": DEFAULT_MIN_FACTOR_CODES,
        "min_support_days": DEFAULT_MIN_SUPPORT_DAYS,
        "selected_factor_count": int(len(selected_names)),
        "selected_factors": selected_names,
        "output_dir": str(run_dir),
        "backtest_stats": stats,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("mixed-family low-correlation strategy complete")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"saved outputs to: {run_dir}")
    print("\nselected factors:")
    for name in selected_names:
        print(" -", name)
    print("\nportfolio stats:")
    print(json.dumps(stats["portfolio"], indent=2, ensure_ascii=False, default=str))
    print("\nbenchmark stats:")
    print(json.dumps(stats["benchmark"], indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
