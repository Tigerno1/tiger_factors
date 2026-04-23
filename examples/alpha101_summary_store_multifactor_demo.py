"""Build an Alpha101 multifactor portfolio from stored factors and stored summaries.

This demo assumes you already have:

1. Alpha101 factor panels saved in the local factor store under Quant Disk.
2. Single-factor evaluation summaries already persisted under the evaluation root
   (for example ``/Volumes/Quant_Disk/evaluation/summary/summary_registry.parquet``).

The flow is intentionally simple:

1. load the saved summary registry
2. keep the Alpha101 rows (or an explicit candidate subset)
3. screen factors using the stored single-factor metrics
4. load the matching factor panels from the factor store
5. apply a simple low-correlation selection pass
6. weight and blend the selected factors
7. run the multifactor backtest
8. generate the full multifactor tear sheets
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

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation import MultifactorEvaluation
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import factor_correlation_matrix
from tiger_factors.factor_screener import screen_factor_registry
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel


DEFAULT_STORE_ROOT = Path("/Volumes/Quant_Disk/factor_store")
DEFAULT_SUMMARY_REGISTRY = Path("/Volumes/Quant_Disk/tiger_quant/data/tmp/summary/summary_registry.parquet")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "alpha101_summary_store_multifactor_demo"
DEFAULT_PROVIDER = "tiger"
DEFAULT_REGION = "us"
DEFAULT_SEC_TYPE = "stock"
DEFAULT_FREQ = "1d"
DEFAULT_PRICE_PROVIDER = "tiger"
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_TOP_N_INITIAL = 20
DEFAULT_CORR_THRESHOLD = 0.65
DEFAULT_LONG_PCT = 0.20
DEFAULT_WEIGHT_METHOD = "softmax"
DEFAULT_WEIGHT_TEMPERATURE = 1.0
DEFAULT_TRANSACTION_COST_BPS = 8.0
DEFAULT_SLIPPAGE_BPS = 4.0
DEFAULT_REPORT_NAME = "alpha101_summary_store_multifactor"

STORE_ROOT = DEFAULT_STORE_ROOT
SUMMARY_REGISTRY = DEFAULT_SUMMARY_REGISTRY
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PROVIDER = DEFAULT_PROVIDER
REGION = DEFAULT_REGION
SEC_TYPE = DEFAULT_SEC_TYPE
FREQ = DEFAULT_FREQ
PRICE_PROVIDER = DEFAULT_PRICE_PROVIDER
VARIANT = None
FACTOR_PREFIX = "alpha_"
FACTOR_NAMES: list[str] | None = None
START = None
END = None
TOP_N_INITIAL = DEFAULT_TOP_N_INITIAL
CORR_THRESHOLD = DEFAULT_CORR_THRESHOLD
LONG_PCT = DEFAULT_LONG_PCT
REBALANCE_FREQ = DEFAULT_REBALANCE_FREQ
WEIGHT_METHOD = DEFAULT_WEIGHT_METHOD
WEIGHT_TEMPERATURE = DEFAULT_WEIGHT_TEMPERATURE
SCORE_FIELD = "fitness"
MIN_FITNESS = 0.0
MIN_IC_MEAN = 0.0
MIN_RANK_IC_MEAN = 0.0
MIN_SHARPE = 0.0
MAX_TURNOVER = 0.70
TRANSACTION_COST_BPS = DEFAULT_TRANSACTION_COST_BPS
SLIPPAGE_BPS = DEFAULT_SLIPPAGE_BPS
OPEN_BROWSER = False


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def _load_summary_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary registry not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported summary registry format: {path.suffix}")


def _select_alpha_rows(registry: pd.DataFrame, *, factor_prefix: str, factor_names: list[str] | None) -> pd.DataFrame:
    frame = registry.copy()
    if "factor_name" not in frame.columns:
        raise ValueError("Summary registry must contain a factor_name column.")
    frame["factor_name"] = frame["factor_name"].astype(str)
    if factor_names:
        wanted = {str(name) for name in factor_names}
        frame = frame.loc[frame["factor_name"].isin(wanted)]
    else:
        frame = frame.loc[frame["factor_name"].str.startswith(str(factor_prefix))]
    if frame.empty:
        raise ValueError("No matching Alpha101 summary rows were found in the registry.")
    return frame.reset_index(drop=True)


def _resolve_direction(row: pd.Series) -> float:
    direction_hint = row.get("direction_hint")
    if isinstance(direction_hint, str):
        if direction_hint == "reverse_factor":
            return -1.0
        if direction_hint == "use_as_is":
            return 1.0
    ic_mean = pd.to_numeric(pd.Series([row.get("ic_mean")]), errors="coerce").iloc[0]
    if pd.notna(ic_mean) and float(ic_mean) < 0:
        return -1.0
    return 1.0


def _score_series(frame: pd.DataFrame, score_field: str) -> pd.Series:
    factor_index = frame["factor_name"].astype(str) if "factor_name" in frame.columns else frame.index.astype(str)
    if score_field == "fitness":
        if "directional_fitness" in frame.columns:
            values = pd.to_numeric(frame["directional_fitness"], errors="coerce")
        else:
            values = pd.to_numeric(frame["fitness"], errors="coerce").abs()
        return pd.Series(values.to_numpy(), index=factor_index, dtype=float)
    if score_field == "ic_ir":
        if "directional_ic_ir" in frame.columns:
            values = pd.to_numeric(frame["directional_ic_ir"], errors="coerce")
        else:
            values = pd.to_numeric(frame["ic_ir"], errors="coerce").abs()
        return pd.Series(values.to_numpy(), index=factor_index, dtype=float)
    if score_field == "sharpe":
        if "directional_sharpe" in frame.columns:
            values = pd.to_numeric(frame["directional_sharpe"], errors="coerce")
        else:
            values = pd.to_numeric(frame["sharpe"], errors="coerce").abs()
        return pd.Series(values.to_numpy(), index=factor_index, dtype=float)
    raise ValueError(f"Unsupported score field: {score_field}")


def _apply_weight_method(scores: pd.Series, *, method: str, temperature: float) -> pd.Series:
    clean = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("No valid scores are available to compute factor weights.")
    if method == "equal":
        return pd.Series(1.0 / len(clean), index=clean.index, dtype=float)
    shifted = clean - float(clean.min())
    shifted = shifted.clip(lower=0.0)
    if method == "positive":
        positive = shifted + 1e-12
        return positive / positive.sum()
    if method == "softmax":
        scale = max(float(temperature), 1e-6)
        logits = clean / scale
        logits = logits - float(logits.max())
        weights = np.exp(logits)
        return pd.Series(weights / weights.sum(), index=clean.index, dtype=float)
    raise ValueError(f"Unsupported weight method: {method}")


def _greedy_low_corr_selection(
    ranked_factors: list[str],
    corr_matrix: pd.DataFrame,
    *,
    threshold: float,
) -> list[str]:
    selected: list[str] = []
    for factor_name in ranked_factors:
        if factor_name not in corr_matrix.index:
            selected.append(factor_name)
            continue
        too_close = False
        for chosen in selected:
            if chosen not in corr_matrix.columns:
                continue
            corr_value = corr_matrix.loc[factor_name, chosen]
            if pd.notna(corr_value) and abs(float(corr_value)) > threshold:
                too_close = True
                break
        if not too_close:
            selected.append(factor_name)
    return selected


def _infer_universe_and_dates(factor_panels: dict[str, pd.DataFrame], *, start: str | None, end: str | None) -> tuple[list[str], str, str]:
    all_codes: set[str] = set()
    date_bounds: list[pd.Timestamp] = []
    for panel in factor_panels.values():
        if panel.empty:
            continue
        all_codes.update(str(code) for code in panel.columns)
        date_bounds.append(pd.Timestamp(panel.index.min()))
        date_bounds.append(pd.Timestamp(panel.index.max()))
    if not all_codes or not date_bounds:
        raise ValueError("Could not infer universe or dates from the selected factor panels.")
    resolved_start = start or str(min(date_bounds).date())
    resolved_end = end or str(max(date_bounds).date())
    return sorted(all_codes), resolved_start, resolved_end


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


def _serializable_stats(stats: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(stats, default=str))


def main() -> None:
    store_root = Path(STORE_ROOT)
    summary_registry_path = Path(SUMMARY_REGISTRY)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # registry = _load_summary_registry(summary_registry_path)
    registry = pd.read_parquet(summary_registry_path)
    print(registry)
    alpha_registry = _select_alpha_rows(
        registry,
        factor_prefix=FACTOR_PREFIX,
        factor_names=FACTOR_NAMES,
    )

    screen_config = FactorMetricFilterConfig(
        min_fitness=MIN_FITNESS,
        min_ic_mean=MIN_IC_MEAN,
        min_rank_ic_mean=MIN_RANK_IC_MEAN,
        min_sharpe=MIN_SHARPE,
        max_turnover=MAX_TURNOVER,
        sort_field=SCORE_FIELD,
        tie_breaker_field="ic_ir",
    )
    screened = screen_factor_registry(alpha_registry, config=screen_config)
#     usable = screened.loc[screened["usable"]].copy()
#     if usable.empty:
#         raise RuntimeError("No Alpha101 factors passed the summary-based screening rules.")

#     candidate_frame = usable.head(max(int(TOP_N_INITIAL), 1)).copy()
#     candidate_names = candidate_frame["factor_name"].astype(str).tolist()

#     factor_store = FactorStore(root_dir=store_root)
#     library = TigerFactorLibrary(output_dir=store_root, price_provider=PRICE_PROVIDER, verbose=True)
#     factor_panels = library.load_factor_panels(
#         factor_names=candidate_names,
#         provider=PROVIDER,
#         freq=FREQ,
#         variant=_normalize_variant(VARIANT),
#         start=START,
#         end=END,
#     )
#     if not factor_panels:
#         raise RuntimeError("No factor panels could be loaded from the factor store for the screened Alpha101 factors.")

#     candidate_frame = candidate_frame.loc[candidate_frame["factor_name"].isin(factor_panels.keys())].copy()
#     if candidate_frame.empty:
#         raise RuntimeError("None of the screened factors could be loaded from the factor store.")

#     factor_directions = {
#         str(row["factor_name"]): _resolve_direction(row)
#         for _, row in candidate_frame.iterrows()
#     }
#     directed_panels = {
#         name: (-panel if float(factor_directions.get(name, 1.0)) < 0 else panel)
#         for name, panel in factor_panels.items()
#     }

#     corr_matrix = factor_correlation_matrix(directed_panels)
#     ranked_names = candidate_frame["factor_name"].astype(str).tolist()
#     selected_names = _greedy_low_corr_selection(ranked_names, corr_matrix, threshold=float(CORR_THRESHOLD))
#     if not selected_names:
#         raise RuntimeError("Correlation filtering removed every candidate factor.")

#     selected_frame = candidate_frame.set_index("factor_name").loc[selected_names].reset_index()
#     raw_scores = _score_series(selected_frame, SCORE_FIELD)
#     weights = _apply_weight_method(raw_scores, method=WEIGHT_METHOD, temperature=WEIGHT_TEMPERATURE)
#     weights = weights.reindex(selected_names).fillna(0.0)
#     weights = weights / weights.sum()

#     selected_panels = {name: directed_panels[name] for name in selected_names}
#     codes, start, end = _infer_universe_and_dates(selected_panels, start=START, end=END)
#     close_panel = _load_close_panel(
#         factor_store,
#         provider=PROVIDER,
#         region=REGION,
#         sec_type=SEC_TYPE,
#         freq=FREQ,
#     )
#     close_panel = close_panel.reindex(index=pd.DatetimeIndex(close_panel.index)).sort_index()
#     close_panel = close_panel.loc[(close_panel.index >= pd.Timestamp(start)) & (close_panel.index <= pd.Timestamp(end))]
#     close_panel = close_panel.reindex(columns=codes)
#     close_panel = close_panel.dropna(axis=1, how="all")
#     if close_panel.empty:
#         raise RuntimeError("Could not align a matching close panel from the local factor store.")

#     backtest_result = multi_factor_backtest(
#         selected_panels,
#         close_panel,
#         weights=weights.to_dict(),
#         standardize=True,
#         rebalance_freq=REBALANCE_FREQ,
#         long_pct=float(LONG_PCT),
#         long_short=True,
#         annual_trading_days=252,
#         transaction_cost_bps=float(TRANSACTION_COST_BPS),
#         slippage_bps=float(SLIPPAGE_BPS),
#     )

#     factor_data: dict[str, pd.DataFrame] = dict(selected_panels)
#     factor_data["combined_factor"] = backtest_result["composite_factor"]
#     backtest = backtest_result["backtest"]
#     evaluation = MultifactorEvaluation(
#         backtest=backtest,
#         positions_frame=backtest.attrs.get("positions"),
#         close_panel_frame=close_panel,
#         factor_data=factor_data,
#         output_dir=output_dir,
#         report_name=DEFAULT_REPORT_NAME,
#         capital_base=1_000_000.0,
#     )
#     bundle = evaluation.full(
#         backtest,
#         output_dir=output_dir,
#         report_name=DEFAULT_REPORT_NAME,
#         open_browser=bool(OPEN_BROWSER),
#     )

#     selected_summary = selected_frame.copy()
#     selected_summary["direction"] = selected_summary["factor_name"].map(factor_directions)
#     selected_summary["weight"] = selected_summary["factor_name"].map(weights.to_dict())
#     backtest_to_write = backtest.copy()
#     backtest_to_write.attrs = {}

#     screened.to_parquet(output_dir / "alpha101_summary_screened.parquet", index=False)
#     selected_summary.to_parquet(output_dir / "alpha101_multifactor_selected.parquet", index=False)
#     corr_matrix.to_parquet(output_dir / "alpha101_multifactor_correlation_matrix.parquet")
#     backtest_result["composite_factor"].to_parquet(output_dir / "alpha101_multifactor_combined_factor.parquet")
#     backtest_to_write.to_parquet(output_dir / "alpha101_multifactor_backtest.parquet")
#     pd.DataFrame(backtest_result["stats"]).T.to_parquet(output_dir / "alpha101_multifactor_backtest_stats.parquet")

#     manifest = {
#         "store_root": str(store_root),
#         "summary_registry": str(summary_registry_path),
#         "provider": PROVIDER,
#         "freq": FREQ,
#         "price_provider": PRICE_PROVIDER,
#         "variant": _normalize_variant(VARIANT),
#         "factor_prefix": FACTOR_PREFIX,
#         "candidate_count": int(len(alpha_registry)),
#         "screened_usable_count": int(len(usable)),
#         "candidate_names": candidate_names,
#         "selected_factors": selected_names,
#         "factor_directions": factor_directions,
#         "factor_weights": {name: float(weights.loc[name]) for name in selected_names},
#         "score_field": SCORE_FIELD,
#         "weight_method": WEIGHT_METHOD,
#         "weight_temperature": float(WEIGHT_TEMPERATURE),
#         "corr_threshold": float(CORR_THRESHOLD),
#         "start": start,
#         "end": end,
#         "long_pct": float(LONG_PCT),
#         "rebalance_freq": REBALANCE_FREQ,
#         "transaction_cost_bps": float(TRANSACTION_COST_BPS),
#         "slippage_bps": float(SLIPPAGE_BPS),
#         "output_dir": str(output_dir),
#         "report_path": None if bundle.report_path is None else str(bundle.report_path),
#         "portfolio_stats": _serializable_stats(backtest_result["stats"]["portfolio"]),
#         "benchmark_stats": _serializable_stats(backtest_result["stats"]["benchmark"]),
#     }
#     (output_dir / "alpha101_multifactor_manifest.json").write_text(
#         json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
#         encoding="utf-8",
#     )

#     print("Alpha101 summary-store multifactor workflow complete.")
#     print(f"Summary registry: {summary_registry_path}")
#     print(f"Output dir: {output_dir}")
#     if bundle.report_path is not None:
#         print(f"Combined report: {bundle.report_path}")
#     print(f"Selected factors: {selected_names}")
#     print(f"Weights: {json.dumps({name: float(weights.loc[name]) for name in selected_names}, indent=2)}")
#     print("\nPortfolio stats:")
#     print(json.dumps(_serializable_stats(backtest_result["stats"]["portfolio"]), indent=2))


if __name__ == "__main__":
    main()
#     main()
