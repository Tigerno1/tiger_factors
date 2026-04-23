from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.pipeline import (
    apply_weight_bounds,
    blend_factor_panels,
    factor_correlation_matrix,
    greedy_select_by_correlation,
    run_factor_backtest,
    score_to_weights,
)
from tiger_factors.factor_screener import screen_factor_metrics
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_maker.vectorization.indicators import Alpha101IndicatorTransformer


DEFAULT_OUTPUT_ROOT = Path("/Volumes/Quant_Disk/evaluation/alpha101_smoothed_screening")


@dataclass(frozen=True)
class Alpha101SmoothedScreeningResult:
    codes: list[str]
    factor_frame: pd.DataFrame
    smoothed_factor_frame: pd.DataFrame
    evaluation_registry: pd.DataFrame
    screened_registry: pd.DataFrame
    selected_factors: list[str]
    factor_weights: dict[str, float]
    combined_factor: pd.DataFrame
    lagged_combined_factor: pd.DataFrame
    backtest_returns: pd.DataFrame
    backtest_stats: dict[str, dict[str, float]]
    output_root: str
    manifest_path: str
    backtest_manifest_path: str


def _resolve_alpha_ids(alpha_ids: Sequence[int] | str | None) -> list[int]:
    if alpha_ids is None:
        return list(range(1, 102))
    if isinstance(alpha_ids, str):
        if alpha_ids.strip().lower() == "all":
            return list(range(1, 102))
        raise ValueError("alpha_ids must be 'all' or a sequence of integers.")
    return [int(alpha_id) for alpha_id in alpha_ids]


def _smooth_group_series(
    series: pd.Series,
    *,
    method: str,
    window: int,
    min_periods: int | None,
    ewm_span: int,
) -> pd.Series:
    if method == "rolling_mean":
        return series.rolling(window, min_periods=min_periods).mean()
    if method == "rolling_median":
        return series.rolling(window, min_periods=min_periods).median()
    if method == "ema":
        return series.ewm(span=ewm_span, adjust=False, min_periods=min_periods).mean()
    raise ValueError(f"Unsupported smoothing method: {method}")


def smooth_alpha101_factor_frame(
    factor_frame: pd.DataFrame,
    *,
    method: str = "rolling_mean",
    window: int = 5,
    min_periods: int | None = None,
    ewm_span: int = 10,
) -> pd.DataFrame:
    if factor_frame.empty:
        return factor_frame.copy()
    frame = factor_frame.copy().sort_values(["code", "date_"], kind="stable").reset_index(drop=True)
    factor_columns = [column for column in frame.columns if column not in {"date_", "code"}]
    if not factor_columns:
        return frame

    grouped = frame.groupby("code", sort=False)
    smoothed_parts: list[pd.DataFrame] = []
    for factor_column in factor_columns:
        smoothed = grouped[factor_column].transform(
            lambda series: _smooth_group_series(
                series,
                method=method,
                window=window,
                min_periods=min_periods,
                ewm_span=ewm_span,
            )
        )
        smoothed_parts.append(smoothed.rename(factor_column))

    smoothed_frame = frame[["date_", "code"]].copy()
    for part in smoothed_parts:
        smoothed_frame[part.name] = pd.to_numeric(part, errors="coerce")
    return smoothed_frame


def _default_screening_config() -> FactorMetricFilterConfig:
    return FactorMetricFilterConfig(
        min_fitness=0.10,
        min_ic_mean=0.01,
        min_rank_ic_mean=0.01,
        min_sharpe=0.40,
        max_turnover=0.40,
        min_decay_score=0.20,
        min_capacity_score=0.20,
        max_correlation_penalty=0.60,
        min_regime_robustness=0.60,
        min_out_of_sample_stability=0.60,
        sort_field="fitness",
        tie_breaker_field="ic_ir",
    )


def _default_backtest_params() -> dict[str, object]:
    return {
        "top_n": 20,
        "corr_threshold": 0.75,
        "weight_method": "positive",
        "weight_temperature": 1.0,
        "min_factor_weight": None,
        "max_factor_weight": None,
        "standardize": True,
        "long_pct": 0.20,
        "long_short": True,
        "rebalance_freq": "ME",
        "annual_trading_days": 252,
        "transaction_cost_bps": 8.0,
        "slippage_bps": 4.0,
    }


class Alpha101SmoothedScreeningEngine:
    """Compute smoothed Alpha101 factors and screen them without saving single-factor files."""

    def __init__(
        self,
        *,
        library: TigerFactorLibrary | None = None,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        calendar: str = "XNYS",
        start: str,
        end: str,
        alpha_ids: Sequence[int] | str | None = None,
        smoothing_method: str = "rolling_mean",
        smoothing_window: int = 5,
        smoothing_min_periods: int | None = None,
        ewm_span: int = 10,
        screening_config: FactorMetricFilterConfig | None = None,
        backtest_top_n: int = 20,
        backtest_corr_threshold: float = 0.75,
        backtest_weight_method: str = "positive",
        backtest_weight_temperature: float = 1.0,
        backtest_min_factor_weight: float | None = None,
        backtest_max_factor_weight: float | None = None,
        backtest_standardize: bool = True,
        backtest_long_pct: float = 0.20,
        backtest_long_short: bool = True,
        backtest_rebalance_freq: str = "ME",
        backtest_annual_trading_days: int = 252,
        backtest_transaction_cost_bps: float = 8.0,
        backtest_slippage_bps: float = 4.0,
        price_provider: str = "simfin",
        classification_provider: str = "simfin",
        classification_company_name: str = "companies",
        classification_industry_name: str = "industry",
        universe_provider: str = "github",
        universe_name: str = "sp500_constituents",
        region: str = "us",
        sec_type: str = "stock",
    ) -> None:
        self.library = library or TigerFactorLibrary(output_dir=output_root, price_provider=price_provider, verbose=False)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.transformer = Alpha101IndicatorTransformer(
            calendar=calendar,
            start=start,
            end=end,
            universe_provider=universe_provider,
            universe_name=universe_name,
            price_provider=price_provider,
            classification_provider=classification_provider,
            classification_company_name=classification_company_name,
            classification_industry_name=classification_industry_name,
            region=region,
            sec_type=sec_type,
        )
        self.alpha_ids = _resolve_alpha_ids(alpha_ids)
        self.smoothing_method = smoothing_method
        self.smoothing_window = int(smoothing_window)
        self.smoothing_min_periods = smoothing_min_periods
        self.ewm_span = int(ewm_span)
        self.screening_config = screening_config or _default_screening_config()
        self.backtest_top_n = int(backtest_top_n)
        self.backtest_corr_threshold = float(backtest_corr_threshold)
        self.backtest_weight_method = backtest_weight_method
        self.backtest_weight_temperature = float(backtest_weight_temperature)
        self.backtest_min_factor_weight = backtest_min_factor_weight
        self.backtest_max_factor_weight = backtest_max_factor_weight
        self.backtest_standardize = bool(backtest_standardize)
        self.backtest_long_pct = float(backtest_long_pct)
        self.backtest_long_short = bool(backtest_long_short)
        self.backtest_rebalance_freq = backtest_rebalance_freq
        self.backtest_annual_trading_days = int(backtest_annual_trading_days)
        self.backtest_transaction_cost_bps = float(backtest_transaction_cost_bps)
        self.backtest_slippage_bps = float(backtest_slippage_bps)

    def _build_factor_panels(
        self,
        factor_frame: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> dict[str, pd.DataFrame]:
        if factor_frame.empty:
            return {}
        frame = factor_frame.copy()
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_", "code"])

        panels: dict[str, pd.DataFrame] = {}
        for factor_name in factor_names:
            if factor_name not in frame.columns:
                continue
            subset = frame[["date_", "code", factor_name]].dropna(subset=[factor_name])
            if subset.empty:
                continue
            wide = subset.pivot_table(index="date_", columns="code", values=factor_name, aggfunc="last").sort_index()
            if not wide.empty:
                panels[factor_name] = wide
        return panels

    def run(
        self,
        *,
        codes: Sequence[str] | None = None,
        dividends: bool = False,
        history: bool = False,
        save_adjusted_price: bool = False,
        compute_workers: int | None = 1,
        save_workers: int | None = 1,
    ) -> Alpha101SmoothedScreeningResult:
        result = self.transformer.compute_all_alpha101_parallel(
            alpha_ids=self.alpha_ids,
            codes=codes,
            dividends=dividends,
            history=history,
            output_dir=self.output_root,
            save_factors=False,
            save_adj_price=save_adjusted_price,
            compute_workers=compute_workers,
            save_workers=save_workers,
        )
        factor_frame = result.factor_frame.copy()
        smoothed_frame = smooth_alpha101_factor_frame(
            factor_frame,
            method=self.smoothing_method,
            window=self.smoothing_window,
            min_periods=self.smoothing_min_periods,
            ewm_span=self.ewm_span,
        )
        price_frame = result.adjusted_frame[["date_", "code", "close"]].copy()

        registry_rows: list[dict[str, object]] = []
        factor_columns = [column for column in smoothed_frame.columns if column not in {"date_", "code"}]
        for factor_column in factor_columns:
            factor_df = smoothed_frame[["date_", "code", factor_column]].dropna(subset=[factor_column]).reset_index(drop=True)
            if factor_df.empty:
                continue
            evaluation = FactorEvaluationEngine(
                factor_frame=factor_df,
                price_frame=price_frame,
                factor_column=factor_column,
            ).evaluate()
            row = {
                "factor_name": factor_column,
                "factor_column": factor_column,
                "factor_rows": int(len(factor_df)),
                "factor_codes": int(factor_df["code"].nunique()) if "code" in factor_df.columns else 0,
                "factor_date_min": str(pd.to_datetime(factor_df["date_"]).min()),
                "factor_date_max": str(pd.to_datetime(factor_df["date_"]).max()),
                "smoothing_method": self.smoothing_method,
                "smoothing_window": int(self.smoothing_window),
                "smoothing_min_periods": self.smoothing_min_periods,
                "ewm_span": int(self.ewm_span),
                **asdict(evaluation),
            }
            registry_rows.append(row)

        evaluation_registry = pd.DataFrame(registry_rows)
        screened_registry = (
            screen_factor_metrics(evaluation_registry, config=self.screening_config)
            if not evaluation_registry.empty
            else evaluation_registry
        )

        selected_factors: list[str] = []
        factor_weights: dict[str, float] = {}
        combined_factor = pd.DataFrame()
        lagged_combined_factor = pd.DataFrame()
        backtest_returns = pd.DataFrame()
        backtest_stats: dict[str, dict[str, float]] = {
            "portfolio": {},
            "benchmark": {},
        }

        if not screened_registry.empty and "factor_name" in screened_registry.columns:
            screened_ordered = screened_registry.copy()
            if "directional_fitness" in screened_ordered.columns:
                screened_ordered["directional_fitness"] = pd.to_numeric(
                    screened_ordered["directional_fitness"], errors="coerce"
                )
            if "fitness" in screened_ordered.columns:
                screened_ordered["fitness"] = pd.to_numeric(screened_ordered["fitness"], errors="coerce")
            if "ic_ir" in screened_ordered.columns:
                screened_ordered["ic_ir"] = pd.to_numeric(screened_ordered["ic_ir"], errors="coerce")
            sort_columns = [column for column in ["directional_fitness", "ic_ir"] if column in screened_ordered.columns]
            if sort_columns:
                screened_ordered = screened_ordered.sort_values(sort_columns, ascending=[False] * len(sort_columns))
            candidate_names = (
                screened_ordered["factor_name"].dropna().astype(str).tolist()[: max(self.backtest_top_n, 1)]
            )
            factor_panels = self._build_factor_panels(smoothed_frame, candidate_names)
            if factor_panels:
                close_panel = coerce_price_panel(price_frame)
                common_index = None
                common_columns = None
                for panel in factor_panels.values():
                    common_index = panel.index if common_index is None else common_index.intersection(panel.index)
                    common_columns = panel.columns if common_columns is None else common_columns.intersection(panel.columns)
                if common_index is not None and common_columns is not None and len(common_index) > 0 and len(common_columns) > 0:
                    aligned_panels = {
                        name: panel.reindex(index=common_index, columns=common_columns)
                        for name, panel in factor_panels.items()
                    }
                    direction_map = {}
                    if "direction_hint" in screened_ordered.columns:
                        for _, row in screened_ordered.iterrows():
                            direction_map[str(row["factor_name"])] = -1.0 if row.get("direction_hint") == "reverse_factor" else 1.0
                    else:
                        for _, row in screened_ordered.iterrows():
                            direction_map[str(row["factor_name"])] = -1.0 if float(row.get("ic_mean", 0.0)) < 0 else 1.0
                    aligned_panels = {
                        name: (-panel if direction_map.get(name, 1.0) < 0 else panel)
                        for name, panel in aligned_panels.items()
                    }
                    aligned_close = close_panel.reindex(index=common_index, columns=common_columns).ffill()
                    score_field = "directional_fitness" if "directional_fitness" in screened_ordered.columns else "fitness"
                    if score_field not in screened_ordered.columns:
                        score_field = "ic_ir"
                    scores = {
                        name: float(
                            screened_ordered.loc[screened_ordered["factor_name"] == name, score_field].iloc[0]
                        )
                        for name in aligned_panels.keys()
                        if name in set(screened_ordered["factor_name"].astype(str).tolist())
                    }
                    if scores:
                        correlation_matrix = factor_correlation_matrix(
                            aligned_panels,
                            standardize=self.backtest_standardize,
                        )
                        selected_factors = greedy_select_by_correlation(
                            scores,
                            correlation_matrix,
                            self.backtest_corr_threshold,
                        )
                        if not selected_factors:
                            selected_factors = list(aligned_panels.keys())[:1]
                        selected_scores = {name: scores[name] for name in selected_factors if name in scores}
                        factor_weights = score_to_weights(
                            selected_scores,
                            selected=selected_factors,
                            method=self.backtest_weight_method,
                            temperature=self.backtest_weight_temperature,
                        )
                        factor_weights = apply_weight_bounds(
                            factor_weights,
                            min_weight=self.backtest_min_factor_weight,
                            max_weight=self.backtest_max_factor_weight,
                            total=1.0,
                        )
                        selected_panels = {name: aligned_panels[name] for name in selected_factors if name in aligned_panels}
                        if selected_panels and factor_weights:
                            combined_factor = blend_factor_panels(
                                selected_panels,
                                factor_weights,
                                standardize=self.backtest_standardize,
                            )
                            lagged_combined_factor = combined_factor.shift(1)
                            backtest_returns, backtest_stats = run_factor_backtest(
                                lagged_combined_factor,
                                aligned_close,
                                long_pct=self.backtest_long_pct,
                                rebalance_freq=self.backtest_rebalance_freq,
                                long_short=self.backtest_long_short,
                                annual_trading_days=self.backtest_annual_trading_days,
                                transaction_cost_bps=self.backtest_transaction_cost_bps,
                                slippage_bps=self.backtest_slippage_bps,
                            )

        registry_path = self.output_root / "alpha101_smoothed_evaluation_registry.parquet"
        screened_path = self.output_root / "alpha101_smoothed_screened_registry.parquet"
        manifest_path = self.output_root / "alpha101_smoothed_screening_manifest.json"
        backtest_dir = self.output_root / "backtest"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        evaluation_registry.to_parquet(registry_path, index=False)
        screened_registry.to_parquet(screened_path, index=False)
        if not combined_factor.empty:
            combined_factor.to_parquet(backtest_dir / "alpha101_smoothed_combined_factor.parquet")
        if not lagged_combined_factor.empty:
            lagged_combined_factor.to_parquet(backtest_dir / "alpha101_smoothed_combined_factor_lagged.parquet")
        if not backtest_returns.empty:
            backtest_returns.to_parquet(backtest_dir / "alpha101_smoothed_backtest_returns.parquet")
        if backtest_stats:
            pd.DataFrame(backtest_stats).T.to_parquet(backtest_dir / "alpha101_smoothed_backtest_stats.parquet")
        if selected_factors:
            pd.Series(selected_factors, name="factor").to_csv(
                backtest_dir / "alpha101_smoothed_selected_factors.csv",
                index=False,
            )
        if factor_weights:
            pd.Series(factor_weights, name="weight").to_csv(
                backtest_dir / "alpha101_smoothed_factor_weights.csv",
                index=False,
            )
        manifest_payload = {
            "output_root": str(self.output_root),
            "codes": int(len(result.codes)),
            "alpha_ids": self.alpha_ids,
            "factor_count": int(len(evaluation_registry)),
            "screened_count": int(len(screened_registry)),
            "selected_count": int(len(selected_factors)),
            "smoothing_method": self.smoothing_method,
            "smoothing_window": int(self.smoothing_window),
            "smoothing_min_periods": self.smoothing_min_periods,
            "ewm_span": int(self.ewm_span),
            "screening_config": asdict(self.screening_config),
            "registry_path": str(registry_path),
            "screened_path": str(screened_path),
            "backtest_dir": str(backtest_dir),
            "selected_factors": selected_factors,
            "factor_weights": factor_weights,
            "backtest_stats": backtest_stats,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        backtest_manifest_path = backtest_dir / "alpha101_smoothed_backtest_manifest.json"
        backtest_manifest_payload = {
            "output_root": str(self.output_root),
            "backtest_dir": str(backtest_dir),
            "selected_factors": selected_factors,
            "factor_weights": factor_weights,
            "backtest_stats": backtest_stats,
            "backtest_returns_path": str(backtest_dir / "alpha101_smoothed_backtest_returns.parquet"),
            "backtest_stats_path": str(backtest_dir / "alpha101_smoothed_backtest_stats.parquet"),
            "combined_factor_path": str(backtest_dir / "alpha101_smoothed_combined_factor.parquet"),
            "lagged_combined_factor_path": str(backtest_dir / "alpha101_smoothed_combined_factor_lagged.parquet"),
        }
        backtest_manifest_path.write_text(
            json.dumps(backtest_manifest_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        return Alpha101SmoothedScreeningResult(
            codes=list(result.codes),
            factor_frame=factor_frame,
            smoothed_factor_frame=smoothed_frame,
            evaluation_registry=evaluation_registry,
            screened_registry=screened_registry,
            selected_factors=selected_factors,
            factor_weights=factor_weights,
            combined_factor=combined_factor,
            lagged_combined_factor=lagged_combined_factor,
            backtest_returns=backtest_returns,
            backtest_stats=backtest_stats,
            output_root=str(self.output_root),
            manifest_path=str(manifest_path),
            backtest_manifest_path=str(backtest_manifest_path),
        )


__all__ = [
    "Alpha101SmoothedScreeningEngine",
    "Alpha101SmoothedScreeningResult",
    "smooth_alpha101_factor_frame",
]
