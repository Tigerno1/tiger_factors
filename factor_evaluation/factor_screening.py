from __future__ import annotations

import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_evaluation.performance import factor_returns
from tiger_factors.factor_evaluation.performance import mean_return_by_quantile
from tiger_factors.factor_evaluation.utils import get_clean_factor_and_forward_returns as tiger_clean_factor_and_forward_returns
from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics


FACTOR_FILE_PATTERN = re.compile(r"^\d+_.+\.parquet$")


@dataclass(frozen=True)
class FactorScreeningRunResult:
    bundle_dir: str
    evaluation_path: str
    screened_path: str
    factor_count: int
    screened_count: int


def _normalize_return_series(series: pd.Series, *, factor_name: str) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return cleaned
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()].sort_index()
    cleaned.name = factor_name
    return cleaned


def _top_quantile_series_from_mean_returns(
    factor_data: pd.DataFrame,
    *,
    selected_period: str | int | pd.Timedelta,
) -> pd.Series | None:
    mean_ret, _ = mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
    )
    period_label = period_to_label(selected_period)
    if period_label not in mean_ret.columns:
        return None
    frame = mean_ret[period_label].unstack("factor_quantile").sort_index()
    if frame.empty:
        return None
    quantile_cols = pd.Index(frame.columns)
    if quantile_cols.empty:
        return None
    top_quantile = pd.to_numeric(quantile_cols, errors="coerce").max()
    if pd.isna(top_quantile):
        top_quantile = quantile_cols[-1]
    series = pd.to_numeric(frame[top_quantile], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()].sort_index()
    series.name = "long_only"
    return series


def _resolve_return_period(preferred_period: str | int | pd.Timedelta | None) -> str | int | pd.Timedelta:
    if preferred_period is not None:
        return preferred_period
    return 1


def _config_value(config: object | None, name: str, default: object) -> object:
    if config is None:
        return default
    return getattr(config, name, default)


def build_single_factor_return_long_frame(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factor_name: str | None = None,
    return_modes: tuple[str, ...] = ("long_short", "long_only"),
    preferred_return_period: str | int | pd.Timedelta | None = None,
    long_short_config: object | None = None,
) -> pd.DataFrame:
    """Convert one factor into ``date_ / factor / return / return_mode`` rows."""
    name = str(factor_name).strip() if factor_name is not None else getattr(factor, "name", None) or "factor"
    selected_period = _resolve_return_period(preferred_return_period)
    modes = tuple(str(mode).strip().lower() for mode in return_modes if str(mode).strip())
    if not modes:
        return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])

    factor_series = coerce_factor_series(factor)
    prepared = tiger_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=prices,
        quantiles=int(_config_value(long_short_config, "quantiles", 5)),
        periods=(selected_period,),
        filter_zscore=_config_value(long_short_config, "filter_zscore", 20),
        max_loss=_config_value(long_short_config, "max_loss", 0.35),
        cumulative_returns=bool(_config_value(long_short_config, "cumulative_returns", True)),
    )
    factor_data = prepared.factor_data

    frames: list[pd.DataFrame] = []
    if "long_short" in modes:
        series = factor_returns(
            factor_data,
            demeaned=bool(_config_value(long_short_config, "long_short", True)),
            group_adjust=bool(_config_value(long_short_config, "group_neutral", False)),
            equal_weight=bool(_config_value(long_short_config, "equal_weight", False)),
        )
        period_label = period_to_label(selected_period)
        if period_label in series.columns:
            long_short_series = _normalize_return_series(series[period_label], factor_name=name)
            if not long_short_series.empty:
                frames.append(
                    pd.DataFrame(
                        {
                            "date_": pd.to_datetime(pd.Index(long_short_series.index), errors="coerce").to_numpy(),
                            "factor": name,
                            "return": pd.to_numeric(long_short_series, errors="coerce").to_numpy(),
                            "return_mode": "long_short",
                        }
                    ).dropna(subset=["date_", "return"])[["date_", "factor", "return", "return_mode"]]
                )

    if "long_only" in modes:
        long_only = _top_quantile_series_from_mean_returns(factor_data, selected_period=selected_period)
        if long_only is not None and not long_only.empty:
            frames.append(
                pd.DataFrame(
                    {
                        "date_": pd.to_datetime(pd.Index(long_only.index), errors="coerce").to_numpy(),
                        "factor": name,
                        "return": pd.to_numeric(long_only, errors="coerce").to_numpy(),
                        "return_mode": "long_only",
                    }
                ).dropna(subset=["date_", "return"])[["date_", "factor", "return", "return_mode"]]
            )

    if not frames:
        return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["date_", "factor", "return"])
        .sort_values(["date_", "factor", "return_mode"], kind="stable")
        .reset_index(drop=True)
    )


def _discover_factor_files(root: Path) -> list[Path]:
    factor_files = [path for path in root.rglob("*.parquet") if FACTOR_FILE_PATTERN.match(path.name)]
    return sorted(factor_files)


def _group_factor_files_by_bundle(root: Path) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for factor_path in _discover_factor_files(root):
        grouped.setdefault(factor_path.parent, []).append(factor_path)
    return grouped


def _strip_factor_prefix(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        return stem
    prefix, remainder = stem.split("_", 1)
    return remainder if prefix.isdigit() else stem


def _bundle_output_paths(
    *,
    input_root: Path,
    bundle_dir: Path,
    output_root: Path | None,
) -> tuple[Path, Path, Path]:
    if output_root is None:
        target_dir = bundle_dir
    else:
        try:
            relative = bundle_dir.relative_to(input_root)
        except ValueError:
            relative = Path(bundle_dir.name)
        target_dir = output_root / relative
    target_dir.mkdir(parents=True, exist_ok=True)
    return (
        target_dir / "single_factor_evaluation.parquet",
        target_dir / "single_factor_screened.parquet",
        target_dir / "single_factor_screening.json",
    )


def _combined_output_paths(output_root: Path) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    return (
        output_root / "single_factor_evaluation_registry.parquet",
        output_root / "single_factor_screened_registry.parquet",
        output_root / "single_factor_screening_manifest.json",
    )


def evaluate_and_screen_factor_root(
    *,
    input_root: str | Path,
    output_root: str | Path | None = None,
    price_provider: str = "simfin",
    screening_config: FactorMetricFilterConfig | None = None,
    library: TigerFactorLibrary | None = None,
    as_ex: bool | None = None,
) -> pd.DataFrame:
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Factor root not found: {root}")

    library = library or TigerFactorLibrary(output_dir=root, price_provider=price_provider, verbose=False)
    cfg = screening_config or FactorMetricFilterConfig(
        min_fitness=0.10,
        min_ic_mean=0.01,
        min_rank_ic_mean=0.01,
        min_sharpe=0.40,
        max_turnover=0.50,
        min_decay_score=0.20,
        min_capacity_score=0.20,
        max_correlation_penalty=0.60,
        min_regime_robustness=0.60,
        min_out_of_sample_stability=0.60,
        sort_field="fitness",
        tie_breaker_field="ic_ir",
    )

    grouped = _group_factor_files_by_bundle(root)
    registry_rows: list[dict[str, object]] = []

    for bundle_dir, factor_paths in sorted(grouped.items(), key=lambda item: str(item[0])):
        if not factor_paths:
            continue

        first_factor = pd.read_parquet(factor_paths[0])
        if first_factor.empty:
            continue
        if "date_" not in first_factor.columns or "code" not in first_factor.columns:
            continue

        codes = sorted(first_factor["code"].dropna().astype(str).unique().tolist())
        date_min = pd.to_datetime(first_factor["date_"]).min()
        date_max = pd.to_datetime(first_factor["date_"]).max()
        price_start = str((pd.Timestamp(date_min) - pd.Timedelta(days=20)).date())
        price_end = str((pd.Timestamp(date_max) + pd.Timedelta(days=20)).date())
        price_frame = library.fetch_price_data(
            codes=codes,
            start=price_start,
            end=price_end,
            provider=price_provider,
            as_ex=as_ex,
        )

        bundle_rows: list[dict[str, object]] = []
        for factor_path in factor_paths:
            factor_df = pd.read_parquet(factor_path)
            factor_columns = [column for column in factor_df.columns if column not in {"date_", "code"}]
            if not factor_columns:
                continue
            factor_column = factor_columns[0]
            factor_name = _strip_factor_prefix(factor_path)

            engine = FactorEvaluationEngine(
                factor_frame=factor_df,
                price_frame=price_frame,
                factor_column=factor_column,
            )
            evaluation = engine.evaluate()
            row = {
                "bundle_dir": str(bundle_dir),
                "factor_name": factor_name,
                "factor_column": factor_column,
                "factor_path": str(factor_path),
                "factor_rows": int(len(factor_df)),
                "factor_codes": int(factor_df["code"].nunique()) if "code" in factor_df.columns else 0,
                "factor_date_min": str(pd.to_datetime(factor_df["date_"]).min()),
                "factor_date_max": str(pd.to_datetime(factor_df["date_"]).max()),
                **asdict(evaluation),
            }
            row["ic_mean_abs"] = abs(float(row.get("ic_mean", 0.0)))
            row["rank_ic_mean_abs"] = abs(float(row.get("rank_ic_mean", 0.0)))
            row["direction_hint"] = "reverse_factor" if float(row.get("ic_mean", 0.0)) < 0 else "use_as_is"
            row["directional_fitness"] = abs(float(row.get("fitness", 0.0)))
            row["directional_sharpe"] = abs(float(row.get("sharpe", 0.0)))
            row["directional_ic_ir"] = abs(float(row.get("ic_ir", 0.0)))
            bundle_rows.append(row)
            registry_rows.append(row)

        if not bundle_rows:
            continue

        bundle_registry = pd.DataFrame(bundle_rows).sort_values(
            ["directional_fitness", "ic_mean_abs", "factor_name"], ascending=[False, False, True], na_position="last"
        ).reset_index(drop=True)
        screened = screen_factor_metrics(bundle_registry, config=cfg)

        evaluation_path, screened_path, manifest_path = _bundle_output_paths(
            input_root=root,
            bundle_dir=bundle_dir,
            output_root=Path(output_root) if output_root is not None else None,
        )
        bundle_registry.to_parquet(evaluation_path, index=False)
        screened.to_parquet(screened_path, index=False)
        manifest_payload = {
            "bundle_dir": str(bundle_dir),
            "price_provider": price_provider,
            "factor_count": int(len(bundle_registry)),
            "screened_count": int(len(screened)),
            "evaluation_path": str(evaluation_path),
            "screened_path": str(screened_path),
            "screening_config": asdict(cfg),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    registry = pd.DataFrame(registry_rows)
    if output_root is not None:
        output_base = Path(output_root)
        registry_path, screened_registry_path, manifest_path = _combined_output_paths(output_base)
        registry.to_parquet(registry_path, index=False)
        screened_registry = screen_factor_metrics(registry, config=cfg) if not registry.empty else registry
        screened_registry.to_parquet(screened_registry_path, index=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "input_root": str(root),
                    "output_root": str(output_base),
                    "factor_count": int(len(registry)),
                    "screened_count": int(len(screened_registry)),
                    "screening_config": asdict(cfg),
                    "registry_path": str(registry_path),
                    "screened_registry_path": str(screened_registry_path),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    return registry


__all__ = [
    "FACTOR_FILE_PATTERN",
    "FactorScreeningRunResult",
    "build_single_factor_return_long_frame",
    "evaluate_and_screen_factor_root",
]
