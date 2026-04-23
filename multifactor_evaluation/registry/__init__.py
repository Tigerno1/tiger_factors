from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from tiger_factors.multifactor_evaluation.screening import factor_metric_frame
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig


@dataclass(frozen=True)
class FactorRegistryConfig:
    summary_relative_path: str = "summary/evaluation.parquet"
    factor_name_column: str = "factor_name"
    strategy_name_column: str = "strategy_name"
    source_path_column: str = "source_path"


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported registry source file type: {path.suffix!r}")


def _coerce_strategy_name(source: Path, *, explicit: str | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if source.is_dir():
        return source.name
    if source.parent.name == "summary" and source.parent.parent.name:
        return source.parent.parent.name
    return source.stem


def _load_strategy_summary_frame(
    source: Path | pd.DataFrame | Mapping[str, Any],
    *,
    config: FactorRegistryConfig,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = factor_metric_frame(source)
        resolved_strategy_name = strategy_name
        if resolved_strategy_name is None and config.strategy_name_column in frame.columns:
            existing = frame[config.strategy_name_column].dropna()
            if not existing.empty:
                resolved_strategy_name = str(existing.iloc[0])
        if resolved_strategy_name is None:
            resolved_strategy_name = "strategy"
        frame[config.strategy_name_column] = resolved_strategy_name
        frame[config.source_path_column] = None
        frame[config.factor_name_column] = resolved_strategy_name
        return frame

    if isinstance(source, Mapping):
        frame = pd.DataFrame([dict(source)])
        resolved_strategy_name = strategy_name or str(frame.get(config.strategy_name_column, pd.Series(dtype=object)).dropna().iloc[0]) if config.strategy_name_column in frame.columns and not frame[config.strategy_name_column].dropna().empty else strategy_name
        if resolved_strategy_name is None or not str(resolved_strategy_name).strip():
            resolved_strategy_name = "strategy"
        frame[config.strategy_name_column] = resolved_strategy_name
        frame[config.source_path_column] = None
        if config.factor_name_column not in frame.columns:
            frame[config.factor_name_column] = resolved_strategy_name
        else:
            frame[config.factor_name_column] = frame[config.factor_name_column].fillna(resolved_strategy_name)
        return frame

    path = Path(source)
    if path.is_dir():
        path = path / config.summary_relative_path
    if not path.exists():
        raise FileNotFoundError(f"Strategy summary not found: {path}")

    frame = factor_metric_frame(_load_table(path))
    resolved_strategy_name = _coerce_strategy_name(path, explicit=strategy_name)
    frame[config.strategy_name_column] = resolved_strategy_name
    frame[config.source_path_column] = str(path)
    if config.factor_name_column not in frame.columns:
        frame[config.factor_name_column] = resolved_strategy_name
    else:
        frame[config.factor_name_column] = frame[config.factor_name_column].fillna(resolved_strategy_name)
    return frame


def build_factor_registry(
    sources: Mapping[str, Path | pd.DataFrame | Mapping[str, Any]]
    | Iterable[Path | pd.DataFrame | Mapping[str, Any]],
    *,
    config: FactorRegistryConfig | None = None,
) -> pd.DataFrame:
    cfg = config if config is not None else FactorRegistryConfig()
    frames: list[pd.DataFrame] = []

    if isinstance(sources, Mapping):
        items = sources.items()
    else:
        items = (
            (_coerce_strategy_name(Path(source)) if isinstance(source, Path) else f"strategy_{idx + 1}", source)
            for idx, source in enumerate(sources)
        )

    for explicit_name, source in items:
        frame = _load_strategy_summary_frame(source, config=cfg, strategy_name=explicit_name)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    registry = pd.concat(frames, ignore_index=True)
    if cfg.strategy_name_column in registry.columns:
        registry[cfg.strategy_name_column] = registry[cfg.strategy_name_column].astype(str)
    return registry


def build_factor_registry_from_root(
    root: str | Path,
    *,
    config: FactorRegistryConfig | None = None,
    summary_relative_path: str | None = None,
) -> pd.DataFrame:
    cfg = config if config is not None else FactorRegistryConfig()
    if summary_relative_path is not None:
        cfg = FactorRegistryConfig(
            summary_relative_path=summary_relative_path,
            factor_name_column=cfg.factor_name_column,
            strategy_name_column=cfg.strategy_name_column,
            source_path_column=cfg.source_path_column,
        )

    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Registry root not found: {base}")

    summary_paths = sorted(base.glob(f"*/{cfg.summary_relative_path}"))
    if not summary_paths:
        return pd.DataFrame()
    return build_factor_registry(summary_paths, config=cfg)


def screen_factor_registry(
    registry: pd.DataFrame | Mapping[str, Any],
    *,
    config: FactorMetricFilterConfig | None = None,
) -> pd.DataFrame:
    return screen_factor_metrics(registry, config=config)


__all__ = [
    "FactorRegistryConfig",
    "build_factor_registry",
    "build_factor_registry_from_root",
    "screen_factor_registry",
]
