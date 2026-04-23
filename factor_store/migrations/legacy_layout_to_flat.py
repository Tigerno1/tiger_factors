from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


DEFAULT_FACTOR_SOURCE_ROOT = Path("/Volumes/Quant_Disk/factor/us/stock/1d")
DEFAULT_PRICE_SOURCE_ROOT = Path("/Volumes/Quant_Disk/tiger_quant/data/adj_price")
DEFAULT_STORE_ROOT = Path("/Volumes/Quant_Disk")


@dataclass(frozen=True, slots=True)
class LegacyMigrationSummary:
    migrated_factors: list[str]
    migrated_prices: list[str]
    removed_sources: list[str]
    report_path: Path


def _cleanup_empty_parents(path: Path, stop: Path) -> None:
    current = path
    while current != stop and current != current.parent:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _rename_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "date_" in frame.columns:
        return frame
    if "trading_day" in frame.columns:
        return frame.rename(columns={"trading_day": "date_"})
    if "date" in frame.columns:
        return frame.rename(columns={"date": "date_"})
    raise ValueError("price frame is missing a date column; expected date_, trading_day, or date")


def migrate_legacy_factor_tree(
    *,
    store: FactorStore,
    source_root: Path = DEFAULT_FACTOR_SOURCE_ROOT,
    delete_source: bool = True,
) -> list[str]:
    migrated: list[str] = []
    if not source_root.exists():
        return migrated

    for source_path in sorted(source_root.rglob("data.parquet")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source_root)
        if len(relative.parts) < 3:
            continue
        factor_name = relative.parts[-3]
        variant = relative.parts[-2]
        if variant == "raw":
            variant = None
        if not factor_name:
            continue
        frame = pd.read_parquet(source_path)
        if "value" not in frame.columns:
            candidate = next((column for column in frame.columns if column not in {"code", "date_"}), None)
            if candidate is None:
                raise ValueError(f"cannot infer factor value column for {source_path}")
            if candidate != "value":
                frame = frame.loc[:, [column for column in frame.columns if column in {"code", "date_", candidate}]].rename(
                    columns={candidate: "value"}
                )
        spec = FactorSpec(
            region="us",
            sec_type="stock",
            freq="1d",
            table_name=factor_name,
            variant=variant,
            provider="tiger",
        )
        store.save_factor(spec, frame, force_updated=True)
        migrated.append(str(source_path))

        if delete_source:
            manifest_path = source_path.parent / "manifest.json"
            source_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            _cleanup_empty_parents(source_path.parent, source_root)

    return migrated


def migrate_legacy_price_files(
    *,
    store: FactorStore,
    source_root: Path = DEFAULT_PRICE_SOURCE_ROOT,
    delete_source: bool = True,
) -> list[str]:
    migrated: list[str] = []
    if not source_root.exists():
        return migrated

    preferred = source_root / "alpha101_adjusted_20200601_20240601.parquet"
    if preferred.exists():
        source_path = preferred
        variant = None
        frame = pd.read_parquet(source_path)
        frame = _rename_date_column(frame)
        spec = AdjPriceSpec(
            region="us",
            sec_type="stock",
            freq="1d",
            variant=variant,
        )
        store.save_adj_price(spec, frame, force_updated=True)
        migrated.append(str(source_path))
        if delete_source:
            source_path.unlink(missing_ok=True)

    extra_candidates = [source_root / "alpha101_adjusted.parquet"]
    for source_path in extra_candidates:
        if source_path.exists() and delete_source:
            source_path.unlink(missing_ok=True)
            migrated.append(str(source_path))

    if not preferred.exists():
        fallback = source_root / "alpha101_adjusted.parquet"
        if fallback.exists():
            frame = pd.read_parquet(fallback)
            frame = _rename_date_column(frame)
            spec = AdjPriceSpec(
                region="us",
                sec_type="stock",
                freq="1d",
                variant=None,
            )
            store.save_adj_price(spec, frame, force_updated=True)
            migrated.append(str(fallback))
            if delete_source:
                fallback.unlink(missing_ok=True)

    if delete_source:
        try:
            source_root.rmdir()
        except OSError:
            pass

    return migrated


def run_legacy_layout_migration(
    *,
    store_root: Path = DEFAULT_STORE_ROOT,
    factor_source_root: Path = DEFAULT_FACTOR_SOURCE_ROOT,
    price_source_root: Path = DEFAULT_PRICE_SOURCE_ROOT,
    delete_source: bool = True,
) -> LegacyMigrationSummary:
    store = FactorStore(store_root)
    migrated_factors = migrate_legacy_factor_tree(
        store=store,
        source_root=factor_source_root,
        delete_source=delete_source,
    )
    migrated_prices = migrate_legacy_price_files(
        store=store,
        source_root=price_source_root,
        delete_source=delete_source,
    )
    removed_sources = migrated_factors + migrated_prices

    report = LegacyMigrationSummary(
        migrated_factors=migrated_factors,
        migrated_prices=migrated_prices,
        removed_sources=removed_sources,
        report_path=Path("/tmp/legacy_layout_to_flat.json"),
    )
    report.report_path.parent.mkdir(parents=True, exist_ok=True)
    report.report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main() -> None:
    summary = run_legacy_layout_migration()
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
