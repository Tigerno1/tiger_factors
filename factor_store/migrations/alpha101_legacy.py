from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import AdjPriceSpec
from .. import FactorStore
from .. import FactorSpec


LEGACY_FACTOR_DIR_PATTERN = re.compile(r"^alpha_\d{3}$")
LEGACY_ADJ_PRICE_GLOB = "alpha101_adjusted_*.parquet"


@dataclass(frozen=True, slots=True)
class Alpha101LegacyMigrationResult:
    source_root: Path
    destination_root: Path
    migrated_factors: list[str]
    migrated_adj_price: str | None
    removed_sources: list[str]
    manifest_path: Path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _migrate_factor_dir(
    *,
    store: FactorStore,
    legacy_factor_dir: Path,
    delete_source: bool,
) -> tuple[str, str, str]:
    factor_name = legacy_factor_dir.name
    parquet_path = legacy_factor_dir / f"{factor_name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    frame = pd.read_parquet(parquet_path)
    if "value" not in frame.columns and factor_name in frame.columns:
        frame = frame.rename(columns={factor_name: "value"})
    metadata = _load_json(legacy_factor_dir / f"{factor_name}.json") or {}
    metadata = {
        **metadata,
        "source_root": str(legacy_factor_dir.parent),
        "legacy_parquet_path": str(parquet_path),
        "legacy_metadata_path": str(legacy_factor_dir / f"{factor_name}.json"),
        "family": "alpha101",
        "migrated_from": "tiger_factors.factor_store",
    }
    spec = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name=factor_name,
        variant=None,
        provider="tiger",
    )
    result = store.save_factor(spec, frame, force_updated=True, metadata=metadata)

    if delete_source:
        shutil.rmtree(legacy_factor_dir)

    return factor_name, str(result.files[0]), str(result.manifest_path)


def _migrate_adj_price(
    *,
    store: FactorStore,
    legacy_adj_dir: Path,
    delete_source: bool,
) -> str | None:
    parquet_files = sorted(legacy_adj_dir.glob(LEGACY_ADJ_PRICE_GLOB))
    if not parquet_files:
        return None

    frames = [pd.read_parquet(path) for path in parquet_files]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "close" in frame.columns:
        frame = frame.copy()
        frame["adj_close"] = pd.to_numeric(frame["close"], errors="coerce")
    metadata = {
        "source_root": str(legacy_adj_dir),
        "legacy_files": [str(path) for path in parquet_files],
        "family": "alpha101",
        "migrated_from": "tiger_factors.factor_maker.vectorization.indicators.alpha101_indicator",
    }
    spec = AdjPriceSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        variant="fwd",
        provider="tiger",
    )
    result = store.save_adj_price(spec, frame, force_updated=True, metadata=metadata)

    if delete_source:
        for path in parquet_files:
            path.unlink(missing_ok=True)
        metadata_path = legacy_adj_dir / "manifest.json"
        if metadata_path.exists():
            metadata_path.unlink()
        try:
            legacy_adj_dir.rmdir()
        except OSError:
            pass

    return str(result.files[0])


def migrate_alpha101_legacy_factors(
    *,
    source_root: str | Path = "/Volumes/Quant_Disk/factor",
    destination_root: str | Path = "/Volumes/Quant_Disk/factor",
    delete_source: bool = True,
    migrate_adj_price: bool = True,
) -> Alpha101LegacyMigrationResult:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    store = FactorStore(destination_root)

    migrated_factors: list[str] = []
    removed_sources: list[str] = []

    for legacy_factor_dir in sorted(source_root.iterdir() if source_root.exists() else []):
        if not legacy_factor_dir.is_dir():
            continue
        if not LEGACY_FACTOR_DIR_PATTERN.match(legacy_factor_dir.name):
            continue
        factor_name, _, _ = _migrate_factor_dir(
            store=store,
            legacy_factor_dir=legacy_factor_dir,
            delete_source=delete_source,
        )
        migrated_factors.append(factor_name)
        if delete_source:
            removed_sources.append(str(legacy_factor_dir))

    migrated_adj_price_path: str | None = None
    if migrate_adj_price:
        legacy_adj_dir = source_root / "adj_price"
        if legacy_adj_dir.exists():
            migrated_adj_price_path = _migrate_adj_price(
                store=store,
                legacy_adj_dir=legacy_adj_dir,
                delete_source=delete_source,
            )
            if delete_source and migrated_adj_price_path is not None:
                removed_sources.append(str(legacy_adj_dir))

    manifest_dir = destination_root / "migrations"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "alpha101_legacy_migration.json"
    manifest_payload = {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "migrated_factors": migrated_factors,
        "migrated_adj_price": migrated_adj_price_path,
        "delete_source": delete_source,
        "migrate_adj_price": migrate_adj_price,
        "removed_sources": removed_sources,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return Alpha101LegacyMigrationResult(
        source_root=source_root,
        destination_root=destination_root,
        migrated_factors=migrated_factors,
        migrated_adj_price=migrated_adj_price_path,
        removed_sources=removed_sources,
        manifest_path=manifest_path,
    )


def main() -> None:
    result = migrate_alpha101_legacy_factors()
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
