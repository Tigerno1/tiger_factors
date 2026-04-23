from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


SOURCE_ROOT = Path("/Volumes/Quant_Disk")
DEST_ROOT = DEFAULT_FACTOR_STORE_ROOT


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _migrate_factor_tree(store: FactorStore) -> int:
    source_root = SOURCE_ROOT / "factor"
    if not source_root.exists():
        return 0

    migrated = 0
    for manifest_path in sorted(source_root.rglob("manifest.json")):
        if not manifest_path.is_file():
            continue
        payload = _load_json(manifest_path)
        spec_payload = payload.get("spec")
        if not isinstance(spec_payload, dict) or spec_payload.get("kind") != "factor":
            continue

        table_name = str(spec_payload.get("table_name"))
        if not table_name:
            continue
        variant = spec_payload.get("variant")
        spec = FactorSpec(
            provider="tiger",
            region=str(spec_payload.get("region", "us")),
            sec_type=str(spec_payload.get("sec_type", "stock")),
            freq=str(spec_payload.get("freq", "1d")),
            table_name=table_name,
            variant=str(variant) if variant is not None else None,
        )

        source_dir = manifest_path.parent
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            continue
        frames: list[pd.DataFrame] = []
        for filename in files:
            path = source_dir / str(filename)
            if not path.exists():
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True, sort=False)
        store.save_factor(spec, frame, force_updated=True)
        shutil.rmtree(source_dir)
        migrated += 1

    if source_root.exists():
        for path in sorted(source_root.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            source_root.rmdir()
        except OSError:
            pass
    return migrated


def _migrate_legacy_factor_dirs() -> int:
    source_base = SOURCE_ROOT / "factor" / "us" / "stock" / "1d"
    if not source_base.exists():
        return 0

    dest_dir = DEST_ROOT / "factor" / "tiger" / "us" / "stock" / "1d"
    dest_dir.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for source_dir in sorted(path for path in source_base.iterdir() if path.is_dir()):
        alpha_name = source_dir.name
        dest_manifest = dest_dir / f"manifest__{alpha_name}.json"
        if dest_manifest.exists():
            shutil.rmtree(source_dir)
            migrated += 1
            continue

        manifest = source_dir / "manifest.json"
        if manifest.exists():
            shutil.move(str(manifest), str(dest_manifest))

        for parquet_path in sorted(source_dir.glob("*.parquet")):
            target = dest_dir / parquet_path.name
            if target.exists():
                continue
            shutil.move(str(parquet_path), str(target))

        shutil.rmtree(source_dir, ignore_errors=True)
        migrated += 1

    for path in sorted(source_base.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        source_base.rmdir()
    except OSError:
        pass
    return migrated


def _migrate_providerless_factor_dirs() -> int:
    source_base = DEST_ROOT / "factor" / "us" / "stock" / "1d"
    if not source_base.exists():
        return 0

    dest_base = DEST_ROOT / "factor" / "tiger" / "us" / "stock" / "1d"
    dest_base.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for source_dir in sorted(path for path in source_base.iterdir() if path.is_dir()):
        target_dir = dest_base / source_dir.name
        if target_dir.exists():
            shutil.rmtree(source_dir)
            migrated += 1
            continue
        shutil.move(str(source_dir), str(target_dir))
        migrated += 1

    for path in sorted(source_base.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        source_base.rmdir()
    except OSError:
        pass
    return migrated


def _migrate_providerless_factor_summary_table() -> int:
    source_base = DEST_ROOT / "factor" / "us" / "stock" / "1d"
    if not source_base.exists():
        return 0

    dest_base = DEST_ROOT / "factor" / "tiger" / "us" / "stock" / "1d"
    dest_base.mkdir(parents=True, exist_ok=True)

    migrated = 0
    summary_file = source_base / "alpha101_summary_table.parquet"
    if summary_file.exists():
        target_file = dest_base / summary_file.name
        if not target_file.exists():
            shutil.move(str(summary_file), str(target_file))
        else:
            summary_file.unlink(missing_ok=True)
        migrated += 1

    manifest_file = source_base / "manifest.json"
    if manifest_file.exists():
        target_manifest = dest_base / "manifest__alpha101_summary_table.json"
        if not target_manifest.exists():
            shutil.move(str(manifest_file), str(target_manifest))
        else:
            manifest_file.unlink(missing_ok=True)
        migrated += 1

    for path in sorted(source_base.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        source_base.rmdir()
    except OSError:
        pass
    return migrated


def _migrate_providerless_evaluation_dirs() -> int:
    source_base = DEST_ROOT / "us" / "stock"
    if not source_base.exists():
        return 0

    dest_base = DEST_ROOT / "evaluation" / "tiger" / "us" / "stock" / "1d"
    dest_base.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for source_dir in sorted(path for path in source_base.iterdir() if path.is_dir()):
        target_dir = dest_base / source_dir.name
        if target_dir.exists():
            shutil.rmtree(source_dir)
        else:
            shutil.move(str(source_dir), str(target_dir))
        migrated += 1

    for path in sorted(source_base.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        source_base.rmdir()
    except OSError:
        pass
    for path in sorted(DEST_ROOT.glob("us/*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        (DEST_ROOT / "us").rmdir()
    except OSError:
        pass
    return migrated


def _move_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for child in sorted(source.iterdir()):
            target = destination / child.name
            if target.exists():
                if child.is_dir() and target.is_dir():
                    _move_tree(child, target)
                continue
            shutil.move(str(child), str(target))
        try:
            source.rmdir()
        except OSError:
            pass
        return
    shutil.move(str(source), str(destination))


def _merge_tree_contents(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    destination.mkdir(parents=True, exist_ok=True)

    moved = 0
    for file_path in sorted(source.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name == ".DS_Store":
            file_path.unlink(missing_ok=True)
            continue
        relative = file_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            file_path.unlink(missing_ok=True)
        else:
            shutil.move(str(file_path), str(target))
            moved += 1

    for path in sorted(source.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        source.rmdir()
    except OSError:
        pass
    return moved


def main() -> int:
    store = FactorStore(DEST_ROOT)

    factor_count = _migrate_factor_tree(store)
    factor_count += _migrate_legacy_factor_dirs()
    factor_count += _migrate_providerless_factor_dirs()
    factor_count += _migrate_providerless_factor_summary_table()
    factor_count += _migrate_providerless_evaluation_dirs()
    _merge_tree_contents(SOURCE_ROOT / "factor", DEST_ROOT / "factor")
    _merge_tree_contents(SOURCE_ROOT / "price", DEST_ROOT / "price")
    _move_tree(SOURCE_ROOT / "others", DEST_ROOT / "others")
    _move_tree(SOURCE_ROOT / "evaluation", DEST_ROOT / "evaluation")

    print(f"migrated {factor_count} factor bundles into {DEST_ROOT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
