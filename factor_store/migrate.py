from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from .conf import DEFAULT_FACTOR_STORE_ROOT
from .spec import AdjPriceSpec
from .spec import FactorSpec
from .store import DatasetSaveResult
from .store import FactorStore


DEFAULT_SCATTERED_FACTOR_SOURCES: tuple[tuple[str, Path], ...] = (
    ("financial", Path("/Volumes/Quant_Disk/evaluation/financial_factors")),
    ("valuation", Path("/Volumes/Quant_Disk/factor/valuation")),
    ("valuation_quick", Path("/Volumes/Quant_Disk/factor/valuation_quick")),
)
DEFAULT_ADJ_PRICE_SOURCE = Path(
    "/Volumes/Quant_Disk/adj_price/us/stock/1d/front_adj/data.parquet"
)


@dataclass(frozen=True, slots=True)
class MigrationEntry:
    family: str
    source_manifest: str | None
    source_parquet: str
    target_kind: str
    target_region: str
    target_sec_type: str
    target_freq: str
    target_dataset_name: str
    target_variant: str
    target_dataset_dir: str
    rows: int


@dataclass(frozen=True, slots=True)
class MigrationReport:
    store_root: str
    entries: tuple[MigrationEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "store_root": self.store_root,
            "entries": [asdict(entry) for entry in self.entries],
        }


def _sanitize_piece(value: object) -> str:
    text = str(value).strip()
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(" ", "_")
    return text


def _canonical_factor_table_name(family: str, subfamily: str | None, factor_name: str) -> str:
    parts = [_sanitize_piece(family)]
    if subfamily:
        parts.append(_sanitize_piece(subfamily))
    parts.append(_sanitize_piece(factor_name))
    return "__".join(part for part in parts if part)


def _split_factor_name(factor_name: str) -> tuple[str, str]:
    name = Path(str(factor_name)).stem
    if "__" in name:
        base, variant = name.rsplit("__", 1)
    else:
        base, variant = name, "raw"
    return _sanitize_piece(base), _sanitize_piece(variant)


def _is_factor_bundle_manifest(payload: dict[str, object]) -> bool:
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return False
    first = files[0]
    return isinstance(first, dict) and "factor" in first and "parquet_path" in first


def _iter_factor_manifests(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*manifest.json")):
        if path.is_file():
            yield path


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_subfamily(manifest: dict[str, object], manifest_path: Path) -> str | None:
    for key in ("variant_name", "variant"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_piece(value)
    if len(manifest_path.parents) >= 3:
        return _sanitize_piece(manifest_path.parents[1].name)
    return None


def _save_report(report: MigrationReport, store_root: Path) -> Path:
    migrations_dir = store_root / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    report_path = migrations_dir / "scatter_to_factor_store_migration.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return report_path


def migrate_factor_bundle_manifest(
    manifest_path: str | Path,
    *,
    family: str,
    store: FactorStore | None = None,
    provider: str = "tiger",
    region: str = "us",
    sec_type: str = "stock",
) -> list[DatasetSaveResult]:
    store = store or FactorStore(DEFAULT_FACTOR_STORE_ROOT)
    manifest_path = Path(manifest_path)
    payload = _load_manifest(manifest_path)
    if not _is_factor_bundle_manifest(payload):
        return []

    freq = _sanitize_piece(payload.get("statement_freq", ""))
    if not freq:
        raise ValueError(f"manifest missing statement_freq: {manifest_path}")
    subfamily = _infer_subfamily(payload, manifest_path)

    results: list[DatasetSaveResult] = []
    for item in payload["files"]:
        if not isinstance(item, dict):
            continue
        factor_name = str(item["factor"])
        source_parquet = Path(str(item["parquet_path"]))
        base_name, variant = _split_factor_name(factor_name)
        table_name = _canonical_factor_table_name(family, subfamily, base_name)

        frame = pd.read_parquet(source_parquet)
        value_column = None
        if "value" in frame.columns:
            value_column = "value"
        elif factor_name in frame.columns:
            value_column = factor_name
        elif base_name in frame.columns:
            value_column = base_name
        else:
            meta_columns = {"code", "date_", "date", "ticker", "simfin_id"}
            candidates = [column for column in frame.columns if column not in meta_columns]
            if len(candidates) == 1:
                value_column = candidates[0]
        if value_column is None:
            raise ValueError(f"cannot infer value column for {source_parquet}")
        if value_column != "value":
            frame = frame.loc[:, ["code", "date_", value_column]].rename(columns={value_column: "value"})
        else:
            frame = frame.loc[:, ["code", "date_", "value"]]

        spec = FactorSpec(
            region=region,
            sec_type=sec_type,
            freq=freq,
            table_name=table_name,
            variant=variant,
            provider=provider,
        )
        results.append(store.save_factor(spec, frame, force_updated=True))
    return results


def migrate_adj_price_file(
    source_path: str | Path,
    *,
    store: FactorStore | None = None,
    region: str = "us",
    sec_type: str = "stock",
    freq: str = "1d",
    variant: str = "fwd",
) -> DatasetSaveResult:
    store = store or FactorStore(DEFAULT_FACTOR_STORE_ROOT)
    source_path = Path(source_path)
    frame = pd.read_parquet(source_path)
    if "code" not in frame.columns or "date_" not in frame.columns:
        raise ValueError(f"adj price file must contain code/date_: {source_path}")
    columns = tuple(column for column in frame.columns if column not in {"code", "date_"})
    spec = AdjPriceSpec(
        region=region,
        sec_type=sec_type,
        freq=freq,
        variant=variant,
        columns=columns,
    )
    return store.save_adj_price(spec, frame, force_updated=True)


def consolidate_default_scattered_factors(
    *,
    store_root: str | Path = DEFAULT_FACTOR_STORE_ROOT,
    include_valuation_quick: bool = True,
    include_adj_price: bool = True,
) -> MigrationReport:
    store = FactorStore(store_root)
    entries: list[MigrationEntry] = []

    for family, root in DEFAULT_SCATTERED_FACTOR_SOURCES:
        if family == "valuation_quick" and not include_valuation_quick:
            continue
        if not root.exists():
            continue
        for manifest_path in _iter_factor_manifests(root):
            payload = _load_manifest(manifest_path)
            if not _is_factor_bundle_manifest(payload):
                continue
            results = migrate_factor_bundle_manifest(
                manifest_path,
                family=family,
                store=store,
            )
            for result, item in zip(results, payload["files"], strict=False):
                if not isinstance(item, dict):
                    continue
                factor_name = str(item["factor"])
                base_name, variant = _split_factor_name(factor_name)
                subfamily = _infer_subfamily(payload, manifest_path)
                entries.append(
                    MigrationEntry(
                        family=family,
                        source_manifest=str(manifest_path),
                        source_parquet=str(item["parquet_path"]),
                        target_kind="factor",
                        target_region="us",
                        target_sec_type="stock",
                        target_freq=_sanitize_piece(payload.get("statement_freq", "")),
                        target_dataset_name=_canonical_factor_table_name(family, subfamily, base_name),
                        target_variant=variant,
                        target_dataset_dir=str(result.dataset_dir),
                        rows=int(result.rows),
                    )
                )

    if include_adj_price and DEFAULT_ADJ_PRICE_SOURCE.exists():
        result = migrate_adj_price_file(DEFAULT_ADJ_PRICE_SOURCE, store=store)
        entries.append(
            MigrationEntry(
                family="adj_price",
                source_manifest=None,
                source_parquet=str(DEFAULT_ADJ_PRICE_SOURCE),
                target_kind="price",
                target_region="us",
                target_sec_type="stock",
                target_freq="1d",
                target_dataset_name="adj_price",
                target_variant="fwd",
                target_dataset_dir=str(result.dataset_dir),
                rows=int(result.rows),
            )
        )

    report = MigrationReport(store_root=str(store.root_dir), entries=tuple(entries))
    _save_report(report, Path(store.root_dir))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate scattered factor parquet files into tiger_factors.factor_store.")
    parser.add_argument(
        "--store-root",
        default=DEFAULT_FACTOR_STORE_ROOT,
        type=Path,
    )
    parser.add_argument("--skip-adj-price", action="store_true")
    parser.add_argument("--skip-valuation-quick", action="store_true")
    args = parser.parse_args(argv)

    report = consolidate_default_scattered_factors(
        store_root=args.store_root,
        include_valuation_quick=not args.skip_valuation_quick,
        include_adj_price=not args.skip_adj_price,
    )
    print(f"migrated {len(report.entries)} datasets into {report.store_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
