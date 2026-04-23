from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store.migrate import migrate_factor_bundle_manifest


def test_migrate_factor_bundle_manifest_renames_and_splits(tmp_path: Path) -> None:
    source_dir = tmp_path / "src" / "ttm" / "base" / "ttm_bundle"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_parquet = source_dir / "001_roe__raw.parquet"
    pd.DataFrame(
        {
            "date_": ["2024-01-31", "2024-02-29"],
            "code": ["AAPL", "MSFT"],
            "roe__raw": [1.0, 2.0],
        }
    ).to_parquet(source_parquet, index=False)
    manifest = {
        "statement_freq": "1y",
        "variant_name": "base",
        "files": [
            {
                "index": 1,
                "factor": "roe__raw",
                "parquet_path": str(source_parquet),
                "rows": 2,
            }
        ],
    }
    manifest_path = source_dir / "ttm_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = FactorStore(tmp_path / "store")
    results = migrate_factor_bundle_manifest(manifest_path, family="financial", store=store)

    assert len(results) == 1
    assert results[0].dataset_dir.as_posix().endswith("factor/tiger/us/stock/1y")
    assert results[0].files[0].name == "financial__base__roe__2024-01.parquet"

    spec = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1y",
        table_name="financial__base__roe",
        provider="tiger",
    )
    loaded = store.get_factor(spec)

    assert list(loaded.columns) == ["code", "date_", "value"]
    assert loaded["value"].tolist() == [1.0, 2.0]
