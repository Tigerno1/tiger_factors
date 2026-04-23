from __future__ import annotations

import json
from datetime import date
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Iterable

from ...spec import MacroSpec
from ...store import FactorStore
from ...store import DatasetSaveResult
from ...store import MacroBatchResult


def _load_default_fred_series() -> tuple[str, ...]:
    series_file = Path(__file__).with_name("fred_series.txt")
    if not series_file.exists():
        return ()
    series: list[str] = []
    for raw_line in series_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        series.append(line)
    return tuple(series)


DEFAULT_FRED_ECONOMY_SERIES = _load_default_fred_series()


@dataclass(frozen=True, slots=True)
class EconomySeriesTask:
    source: str
    series_name: str
    region: str = "us"
    freq: str = "auto"
    variant: str | None = None

    def to_spec(self) -> MacroSpec:
        return MacroSpec(
            region=self.region,
            freq=self.freq,
            table_name=self.series_name,
            variant=self.variant,
            provider=self.source,
            source_name=self.series_name,
        )


class EconomyDownloader:
    def __init__(self, store: FactorStore | None = None):
        self.store = store or FactorStore()

    def download_tasks(
        self,
        tasks: Iterable[EconomySeriesTask],
        *,
        start: object | None = None,
        end: object | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = False,
    ) -> MacroBatchResult:
        grouped: dict[str, list[MacroSpec]] = defaultdict(list)
        for task in tasks:
            grouped[task.source.lower()].append(task.to_spec())

        if not grouped:
            return MacroBatchResult(manifest_path=Path(), results={})

        merged_results: dict[str, DatasetSaveResult] = {}
        for source, specs in grouped.items():
            batch = self.store.download_macro_batch(
                specs,
                start=start,
                end=end,
                data_source=source,
                api_key=api_key,
                reader=reader,
                skip_failed=skip_failed,
            )
            merged_results.update(batch)

        manifest_path = self.store.root_dir / "macro" / "economy_batch_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sources": sorted(grouped),
            "results": {
                key: {
                    "dataset_dir": str(result.dataset_dir),
                    "manifest_path": str(result.manifest_path),
                    "files": [str(path) for path in result.files],
                    "rows": result.rows,
                    "date_min": result.date_min,
                    "date_max": result.date_max,
                }
                for key, result in merged_results.items()
            },
        }
        manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return MacroBatchResult(manifest_path=manifest_path, results=merged_results)

    def download_fred_core(
        self,
        *,
        start: object | None = None,
        end: object | None = None,
        region: str = "us",
        freq: str = "auto",
        variant: str | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = False,
    ) -> MacroBatchResult:
        return self.store.download_fred_macro_series(
            DEFAULT_FRED_ECONOMY_SERIES,
            region=region,
            freq=freq,
            variant=variant,
            start=start,
            end=end,
            api_key=api_key,
            reader=reader,
            skip_failed=skip_failed,
        )

    def download_fred_core_from_file(
        self,
        series_list_path: str | Path,
        *,
        region: str = "us",
        freq: str = "auto",
        variant: str | None = None,
        start: object | None = None,
        end: object | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = False,
        write_manifest: bool = True,
        column: str | int | None = None,
        has_header: bool = False,
    ) -> MacroBatchResult:
        return self.store.download_fred_macro_series_from_file(
            series_list_path,
            region=region,
            freq=freq,
            variant=variant,
            start=start,
            end=end,
            api_key=api_key,
            reader=reader,
            skip_failed=skip_failed,
            write_manifest=write_manifest,
            column=column,
            has_header=has_header,
        )

    def download_fred_all_since_2000(
        self,
        *,
        region: str = "us",
        freq: str = "auto",
        variant: str | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = True,
    ) -> MacroBatchResult:
        return self.download_fred_core(
            start="2000-01-01",
            end=date.today(),
            region=region,
            freq=freq,
            variant=variant,
            api_key=api_key,
            reader=reader,
            skip_failed=skip_failed,
        )


def main(root_dir: str | Path | None = None) -> MacroBatchResult:
    downloader = EconomyDownloader(store=FactorStore(root_dir))
    result = downloader.download_fred_all_since_2000(variant=None, skip_failed=True)
    print(
        f"downloaded {len(result.results)} macro series to {result.manifest_path}",
        flush=True,
    )
    return result


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
