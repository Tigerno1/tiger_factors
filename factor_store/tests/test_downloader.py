from __future__ import annotations

from pathlib import Path
from datetime import date

import pandas as pd

from tiger_factors.factor_store import EconomyDownloader
from tiger_factors.factor_store import EconomySeriesTask
from tiger_factors.factor_store import MacroBatchResult
from tiger_factors.factor_store.downloader import DEFAULT_FRED_ECONOMY_SERIES
from tiger_factors.factor_store.downloader import main


def test_economy_downloader_groups_sources_and_writes_manifest(tmp_path: Path) -> None:
    from tiger_factors.factor_store import FactorStore

    downloader = EconomyDownloader(store=FactorStore(tmp_path))

    tasks = [
        EconomySeriesTask(source="FRED", series_name="FEDFUNDS", freq="1m"),
        EconomySeriesTask(source="FRED", series_name="DGS10", freq="1m"),
    ]
    raw_frames = {
        "FEDFUNDS": pd.DataFrame({"FEDFUNDS": [5.33]}, index=pd.to_datetime(["2024-01-31"])),
        "DGS10": pd.DataFrame({"DGS10": [4.10]}, index=pd.to_datetime(["2024-01-31"])),
    }
    calls: list[tuple[object, object]] = []

    def reader(name: object, source: object, **kwargs: object) -> pd.DataFrame:
        calls.append((name, source))
        return raw_frames[str(name)]

    batch = downloader.download_tasks(tasks, reader=reader)

    assert set(batch.results) == {"fred/us/1m/fedfunds", "fred/us/1m/dgs10"}
    assert calls == [("FEDFUNDS", "fred"), ("DGS10", "fred")]
    assert batch.manifest_path.exists()
    assert (tmp_path / "macro" / "fred" / "us" / "1m" / "fedfunds" / "fedfunds.parquet").exists()
    assert (tmp_path / "macro" / "fred" / "us" / "1m" / "dgs10" / "dgs10.parquet").exists()


def test_default_fred_series_loaded_from_data_file() -> None:
    assert "FEDFUNDS" in DEFAULT_FRED_ECONOMY_SERIES
    assert "DGS10" in DEFAULT_FRED_ECONOMY_SERIES


def test_main_uses_2000_to_today(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_download_fred_core(self, **kwargs: object) -> MacroBatchResult:
        calls.append(kwargs)
        manifest = tmp_path / "macro" / "fred_batch_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        return MacroBatchResult(manifest_path=manifest, results={})

    monkeypatch.setattr(EconomyDownloader, "download_fred_core", fake_download_fred_core)

    result = main(root_dir=tmp_path)

    assert isinstance(result, MacroBatchResult)
    assert calls and calls[0]["start"] == "2000-01-01"
    assert calls[0]["end"] == date.today()
    assert calls[0]["variant"] is None
    assert calls[0]["skip_failed"] is True
    assert result.manifest_path.exists()
