from __future__ import annotations

import json

import pandas as pd

from tiger_factors.factor_algorithm.traditional_factors.pipeline import TraditionalFactorPipelineEngine


def test_traditional_factor_pipeline_saves_canonical_factor(tmp_path, monkeypatch):
    engine = TraditionalFactorPipelineEngine(output_root=tmp_path, verbose=False)

    data = pd.DataFrame(
        {
            "permno": [10001, 10002],
            "time_avail_m": pd.to_datetime(["2020-01-31", "2020-02-29"]),
        }
    )

    monkeypatch.setattr(
        "tiger_factors.factor_algorithm.traditional_factors.pipeline.run_original_factor",
        lambda name, data, **kwargs: pd.Series([1.25, 2.5], name=name),
    )

    factor_frame, saved = engine.compute_and_save_factor("Size", data=data)

    assert list(factor_frame.columns) == ["date_", "code", "value"]
    assert factor_frame["code"].tolist() == ["10001", "10002"]
    assert saved.parquet_path.exists()
    assert saved.metadata_path.exists()
    manifest = json.loads(saved.metadata_path.read_text())
    assert manifest["factor_name"] == "Size"


def test_traditional_factor_pipeline_compute_all_handles_subset(tmp_path, monkeypatch):
    engine = TraditionalFactorPipelineEngine(output_root=tmp_path, verbose=False)
    data = pd.DataFrame(
        {
            "permno": [10001, 10002],
            "yyyymm": [202001, 202002],
        }
    )

    monkeypatch.setattr(
        "tiger_factors.factor_algorithm.traditional_factors.pipeline.run_original_factor",
        lambda name, data, **kwargs: pd.Series([0.1, 0.2], name=name),
    )
    monkeypatch.setattr(
        "tiger_factors.factor_algorithm.traditional_factors.pipeline.available_factors",
        lambda: ["Size", "BM"],
    )

    result = engine.compute_all(data=data, save=True)

    assert set(result.factor_frames) == {"Size", "BM"}
    assert result.saved_factor_results is not None
    assert set(result.saved_factor_results) == {"Size", "BM"}
    assert result.manifest_path is not None
