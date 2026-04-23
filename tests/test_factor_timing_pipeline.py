from __future__ import annotations

import pandas as pd

from tiger_factors.factor_algorithm.factor_timing.pipeline import FactorTimingPipelineEngine


def test_factor_timing_pipeline_saves_canonical_factor(tmp_path):
    engine = FactorTimingPipelineEngine(output_root=tmp_path, verbose=False)
    price_df = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0, 13.0],
            "BBB": [20.0, 22.0, 24.0, 26.0],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]),
    )

    factor_frame, saved = engine.compute_and_save_factor("MOM9", price_df)

    assert list(factor_frame.columns) == ["date_", "code", "value"]
    assert set(factor_frame["code"]) == {"AAA", "BBB"}
    assert saved.parquet_path.exists()
    assert saved.metadata_path.exists()


def test_factor_timing_pipeline_compute_all_subset(tmp_path, monkeypatch):
    engine = FactorTimingPipelineEngine(output_root=tmp_path, verbose=False)
    price_df = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0],
            "BBB": [20.0, 19.0, 18.0],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )

    monkeypatch.setattr(
        "tiger_factors.factor_algorithm.factor_timing.pipeline.available_factors",
        lambda: ["MOM9", "MOM10"],
    )

    result = engine.compute_all(price_df, save=True)

    assert set(result.factor_frames) == {"MOM9", "MOM10"}
    assert result.saved_factor_results is not None
    assert set(result.saved_factor_results) == {"MOM9", "MOM10"}
    assert result.manifest_path is not None
