from __future__ import annotations

import pandas as pd

from tiger_factors.factor_maker.pipeline import Pipeline
from tiger_factors.factor_maker.pipeline import PipelineEngine
from tiger_factors.factor_maker.pipeline import Returns
from tiger_factors.factor_maker.pipeline import USEquityPricing
from tiger_factors.factor_evaluation.evaluation import evaluate_from_pipeline
from tiger_factors.factor_store import FactorSpec


class FakeLibrary:
    def __init__(self) -> None:
        self.price_provider = "yahoo"
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def price_panel(self, *, codes, start, end, provider=None, field="close", as_ex=None):
        self.calls.append((field, tuple(codes)))
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        data = {
            "AAPL": [10, 11, 12, 13, 14, 15, 16, 17],
            "MSFT": [20, 19, 18, 19, 20, 21, 22, 23],
        }
        frame = pd.DataFrame(data, index=dates)
        if field == "volume":
            frame = frame * 100
        return frame.reindex(columns=codes)

    def fetch_price_data(self, *, codes, start, end, provider=None, freq="1d", as_ex=None):
        panel = self.price_panel(codes=codes, start=start, end=end, provider=provider, field="close", as_ex=as_ex)
        long_df = (
            panel.rename_axis(index="date_")
            .reset_index()
            .melt(id_vars="date_", var_name="code", value_name="close")
        )
        return long_df


def test_run_pipeline_factor_research(tmp_path) -> None:
    library = FakeLibrary()
    engine = PipelineEngine(library=library, as_ex=True)
    pipeline = Pipeline(
        columns={
            "close": USEquityPricing.column("close"),
            "momentum_3": Returns(window_length=3),
        },
        screen=USEquityPricing.column("close").notnull(),
    )

    result = evaluate_from_pipeline(
        spec=FactorSpec(
            provider="tiger",
            region="us",
            sec_type="stock",
            freq="1d",
            table_name="momentum_3",
        ),
        pipeline=pipeline,
        pipeline_engine=engine,
        codes=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-08",
        include_native_report=False,
        include_horizon=False,
    )

    assert result.report is not None
    assert result.report_bundle is not None
    assert result.report.output_dir.exists()
    assert result.evaluation is not None
