from __future__ import annotations

import pandas as pd

from tiger_factors.factor_evaluation.main import _load_panel, _to_wide


def test_reports_load_and_wide_roundtrip(tmp_path) -> None:
    csv_path = tmp_path / "factor.csv"
    pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA"],
            "my_factor": [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    loaded = _load_panel(csv_path, value_col="my_factor")
    wide = _to_wide(loaded, date_col="date_", code_col="code", value_col="my_factor")

    assert list(wide.columns) == ["AAA", "BBB"]
    assert float(wide.loc[pd.Timestamp("2024-01-01"), "AAA"]) == 1.0
    assert float(wide.loc[pd.Timestamp("2024-01-01"), "BBB"]) == 2.0
