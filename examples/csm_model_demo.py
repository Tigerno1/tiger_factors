"""Minimal CSM research demo.

This script shows how to fit a cross-sectional selection model on a long
panel, score the universe, and take the top names per date.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import infer_csm_feature_columns


def _build_sample_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=12)
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            momentum = code_idx + 0.05 * date_idx + rng.normal(0.0, 0.02)
            value = (len(codes) - code_idx) + 0.03 * date_idx + rng.normal(0.0, 0.02)
            quality = 0.3 * momentum - 0.2 * value + rng.normal(0.0, 0.02)
            forward_return = 0.5 * momentum - 0.35 * value + 0.15 * quality + rng.normal(0.0, 0.02)
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "momentum": momentum,
                    "value": value,
                    "quality": quality,
                    "forward_return": forward_return,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    panel = _build_sample_panel()
    train = panel[panel["date_"] <= pd.Timestamp("2024-01-08")].copy()
    test = panel[panel["date_"] > pd.Timestamp("2024-01-08")].copy()
    feature_columns = infer_csm_feature_columns(train)

    model = build_csm_model(
        feature_columns,
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=3,
        normalize_score_by_date=True,
    )
    model.fit(train)

    print("feature weights:")
    print(model.weights_)
    print("\nfeature stats:")
    print(model.feature_stats_.to_string(index=False))

    scored = model.predict(test)
    print("\nscored head:")
    print(scored.loc[:, ["date_", "code", "csm_score", "csm_rank"]].head(12).to_string(index=False))

    selected = model.select(test, top_n=2, bottom_n=2, long_only=False)
    print("\nselected head:")
    print(
        selected.loc[:, ["date_", "code", "csm_score", "csm_rank", "csm_side", "csm_target_weight"]]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
