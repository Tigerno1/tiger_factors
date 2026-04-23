from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_algorithm import traditional_factor_group_frame
from tiger_factors.factor_algorithm import traditional_factor_group_index
from tiger_factors.factor_algorithm import traditional_factor_group_names
from tiger_factors.factor_algorithm import traditional_factor_group_for_signal
from tiger_factors.factor_algorithm.traditional_factors import available_factors
from tiger_factors.factor_algorithm.traditional_factors import traditional_factor_group_summary


def test_traditional_factor_group_index_covers_all_vendored_signals() -> None:
    frame = traditional_factor_group_frame()
    assert set(frame["signal_name"]) == set(available_factors())

    summary = traditional_factor_group_summary()
    assert int(summary["count"].sum()) == len(available_factors())


def test_traditional_factor_group_lookup_routes_common_signals() -> None:
    assert traditional_factor_group_for_signal("AM") == "value"
    assert traditional_factor_group_for_signal("BookLeverage") == "leverage"
    assert traditional_factor_group_for_signal("Beta") == "risk"
    assert traditional_factor_group_for_signal("Mom12m") == "momentum"
    assert traditional_factor_group_for_signal("Accruals") == "accruals"
    assert traditional_factor_group_for_signal("AnalystRevision") == "analyst"


def test_traditional_factor_group_index_has_expected_family_names() -> None:
    families = traditional_factor_group_names()
    assert "value" in families
    assert "profitability" in families
    assert "investment" in families
    assert "risk" in families
    assert "liquidity" in families
    assert "momentum" in families

    group_index = traditional_factor_group_index()
    assert "AM" in group_index["value"]
    assert "AssetGrowth" in group_index["investment"]
    assert "Illiquidity" in group_index["liquidity"]

