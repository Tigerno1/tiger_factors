from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation import TigerFactorData
from tiger_factors.factor_evaluation import bootstrap_confidence_interval
from tiger_factors.factor_evaluation import permutation_test
from tiger_factors.factor_evaluation import validate_factor_data
from tiger_factors.factor_evaluation import validate_series
from tiger_factors.factor_frame import FactorDefinition
from tiger_factors.factor_frame import FactorDefinitionRegistry
from tiger_factors.factor_frame import FactorGroupEngine
from tiger_factors.factor_frame import FactorGroupSpec
from tiger_factors.multifactor_evaluation import benjamini_hochberg
from tiger_factors.multifactor_evaluation import bonferroni_adjust
from tiger_factors.multifactor_evaluation import estimate_pi0
from tiger_factors.multifactor_evaluation import holm_adjust
from tiger_factors.multifactor_evaluation import storey_qvalues
from tiger_factors.multifactor_evaluation import validate_factor_family


class _PanelDefinition(FactorDefinition):
    def __init__(self, name: str, panel: pd.DataFrame) -> None:
        super().__init__(name=name)
        self._panel = panel

    def compute(self, ctx, state):  # type: ignore[override]
        return self._panel


def _sample_panel(scale: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "AAA": np.array([1.0, 2.0, 3.0, 4.0]) * scale,
            "BBB": np.array([2.0, 3.0, 4.0, 5.0]) * scale,
        },
        index=dates,
    )


def _sample_tiger_factor_data() -> TigerFactorData:
    dates = pd.bdate_range("2024-01-01", periods=6)
    codes = ["AAA", "BBB", "CCC"]
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            factor = float(code_idx) + 0.2 * date_idx
            forward = 0.15 * factor - 0.05 * code_idx + 0.01 * date_idx
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "factor": factor,
                    "1D": forward,
                }
            )
    factor_data = pd.DataFrame(rows).set_index(["date_", "code"]).sort_index()
    factor_frame = factor_data.reset_index()
    factor_series = factor_data["factor"]
    factor_panel = factor_data["factor"].unstack()
    forward_returns = factor_data["1D"].unstack()
    prices = (1.0 + forward_returns.fillna(0.0)).cumprod()
    price_frame = prices.stack().rename("close").reset_index()
    return TigerFactorData(
        factor_data=factor_data,
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_series=factor_series,
        prices=prices,
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_column="factor",
        date_column="date_",
        code_column="code",
        price_column="close",
        periods=("1D",),
        quantiles=5,
    )


def test_factor_group_engine_combines_registry_members() -> None:
    registry = FactorDefinitionRegistry()
    registry.register_many(
        _PanelDefinition("momentum", _sample_panel(1.0)),
        _PanelDefinition("value", _sample_panel(2.0)),
    )
    engine = FactorGroupEngine(definition_registry=registry)
    engine.register_group(
        FactorGroupSpec(
            name="core_family",
            members=("momentum", "value"),
            weights={"momentum": 0.25, "value": 0.75},
        )
    )

    result = engine.run(object())
    assert "core_family" in result.group_panels
    assert not result.summary.empty
    assert {"date_", "code", "group", "value"}.issubset(result.long_frame.columns)

    expected = _sample_panel(1.0) * 0.25 + _sample_panel(2.0) * 0.75
    pd.testing.assert_frame_equal(result.group_panels["core_family"], expected)


def test_factor_group_engine_supports_mean_combination() -> None:
    engine = FactorGroupEngine(
        member_sources={
            "a": lambda ctx: _sample_panel(1.0),
            "b": lambda ctx: _sample_panel(3.0),
        }
    )
    engine.register_group(FactorGroupSpec(name="blend", members=("a", "b"), combine_method="mean"))

    result = engine.run(object())
    expected = (_sample_panel(1.0) + _sample_panel(3.0)) / 2.0
    pd.testing.assert_frame_equal(result.group_panels["blend"], expected)


def test_validation_helpers_and_family_adjustments() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="metric")
    ci = bootstrap_confidence_interval(series, n_bootstrap=64, random_state=1)
    perm = permutation_test(series, n_permutations=64, random_state=1)
    validation = validate_series(series, metric_name="metric", n_bootstrap=64, n_permutations=64, random_state=1)

    assert ci["observed"] == 3.0
    assert 0.0 <= float(perm["p_value"]) <= 1.0
    assert validation.metric_name == "metric"
    assert validation.n_obs == 5

    p_values = pd.Series([0.001, 0.02, 0.2], index=["alpha", "beta", "gamma"])
    bonf = bonferroni_adjust(p_values, alpha=0.05)
    holm = holm_adjust(p_values, alpha=0.05)
    bh = benjamini_hochberg(p_values, alpha=0.05)
    storey = storey_qvalues(p_values, alpha=0.05)

    assert bonf.rejected_count == 1
    assert holm.rejected_count >= bonf.rejected_count
    assert bh.rejected_count >= 1
    assert 0.0 <= float(storey.pi0 or 0.0) <= 1.0
    assert estimate_pi0(p_values) <= 1.0

    family_report = validate_factor_family(
        pd.DataFrame(
            {
                "factor_name": ["alpha", "beta", "gamma"],
                "p_value": [0.001, 0.02, 0.2],
                "fitness": [1.0, 0.8, 0.2],
            }
        ),
        method="bh",
        alpha=0.05,
        cluster_threshold=0.5,
        factor_dict={
            "alpha": _sample_panel(1.0),
            "beta": _sample_panel(1.0) * 1.01,
            "gamma": _sample_panel(-1.0),
        },
    )
    assert "table" in family_report
    assert "adjustment" in family_report
    assert family_report["clusters"] is not None


def test_validate_factor_data_returns_core_and_validation() -> None:
    tfd = _sample_tiger_factor_data()
    report = validate_factor_data(tfd, period="1D", n_bootstrap=32, n_permutations=32, random_state=7)

    assert report["period"] == "1D"
    assert "core" in report
    assert "ic_validation" in report
    assert "long_short_validation" in report
    assert report["ic_validation"]["n_obs"] > 0
