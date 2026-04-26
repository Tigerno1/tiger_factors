from __future__ import annotations

import pandas as pd
import pytest

from tiger_factors.factor_allocation import LongShortReturnConfig
from tiger_factors.factor_allocation import RiskfolioConfig
from tiger_factors.factor_allocation import allocate_selected_factors
from tiger_factors.factor_allocation import build_long_short_return_panel
from tiger_factors.factor_allocation import compute_factor_long_short_returns
from tiger_factors.factor_allocation import optimize_factor_weights_with_riskfolio


def _sample_prices_and_factors() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    dates = pd.bdate_range("2024-01-01", periods=18)
    codes = ["A", "B", "C", "D", "E", "F"]
    daily_returns = {
        "A": 0.018,
        "B": 0.012,
        "C": 0.006,
        "D": -0.003,
        "E": -0.008,
        "F": -0.012,
    }
    prices = pd.DataFrame(index=dates, columns=codes, dtype=float)
    prices.iloc[0] = [100.0] * len(codes)
    for i in range(1, len(dates)):
        for code in codes:
            prices.loc[dates[i], code] = prices.loc[dates[i - 1], code] * (1.0 + daily_returns[code])

    factor_1 = pd.DataFrame(
        [list(range(len(codes), 0, -1)) for _ in dates],
        index=dates,
        columns=codes,
        dtype=float,
    ).stack(future_stack=True)
    factor_1.name = "momentum"

    factor_2 = (-factor_1).rename("reversal")
    return prices, {"momentum": factor_1, "reversal": factor_2}


def test_compute_factor_long_short_returns_supports_tiger_source() -> None:
    prices, factor_dict = _sample_prices_and_factors()

    series = compute_factor_long_short_returns(
        factor_dict["momentum"],
        prices,
        config=LongShortReturnConfig(periods=(1, 3), quantiles=3, selected_period=1, source="tiger"),
    )

    assert isinstance(series, pd.Series)
    assert series.name == "momentum"
    assert not series.empty


def test_build_long_short_return_panel_supports_alphalens_source() -> None:
    prices, factor_dict = _sample_prices_and_factors()

    panel = build_long_short_return_panel(
        factor_dict,
        prices,
        config=LongShortReturnConfig(periods=(1,), quantiles=3, selected_period=1, source="alphalens"),
    )

    assert list(panel.columns) == ["momentum", "reversal"]
    assert not panel.empty


def test_optimize_factor_weights_with_riskfolio_uses_lazy_import(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "momentum": [0.01, 0.02, -0.01],
            "reversal": [0.005, -0.002, 0.004],
        },
        index=pd.bdate_range("2024-01-01", periods=3),
    )

    class FakePortfolio:
        def __init__(self, returns):
            self.returns = returns

        def assets_stats(self, method_mu="hist", method_cov="hist"):
            self.method_mu = method_mu
            self.method_cov = method_cov

        def optimization(self, **kwargs):
            return pd.DataFrame({"weights": [0.7, 0.3]}, index=["momentum", "reversal"])

    class FakeRiskfolio:
        Portfolio = FakePortfolio

    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.allocation._import_riskfolio",
        lambda: FakeRiskfolio,
    )

    weights = optimize_factor_weights_with_riskfolio(panel, config=RiskfolioConfig())

    assert weights.index.tolist() == ["momentum", "reversal"]
    assert weights.sum() == 1.0


def test_riskfolio_config_passes_extended_kwargs(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "momentum": [0.01, 0.02, -0.01],
            "reversal": [0.005, -0.002, 0.004],
        },
        index=pd.bdate_range("2024-01-01", periods=3),
    )

    captured: dict[str, dict[str, object]] = {}

    class FakePortfolio:
        def __init__(self, returns, **kwargs):
            self.returns = returns
            captured["portfolio_kwargs"] = kwargs

        def assets_stats(self, method_mu="hist", method_cov="hist", **kwargs):
            self.method_mu = method_mu
            self.method_cov = method_cov
            captured["assets_stats_kwargs"] = kwargs

        def optimization(self, **kwargs):
            captured["optimization_kwargs"] = kwargs
            return pd.DataFrame({"weights": [0.7, 0.3]}, index=["momentum", "reversal"])

    class FakeRiskfolio:
        Portfolio = FakePortfolio

    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.allocation._import_riskfolio",
        lambda: FakeRiskfolio,
    )

    weights = optimize_factor_weights_with_riskfolio(
        panel,
        config=RiskfolioConfig(
            portfolio_kwargs={"budget": 1.0},
            assets_stats_kwargs={"method_kurt": "hist", "dict_mu": {"momentum": 1.0}},
            optimization_kwargs={"allow_short": False},
        ),
    )

    assert captured["portfolio_kwargs"] == {"budget": 1.0}
    assert captured["assets_stats_kwargs"] == {"method_kurt": "hist", "dict_mu": {"momentum": 1.0}}
    assert captured["optimization_kwargs"]["allow_short"] is False
    assert weights.index.tolist() == ["momentum", "reversal"]
    assert weights.sum() == 1.0


def test_allocate_selected_factors_filters_by_screening_summary(monkeypatch) -> None:
    prices, factor_dict = _sample_prices_and_factors()
    screening_summary = pd.DataFrame(
        {
            "factor_name": ["momentum", "reversal"],
            "usable": [True, False],
        }
    )

    monkeypatch.setattr(
        "tiger_factors.multifactor_evaluation.allocation.optimize_factor_weights_with_riskfolio",
        lambda panel, config=None: pd.Series({"momentum": 1.0}, name="weight"),
    )

    panel, weights = allocate_selected_factors(
        factor_dict,
        prices,
        screening_summary=screening_summary,
        long_short_config=LongShortReturnConfig(periods=(1,), quantiles=3, selected_period=1, source="tiger"),
    )

    assert list(panel.columns) == ["momentum"]
    assert weights.index.tolist() == ["momentum"]


def test_real_riskfolio_smoke_optimization() -> None:
    pytest.importorskip("riskfolio")
    panel = pd.DataFrame(
        {
            "a": [0.01, 0.02, -0.01, 0.015],
            "b": [0.005, -0.002, 0.004, 0.006],
        },
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    weights = optimize_factor_weights_with_riskfolio(panel)

    assert set(weights.index) == {"a", "b"}
    assert weights.sum() == 1.0
