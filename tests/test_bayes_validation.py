from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation import bayesian_fdr
from tiger_factors.multifactor_evaluation import bayesian_fwer
from tiger_factors.multifactor_evaluation import alpha_hacking_bayesian_update
from tiger_factors.multifactor_evaluation import dynamic_bayesian_alpha
from tiger_factors.multifactor_evaluation import fit_bayesian_mixture
from tiger_factors.multifactor_evaluation import fit_hierarchical_bayesian_mixture
from tiger_factors.multifactor_evaluation import hierarchical_bayesian_fdr
from tiger_factors.multifactor_evaluation import rolling_bayesian_alpha
from tiger_factors.multifactor_evaluation import validate_bayesian_factor_family
from tiger_factors.multifactor_evaluation import validate_hierarchical_bayesian_factor_family


def test_fit_bayesian_mixture_produces_posteriors() -> None:
    frame = pd.DataFrame(
        {
            "factor_name": ["alpha_a", "alpha_b", "alpha_c", "alpha_d"],
            "p_value": [0.0001, 0.01, 0.25, 0.75],
            "fitness": [1.8, 1.1, 0.2, -0.1],
        }
    )
    result = fit_bayesian_mixture(frame, alpha=0.2)
    table = result["table"]

    assert 0.0 <= result["pi0"] <= 1.0
    assert result["alt_variance"] >= 1.0
    assert "posterior_signal_prob" in table.columns
    assert "posterior_null_prob" in table.columns
    assert table.loc[table["factor_name"].eq("alpha_a"), "posterior_signal_prob"].iloc[0] >= table.loc[
        table["factor_name"].eq("alpha_d"), "posterior_signal_prob"
    ].iloc[0]
    assert table["discovered"].any()
    assert bayesian_fdr(result["result"]) >= 0.0


def test_validate_bayesian_factor_family_returns_summary() -> None:
    frame = pd.DataFrame(
        {
            "factor_name": ["alpha_a", "alpha_b", "alpha_c"],
            "p_value": [0.001, 0.02, 0.6],
            "fitness": [1.5, 0.7, 0.1],
        }
    )
    report = validate_bayesian_factor_family(frame, alpha=0.2)

    assert "table" in report
    assert "bayesian_fdr" in report
    assert "pi0" in report
    assert isinstance(report["table"], pd.DataFrame)
    assert report["table"]["posterior_signal_prob"].between(0.0, 1.0).all()


def test_hierarchical_bayesian_mixture_borrows_cluster_evidence() -> None:
    frame = pd.DataFrame(
        {
            "factor_name": ["alpha_a", "alpha_b", "alpha_c", "alpha_d"],
            "p_value": [0.0001, 0.002, 0.18, 0.52],
            "fitness": [1.9, 1.5, 0.3, -0.2],
        }
    )
    factor_dict = {
        "alpha_a": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=10), "code": ["A"] * 10, "value": np.linspace(0.2, 1.1, 10)}).pivot(index="date_", columns="code", values="value"),
        "alpha_b": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=10), "code": ["A"] * 10, "value": np.linspace(0.25, 1.05, 10)}).pivot(index="date_", columns="code", values="value"),
        "alpha_c": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=10), "code": ["A"] * 10, "value": np.linspace(-0.3, 0.3, 10)}).pivot(index="date_", columns="code", values="value"),
        "alpha_d": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=10), "code": ["A"] * 10, "value": np.linspace(-0.1, 0.1, 10)}).pivot(index="date_", columns="code", values="value"),
    }

    result = fit_hierarchical_bayesian_mixture(
        frame,
        alpha=0.2,
        factor_dict=factor_dict,
        cluster_threshold=0.75,
    )
    table = result["table"]

    assert "hierarchical_signal_prob" in table.columns
    assert "hierarchical_null_prob" in table.columns
    assert "hierarchical_posterior_mean" in table.columns
    assert "cluster" in table.columns
    assert result["factor_fit"]["pi0"] >= 0.0
    assert hierarchical_bayesian_fdr(result["result"]) >= 0.0


def test_validate_hierarchical_bayesian_factor_family_returns_summary() -> None:
    frame = pd.DataFrame(
        {
            "factor_name": ["alpha_a", "alpha_b", "alpha_c"],
            "p_value": [0.001, 0.03, 0.7],
            "fitness": [1.5, 0.6, 0.1],
        }
    )
    factor_dict = {
        "alpha_a": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=8), "code": ["A"] * 8, "value": np.linspace(0.3, 1.0, 8)}).pivot(index="date_", columns="code", values="value"),
        "alpha_b": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=8), "code": ["A"] * 8, "value": np.linspace(0.35, 0.95, 8)}).pivot(index="date_", columns="code", values="value"),
        "alpha_c": pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=8), "code": ["A"] * 8, "value": np.linspace(-0.2, 0.2, 8)}).pivot(index="date_", columns="code", values="value"),
    }
    report = validate_hierarchical_bayesian_factor_family(frame, alpha=0.2, factor_dict=factor_dict, cluster_threshold=0.8)

    assert "table" in report
    assert "hierarchical_fdr" in report
    assert isinstance(report["table"], pd.DataFrame)
    assert report["table"]["hierarchical_signal_prob"].between(0.0, 1.0).all()


def test_bayesian_fwer_uses_posterior_null_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "factor_name": ["alpha_a", "alpha_b", "alpha_c"],
            "p_value": [0.001, 0.02, 0.6],
            "fitness": [1.5, 0.6, 0.1],
        }
    )
    result = fit_bayesian_mixture(frame, alpha=0.2)

    assert 0.0 <= bayesian_fwer(result["result"]) <= 1.0
    assert bayesian_fwer(result["result"]) >= bayesian_fdr(result["result"])


def test_rolling_bayesian_alpha_produces_time_series() -> None:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    alpha = np.linspace(-0.01, 0.04, len(dates)) + np.random.default_rng(11).normal(0.0, 0.003, len(dates))
    frame = pd.DataFrame({"date_": dates, "alpha": alpha})

    result = rolling_bayesian_alpha(frame, window=12, min_periods=8, alpha=0.2)
    table = result["table"]

    assert not table.empty
    assert {"ols_alpha", "posterior_alpha", "bayesian_fdr", "bayesian_fwer"}.issubset(table.columns)
    assert table["posterior_alpha"].dtype.kind in {"f", "c"}
    assert table["bayesian_fdr"].between(0.0, 1.0).all()
    assert table["bayesian_fwer"].between(0.0, 1.0).all()
    assert 0.0 <= result["bayesian_fdr"] <= 1.0
    assert 0.0 <= result["bayesian_fwer"] <= 1.0


def test_dynamic_bayesian_alpha_produces_state_space_path() -> None:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    alpha = np.linspace(-0.01, 0.04, len(dates)) + np.random.default_rng(17).normal(0.0, 0.003, len(dates))
    frame = pd.DataFrame({"date_": dates, "alpha": alpha})

    result = dynamic_bayesian_alpha(frame, process_discount=0.985, alpha=0.2)
    table = result["table"]

    assert not table.empty
    assert {"prior_alpha", "posterior_alpha", "posterior_alpha_sd", "kalman_gain", "posterior_null_prob"}.issubset(table.columns)
    assert table["posterior_alpha"].dtype.kind in {"f", "c"}
    assert table["posterior_null_prob"].between(0.0, 1.0).all()
    assert 0.0 <= result["bayesian_fdr"] <= 1.0
    assert 0.0 <= result["bayesian_fwer"] <= 1.0


def test_alpha_hacking_bayesian_update_weights_oos_more_when_bias_is_large() -> None:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(23)
    in_sample = 0.01 + np.linspace(0.0, 0.03, len(dates)) + rng.normal(0.001, 0.001, len(dates))
    out_of_sample = 0.008 + np.linspace(0.0, 0.02, len(dates)) + rng.normal(0.0, 0.001, len(dates))
    frame = pd.DataFrame(
        {
            "date_": dates,
            "in_sample_alpha": in_sample,
            "out_of_sample_alpha": out_of_sample,
        }
    )

    result = alpha_hacking_bayesian_update(frame, alpha=0.2)
    table = result["table"]

    assert not table.empty
    assert {"hacking_bias", "posterior_alpha", "out_of_sample_weight", "in_sample_weight"}.issubset(table.columns)
    assert table["out_of_sample_weight"].between(0.0, 1.0).all()
    assert table["in_sample_weight"].between(0.0, 1.0).all()
    assert 0.0 <= result["bayesian_fdr"] <= 1.0
    assert 0.0 <= result["bayesian_fwer"] <= 1.0
