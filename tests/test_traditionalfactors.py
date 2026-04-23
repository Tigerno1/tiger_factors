from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_algorithm.traditional_factors import available_factors
from tiger_factors.factor_algorithm.traditional_factors import available_factor_templates
from tiger_factors.factor_algorithm.traditional_factors import factor_metadata
from tiger_factors.factor_algorithm.traditional_factors import get_factor_template
from tiger_factors.factor_algorithm.traditional_factors import Size
from tiger_factors.factor_algorithm import TraditionalPortfolioEngine
from tiger_factors.factor_algorithm import factor_metadata as root_factor_metadata
from tiger_factors.factor_algorithm import run_original_factor


def test_available_factors_exposes_vendored_signal_catalog():
    factors = available_factors()
    assert "Size" in factors
    assert "ChInvIA" in factors
    assert len(factors) >= 300


def test_size_runs_vendored_upstream_script_with_single_input():
    panel = pd.DataFrame(
        {
            "permno": [10001, 10002],
            "time_avail_m": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "mve_c": [10.0, np.e**2],
        }
    )

    result = Size(panel)

    expected = np.log(panel["mve_c"])
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())
    assert result.name == "Size"


def test_multi_input_factor_reports_missing_upstream_datasets_cleanly():
    panel = pd.DataFrame(
        {
            "permno": [10001],
            "time_avail_m": pd.to_datetime(["2020-01-31"]),
            "sicCRSP": ["35"],
        }
    )

    try:
        from tiger_factors.factor_algorithm.traditional_factors import ChInvIA

        ChInvIA(panel)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("ChInvIA should require extra upstream datasets")

    assert "m_aCompustat.parquet" in message
    assert factor_metadata("ChInvIA").script_relpath.endswith("Predictors/ChInvIA.py")


def test_traditional_factor_templates_are_explicit_placeholders():
    templates = available_factor_templates()
    assert len(templates) == 318
    assert templates[0] == "factor_001"
    with pytest.raises(NotImplementedError, match="placeholder template"):
        get_factor_template("factor_001")(pd.DataFrame({"x": [1]}))


def test_factor_algorithm_root_exports_openassetpricing_wrappers():
    assert TraditionalPortfolioEngine.__name__ == "TraditionalPortfolioEngine"
    assert root_factor_metadata("Size").script_relpath.endswith("Predictors/Size.py")
    assert run_original_factor.__name__ == "run_original_factor"
