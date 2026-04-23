from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest


def _torch_importable() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


if not _torch_importable():
    pytest.skip("torch is installed but not importable in this environment", allow_module_level=True)

import torch
from tiger_factors.factor_ml.alpha_constraints import FormulaConstraints
from tiger_factors.factor_evaluation import evaluate_factor_panel
from tiger_factors.factor_ml.alphagpt import AlphaFeatureEngineer, AlphaFormulaVM


def test_formula_validator_rejects_deep_invalid_formula():
    vm = AlphaFormulaVM(constraints=FormulaConstraints(max_depth=2))
    invalid = [0, 1, AlphaFeatureEngineer.INPUT_DIM, AlphaFeatureEngineer.INPUT_DIM + 1]
    result = vm.validate(invalid)
    assert not result.valid


def test_formula_compile_and_execute_uses_dag_path():
    feat_tensor = torch.arange(30, dtype=torch.float32).reshape(1, AlphaFeatureEngineer.INPUT_DIM, -1)
    vm = AlphaFormulaVM()
    compiled = vm.compile([0, 1, AlphaFeatureEngineer.INPUT_DIM])
    result = vm.execute(compiled, feat_tensor)
    assert result is not None
    assert torch.equal(result, feat_tensor[:, 0, :] + feat_tensor[:, 1, :])


def test_factor_evaluation_returns_fitness():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.5, 1.0, 2.5], [0.5, 1.2, 1.8]],
        index=pd.date_range("2024-01-01", periods=3),
        columns=["A", "B", "C"],
    )
    forward_returns = pd.DataFrame(
        [[0.01, 0.02, 0.03], [0.03, 0.01, 0.02], [0.00, 0.01, 0.02]],
        index=factor.index,
        columns=factor.columns,
    )
    evaluation = evaluate_factor_panel(factor, forward_returns)
    assert evaluation.fitness == evaluation.fitness
    assert evaluation.sharpe == evaluation.sharpe
