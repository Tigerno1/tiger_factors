from __future__ import annotations

import subprocess
import sys

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
from tiger_factors.factor_ml.alphagpt import AlphaFeatureEngineer, AlphaFormulaVM


def test_alphagpt_feature_engineer_builds_expected_shape():
    raw = {
        "open": torch.tensor([[10.0, 11.0, 12.0, 13.0]]),
        "high": torch.tensor([[11.0, 12.0, 13.0, 14.0]]),
        "low": torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
        "close": torch.tensor([[10.5, 11.5, 12.5, 13.5]]),
        "volume": torch.tensor([[100.0, 120.0, 140.0, 160.0]]),
        "adv": torch.tensor([[1000.0, 1100.0, 1050.0, 1200.0]]),
        "market_cap": torch.tensor([[5000.0, 5050.0, 5100.0, 5200.0]]),
    }
    features = AlphaFeatureEngineer.compute_features(raw)
    assert features.shape == (1, AlphaFeatureEngineer.INPUT_DIM, 4)


def test_alphagpt_vm_executes_simple_formula():
    feat_tensor = torch.arange(30, dtype=torch.float32).reshape(1, AlphaFeatureEngineer.INPUT_DIM, -1)
    vm = AlphaFormulaVM()
    # feature 0 + feature 1
    result = vm.execute([0, 1, AlphaFeatureEngineer.INPUT_DIM], feat_tensor)
    assert result is not None
    assert torch.equal(result, feat_tensor[:, 0, :] + feat_tensor[:, 1, :])
