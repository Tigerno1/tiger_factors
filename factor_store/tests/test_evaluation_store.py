from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tiger_factors.factor_store.evaluation_store import EvaluationStore
from tiger_factors.factor_store import FactorSpec


def test_evaluation_store_ensure_run_dir(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)

    result = store.ensure_run_dir("alpha", "20260406")

    assert result.root_dir == tmp_path
    assert result.run_dir == tmp_path / "alpha" / "20260406"
    assert result.run_dir.exists()


def test_evaluation_store_save_and_load_summary(tmp_path: Path) -> None:
    spec = FactorSpec(
        region="US",
        sec_type="stock",
        freq="1d",
        table_name="alpha_001",
        provider="tiger",
    )
    store = EvaluationStore(tmp_path)
    summary = pd.DataFrame(
        [
            {
                "ic_mean": 0.12,
                "sharpe": 1.5,
                "best_horizon": 5,
                "best_long_short_return": 0.08,
            }
        ]
    )

    result = store.save_evaluation(summary, metadata={"family": "momentum"}, spec=spec)

    assert result.saved is True
    assert result.root_dir == tmp_path
    assert result.run_dir == tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_001"
    assert result.summary_path == tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_001" / "summary" / "summary.parquet"
    assert result.manifest_path == tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_001" / "manifest.json"
    assert result.summary_path.exists()
    assert result.manifest_path.exists()

    loaded = pd.read_parquet(result.summary_path)
    assert loaded.equals(summary)

    with pytest.raises(FileExistsError, match="evaluation already exists"):
        store.save_evaluation(summary, metadata={"family": "momentum"}, spec=spec)
    store.save_evaluation(summary, metadata={"family": "momentum"}, spec=spec, force_updated=True)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["spec"] == spec.to_dict()
    assert manifest["rows"] == 1
    assert manifest["columns"] == [
        "ic_mean",
        "sharpe",
        "best_horizon",
        "best_long_short_return",
    ]
    assert manifest["metadata"] == {"family": "momentum"}


def test_evaluation_store_save_false_keeps_memory_only(tmp_path: Path) -> None:
    spec = FactorSpec(
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="alpha_002",
        provider="tiger",
    )
    store = EvaluationStore(tmp_path)

    result = store.save_evaluation(
        {"ic_mean": 0.03, "sharpe": 0.7},
        save=False,
        spec=spec,
    )

    assert result.saved is False
    assert result.root_dir is None
    assert result.run_dir is None
    assert result.summary_path is None
    assert result.manifest_path is None
    assert result.data == {"ic_mean": 0.03, "sharpe": 0.7}
    assert not (tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_002").exists()


def test_evaluation_store_uses_factor_spec_directly(tmp_path: Path) -> None:
    spec = FactorSpec(
        region="US",
        sec_type="stock",
        freq="1d",
        table_name="momentum_12m_1m",
        provider="simfin",
    )
    summary = pd.DataFrame([{"ic_mean": 0.08, "sharpe": 1.2}])
    store = EvaluationStore(tmp_path)

    result = store.save_evaluation(summary, metadata={"family": "price"}, spec=spec)

    assert result.spec.region == "us"
    assert result.spec.sec_type == "stock"
    assert result.spec.freq == "1d"
    assert result.spec.table_name == "momentum_12m_1m"
    assert result.spec.provider == "simfin"
    assert result.spec.variant is None
    assert result.run_dir == tmp_path / "evaluation" / "simfin" / "us" / "stock" / "1d" / "momentum_12m_1m"
    assert result.summary_path == tmp_path / "evaluation" / "simfin" / "us" / "stock" / "1d" / "momentum_12m_1m" / "summary" / "summary.parquet"
    with pytest.raises(FileExistsError, match="evaluation already exists"):
        store.save_evaluation(summary, metadata={"family": "price"}, spec=spec)
    store.save_evaluation(summary, metadata={"family": "price"}, spec=spec, force_updated=True)


def test_evaluation_store_reads_unique_alternate_group(tmp_path: Path) -> None:
    legacy_spec = FactorSpec(
        region="US",
        sec_type="stock",
        freq="1d",
        table_name="alpha_021",
        provider="tiger",
        group="apha101",
    )
    requested_spec = FactorSpec(
        region="US",
        sec_type="stock",
        freq="1d",
        table_name="alpha_021",
        provider="tiger",
        group="alpha_101",
    )
    summary = pd.DataFrame([{"ic_mean": 0.08, "sharpe": 1.2}])
    returns = pd.DataFrame({"date_": pd.date_range("2024-01-01", periods=2), "long_short": [0.01, -0.02]})
    store = EvaluationStore(tmp_path)

    store.save_evaluation(summary, metadata={"family": "legacy"}, spec=legacy_spec)
    store.save_returns(returns, spec=legacy_spec, table_name="factor_portfolio_returns")

    assert store.section(requested_spec, "summary").get_table().equals(summary)
    assert store.section(requested_spec, "returns").get_table("factor_portfolio_returns").equals(returns)
    assert store.list_tables(requested_spec, "returns") == ["factor_portfolio_returns"]


def test_evaluation_store_can_roundtrip_nested_summary_payload(tmp_path: Path) -> None:
    spec = FactorSpec(
        region="US",
        sec_type="stock",
        freq="1d",
        table_name="alpha_003",
        provider="tiger",
    )
    store = EvaluationStore(tmp_path)
    summary = {
        "metrics": {
            "ic_mean": 0.05,
            "sharpe": 0.9,
        },
        "table": pd.DataFrame(
            [
                {"bucket": "long", "return": 0.12},
                {"bucket": "short", "return": -0.03},
            ]
        ),
    }

    result = store.save_evaluation(summary, spec=spec)
    assert result.saved is True
    assert result.run_dir == tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_003"
    assert result.summary_path == tmp_path / "evaluation" / "tiger" / "us" / "stock" / "1d" / "alpha_003" / "summary" / "summary.json"

    loaded_payload = EvaluationStore._decode_payload(
        json.loads(result.summary_path.read_text(encoding="utf-8"))
    )
    loaded = loaded_payload
    assert isinstance(loaded, dict)
    assert loaded["metrics"] == {"ic_mean": 0.05, "sharpe": 0.9}
    assert isinstance(loaded["table"], pd.DataFrame)
    assert loaded["table"].equals(summary["table"])
