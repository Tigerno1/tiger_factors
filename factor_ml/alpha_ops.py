from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import Callable

import torch


class ValueKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    BOOLEAN = "boolean"


class AxisKind(str, Enum):
    ANY = "any"
    PANEL = "panel"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    PORTFOLIO = "portfolio"


class SemanticKind(str, Enum):
    GENERIC = "generic"
    PRICE = "price"
    VOLUME = "volume"
    RETURN = "return"
    RANK = "rank"
    SCORE = "score"
    LIQUIDITY = "liquidity"
    RISK = "risk"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class TensorSpec:
    kind: ValueKind = ValueKind.MATRIX
    axis: AxisKind = AxisKind.PANEL
    semantic: SemanticKind = SemanticKind.GENERIC


GENERIC_PANEL = TensorSpec()
PRICE_PANEL = TensorSpec(semantic=SemanticKind.PRICE)
RETURN_PANEL = TensorSpec(semantic=SemanticKind.RETURN)
VOLUME_PANEL = TensorSpec(semantic=SemanticKind.VOLUME)
LIQUIDITY_PANEL = TensorSpec(semantic=SemanticKind.LIQUIDITY)
RISK_PANEL = TensorSpec(semantic=SemanticKind.RISK)
RANK_PANEL = TensorSpec(semantic=SemanticKind.RANK)
SCORE_PANEL = TensorSpec(semantic=SemanticKind.SCORE)
BOOLEAN_PANEL = TensorSpec(kind=ValueKind.BOOLEAN, semantic=SemanticKind.BOOLEAN)


@dataclass(frozen=True)
class AlphaOperator:
    name: str
    func: Callable[..., torch.Tensor]
    arity: int
    category: str
    lookback: int = 0
    input_types: tuple[TensorSpec, ...] = ()
    output_type: TensorSpec = GENERIC_PANEL
    complexity: int = 1


def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)


def _rolling_apply(x: torch.Tensor, window: int, reducer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if window <= 1:
        return x
    pad = torch.zeros((x.shape[0], window - 1), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    unfolded = x_pad.unfold(1, window, 1)
    return reducer(unfolded)


def _ts_delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    return x - _ts_delay(x, d)


def _ts_mean(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    return _rolling_apply(x, window, lambda t: t.mean(dim=-1))


def _ts_std(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    return _rolling_apply(x, window, lambda t: t.std(dim=-1))


def _ts_sum(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    return _rolling_apply(x, window, lambda t: t.sum(dim=-1))


def _ts_rank(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    def reducer(t: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(torch.argsort(t, dim=-1), dim=-1).to(torch.float32)
        denom = max(t.shape[-1] - 1, 1)
        return order[..., -1] / denom

    return _rolling_apply(x, window, reducer)


def _ts_zscore(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    mean = _ts_mean(x, window)
    std = _ts_std(x, window) + 1e-6
    return (x - mean) / std


def _ts_argmax(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    return _rolling_apply(x, window, lambda t: torch.argmax(t, dim=-1).to(torch.float32))


def _ts_argmin(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    return _rolling_apply(x, window, lambda t: torch.argmin(t, dim=-1).to(torch.float32))


def _winsorize(x: torch.Tensor, limit: float = 3.0) -> torch.Tensor:
    return torch.clamp(x, -limit, limit)


def _signed_power(x: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-12, power)


def _decay_linear(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    weights = torch.arange(1, window + 1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    return _rolling_apply(x, window, lambda t: (t * weights).sum(dim=-1))


def _where(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).to(x.dtype)
    return mask * x + (1.0 - mask) * y


def _corr(x: torch.Tensor, y: torch.Tensor, window: int = 5) -> torch.Tensor:
    x_mean = _ts_mean(x, window)
    y_mean = _ts_mean(y, window)
    cov = _ts_mean((x - x_mean) * (y - y_mean), window)
    x_std = _ts_std(x, window)
    y_std = _ts_std(y, window)
    return cov / (x_std * y_std + 1e-6)


def _cov(x: torch.Tensor, y: torch.Tensor, window: int = 5) -> torch.Tensor:
    x_mean = _ts_mean(x, window)
    y_mean = _ts_mean(y, window)
    return _ts_mean((x - x_mean) * (y - y_mean), window)


def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(torch.argsort(x, dim=0), dim=0).to(torch.float32)
    denom = max(x.shape[0] - 1, 1)
    return order / denom


def _cs_zscore(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std


def _cross_sectional_regression_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    beta = (x_centered * y_centered).sum(dim=0, keepdim=True) / ((y_centered**2).sum(dim=0, keepdim=True) + 1e-6)
    residual = x_centered - beta * y_centered
    return residual


def _neutralize(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _cross_sectional_regression_residual(x, y)


def _residualize(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _cross_sectional_regression_residual(x, y)


def _gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x > y).to(torch.float32)


def _lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x < y).to(torch.float32)


def _blend50(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + y)


def _scale2(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x


@lru_cache(maxsize=1)
def default_operator_registry() -> tuple[AlphaOperator, ...]:
    return (
        AlphaOperator("ADD", lambda x, y: x + y, 2, "arithmetic", input_types=(GENERIC_PANEL, GENERIC_PANEL)),
        AlphaOperator("SUB", lambda x, y: x - y, 2, "arithmetic", input_types=(GENERIC_PANEL, GENERIC_PANEL)),
        AlphaOperator("MUL", lambda x, y: x * y, 2, "arithmetic", input_types=(GENERIC_PANEL, GENERIC_PANEL)),
        AlphaOperator("DIV", lambda x, y: x / (y + 1e-6), 2, "arithmetic", input_types=(GENERIC_PANEL, GENERIC_PANEL)),
        AlphaOperator("BLEND50", _blend50, 2, "composition", input_types=(GENERIC_PANEL, GENERIC_PANEL), complexity=2),
        AlphaOperator("NEG", lambda x: -x, 1, "transform", input_types=(GENERIC_PANEL,)),
        AlphaOperator("ABS", torch.abs, 1, "transform", input_types=(GENERIC_PANEL,)),
        AlphaOperator("SIGN", torch.sign, 1, "transform", input_types=(GENERIC_PANEL,)),
        AlphaOperator("CLIP1", lambda x: _winsorize(x, 1.0), 1, "transform", input_types=(GENERIC_PANEL,)),
        AlphaOperator("CLIP3", lambda x: _winsorize(x, 3.0), 1, "transform", input_types=(GENERIC_PANEL,)),
        AlphaOperator("SCALE2", _scale2, 1, "composition", input_types=(GENERIC_PANEL,), complexity=2),
        AlphaOperator(
            "SIGNED_POWER2",
            lambda x: _signed_power(x, 2.0),
            1,
            "transform",
            input_types=(GENERIC_PANEL,),
            output_type=SCORE_PANEL,
            complexity=2,
        ),
        AlphaOperator("DELAY1", lambda x: _ts_delay(x, 1), 1, "timeseries", lookback=1, input_types=(GENERIC_PANEL,)),
        AlphaOperator("DELTA1", lambda x: _ts_delta(x, 1), 1, "timeseries", lookback=1, input_types=(GENERIC_PANEL,)),
        AlphaOperator("TS_MEAN5", lambda x: _ts_mean(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,)),
        AlphaOperator("TS_STD5", lambda x: _ts_std(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,), output_type=RISK_PANEL),
        AlphaOperator("TS_SUM5", lambda x: _ts_sum(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,)),
        AlphaOperator("TS_RANK5", lambda x: _ts_rank(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,), output_type=RANK_PANEL, complexity=2),
        AlphaOperator("TS_ZSCORE5", lambda x: _ts_zscore(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,), output_type=SCORE_PANEL, complexity=2),
        AlphaOperator("DECAY_LINEAR5", lambda x: _decay_linear(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,), complexity=2),
        AlphaOperator("ARGMAX5", lambda x: _ts_argmax(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,), output_type=RANK_PANEL, complexity=2),
        AlphaOperator("ARGMIN5", lambda x: _ts_argmin(x, 5), 1, "timeseries", lookback=5, input_types=(GENERIC_PANEL,), output_type=RANK_PANEL, complexity=2),
        AlphaOperator("CORR5", lambda x, y: _corr(x, y, 5), 2, "timeseries", lookback=5, input_types=(GENERIC_PANEL, GENERIC_PANEL), output_type=SCORE_PANEL, complexity=3),
        AlphaOperator("COV5", lambda x, y: _cov(x, y, 5), 2, "timeseries", lookback=5, input_types=(GENERIC_PANEL, GENERIC_PANEL), output_type=SCORE_PANEL, complexity=3),
        AlphaOperator("CS_RANK", _cs_rank, 1, "cross_sectional", input_types=(GENERIC_PANEL,), output_type=RANK_PANEL, complexity=2),
        AlphaOperator("CS_ZSCORE", _cs_zscore, 1, "cross_sectional", input_types=(GENERIC_PANEL,), output_type=SCORE_PANEL, complexity=2),
        AlphaOperator("WINSORIZE3", lambda x: _winsorize(x, 3.0), 1, "cross_sectional", input_types=(GENERIC_PANEL,), output_type=SCORE_PANEL),
        AlphaOperator("NEUTRALIZE", _neutralize, 2, "cross_sectional", input_types=(GENERIC_PANEL, GENERIC_PANEL), output_type=SCORE_PANEL, complexity=3),
        AlphaOperator("RESIDUALIZE", _residualize, 2, "cross_sectional", input_types=(GENERIC_PANEL, GENERIC_PANEL), output_type=SCORE_PANEL, complexity=3),
        AlphaOperator("GT", _gt, 2, "logical", input_types=(GENERIC_PANEL, GENERIC_PANEL), output_type=BOOLEAN_PANEL),
        AlphaOperator("LT", _lt, 2, "logical", input_types=(GENERIC_PANEL, GENERIC_PANEL), output_type=BOOLEAN_PANEL),
        AlphaOperator("WHERE", _where, 3, "logical", input_types=(BOOLEAN_PANEL, GENERIC_PANEL, GENERIC_PANEL), output_type=GENERIC_PANEL, complexity=2),
    )


def operator_name_list() -> list[str]:
    return [op.name for op in default_operator_registry()]


def operator_registry_by_token(feature_dim: int) -> dict[int, AlphaOperator]:
    return {feature_dim + i: op for i, op in enumerate(default_operator_registry())}


OPS_CONFIG = [asdict(op) for op in default_operator_registry()]

__all__ = [
    "ValueKind",
    "AxisKind",
    "SemanticKind",
    "TensorSpec",
    "GENERIC_PANEL",
    "PRICE_PANEL",
    "RETURN_PANEL",
    "VOLUME_PANEL",
    "LIQUIDITY_PANEL",
    "RISK_PANEL",
    "RANK_PANEL",
    "SCORE_PANEL",
    "BOOLEAN_PANEL",
    "AlphaOperator",
    "default_operator_registry",
    "operator_name_list",
    "operator_registry_by_token",
    "OPS_CONFIG",
]
