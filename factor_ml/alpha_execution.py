from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch

from tiger_factors.factor_ml.alpha_constraints import FormulaConstraints, FormulaValidator
from tiger_factors.factor_ml.alpha_ops import AlphaOperator, TensorSpec, default_operator_registry


@dataclass(frozen=True)
class ExecutionNode:
    key: str
    kind: str
    token: int
    name: str
    children: tuple["ExecutionNode", ...] = ()


@dataclass(frozen=True)
class CompiledFormula:
    root: ExecutionNode
    feature_dim: int
    token_count: int


def _node_key(kind: str, name: str, children: tuple[ExecutionNode, ...]) -> str:
    payload = f"{kind}:{name}:{','.join(child.key for child in children)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class AlphaDAGExecutor:
    def __init__(
        self,
        feature_dim: int,
        *,
        constraints: FormulaConstraints | None = None,
        operators: tuple[AlphaOperator, ...] | None = None,
        feature_specs: list[TensorSpec] | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.constraints = constraints or FormulaConstraints()
        self.feature_specs = feature_specs or []
        self.validator = FormulaValidator(feature_dim, self.constraints, feature_specs=self.feature_specs)
        self.operators = operators or default_operator_registry()
        self.operator_map = {feature_dim + i: op for i, op in enumerate(self.operators)}
        self._compile_cache: dict[tuple[int, ...], CompiledFormula] = {}

    def compile(self, formula_tokens: list[int]) -> CompiledFormula:
        token_key = tuple(int(token) for token in formula_tokens)
        cached = self._compile_cache.get(token_key)
        if cached is not None:
            return cached

        result = self.validator.validate(list(token_key))
        if not result.valid:
            raise ValueError("; ".join(result.errors))

        stack: list[ExecutionNode] = []
        dedup: dict[str, ExecutionNode] = {}
        for token in token_key:
            if token < self.feature_dim:
                node = ExecutionNode(
                    key=_node_key("feature", f"f{token}", ()),
                    kind="feature",
                    token=token,
                    name=f"f{token}",
                )
            else:
                op = self.operator_map[token]
                children = tuple(stack.pop() for _ in range(op.arity))[::-1]
                node = ExecutionNode(
                    key=_node_key("op", op.name, children),
                    kind="op",
                    token=token,
                    name=op.name,
                    children=children,
                )
            node = dedup.setdefault(node.key, node)
            stack.append(node)

        compiled = CompiledFormula(root=stack[0], feature_dim=self.feature_dim, token_count=len(token_key))
        self._compile_cache[token_key] = compiled
        return compiled

    def compile_many(self, formulas: list[list[int]]) -> list[CompiledFormula]:
        return [self.compile(tokens) for tokens in formulas]

    def evaluate(
        self,
        formula_tokens: list[int] | CompiledFormula,
        feat_tensor: torch.Tensor,
        *,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        compiled = formula_tokens if isinstance(formula_tokens, CompiledFormula) else self.compile(formula_tokens)
        memo = cache if cache is not None else {}
        return self._evaluate_node(compiled.root, feat_tensor, memo)

    def evaluate_many(
        self,
        formulas: list[list[int] | CompiledFormula],
        feat_tensor: torch.Tensor,
        *,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        compiled = [item if isinstance(item, CompiledFormula) else self.compile(item) for item in formulas]
        memo = cache if cache is not None else {}
        return [self._evaluate_node(item.root, feat_tensor, memo) for item in compiled]

    def _evaluate_node(self, node: ExecutionNode, feat_tensor: torch.Tensor, memo: dict[str, torch.Tensor]) -> torch.Tensor:
        cached = memo.get(node.key)
        if cached is not None:
            return cached
        if node.kind == "feature":
            value = feat_tensor[:, node.token, :]
        else:
            operator = self.operator_map[node.token]
            args = [self._evaluate_node(child, feat_tensor, memo) for child in node.children]
            value = operator.func(*args)
            if torch.isnan(value).any() or torch.isinf(value).any():
                value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
        memo[node.key] = value
        return value


__all__ = [
    "ExecutionNode",
    "CompiledFormula",
    "AlphaDAGExecutor",
]
