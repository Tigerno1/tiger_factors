from __future__ import annotations

from dataclasses import dataclass, field

from tiger_factors.factor_ml.alpha_ops import SemanticKind, TensorSpec, default_operator_registry


@dataclass(frozen=True)
class FormulaConstraints:
    max_tokens: int = 32
    max_nodes: int = 32
    max_depth: int = 8
    min_operators: int = 1
    max_consecutive_unary: int = 4
    max_lookback: int = 20
    max_complexity: int = 64
    forbid_repeated_unary: bool = True


@dataclass(frozen=True)
class FormulaValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    operator_count: int = 0
    max_depth: int = 0
    max_lookback: int = 0
    complexity: int = 0


def _spec_matches(expected: TensorSpec, actual: TensorSpec) -> bool:
    kind_ok = expected.kind == actual.kind
    axis_ok = expected.axis == actual.axis or expected.axis.value == "any"
    semantic_ok = expected.semantic in {SemanticKind.GENERIC, actual.semantic}
    return kind_ok and axis_ok and semantic_ok


class FormulaValidator:
    def __init__(
        self,
        feature_dim: int,
        constraints: FormulaConstraints | None = None,
        feature_specs: list[TensorSpec] | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.constraints = constraints or FormulaConstraints()
        self.operators = {feature_dim + i: op for i, op in enumerate(default_operator_registry())}
        self.feature_specs = feature_specs or []

    def validate(self, formula_tokens: list[int]) -> FormulaValidationResult:
        errors: list[str] = []
        if len(formula_tokens) > self.constraints.max_tokens:
            errors.append(f"formula exceeds max_tokens={self.constraints.max_tokens}")
        if len(formula_tokens) > self.constraints.max_nodes:
            errors.append(f"formula exceeds max_nodes={self.constraints.max_nodes}")

        stack_depths: list[int] = []
        stack_specs: list[TensorSpec] = []
        operator_count = 0
        max_depth = 0
        max_lookback = 0
        consecutive_unary = 0
        complexity = 0
        previous_unary_token: int | None = None

        for token in formula_tokens:
            token = int(token)
            if token < self.feature_dim:
                stack_depths.append(1)
                if token < len(self.feature_specs):
                    stack_specs.append(self.feature_specs[token])
                else:
                    stack_specs.append(TensorSpec())
                consecutive_unary = 0
                previous_unary_token = None
                max_depth = max(max_depth, 1)
                continue

            op = self.operators.get(token)
            if op is None:
                errors.append(f"unknown token {token}")
                continue

            operator_count += 1
            max_lookback = max(max_lookback, op.lookback)
            complexity += op.complexity

            if len(stack_depths) < op.arity:
                errors.append(f"operator {op.name} underflow")
                continue

            args = [stack_depths.pop() for _ in range(op.arity)]
            arg_specs = [stack_specs.pop() for _ in range(op.arity)]
            node_depth = max(args) + 1
            stack_depths.append(node_depth)
            max_depth = max(max_depth, node_depth)
            for expected, actual in zip(op.input_types, arg_specs[::-1]):
                if not _spec_matches(expected, actual):
                    errors.append(
                        f"type mismatch for {op.name}: expected {expected.semantic.value}/{expected.axis.value}, "
                        f"got {actual.semantic.value}/{actual.axis.value}"
                    )
            stack_specs.append(op.output_type)

            if op.arity == 1:
                consecutive_unary += 1
                if self.constraints.forbid_repeated_unary and previous_unary_token == token:
                    errors.append(f"repeated unary operator chain detected for {op.name}")
                previous_unary_token = token
            else:
                consecutive_unary = 0
                previous_unary_token = None
            if consecutive_unary > self.constraints.max_consecutive_unary:
                errors.append(
                    f"too many consecutive unary operators (> {self.constraints.max_consecutive_unary})"
                )

        if operator_count < self.constraints.min_operators:
            errors.append(f"formula needs at least {self.constraints.min_operators} operators")
        if len(stack_depths) != 1:
            errors.append("formula must resolve to exactly one output")
        if max_depth > self.constraints.max_depth:
            errors.append(f"formula exceeds max_depth={self.constraints.max_depth}")
        if max_lookback > self.constraints.max_lookback:
            errors.append(f"formula exceeds max_lookback={self.constraints.max_lookback}")
        if complexity > self.constraints.max_complexity:
            errors.append(f"formula exceeds max_complexity={self.constraints.max_complexity}")

        return FormulaValidationResult(
            valid=not errors,
            errors=errors,
            operator_count=operator_count,
            max_depth=max_depth,
            max_lookback=max_lookback,
            complexity=complexity,
        )


__all__ = [
    "FormulaConstraints",
    "FormulaValidationResult",
    "FormulaValidator",
]
