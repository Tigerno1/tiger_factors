from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from tiger_factors.factor_frame.definition import FactorDefinition


@dataclass
class FactorDefinitionRegistry:
    """Lightweight registry for structured factor definitions."""

    definitions: dict[str, FactorDefinition] = field(default_factory=dict)

    def register(self, definition: FactorDefinition) -> "FactorDefinitionRegistry":
        self.definitions[str(definition.name)] = definition
        return self

    def add(self, definition: FactorDefinition) -> "FactorDefinitionRegistry":
        return self.register(definition)

    def register_many(self, *definitions: FactorDefinition) -> "FactorDefinitionRegistry":
        for definition in definitions:
            self.register(definition)
        return self

    def add_many(self, *definitions: FactorDefinition) -> "FactorDefinitionRegistry":
        return self.register_many(*definitions)

    def get(self, name: str) -> FactorDefinition:
        key = str(name)
        try:
            return self.definitions[key]
        except KeyError as exc:
            available = self.names()
            suggestions = difflib.get_close_matches(key, available, n=3, cutoff=0.45)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise KeyError(
                f"Unknown factor definition {name!r}. Available definitions: {available}.{suggestion_text}"
            ) from exc

    def build_factor(
        self,
        name: str,
        *,
        save: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        return self.get(name).to_factor(save=save, metadata=metadata)

    def names(self) -> tuple[str, ...]:
        return tuple(self.definitions.keys())

    def items(self):
        return self.definitions.items()

    def __contains__(self, name: str) -> bool:
        return str(name) in self.definitions

    def __len__(self) -> int:
        return len(self.definitions)


__all__ = ["FactorDefinitionRegistry"]
