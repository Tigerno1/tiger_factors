from tiger_factors.multifactor_evaluation.screening import FactorSummaryTableConfig
from tiger_factors.multifactor_evaluation.screening import build_factor_summary_table as _build_factor_summary_table


def build_factor_summary_table(frame, *, config: FactorSummaryTableConfig | None = None):
    return _build_factor_summary_table(frame, config=config)


__all__ = [
    "FactorSummaryTableConfig",
    "build_factor_summary_table",
]

