from tiger_factors.multifactor_evaluation.reporting.portfolio import PositionReportResult
from tiger_factors.multifactor_evaluation.reporting.portfolio import create_position_report


def create_position_tear_sheet(*args, **kwargs):
    return create_position_report(*args, **kwargs)


__all__ = [
    "PositionReportResult",
    "create_position_report",
    "create_position_tear_sheet",
]
