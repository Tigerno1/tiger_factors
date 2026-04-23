from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tiger_factors.factor_evaluation.input import TableLike
from tiger_factors.factor_evaluation.input import load_factor_frame
from tiger_factors.factor_evaluation.input import load_price_frame
from tiger_factors.factor_evaluation.utils import factor_frame_to_series
from tiger_factors.factor_evaluation.utils import price_frame_to_wide


@dataclass(frozen=True)
class AlphalensInput:
    factor_frame: pd.DataFrame
    price_frame: pd.DataFrame
    factor_series: pd.Series
    prices: pd.DataFrame
    factor_column: str
    date_column: str = "date_"
    code_column: str = "code"
    price_column: str = "close"


def build_alphalens_input(
    *,
    factor_frame: TableLike,
    price_frame: TableLike,
    factor_column: str,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> AlphalensInput:
    factor_frame_df = load_factor_frame(
        factor_frame,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
    )
    price_frame_df = load_price_frame(
        price_frame,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )
    factor_series = factor_frame_to_series(
        factor_frame_df,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
    )
    prices = price_frame_to_wide(
        price_frame_df,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )
    return AlphalensInput(
        factor_frame=factor_frame_df,
        price_frame=price_frame_df,
        factor_series=factor_series,
        prices=prices,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )


__all__ = [
    "AlphalensInput",
    "build_alphalens_input",
]
