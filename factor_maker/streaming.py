from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiger_execution import create_array_manager, make_bar


@dataclass
class StreamingFactorEngine:
    buffer_size: int = 256
    additional_status: bool = True
    prefer_rust: bool = True

    def __post_init__(self) -> None:
        self.array_manager = create_array_manager(
            self.buffer_size,
            additional_status=self.additional_status,
            prefer_rust=self.prefer_rust,
        )

    def update_bar(
        self,
        *,
        datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
        open_interest: float = 0.0,
    ) -> None:
        bar = make_bar(
            datetime=datetime,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            open_interest=open_interest,
        )
        self.array_manager.update_bar(bar)

    def latest_snapshot(self) -> dict[str, float]:
        close = np.asarray(self.array_manager.close, dtype=float)
        returns = np.asarray(self.array_manager.returns, dtype=float)
        volatility = np.asarray(self.array_manager.volatility, dtype=float)
        amplitude = np.asarray(self.array_manager.amplitude, dtype=float)
        wt = np.asarray(self.array_manager.wt, dtype=float)
        return {
            "close": float(close[-1]) if len(close) else 0.0,
            "return": float(returns[-1]) if len(returns) else 0.0,
            "volatility": float(volatility[-1]) if len(volatility) else 0.0,
            "amplitude": float(amplitude[-1]) if len(amplitude) else 0.0,
            "wt": float(wt[-1]) if len(wt) else 0.0,
        }


__all__ = ["StreamingFactorEngine"]
