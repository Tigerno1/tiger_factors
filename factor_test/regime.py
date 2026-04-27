from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.utils import TigerFactorData
from tiger_factors.factor_test.market_state import MarketStateResult
from tiger_factors.factor_test.market_state import market_state_test


@dataclass(frozen=True)
class FactorRegimeICResult:
    method: str
    period_column: str
    state_column: str
    table: pd.DataFrame
    current_state: str | None
    state_counts: dict[str, int]
    best_state: str | None
    worst_state: str | None
    ic_spread: float
    rank_ic_spread: float
    passed: bool
    market_state: MarketStateResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "state_column": self.state_column,
            "current_state": self.current_state,
            "state_counts": self.state_counts,
            "best_state": self.best_state,
            "worst_state": self.worst_state,
            "ic_spread": self.ic_spread,
            "rank_ic_spread": self.rank_ic_spread,
            "passed": self.passed,
            "table": self.table.to_dict(orient="records"),
            "market_state": self.market_state.to_dict(),
        }


@dataclass(frozen=True)
class FactorRegimeStabilityResult:
    method: str
    period_column: str
    current_state: str | None
    historical_table: pd.DataFrame
    regime_ic: FactorRegimeICResult
    market_state: MarketStateResult
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "current_state": self.current_state,
            "passed": self.passed,
            "historical_table": self.historical_table.to_dict(orient="records"),
            "regime_ic": self.regime_ic.to_dict(),
            "market_state": self.market_state.to_dict(),
        }


@dataclass(frozen=True)
class FactorRegimeDecayResult:
    method: str
    period_column: str
    state_column: str
    table: pd.DataFrame
    transition_table: pd.DataFrame
    current_state: str | None
    state_counts: dict[str, int]
    decay_ratio: float
    rank_decay_ratio: float
    decay_score: float
    passed: bool
    market_state: MarketStateResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "state_column": self.state_column,
            "current_state": self.current_state,
            "state_counts": self.state_counts,
            "decay_ratio": self.decay_ratio,
            "rank_decay_ratio": self.rank_decay_ratio,
            "decay_score": self.decay_score,
            "passed": self.passed,
            "table": self.table.to_dict(orient="records"),
            "transition_table": self.transition_table.to_dict(orient="records"),
            "market_state": self.market_state.to_dict(),
        }


@dataclass(frozen=True)
class FactorRegimeTurningPointResult:
    method: str
    period_column: str
    state_column: str
    table: pd.DataFrame
    transition_table: pd.DataFrame
    current_state: str | None
    state_counts: dict[str, int]
    turning_point_ratio: float
    rank_turning_point_ratio: float
    sign_flip_ratio: float
    passed: bool
    market_state: MarketStateResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "state_column": self.state_column,
            "current_state": self.current_state,
            "state_counts": self.state_counts,
            "turning_point_ratio": self.turning_point_ratio,
            "rank_turning_point_ratio": self.rank_turning_point_ratio,
            "sign_flip_ratio": self.sign_flip_ratio,
            "passed": self.passed,
            "table": self.table.to_dict(orient="records"),
            "transition_table": self.transition_table.to_dict(orient="records"),
            "market_state": self.market_state.to_dict(),
        }


@dataclass(frozen=True)
class FactorRegimeReportResult:
    method: str
    period_column: str
    market_state: MarketStateResult
    regime_ic: FactorRegimeICResult
    decay: FactorRegimeDecayResult
    turning_point: FactorRegimeTurningPointResult
    stability: FactorRegimeStabilityResult
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "passed": self.passed,
            "market_state": self.market_state.to_dict(),
            "regime_ic": self.regime_ic.to_dict(),
            "decay": self.decay.to_dict(),
            "turning_point": self.turning_point.to_dict(),
            "stability": self.stability.to_dict(),
        }


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    joined = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    x = joined.iloc[:, 0].to_numpy(dtype=float)
    y = joined.iloc[:, 1].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def _forward_return_columns(frame: pd.DataFrame) -> list[str]:
    cols = [str(col) for col in frame.columns if str(col) not in {"factor", "date_", "code"}]
    return cols


def _daily_regime_ic(
    factor_data: pd.DataFrame,
    *,
    period_column: str,
    state_table: pd.DataFrame,
    state_column: str = "state",
) -> pd.DataFrame:
    frame = factor_data.copy()
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 2:
        if {"date_", "code"}.issubset(frame.columns):
            frame = frame.set_index(["date_", "code"]).sort_index()
        else:
            raise ValueError("factor_data must be indexed by date and code or contain date_ / code columns")

    if period_column not in frame.columns:
        raise KeyError(f"Missing forward return column {period_column!r}")

    factor = pd.to_numeric(frame["factor"], errors="coerce")
    forward = pd.to_numeric(frame[period_column], errors="coerce")
    date_index = pd.Index(frame.index.get_level_values(0), name="date_")

    def _group_corr(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "ic": _safe_corr(group["factor"], group[period_column], method="pearson"),
                "rank_ic": _safe_corr(group["factor"], group[period_column], method="spearman"),
            }
        )

    daily = pd.concat([factor.rename("factor"), forward.rename(period_column)], axis=1)
    daily = daily.groupby(date_index, sort=True).apply(_group_corr).reset_index()
    merged = daily.merge(state_table[["date_", state_column]], on="date_", how="left")
    return merged.dropna(subset=[state_column])


def _transition_window_table(
    daily: pd.DataFrame,
    market_state: MarketStateResult,
    *,
    pre_window: int = 5,
    post_window: int = 5,
) -> pd.DataFrame:
    state_table = market_state.table.copy()
    state_table["date_"] = pd.to_datetime(state_table["date_"], errors="coerce")
    state_table = state_table.dropna(subset=["date_"]).sort_values("date_").reset_index(drop=True)
    if state_table.empty or len(state_table) < 2:
        return pd.DataFrame(
            columns=[
                "date_",
                "from_state",
                "to_state",
                "pre_ic_mean",
                "post_ic_mean",
                "pre_rank_ic_mean",
                "post_rank_ic_mean",
                "ic_delta",
                "rank_ic_delta",
                "ic_ratio",
                "rank_ic_ratio",
                "sign_flip",
            ]
        )

    daily = daily.copy()
    daily["date_"] = pd.to_datetime(daily["date_"], errors="coerce")
    daily = daily.dropna(subset=["date_"]).sort_values("date_").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    pre_window = int(max(pre_window, 1))
    post_window = int(max(post_window, 1))
    for idx in range(1, len(state_table)):
        prev_row = state_table.iloc[idx - 1]
        curr_row = state_table.iloc[idx]
        from_state = str(prev_row["state"])
        to_state = str(curr_row["state"])
        if from_state == to_state:
            continue
        cut_date = pd.Timestamp(curr_row["date_"])
        pre_slice = daily[daily["date_"] < cut_date].tail(pre_window)
        post_slice = daily[daily["date_"] >= cut_date].head(post_window)
        if pre_slice.empty or post_slice.empty:
            continue
        pre_ic_mean = float(pd.to_numeric(pre_slice["ic"], errors="coerce").dropna().mean())
        post_ic_mean = float(pd.to_numeric(post_slice["ic"], errors="coerce").dropna().mean())
        pre_rank_ic_mean = float(pd.to_numeric(pre_slice["rank_ic"], errors="coerce").dropna().mean())
        post_rank_ic_mean = float(pd.to_numeric(post_slice["rank_ic"], errors="coerce").dropna().mean())
        ic_ratio = float(post_ic_mean / pre_ic_mean) if abs(pre_ic_mean) > 1e-12 else float("nan")
        rank_ic_ratio = float(post_rank_ic_mean / pre_rank_ic_mean) if abs(pre_rank_ic_mean) > 1e-12 else float("nan")
        sign_flip = bool(
            (abs(pre_ic_mean) > 1e-12 and abs(post_ic_mean) > 1e-12 and np.sign(pre_ic_mean) != np.sign(post_ic_mean))
            or (abs(pre_rank_ic_mean) > 1e-12 and abs(post_rank_ic_mean) > 1e-12 and np.sign(pre_rank_ic_mean) != np.sign(post_rank_ic_mean))
        )
        records.append(
            {
                "date_": cut_date,
                "from_state": from_state,
                "to_state": to_state,
                "pre_ic_mean": pre_ic_mean,
                "post_ic_mean": post_ic_mean,
                "pre_rank_ic_mean": pre_rank_ic_mean,
                "post_rank_ic_mean": post_rank_ic_mean,
                "ic_delta": float(post_ic_mean - pre_ic_mean),
                "rank_ic_delta": float(post_rank_ic_mean - pre_rank_ic_mean),
                "ic_ratio": ic_ratio,
                "rank_ic_ratio": rank_ic_ratio,
                "sign_flip": sign_flip,
            }
        )
    return pd.DataFrame(records)


def factor_regime_ic_test(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    use_hmm: bool = False,
    n_states: int = 3,
    market_state_kwargs: dict[str, Any] | None = None,
    min_current_ic_mean: float = 0.0,
    min_current_rank_ic_mean: float = 0.0,
) -> FactorRegimeICResult:
    market_state_kwargs = dict(market_state_kwargs or {})
    market_state = market_state_test(
        factor_data.prices,
        n_states=n_states,
        use_hmm=use_hmm,
        **market_state_kwargs,
    )
    frame = factor_data.factor_data.copy()
    periods = _forward_return_columns(frame)
    if not periods:
        empty = pd.DataFrame(columns=["state", "ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "n_obs"])
        return FactorRegimeICResult(
            method="regime_ic",
            period_column="",
            state_column="state",
            table=empty,
            current_state=market_state.current_state,
            state_counts=market_state.state_counts,
            best_state=None,
            worst_state=None,
            ic_spread=0.0,
            rank_ic_spread=0.0,
            passed=False,
            market_state=market_state,
        )

    chosen_period = str(period) if period is not None else str(periods[0])
    if chosen_period not in frame.columns:
        chosen_period = str(periods[0])

    daily = _daily_regime_ic(frame, period_column=chosen_period, state_table=market_state.table)
    if daily.empty:
        empty = pd.DataFrame(columns=["state", "ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "n_obs"])
        return FactorRegimeICResult(
            method="regime_ic",
            period_column=chosen_period,
            state_column="state",
            table=empty,
            current_state=market_state.current_state,
            state_counts=market_state.state_counts,
            best_state=None,
            worst_state=None,
            ic_spread=0.0,
            rank_ic_spread=0.0,
            passed=False,
            market_state=market_state,
        )

    records: list[dict[str, Any]] = []
    for state, group in daily.groupby("state", dropna=False):
        ic = pd.to_numeric(group["ic"], errors="coerce").dropna()
        rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        ic_mean = float(ic.mean()) if not ic.empty else 0.0
        rank_ic_mean = float(rank_ic.mean()) if not rank_ic.empty else 0.0
        ic_ir = float(ic_mean / (ic.std(ddof=0) + 1e-6)) if not ic.empty else 0.0
        rank_ic_ir = float(rank_ic_mean / (rank_ic.std(ddof=0) + 1e-6)) if not rank_ic.empty else 0.0
        records.append(
            {
                "state": str(state),
                "ic_mean": ic_mean,
                "rank_ic_mean": rank_ic_mean,
                "ic_ir": ic_ir,
                "rank_ic_ir": rank_ic_ir,
                "n_obs": int(len(group)),
            }
        )

    table = pd.DataFrame(records).sort_values(["ic_mean", "rank_ic_mean"], ascending=False).reset_index(drop=True)
    if table.empty:
        best_state = None
        worst_state = None
        ic_spread = 0.0
        rank_ic_spread = 0.0
    else:
        best_state = str(table.iloc[0]["state"])
        worst_state = str(table.iloc[-1]["state"])
        ic_spread = float(table.iloc[0]["ic_mean"] - table.iloc[-1]["ic_mean"])
        rank_ic_spread = float(table.iloc[0]["rank_ic_mean"] - table.iloc[-1]["rank_ic_mean"])

    current_row = table.loc[table["state"].eq(str(market_state.current_state))] if market_state.current_state is not None else pd.DataFrame()
    current_ic_mean = float(current_row["ic_mean"].iloc[0]) if not current_row.empty else 0.0
    current_rank_ic_mean = float(current_row["rank_ic_mean"].iloc[0]) if not current_row.empty else 0.0
    passed = (
        abs(current_ic_mean) >= float(min_current_ic_mean)
        and abs(current_rank_ic_mean) >= float(min_current_rank_ic_mean)
    )

    return FactorRegimeICResult(
        method="regime_ic",
        period_column=chosen_period,
        state_column="state",
        table=table,
        current_state=market_state.current_state,
        state_counts=market_state.state_counts,
        best_state=best_state,
        worst_state=worst_state,
        ic_spread=ic_spread,
        rank_ic_spread=rank_ic_spread,
        passed=passed,
        market_state=market_state,
    )


def factor_regime_stability_test(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    use_hmm: bool = False,
    n_states: int = 3,
    market_state_kwargs: dict[str, Any] | None = None,
    min_current_ic_mean: float = 0.0,
    min_current_rank_ic_mean: float = 0.0,
    min_ic_spread: float = 0.0,
    min_rank_ic_spread: float = 0.0,
) -> FactorRegimeStabilityResult:
    regime_ic = factor_regime_ic_test(
        factor_data,
        period=period,
        use_hmm=use_hmm,
        n_states=n_states,
        market_state_kwargs=market_state_kwargs,
        min_current_ic_mean=min_current_ic_mean,
        min_current_rank_ic_mean=min_current_rank_ic_mean,
    )
    historical_table = regime_ic.table.copy()
    spread_ok = (
        abs(regime_ic.ic_spread) >= float(min_ic_spread)
        or abs(regime_ic.rank_ic_spread) >= float(min_rank_ic_spread)
        or len(historical_table) <= 1
    )
    passed = bool(regime_ic.passed) and spread_ok
    return FactorRegimeStabilityResult(
        method="regime_stability",
        period_column=regime_ic.period_column,
        current_state=regime_ic.current_state,
        historical_table=historical_table,
        regime_ic=regime_ic,
        market_state=regime_ic.market_state,
        passed=passed,
    )


def factor_regime_decay_test(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    use_hmm: bool = False,
    n_states: int = 3,
    market_state_kwargs: dict[str, Any] | None = None,
    pre_window: int = 5,
    post_window: int = 5,
    min_decay_ratio: float = 0.5,
    min_rank_decay_ratio: float = 0.5,
) -> FactorRegimeDecayResult:
    market_state_kwargs = dict(market_state_kwargs or {})
    market_state = market_state_test(
        factor_data.prices,
        n_states=n_states,
        use_hmm=use_hmm,
        **market_state_kwargs,
    )
    frame = factor_data.factor_data.copy()
    periods = _forward_return_columns(frame)
    if not periods:
        empty = pd.DataFrame(columns=["state", "ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "n_obs"])
        transition_empty = _transition_window_table(pd.DataFrame(columns=["date_", "ic", "rank_ic"]), market_state)
        return FactorRegimeDecayResult(
            method="regime_decay",
            period_column="",
            state_column="state",
            table=empty,
            transition_table=transition_empty,
            current_state=market_state.current_state,
            state_counts=market_state.state_counts,
            decay_ratio=0.0,
            rank_decay_ratio=0.0,
            decay_score=0.0,
            passed=False,
            market_state=market_state,
        )

    chosen_period = str(period) if period is not None else str(periods[0])
    if chosen_period not in frame.columns:
        chosen_period = str(periods[0])

    daily = _daily_regime_ic(frame, period_column=chosen_period, state_table=market_state.table)
    if daily.empty:
        empty = pd.DataFrame(columns=["state", "ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "n_obs"])
        transition_empty = _transition_window_table(pd.DataFrame(columns=["date_", "ic", "rank_ic"]), market_state)
        return FactorRegimeDecayResult(
            method="regime_decay",
            period_column=chosen_period,
            state_column="state",
            table=empty,
            transition_table=transition_empty,
            current_state=market_state.current_state,
            state_counts=market_state.state_counts,
            decay_ratio=0.0,
            rank_decay_ratio=0.0,
            decay_score=0.0,
            passed=False,
            market_state=market_state,
        )

    records: list[dict[str, Any]] = []
    for state, group in daily.groupby("state", dropna=False):
        ic = pd.to_numeric(group["ic"], errors="coerce").dropna()
        rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        ic_mean = float(ic.mean()) if not ic.empty else 0.0
        rank_ic_mean = float(rank_ic.mean()) if not rank_ic.empty else 0.0
        ic_ir = float(ic_mean / (ic.std(ddof=0) + 1e-6)) if not ic.empty else 0.0
        rank_ic_ir = float(rank_ic_mean / (rank_ic.std(ddof=0) + 1e-6)) if not rank_ic.empty else 0.0
        records.append(
            {
                "state": str(state),
                "ic_mean": ic_mean,
                "rank_ic_mean": rank_ic_mean,
                "ic_ir": ic_ir,
                "rank_ic_ir": rank_ic_ir,
                "n_obs": int(len(group)),
            }
        )

    table = pd.DataFrame(records).sort_values(["ic_mean", "rank_ic_mean"], ascending=False).reset_index(drop=True)
    transition_table = _transition_window_table(daily, market_state, pre_window=pre_window, post_window=post_window)
    if transition_table.empty:
        decay_ratio = 1.0
        rank_decay_ratio = 1.0
    else:
        decay_ratio = float(np.nanmean(np.abs(transition_table["ic_ratio"].to_numpy(dtype=float))))
        rank_decay_ratio = float(np.nanmean(np.abs(transition_table["rank_ic_ratio"].to_numpy(dtype=float))))
        if not np.isfinite(decay_ratio):
            decay_ratio = 0.0
        if not np.isfinite(rank_decay_ratio):
            rank_decay_ratio = 0.0
    decay_score = float(np.clip(0.5 * decay_ratio + 0.5 * rank_decay_ratio, 0.0, 1.0))
    passed = decay_ratio >= float(min_decay_ratio) and rank_decay_ratio >= float(min_rank_decay_ratio)
    return FactorRegimeDecayResult(
        method="regime_decay",
        period_column=chosen_period,
        state_column="state",
        table=table,
        transition_table=transition_table,
        current_state=market_state.current_state,
        state_counts=market_state.state_counts,
        decay_ratio=decay_ratio,
        rank_decay_ratio=rank_decay_ratio,
        decay_score=decay_score,
        passed=passed,
        market_state=market_state,
    )


def factor_regime_turning_point_test(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    use_hmm: bool = False,
    n_states: int = 3,
    market_state_kwargs: dict[str, Any] | None = None,
    pre_window: int = 5,
    post_window: int = 5,
    min_sign_flip_ratio: float = 0.25,
    min_abs_turning_ratio: float = 0.5,
) -> FactorRegimeTurningPointResult:
    decay_result = factor_regime_decay_test(
        factor_data,
        period=period,
        use_hmm=use_hmm,
        n_states=n_states,
        market_state_kwargs=market_state_kwargs,
        pre_window=pre_window,
        post_window=post_window,
        min_decay_ratio=0.0,
        min_rank_decay_ratio=0.0,
    )
    transition_table = decay_result.transition_table.copy()
    if transition_table.empty:
        empty = pd.DataFrame(
            columns=[
                "date_",
                "from_state",
                "to_state",
                "pre_ic_mean",
                "post_ic_mean",
                "pre_rank_ic_mean",
                "post_rank_ic_mean",
                "ic_delta",
                "rank_ic_delta",
                "ic_ratio",
                "rank_ic_ratio",
                "sign_flip",
            ]
        )
        return FactorRegimeTurningPointResult(
            method="regime_turning_point",
            period_column=decay_result.period_column,
            state_column="state",
            table=empty,
            transition_table=transition_table,
            current_state=decay_result.current_state,
            state_counts=decay_result.state_counts,
            turning_point_ratio=0.0,
            rank_turning_point_ratio=0.0,
            sign_flip_ratio=0.0,
            passed=False,
            market_state=decay_result.market_state,
        )

    sign_flip_ratio = float(transition_table["sign_flip"].mean()) if not transition_table.empty else 0.0
    turning_point_ratio = float(np.nanmean(np.abs(transition_table["ic_delta"].to_numpy(dtype=float))))
    rank_turning_point_ratio = float(np.nanmean(np.abs(transition_table["rank_ic_delta"].to_numpy(dtype=float))))
    if not np.isfinite(turning_point_ratio):
        turning_point_ratio = 0.0
    if not np.isfinite(rank_turning_point_ratio):
        rank_turning_point_ratio = 0.0
    passed = (
        sign_flip_ratio >= float(min_sign_flip_ratio)
        or turning_point_ratio >= float(min_abs_turning_ratio)
        or rank_turning_point_ratio >= float(min_abs_turning_ratio)
    )
    return FactorRegimeTurningPointResult(
        method="regime_turning_point",
        period_column=decay_result.period_column,
        state_column="state",
        table=transition_table,
        transition_table=transition_table,
        current_state=decay_result.current_state,
        state_counts=decay_result.state_counts,
        turning_point_ratio=turning_point_ratio,
        rank_turning_point_ratio=rank_turning_point_ratio,
        sign_flip_ratio=sign_flip_ratio,
        passed=passed,
        market_state=decay_result.market_state,
    )


def factor_regime_report(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    use_hmm: bool = False,
    n_states: int = 3,
    market_state_kwargs: dict[str, Any] | None = None,
    pre_window: int = 5,
    post_window: int = 5,
    min_current_ic_mean: float = 0.0,
    min_current_rank_ic_mean: float = 0.0,
    min_ic_spread: float = 0.0,
    min_rank_ic_spread: float = 0.0,
    min_decay_ratio: float = 0.5,
    min_rank_decay_ratio: float = 0.5,
    min_sign_flip_ratio: float = 0.25,
    min_abs_turning_ratio: float = 0.5,
) -> FactorRegimeReportResult:
    market_state_kwargs = dict(market_state_kwargs or {})
    market_state = market_state_test(
        factor_data.prices,
        n_states=n_states,
        use_hmm=use_hmm,
        **market_state_kwargs,
    )
    regime_ic = factor_regime_ic_test(
        factor_data,
        period=period,
        use_hmm=use_hmm,
        n_states=n_states,
        market_state_kwargs=market_state_kwargs,
        min_current_ic_mean=min_current_ic_mean,
        min_current_rank_ic_mean=min_current_rank_ic_mean,
    )
    decay = factor_regime_decay_test(
        factor_data,
        period=period,
        use_hmm=use_hmm,
        n_states=n_states,
        market_state_kwargs=market_state_kwargs,
        pre_window=pre_window,
        post_window=post_window,
        min_decay_ratio=min_decay_ratio,
        min_rank_decay_ratio=min_rank_decay_ratio,
    )
    turning_point = factor_regime_turning_point_test(
        factor_data,
        period=period,
        use_hmm=use_hmm,
        n_states=n_states,
        market_state_kwargs=market_state_kwargs,
        pre_window=pre_window,
        post_window=post_window,
        min_sign_flip_ratio=min_sign_flip_ratio,
        min_abs_turning_ratio=min_abs_turning_ratio,
    )
    stability = factor_regime_stability_test(
        factor_data,
        period=period,
        use_hmm=use_hmm,
        n_states=n_states,
        market_state_kwargs=market_state_kwargs,
        min_current_ic_mean=min_current_ic_mean,
        min_current_rank_ic_mean=min_current_rank_ic_mean,
        min_ic_spread=min_ic_spread,
        min_rank_ic_spread=min_rank_ic_spread,
    )
    passed = all([regime_ic.passed, decay.passed, turning_point.passed, stability.passed])
    return FactorRegimeReportResult(
        method="regime_report",
        period_column=regime_ic.period_column or decay.period_column or stability.period_column,
        market_state=market_state,
        regime_ic=regime_ic,
        decay=decay,
        turning_point=turning_point,
        stability=stability,
        passed=passed,
    )


__all__ = [
    "FactorRegimeICResult",
    "FactorRegimeDecayResult",
    "FactorRegimeStabilityResult",
    "FactorRegimeTurningPointResult",
    "FactorRegimeReportResult",
    "factor_regime_ic_test",
    "factor_regime_decay_test",
    "factor_regime_stability_test",
    "factor_regime_turning_point_test",
    "factor_regime_report",
]
