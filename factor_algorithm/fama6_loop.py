from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tiger_factors.factor_store import FactorResult, TigerFactorLibrary, to_long_factor
from tiger_factors.utils.combine import (
    blend_factor_panels,
    factor_correlation_matrix,
    greedy_select_by_correlation,
)
from tiger_factors.utils.scoring import score_factor_panels, summarize_factor_evaluation
from tiger_factors.utils.weighting import score_to_weights


@dataclass(frozen=True)
class Fama6Artifacts:
    single_factor_results: dict[str, FactorResult]
    multi_factor_result: FactorResult
    diagnostics_path: Path


def _asof_align(
    library: TigerFactorLibrary,
    fundamentals: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    columns: list[str],
    codes: list[str],
) -> dict[str, pd.DataFrame]:
    aligned = library.align_fundamental_to_trading_dates(
        fundamentals,
        trading_dates,
        value_columns=columns,
        use_point_in_time=True,
        lag_sessions=1,
    )
    return {name: frame.reindex(index=trading_dates, columns=codes) for name, frame in aligned.items()}


def build_fama6_loop(
    *,
    codes: list[str],
    start: str,
    end: str,
    output_dir: str | Path,
    price_provider: str = "yahoo",
    forward_days: int = 21,
    corr_threshold: float = 0.75,
    verbose: bool = True,
    as_ex: bool | None = None,
) -> Fama6Artifacts:
    library = TigerFactorLibrary(
        output_dir=output_dir,
        price_provider=price_provider,
        verbose=verbose,
    )

    buffer_start = str((pd.Timestamp(start) - pd.Timedelta(days=750)).date())
    close = library.price_panel(codes=codes, start=buffer_start, end=end, provider=price_provider, field="close", as_ex=as_ex)
    close = close.reindex(columns=codes).sort_index()
    trading_dates = pd.DatetimeIndex(close.index)

    if close.empty:
        raise ValueError("No price data fetched; cannot build factors.")

    balance = library.fetch_fundamental_data(
        name="balance_sheet",
        freq="1q",
        start=buffer_start,
        end=end,
        codes=codes,
        as_ex=as_ex,
    )
    income = library.fetch_fundamental_data(
        name="income_statement",
        freq="1q",
        start=buffer_start,
        end=end,
        codes=codes,
        as_ex=as_ex,
    )

    bal = _asof_align(library, balance, trading_dates, ["shares_basic", "total_assets", "total_equity"], codes)
    inc = _asof_align(library, income, trading_dates, ["net_income"], codes)

    shares = bal["shares_basic"]
    total_assets = bal["total_assets"]
    total_equity = bal["total_equity"]
    net_income = inc["net_income"]

    mcap = close * shares
    log_mcap = np.log(mcap.replace(0, np.nan))

    returns = close.pct_change()
    market_return = returns.mean(axis=1)

    # Fama6-style proxies: MKT beta, SMB(size), HML(value), RMW(profitability), CMA(investment), MOM(momentum)
    rolling_cov = returns.apply(lambda s: s.rolling(126, min_periods=63).cov(market_return), axis=0)
    rolling_var = market_return.rolling(126, min_periods=63).var()
    beta = rolling_cov.div(rolling_var, axis=0)
    size = -log_mcap
    value = total_equity.div(mcap.replace(0, np.nan))
    profitability = net_income.div(total_equity.replace(0, np.nan))
    investment = -total_assets.pct_change(252)
    momentum = close.pct_change(252) - close.pct_change(21)

    raw_factors = {
        "fama6_beta": beta,
        "fama6_size": size,
        "fama6_value": value,
        "fama6_profitability": profitability,
        "fama6_investment": investment,
        "fama6_momentum": momentum,
    }

    # Restrict output window.
    start_ts = pd.Timestamp(start)
    for key in list(raw_factors.keys()):
        raw_factors[key] = raw_factors[key].loc[raw_factors[key].index >= start_ts]

    fwd_ret = close.pct_change(forward_days).shift(-forward_days)
    fwd_ret = fwd_ret.loc[fwd_ret.index >= start_ts]

    single_results: dict[str, FactorResult] = {}
    factor_scores, eval_summary = score_factor_panels(raw_factors, fwd_ret, score_field="fitness")

    for name, panel in raw_factors.items():
        panel = panel.reindex(columns=codes)
        long_df = to_long_factor(panel, name)
        single_results[name] = library.save_factor(
            factor_name=name,
            factor_df=long_df,
            metadata={
                "family": "fama6_proxy_single",
                "forward_days": forward_days,
                **eval_summary[name],
            },
        )

    corr = factor_correlation_matrix(raw_factors)
    selected = greedy_select_by_correlation(factor_scores, corr, corr_threshold)
    if not selected:
        raise ValueError("No factor selected for multi-factor composition.")

    factor_weights = score_to_weights(factor_scores, selected=selected, method="positive")
    multi_panel = blend_factor_panels(raw_factors, factor_weights, standardize=True)
    multi_long = to_long_factor(multi_panel, "fama6_multi_factor")
    multi_eval = summarize_factor_evaluation(multi_panel, fwd_ret)
    multi_result = library.save_factor(
        factor_name="fama6_multi_factor",
        factor_df=multi_long,
        metadata={
            "family": "fama6_proxy_multi",
            "selected_factors": selected,
            "factor_weights": factor_weights,
            "corr_threshold": corr_threshold,
            "forward_days": forward_days,
            "fitness": float(multi_eval["fitness"]),
            "ic_mean": float(multi_eval["ic_mean"]),
            "rank_ic_mean": float(multi_eval["rank_ic_mean"]),
            "turnover": float(multi_eval["turnover"]),
        },
    )

    diag_dir = Path(output_dir) / "fama6_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diag_dir / "fama6_summary.json"
    diagnostics = {
        "codes": codes,
        "start": start,
        "end": end,
        "forward_days": forward_days,
        "corr_threshold": corr_threshold,
        "single_factor_scores": factor_scores,
        "selected_factors": selected,
        "factor_weights": factor_weights,
        "single_factor_eval": eval_summary,
        "multi_factor_eval": {
            "fitness": float(multi_eval["fitness"]),
            "ic_mean": float(multi_eval["ic_mean"]),
            "rank_ic_mean": float(multi_eval["rank_ic_mean"]),
            "turnover": float(multi_eval["turnover"]),
        },
        "factor_corr": corr.to_dict(),
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")

    if verbose:
        print(f"saved diagnostics -> {diagnostics_path}")
        print(f"selected factors: {selected}")

    return Fama6Artifacts(
        single_factor_results=single_results,
        multi_factor_result=multi_result,
        diagnostics_path=diagnostics_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end Fama6 proxy factor loop (single + multi factor).")
    parser.add_argument("--codes", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT NVDA")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "src" / "output" / "factors"))
    parser.add_argument("--price-provider", default="yahoo")
    parser.add_argument("--forward-days", type=int, default=21)
    parser.add_argument("--corr-threshold", type=float, default=0.75)
    parser.add_argument("--as-ex", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_fama6_loop(
        codes=args.codes,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        price_provider=args.price_provider,
        forward_days=args.forward_days,
        corr_threshold=args.corr_threshold,
        verbose=not args.quiet,
        as_ex=args.as_ex,
    )
    print(
        json.dumps(
            {
                "single_factors": sorted(artifacts.single_factor_results.keys()),
                "multi_factor": artifacts.multi_factor_result.name,
                "multi_factor_rows": int(len(artifacts.multi_factor_result.data)),
                "diagnostics": str(artifacts.diagnostics_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["Fama6Artifacts", "build_fama6_loop", "main"]
