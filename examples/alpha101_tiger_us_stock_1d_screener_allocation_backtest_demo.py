"""Tiger US stock 1d Alpha101 demo: screener -> allocation -> backtest.

This demo is intentionally explicit and follows the three-stage workflow:

1. load stored Alpha101 factor panels from the Tiger factor store
2. screen them with single-factor metrics plus low-correlation selection
3. allocate weights from factor long-short return panels
4. backtest the weighted composite factor

The defaults target the local Tiger provider / US / stock / 1d store layout.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.examples.factor_store_multi_factor_reporting import save_factor_backtest_plot
from tiger_factors.factor_store import AdjPriceSpec
from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import FactorStore
from tiger_factors.multifactor_evaluation.allocation import LongShortReturnConfig
from tiger_factors.multifactor_evaluation.allocation import RiskfolioConfig
from tiger_factors.multifactor_evaluation.allocation import allocate_from_return_panel
from tiger_factors.multifactor_evaluation.allocation import build_long_short_return_panel
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest
from tiger_factors.multifactor_evaluation.pipeline import factor_correlation_matrix
from tiger_factors.multifactor_evaluation.pipeline import score_factor_panels
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.screening import FactorSummaryTableConfig
from tiger_factors.multifactor_evaluation.screening import build_factor_summary_table
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics
from tiger_factors.multifactor_evaluation.selection import greedy_select_by_correlation
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "alpha101_tiger_us_stock_1d_screener_allocation_backtest_demo"
DEFAULT_PROVIDER = "tiger"
DEFAULT_REGION = "us"
DEFAULT_SEC_TYPE = "stock"
DEFAULT_FREQ = "1d"
DEFAULT_GROUP: str | None = "alpha_101"
DEFAULT_VARIANT: str | None = None
DEFAULT_FORWARD_DAYS = 21
DEFAULT_TOP_N_INITIAL = 12
DEFAULT_CORR_THRESHOLD = 0.75
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_LONG_PCT = 0.20
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_ANNUAL_TRADING_DAYS = 252
DEFAULT_REPORT_NAME = "alpha101_tiger_us_stock_1d"
DEFAULT_FACTOR_PREFIX = "alpha_"
DEFAULT_PRICE_PROVIDER_FALLBACKS = ("tiger", "simfin")

@dataclass(slots=True)
class DemoConfig:
    store_root: Path = DEFAULT_FACTOR_STORE_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    provider: str = DEFAULT_PROVIDER
    price_provider: str | None = None
    region: str = DEFAULT_REGION
    sec_type: str = DEFAULT_SEC_TYPE
    freq: str = DEFAULT_FREQ
    group: str | None = DEFAULT_GROUP
    variant: str | None = DEFAULT_VARIANT
    factor_names: list[str] | None = None
    factor_prefix: str = DEFAULT_FACTOR_PREFIX
    start: str | None = None
    end: str | None = None
    forward_days: int = DEFAULT_FORWARD_DAYS
    top_n_initial: int = DEFAULT_TOP_N_INITIAL
    corr_threshold: float = DEFAULT_CORR_THRESHOLD
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ
    long_pct: float = DEFAULT_LONG_PCT
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    annual_trading_days: int = DEFAULT_ANNUAL_TRADING_DAYS
    report_name: str = DEFAULT_REPORT_NAME
    skip_report: bool = True


CONFIG = DemoConfig()


def _normalize_optional_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def _load_factor_panel(
    store: FactorStore,
    *,
    factor_name: str,
    provider: str,
    region: str,
    sec_type: str,
    freq: str,
    group: str | None,
    variant: str | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    from tiger_factors.factor_store import FactorSpec

    spec = FactorSpec(
        provider=provider,
        region=region,
        sec_type=sec_type,
        freq=freq,
        group=group,
        table_name=factor_name,
        variant=variant,
    )
    frame = store.get_factor(spec, start=start, end=end)
    factor_series = coerce_factor_series(frame)
    panel = factor_series.unstack("code").sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = "date_"
    return panel


def _load_close_panel(
    store: FactorStore,
    *,
    providers: list[str],
    region: str,
    sec_type: str,
    freq: str,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, str]:
    attempted_specs: list[str] = []
    for provider in providers:
        spec = AdjPriceSpec(provider=provider, region=region, sec_type=sec_type, freq=freq)
        price_frame = store.get_adj_price(spec, start=start, end=end)
        close_panel = coerce_price_panel(price_frame)
        attempted_specs.append(str(spec))
        if not close_panel.empty:
            return close_panel, provider
    raise RuntimeError(
        "Empty close panel loaded from store. Attempted specs: "
        + ", ".join(attempted_specs)
    )


def _align_common_universe(
    factor_panels: dict[str, pd.DataFrame],
    close_panel: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if not factor_panels:
        raise ValueError("No factor panels were loaded.")

    common_codes: set[str] | None = None
    common_dates: pd.DatetimeIndex | None = None
    for panel in factor_panels.values():
        codes = {str(code) for code in panel.columns}
        dates = pd.DatetimeIndex(panel.index)
        common_codes = codes if common_codes is None else (common_codes & codes)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

    close_dates = pd.DatetimeIndex(close_panel.index)
    close_codes = {str(code) for code in close_panel.columns}
    common_codes = (common_codes or set()) & close_codes
    common_dates = close_dates if common_dates is None else common_dates.intersection(close_dates)

    if not common_codes or common_dates is None or common_dates.empty:
        raise ValueError("No overlapping dates/codes were found between factors and close panel.")

    ordered_codes = sorted(common_codes)
    ordered_dates = pd.DatetimeIndex(common_dates).sort_values()
    aligned_factors = {
        name: panel.reindex(index=ordered_dates, columns=ordered_codes).sort_index()
        for name, panel in factor_panels.items()
    }
    aligned_close = close_panel.reindex(index=ordered_dates, columns=ordered_codes).sort_index().ffill()
    return aligned_factors, aligned_close


def _serializable_stats(stats: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(stats, default=str))


def _discover_factor_names(
    store: FactorStore,
    *,
    provider: str,
    region: str,
    sec_type: str,
    freq: str,
    group: str | None,
    variant: str | None,
    factor_prefix: str,
) -> list[str]:
    base_dir = store.root_dir / "factor" / provider / region / sec_type / freq
    if group is not None:
        base_dir = base_dir / group
    if not base_dir.exists():
        raise FileNotFoundError(f"Factor directory does not exist: {base_dir}")

    discovered: list[str] = []
    variant_suffix = None if variant is None else f"__{variant}"
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        stem = path.name
        if variant_suffix is None:
            if "__" in stem:
                continue
            factor_name = stem
        else:
            if not stem.endswith(variant_suffix):
                continue
            factor_name = stem[: -len(variant_suffix)]
        if factor_name.startswith(factor_prefix):
            discovered.append(factor_name)
    return discovered


def main() -> None:
    cfg = CONFIG
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group = _normalize_optional_token(cfg.group)
    variant = _normalize_optional_token(cfg.variant)
    requested_price_provider = _normalize_optional_token(cfg.price_provider)
    store = FactorStore(root_dir=cfg.store_root)

    if cfg.factor_names:
        factor_names = [str(name).strip() for name in cfg.factor_names if str(name).strip()]
    else:
        factor_names = _discover_factor_names(
            store=store,
            provider=cfg.provider,
            region=cfg.region,
            sec_type=cfg.sec_type,
            freq=cfg.freq,
            group=group,
            variant=variant,
            factor_prefix=str(cfg.factor_prefix).strip(),
        )
    if not factor_names:
        raise ValueError("No factor names were provided.")

    price_providers: list[str] = []
    if requested_price_provider is not None:
        price_providers.append(requested_price_provider)
    for provider in DEFAULT_PRICE_PROVIDER_FALLBACKS:
        if provider not in price_providers:
            price_providers.append(provider)

    close_panel, price_provider_used = _load_close_panel(
        store,
        providers=price_providers,
        region=cfg.region,
        sec_type=cfg.sec_type,
        freq=cfg.freq,
        start=cfg.start,
        end=cfg.end,
    )

    factor_panels: dict[str, pd.DataFrame] = {}
    for factor_name in factor_names:
        panel = _load_factor_panel(
            store,
            factor_name=factor_name,
            provider=cfg.provider,
            region=cfg.region,
            sec_type=cfg.sec_type,
            freq=cfg.freq,
            group=group,
            variant=variant,
            start=cfg.start,
            end=cfg.end,
        )
        if not panel.empty:
            factor_panels[factor_name] = panel

    if not factor_panels:
        raise ValueError("No Alpha101 factor panels could be loaded from the local Tiger factor store.")

    aligned_factors, aligned_close = _align_common_universe(factor_panels, close_panel)
    forward_returns = aligned_close.pct_change(cfg.forward_days, fill_method=None).shift(-cfg.forward_days)

    _, summaries = score_factor_panels(aligned_factors, forward_returns, score_field="fitness")
    summary = pd.DataFrame(summaries).T.reset_index(names="factor_name")
    raw_summary = summary.copy()

    screened = screen_factor_metrics(
        summary,
        config=FactorMetricFilterConfig(
            sort_field="fitness",
            tie_breaker_field="ic_ir",
        ),
    )
    if screened.empty:
        screened = raw_summary.sort_values("fitness", ascending=False)
    elif "factor_name" in screened.columns:
        screened = screened.set_index("factor_name", drop=False)

    score_table = build_factor_summary_table(
        screened,
        config=FactorSummaryTableConfig(
            x_metric="directional_ic_ir",
            y_metric="directional_sharpe",
            score_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            score_weights=(0.5, 0.25, 0.25),
        ),
    )

    top_candidates = [str(name) for name in screened["factor_name"].head(cfg.top_n_initial).tolist() if name in aligned_factors]
    candidate_panels = {name: aligned_factors[name] for name in top_candidates}
    if not candidate_panels:
        raise ValueError("No candidate factors survived the initial screener stage.")

    candidate_scores = {name: float(screened.loc[name, "fitness"]) for name in top_candidates if name in screened.index}
    corr = factor_correlation_matrix(candidate_panels, standardize=True)
    selected_factor_names = greedy_select_by_correlation(candidate_scores, corr, threshold=float(cfg.corr_threshold))
    if not selected_factor_names:
        selected_factor_names = top_candidates[:1]

    selected_panels = {name: aligned_factors[name] for name in selected_factor_names}
    return_panel = build_long_short_return_panel(
        selected_panels,
        aligned_close,
        config=LongShortReturnConfig(
            periods=(cfg.forward_days,),
            selected_period=cfg.forward_days,
            quantiles=5,
            long_short=True,
            source="tiger",
        ),
    )
    factor_weights = allocate_from_return_panel(
        return_panel,
        config=RiskfolioConfig(
            model="Classic",
            rm="MV",
            obj="Sharpe",
            rf=0.0,
            l=2.0,
            hist=True,
            method_mu="hist",
            method_cov="hist",
            max_kelly=False,
            weight_bounds=(0.0, 1.0),
        ),
    )

    backtest_result = multi_factor_backtest(
        selected_panels,
        aligned_close,
        weights=factor_weights.to_dict(),
        standardize=True,
        rebalance_freq=cfg.rebalance_freq,
        long_pct=float(cfg.long_pct),
        long_short=True,
        annual_trading_days=int(cfg.annual_trading_days),
        transaction_cost_bps=float(cfg.transaction_cost_bps),
        slippage_bps=float(cfg.slippage_bps),
    )

    raw_summary.to_csv(output_dir / "alpha101_raw_summary.csv")
    screened.to_csv(output_dir / "alpha101_screened_summary.csv")
    score_table.to_csv(output_dir / "alpha101_score_table.csv", index=False)
    corr.to_csv(output_dir / "alpha101_factor_correlation.csv")
    return_panel.to_csv(output_dir / "alpha101_return_panel.csv")
    factor_weights.to_csv(output_dir / "alpha101_factor_weights.csv", header=True)
    backtest_result["composite_factor"].to_csv(output_dir / "alpha101_composite_factor.csv")
    backtest_result["backtest"].to_csv(output_dir / "alpha101_backtest.csv")
    pd.DataFrame(backtest_result["stats"]).T.to_csv(output_dir / "alpha101_backtest_stats.csv")
    pd.Series(selected_factor_names, name="factor_name").to_csv(output_dir / "alpha101_selected_factors.csv", index=False)

    figure_path = save_factor_backtest_plot(
        backtest_result["backtest"],
        output_dir=output_dir,
        report_name=cfg.report_name,
    )

    report_summary: dict[str, object] | None = None
    if not cfg.skip_report:
        report = run_portfolio_from_backtest(
            backtest_result["backtest"],
            output_dir=output_dir / "portfolio",
            report_name=cfg.report_name,
        )
        report_summary = report.to_summary() if report is not None else None

    manifest = {
        "store_root": str(cfg.store_root),
        "provider": cfg.provider,
        "price_provider_requested": requested_price_provider,
        "price_provider_used": price_provider_used,
        "region": cfg.region,
        "sec_type": cfg.sec_type,
        "freq": cfg.freq,
        "group": group,
        "variant": variant,
        "factor_names": factor_names,
        "discovered_factor_count": int(len(factor_names)),
        "loaded_factor_names": list(aligned_factors.keys()),
        "selected_factor_names": selected_factor_names,
        "top_n_initial": int(cfg.top_n_initial),
        "corr_threshold": float(cfg.corr_threshold),
        "forward_days": int(cfg.forward_days),
        "rebalance_freq": cfg.rebalance_freq,
        "long_pct": float(cfg.long_pct),
        "transaction_cost_bps": float(cfg.transaction_cost_bps),
        "slippage_bps": float(cfg.slippage_bps),
        "annual_trading_days": int(cfg.annual_trading_days),
        "factor_weights": {str(name): float(weight) for name, weight in factor_weights.items()},
        "backtest_stats": _serializable_stats(backtest_result["stats"]),
        "equity_curve_path": None if figure_path is None else str(figure_path),
        "portfolio_report": report_summary,
    }
    (output_dir / "alpha101_demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("selected factors:")
    print(pd.Series(selected_factor_names, name="factor_name").to_string(index=False))
    print("\nallocation weights:")
    print(factor_weights.to_string())
    print("\nbacktest stats:")
    print(pd.DataFrame(backtest_result["stats"]).T.to_string())
    if figure_path is not None:
        print(f"\nequity curve: {figure_path}")
    if report_summary is not None:
        print("\nportfolio report summary:")
        print(json.dumps(report_summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
