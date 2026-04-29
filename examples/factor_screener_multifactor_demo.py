"""Run the screener workflow, then let Riskfolio allocate stock holdings.

This demo shows the intended boundary between the layers:

1. ``Screener`` dispatches the factor screener and the correlation screener
2. selected factors are used only to build a composite stock score
3. Riskfolio optimizes the actual stock holdings inside the selected stock set

The script expects the factor evaluation artifacts to already exist in the
local Tiger factor store. It is intentionally explicit so you can swap in your
own factor names, providers, and universe controls.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import CorrelationScreenerSpec
from tiger_factors.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener import ScreeningEffectivenessSpec
from tiger_factors.factor_screener import Screener
from tiger_factors.factor_allocation import RiskfolioConfig
from tiger_factors.factor_allocation import allocate_from_return_panel
from tiger_factors.factor_portfolio import TigerTradeConstraintConfig
from tiger_factors.factor_portfolio import TigerTradeConstraintData
from tiger_factors.factor_portfolio import apply_trade_constraints_to_scores
from tiger_factors.factor_portfolio import apply_trade_constraints_to_weights
from tiger_factors.factor_portfolio import build_tradeable_universe_mask
from tiger_factors.factor_portfolio import run_weight_panel_backtest
from tiger_factors.factor_portfolio import standardize_cross_section
from tiger_factors.factor_portfolio import summarize_trade_constraints
from tiger_factors.factor_portfolio import weights_to_positions_frame
from tiger_factors.multifactor_evaluation.reporting.analysis_report import create_analysis_report


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "factor_screener_multifactor_demo"
DEFAULT_FACTOR_NAMES = (
    "alpha_021",
    "alpha_030",
    "alpha_047",
    "alpha_005",
    "alpha_076",
    "alpha_068",
    "alpha_066",
    "alpha_092",
)


FACTOR_ROOT = str(DEFAULT_FACTOR_STORE_ROOT)
FACTOR_PROVIDER = "tiger"
FACTOR_VARIANT: str | None = None
FACTOR_GROUP: str | None = "alpha_101"
REGION = "us"
SEC_TYPE = "stock"
FREQ = "1d"
FACTOR_NAMES: tuple[str, ...] = DEFAULT_FACTOR_NAMES
CODES: tuple[str, ...] | None = None
START: str | None = "2020-01-01"
END: str | None = "2024-12-31"
LONG_PCT = 0.20
REBALANCE_FREQ = "ME"
RISKFOLIO_LOOKBACK_DAYS = 252
MIN_OPTIMIZATION_OBSERVATIONS = 60
MIN_SELECTED_ASSETS = 2
ANNUAL_TRADING_DAYS = 252
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 2.0
RETURN_MODE = "long_short"
PRICE_PROVIDER = "yahoo"
CLASSIFICATION_PROVIDER = "simfin"
CLASSIFICATION_DATASET = "companies"
INDUSTRY_COLUMN = "sector"
OUTPUT_DIR = str(DEFAULT_OUTPUT_DIR)
REPORT_NAME = "factor_screener_multifactor_demo"
OPEN_BROWSER = False
TOP_HOLDINGS_N = 30
STOCK_GROSS_EXPOSURE = 1.0
SHORTABLE_CODES: tuple[str, ...] | None = None
HALTED_CODES: tuple[str, ...] = ()


CONFIG_SINGLE_FACTOR = FactorMetricFilterConfig(
    min_fitness=0.02,
    min_ic_mean=0.005,
    min_rank_ic_mean=0.005,
    min_sharpe=0.40,
    max_turnover=0.50,
    min_decay_score=0.20,
    min_capacity_score=0.20,
    max_correlation_penalty=0.60,
    min_regime_robustness=0.60,
    min_out_of_sample_stability=0.60,
    sort_field="fitness",
    tie_breaker_field="ic_ir",
)
CONFIG_RISKFOLIO = RiskfolioConfig(
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
)
CONFIG_TRADE_CONSTRAINTS = TigerTradeConstraintConfig(
    min_price=5.0,
    min_market_cap=500_000_000.0,
    min_dollar_volume=10_000_000.0,
    dollar_volume_window=20,
    require_price=True,
    require_volume=False,
    exclude_halted=True,
    require_shortable_for_short=True,
    max_single_name_weight=0.05,
    max_industry_weight=0.30,
    min_eligible_assets=10,
    normalize_after_constraints=True,
)
CONFIG_CORRELATION_STEPS = (
    CorrelationScreenerSpec(
        evaluation_source="factor",
        method="greedy",
        threshold=0.75,
        score_field="fitness",
    ),
    CorrelationScreenerSpec(
        evaluation_source="ic",
        method="greedy",
        threshold=0.60,
        score_field="fitness",
    ),
    CorrelationScreenerSpec(
        evaluation_source="return",
        method="greedy",
        threshold=0.00,
        score_field="fitness",
    ),
)


def _build_equal_weight_composite_signal(factor_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    standardized = {
        name: standardize_cross_section(panel)
        for name, panel in factor_panels.items()
        if isinstance(panel, pd.DataFrame) and not panel.empty
    }
    if not standardized:
        return pd.DataFrame()

    common_index = sorted(set().union(*(panel.index for panel in standardized.values())))
    common_columns = sorted(set().union(*(panel.columns.astype(str) for panel in standardized.values())))
    aligned = [
        panel.reindex(index=common_index, columns=common_columns)
        for panel in standardized.values()
    ]
    stacked = np.stack([panel.to_numpy(dtype=float) for panel in aligned])
    valid_count = np.isfinite(stacked).sum(axis=0)
    summed = np.nansum(stacked, axis=0)
    composite = np.divide(
        summed,
        valid_count,
        out=np.full_like(summed, np.nan, dtype=float),
        where=valid_count > 0,
    )
    return pd.DataFrame(composite, index=pd.DatetimeIndex(common_index), columns=common_columns).sort_index()


def _pivot_price_field(price_df: pd.DataFrame, field: str, columns: list[str]) -> pd.DataFrame:
    if price_df.empty or field not in price_df.columns:
        return pd.DataFrame()
    frame = price_df.loc[:, ["date_", "code", field]].copy()
    frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce").dt.tz_localize(None)
    frame["code"] = frame["code"].astype(str)
    frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return (
        frame.dropna(subset=["date_", "code"])
        .pivot_table(index="date_", columns="code", values=field, aggfunc="last")
        .sort_index()
        .reindex(columns=columns)
    )


def _load_price_and_constraint_panels(
    library: TigerFactorLibrary,
    *,
    codes: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_df = library.fetch_price_data(
        codes=codes,
        start=start,
        end=end,
        provider=PRICE_PROVIDER,
    )
    if price_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    close_raw = _pivot_price_field(price_df, "close", codes)
    close_for_returns = close_raw.ffill()
    volume = _pivot_price_field(price_df, "volume", codes)

    market_cap = _pivot_price_field(price_df, "market_value", codes)
    if market_cap.empty:
        shares = _pivot_price_field(price_df, "shares_outstanding", codes)
        if not shares.empty:
            market_cap = close_for_returns * shares.ffill()
    return close_raw, close_for_returns, volume, market_cap


def _load_industry_labels(
    library: TigerFactorLibrary,
    *,
    codes: list[str],
    start: str,
    end: str,
) -> pd.Series | None:
    try:
        companies = library.fetch_fundamental_data(
            provider=CLASSIFICATION_PROVIDER,
            name=CLASSIFICATION_DATASET,
            freq="static" if CLASSIFICATION_DATASET == "companies" else "1d",
            codes=codes,
            start=start,
            end=end,
        )
    except Exception as exc:
        print(f"  industry exposure skipped: {exc}")
        return None

    if companies.empty or "code" not in companies.columns:
        return None
    industry_column = next(
        (column for column in (INDUSTRY_COLUMN, "sector", "industry", "subindustry") if column in companies.columns),
        None,
    )
    if industry_column is None:
        return None
    labels = companies.loc[:, ["code", industry_column]].dropna(subset=["code"]).copy()
    labels["code"] = labels["code"].astype(str)
    labels[industry_column] = labels[industry_column].astype(str)
    if labels.empty:
        return None
    return labels.groupby("code")[industry_column].last()


def _static_shortable_series(codes: list[str]) -> pd.Series | None:
    if SHORTABLE_CODES is None:
        return None
    shortable = set(map(str, SHORTABLE_CODES))
    return pd.Series({code: code in shortable for code in codes}, dtype=bool)


def _static_halted_series(codes: list[str]) -> pd.Series | None:
    if not HALTED_CODES:
        return None
    halted = set(map(str, HALTED_CODES))
    return pd.Series({code: code in halted for code in codes}, dtype=bool)


def _select_assets_from_scores(scores: pd.Series, *, largest: bool) -> list[str]:
    clean = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    k = max(int(np.ceil(len(clean) * float(LONG_PCT))), 1)
    selected = clean.nlargest(k) if largest else clean.nsmallest(k)
    return selected.index.astype(str).tolist()


def _equal_weights(assets: list[str], gross: float) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    weight = float(gross) / float(len(assets))
    return pd.Series(weight, index=pd.Index(assets, dtype=str), dtype=float)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy(deep=False)
    output.attrs = {}
    output.to_parquet(path)


def _riskfolio_leg_weights(
    returns: pd.DataFrame,
    assets: list[str],
    *,
    gross: float,
) -> pd.Series:
    selected = [str(code) for code in assets if str(code) in returns.columns]
    if not selected:
        return pd.Series(dtype=float)

    panel = returns.loc[:, selected].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    panel = panel.dropna(axis=1, how="all")
    selected = panel.columns.astype(str).tolist()
    if len(selected) < MIN_SELECTED_ASSETS or len(panel.dropna(how="all")) < MIN_OPTIMIZATION_OBSERVATIONS:
        return _equal_weights(selected, gross)

    try:
        weights = allocate_from_return_panel(panel, config=CONFIG_RISKFOLIO)
    except Exception as exc:
        print(f"  riskfolio fallback to equal weights for {len(selected)} assets: {exc}")
        return _equal_weights(selected, gross)

    weights = pd.to_numeric(weights.reindex(selected), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0)
    total = float(weights.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return _equal_weights(selected, gross)
    return weights / total * float(gross)


def _build_stock_weight_panel(
    composite_signal: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    long_short: bool,
    trade_data: TigerTradeConstraintData | None = None,
    trade_constraints: TigerTradeConstraintConfig | None = None,
) -> pd.DataFrame:
    signal = composite_signal.sort_index()
    close = close_panel.reindex(columns=signal.columns).ffill().sort_index()
    stock_returns = close.pct_change(fill_method=None)
    rebalance_scores = signal.resample(REBALANCE_FREQ).last().dropna(how="all")
    if rebalance_scores.empty:
        return pd.DataFrame(index=rebalance_scores.index, columns=signal.columns, dtype=float)

    long_rebalance_scores = rebalance_scores
    short_rebalance_scores = rebalance_scores
    if trade_data is not None and trade_constraints is not None:
        long_rebalance_scores = apply_trade_constraints_to_scores(
            rebalance_scores,
            trade_data,
            trade_constraints,
            side="long",
        ).values
        if long_short:
            short_rebalance_scores = apply_trade_constraints_to_scores(
                rebalance_scores,
                trade_data,
                trade_constraints,
                side="short",
            ).values

    long_gross = float(STOCK_GROSS_EXPOSURE) if not long_short else float(STOCK_GROSS_EXPOSURE) / 2.0
    short_gross = float(STOCK_GROSS_EXPOSURE) / 2.0
    rows: list[pd.Series] = []

    for rebalance_date in rebalance_scores.index:
        trailing = stock_returns.loc[stock_returns.index < rebalance_date].tail(RISKFOLIO_LOOKBACK_DAYS)
        long_assets = _select_assets_from_scores(long_rebalance_scores.loc[rebalance_date], largest=True)
        short_assets: list[str] = []
        if long_short:
            short_assets = _select_assets_from_scores(short_rebalance_scores.loc[rebalance_date], largest=False)
            long_set = set(long_assets)
            short_assets = [code for code in short_assets if code not in long_set]

        row = pd.Series(0.0, index=signal.columns, dtype=float)
        long_weights = _riskfolio_leg_weights(trailing, long_assets, gross=long_gross)
        row.loc[long_weights.index] = long_weights

        if long_short and short_assets:
            short_weights = _riskfolio_leg_weights(-trailing, short_assets, gross=short_gross)
            row.loc[short_weights.index] = -short_weights

        rows.append(row.rename(pd.Timestamp(rebalance_date)))

    weights = pd.DataFrame(rows).sort_index()
    weights.index.name = "date_"
    weights = weights.reindex(columns=signal.columns).fillna(0.0)
    if trade_data is not None and trade_constraints is not None:
        weights = apply_trade_constraints_to_weights(weights, trade_data, trade_constraints)
    return weights


def _latest_holdings(weight_panel: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    positions = weights_to_positions_frame(weight_panel)
    if positions.empty:
        return positions
    latest_date = positions["date_"].max()
    latest = positions.loc[positions["date_"] == latest_date].copy()
    latest = latest.loc[pd.to_numeric(latest["weight"], errors="coerce").abs() > 1e-12]
    latest["side"] = np.where(latest["weight"] > 0, "long", "short")
    return latest.sort_values("weight", key=lambda series: series.abs(), ascending=False).head(int(top_n)).reset_index(drop=True)


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    factor_names = [str(name) for name in FACTOR_NAMES]
    store = FactorStore(root_dir=FACTOR_ROOT)
    library = TigerFactorLibrary(
        store=store,
        region=REGION,
        sec_type=SEC_TYPE,
        price_provider=PRICE_PROVIDER,
        verbose=False,
    )

    candidate_specs = [
        FactorSpec(
            provider=FACTOR_PROVIDER,
            region=REGION,
            sec_type=SEC_TYPE,
            freq=FREQ,
            table_name=factor_name,
            variant=FACTOR_VARIANT,
            group=FACTOR_GROUP,
        )
        for factor_name in factor_names
    ]

    # 1) Build spec inputs for the workflow. The workflow will resolve factor
    # evaluation artifacts from the store internally.
    screener_spec = FactorScreenerSpec(
        screening_config=CONFIG_SINGLE_FACTOR,
    )

    correlation_specs = tuple(
        CONFIG_CORRELATION_STEPS
    )

    # 2) Run the total screener. It first applies the factor screener, then
    # dispatches the correlation screener on the surviving factor specs.
    workflow_result = Screener(
        screener_spec,
        correlation_specs,
        factor_specs=tuple(candidate_specs),
        store=store,
        return_mode=RETURN_MODE,
    ).run()
    screening_effectiveness = workflow_result.validate_effectiveness(
        spec=ScreeningEffectivenessSpec(min_retained_ratio=0.8),
    )

    return_panel = workflow_result.return_panel
    selected_factor_names = [spec.table_name for spec in workflow_result.selected_factor_specs]

    # 3) Recover selected factor panels and blend them equally into a composite
    # stock score. Factors decide only which signals survive screening; they no
    # longer receive portfolio weights.
    selected_factor_panels = library.load_factor_panels(
        factor_names=selected_factor_names,
        provider=FACTOR_PROVIDER,
        freq=FREQ,
        variant=FACTOR_VARIANT,
        group=FACTOR_GROUP,
        codes=None if CODES is None else list(CODES),
        start=START,
        end=END,
    )
    selected_factor_panels = {
        name: panel
        for name, panel in selected_factor_panels.items()
        if name in selected_factor_names and not panel.empty
    }
    if not selected_factor_panels:
        raise RuntimeError("No selected factor panels could be loaded from the factor store.")

    composite_signal = _build_equal_weight_composite_signal(selected_factor_panels)
    universe_codes = sorted(composite_signal.columns.astype(str).tolist())
    if not universe_codes:
        raise RuntimeError("Selected factor panels produced an empty stock universe.")

    price_start = START or str(composite_signal.index.min().date())
    price_end = END or str(composite_signal.index.max().date())
    close_raw_panel, close_panel, volume_panel, market_cap_panel = _load_price_and_constraint_panels(
        library,
        codes=universe_codes,
        start=price_start,
        end=price_end,
    )
    if close_panel.empty:
        raise RuntimeError("Could not load a close panel for the selected stock universe.")
    industry_labels = _load_industry_labels(
        library,
        codes=universe_codes,
        start=price_start,
        end=price_end,
    )
    trade_data = TigerTradeConstraintData(
        close=close_raw_panel,
        volume=None if volume_panel.empty else volume_panel,
        market_cap=None if market_cap_panel.empty else market_cap_panel,
        industry=industry_labels,
        shortable=_static_shortable_series(universe_codes),
        halted=_static_halted_series(universe_codes),
    )
    long_tradeable_mask = build_tradeable_universe_mask(trade_data, CONFIG_TRADE_CONSTRAINTS, side="long")
    short_tradeable_mask = build_tradeable_universe_mask(trade_data, CONFIG_TRADE_CONSTRAINTS, side="short")
    long_tradeable_summary = summarize_trade_constraints(long_tradeable_mask)
    short_tradeable_summary = summarize_trade_constraints(short_tradeable_mask)

    # 4) Riskfolio decides stock holdings inside the factor-selected universe.
    # For long-short, long and short legs are optimized separately. The factor
    # scores only decide which assets enter each leg.
    stock_weight_panel = _build_stock_weight_panel(
        composite_signal,
        close_panel,
        long_short=RETURN_MODE == "long_short",
        trade_data=trade_data,
        trade_constraints=CONFIG_TRADE_CONSTRAINTS,
    )
    latest_holdings = _latest_holdings(stock_weight_panel, top_n=TOP_HOLDINGS_N)
    per_factor_latest_holdings = []
    for factor_name, panel in selected_factor_panels.items():
        factor_signal = standardize_cross_section(panel)
        factor_weight_panel = _build_stock_weight_panel(
            factor_signal,
            close_panel,
            long_short=RETURN_MODE == "long_short",
            trade_data=trade_data,
            trade_constraints=CONFIG_TRADE_CONSTRAINTS,
        )
        latest = _latest_holdings(factor_weight_panel, top_n=TOP_HOLDINGS_N)
        if latest.empty:
            continue
        latest.insert(0, "factor_name", factor_name)
        per_factor_latest_holdings.append(latest)
    per_factor_latest_holdings_frame = (
        pd.concat(per_factor_latest_holdings, ignore_index=True)
        if per_factor_latest_holdings
        else pd.DataFrame(columns=["factor_name", "date_", "code", "weight"])
    )

    # 5) Backtest the stock-level Riskfolio target weights.
    stock_backtest, stock_stats = run_weight_panel_backtest(
        stock_weight_panel,
        close_panel,
        rebalance_freq=REBALANCE_FREQ,
        annual_trading_days=ANNUAL_TRADING_DAYS,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )

    # 6) Generate an analysis report from the stock-level backtest returns.
    report = create_analysis_report(
        returns=stock_backtest["portfolio"],
        benchmark_returns=stock_backtest["benchmark"],
        output_dir=output_dir / "analysis",
        report_name=REPORT_NAME,
        open_browser=OPEN_BROWSER,
    )

    manifest = {
        "base_config": {
            "factor_root": FACTOR_ROOT,
            "factor_provider": FACTOR_PROVIDER,
            "factor_variant": FACTOR_VARIANT,
            "factor_group": FACTOR_GROUP,
            "region": REGION,
            "sec_type": SEC_TYPE,
            "freq": FREQ,
            "factor_names": list(FACTOR_NAMES),
            "codes": None if CODES is None else list(CODES),
            "long_pct": LONG_PCT,
            "rebalance_freq": REBALANCE_FREQ,
            "riskfolio_lookback_days": RISKFOLIO_LOOKBACK_DAYS,
            "min_optimization_observations": MIN_OPTIMIZATION_OBSERVATIONS,
            "min_selected_assets": MIN_SELECTED_ASSETS,
            "annual_trading_days": ANNUAL_TRADING_DAYS,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "output_dir": OUTPUT_DIR,
            "report_name": REPORT_NAME,
            "open_browser": OPEN_BROWSER,
            "portfolio_construction": "factor_screen_selects_signals_riskfolio_allocates_stocks",
        },
        "single_factor_screening_config": asdict(CONFIG_SINGLE_FACTOR),
        "stock_riskfolio_config": asdict(CONFIG_RISKFOLIO),
        "trade_constraint_config": asdict(CONFIG_TRADE_CONSTRAINTS),
        "trade_constraint_inputs": {
            "raw_close": not close_raw_panel.empty,
            "volume": not volume_panel.empty,
            "market_cap": not market_cap_panel.empty,
            "industry": industry_labels is not None,
            "shortable": SHORTABLE_CODES is not None,
            "halted": bool(HALTED_CODES),
        },
        "correlation_screening_configs": [asdict(config) for config in CONFIG_CORRELATION_STEPS],
        "factor_names": factor_names,
        "candidate_factor_specs": [asdict(spec) for spec in candidate_specs],
        "selected_factor_names": selected_factor_names,
        "selected_factor_specs": [asdict(spec) for spec in workflow_result.selected_factor_specs],
        "factor_selected_factor_names": [spec.table_name for spec in workflow_result.factor_selected_factor_specs],
        "factor_selected_factor_specs": [asdict(spec) for spec in workflow_result.factor_selected_factor_specs],
        "output_dir": str(output_dir),
        "return_panel_columns": list(return_panel.columns),
        "factor_blend_weights": {
            name: 1.0 / float(len(selected_factor_panels))
            for name in selected_factor_panels
        },
        "selected_factor_panel_shapes": {
            name: list(panel.shape)
            for name, panel in selected_factor_panels.items()
        },
        "stock_universe_count": int(len(universe_codes)),
        "latest_long_tradeable_count": 0 if long_tradeable_summary.empty else int(long_tradeable_summary["eligible_count"].iloc[-1]),
        "latest_short_tradeable_count": 0 if short_tradeable_summary.empty else int(short_tradeable_summary["eligible_count"].iloc[-1]),
        "stock_weight_panel_shape": list(stock_weight_panel.shape),
        "latest_holdings_date": None if latest_holdings.empty else str(latest_holdings["date_"].max()),
        "latest_holdings_count": int(len(latest_holdings)),
        "per_factor_latest_holdings_count": int(len(per_factor_latest_holdings_frame)),
        "workflow_summary": workflow_result.to_summary(),
        "screening_effectiveness_summary": screening_effectiveness.to_summary(),
        "screening_effectiveness_passed": bool(screening_effectiveness.passed),
        "screening_effectiveness_failed_rules": list(screening_effectiveness.failed_rules),
        "factor_screener_summary": workflow_result.factor_result.to_summary(),
        "correlation_screener_summary": workflow_result.correlation_result.to_summary(),
        "correlation_screener_chain": [result.to_summary() for result in workflow_result.correlation_results],
        "stock_backtest_stats": stock_stats,
        "analysis_report_path": None if report.get_report(open_browser=False) is None else str(report.get_report(open_browser=False)),
    }
    screening_effectiveness.save(output_dir / "screener" / "screening_effectiveness")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(stock_weight_panel, output_dir / "riskfolio_stock_weights.parquet")
    _write_parquet(weights_to_positions_frame(stock_weight_panel), output_dir / "riskfolio_stock_positions.parquet")
    latest_holdings.to_csv(output_dir / "riskfolio_latest_holdings.csv", index=False)
    _write_parquet(composite_signal, output_dir / "selected_factor_composite_signal.parquet")
    long_tradeable_summary.to_csv(output_dir / "tradeable_universe_long_summary.csv")
    short_tradeable_summary.to_csv(output_dir / "tradeable_universe_short_summary.csv")
    if not per_factor_latest_holdings_frame.empty:
        per_factor_latest_holdings_frame.to_csv(output_dir / "selected_factor_latest_holdings_by_factor.csv", index=False)
    _write_parquet(stock_backtest, output_dir / "riskfolio_stock_backtest.parquet")
    (output_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("screened factors:")
    print(f"  candidates: {factor_names}")
    print(f"  factor-selected specs: {[spec.table_name for spec in workflow_result.factor_selected_factor_specs]}")
    print(f"  selected specs: {[spec.table_name for spec in workflow_result.selected_factor_specs]}")
    print(f"  correlation chain: {[result.spec.evaluation_source + ':' + result.spec.method for result in workflow_result.correlation_results]}")
    print(f"  screening effectiveness passed: {screening_effectiveness.passed}")
    print(f"  return panel columns: {list(return_panel.columns)}")
    print(f"  stock universe count: {len(universe_codes)}")
    if not long_tradeable_summary.empty:
        print(f"  latest long-tradeable count: {int(long_tradeable_summary['eligible_count'].iloc[-1])}")
    if not short_tradeable_summary.empty:
        print(f"  latest short-tradeable count: {int(short_tradeable_summary['eligible_count'].iloc[-1])}")
    print("\nlatest Riskfolio stock holdings:")
    print(latest_holdings.to_string(index=False))
    if not per_factor_latest_holdings_frame.empty:
        print("\nlatest stock candidates by factor:")
        print(per_factor_latest_holdings_frame.to_string(index=False))
    print("\nbacktest stats:")
    print(pd.DataFrame(stock_stats).T.to_string())
    print("\noutputs:")
    print(f"  screener detail: {output_dir / 'screener'}")
    print(f"  screening effectiveness: {output_dir / 'screener' / 'screening_effectiveness'}")
    print(f"  composite signal: {output_dir / 'selected_factor_composite_signal.parquet'}")
    print(f"  long tradeable summary: {output_dir / 'tradeable_universe_long_summary.csv'}")
    print(f"  short tradeable summary: {output_dir / 'tradeable_universe_short_summary.csv'}")
    print(f"  Riskfolio stock weights: {output_dir / 'riskfolio_stock_weights.parquet'}")
    print(f"  Riskfolio stock positions: {output_dir / 'riskfolio_stock_positions.parquet'}")
    print(f"  Riskfolio latest holdings: {output_dir / 'riskfolio_latest_holdings.csv'}")
    print(f"  stock candidates by factor: {output_dir / 'selected_factor_latest_holdings_by_factor.csv'}")
    print(f"  stock backtest: {output_dir / 'riskfolio_stock_backtest.parquet'}")
    print(f"  analysis report: {report.get_report(open_browser=False)}")
    print(f"  workflow manifest: {output_dir / 'workflow_manifest.json'}")


if __name__ == "__main__":
    main()
