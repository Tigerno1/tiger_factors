"""Multifactors research entry point for Alpha101 factor generation and screening."""

from __future__ import annotations

import json
import os
import warnings
import zipfile
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide="ignore", invalid="ignore")

from tiger_factors.factor_algorithm.alpha101 import Alpha101Engine
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation import FactorPipelineConfig
from tiger_factors.multifactor_evaluation import screen_factor_panels
from tiger_factors.multifactor_evaluation.reporting.persistence import persist_multifactors_outputs


DEFAULT_UNIVERSE = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "Financials": ["JPM", "BAC", "GS", "V", "MA"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABT"],
    "Consumer": ["WMT", "COST", "PG", "KO", "PEP"],
    "Industrials": ["CAT", "GE", "HON", "MMM", "BA"],
    "Energy": ["XOM", "CVX"],
    "Materials": ["LIN", "APD"],
    "Real Estate": ["PLD", "AMT"],
    "Telecom": ["T", "VZ"],
}

LEAN_DAILY_ROOT = PROJECT_ROOT / "resources" / "Lean-master" / "Data" / "equity" / "usa" / "daily"


def flatten_universe(universe: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    codes = [code for codes_in_sector in universe.values() for code in codes_in_sector]
    sector_map = {code: sector for sector, codes_in_sector in universe.items() for code in codes_in_sector}
    return codes, sector_map


def _yf_to_long(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["date_", "code", "open", "high", "low", "close", "volume"])

    records: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0).unique())
        price_fields = {"Open", "High", "Low", "Close", "Volume"}
        slice_level = 1 if price_fields.issubset(level0) else 0
        for ticker in symbols:
            try:
                sub = raw.xs(ticker, axis=1, level=slice_level)[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception:
                continue
            sub.columns = ["open", "high", "low", "close", "volume"]
            sub["code"] = ticker
            sub.index.name = "date_"
            records.append(sub.reset_index())
    else:
        sub = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        sub.columns = ["open", "high", "low", "close", "volume"]
        sub["code"] = symbols[0]
        sub.index.name = "date_"
        records.append(sub.reset_index())

    df = pd.concat(records, ignore_index=True)
    df["date_"] = pd.to_datetime(df["date_"]).dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values(["code", "date_"]).reset_index(drop=True)
    return df


def to_wide_factor_panels(factors_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    factor_cols = [column for column in factors_long.columns if column.startswith("alpha_")]
    panels: dict[str, pd.DataFrame] = {}
    for factor_col in factor_cols:
        subset = factors_long[["date_", "code", factor_col]].dropna(subset=[factor_col])
        if subset.empty:
            continue
        wide = subset.pivot_table(index="date_", columns="code", values=factor_col, aggfunc="last").sort_index()
        if len(wide) < 20:
            continue
        panels[factor_col] = wide
    return panels


def _build_local_prices_long(
    codes: list[str],
    *,
    start: str,
    end: str,
    db_path: str | Path,
    price_provider: str,
    classification_provider: str,
    classification_dataset: str,
) -> pd.DataFrame:
    try:
        os.environ["TIGER_DB_URL_YAHOO_US_STOCK"] = f"sqlite:///{Path(db_path).resolve()}"
        library = TigerFactorLibrary(verbose=False)
        prices_long = library.build_alpha101_input(
            codes=codes,
            start=start,
            end=end,
            price_provider=price_provider,
            classification_provider=classification_provider,
            classification_dataset=classification_dataset,
        )
        if not prices_long.empty:
            return prices_long
    except Exception as exc:
        print(f"Local Tiger DB path unavailable, falling back to Lean sample data: {exc}")

    frames: list[pd.DataFrame] = []
    for code in codes:
        zip_path = LEAN_DAILY_ROOT / f"{code.lower()}.zip"
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as archive:
            csv_name = archive.namelist()[0]
            raw = archive.read(csv_name).decode("utf-8")
        frame = pd.read_csv(
            io.StringIO(raw),
            header=None,
            names=["raw_date", "open", "high", "low", "close", "volume"],
        )
        if frame.empty:
            continue
        frame["date_"] = pd.to_datetime(frame["raw_date"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
        frame["code"] = code.upper()
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") / 10000.0
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame = frame.dropna(subset=["date_", "close"]).copy()
        frame["vwap"] = frame[["open", "high", "low", "close"]].mean(axis=1)
        frames.append(frame[["date_", "code", "open", "high", "low", "close", "volume", "vwap"]])

    if not frames:
        raise RuntimeError("No local Tiger data or Lean sample data could be built for multifactors research.")

    prices_long = pd.concat(frames, ignore_index=True).sort_values(["date_", "code"]).reset_index(drop=True)
    return prices_long


def main() -> None:
    start = "2020-01-01"
    end = "2024-12-31"
    fast = False
    use_local_data = False
    db_path = str(Path("/Volumes/Quant_Disk/data/yahoo_us_stock.db"))
    price_provider = "yahoo"
    classification_provider = "simfin"
    classification_dataset = "companies"
    max_codes = 0
    forward_days = 5
    top_n_initial = 25
    corr_threshold = 0.65
    long_pct = 0.20
    weight_method = "softmax"
    weight_temp = 1.0
    min_factor_weight = None
    max_factor_weight = None
    transaction_cost_bps = 8.0
    slippage_bps = 4.0
    output_dir_name = "multifactors_research"
    skip_save_factors = False
    persist_outputs = False
    params = {
        "start": start,
        "end": end,
        "forward_days": forward_days,
        "top_n_initial": top_n_initial,
        "corr_threshold": corr_threshold,
        "long_pct": long_pct,
        "weight_method": weight_method,
        "weight_temp": weight_temp,
        "min_factor_weight": min_factor_weight,
        "max_factor_weight": max_factor_weight,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "fast": fast,
        "use_local_data": use_local_data,
        "persist_outputs": persist_outputs,
    }

    factor_ids = list(range(1, 21)) if fast else list(range(1, 102))

    data_root = Path(os.getenv("TIGER_TMP_DATA_PATH", "/Volumes/Quant_Disk/data/tmp"))
    if not data_root.exists():
        data_root = Path(__file__).resolve().parent / "output"
    output_dir = data_root / output_dir_name
    if persist_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    if use_local_data:
        codes = sorted({path.stem.upper() for path in LEAN_DAILY_ROOT.glob("*.zip")})
        if max_codes and max_codes > 0:
            codes = codes[: int(max_codes)]
        sector_map = {code: "Unknown" for code in codes}
        print(f"[1/8] Loading multifactors local Tiger/Lean data: {len(codes)} stocks ({start} -> {end})")
        prices_long = _build_local_prices_long(
            codes,
            start=start,
            end=end,
            db_path=db_path,
            price_provider=price_provider,
            classification_provider=classification_provider,
            classification_dataset=classification_dataset,
        )
        if "sector" not in prices_long.columns:
            prices_long["sector"] = prices_long["code"].map(sector_map).fillna("Unknown")
        if "industry" not in prices_long.columns:
            prices_long["industry"] = prices_long["sector"]
        if "subindustry" not in prices_long.columns:
            prices_long["subindustry"] = prices_long["industry"]
        if "market_value" not in prices_long.columns:
            prices_long["market_value"] = prices_long["close"] * prices_long["volume"]
        print(
            f"Loaded multifactors local panel: rows={len(prices_long):,}, codes={prices_long['code'].nunique()}, "
            f"dates={prices_long['date_'].nunique()}"
        )
    else:
        codes, sector_map = flatten_universe(DEFAULT_UNIVERSE)
        print(f"[1/8] Downloading multifactors prices: {len(codes)} stocks ({start} -> {end})")
        prices_long = _yf_to_long(codes, start, end)
        prices_long["sector"] = prices_long["code"].map(sector_map).fillna("Unknown")
        if prices_long.empty:
            raise RuntimeError("No price data downloaded. Check network/symbol settings.")

    print(f"[2/8] Computing multifactors factors: {len(factor_ids)} factors")
    engine = Alpha101Engine(prices_long)
    all_factors_long = engine.compute_all(alpha_ids=factor_ids)
    all_factors_long = all_factors_long[
        (all_factors_long["date_"] >= pd.Timestamp(start))
        & (all_factors_long["date_"] <= pd.Timestamp(end))
    ].reset_index(drop=True)

    factor_cols = [column for column in all_factors_long.columns if column.startswith("alpha_")]

    print("[3/8] Preparing multifactors factor summaries")
    if persist_outputs and not skip_save_factors:
        factor_save_dir = output_dir / "multifactors_factors"
        library = TigerFactorLibrary(output_dir=factor_save_dir, verbose=False)
        for factor_col in factor_cols:
            factor_df = all_factors_long[["date_", "code", factor_col]].dropna(subset=[factor_col]).copy()
            library.save_factor(
                factor_name=factor_col,
                factor_df=factor_df,
                metadata={"family": "multifactors", "source": "yfinance", "universe_size": len(codes)},
            )

    print("[4/8] Building multifactors factor panels and forward returns")
    factors_wide = to_wide_factor_panels(all_factors_long)
    if not factors_wide:
        raise RuntimeError("No valid factor panel generated.")

    close_wide = (
        prices_long.pivot_table(index="date_", columns="code", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    forward_returns = close_wide.pct_change(forward_days, fill_method=None).shift(-forward_days)

    print("[5/8] Multifactors IC / fitness evaluation")
    pipeline_result = screen_factor_panels(
        factors_wide,
        close_wide,
        config=FactorPipelineConfig(
            forward_days=forward_days,
            top_n_initial=top_n_initial,
            corr_threshold=corr_threshold,
            score_field="fitness",
            selection_score_field="ic_ir",
            weight_method=weight_method,
            weight_temperature=weight_temp,
            min_factor_weight=min_factor_weight,
            max_factor_weight=max_factor_weight,
            long_pct=long_pct,
            long_short=True,
            rebalance_freq="ME",
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            persist_outputs=persist_outputs,
        ),
        output_dir=output_dir / "multifactors_pipeline" if persist_outputs else None,
        report_dir=output_dir / "multifactors_reports" if persist_outputs else None,
        report_factor_name="multifactors_combined_factor",
    )
    selected_factors = pipeline_result.selected_factors
    chosen_weights = pipeline_result.factor_weights
    combined_factor = pipeline_result.combined_factor
    backtest_df = pipeline_result.backtest_returns
    stats = pipeline_result.backtest_stats
    if backtest_df.empty:
        raise RuntimeError("Backtest returned empty result. Check date range and data coverage.")

    if persist_outputs:
        _persist_multifactors_outputs(
            output_dir=output_dir,
            pipeline_summary=pipeline_result.summary,
            factor_score_table=pipeline_result.score_table,
            correlation_matrix=pipeline_result.correlation_matrix,
            prices_long=prices_long,
            factors_wide=factors_wide,
            selected_factors=selected_factors,
            chosen_weights=chosen_weights,
            combined_factor=combined_factor,
            backtest_df=backtest_df,
            stats=stats,
            factor_cols=factor_cols,
            params=params,
            long_pct=long_pct,
            weight_method=weight_method,
            forward_returns=forward_returns,
        )

    print("\n[8/8] Multifactors research complete.")
    if persist_outputs:
        print(f"Output dir: {output_dir}")
    else:
        print("No files were written; selected factors are returned in memory only.")
    print(f"Top selected factors: {selected_factors[:10]}")
    print("\nPortfolio stats:")
    print(json.dumps(stats["portfolio"], indent=2))


if __name__ == "__main__":
    main()
