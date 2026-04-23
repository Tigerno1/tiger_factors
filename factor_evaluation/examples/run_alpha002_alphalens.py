from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tiger_matplotlib")
DEFAULT_FACTOR_PATH = "/Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_002.parquet"
DEFAULT_PRICE_PATH = "/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet"
DEFAULT_OUTPUT_DIR = "/Volumes/Quant_Disk/evaluation/alpha_002_alphalens"
DEFAULT_FACTOR_COLUMN = "alpha_002"
DEFAULT_PERIODS = (1, 5, 10)
DEFAULT_QUANTILES = 5

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiger_factors.factor_evaluation import build_alphalens_input

try:
    import alphalens as al  # type: ignore
except Exception:  # pragma: no cover
    al = None


def _load_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"Unsupported file format: {resolved.suffix}")


def _patch_forward_returns_freq() -> None:
    if al is None:
        return
    from alphalens import utils as al_utils

    source = inspect.getsource(al_utils.compute_forward_returns)
    source = source.replace("    df.index.levels[0].freq = freq\n", "    # Tiger compatibility: skip freq pinning for pandas 3.\n")
    source = source.replace("prices.pct_change(period)", "prices.pct_change(period, fill_method=None)")
    source = source.replace("prices.pct_change()", "prices.pct_change(fill_method=None)")
    namespace: dict[str, object] = {}
    exec(source, al_utils.__dict__, namespace)
    al_utils.compute_forward_returns = namespace["compute_forward_returns"]  # type: ignore[assignment]


def _patch_alpha_beta() -> None:
    if al is None:
        return

    import numpy as np
    import pandas as pd
    from alphalens import performance as al_performance
    from alphalens import utils as al_utils
    from statsmodels.api import OLS, add_constant

    def _safe_factor_alpha_beta(
        factor_data,
        returns=None,
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
    ):
        if returns is None:
            returns = al_performance.factor_returns(
                factor_data,
                demeaned=demeaned,
                group_adjust=group_adjust,
                equal_weight=equal_weight,
            )

        universe_ret = (
            factor_data.groupby(level="date")[al_utils.get_forward_returns_columns(factor_data.columns)]
            .mean()
            .reindex(returns.index, axis=0)
        )

        if isinstance(returns, pd.Series):
            returns.name = universe_ret.columns.values[0]
            returns = pd.DataFrame(returns)

        alpha_beta = pd.DataFrame()
        for period in returns.columns.values:
            x = universe_ret[period].values
            y = returns[period].values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if valid_mask.sum() < 2:
                alpha_beta.loc["Ann. alpha", period] = np.nan
                alpha_beta.loc["beta", period] = np.nan
                continue

            x = add_constant(x[valid_mask])
            y = y[valid_mask]
            reg_fit = OLS(y, x, missing="drop").fit()
            try:
                alpha, beta = reg_fit.params
            except ValueError:
                alpha_beta.loc["Ann. alpha", period] = np.nan
                alpha_beta.loc["beta", period] = np.nan
            else:
                freq_adjust = pd.Timedelta("252Days") / pd.Timedelta(period)
                alpha_beta.loc["Ann. alpha", period] = (1 + alpha) ** freq_adjust - 1
                alpha_beta.loc["beta", period] = beta

        return alpha_beta

    al_performance.factor_alpha_beta = _safe_factor_alpha_beta  # type: ignore[assignment]


def _save_figures(output_dir: Path) -> list[Path]:
    saved_files: list[Path] = []
    for fig_num in plt.get_fignums():
        figure = plt.figure(fig_num)
        if not figure.axes:
            continue
        title = ""
        for axis in figure.axes:
            if axis.get_title():
                title = axis.get_title()
                break
        safe_title = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in title).strip("_")
        filename = f"figure_{fig_num:02d}"
        if safe_title:
            filename = f"{filename}_{safe_title[:60]}"
        filepath = output_dir / f"{filename}.png"
        figure.savefig(filepath, dpi=200, bbox_inches="tight")
        saved_files.append(filepath)
    return saved_files


def _run_tear_sheet_without_closing(factor_data: pd.DataFrame, output_dir: Path) -> list[Path]:
    original_show = plt.show
    original_close = plt.close
    try:
        plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
        plt.close = lambda *args, **kwargs: None  # type: ignore[assignment]
        al.tears.create_full_tear_sheet(factor_data, long_short=True)
        return _save_figures(output_dir)
    finally:
        plt.show = original_show  # type: ignore[assignment]
        plt.close = original_close  # type: ignore[assignment]
        plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Alphalens report for alpha_002.")
    parser.add_argument("--factor-path", default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--price-path", default=DEFAULT_PRICE_PATH)
    parser.add_argument("--factor-column", default=DEFAULT_FACTOR_COLUMN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--periods", nargs="*", type=int, default=list(DEFAULT_PERIODS))
    return parser.parse_args()


def main() -> None:
    if al is None:
        raise RuntimeError("alphalens is not installed in this environment.")

    _patch_forward_returns_freq()
    _patch_alpha_beta()

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_frame = _load_table(args.factor_path)
    price_frame = _load_table(args.price_path)
    adapter = build_alphalens_input(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=args.factor_column,
    )

    factor_data = al.utils.get_clean_factor_and_forward_returns(
        adapter.factor_series,
        adapter.prices,
        quantiles=args.quantiles,
        periods=tuple(args.periods),
        max_loss=0.35,
    )

    factor_data.to_parquet(output_dir / "factor_data.parquet")
    adapter.factor_frame.to_parquet(output_dir / "factor_frame.parquet", index=False)
    adapter.price_frame.to_parquet(output_dir / "price_frame.parquet", index=False)
    adapter.factor_series.to_frame(name=args.factor_column).to_parquet(output_dir / "factor_series.parquet")
    adapter.prices.to_parquet(output_dir / "prices.parquet")

    saved_figures = _run_tear_sheet_without_closing(factor_data, output_dir)
    manifest = {
        "factor_column": args.factor_column,
        "factor_path": str(Path(args.factor_path)),
        "price_path": str(Path(args.price_path)),
        "output_dir": str(output_dir),
        "quantiles": args.quantiles,
        "periods": list(args.periods),
        "factor_rows": int(len(adapter.factor_frame)),
        "price_rows": int(len(adapter.price_frame)),
        "factor_data_rows": int(len(factor_data)),
        "factor_data_columns": list(map(str, factor_data.columns)),
        "saved_figures": [str(path.name) for path in saved_figures],
    }
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("ALPHALENS SUMMARY")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved figures to: {output_dir}")


if __name__ == "__main__":
    main()
