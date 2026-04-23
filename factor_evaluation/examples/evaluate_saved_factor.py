from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_evaluation import build_alphalens_input

matplotlib.use("Agg")

try:
    import alphalens as al  # type: ignore
except Exception:  # pragma: no cover
    al = None


def _load_long_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported file format: {path.suffix}')


def _to_panel(frame: pd.DataFrame, *, date_col: str, code_col: str, value_col: str) -> pd.DataFrame:
    panel = frame.pivot(index=date_col, columns=code_col, values=value_col).sort_index()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index, errors='coerce'))
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a saved factor parquet with Tiger reports and optional Alphalens adapter.')
    parser.add_argument('--factor-path', required=True, help='Saved factor parquet/csv in long format.')
    parser.add_argument('--price-path', required=True, help='Adjusted price parquet/csv in long format.')
    parser.add_argument('--factor-column', default='alpha_001', help='Name of the factor column to evaluate.')
    parser.add_argument('--date-col', default='date_', help='Date column name in the long tables.')
    parser.add_argument('--code-col', default='code', help='Code column name in the long tables.')
    parser.add_argument('--price-column', default='close', help='Price column used for forward returns.')
    parser.add_argument('--benchmark-path', default='', help='Optional benchmark returns parquet/csv in long format with date and returns columns.')
    parser.add_argument('--benchmark-date-col', default='date_', help='Benchmark date column name.')
    parser.add_argument('--benchmark-value-col', default='returns', help='Benchmark returns column name.')
    parser.add_argument('--group-path', default='', help='Optional group labels parquet/csv in long or wide format.')
    parser.add_argument('--group-date-col', default='date_', help='Group date column name if group input is long.')
    parser.add_argument('--group-code-col', default='code', help='Group code column name if group input is long.')
    parser.add_argument('--group-value-col', default='group', help='Group value column name if group input is long.')
    parser.add_argument('--output-dir', required=True, help='Directory for Tiger tear sheet outputs.')
    parser.add_argument('--quantiles', type=int, default=5, help='Quantiles for factor returns.')
    parser.add_argument('--alphalens-output-dir', default='', help='Optional directory to write Alphalens-ready inputs.')
    parser.add_argument('--run-alphalens', action='store_true', help='Also build Alphalens inputs and print a ready-to-run example.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    factor_frame = _load_long_table(args.factor_path)
    price_frame = _load_long_table(args.price_path)
    if args.date_col not in factor_frame.columns:
        raise ValueError(f'Missing factor date column: {args.date_col}')
    if args.code_col not in factor_frame.columns:
        raise ValueError(f'Missing factor code column: {args.code_col}')
    if args.factor_column not in factor_frame.columns:
        raise ValueError(f'Missing factor column: {args.factor_column}')
    if args.date_col not in price_frame.columns:
        raise ValueError(f'Missing price date column: {args.date_col}')
    if args.code_col not in price_frame.columns:
        raise ValueError(f'Missing price code column: {args.code_col}')
    if args.price_column not in price_frame.columns:
        raise ValueError(f'Missing price column: {args.price_column}')

    factor_frame = factor_frame.copy()
    price_frame = price_frame.copy()
    factor_frame[args.date_col] = pd.to_datetime(factor_frame[args.date_col], errors='coerce')
    price_frame[args.date_col] = pd.to_datetime(price_frame[args.date_col], errors='coerce')
    factor_frame[args.code_col] = factor_frame[args.code_col].astype(str)
    price_frame[args.code_col] = price_frame[args.code_col].astype(str)

    benchmark_returns = None
    if args.benchmark_path:
        benchmark_frame = _load_long_table(args.benchmark_path)
        if args.benchmark_date_col not in benchmark_frame.columns:
            raise ValueError(f'Missing benchmark date column: {args.benchmark_date_col}')
        if args.benchmark_value_col not in benchmark_frame.columns:
            raise ValueError(f'Missing benchmark returns column: {args.benchmark_value_col}')
        benchmark_frame = benchmark_frame.copy()
        benchmark_frame[args.benchmark_date_col] = pd.to_datetime(benchmark_frame[args.benchmark_date_col], errors='coerce')
        benchmark_returns = benchmark_frame.set_index(args.benchmark_date_col)[args.benchmark_value_col].sort_index()

    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=args.factor_column,
        date_column=args.date_col,
        code_column=args.code_col,
        price_column=args.price_column,
        benchmark_frame=benchmark_returns,
        group_labels=args.group_path or None,
        factor_store=FactorStore(root_dir=args.output_dir),
    )
    summary = engine.evaluate(benchmark_returns=benchmark_returns)
    print('FACTOR SUMMARY')
    print(summary)

    report = engine.full(
        output_dir=args.output_dir,
        quantiles=args.quantiles,
        by_group=bool(args.group_path),
    )
    print(f'\nSaved Tiger report to: {report.output_dir}')
    print('Tiger figures:')
    for name, path in report.figure_paths.items():
        print(f'  {name}: {path}')

    if args.run_alphalens:
        alphalens_input = build_alphalens_input(
            factor_frame=args.factor_path,
            price_frame=args.price_path,
            factor_column=args.factor_column,
            date_column=args.date_col,
            code_column=args.code_col,
            price_column=args.price_column,
        )
        print('\nALPHALENS INPUT READY')
        print(f'factor_series shape: {alphalens_input.factor_series.shape}')
        print(f'prices shape: {alphalens_input.prices.shape}')
        if args.alphalens_output_dir:
            out = Path(args.alphalens_output_dir)
            out.mkdir(parents=True, exist_ok=True)
            alphalens_input.factor_frame.to_parquet(out / 'factor_frame.parquet', index=False)
            alphalens_input.price_frame.to_parquet(out / 'price_frame.parquet', index=False)
            alphalens_input.factor_series.to_frame(name=args.factor_column).to_parquet(out / 'factor_series.parquet')
            alphalens_input.prices.to_parquet(out / 'prices.parquet')
            print(f'Alphalens-ready inputs saved to: {out}')
        if al is None:
            print('\nAlphalens is not installed in this environment.')
            print('Install it and run the following snippet:')
            print('''
import alphalens as al
from tiger_factors.factor_evaluation import build_alphalens_input

adapter = build_alphalens_input(
    factor_frame="'''+str(args.factor_path)+'''",
    price_frame="'''+str(args.price_path)+'''",
    factor_column="'''+args.factor_column+'''",
)
factor_data = al.utils.get_clean_factor_and_forward_returns(
    adapter.factor_series,
    adapter.prices,
    quantiles=5,
    periods=(1, 5, 10),
    max_loss=0.35,
)
al.tears.create_full_tear_sheet(factor_data)
''')
        else:
            print('\nRunning Alphalens tear sheet...')
            factor_data = al.utils.get_clean_factor_and_forward_returns(
                alphalens_input.factor_series,
                alphalens_input.prices,
                quantiles=5,
                periods=(1, 5, 10),
                max_loss=0.35,
            )
            al.tears.create_full_tear_sheet(factor_data)


if __name__ == '__main__':
    main()
