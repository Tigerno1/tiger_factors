# Tiger Evaluation

`tiger_factors.factor_evaluation` is the Tiger-native factor evaluation module.

Evaluation and IO are separated:

- core evaluation uses `DataFrame`s
- path loading lives in `build_tiger_evaluation_input(...)`

It accepts either long-format `DataFrame`s or parquet/csv paths for:

- factor data: `date_`, `code`, factor column
- adjusted price data: `date_`, `code`, `close`

Main entrypoints:

```python
from tiger_factors.factor_evaluation import FactorEvaluationEngine
from tiger_factors.factor_evaluation import SingleFactorEvaluation

engine = FactorEvaluationEngine(
    factor_frame=factor_df,
    price_frame=adj_price_df,
    factor_column="alpha_001",
)

core = engine.evaluate()
summary = engine.summary()
report = engine.full()
bundle_summary = engine.create_report_bundle_summary(report)
horizon_result = engine.analyze_horizons([1, 3, 5, 10, 20])
horizon_summary = engine.summarize_best_horizon([1, 3, 5, 10, 20])

single = SingleFactorEvaluation(
    factor_frame=factor_df,
    price_frame=adj_price_df,
    factor_column="alpha_001",
    spec=spec,
)
single.summary()
single.horizon()
single.full()
summary_tables = single.summary.tables()
summary_report = single.summary.report()
summary_df = single.summary.get_table("summary")
```

Evaluation follows a split rule:

- `evaluate()` returns the core metric object
- `summary()` returns the final one-row summary table
- section methods expose supporting evidence
- `full()` returns the figure-only overview report

For `SingleFactorEvaluation`, the section commands are save-by-default and do
not return a value:

- `summary()` writes the `summary/` section
- `horizon()` writes the `horizon/` section
- `full()` writes the `full/` section

Each section accessor also exposes:

- `tables()`
- `imgs()`
- `report()`
- `get_table(name)`
- `get_img(name)`
- `get_report()`

Report outputs follow a split rule:

- the store-backed `summary/`, `horizon/`, and `full/` sections live under
  the evaluation store using the factor-spec-derived path layout
- report HTML is generated separately under the report output root
- `format=` controls which artifacts are persisted (`img`, `table`, `img_table`, `report`, `all`)
- call `get_report()` on a section accessor to open the HTML report in your browser

CLI example:

```bash
python -m tiger_factors.factor_evaluation.examples.run_full_evaluation \
  --factor-path /Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_001.parquet \
  --price-path /Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet \
  --factor-column alpha_001 \
  --output-dir /Volumes/Quant_Disk/evaluation/alpha_001_formal \
  --horizons 1 3 5 10 20
```

Alpha 002 example:

```bash
python -m tiger_factors.factor_evaluation.examples.run_alpha002_evaluation \
  --factor-path /Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_002.parquet \
  --price-path /Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet \
  --output-dir /Volumes/Quant_Disk/evaluation/alpha_002_formal
```

If you already have the table in memory, pass it straight into
`FactorEvaluationEngine(factor_frame=..., price_frame=..., factor_column="alpha_002")`.

Alpha 101 summary-only example:

```bash
python -m tiger_factors.factor_evaluation.examples.run_alpha101_summary_evaluation
```

This loads the stored `alpha_101.parquet` factor table from the provider-aware
`factor/<provider>/<region>/<sec_type>/<freq>/` layout and the
shared adjusted price table from `Quant_Disk`, then writes only the
`summary/` tear sheet.
This summary-only sheet writes the core summary parquet only and does not
generate report PNGs.

Alpha 101 summary batch example:

```bash
python -m tiger_factors.factor_evaluation.examples.run_alpha101_summary_batch
```

This loops over every stored `alpha_*.parquet` factor in the provider-aware
`/Volumes/Quant_Disk/factor/<provider>/...` tree, writes each summary tear
sheet under
`/Volumes/Quant_Disk/evaluation/<provider>/<region>/<sec_type>/<freq>/<factor_name>/summary/summary.parquet`, and saves a
`summary_registry.parquet` plus `summary_manifest.json` at the batch root.

`factor_name` here follows the same stem rule used by the store layer:

- with `variant`: `table_name__variant`
- without `variant`: `table_name`

Alpha 002 Alphalens example:

```bash
python -m tiger_factors.factor_evaluation.examples.run_alpha002_alphalens \
  --factor-path /Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_002.parquet \
  --price-path /Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet \
  --output-dir /Volumes/Quant_Disk/evaluation/alpha_002_alphalens
```

This example saves:

- `factor_data.parquet`
- `factor_frame.parquet`
- `price_frame.parquet`
- `factor_series.parquet`
- `prices.parquet`
- `summary.json`
- generated Alphalens figures as `figure_*.png`

Saved-factor Tiger evaluation example:

```bash
python -m tiger_factors.factor_evaluation.examples.evaluate_saved_factor \
  --factor-path /Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_001.parquet \
  --price-path /Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet \
  --factor-column alpha_001 \
  --output-dir /Volumes/Quant_Disk/evaluation/alpha_001_saved_factor
```

Minimal hardcoded Tiger evaluation smoke:

```bash
python -m tiger_factors.factor_evaluation.examples.tiger_factor_evaluation_run
```

Section tear sheets are table-first and do not emit figures by default.

`create_full_tear_sheet()` writes the report into the chosen root directory with:

- section folders:
  - `summary/`
  - `returns/`
  - `information/`
  - `turnover/`
  - `event_returns/`
  - `horizon/`
- section tables are named explicitly inside each folder:
  - `summary/summary.parquet`
  - `returns/<table_name>.parquet`
  - `information/<table_name>.parquet`
  - `turnover/<table_name>.parquet`
  - `event_returns/<table_name>.parquet`
  - `horizon/<table_name>.parquet`
- section figures are stored inside the matching section folders and the root
  `report.html` references them directly
- root-level manifest:
  - `manifest.json`

The manifests include:

- `created_at`
- factor/price column settings
- row counts
- tear sheet parameters
- horizon configuration
- output file paths

```python
from tiger_factors.factor_evaluation import FactorEvaluationEngine, build_tiger_evaluation_input

prepared = build_tiger_evaluation_input(
    factor_frame="/Volumes/Quant_Disk/factor/tiger/us/stock/1d/alpha_001.parquet",
    price_frame="/Volumes/Quant_Disk/price/tiger/us/stock/1d/adj_price.parquet",
    factor_column="alpha_001",
)
engine = FactorEvaluationEngine(
    factor_frame=prepared.factor_frame,
    price_frame=prepared.price_frame,
    factor_column="alpha_001",
)
```

For validation helpers that bootstrap, permutation-test, and split-check a
single factor's evidence, see `tiger_factors.factor_evaluation.validation`.

Available methods:

- `summary()`
- `get_clean_factor_and_forward_returns()`
- `evaluate()`
- `returns()`
- `information()`
- `turnover()`
- `event_returns()`
- `event_study()`
- `full()`
- `create_summary_tear_sheet()`
- `create_returns_tear_sheet()`
- `create_information_tear_sheet()`
- `create_turnover_tear_sheet()`
- `create_event_returns_tear_sheet()`
- `create_event_study_tear_sheet()`
- `create_full_tear_sheet()`
- `create_native_full_tear_sheet()`
- `create_report_bundle_summary()`
- `holding_period_analyzer()`
- `analyze_horizons()`
- `summarize_best_horizon()`
- `plot_horizon_result()`

`plot_horizon_result()` writes to the report root by default when `output_path` is omitted.

Input helpers exposed here:

- `build_alphalens_input()`
- `build_tiger_evaluation_input()`

Default output root:

```text
/Volumes/Quant_Disk/evaluation/<factor_column>/
```
