# Factor Store

`FactorStore` is the Tiger-native storage layer for factor data, adjusted price data, macro data, and custom others data.

## Root Layout

All datasets live under `Quant_Disk` by default:

```text
/Volumes/Quant_Disk/
  factor/
  price/
  macro/
  evaluation/
  others/
```

## Dataset Layout

### Factor

Factor data is stored as a long table:

```text
factor/{provider}/{region}/{sec_type}/{freq}/{table_name}.parquet
factor/{provider}/{region}/{sec_type}/{freq}/{table_name}__{variant}.parquet
```

Fama-style factor bundles are saved as individual factor files through the
regular `save_factor(...)` API. Use `variant="fama"` and one of the canonical
factor names `mkt`, `smb`, `hml`, `rmw`, `cma`, or `umd`. Descriptive aliases
such as `market`, `size`, `value`, `profitability`, `investment`, and
`momentum` are also supported.

Quick reference:

| Family | Canonical names | Descriptive aliases |
| --- | --- | --- |
| 3 | `mkt`, `smb`, `hml` | `market`, `size`, `value` |
| 5 | `mkt`, `smb`, `hml`, `rmw`, `cma` | `market`, `size`, `value`, `profitability`, `investment` |
| 6 | `mkt`, `smb`, `hml`, `rmw`, `cma`, `umd` | `market`, `size`, `value`, `profitability`, `investment`, `momentum` |

The default `GTJA191` `alpha_030` lookup uses the canonical Fama 3 bundle:

```text
factor/{provider}/{region}/{sec_type}/{freq}/mkt__fama.parquet
factor/{provider}/{region}/{sec_type}/{freq}/smb__fama.parquet
factor/{provider}/{region}/{sec_type}/{freq}/hml__fama.parquet
```

For 5-factor and 6-factor Fama families, the same rule extends to:

```text
rmw__fama.parquet
cma__fama.parquet
umd__fama.parquet
```

Example:

```python
from tiger_factors.factor_store import FactorSpec

fama3 = library.build_fama3_panel(
    codes=["AAPL", "MSFT", "AMZN"],
    start="2020-01-01",
    end="2024-12-31",
    price_provider="yahoo",
    fama_provider="simfin",
)

for factor_name in library.fama_factor_names(3):
    factor_frame = (
        fama3.loc[:, ["date_", factor_name]]
        .rename(columns={factor_name: "value"})
        .assign(code=factor_name)
        .loc[:, ["date_", "code", "value"]]
    )
    library.save_factor(
        factor_name=factor_name,
        factor_df=factor_frame,
        spec=FactorSpec(
            provider="tiger",
            region="us",
            sec_type="stock",
            freq="1d",
            table_name=factor_name,
            variant="fama",
        ),
        force_updated=True,
    )

library.gtja191(
    alpha_id=30,
    codes=["AAPL", "MSFT", "AMZN"],
    start="2020-01-01",
    end="2024-12-31",
    price_provider="yahoo",
    fama_provider="simfin",
    use_cached_fama3=True,
)

# For Fama 5 and 6, use the same save_factor pattern with:
# - library.fama_factor_names(5) -> mkt, smb, hml, rmw, cma
# - library.fama_factor_names(6) -> mkt, smb, hml, rmw, cma, umd
```

For `save_fama5(...)` and `save_fama6(...)`, pass a wide frame with a `date_`
column plus the canonical or descriptive factor columns for that family.
For example, a Fama 5 frame can contain:

```text
date_
mkt / market
smb / size
hml / value
rmw / profitability
cma / investment
```

The normalized factor shape is:

```text
code
date_
value
```

To read a stored factor back into a wide panel, use
`TigerFactorLibrary.load_factor_panel(...)`. It returns a date-indexed
`date_ x code` frame that is ready for joins or backtests. Under the hood,
`FactorStore.get_factor(...)` reads factor files through the provider-layer
layout and, when available, uses the Ibis + DuckDB query path for pushdown
filtering:

```python
library = TigerFactorLibrary(output_dir="/Volumes/Quant_Disk")
bm_panel = library.load_factor_panel(
    factor_name="BM",
    provider="tiger",
    freq="1d",
)
```

To assemble several stored factors into a Tiger-style long research frame in
one call, use `TigerFactorLibrary.load_factor_frame(...)`:

```python
factor_frame = library.load_factor_frame(
    factor_names=["BM", "FSCORE"],
    provider="tiger",
    freq="1d",
)
```

For a generic multi-factor store demo that prints panel coverage and can save
the merged research frame to CSV, see
`tiger_factors.examples.factor_store_multi_factor_demo`.

For a companion equal-weight composite backtest over any stored factor
basket, see `tiger_factors.examples.factor_store_multi_factor_backtest_demo`.
The same demo also accepts custom weights via `--weights-json` as either a
JSON literal or a JSON file path.

### Adj Price

Adjusted price data is stored as:

```text
price/{provider}/{region}/{sec_type}/{freq}/adj_price[__variant].parquet
```

The adj price schema is centered on OHLCV:

```text
code
date_
open
high
low
close
volume
```

Extra columns may exist, but they do not define the core schema.

### Macro

Macro data is stored as:

```text
macro/{provider}/{region}/{freq}/{table_name}.parquet
macro/{provider}/{region}/{freq}/{table_name}__{variant}.parquet
```

The normalized macro shape is:

```text
date_
value
```

### Evaluation

Factor evaluation results live under the `evaluation` namespace on `FactorStore`:

```text
evaluation/{provider}/{region}/{sec_type}/{freq}/{factor_name}/summary/summary.parquet
evaluation/{provider}/{region}/{sec_type}/{freq}/{factor_name}/{section}/{table_name}.parquet
```

`factor_name` follows the same stem rule used everywhere else:

- with `variant`: `table_name__variant`
- without `variant`: `table_name`

`returns/`, `information/`, `turnover/`, `event_returns/`, and `horizon/`
are section folders, and each section can contain multiple explicitly named
tables.

The public read API is exposed as `FactorStore.evaluation`:

- `evaluation.summary(...).get_table()`
- `evaluation.section(...).get_table(...)`
- `evaluation.section(...).tables()`
- `evaluation.section(...).imgs()`
- `evaluation.section(...).get_img(...)`
- `evaluation.section(...).get_report(...)`

Persistence lives on `FactorStore.evaluation_store`:

- `save_summary(...)`
- `save_returns(...)`
- `save_information(...)`
- `save_turnover(...)`
- `save_event_returns(...)`
- `save_event_study(...)`
- `save_full(...)`

`SingleFactorEvaluation` uses `FactorStore.evaluation_store` internally when you call `save=True` on `summary`, `returns`, `information`, `turnover`, `event_returns`, `event_study`, or `full`.

### Others

Custom tabular datasets that do not belong to factor, adj price, macro, or evaluation live here:

```text
others/{provider}/{region}/{sec_type}/{freq}/{table_name}.parquet
others/{provider}/{region}/{sec_type}/{freq}/{table_name}__{variant}.parquet
```

`others` is intentionally separate from the core dataset families and is meant for custom spec-backed storage without factor-style schema normalization.
Its `provider`, `region`, `sec_type`, and `freq` values must come from the repository's known constants.
It is a good fit for summary tables or any other custom tabular output that should not be treated as factor, adj price, macro, or evaluation.

## Read API

The public read helpers are:

- `get_factor(...)`
- `TigerFactorLibrary.load_factor_panel(...)`
- `TigerFactorLibrary.load_factor_long(...)`
- `TigerFactorLibrary.load_factor_frame(...)`
- `get_adj_price(...)`
- `get_macro(...)`
- `get_others(...)`
- `evaluation.summary(...).get_table()`
- `evaluation.section(...).get_table(...)`
- `evaluation.section(...).tables()`
- `evaluation.section(...).imgs()`
- `evaluation.section(...).get_img(...)`
- `evaluation.section(...).get_report(...)`

## Save API

The public save helpers are:

- `save_factor(...)`
- `save_adj_price(...)`
- `save_macro(...)`
- `save_others(...)`
- `evaluation_store.save_summary(...)`
- `evaluation_store.save_returns(...)`
- `evaluation_store.save_information(...)`
- `evaluation_store.save_turnover(...)`
- `evaluation_store.save_event_returns(...)`
- `evaluation_store.save_event_study(...)`
- `evaluation_store.save_full(...)`

## Storage Rules

- `factor` and `adj_price` default to monthly partitions
- `factor` is provider-aware and keeps the provider as the first path segment
- `evaluation` stores section tables under the `evaluation/` namespace
- `macro` follows its own dataset spec and may be stored as a single file
- `others` is a separate custom bucket with explicit spec and no factor/adj price/macro mixing
- `force_updated=False` means existing data is preserved and save will fail
- `force_updated=True` means the target months or files are overwritten
- invalid data is rejected before any write happens

## Specs

- `FactorSpec` describes factor storage metadata
- `AdjPriceSpec` describes adjusted price storage metadata
- `MacroSpec` describes macro storage metadata
- `OthersSpec` describes custom others storage metadata
