
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_api.sdk.client import fetch  
from tiger_reference.adjustments import adj_df

from tiger_factors.factor_store import FactorStore, AdjPriceSpec

data = fetch(
    provider = "simfin",
    region = "us",
    sec_type="stock",
    freq = "1d",
    name = "eod_price",
    return_type="df"
)

adj_price = adj_df(data, dividends=True)
print(adj_price)
spec = AdjPriceSpec(
    provider="simfin",
    region = "us",
    sec_type = "stock",
    freq = "1d",
    table_name="adj_price"
)

store = FactorStore()
store.save_adj_price(spec, adj_price, force_update=True) 