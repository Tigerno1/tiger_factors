from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tiger_factors.factor_store import FactorStore, FactorSpec

store = FactorStore()


spec = FactorSpec(
    provider = "tiger",
    region = "us",
    sec_type = "stock", 
    freq = "1d",
    group = "alpha_101",
    table_name = "alpha_001"
)


data = store.get_factor(spec=spec)
print(data.head())