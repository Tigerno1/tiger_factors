from pathlib import Path
import pandas as pd 

p = Path("/Volumes/Quant_Disk/factor_store/factor/tiger/us/stock/1d/alpha_101/alpha_002/")
data = pd.read_parquet(
    "/Volumes/Quant_Disk/factor_store/factor/tiger/us/stock/1d/alpha_101/alpha_002/alpha_002__2023-05.parquet"
)

print(data)

data2 = pd.read_parquet('/Volumes/Quant_Disk/factor_store/price/tiger/us/stock/1d/adj_price/adj_price__2021-11.parquet')
print(data2.head())