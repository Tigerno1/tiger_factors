"""
Timing Factor Library
36 factors from Gupta & Kelly (2019), Moreira & Muir (2017), Campbell & Shiller (1988), etc.
Each function returns a pandas Series indexed by symbol.
Input: price_df (DataFrame, indexed by date, columns by symbol), t (int, time index)
"""
import pandas as pd
import numpy as np

def MOM1(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t]
    vol = price_df.pct_change(1).iloc[t-36:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM2(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-2:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-36:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM3(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-5:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-36:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM4(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-11:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-120:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM5(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-35:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-120:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM6(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-59:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-120:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM7(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-11:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-36:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM8(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-59:t+1].mean()
    vol = price_df.pct_change(1).iloc[t-120:t].std() * np.sqrt(12)
    signal = ret / vol
    return signal.clip(-2, 2)

def MOM9(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t])

def MOM10(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t-2:t+1].sum())

def MOM11(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t-5:t+1].sum())

def MOM12(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t-11:t+1].sum())

def VOL1(price_df: pd.DataFrame, t: int) -> pd.Series:
    daily_ret = price_df.pct_change(1).iloc[t]
    var_t = daily_ret.var()
    avg_var = price_df.pct_change(1).iloc[:t].var().mean()
    return 1 / var_t / avg_var

def VOL2(price_df: pd.DataFrame, t: int) -> pd.Series:
    std_t = price_df.pct_change(1).iloc[t].std()
    avg_std = price_df.pct_change(1).iloc[:t].std().mean()
    return 1 / std_t / avg_std

def VOL3(price_df: pd.DataFrame, t: int) -> pd.Series:
    # AR(1) log variance estimation placeholder
    std_t = price_df.pct_change(1).iloc[t].std()
    avg_std = price_df.pct_change(1).iloc[:t].std().mean()
    return 1 / std_t / avg_std

def VOL4(price_df: pd.DataFrame, t: int) -> pd.Series:
    # VIX/variance placeholder
    std_t = price_df.pct_change(1).iloc[t].std()
    avg_std = price_df.pct_change(1).iloc[:t].std().mean()
    return 1 / std_t / avg_std

def VOL5(price_df: pd.DataFrame, t: int) -> pd.Series:
    std_t = price_df.pct_change(1).iloc[t].std()
    avg_std = price_df.pct_change(1).iloc[:t].std().mean()
    return 1 / std_t / avg_std

def VOL6(price_df: pd.DataFrame, t: int) -> pd.Series:
    # CBOE VIX index placeholder
    return pd.Series(np.nan, index=price_df.columns)

def VOL7(price_df: pd.DataFrame, t: int) -> pd.Series:
    # CBOE SKEW index placeholder
    return pd.Series(np.nan, index=price_df.columns)

def REV1(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-59:t+1].mean()
    return 1 - ret

def REV2(price_df: pd.DataFrame, t: int) -> pd.Series:
    ret = price_df.pct_change(1).iloc[t-119:t+1].mean()
    return 1 - ret

def TS_MOM1(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t]) * 0.4

def TS_MOM2(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t-2:t+1].sum()) * 0.4

def TS_MOM3(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t-5:t+1].sum()) * 0.4

def TS_MOM4(price_df: pd.DataFrame, t: int) -> pd.Series:
    return np.sign(price_df.pct_change(1).iloc[t-11:t+1].sum()) * 0.4

def VAL1(price_df: pd.DataFrame, t: int, btm_df: pd.DataFrame) -> pd.Series:
    # book-to-market spread, placeholder
    return btm_df.iloc[t] - btm_df.iloc[t-1]

def VAL2(price_df: pd.DataFrame, t: int, btm_df: pd.DataFrame) -> pd.Series:
    return btm_df.iloc[t] - btm_df.iloc[t-1]

def VAL3(price_df: pd.DataFrame, t: int, btm_df: pd.DataFrame) -> pd.Series:
    return btm_df.iloc[t] - btm_df.iloc[t-1]

def VAL4(price_df: pd.DataFrame, t: int, btm_df: pd.DataFrame) -> pd.Series:
    return btm_df.iloc[t] - btm_df.iloc[t-1]

def VAL5(price_df: pd.DataFrame, t: int, btm_df: pd.DataFrame) -> pd.Series:
    return btm_df.iloc[t] - btm_df.iloc[t-1]

def VAL6(price_df: pd.DataFrame, t: int, btm_df: pd.DataFrame) -> pd.Series:
    return btm_df.iloc[t] - btm_df.iloc[t-1]

def SPREAD1(price_df: pd.DataFrame, t: int, spread_df: pd.DataFrame) -> pd.Series:
    return spread_df.iloc[t] - spread_df.iloc[t-1]

def SPREAD2(price_df: pd.DataFrame, t: int, spread_df: pd.DataFrame) -> pd.Series:
    return spread_df.iloc[t] - spread_df.iloc[t-1]

def IPS1(price_df: pd.DataFrame, t: int, ips_df: pd.DataFrame) -> pd.Series:
    return ips_df.iloc[t] - ips_df.iloc[t-1]

def IPS2(price_df: pd.DataFrame, t: int, ips_df: pd.DataFrame) -> pd.Series:
    return ips_df.iloc[t] - ips_df.iloc[t-1]

def IPS3(price_df: pd.DataFrame, t: int, ips_df: pd.DataFrame) -> pd.Series:
    return ips_df.iloc[t] - ips_df.iloc[t-1]

def IPS4(price_df: pd.DataFrame, t: int, ips_df: pd.DataFrame) -> pd.Series:
    return ips_df.iloc[t] - ips_df.iloc[t-1]

def IPS5(price_df: pd.DataFrame, t: int, ips_df: pd.DataFrame) -> pd.Series:
    return ips_df.iloc[t] - ips_df.iloc[t-1]

def IPS6(price_df: pd.DataFrame, t: int, ips_df: pd.DataFrame) -> pd.Series:
    return ips_df.iloc[t] - ips_df.iloc[t-1]
