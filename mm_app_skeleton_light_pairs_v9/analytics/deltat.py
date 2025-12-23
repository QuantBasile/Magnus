# Python 3.12
from __future__ import annotations
import pandas as pd
import numpy as np


def pnl_leg_series(df: pd.DataFrame) -> pd.Series:
    pnl = pd.Series(np.nan, index=df.index, dtype="float64")
    if "PairPnL_L1" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["PairPnL_L1"], errors="coerce"))
    if "GroupPnL_L2" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["GroupPnL_L2"], errors="coerce"))
    if "PairPnL_L3" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["PairPnL_L3"], errors="coerce"))
    return pnl


def _primary_group_key(df: pd.DataFrame) -> pd.Series:
    key = pd.Series(pd.NA, index=df.index, dtype="string")
    if "PairID_L1" in df.columns:
        key = key.fillna(
            df["PairID_L1"].where(df["PairID_L1"] >= 0).astype("Int64").astype("string").radd("L1-")
        )
    if "GroupID_L2" in df.columns:
        key = key.fillna(
            df["GroupID_L2"].where(df["GroupID_L2"] >= 0).astype("Int64").astype("string").radd("L2-")
        )
    if "PairID_L3" in df.columns:
        key = key.fillna(
            df["PairID_L3"].where(df["PairID_L3"] >= 0).astype("Int64").astype("string").radd("L3-")
        )
    return key


def deltat_bins_and_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      bin_label, pnl
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["bin", "pnl"])

    if "TradeTime" not in df.columns:
        return pd.DataFrame(columns=["bin", "pnl"])

    key = _primary_group_key(df)
    df = df.loc[key.notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["bin", "pnl"])

    tt = pd.to_datetime(df["TradeTime"], errors="coerce")
    df["_t"] = tt
    df["_key"] = key
    df["_pnl"] = pnl_leg_series(df)

    # Δt = max(time) - min(time) per group
    dt_sec = (
        df.groupby("_key")["_t"]
        .transform(lambda x: (x.max() - x.min()).total_seconds())
    )

    dt_min = dt_sec / 60.0
    df["_dt_min"] = dt_min

    # ---- Bin assignment ----
    def assign_bin(x):
        if pd.isna(x):
            return None
        if x < 10:
            return f"{int(np.floor(x))}-{int(np.floor(x)+1)} min"
        if x < 60:
            lo = int((x // 10) * 10)
            return f"{lo}-{lo+10} min"
        return "> 1h"

    df["bin"] = df["_dt_min"].map(assign_bin)

    out = (
        df.groupby("bin", dropna=True)["_pnl"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"_pnl": "pnl"})
    )

    return out
