# Python 3.12
from __future__ import annotations
import numpy as np
import pandas as pd


def pnl_leg_series(df: pd.DataFrame) -> pd.Series:
    pnl = pd.Series(np.nan, index=df.index, dtype="float64")
    if "PairPnL_L1" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["PairPnL_L1"], errors="coerce"))
    if "GroupPnL_L2" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["GroupPnL_L2"], errors="coerce"))
    if "PairPnL_L3" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["PairPnL_L3"], errors="coerce"))
    return pnl


def intraday_hourly_pnl(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by hour 0..23 with total PnL per hour-bin.
    """
    if df is None or df.empty or "TradeTime" not in df.columns:
        return pd.Series([0.0] * 24, index=list(range(24)), dtype="float64")

    tt = pd.to_datetime(df["TradeTime"], errors="coerce")
    hours = tt.dt.hour

    pnl = pnl_leg_series(df)
    work = pd.DataFrame({"h": hours, "pnl": pnl})
    work = work.dropna(subset=["h"])
    work["h"] = work["h"].astype(int)

    s = work.groupby("h")["pnl"].sum(min_count=1)
    s = s.reindex(range(24)).fillna(0.0).astype("float64")
    return s
