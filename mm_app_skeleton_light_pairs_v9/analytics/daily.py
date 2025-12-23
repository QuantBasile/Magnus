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


def _primary_group_key(df: pd.DataFrame) -> pd.Series:
    primary = pd.Series(pd.NA, index=df.index, dtype="string")
    if "PairID_L1" in df.columns:
        primary = primary.fillna(
            pd.Series(df["PairID_L1"]).where(df["PairID_L1"] >= 0, pd.NA)
            .astype("Int64").astype("string").radd("L1-")
        )
    if "GroupID_L2" in df.columns:
        primary = primary.fillna(
            pd.Series(df["GroupID_L2"]).where(df["GroupID_L2"] >= 0, pd.NA)
            .astype("Int64").astype("string").radd("L2-")
        )
    if "PairID_L3" in df.columns:
        primary = primary.fillna(
            pd.Series(df["PairID_L3"]).where(df["PairID_L3"] >= 0, pd.NA)
            .astype("Int64").astype("string").radd("L3-")
        )
    return primary


def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per day:
    Date, Total PnL, Pair PnL, NonPair PnL, Total Trades, Pair Trades, Pair %
    """
    if df is None or df.empty or "TradeTime" not in df.columns:
        return pd.DataFrame()

    tt = pd.to_datetime(df["TradeTime"], errors="coerce")
    day = tt.dt.strftime("%Y-%m-%d")

    pnl = pnl_leg_series(df)
    gk = _primary_group_key(df)
    pair_mask = gk.notna()

    # trades counts
    total_tr = day.value_counts().sort_index()
    pair_tr = day[pair_mask].value_counts().reindex(total_tr.index).fillna(0).astype(int)

    # pnl sums (ignore NaN)
    total_pnl = pnl.groupby(day).sum(min_count=1).reindex(total_tr.index).fillna(0.0)
    pair_pnl = pnl[pair_mask].groupby(day[pair_mask]).sum(min_count=1).reindex(total_tr.index).fillna(0.0)
    nonpair_pnl = total_pnl - pair_pnl

    res = pd.DataFrame({
        "Date": total_tr.index,
        "Total PnL": total_pnl.values,
        "Pair PnL": pair_pnl.values,
        "NonPair PnL": nonpair_pnl.values,
        "Total Trades": total_tr.values,
        "Pair Trades": pair_tr.values,
    })
    res["Pair %"] = np.where(res["Total Trades"] > 0, 100.0 * res["Pair Trades"] / res["Total Trades"], 0.0)
    return res
