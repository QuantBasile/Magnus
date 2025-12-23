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


def _day_series(df: pd.DataFrame) -> pd.Series:
    tt = pd.to_datetime(df.get("TradeTime", pd.Series(index=df.index)), errors="coerce")
    return tt


def _pair_mask(df: pd.DataFrame) -> pd.Series:
    # any level present => "paired"
    m = pd.Series(False, index=df.index)
    if "PairID_L1" in df.columns:
        m |= pd.to_numeric(df["PairID_L1"], errors="coerce").fillna(-1).astype(int) >= 0
    if "GroupID_L2" in df.columns:
        m |= pd.to_numeric(df["GroupID_L2"], errors="coerce").fillna(-1).astype(int) >= 0
    if "PairID_L3" in df.columns:
        m |= pd.to_numeric(df["PairID_L3"], errors="coerce").fillna(-1).astype(int) >= 0
    return m


def best_worst_underlyings(df: pd.DataFrame, n: int = 3) -> tuple[list[str], list[str]]:
    if df is None or df.empty:
        return [], []
    if "UND_NAME" not in df.columns:
        return [], []

    pnl = pnl_leg_series(df)
    und = df["UND_NAME"].astype("string").fillna("")
    agg = pnl.groupby(und).sum(min_count=1).sort_values()

    # remove empty key if present
    if "" in agg.index:
        agg = agg.drop(index="")

    worst = agg.head(n).index.tolist()
    best = agg.tail(n).index.tolist()
    return best, worst


def cum_pnl_by_underlying(df: pd.DataFrame, underlyings: list[str]) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    """
    Returns dict: und -> (time, cum_pnl)
    cum is built in chronological order.
    """
    res: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    if df is None or df.empty or not underlyings:
        return res
    if "UND_NAME" not in df.columns or "TradeTime" not in df.columns:
        return res

    tt = _day_series(df)
    pnl = pnl_leg_series(df)
    und = df["UND_NAME"].astype("string").fillna("")

    work = pd.DataFrame({"t": tt, "pnl": pnl, "und": und})
    work = work.dropna(subset=["t"])
    work = work.sort_values("t", kind="mergesort")

    for u in underlyings:
        w = work[work["und"] == u]
        if w.empty:
            continue
        y = pd.to_numeric(w["pnl"], errors="coerce").fillna(0.0).to_numpy().cumsum()
        res[str(u)] = (pd.DatetimeIndex(w["t"]), y)
    return res


def cum_pnl_total(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if df is None or df.empty or "TradeTime" not in df.columns:
        return pd.DatetimeIndex([]), np.array([])
    tt = _day_series(df)
    pnl = pnl_leg_series(df)

    work = pd.DataFrame({"t": tt, "pnl": pnl}).dropna(subset=["t"]).sort_values("t", kind="mergesort")
    y = pd.to_numeric(work["pnl"], errors="coerce").fillna(0.0).to_numpy().cumsum()
    return pd.DatetimeIndex(work["t"]), y


def underlying_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table for selected level dataframe:
    Underlying, Total PnL, Pair Trades
    (Total Trades is filled in the View from df_base, same as Counterparty.)
    """
    if df is None or df.empty or "UND_NAME" not in df.columns:
        return pd.DataFrame(columns=["Underlying", "Total PnL", "Pair Trades"])

    pnl = pnl_leg_series(df)
    und = df["UND_NAME"].astype("string").fillna("")
    pair_tr = _pair_mask(df)

    # sum pnl
    total_pnl = pnl.groupby(und).sum(min_count=1)

    # count pair trades (row count)
    pair_trades = pair_tr.groupby(und).sum()

    tbl = pd.DataFrame({
        "Underlying": total_pnl.index.astype("string"),
        "Total PnL": total_pnl.values,
        "Pair Trades": pair_trades.reindex(total_pnl.index).fillna(0).astype(int).values,
    })

    # drop blank underlying row if any
    tbl = tbl[tbl["Underlying"].astype("string") != ""].copy()
    return tbl
