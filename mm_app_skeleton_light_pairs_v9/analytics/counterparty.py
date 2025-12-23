# Python 3.12
from __future__ import annotations
import numpy as np
import pandas as pd


def pnl_leg_series(df: pd.DataFrame) -> pd.Series:
    """PnL/leg float series (NaN if missing)."""
    pnl = pd.Series(np.nan, index=df.index, dtype="float64")
    if "PairPnL_L1" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["PairPnL_L1"], errors="coerce"))
    if "GroupPnL_L2" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["GroupPnL_L2"], errors="coerce"))
    if "PairPnL_L3" in df.columns:
        pnl = pnl.fillna(pd.to_numeric(df["PairPnL_L3"], errors="coerce"))
    return pnl


def best_worst_counterparties(df: pd.DataFrame, *, n: int = 5) -> tuple[list[str], list[str]]:
    """Pick best/worst by TOTAL PnL (after row filters)."""
    if df is None or df.empty or "Counterparty" not in df.columns:
        return [], []
    cp = df["Counterparty"].astype("string").fillna("")
    pnl = pnl_leg_series(df)
    totals = pnl.groupby(cp).sum().sort_values()
    totals = totals[totals.index != ""]
    worst = totals.head(n).index.tolist()
    best = totals.tail(n).index.tolist()
    return best, worst


def cum_pnl_by_cp(df: pd.DataFrame, cps: list[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    """dict cp -> (time, cumPnL)"""
    out = {}
    if df is None or df.empty or not cps:
        return out

    tt = pd.to_datetime(df["TradeTime"], errors="coerce")
    pnl = pnl_leg_series(df)

    for cp in cps:
        m = df["Counterparty"].astype("string") == cp
        d = pd.DataFrame({"t": tt[m], "p": pnl[m]}).dropna(subset=["t"])
        if d.empty:
            continue
        d = d.sort_values("t", kind="mergesort")
        d["p"] = pd.to_numeric(d["p"], errors="coerce").fillna(0.0)
        out[cp] = (d["t"], d["p"].cumsum())
    return out


def cum_pnl_total(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="float64")
    tt = pd.to_datetime(df["TradeTime"], errors="coerce")
    pnl = pnl_leg_series(df)
    d = pd.DataFrame({"t": tt, "p": pnl}).dropna(subset=["t"])
    if d.empty:
        return pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="float64")
    d = d.sort_values("t", kind="mergesort")
    d["p"] = pd.to_numeric(d["p"], errors="coerce").fillna(0.0)
    return d["t"], d["p"].cumsum()


def cp_table(df_base: pd.DataFrame, df_sel: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Counterparty, Total PnL, Total Trades, Pair Trades

    - Total Trades: computed on df_base (after filters)
    - Pair Trades: computed on df_sel (selected set: pairs/level). If df_sel is None, uses df_base.
    - Total PnL: computed on df_sel if provided & non-empty, else on df_base.
    """
    if df_base is None or df_base.empty or "Counterparty" not in df_base.columns:
        return pd.DataFrame()

    cp_base = df_base["Counterparty"].astype("string").fillna("")
    total_trades = cp_base.value_counts()

    # if df_sel not provided -> fall back to df_base
    if df_sel is None or df_sel.empty or "Counterparty" not in df_sel.columns:
        df_sel = df_base

    cp_sel = df_sel["Counterparty"].astype("string").fillna("")

    # Pair trades: rows belonging to any primary group (L1>L2>L3) on df_sel
    gk = pd.Series(pd.NA, index=df_sel.index, dtype="string")
    if "PairID_L1" in df_sel.columns:
        gk = gk.fillna(pd.Series(df_sel["PairID_L1"]).where(df_sel["PairID_L1"] >= 0, pd.NA).astype("Int64").astype("string").radd("L1-"))
    if "GroupID_L2" in df_sel.columns:
        gk = gk.fillna(pd.Series(df_sel["GroupID_L2"]).where(df_sel["GroupID_L2"] >= 0, pd.NA).astype("Int64").astype("string").radd("L2-"))
    if "PairID_L3" in df_sel.columns:
        gk = gk.fillna(pd.Series(df_sel["PairID_L3"]).where(df_sel["PairID_L3"] >= 0, pd.NA).astype("Int64").astype("string").radd("L3-"))

    pair_trades = cp_sel[gk.notna()].value_counts()

    # Total PnL: use pnl_leg_series on df_sel
    pnl = pnl_leg_series(df_sel)
    total_pnl = pnl.groupby(cp_sel).sum(min_count=1)

    idx = total_trades.index
    res = pd.DataFrame({
        "Counterparty": idx,
        "Total PnL": total_pnl.reindex(idx).fillna(0.0).values,
        "Total Trades": total_trades.reindex(idx).fillna(0).astype(int).values,
        "Pair Trades": pair_trades.reindex(idx).fillna(0).astype(int).values,
    })
    res = res[res["Counterparty"] != ""].copy()
    res = res.sort_values("Total PnL", ascending=False, kind="mergesort").reset_index(drop=True)
    return res

