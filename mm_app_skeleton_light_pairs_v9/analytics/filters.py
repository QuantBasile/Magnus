# Python 3.12
from __future__ import annotations
import pandas as pd


def _primary_group_key(df: pd.DataFrame) -> pd.Series:
    """L1-#, else L2-#, else L3-#; NA if none."""
    primary = pd.Series(pd.NA, index=df.index, dtype="string")

    if "PairID_L1" in df.columns:
        primary = primary.fillna(
            pd.Series(df["PairID_L1"])
            .where(df["PairID_L1"] >= 0, pd.NA)
            .astype("Int64").astype("string").radd("L1-")
        )
    if "GroupID_L2" in df.columns:
        primary = primary.fillna(
            pd.Series(df["GroupID_L2"])
            .where(df["GroupID_L2"] >= 0, pd.NA)
            .astype("Int64").astype("string").radd("L2-")
        )
    if "PairID_L3" in df.columns:
        primary = primary.fillna(
            pd.Series(df["PairID_L3"])
            .where(df["PairID_L3"] >= 0, pd.NA)
            .astype("Int64").astype("string").radd("L3-")
        )
    return primary


def apply_row_filters(
    df: pd.DataFrame,
    *,
    day: str = "ALL",
    level: str = "ALL",
    cp_sub: str = "",
    und_sub: str = "",
) -> pd.DataFrame:
    """Row filters matching Pairs tab semantics."""
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    if "TradeTime" not in out.columns:
        out["TradeTime"] = pd.NaT
    tt = pd.to_datetime(out["TradeTime"], errors="coerce")

    mask = pd.Series(True, index=out.index)

    if day and day != "ALL":
        mask &= (tt.dt.strftime("%Y-%m-%d") == day)

    if cp_sub:
        if "Counterparty" not in out.columns:
            out["Counterparty"] = ""
        mask &= out["Counterparty"].astype("string").fillna("").str.contains(cp_sub, case=False, na=False)

    if und_sub:
        if "UND_NAME" not in out.columns:
            out["UND_NAME"] = ""
        mask &= out["UND_NAME"].astype("string").fillna("").str.contains(und_sub, case=False, na=False)

    out = out.loc[mask].copy()

    lvl = (level or "ALL").strip().upper()
    if lvl in ("L1", "L2", "L3"):
        gk = _primary_group_key(out)
        out = out.loc[gk.astype("string").str.startswith(f"{lvl}-", na=False)].copy()

    return out
