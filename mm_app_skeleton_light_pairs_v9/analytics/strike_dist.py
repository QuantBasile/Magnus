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


def strike_ref_dist_pct_abs(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="float64")

    if "Strike" not in df.columns or "Ref" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    strike = pd.to_numeric(df["Strike"], errors="coerce")
    ref = pd.to_numeric(df["Ref"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (strike - ref) / ref * 100.0
    return pct.abs().astype("float64")


def pnl_by_strike_dist_bucket(df: pd.DataFrame, *, bucket_pct: float = 2.0) -> pd.DataFrame:
    """
    Returns DataFrame columns: bucket, pnl
    bucket labels like '0-2%', '2-4%', ...
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["bucket", "pnl"])

    dist = strike_ref_dist_pct_abs(df)
    pnl = pnl_leg_series(df)

    work = pd.DataFrame({"dist": dist, "pnl": pnl})
    work = work.dropna(subset=["dist"])
    if work.empty:
        return pd.DataFrame(columns=["bucket", "pnl"])

    # build edges up to max
    mx = float(work["dist"].max())
    if not np.isfinite(mx) or mx <= 0:
        edges = np.array([0.0, bucket_pct])
    else:
        n = int(np.ceil(mx / bucket_pct))
        n = max(n, 1)
        edges = np.arange(0.0, (n + 1) * bucket_pct, bucket_pct)

    cats = pd.cut(
        work["dist"],
        bins=edges,
        right=False,
        include_lowest=True,
    )

    out = (
        work.groupby(cats, dropna=True, observed=False)["pnl"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"dist": "bucket", "pnl": "pnl"})
    )

    # prettier labels
    labels = []
    for itv in out["bucket"]:
        lo = float(itv.left)
        hi = float(itv.right)
        labels.append(f"{lo:.0f}-{hi:.0f}%")
    out["bucket"] = labels

    return out
