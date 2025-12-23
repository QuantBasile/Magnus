# Python 3.12
from __future__ import annotations
import pandas as pd

from analytics.filters import apply_row_filters


def apply_filters_dropdown(
    df: pd.DataFrame,
    *,
    day: str = "ALL",
    level: str = "ALL",
    cp: str = "ALL",
    und: str = "ALL",
    level_mode: str = "as_requested",
):
    """
    level_mode:
      - "as_requested": apply_row_filters(... level=level)
      - "no_level": apply_row_filters(... level="ALL")
    """
    if df is None or df.empty:
        return df

    lvl = "ALL" if level_mode == "no_level" else level
    out = apply_row_filters(df, day=day, level=lvl, cp_sub="", und_sub="")

    if cp and cp != "ALL" and "Counterparty" in out.columns:
        out = out[out["Counterparty"].astype("string") == cp]

    if und and und != "ALL" and "UND_NAME" in out.columns:
        out = out[out["UND_NAME"].astype("string") == und]

    return out
