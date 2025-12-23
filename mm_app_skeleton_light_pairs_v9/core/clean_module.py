# clean_module.py
# Python 3.12
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_data(
    df: pd.DataFrame,
    *,
    tz: str | None = None,
    add_keys: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Minimal clean for the MM app + detect_pairs.

    Output columns (core):
      TradeTime (datetime64[ns])
      ISIN (string)
      b/s (string: "buy" / "sell")
      Quantity (int32)
      Price (float64)
      Counterparty (string)
      Ref (float64)
      UND_NAME (string)

    Output columns (keys required by detect_pairs):
      TradeTime_ns (int64; -1 for NaT)
      YearMonthDay (int32: YYYYMMDD; 0 for NaT)
      ISIN_id (int32)
      CP_id (int32)
      Side (int8: +1 buy, -1 sell, 0 unknown)

    Everything else is intentionally NOT computed.
    """

    meta: dict = {"warnings": [], "n_rows_in": int(len(df))}

    def _warn(msg: str):
        meta["warnings"].append(msg)

    def _ensure_str(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip()

    def _coerce_datetime_ns(s: pd.Series) -> pd.Series:
        out = pd.to_datetime(s, errors="coerce", utc=False)
        if tz is not None:
            try:
                if getattr(out.dt, "tz", None) is None:
                    out = out.dt.tz_localize(tz)
                else:
                    out = out.dt.tz_convert(tz)
                out = out.dt.tz_localize(None)
            except Exception as e:
                _warn(f"TZ conversion failed: {e}")
        return out.astype("datetime64[ns]")

    def _parse_float64(s: pd.Series) -> pd.Series:
        # handle comma decimals if they appear
        if s.dtype == object:
            s2 = s.astype("string").str.replace(",", ".", regex=False)
        else:
            s2 = s
        return pd.to_numeric(s2, errors="coerce").astype("float64")

    def _parse_int32(s: pd.Series) -> pd.Series:
        out = pd.to_numeric(s, errors="coerce")
        if strict:
            bad = out.notna() & (np.floor(out) != out)
            if bad.any():
                raise ValueError(f"Non-integer values found in integer column: {s.name}")
        out = np.floor(out)
        return out.fillna(-1).astype("int32")

    # ---- Keep only what we need ----
    needed = [
        "TradeTime",
        "ISIN",
        "b/s",
        "Quantity",
        "Price",
        "Counterparty",
        "Ref",
        "UND_NAME",
        "Strike",
        "CALL_OPTION"
    ]
    present = [c for c in needed if c in df.columns]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        msg = f"Missing expected columns (will be created empty): {missing}"
        if strict:
            raise KeyError(msg)
        _warn(msg)

    dfc = df[present].copy()

    # Create missing as empty
    for c in missing:
        dfc[c] = pd.Series([pd.NA] * len(df), dtype="string")

    # ---- Parse TradeTime ----
    dfc["TradeTime"] = _coerce_datetime_ns(dfc["TradeTime"])
    if dfc["TradeTime"].isna().any():
        nbad = int(dfc["TradeTime"].isna().sum())
        msg = f"TradeTime: {nbad} rows could not be parsed (set to NaT)."
        if strict:
            raise ValueError(msg)
        _warn(msg)

    # ---- Strings ----
    for col in ["ISIN", "Counterparty", "UND_NAME", "b/s","CALL_OPTION"]:
        dfc[col] = _ensure_str(dfc[col])

    # ---- Normalize b/s to buy/sell ----
    side = dfc["b/s"].str.lower().replace({"b": "buy", "s": "sell"})
    bad = ~(side.isin(["buy", "sell"]) | side.isna())
    if bad.any():
        msg = f"b/s: invalid values (examples: {side[bad].dropna().unique()[:5]})"
        if strict:
            raise ValueError(msg)
        _warn(msg)
        side = side.where(~bad, pd.NA)
    dfc["b/s"] = side.astype("string")

    # ---- Numerics (overwrite originals, no duplicates) ----
    dfc["Quantity"] = _parse_int32(dfc["Quantity"])
    dfc["Price"] = _parse_float64(dfc["Price"])
    dfc["Ref"] = _parse_float64(dfc["Ref"])
    dfc["Strike"] = _parse_float64(dfc["Strike"])

    # ---- Keys required by detect_pairs ----
    if add_keys:
        tt = dfc["TradeTime"]

        # TradeTime_ns (int64; -1 for NaT)
        t_ns = tt.values.astype("datetime64[ns]").astype("int64")
        t_ns = np.where(tt.isna().to_numpy(), -1, t_ns)
        dfc["TradeTime_ns"] = t_ns.astype("int64")

        # YearMonthDay (int32 YYYYMMDD; 0 for NaT)
        y = tt.dt.year.fillna(0).astype(np.int32)
        m = tt.dt.month.fillna(0).astype(np.int32)
        d = tt.dt.day.fillna(0).astype(np.int32)
        dfc["YearMonthDay"] = (y * 10000 + m * 100 + d).astype(np.int32)

        # Factorized ids
        isin_id, isin_uniques = pd.factorize(dfc["ISIN"].astype("string"), sort=False)
        dfc["ISIN_id"] = isin_id.astype(np.int32)
        meta["ISIN_uniques"] = isin_uniques.astype("string")

        cp_id, cp_uniques = pd.factorize(dfc["Counterparty"].astype("string"), sort=False)
        dfc["CP_id"] = cp_id.astype(np.int32)
        meta["CP_uniques"] = cp_uniques.astype("string")

        # Side int8
        bs = dfc["b/s"]
        side_int = np.zeros(len(dfc), dtype=np.int8)
        side_int[bs == "buy"] = 1
        side_int[bs == "sell"] = -1
        dfc["Side"] = side_int

    # Final column order (nice & stable)
    ordered = [
        "TradeTime", "ISIN","CALL_OPTION",  "b/s", "Quantity", "Price",
        "Counterparty", "UND_NAME", "Ref", "Strike"
    ]
    if add_keys:
        ordered += ["TradeTime_ns", "YearMonthDay", "ISIN_id", "CP_id", "Side"]

    dfc = dfc.loc[:, [c for c in ordered if c in dfc.columns]].copy()

    meta["n_rows_out"] = int(len(dfc))
    return dfc, meta
