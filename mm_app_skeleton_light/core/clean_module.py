# clean_module.py
# Python 3.12
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_data(
    df: pd.DataFrame,
    *,
    tz: str | None = None,
    keep_original: bool = True,
    add_keys: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean + normalize your MM trade logs WITHOUT creating duplicated numeric columns.

    Keeps original names:
      Ref, Quantity, Price, Strike, Ratio (converted to efficient numeric dtypes)

    Adds only "helper" columns (not duplicates):
      TradeTime_ns, TradeDate, YearMonthDay, YearMonth, YearWeek,
      Side (+1 buy / -1 sell), ISIN_id, CP_id,
      StrikeDistPct, StrikeDistAbsPct, MoneynessPct,
      Expiry_D, KnockDate_D (datetime64[D]) for fast filtering (original Expiry/Knock_Date remain strings)

    Also fixes pandas to_datetime warning by specifying format="%Y-%m-%d" when parsing date-only strings.
    """

    meta: dict = {"warnings": [], "n_rows_in": int(len(df))}

    def _warn(msg: str):
        meta["warnings"].append(msg)

    def _ensure_str(s: pd.Series) -> pd.Series:
        return s.astype("string")

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

    def _parse_date_to_string_ymd(s: pd.Series) -> pd.Series:
        """
        Returns string 'YYYY-MM-DD' (or <NA>).
        Uses explicit format to avoid UserWarning when strings are in that format.
        """
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = s
        else:
            dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d", exact=False)
        d = dt.dt.floor("D")
        return d.dt.strftime("%Y-%m-%d").astype("string")

    def _parse_float64(s: pd.Series) -> pd.Series:
        # handle comma decimals if they appear
        if s.dtype == object:
            s2 = s.astype("string").str.replace(",", ".", regex=False)
        else:
            s2 = s
        out = pd.to_numeric(s2, errors="coerce").astype("float64")
        return out

    def _parse_int32(s: pd.Series) -> pd.Series:
        out = pd.to_numeric(s, errors="coerce")
        if strict:
            bad = out.notna() & (np.floor(out) != out)
            if bad.any():
                raise ValueError(f"Non-integer values found in integer column: {s.name}")
        out = np.floor(out)
        # keep NaN as -1 for fast numpy usage; if you prefer NA, tell me and I’ll adapt
        out = out.fillna(-1).astype("int32")
        return out

    required = [
        "TradeTime", "TradeNo", "ISIN", "b/s", "Quantity", "Price", "Counterparty",
        "Ref", "CALL_OPTION", "Expiry", "Knock_Date", "Ratio", "Strike", "UND_NAME", "UND_TYPE"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        if strict:
            raise KeyError(msg)
        _warn(msg)

    if keep_original:
        dfc = df.copy()
    else:
        present = [c for c in required if c in df.columns]
        dfc = df[present].copy()

    # --- TradeTime ---
    if "TradeTime" in dfc.columns:
        dfc["TradeTime"] = _coerce_datetime_ns(dfc["TradeTime"])
        if dfc["TradeTime"].isna().any():
            nbad = int(dfc["TradeTime"].isna().sum())
            msg = f"TradeTime: {nbad} rows could not be parsed (set to NaT)."
            if strict:
                raise ValueError(msg)
            _warn(msg)

    # --- Strings ---
    for col in ["TradeNo", "ISIN", "Counterparty", "UND_NAME", "UND_TYPE", "CALL_OPTION", "b/s", "Expiry", "Knock_Date"]:
        if col in dfc.columns:
            dfc[col] = _ensure_str(dfc[col]).str.strip()

    # --- Side normalization ---
    if "b/s" in dfc.columns:
        side = dfc["b/s"].str.lower().replace({"b": "buy", "s": "sell"})
        bad = ~(side.isin(["buy", "sell"]) | side.isna())
        if bad.any():
            msg = f"b/s: invalid values (examples: {side[bad].dropna().unique()[:5]})"
            if strict:
                raise ValueError(msg)
            _warn(msg)
            side = side.where(~bad, pd.NA)
        dfc["b/s"] = side.astype("string")

    # --- CALL_OPTION normalization ---
    if "CALL_OPTION" in dfc.columns:
        co = dfc["CALL_OPTION"].str.upper().replace({"CALL": "C", "PUT": "P"})
        bad = ~(co.isin(["C", "P"]) | co.isna())
        if bad.any():
            msg = f"CALL_OPTION: invalid values (examples: {co[bad].dropna().unique()[:5]})"
            if strict:
                raise ValueError(msg)
            _warn(msg)
            co = co.where(~bad, pd.NA)
        dfc["CALL_OPTION"] = co.astype("string")

    # --- Expiry / Knock_Date as string YYYY-MM-DD (no warning) ---
    for col in ["Expiry", "Knock_Date"]:
        if col in dfc.columns:
            dfc[col] = _parse_date_to_string_ymd(dfc[col])

    # --- Numerics: overwrite originals (NO duplicates) ---
    if "Quantity" in dfc.columns:
        dfc["Quantity"] = _parse_int32(dfc["Quantity"])
    for col in ["Price", "Ref", "Strike", "Ratio"]:
        if col in dfc.columns:
            dfc[col] = _parse_float64(dfc[col])

    # =========================
    # Add helper keys (no duplicates of core vars)
    # =========================
    if add_keys and "TradeTime" in dfc.columns:
        tt = dfc["TradeTime"]

        dfc["TradeDate"] = tt.values.astype("datetime64[D]")

        t_ns = tt.values.astype("datetime64[ns]").astype("int64")
        t_ns = np.where(tt.isna().to_numpy(), -1, t_ns)
        dfc["TradeTime_ns"] = t_ns.astype("int64")

        iso = tt.dt.isocalendar()
        year = iso["year"].fillna(0).astype(np.int32)
        week = iso["week"].fillna(0).astype(np.int32)
        dfc["YearWeek"] = (year * 100 + week).astype(np.int32)

        y = tt.dt.year.fillna(0).astype(np.int32)
        m = tt.dt.month.fillna(0).astype(np.int32)
        d = tt.dt.day.fillna(0).astype(np.int32)

        dfc["YearMonth"] = (y * 100 + m).astype(np.int32)
        dfc["YearMonthDay"] = (y * 10000 + m * 100 + d).astype(np.int32)

    if add_keys:
        if "ISIN" in dfc.columns:
            isin_id, isin_uniques = pd.factorize(dfc["ISIN"], sort=False)
            dfc["ISIN_id"] = isin_id.astype(np.int32)
            meta["ISIN_uniques"] = isin_uniques.astype("string")

        if "Counterparty" in dfc.columns:
            cp_id, cp_uniques = pd.factorize(dfc["Counterparty"], sort=False)
            dfc["CP_id"] = cp_id.astype(np.int32)
            meta["CP_uniques"] = cp_uniques.astype("string")

        if "b/s" in dfc.columns:
            side_s = dfc["b/s"]
            side_int = np.zeros(len(dfc), dtype=np.int8)
            side_int[side_s == "buy"] = 1
            side_int[side_s == "sell"] = -1
            dfc["Side"] = side_int

        # Date helpers as datetime64[D] for fast filtering (original strings remain)
        for col, new in [("Expiry", "Expiry_D"), ("Knock_Date", "KnockDate_D")]:
            if col in dfc.columns:
                dt = pd.to_datetime(dfc[col], errors="coerce", format="%Y-%m-%d", exact=False).dt.floor("D")
                dfc[new] = dt.values.astype("datetime64[D]")

        # --- ONLY: StrikeAbsDistPct ---
        if ("Strike" in dfc.columns) and ("Ref" in dfc.columns):
            refv = dfc["Ref"].to_numpy(np.float64, copy=False)
            strv = dfc["Strike"].to_numpy(np.float64, copy=False)
        
            with np.errstate(divide="ignore", invalid="ignore"):
                dist_abs_pct = np.abs((strv - refv) / refv) * 100.0
        
            dist_abs_pct = np.where(np.isfinite(dist_abs_pct), dist_abs_pct, np.nan)
            dfc["StrikeAbsDistPct"] = dist_abs_pct.astype(np.float64)


    meta["n_rows_out"] = int(len(dfc))
    return dfc, meta
