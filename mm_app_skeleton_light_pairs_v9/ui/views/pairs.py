# Python 3.12
from __future__ import annotations

from tkinter import ttk
import tkinter as tk
import pandas as pd
import numpy as np

from ui.widgets.grouped_pairs_table import GroupedPairsTable


def _detect_side_column(df: pd.DataFrame) -> str | None:
    candidates = ["b/s", "B/S", "side", "Side", "BUYSELL", "BuySell", "buy_sell", "direction", "Direction"]
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["b/s", "side", "buysell", "buy_sell", "direction"]:
        if key in lower_map:
            return lower_map[key]
    return None


def _detect_ref_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Ref", "REF", "Reference", "reference",
        "RefPrice", "ReferencePrice",
        "Mid", "MidPrice",
        "Theo", "TheoPrice",
        "UnderlyingRef", "UND_REF",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["ref", "reference", "refprice", "mid", "midprice", "theo", "theoprice"]:
        if key in lower_map:
            return lower_map[key]
    return None


def _to_float_series(x: pd.Series) -> pd.Series:
    """
    Robust numeric conversion:
    - handles "1.234,56" -> 1234.56 (EU format)
    - handles "1234.56" -> 1234.56
    """
    s = x.astype("string").fillna("")
    s = s.str.replace(" ", "", regex=False)

    has_comma = s.str.contains(",", na=False)
    s2 = s.copy()
    s2.loc[has_comma] = (
        s2.loc[has_comma]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out = pd.to_numeric(s2, errors="coerce")
    return out


class PairsView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self._df_master: pd.DataFrame | None = None
        self._available_days: list[str] = ["ALL"]
        self._available_cps: list[str] = ["ALL"]
        self._available_unds: list[str] = ["ALL"]

        card = ttk.Frame(self, style="Panel.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        card.grid_rowconfigure(2, weight=1)
        card.grid_columnconfigure(0, weight=1)

        # Header
        header = ttk.Frame(card, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        header.grid_columnconfigure(1, weight=1)

        ttk.Label(header, text="Pairs / Groups", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="(expand groups)", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=10)

        # Filters (match UnderlyingView style: dropdowns with ALL)
        filters = ttk.Frame(card, style="Panel.TFrame")
        filters.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 8))

        ttk.Label(filters, text="Level:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.var_level = tk.StringVar(value="ALL")
        ttk.Combobox(
            filters, width=6, state="readonly",
            values=["ALL", "L1", "L2", "L3"],
            textvariable=self.var_level
        ).grid(row=0, column=1, padx=(6, 12))

        ttk.Label(filters, text="Day:", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.var_day = tk.StringVar(value="ALL")
        self.cmb_day = ttk.Combobox(
            filters, width=12, state="readonly",
            values=self._available_days,
            textvariable=self.var_day
        )
        self.cmb_day.grid(row=0, column=3, padx=(6, 12))

        ttk.Label(filters, text="Counterparty:", style="Muted.TLabel").grid(row=0, column=4, sticky="w")
        self.var_cp = tk.StringVar(value="ALL")
        self.cmb_cp = ttk.Combobox(
            filters, width=14, state="readonly",
            values=self._available_cps,
            textvariable=self.var_cp
        )
        self.cmb_cp.grid(row=0, column=5, padx=(6, 12))

        ttk.Label(filters, text="Underlying:", style="Muted.TLabel").grid(row=0, column=6, sticky="w")
        self.var_und = tk.StringVar(value="ALL")
        self.cmb_und = ttk.Combobox(
            filters, width=16, state="readonly",
            values=self._available_unds,
            textvariable=self.var_und
        )
        self.cmb_und.grid(row=0, column=7, padx=(6, 12))

        ttk.Button(filters, text="Apply", command=self._rerender).grid(row=0, column=8, padx=(6, 0))
        ttk.Button(filters, text="Reset", command=self._reset_filters).grid(row=0, column=9, padx=(8, 0))

        # Table (no sorting on headers)
        cols = ["TradeTime", "ISIN", "b/s", "Quantity", "Price", "Ref", "Counterparty", "UND_NAME", "PnL/leg", "Δt", "ΔRef"]
        self.table = GroupedPairsTable(card, cols, height=16, enable_sort=False)
        self.table.grid(row=2, column=0, sticky="nsew", padx=14, pady=(6, 10))

        # Row styles
        self.table.tag_configure("buy", foreground="#0f766e")
        self.table.tag_configure("sell", foreground="#b91c1c")
        self.table.tag_configure("group", background="#eef2f7", font=("Segoe UI", 10, "bold"))

        palette = [
            "#f2f7ff", "#fff7ed", "#f0fdf4", "#faf5ff", "#fef2f2",
            "#eff6ff", "#fffbeb", "#ecfeff", "#fdf2f8", "#f7fee7",
        ]
        for i, c in enumerate(palette):
            self.table.tag_configure(f"g{i}", background=c)

        # KPIs (2x3)
        kpi_wrap = ttk.Frame(card, style="Panel.TFrame")
        kpi_wrap.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        kpi_wrap.grid_columnconfigure((0, 1, 2), weight=1)

        self.kpi_total_trades = self._kpi_card(kpi_wrap, "Total Trades", "-", value_style="KpiValue.TLabel")
        self.kpi_pair_trades = self._kpi_card(kpi_wrap, "Pair Trades", "-", value_style="KpiValue.TLabel")
        self.kpi_pair_pct = self._kpi_card(kpi_wrap, "Pair %", "-", value_style="KpiValue.TLabel")

        self.kpi_total_pnl = self._kpi_card(kpi_wrap, "Total PnL", "-", value_style="KpiValue.TLabel")
        self.kpi_pos = self._kpi_card(kpi_wrap, "Pair Trades +", "-", value_style="KpiValuePos.TLabel")
        self.kpi_neg = self._kpi_card(kpi_wrap, "Pair Trades -", "-", value_style="KpiValueNeg.TLabel")

        self.kpi_total_trades["frame"].grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=(0, 10))
        self.kpi_pair_trades["frame"].grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(0, 10))
        self.kpi_pair_pct["frame"].grid(row=0, column=2, sticky="ew", pady=(0, 10))

        self.kpi_total_pnl["frame"].grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.kpi_pos["frame"].grid(row=1, column=1, sticky="ew", padx=(0, 10))
        self.kpi_neg["frame"].grid(row=1, column=2, sticky="ew")

    def _kpi_card(self, parent, title: str, value: str, *, value_style: str):
        """
        KPI layout: title + value on the SAME ROW.
        Title left, value right.
        """
        fr = ttk.Frame(parent, style="KpiCard.TFrame")
        fr.grid_columnconfigure(0, weight=1)   # push value to the right
        fr.grid_columnconfigure(1, weight=0)

        ttk.Label(fr, text=title, style="KpiTitle.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=10)
        lbl = ttk.Label(fr, text=value, style=value_style)
        lbl.grid(row=0, column=1, sticky="e", padx=12, pady=10)

        return {"frame": fr, "label": lbl}

    def set_data(self, df_master: pd.DataFrame | None):
        self._df_master = df_master

        days = ["ALL"]
        cps = ["ALL"]
        unds = ["ALL"]

        if df_master is not None and not df_master.empty:
            if "TradeTime" in df_master.columns:
                tt = pd.to_datetime(df_master["TradeTime"], errors="coerce")
                uniq = sorted(tt.dt.strftime("%Y-%m-%d").dropna().unique().tolist())
                days += uniq[:500]

            if "Counterparty" in df_master.columns:
                uniq_cps = sorted(df_master["Counterparty"].astype("string").fillna("").unique().tolist())
                uniq_cps = [c for c in uniq_cps if c != ""]
                cps += uniq_cps[:5000]

            if "UND_NAME" in df_master.columns:
                uniq_unds = sorted(df_master["UND_NAME"].astype("string").fillna("").unique().tolist())
                uniq_unds = [u for u in uniq_unds if u != ""]
                unds += uniq_unds[:5000]

        self._available_days = days
        self.cmb_day.configure(values=self._available_days)
        if self.var_day.get() not in self._available_days:
            self.var_day.set("ALL")

        self._available_cps = cps
        self.cmb_cp.configure(values=self._available_cps)
        if self.var_cp.get() not in self._available_cps:
            self.var_cp.set("ALL")

        self._available_unds = unds
        self.cmb_und.configure(values=self._available_unds)
        if self.var_und.get() not in self._available_unds:
            self.var_und.set("ALL")

        self._rerender()

    def _reset_filters(self):
        self.var_level.set("ALL")
        self.var_day.set("ALL")
        self.var_cp.set("ALL")
        self.var_und.set("ALL")
        self._rerender()

    def _set_kpis(self, *, total_trades: int, pair_trades: int, total_pnl: float, pos: int, neg: int):
        pct = (pair_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        self.kpi_total_trades["label"].config(text=f"{total_trades:,}")
        self.kpi_pair_trades["label"].config(text=f"{pair_trades:,}")
        self.kpi_pair_pct["label"].config(text=f"{pct:,.2f}%")

        if total_pnl > 0:
            self.kpi_total_pnl["label"].config(text=f"{total_pnl:,.1f} €", style="KpiValuePos.TLabel")
        elif total_pnl < 0:
            self.kpi_total_pnl["label"].config(text=f"{total_pnl:,.1f} €", style="KpiValueNeg.TLabel")
        else:
            self.kpi_total_pnl["label"].config(text=f"{total_pnl:,.1f} €", style="KpiValue.TLabel")

        self.kpi_pos["label"].config(text=f"{pos:,}")
        self.kpi_neg["label"].config(text=f"{neg:,}")

    def _rerender(self):
        df_master = self._df_master
        if df_master is None or df_master.empty:
            self.table.clear()
            self._set_kpis(total_trades=0, pair_trades=0, total_pnl=0.0, pos=0, neg=0)
            return

        df = df_master.copy()

        for c in ["TradeTime", "ISIN", "Quantity", "Counterparty", "UND_NAME"]:
            if c not in df.columns:
                df[c] = ""

        if "Price" not in df.columns:
            df["Price"] = np.nan

        # side
        side_col = _detect_side_column(df)
        if side_col is None:
            df["b/s"] = ""
        else:
            df["b/s"] = df[side_col].astype("string").fillna("")
        bs = df["b/s"].astype("string").str.strip()
        bs_up = bs.str.upper()
        df["b/s"] = np.where(bs_up.str.startswith("B"), "B", np.where(bs_up.str.startswith("S"), "S", bs))

        # ref numeric + keep for display
        ref_col = _detect_ref_column(df)
        if ref_col is None:
            df["_REFNUM_"] = np.nan
        else:
            df["_REFNUM_"] = _to_float_series(df[ref_col])

        has_L1 = "PairID_L1" in df.columns
        has_L2 = "GroupID_L2" in df.columns
        has_L3 = "PairID_L3" in df.columns
        if not (has_L1 or has_L2 or has_L3):
            self.table.clear()
            self._set_kpis(total_trades=0, pair_trades=0, total_pnl=0.0, pos=0, neg=0)
            return

        tt = pd.to_datetime(df["TradeTime"], errors="coerce")

        # row filters (dropdown exact-match, like UnderlyingView)
        mask = pd.Series(True, index=df.index)

        day = self.var_day.get().strip()
        if day and day != "ALL":
            mask &= (tt.dt.strftime("%Y-%m-%d") == day)

        cp_sel = self.var_cp.get().strip()
        if cp_sel and cp_sel != "ALL" and "Counterparty" in df.columns:
            mask &= (df["Counterparty"].astype("string") == cp_sel)

        und_sel = self.var_und.get().strip()
        if und_sel and und_sel != "ALL" and "UND_NAME" in df.columns:
            mask &= (df["UND_NAME"].astype("string") == und_sel)

        df_f = df.loc[mask].copy()
        tt_f = tt.loc[mask]
        total_trades = int(len(df_f))

        # group key priority L1 > L2 > L3
        primary_key = pd.Series(pd.NA, index=df_f.index, dtype="string")
        if has_L1:
            primary_key = primary_key.fillna(
                pd.Series(df_f["PairID_L1"])
                .where(df_f["PairID_L1"] >= 0, pd.NA)
                .astype("Int64").astype("string").radd("L1-")
            )
        if has_L2:
            primary_key = primary_key.fillna(
                pd.Series(df_f["GroupID_L2"])
                .where(df_f["GroupID_L2"] >= 0, pd.NA)
                .astype("Int64").astype("string").radd("L2-")
            )
        if has_L3:
            primary_key = primary_key.fillna(
                pd.Series(df_f["PairID_L3"])
                .where(df_f["PairID_L3"] >= 0, pd.NA)
                .astype("Int64").astype("string").radd("L3-")
            )

        in_any = primary_key.notna()
        if not bool(in_any.any()):
            self.table.clear()
            self._set_kpis(total_trades=total_trades, pair_trades=0, total_pnl=0.0, pos=0, neg=0)
            return

        level = self.var_level.get().strip().upper()
        if level in ("L1", "L2", "L3"):
            in_any = in_any & primary_key.astype("string").str.startswith(f"{level}-", na=False)

        if not bool(in_any.any()):
            self.table.clear()
            self._set_kpis(total_trades=total_trades, pair_trades=0, total_pnl=0.0, pos=0, neg=0)
            return

        # pnl/leg
        pnl = pd.Series(np.nan, index=df_f.index, dtype="float64")
        if "PairPnL_L1" in df_f.columns:
            pnl = pnl.fillna(pd.to_numeric(df_f["PairPnL_L1"], errors="coerce"))
        if "GroupPnL_L2" in df_f.columns:
            pnl = pnl.fillna(pd.to_numeric(df_f["GroupPnL_L2"], errors="coerce"))
        if "PairPnL_L3" in df_f.columns:
            pnl = pnl.fillna(pd.to_numeric(df_f["PairPnL_L3"], errors="coerce"))

        # Δt per group
        dt_sec = tt_f.groupby(primary_key).transform(lambda s: (s.max() - s.min()).total_seconds())

        # ---------- ΔRef = Ref_último_buy - Ref_sell ----------
        tmp = pd.DataFrame(
            {
                "g": primary_key,
                "t": tt_f,
                "side": df_f["b/s"].astype("string"),
                "ref": df_f["_REFNUM_"],
            },
            index=df_f.index,
        )
        tmp = tmp.dropna(subset=["g"]).sort_values(["g", "t"], kind="mergesort")

        sell_ref = (
            tmp[tmp["side"].str.upper().str.startswith("S", na=False)]
            .groupby("g")["ref"]
            .first()
        )
        last_buy_ref = (
            tmp[tmp["side"].str.upper().str.startswith("B", na=False)]
            .groupby("g")["ref"]
            .last()
        )
        dref = primary_key.map(last_buy_ref) - primary_key.map(sell_ref)
        # --------------------------------------------------------------------------

        out = pd.DataFrame({
            "GroupKey": primary_key.loc[in_any].astype("string"),
            "TradeTime": tt_f.loc[in_any].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ISIN": df_f.loc[in_any, "ISIN"].astype("string"),
            "b/s": df_f.loc[in_any, "b/s"].astype("string"),
            "Quantity": pd.to_numeric(df_f.loc[in_any, "Quantity"], errors="coerce").fillna(0).astype("int64"),
            "Price": pd.to_numeric(df_f.loc[in_any, "Price"], errors="coerce").round(6),
            "Ref": pd.to_numeric(df_f.loc[in_any, "_REFNUM_"], errors="coerce").round(6),
            "Counterparty": df_f.loc[in_any, "Counterparty"].astype("string"),
            "UND_NAME": df_f.loc[in_any, "UND_NAME"].astype("string"),
            "PnL/leg": pd.Series(pnl.loc[in_any]).round(4),
            "Δt": pd.to_numeric(dt_sec.loc[in_any], errors="coerce").round(3),
            "ΔRef": pd.to_numeric(dref.loc[in_any], errors="coerce").round(6),
        })

        # enforce complete groups (>=2 rows) after filters
        gs = out.groupby("GroupKey").size()
        out = out[out["GroupKey"].map(gs) >= 2]

        if out.empty:
            self.table.clear()
            self._set_kpis(total_trades=total_trades, pair_trades=0, total_pnl=0.0, pos=0, neg=0)
            return

        # KPIs on displayed trades
        pair_trades = int(len(out))
        total_pnl = float(np.nansum(out["PnL/leg"].to_numpy(dtype=float)))
        pos = int((out["PnL/leg"] > 0).sum())
        neg = int((out["PnL/leg"] < 0).sum())
        self._set_kpis(total_trades=total_trades, pair_trades=pair_trades, total_pnl=total_pnl, pos=pos, neg=neg)

        # sort groups by worst pnl
        out["_t"] = pd.to_datetime(out["TradeTime"], errors="coerce")
        group_min = out.groupby("GroupKey")["PnL/leg"].min()
        out["_gscore"] = out["GroupKey"].map(group_min)
        out = out.sort_values(
            by=["_gscore", "GroupKey", "PnL/leg", "_t"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).drop(columns=["_t", "_gscore"])

        # group tags palette
        palette_n = 10
        g2tag: dict[str, str] = {}
        tags = []
        for g in out["GroupKey"].tolist():
            if g not in g2tag:
                g2tag[g] = f"g{len(g2tag) % palette_n}"
            tags.append(g2tag[g])
        out["__gtag__"] = tags

        render_df = out[["GroupKey", "TradeTime", "ISIN", "b/s", "Quantity", "Price", "Ref",
                         "Counterparty", "UND_NAME", "PnL/leg", "Δt", "ΔRef"]].copy()
        self.table.set_groups(render_df, group_col="GroupKey", side_col="b/s")

        # tag group headers
        g_first = out.groupby("GroupKey")["__gtag__"].first().to_dict()
        for parent in self.table.tree.get_children(""):
            gname = self.table.tree.item(parent, "text")
            tag = g_first.get(gname)
            if tag:
                self.table.tree.item(parent, tags=("group", tag))
