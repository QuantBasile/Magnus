# Python 3.12
from __future__ import annotations

from tkinter import ttk
import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

from analytics.filters import apply_row_filters
from analytics.deltat import pnl_leg_series
from ui.widgets.mpl_chart import MplChart
from ui.widgets.table import DataTable


def _fmt_km(v, _pos):
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 10_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.0f}"


def _pair_mask_any(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    if "PairID_L1" in df.columns:
        m |= df["PairID_L1"].fillna(-1).astype(int) >= 0
    if "GroupID_L2" in df.columns:
        m |= df["GroupID_L2"].fillna(-1).astype(int) >= 0
    if "PairID_L3" in df.columns:
        m |= df["PairID_L3"].fillna(-1).astype(int) >= 0
    return m


class CallPutView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self._df_master: pd.DataFrame | None = None
        self._available_days = ["ALL"]
        self._available_cps = ["ALL"]
        self._available_unds = ["ALL"]

        card = ttk.Frame(self, style="Panel.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        card.grid_rowconfigure(2, weight=1)

        # ── Header + filters ─────────────────────────────
        top = ttk.Frame(card, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 6))

        ttk.Label(top, text="Call / Put Overview", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        flt = ttk.Frame(top, style="Panel.TFrame")
        flt.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(flt, text="Level:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.var_level = tk.StringVar(value="ALL")
        ttk.Combobox(
            flt, width=6, state="readonly",
            values=["ALL", "L1", "L2", "L3"],
            textvariable=self.var_level
        ).grid(row=0, column=1, padx=(6, 12))

        ttk.Label(flt, text="Day:", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.var_day = tk.StringVar(value="ALL")
        self.cmb_day = ttk.Combobox(
            flt, width=10, state="readonly",
            values=self._available_days, textvariable=self.var_day
        )
        self.cmb_day.grid(row=0, column=3, padx=(6, 12))

        ttk.Label(flt, text="Counterparty:", style="Muted.TLabel").grid(row=0, column=4, sticky="w")
        self.var_cp = tk.StringVar(value="ALL")
        self.cmb_cp = ttk.Combobox(
            flt, width=14, state="readonly",
            values=self._available_cps, textvariable=self.var_cp
        )
        self.cmb_cp.grid(row=0, column=5, padx=(6, 12))

        ttk.Label(flt, text="Underlying:", style="Muted.TLabel").grid(row=0, column=6, sticky="w")
        self.var_und = tk.StringVar(value="ALL")
        self.cmb_und = ttk.Combobox(
            flt, width=16, state="readonly",
            values=self._available_unds, textvariable=self.var_und
        )
        self.cmb_und.grid(row=0, column=7, padx=(6, 12))

        ttk.Button(flt, text="Apply", command=self._rerender).grid(row=0, column=8, padx=(6, 0))
        ttk.Button(flt, text="Reset", command=self._reset).grid(row=0, column=9, padx=(6, 0))

        # ── KPIs: Calls vs Puts ─────────────────────────
        self.kpi_call = self._kpi_block(card, "CALLS")
        self.kpi_put = self._kpi_block(card, "PUTS")

        # IMPORTANT: grid the frame, not the dict
        self.kpi_call["frame"].grid(row=1, column=0, sticky="ew", padx=(14, 7), pady=(6, 10))
        self.kpi_put["frame"].grid(row=1, column=1, sticky="ew", padx=(7, 14), pady=(6, 10))

        # ── Chart ───────────────────────────────────────
        self.chart = MplChart(card, title="Cum PnL: Calls vs Puts")
        self.chart.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=14, pady=(0, 14))



    def _kpi_block(self, parent, title: str):
        fr = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(fr, text=title, style="Title.TLabel").pack(anchor="w", padx=12, pady=(8, 4))

        box = ttk.Frame(fr, style="Panel.TFrame")
        box.pack(fill="x", padx=12, pady=(0, 10))

        return {
            "frame": fr,
            "pnl": self._kpi(box, "Total PnL"),
            "total": self._kpi(box, "Total Trades"),
            "pairs": self._kpi(box, "Pair Trades"),
            "win": self._kpi(box, "Win %"),
        }


    def _kpi(self, parent, title: str):
        fr = ttk.Frame(parent)
        ttk.Label(fr, text=title, style="Muted.TLabel").pack(anchor="w")
        lbl = ttk.Label(fr, text="-", style="KpiValue.TLabel")
        lbl.pack(anchor="w")
        fr.pack(side="left", padx=10)
        return lbl

    def set_data(self, df_master: pd.DataFrame | None):
        self._df_master = df_master

        days, cps, unds = ["ALL"], ["ALL"], ["ALL"]
        if df_master is not None and not df_master.empty:
            if "TradeTime" in df_master.columns:
                tt = pd.to_datetime(df_master["TradeTime"], errors="coerce")
                days += sorted(tt.dt.strftime("%Y-%m-%d").dropna().unique().tolist())
            if "Counterparty" in df_master.columns:
                cps += sorted(df_master["Counterparty"].astype("string").dropna().unique().tolist())
            if "UND_NAME" in df_master.columns:
                unds += sorted(df_master["UND_NAME"].astype("string").dropna().unique().tolist())

        self.cmb_day.configure(values=days)
        self.cmb_cp.configure(values=cps)
        self.cmb_und.configure(values=unds)

        self._rerender()

    def _reset(self):
        self.var_level.set("ALL")
        self.var_day.set("ALL")
        self.var_cp.set("ALL")
        self.var_und.set("ALL")
        self._rerender()

    def _rerender(self):
        df = self._df_master
        if df is None or df.empty:
            return

        day = self.var_day.get()
        level = self.var_level.get()
        cp = self.var_cp.get()
        und = self.var_und.get()

        # Base (no level)
        df_base = apply_row_filters(df, day=day, level="ALL", cp_sub="", und_sub="")
        if cp != "ALL" and "Counterparty" in df_base.columns:
            df_base = df_base[df_base["Counterparty"] == cp]
        if und != "ALL" and "UND_NAME" in df_base.columns:
            df_base = df_base[df_base["UND_NAME"] == und]
        
        total_trades_all = int(len(df_base))


        # Selected (with level, or paired-only when ALL)
        if level == "ALL":
            df_sel = df_base.loc[_pair_mask_any(df_base)].copy()
        else:
            df_sel = apply_row_filters(df, day=day, level=level, cp_sub="", und_sub="")
            if cp != "ALL" and "Counterparty" in df_sel.columns:
                df_sel = df_sel[df_sel["Counterparty"] == cp]
            if und != "ALL" and "UND_NAME" in df_sel.columns:
                df_sel = df_sel[df_sel["UND_NAME"] == und]

        # Guard: if CALL_OPTION missing
        if "CALL_OPTION" not in df_sel.columns:
            # clear visuals to zeros
            self._render_rows([["Call", 0.0, 0, 0.0], ["Put", 0.0, 0, 0.0]], total_trades_all)
            return

        pnl = pnl_leg_series(df_sel)
        df_sel = df_sel.assign(_pnl=pnl)

        # Sort by time for meaningful cumulative curves
        if "TradeTime" in df_sel.columns:
            df_sel = df_sel.assign(_t=pd.to_datetime(df_sel["TradeTime"], errors="coerce")).sort_values("_t")

        rows: list[list[object]] = []

        self.chart.clear()
        ax = self.chart.ax
        ax.set_title("Cum PnL: Calls vs Puts")

        for opt, label in [("C", "Call"), ("P", "Put")]:
            d = df_sel[df_sel["CALL_OPTION"] == opt]
            if d.empty:
                rows.append([label, 0.0, 0, 0.0, 0.0])
                continue

            t = pd.to_datetime(d["TradeTime"], errors="coerce")
            leg = pd.to_numeric(d["_pnl"], errors="coerce").fillna(0.0)
            y = leg.cumsum()

            ax.plot(t, y, drawstyle="steps-post", label=label)

            total_pnl = float(y.iloc[-1]) if len(y) else 0.0
            n = int(len(d))
            avg = total_pnl / n if n else 0.0
            win = float((leg > 0).mean() * 100.0) if n else 0.0

            rows.append([label, total_pnl, n, avg, win])

        # TOTAL
        if not df_sel.empty:
            t_all = pd.to_datetime(df_sel["TradeTime"], errors="coerce")
            y_all = pd.to_numeric(df_sel["_pnl"], errors="coerce").fillna(0.0).cumsum()
            if len(y_all):
                ax.plot(t_all, y_all, drawstyle="steps-post", color="grey", alpha=0.6, label="TOTAL")

        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km))
        ax.grid(True, alpha=0.25)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

        self.chart.draw()

        self._render_rows(rows, total_trades_all)


    def _render_rows(self, rows, total_trades_all: int):
        # KPIs
        for row, kpi in zip(rows, [self.kpi_call, self.kpi_put]):
            total_pnl = float(row[1])
            n = int(row[2])
            win = float(row[4])

            kpi["pnl"].config(
                text=f"{total_pnl:,.1f} €",
                style=("KpiValuePos.TLabel" if total_pnl > 0 else "KpiValueNeg.TLabel" if total_pnl < 0 else "KpiValue.TLabel"),
            )
            kpi["total"].config(text=f"{total_trades_all:,}")
            kpi["pairs"].config(text=f"{n:,}")
            kpi["win"].config(text=f"{win:.1f} %")


