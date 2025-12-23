# Python 3.12
from __future__ import annotations

from tkinter import ttk
import tkinter as tk
import pandas as pd

from analytics.filters import apply_row_filters
from analytics.underlying import (
    best_worst_underlyings,
    cum_pnl_by_underlying,
    cum_pnl_total,
    underlying_table,
)
from ui.widgets.mpl_chart import MplChart
from ui.widgets.table import DataTable
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


def _fmt_y(v, _pos):
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 10_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.0f}"


def _pair_mask_any(df: pd.DataFrame) -> pd.Series:
    """True if row belongs to any pair/group (L1/L2/L3)."""
    m = pd.Series(False, index=df.index)
    if "PairID_L1" in df.columns:
        m |= pd.to_numeric(df["PairID_L1"], errors="coerce").fillna(-1).astype(int) >= 0
    if "GroupID_L2" in df.columns:
        m |= pd.to_numeric(df["GroupID_L2"], errors="coerce").fillna(-1).astype(int) >= 0
    if "PairID_L3" in df.columns:
        m |= pd.to_numeric(df["PairID_L3"], errors="coerce").fillna(-1).astype(int) >= 0
    return m


class UnderlyingView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self._df_master: pd.DataFrame | None = None
        self._available_days: list[str] = ["ALL"]
        self._available_cps: list[str] = ["ALL"]
        self._available_unds: list[str] = ["ALL"]

        card = ttk.Frame(self, style="Panel.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        # 2/3 vs 1/3
        card.grid_columnconfigure(0, weight=2)
        card.grid_columnconfigure(1, weight=1)
        card.grid_rowconfigure(1, weight=1)

        # Header + filters
        top = ttk.Frame(card, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 8))

        ttk.Label(top, text="Underlying", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        flt = ttk.Frame(top, style="Panel.TFrame")
        flt.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(flt, text="Level:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.var_level = tk.StringVar(value="ALL")
        ttk.Combobox(
            flt, width=6, state="readonly",
            values=["ALL", "L1", "L2", "L3"],
            textvariable=self.var_level,
        ).grid(row=0, column=1, padx=(6, 12))

        ttk.Label(flt, text="Day:", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.var_day = tk.StringVar(value="ALL")
        self.cmb_day = ttk.Combobox(
            flt, width=12, state="readonly",
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

        # Left: chart
        self.chart = MplChart(card, title="Cum PnL (Top 3 best + Top 3 worst)")
        self.chart.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        # Right: table
        panel = ttk.Frame(card, style="Panel.TFrame")
        panel.grid(row=1, column=1, sticky="nsew", padx=(0, 14), pady=(0, 14))
        panel.grid_rowconfigure(1, weight=1)

        ttk.Label(panel, text="Underlying table", style="Muted.TLabel") \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        cols = ["Underlying", "Total PnL", "Total Trades", "Pair Trades"]
        self.table = DataTable(panel, cols, height=18)
        self.table.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

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

    def _reset(self):
        self.var_level.set("ALL")
        self.var_day.set("ALL")
        self.var_cp.set("ALL")
        self.var_und.set("ALL")
        self._rerender()

    def _rerender(self):
        df = self._df_master
        if df is None or df.empty:
            self.chart.clear()
            self.chart.draw()
            self.table.clear()
            return

        day = self.var_day.get().strip()
        level = self.var_level.get().strip().upper()
        cp_sel = self.var_cp.get().strip()
        und_sel = self.var_und.get().strip()

        # Base WITHOUT level (denominator for Total Trades)
        df_base = apply_row_filters(df, day=day, level="ALL", cp_sub="", und_sub="")

        # Dropdown exact-match filters
        if cp_sel and cp_sel != "ALL" and "Counterparty" in df_base.columns:
            df_base = df_base[df_base["Counterparty"].astype("string") == cp_sel]
        if und_sel and und_sel != "ALL" and "UND_NAME" in df_base.columns:
            df_base = df_base[df_base["UND_NAME"].astype("string") == und_sel]

        if df_base.empty:
            self.chart.clear()
            self.chart.ax.set_title("Cum PnL (no data after filters)")
            self.chart.draw()
            self.table.clear()
            return

        # Selected set WITH level; if ALL -> paired only
        if level == "ALL":
            df_sel = df_base.loc[_pair_mask_any(df_base)].copy()
        else:
            df_lvl = apply_row_filters(df, day=day, level=level, cp_sub="", und_sub="")
            if cp_sel and cp_sel != "ALL" and "Counterparty" in df_lvl.columns:
                df_lvl = df_lvl[df_lvl["Counterparty"].astype("string") == cp_sel]
            if und_sel and und_sel != "ALL" and "UND_NAME" in df_lvl.columns:
                df_lvl = df_lvl[df_lvl["UND_NAME"].astype("string") == und_sel]
            df_sel = df_lvl

        # ---- Chart ----
        self.chart.clear()
        ax = self.chart.ax
        ax.set_title("Cum PnL (Top 3 best + Top 3 worst)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cum PnL")

        # If user selected a specific underlying -> show only that
        if und_sel and und_sel != "ALL":
            unds = [und_sel]
        else:
            best, worst = best_worst_underlyings(df_sel if not df_sel.empty else df_base, n=3)
            unds = worst + best

        series = cum_pnl_by_underlying(df_sel, unds) if not df_sel.empty else {}
        for u in unds:
            if u in series:
                t, y = series[u]
                ax.plot(t, y, drawstyle="steps-post", label=u)

        base_for_total = df_sel if not df_sel.empty else df_base
        ttot, ytot = cum_pnl_total(base_for_total)
        if len(ttot) > 0:
            ax.plot(ttot, ytot, drawstyle="steps-post", linewidth=2.0, label="TOTAL")

        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

        # X ticks: first, 2 mid, last
        tmin = pd.to_datetime(base_for_total["TradeTime"], errors="coerce").min()
        tmax = pd.to_datetime(base_for_total["TradeTime"], errors="coerce").max()
        if pd.notna(tmin) and pd.notna(tmax) and tmin < tmax:
            ax.set_xlim(tmin, tmax)
            dt = (tmax - tmin) / 3
            xticks = [tmin, tmin + dt, tmin + 2 * dt, tmax]
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
        self.chart.draw()

        # ---- Table ----
        # Table based on selected set
        tbl = underlying_table(df_sel if not df_sel.empty else df_base)
        if tbl is None or tbl.empty:
            self.table.clear()
            return

        tbl2 = tbl.copy()

        # Total Trades from df_base (no level), exact same filtering
        base_counts = df_base["UND_NAME"].astype("string").fillna("").value_counts()
        if "Underlying" in tbl2.columns:
            tbl2["Total Trades"] = tbl2["Underlying"].map(base_counts).fillna(0).astype(int)

        # Sort: most negative on top
        if "Total PnL" in tbl2.columns:
            tbl2 = tbl2.sort_values("Total PnL", ascending=True, kind="mergesort")

        # Format PnL: 1 decimal + €
        if "Total PnL" in tbl2.columns:
            tbl2["Total PnL"] = (
                pd.to_numeric(tbl2["Total PnL"], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"{x:,.1f} €")
            )

        self.table.set_dataframe(tbl2)
