# Python 3.12
from __future__ import annotations

from tkinter import ttk
import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

from analytics.filters import apply_row_filters
from analytics.daily_und import daily_aggregate_by_underlying
from ui.widgets.mpl_chart import MplChart
from ui.widgets.table import DataTable


def _fmt_km(v, _pos):
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 100_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.0f}"


class DailyUnderlyingView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._df_master: pd.DataFrame | None = None
        self._available_days: list[str] = ["ALL"]
        self._available_unds: list[str] = ["ALL"]

        card = ttk.Frame(self, style="Panel.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        card.grid_columnconfigure(0, weight=3)  # plot
        card.grid_columnconfigure(1, weight=1)  # table
        card.grid_rowconfigure(2, weight=1)

        # Title + filters
        top = ttk.Frame(card, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 6))

        ttk.Label(top, text="Daily PnL by Underlying", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        flt = ttk.Frame(top, style="Panel.TFrame")
        flt.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(flt, text="Underlying:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.var_underlying = tk.StringVar(value="ALL")
        self.cmb_und = ttk.Combobox(
            flt, width=22, state="readonly",
            values=self._available_unds, textvariable=self.var_underlying
        )
        self.cmb_und.grid(row=0, column=1, padx=(6, 12))

        ttk.Label(flt, text="Level:", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.var_level = tk.StringVar(value="ALL")
        ttk.Combobox(
            flt, width=6, state="readonly",
            values=["ALL", "L1", "L2", "L3"],
            textvariable=self.var_level
        ).grid(row=0, column=3, padx=(6, 12))

        ttk.Label(flt, text="Day:", style="Muted.TLabel").grid(row=0, column=4, sticky="w")
        self.var_day = tk.StringVar(value="ALL")
        self.cmb_day = ttk.Combobox(
            flt, width=12, state="readonly",
            values=self._available_days, textvariable=self.var_day
        )
        self.cmb_day.grid(row=0, column=5, padx=(6, 12))

        ttk.Label(flt, text="CP contains:", style="Muted.TLabel").grid(row=0, column=6, sticky="w")
        self.var_cp = tk.StringVar(value="")
        ttk.Entry(flt, width=18, textvariable=self.var_cp).grid(row=0, column=7, padx=(6, 12))

        ttk.Button(flt, text="Apply", command=self._rerender).grid(row=0, column=8, padx=(6, 0))
        ttk.Button(flt, text="Reset", command=self._reset).grid(row=0, column=9, padx=(8, 0))

        # KPIs
        kpi = ttk.Frame(card, style="Panel.TFrame")
        kpi.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))
        kpi.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.k_total_pnl = self._kpi(kpi, "Total PnL", "-")
        self.k_total_trades = self._kpi(kpi, "Total Trades", "-")
        self.k_pair_trades = self._kpi(kpi, "Pair Trades", "-")
        self.k_green = self._kpi(kpi, "Green Days %", "-")

        self.k_total_pnl["frame"].grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.k_total_trades["frame"].grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.k_pair_trades["frame"].grid(row=0, column=2, sticky="ew", padx=(0, 10))
        self.k_green["frame"].grid(row=0, column=3, sticky="ew")

        # Chart
        self.chart = MplChart(card, title="Daily PnL")
        self.chart.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

        # Table
        panel = ttk.Frame(card, style="Panel.TFrame")
        panel.grid(row=2, column=1, sticky="nsew", padx=(0, 14), pady=(0, 14))
        panel.grid_rowconfigure(1, weight=1)

        ttk.Label(panel, text="Days (chronological)", style="Muted.TLabel") \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        cols = ["Date", "Total PnL", "Total Trades", "Pair Trades"]
        self.table = DataTable(panel, cols, height=18)
        self.table.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _kpi(self, parent, title, value):
        fr = ttk.Frame(parent, style="KpiCard.TFrame")
        fr.grid_columnconfigure(0, weight=1)
        ttk.Label(fr, text=title, style="KpiTitle.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
        lbl = ttk.Label(fr, text=value, style="KpiValue.TLabel")
        lbl.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))
        return {"frame": fr, "label": lbl}

    def set_data(self, df_master: pd.DataFrame | None):
        self._df_master = df_master

        days = ["ALL"]
        unds = ["ALL"]

        if df_master is not None and not df_master.empty:
            if "TradeTime" in df_master.columns:
                tt = pd.to_datetime(df_master["TradeTime"], errors="coerce")
                uniq_days = sorted(tt.dt.strftime("%Y-%m-%d").dropna().unique().tolist())
                days += uniq_days[:500]

            if "UND_NAME" in df_master.columns:
                uniq_unds = sorted(df_master["UND_NAME"].astype("string").fillna("").unique().tolist())
                uniq_unds = [u for u in uniq_unds if u != ""]
                unds += uniq_unds[:2000]

        self._available_days = days
        self.cmb_day.configure(values=self._available_days)
        if self.var_day.get() not in self._available_days:
            self.var_day.set("ALL")

        self._available_unds = unds
        self.cmb_und.configure(values=self._available_unds)
        if self.var_underlying.get() not in self._available_unds:
            self.var_underlying.set("ALL")

        self._rerender()

    def _reset(self):
        self.var_underlying.set("ALL")
        self.var_level.set("ALL")
        self.var_day.set("ALL")
        self.var_cp.set("")
        self._rerender()

    def _rerender(self):
        df = self._df_master
        if df is None or df.empty:
            self.chart.clear()
            self.chart.draw()
            self.table.clear()
            for k in [self.k_total_pnl, self.k_total_trades, self.k_pair_trades, self.k_green]:
                k["label"].config(text="-", style="KpiValue.TLabel")
            return

        underlying = self.var_underlying.get().strip()
        day = self.var_day.get().strip()
        level = self.var_level.get().strip()
        cp_sub = self.var_cp.get().strip()

        # Base WITHOUT level (denominator for Total Trades)
        df_base = apply_row_filters(df, day=day, level="ALL", cp_sub=cp_sub)
        # Level-filtered dataframe drives chart/table/pair stats
        df_lvl = apply_row_filters(df, day=day, level=level, cp_sub=cp_sub)

        agg_base = daily_aggregate_by_underlying(df_base, underlying=underlying)
        agg_lvl = daily_aggregate_by_underlying(df_lvl, underlying=underlying)

        if agg_base.empty:
            self.chart.clear()
            self.chart.ax.set_title("Daily PnL (no data after filters)")
            self.chart.draw()
            self.table.clear()
            for k in [self.k_total_pnl, self.k_total_trades, self.k_pair_trades, self.k_green]:
                k["label"].config(text="-", style="KpiValue.TLabel")
            return

        # KPIs
        total_trades = int(pd.to_numeric(agg_base["Total Trades"], errors="coerce").fillna(0).sum())

        if not agg_lvl.empty:
            total_pnl = float(pd.to_numeric(agg_lvl["Total PnL"], errors="coerce").fillna(0.0).sum())
            pair_trades = int(pd.to_numeric(agg_lvl["Pair Trades"], errors="coerce").fillna(0).sum())
            green = float((pd.to_numeric(agg_lvl["Total PnL"], errors="coerce").fillna(0.0) > 0).mean() * 100.0)
        else:
            total_pnl = 0.0
            pair_trades = 0
            green = 0.0

        def fmt_eur(x: float) -> str:
            return f"{x:,.1f} €"

        self.k_total_pnl["label"].config(
            text=fmt_eur(total_pnl),
            style=("KpiValuePos.TLabel" if total_pnl > 0 else "KpiValueNeg.TLabel" if total_pnl < 0 else "KpiValue.TLabel"),
        )
        self.k_total_trades["label"].config(text=f"{total_trades:,}", style="KpiValue.TLabel")
        self.k_pair_trades["label"].config(text=f"{pair_trades:,}", style="KpiValue.TLabel")
        self.k_green["label"].config(text=f"{green:,.1f} %", style="KpiValue.TLabel")

        if agg_lvl.empty:
            self.table.clear()
            self.chart.clear()
            self.chart.ax.set_title("Daily PnL (no data after Level filter)")
            self.chart.draw()
            return

        # Table chronological, Total Trades from agg_base (no level)
        tdf = agg_lvl.copy()
        tdf["_d"] = pd.to_datetime(tdf["Date"], errors="coerce")
        tdf = tdf.sort_values("_d", ascending=True, kind="mergesort").drop(columns="_d")

        base_trades_by_day = (
            agg_base.set_index("Date")["Total Trades"]
            if ("Date" in agg_base.columns and "Total Trades" in agg_base.columns)
            else pd.Series(dtype="int64")
        )

        show = tdf[["Date", "Total PnL", "Total Trades", "Pair Trades"]].copy()
        show["Total Trades"] = show["Date"].map(base_trades_by_day).fillna(0).astype(int)
        show["Total PnL"] = pd.to_numeric(show["Total PnL"], errors="coerce").fillna(0.0).map(lambda x: f"{x:,.1f} €")
        self.table.set_dataframe(show)

        # Plot daily histogram green/red
        self.chart.clear()
        ax = self.chart.ax
        ax.set_title("Daily PnL")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")

        dts = pd.to_datetime(tdf["Date"], errors="coerce")
        vals = pd.to_numeric(tdf["Total PnL"], errors="coerce").fillna(0.0).to_numpy()
        colors = np.where(vals >= 0, "#0f766e", "#b91c1c")

        ax.bar(dts, vals, width=0.8, color=colors)

        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km))
        ax.grid(True, alpha=0.25)

        # X ticks: first, 2 mid, last (max 4)
        tmin = dts.min()
        tmax = dts.max()
        if pd.notna(tmin) and pd.notna(tmax) and tmin < tmax:
            ax.set_xlim(tmin, tmax)
            dt = (tmax - tmin) / 3
            xticks = [tmin, tmin + dt, tmin + 2 * dt, tmax]
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        self.chart.draw()
