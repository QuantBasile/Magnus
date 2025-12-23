# Python 3.12
from __future__ import annotations

from tkinter import ttk
import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

from analytics.filters import apply_row_filters
from analytics.intraday import intraday_hourly_pnl, pnl_leg_series
from ui.widgets.mpl_chart import MplChart


def _fmt_km_10k(v, _pos):
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 10_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.0f}"


def _pair_mask_any(df: pd.DataFrame) -> pd.Series:
    """
    True for rows that belong to ANY pair/group (L1/L2/L3).
    """
    m = pd.Series(False, index=df.index)
    if "PairID_L1" in df.columns:
        m |= pd.to_numeric(df["PairID_L1"], errors="coerce").fillna(-1).astype(int) >= 0
    if "GroupID_L2" in df.columns:
        m |= pd.to_numeric(df["GroupID_L2"], errors="coerce").fillna(-1).astype(int) >= 0
    if "PairID_L3" in df.columns:
        m |= pd.to_numeric(df["PairID_L3"], errors="coerce").fillna(-1).astype(int) >= 0
    return m


class IntradayView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._df_master: pd.DataFrame | None = None

        self._available_days: list[str] = ["ALL"]
        self._available_cps: list[str] = ["ALL"]
        self._available_unds: list[str] = ["ALL"]

        card = ttk.Frame(self, style="Panel.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        # Title + filters
        top = ttk.Frame(card, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))

        ttk.Label(top, text="Intraday Pattern", style="Title.TLabel").grid(row=0, column=0, sticky="w")

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
            flt, width=10, state="readonly",
            values=self._available_days, textvariable=self.var_day
        )
        self.cmb_day.grid(row=0, column=3, padx=(6, 12))

        ttk.Label(flt, text="Counterparty:", style="Muted.TLabel").grid(row=0, column=4, sticky="w")
        self.var_cp = tk.StringVar(value="ALL")
        self.cmb_cp = ttk.Combobox(
            flt, width=16, state="readonly",
            values=self._available_cps, textvariable=self.var_cp
        )
        self.cmb_cp.grid(row=0, column=5, padx=(6, 12))

        ttk.Label(flt, text="Underlying:", style="Muted.TLabel").grid(row=0, column=6, sticky="w")
        self.var_und = tk.StringVar(value="ALL")
        self.cmb_und = ttk.Combobox(
            flt, width=20, state="readonly",
            values=self._available_unds, textvariable=self.var_und
        )
        self.cmb_und.grid(row=0, column=7, padx=(6, 12))

        ttk.Button(flt, text="Apply", command=self._rerender).grid(row=0, column=8, padx=(6, 0))
        ttk.Button(flt, text="Reset", command=self._reset).grid(row=0, column=9, padx=(8, 0))

        # KPIs
        kpi = ttk.Frame(card, style="Panel.TFrame")
        kpi.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        kpi.grid_columnconfigure((0, 1, 2), weight=1)

        self.k_total_pnl = self._kpi(kpi, "Total PnL", "-")
        self.k_total_trades = self._kpi(kpi, "Total Trades", "-")
        self.k_pair_trades = self._kpi(kpi, "Pair Trades", "-")

        self.k_total_pnl["frame"].grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.k_total_trades["frame"].grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.k_pair_trades["frame"].grid(row=0, column=2, sticky="ew")

        # Chart
        self.chart = MplChart(card, title="Hourly PnL (1h bins)")
        self.chart.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

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
        cps = ["ALL"]
        unds = ["ALL"]

        if df_master is not None and not df_master.empty:
            if "TradeTime" in df_master.columns:
                tt = pd.to_datetime(df_master["TradeTime"], errors="coerce")
                uniq_days = sorted(tt.dt.strftime("%Y-%m-%d").dropna().unique().tolist())
                days += uniq_days[:800]

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
            for k in [self.k_total_pnl, self.k_total_trades, self.k_pair_trades]:
                k["label"].config(text="-", style="KpiValue.TLabel")
            return

        day = self.var_day.get().strip()
        level = self.var_level.get().strip()
        cp_sel = self.var_cp.get().strip()
        und_sel = self.var_und.get().strip()

        # Base WITHOUT level (denominator)
        df_base = apply_row_filters(df, day=day, level="ALL", cp_sub="", und_sub="")

        # Apply exact-match dropdown filters (CP/UND)
        if cp_sel and cp_sel != "ALL" and "Counterparty" in df_base.columns:
            df_base = df_base[df_base["Counterparty"].astype("string") == cp_sel]
        if und_sel and und_sel != "ALL" and "UND_NAME" in df_base.columns:
            df_base = df_base[df_base["UND_NAME"].astype("string") == und_sel]

        # Selected level dataframe
        if level == "ALL":
            # For ALL: we want "pair trades" = any paired rows (not all trades)
            pm = _pair_mask_any(df_base)
            df_sel = df_base.loc[pm].copy()
        else:
            df_lvl = apply_row_filters(df, day=day, level=level, cp_sub="", und_sub="")
            if cp_sel and cp_sel != "ALL" and "Counterparty" in df_lvl.columns:
                df_lvl = df_lvl[df_lvl["Counterparty"].astype("string") == cp_sel]
            if und_sel and und_sel != "ALL" and "UND_NAME" in df_lvl.columns:
                df_lvl = df_lvl[df_lvl["UND_NAME"].astype("string") == und_sel]
            df_sel = df_lvl

        total_trades = int(len(df_base))
        pair_trades = int(len(df_sel))

        # Total PnL from selected set (paired/level)
        pnl_total = float(pd.to_numeric(pnl_leg_series(df_sel), errors="coerce").fillna(0.0).sum()) if not df_sel.empty else 0.0

        def fmt_eur(x: float) -> str:
            return f"{x:,.1f} €"

        self.k_total_pnl["label"].config(
            text=fmt_eur(pnl_total),
            style=("KpiValuePos.TLabel" if pnl_total > 0 else "KpiValueNeg.TLabel" if pnl_total < 0 else "KpiValue.TLabel"),
        )
        self.k_total_trades["label"].config(text=f"{total_trades:,}", style="KpiValue.TLabel")
        self.k_pair_trades["label"].config(text=f"{pair_trades:,}", style="KpiValue.TLabel")

        # Plot: hourly bins (1h) of accumulated pnl, but show only 08:00..22:00
        s = intraday_hourly_pnl(df_sel)  # index 0..23

        hours = np.arange(8, 23)  # 8..22 inclusive
        vals = s.reindex(hours).to_numpy(dtype=float)
        colors = np.where(vals >= 0, "#0f766e", "#b91c1c")  # green/red

        self.chart.clear()
        ax = self.chart.ax
        ax.set_title("Hourly PnL (1h bins)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("PnL")

        ax.bar(hours, vals, width=0.9, color=colors)

        # Fixed session axis
        ax.set_xlim(7.5, 22.5)
        xt = [8, 10, 12, 14, 16, 18, 20, 22]
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{h:02d}:00" for h in xt])

        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km_10k))
        ax.grid(True, alpha=0.25)

        self.chart.draw()
