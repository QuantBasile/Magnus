# Python 3.12
from __future__ import annotations

from tkinter import ttk
import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

from analytics.filters import apply_row_filters
from analytics.deltat import deltat_bins_and_pnl, pnl_leg_series
from ui.widgets.mpl_chart import MplChart
from ui.widgets.table import DataTable


def _fmt_km_10k(v, _pos):
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 10_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.0f}"


def _pair_mask_any(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    if "PairID_L1" in df.columns:
        m |= pd.to_numeric(df["PairID_L1"], errors="coerce").fillna(-1).astype(int) >= 0
    if "GroupID_L2" in df.columns:
        m |= pd.to_numeric(df["GroupID_L2"], errors="coerce").fillna(-1).astype(int) >= 0
    if "PairID_L3" in df.columns:
        m |= pd.to_numeric(df["PairID_L3"], errors="coerce").fillna(-1).astype(int) >= 0
    return m


class DeltaTView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self._df_master: pd.DataFrame | None = None
        self._available_days: list[str] = ["ALL"]
        self._available_cps: list[str] = ["ALL"]
        self._available_unds: list[str] = ["ALL"]

        card = ttk.Frame(self, style="Panel.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        # Layout: chart 2/3, table 1/3
        card.grid_columnconfigure(0, weight=2)
        card.grid_columnconfigure(1, weight=1)
        card.grid_rowconfigure(2, weight=1)

        # ── Header + filters ──────────────────────────────────────────────
        top = ttk.Frame(card, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 6))

        ttk.Label(top, text="Δt Pattern", style="Title.TLabel").grid(row=0, column=0, sticky="w")

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

        # ── KPIs ──────────────────────────────────────────────────────────
        kpi = ttk.Frame(card, style="Panel.TFrame")
        kpi.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))
        kpi.grid_columnconfigure((0, 1, 2), weight=1)

        self.k_total_pnl = self._kpi(kpi, "Total PnL", "-")
        self.k_total_trades = self._kpi(kpi, "Total Trades", "-")
        self.k_pair_trades = self._kpi(kpi, "Pair Trades", "-")

        self.k_total_pnl["frame"].grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.k_total_trades["frame"].grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.k_pair_trades["frame"].grid(row=0, column=2, sticky="ew")

        # ── Chart (left) ────────────────────────────────────────────────
        self.chart = MplChart(card, title="PnL by Δt bins")
        self.chart.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

        # ── Table (right) ───────────────────────────────────────────────
        panel = ttk.Frame(card, style="Panel.TFrame")
        panel.grid(row=2, column=1, sticky="nsew", padx=(0, 14), pady=(0, 14))
        panel.grid_rowconfigure(1, weight=1)

        ttk.Label(panel, text="Δt bins", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )

        self.table = DataTable(panel, ["Δt", "Total PnL"], height=18)
        self.table.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        # Disable heading-click sorting (make headings inert)
        for col in ["Δt", "Total PnL"]:
            self.table.tree.heading(col, command=lambda: None)
        
        # Align PnL column to the right
        self.table.tree.column("Total PnL", anchor="e")



    def _kpi(self, parent, title, value):
        fr = ttk.Frame(parent, style="KpiCard.TFrame")
        fr.grid_columnconfigure(0, weight=1)
        ttk.Label(fr, text=title, style="KpiTitle.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
        lbl = ttk.Label(fr, text=value, style="KpiValue.TLabel")
        lbl.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))
        return {"frame": fr, "label": lbl}

    def set_data(self, df_master: pd.DataFrame | None):
        self._df_master = df_master

        days, cps, unds = ["ALL"], ["ALL"], ["ALL"]

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
        self.cmb_day.configure(values=days)
        if self.var_day.get() not in days:
            self.var_day.set("ALL")

        self._available_cps = cps
        self.cmb_cp.configure(values=cps)
        if self.var_cp.get() not in cps:
            self.var_cp.set("ALL")

        self._available_unds = unds
        self.cmb_und.configure(values=unds)
        if self.var_und.get() not in unds:
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
            for k in [self.k_total_pnl, self.k_total_trades, self.k_pair_trades]:
                k["label"].config(text="-", style="KpiValue.TLabel")
            return

        day = self.var_day.get().strip()
        level = self.var_level.get().strip().upper()
        cp = self.var_cp.get().strip()
        und = self.var_und.get().strip()

        # Base = without level (denominator for Total Trades)
        df_base = apply_row_filters(df, day=day, level="ALL", cp_sub="", und_sub="")
        if cp != "ALL" and "Counterparty" in df_base.columns:
            df_base = df_base[df_base["Counterparty"].astype("string") == cp]
        if und != "ALL" and "UND_NAME" in df_base.columns:
            df_base = df_base[df_base["UND_NAME"].astype("string") == und]

        # Selected = with level OR paired-only when ALL
        if level == "ALL":
            df_sel = df_base.loc[_pair_mask_any(df_base)].copy()
        else:
            df_lvl = apply_row_filters(df, day=day, level=level, cp_sub="", und_sub="")
            if cp != "ALL" and "Counterparty" in df_lvl.columns:
                df_lvl = df_lvl[df_lvl["Counterparty"].astype("string") == cp]
            if und != "ALL" and "UND_NAME" in df_lvl.columns:
                df_lvl = df_lvl[df_lvl["UND_NAME"].astype("string") == und]
            df_sel = df_lvl

        total_trades = int(len(df_base))
        pair_trades = int(len(df_sel))
        total_pnl = float(pd.to_numeric(pnl_leg_series(df_sel), errors="coerce").fillna(0.0).sum()) if not df_sel.empty else 0.0

        self.k_total_trades["label"].config(text=f"{total_trades:,}", style="KpiValue.TLabel")
        self.k_pair_trades["label"].config(text=f"{pair_trades:,}", style="KpiValue.TLabel")
        self.k_total_pnl["label"].config(
            text=f"{total_pnl:,.1f} €",
            style=("KpiValuePos.TLabel" if total_pnl > 0 else "KpiValueNeg.TLabel" if total_pnl < 0 else "KpiValue.TLabel"),
        )

        # ---- Compute bins ----
        bins = deltat_bins_and_pnl(df_sel)

        # ---- Chart ----
        self.chart.clear()
        ax = self.chart.ax
        ax.set_title("PnL by Δt (pair duration)")
        ax.set_ylabel("PnL")

        if bins is None or bins.empty:
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km_10k))
            ax.grid(True, axis="y", alpha=0.3)
            self.chart.draw()
            self.table.clear()
            return

        # sort bins naturally
        def _sort_key(lbl: str) -> float:
            s = str(lbl).strip()
            if s == "> 1h":
                return 1e9
            try:
                lo = float(s.split("-")[0].strip())
                return lo
            except Exception:
                return 1e8

        bins2 = bins.copy()
        bins2["_k"] = bins2["bin"].astype(str).map(_sort_key)
        bins2 = bins2.sort_values("_k", ascending=True).drop(columns="_k")

        labels = bins2["bin"].astype(str).tolist()
        vals = pd.to_numeric(bins2["pnl"], errors="coerce").fillna(0.0).to_numpy()
        colors = np.where(vals >= 0, "#0f766e", "#b91c1c")

        ax.bar(labels, vals, color=colors)
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_km_10k))
        ax.grid(True, axis="y", alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

        self.chart.draw()

        # ---- Table ----
        tbl = bins2.copy()
        tbl["pnl"] = pd.to_numeric(tbl["pnl"], errors="coerce").fillna(0.0)
        tbl["Total PnL"] = tbl["pnl"].map(lambda x: f"{x:,.1f} €")
        tbl = tbl.rename(columns={"bin": "Δt"})[["Δt", "Total PnL"]]
        self.table.set_dataframe(tbl)
