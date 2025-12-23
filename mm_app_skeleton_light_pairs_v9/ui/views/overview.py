# Python 3.12
from __future__ import annotations

from tkinter import ttk
import pandas as pd

from ui.widgets.table import DataTable


class OverviewView(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="TFrame")

        header = ttk.Label(self, text="Overview", style="Title.TLabel")
        header.pack(anchor="w", padx=14, pady=(12, 6))

        self.kpi_frame = ttk.Frame(self, style="TFrame")
        self.kpi_frame.pack(fill="x", padx=14)

        self.kpi_trades = ttk.Label(self.kpi_frame, text="Trades: -", style="Muted.TLabel")
        self.kpi_trades.pack(side="left", padx=(0, 20))

        self.kpi_pairs = ttk.Label(self.kpi_frame, text="Pairs L1: - | Groups L2: - | Pairs L3: -", style="Muted.TLabel")
        self.kpi_pairs.pack(side="left")

        # Dynamic columns: show ALL columns from the input dataframe
        self.table = DataTable(self, columns=None, height=20)
        self.table.pack(fill="both", expand=True, padx=14, pady=14)

    def render(self, summary: dict, df_input: pd.DataFrame | None) -> None:
        self.kpi_trades.config(text=f"Trades: {summary.get('n_trades', 0)}")
        self.kpi_pairs.config(
            text=f"Pairs L1: {summary.get('n_pairs_L1', 0)} | Groups L2: {summary.get('n_groups_L2', 0)} | Pairs L3: {summary.get('n_pairs_L3', 0)}"
        )
        self.table.set_dataframe(df_input if df_input is not None else pd.DataFrame())
