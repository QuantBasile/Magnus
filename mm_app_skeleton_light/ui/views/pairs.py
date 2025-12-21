# Python 3.12
from __future__ import annotations

from tkinter import ttk
import pandas as pd

from ui.widgets.table import DataTable


class PairsView(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="TFrame")

        header = ttk.Label(self, text="Pairs", style="Title.TLabel")
        header.pack(anchor="w", padx=14, pady=(12, 6))

        self.note = ttk.Label(
            self,
            text="Aquí mostraremos pairs L1 / groups L2 / pairs L3 (por ahora un preview).",
            style="Muted.TLabel"
        )
        self.note.pack(anchor="w", padx=14, pady=(0, 10))

        self.table = DataTable(self, height=22)
        self.table.pack(fill="both", expand=True, padx=14, pady=14)

    def render(self, df_pairs: pd.DataFrame | None) -> None:
        self.table.set_dataframe(df_pairs if df_pairs is not None else pd.DataFrame())
