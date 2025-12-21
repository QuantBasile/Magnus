# Python 3.12
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Sequence, Optional

import pandas as pd


class DataTable(ttk.Frame):
    """Minimal Treeview wrapper: set dataframe, auto columns."""

    def __init__(self, master, *, height: int = 18):
        super().__init__(master, style="Panel.TFrame")

        self.tree = ttk.Treeview(self, columns=(), show="headings", height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set_dataframe(self, df: pd.DataFrame, *, max_rows: int = 2000) -> None:
        # Clear
        for c in self.tree["columns"]:
            self.tree.heading(c, text="")
        self.tree.delete(*self.tree.get_children())

        if df is None or df.empty:
            self.tree["columns"] = ()
            return

        dfx = df.head(max_rows)

        cols = list(dfx.columns)
        self.tree["columns"] = cols

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="w", stretch=True)

        # Insert rows (stringified)
        values = dfx.astype(str).to_numpy()
        for row in values:
            self.tree.insert("", "end", values=tuple(row))
