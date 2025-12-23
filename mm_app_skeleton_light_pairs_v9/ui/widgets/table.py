# Python 3.12
from __future__ import annotations

from tkinter import ttk
from typing import Iterable, Sequence, Optional

import pandas as pd


class DataTable(ttk.Frame):
    """ttk.Treeview table
    - optional sort when clicking headers
    - supports per-row tags for coloring
    - can work in dynamic-column mode (columns=None) where columns come from df
    """

    def __init__(self, parent, columns: Optional[Sequence[str]] = None, *, height: int = 18):
        super().__init__(parent)

        self.columns = list(columns) if columns is not None else []
        self.dynamic = columns is None
        self._df: pd.DataFrame | None = None
        self._sort_state: dict[str, bool] = {}

        self.tree = ttk.Treeview(self, columns=self.columns, show="headings", height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        if self.columns:
            self._setup_columns(self.columns)

    def _setup_columns(self, cols: Sequence[str]):
        self.columns = list(cols)
        self.tree["columns"] = self.columns

        for c in self.columns:
            self.tree.heading(c, text=c, command=lambda col=c: self._on_heading_click(col))
            w = 120
            cl = c.lower()
            anchor="center"
            if cl in ("tradetime", "time", "datetime"):
                anchor="w"
                w = 175
            elif cl in ("isin", "und_name", "counterparty"):
                w = 150
            elif cl in ("quantity",):
                w = 95
            elif cl in ("b/s", "side", "levels", "δt", "dt"):
                w = 80
            elif cl in ("pnl",):
                anchor="e"
                w = 115
            if "pnl" in cl or "trade" in cl:
                anchor = "e"   # derecha
            self.tree.column(c, width=w, anchor=anchor, stretch=True)

    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._df = None

    def set_tag_style(self, tag: str, *, background: str | None = None, foreground: str | None = None):
        kwargs = {}
        if background is not None:
            kwargs["background"] = background
        if foreground is not None:
            kwargs["foreground"] = foreground
        if kwargs:
            self.tree.tag_configure(tag, **kwargs)

    def set_dataframe(self, df: pd.DataFrame | None, *, tags: Iterable[str] | None = None, max_rows: int = 2000):
        self.clear()
        if df is None or df.empty:
            return

        df2 = df.head(max_rows).copy()

        if self.dynamic:
            self._setup_columns(list(df2.columns))

        # Ensure required columns exist
        for c in self.columns:
            if c not in df2.columns:
                df2[c] = ""
        df2 = df2[self.columns]
        self._df = df2

        tag_list = list(tags) if tags is not None else None
        if tag_list is not None and len(tag_list) != len(df2):
            raise ValueError("tags length must match df length")

        for i, row in enumerate(df2.itertuples(index=False)):
            vals = []
            for col, v in zip(self.columns, row):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    vals.append("")
                    continue
            
                cl = col.lower()
            
                # ---- thousands separator for numeric columns ----
                if isinstance(v, (int, float)) and (
                    "pnl" in cl or "trade" in cl
                ):
                    # integers: no decimals
                    if isinstance(v, int):
                        vals.append(f"{v:,}")
                    else:
                        # floats: keep one decimal (PnL)
                        vals.append(f"{v:,.1f}")
                else:
                    vals.append(v)

            itag = (tag_list[i],) if tag_list is not None else ()
            self.tree.insert("", "end", values=vals, tags=itag)

    def _on_heading_click(self, col: str):
        if self._df is None or self._df.empty:
            return

        asc = self._sort_state.get(col, True)
        self._sort_state[col] = not asc

        df = self._df.copy()
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            df_sorted = df.sort_values(by=col, ascending=asc, kind="mergesort", na_position="last")
        else:
            if col.lower() in ("tradetime", "datetime", "time"):
                dt = pd.to_datetime(s, errors="coerce")
                df_sorted = df.assign(_dt=dt).sort_values("_dt", ascending=asc, kind="mergesort", na_position="last").drop(columns=["_dt"])
            else:
                df_sorted = df.astype({col: "string"}).sort_values(by=col, ascending=asc, kind="mergesort", na_position="last")

        # preserve tags if present
        if "__rowtag__" in df_sorted.columns:
            tags = df_sorted["__rowtag__"].tolist()
            df_sorted = df_sorted.drop(columns=["__rowtag__"])
        else:
            tags = None

        self.set_dataframe(df_sorted, tags=tags)
