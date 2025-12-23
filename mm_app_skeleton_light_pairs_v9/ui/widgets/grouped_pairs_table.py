# Python 3.12
from __future__ import annotations

from tkinter import ttk
from typing import Dict, List, Sequence, Tuple

import pandas as pd


class GroupedPairsTable(ttk.Frame):
    """Treeview with group parent rows (expand/collapse).

    Notes:
    - Uses show='tree headings' so we can have a dedicated Group column (#0).
    - Uses dict-record iteration (NOT itertuples _asdict) so columns like 'b/s' work.
    """

    def __init__(self, parent, columns: Sequence[str], *, height: int = 20, enable_sort: bool = False):
        super().__init__(parent)

        self.columns = list(columns)
        self.enable_sort = enable_sort
        self._groups: Dict[str, List[Tuple[str, List[object]]]] = {}  # group -> list[(side_tag, values)]

        self.tree = ttk.Treeview(self, columns=self.columns, show="tree headings", height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Group column (#0)
        self.tree.heading("#0", text="Group")
        self.tree.column("#0", width=120, anchor="center", stretch=False)

        for c in self.columns:
            if self.enable_sort:
                self.tree.heading(c, text=c, command=lambda col=c: self._on_heading_click(col))
            else:
                self.tree.heading(c, text=c)
            w = 120
            cl = c.lower()
            if cl == "tradetime":
                w = 175
            elif cl in ("isin", "und_name", "counterparty"):
                w = 150
            elif cl == "quantity":
                w = 95
            elif cl in ("b/s", "levels", "δt", "dt"):
                w = 80
            elif cl == "pnl":
                w = 115
            self.tree.column(c, width=w, anchor="center", stretch=True)

        self._sort_state: dict[str, bool] = {}
        self.tree.tag_configure("group", font=("Segoe UI", 10, "bold"))

    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._groups.clear()

    def tag_configure(self, tag: str, **kwargs):
        self.tree.tag_configure(tag, **kwargs)

    def set_groups(self, df: pd.DataFrame, *, group_col: str, side_col: str):
        self.clear()
        if df is None or df.empty:
            return

        # Build groups safely even with weird column names
        self._groups = {}
        records = df.to_dict(orient="records")
        for rec in records:
            g = str(rec.get(group_col, ""))
            side = str(rec.get(side_col, ""))
            values = [rec.get(c, "") for c in self.columns]
            side_u = side.strip().upper()
            side_tag = "buy" if side_u.startswith("B") else ("sell" if side_u.startswith("S") else "side")
            self._groups.setdefault(g, []).append((side_tag, values))

        self._render_tree(open_all=True)

    def _render_tree(self, *, open_all: bool = True):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for g in self._groups.keys():
            parent = self.tree.insert("", "end", text=g, values=[""] * len(self.columns), tags=("group",), open=open_all)
            for side_tag, values in self._groups[g]:
                self.tree.insert(parent, "end", text="", values=values, tags=(side_tag,))

    # Optional sorting (disabled by default)
    def _on_heading_click(self, col: str):
        if not self._groups:
            return

        asc = self._sort_state.get(col, True)
        self._sort_state[col] = not asc

        col_idx = self.columns.index(col)

        def sort_key(values: List[object]):
            v = values[col_idx]
            if v == "" or v is None:
                return (1, None)
            if col.lower() == "tradetime":
                try:
                    return (0, pd.to_datetime(v, errors="coerce"))
                except Exception:
                    return (0, v)
            try:
                return (0, float(v))
            except Exception:
                return (0, str(v))

        for g, rows in self._groups.items():
            self._groups[g] = sorted(rows, key=lambda r: sort_key(r[1]), reverse=not asc)

        self._render_tree(open_all=True)
