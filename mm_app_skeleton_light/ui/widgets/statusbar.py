# Python 3.12
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class StatusBar(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Panel.TFrame")
        self.var = tk.StringVar(value="Ready.")
        self.label = ttk.Label(self, textvariable=self.var, style="Muted.TLabel")
        self.label.pack(side="left", padx=10, pady=6, fill="x", expand=True)

    def set(self, msg: str) -> None:
        self.var.set(msg)
