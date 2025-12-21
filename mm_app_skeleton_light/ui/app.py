# Python 3.12
from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from controller.state import AppState
from controller import actions
from ui.theme import Theme
from ui.widgets.statusbar import StatusBar
from ui.views.overview import OverviewView
from ui.views.pairs import PairsView


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MM Analytics (Tkinter)")
        self.geometry("1300x780")
        Theme.apply(self)

        self.state = AppState()

        self._q: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None

        self.topbar = ttk.Frame(self, style="Topbar.TFrame")
        self.topbar.pack(side="top", fill="x")

        self.body = ttk.Frame(self, style="TFrame")
        self.body.pack(side="top", fill="both", expand=True)

        self.status = StatusBar(self)
        self.status.pack(side="bottom", fill="x")

        self._build_topbar()
        self._build_body()

        self.after(100, self._poll_queue)

    def _build_topbar(self):
        pad_y = 10

        ttk.Label(self.topbar, text="Start:", style="Muted.TLabel").pack(side="left", padx=(12, 6), pady=pad_y)
        self.ent_start = ttk.Entry(self.topbar, width=12)
        self.ent_start.pack(side="left", pady=pad_y)
        self.ent_start.insert(0, "2024-12-02")

        ttk.Label(self.topbar, text="End:", style="Muted.TLabel").pack(side="left", padx=(10, 6), pady=pad_y)
        self.ent_end = ttk.Entry(self.topbar, width=12)
        self.ent_end.pack(side="left", pady=pad_y)
        self.ent_end.insert(0, "2024-12-11")

        presets = ["30s", "60s", "2m", "3m", "5m", "10m"]

        ttk.Label(self.topbar, text="L1 window:", style="Muted.TLabel").pack(side="left", padx=(18, 6), pady=pad_y)
        self.cmb_l1 = ttk.Combobox(self.topbar, width=6, values=presets, state="normal")
        self.cmb_l1.pack(side="left", pady=pad_y)
        self.cmb_l1.set("60s")

        ttk.Label(self.topbar, text="L2 window:", style="Muted.TLabel").pack(side="left", padx=(12, 6), pady=pad_y)
        self.cmb_l2 = ttk.Combobox(self.topbar, width=6, values=presets, state="normal")
        self.cmb_l2.pack(side="left", pady=pad_y)
        self.cmb_l2.set("10m")

        ttk.Label(self.topbar, text="L3 window:", style="Muted.TLabel").pack(side="left", padx=(12, 6), pady=pad_y)
        self.cmb_l3 = ttk.Combobox(self.topbar, width=6, values=presets, state="normal")
        self.cmb_l3.pack(side="left", pady=pad_y)
        self.cmb_l3.set("10m")

        ttk.Label(self.topbar, text="ΔQty:", style="Muted.TLabel").pack(side="left", padx=(12, 6), pady=pad_y)
        self.spn_dqty = ttk.Spinbox(self.topbar, from_=0, to=1_000_000, width=7)
        self.spn_dqty.pack(side="left", pady=pad_y)
        self.spn_dqty.delete(0, "end")
        self.spn_dqty.insert(0, "1")

        self.btn_load = ttk.Button(self.topbar, text="Load", command=self.on_load)
        self.btn_load.pack(side="left", padx=(18, 6), pady=pad_y)

        self.btn_pairs = ttk.Button(self.topbar, text="Run pairing", command=self.on_run_pairing, state="disabled")
        self.btn_pairs.pack(side="left", padx=6, pady=pad_y)

        ttk.Label(self.topbar, text=" ", style="Muted.TLabel").pack(side="right", padx=10)

    def _build_body(self):
        self.sidebar = ttk.Frame(self.body, style="Sidebar.TFrame", width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = ttk.Frame(self.body, style="TFrame")
        self.content.pack(side="left", fill="both", expand=True)

        self.views: dict[str, ttk.Frame] = {}
        self.views["overview"] = OverviewView(self.content)
        self.views["pairs"] = PairsView(self.content)

        for v in self.views.values():
            v.place(relx=0, rely=0, relwidth=1, relheight=1)

        ttk.Label(self.sidebar, text="Navigation", style="Title.TLabel").pack(anchor="w", padx=12, pady=(12, 6))
        ttk.Button(self.sidebar, text="Overview", style="Sidebar.TButton", command=lambda: self.show("overview")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Pairs", style="Sidebar.TButton", command=lambda: self.show("pairs")).pack(fill="x", padx=10, pady=4)

        self.show("overview")

    def show(self, key: str):
        self.views[key].tkraise()

    @staticmethod
    def _normalize_window(text: str) -> str:
        t = (text or "").strip().lower().replace(" ", "")
        t = t.replace("secs", "s").replace("sec", "s").replace("seconds", "s").replace("second", "s")
        t = t.replace("mins", "m").replace("min", "m").replace("minutes", "m").replace("minute", "m")
        if t.isdigit():
            return f"{t}s"
        return t

    def _set_busy(self, busy: bool, msg: str = ""):
        if busy:
            self.btn_load.config(state="disabled")
            self.btn_pairs.config(state="disabled")
            self.status.set(msg or "Working...")
        else:
            self.btn_load.config(state="normal")
            self.btn_pairs.config(state="normal" if self.state.df_clean is not None else "disabled")
            self.status.set(msg or "Ready.")

    def _sync_filters_from_ui(self):
        self.state.filters.start_date = self.ent_start.get().strip()
        self.state.filters.end_date = self.ent_end.get().strip()
        self.state.filters.l1_window = self._normalize_window(self.cmb_l1.get())
        self.state.filters.l2_window = self._normalize_window(self.cmb_l2.get())
        self.state.filters.l3_window = self._normalize_window(self.cmb_l3.get())
        try:
            self.state.filters.delta_qty = int(self.spn_dqty.get())
        except ValueError:
            self.state.filters.delta_qty = 1
            self.spn_dqty.delete(0, "end")
            self.spn_dqty.insert(0, "1")

    def on_load(self):
        self._sync_filters_from_ui()
        self._run_bg("load")

    def on_run_pairing(self):
        self._sync_filters_from_ui()
        self._run_bg("pair")

    def _run_bg(self, kind: str):
        if self._worker is not None and self._worker.is_alive():
            return

        self._set_busy(True, "Starting...")

        def progress(msg: str):
            self._q.put(("status", msg))

        def job():
            try:
                if kind == "load":
                    actions.load_data(self.state, progress=progress)
                    self._q.put(("loaded", None))
                elif kind == "pair":
                    actions.run_pairing(self.state, progress=progress)
                    self._q.put(("paired", None))
                else:
                    raise ValueError("Unknown job")
            except Exception as e:
                self._q.put(("error", e))

        self._worker = threading.Thread(target=job, daemon=True)
        self._worker.start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "loaded":
                    self._after_loaded()
                elif kind == "paired":
                    self._after_paired()
                elif kind == "error":
                    self._set_busy(False, "Error.")
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        finally:
            self.after(120, self._poll_queue)

    def _after_loaded(self):
        self._set_busy(False, "Loaded.")
        self.btn_pairs.config(state="normal")
        self._render_all()

    def _after_paired(self):
        self._set_busy(False, "Pairing done.")
        self._render_all()

    def _render_all(self):
        summary = actions.summarize(self.state)

        df_preview = None
        if self.state.df_master is not None:
            df_preview = self.state.df_master.head(200)
        elif self.state.df_clean is not None:
            df_preview = self.state.df_clean.head(200)

        self.views["overview"].render(summary, df_preview)

        if self.state.pairs_L1 is not None and len(self.state.pairs_L1):
            self.views["pairs"].render(self.state.pairs_L1.head(2000))
        else:
            self.views["pairs"].render(None)
