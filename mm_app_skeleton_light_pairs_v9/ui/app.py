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
from ui.views.counterparty import CounterpartyView
from ui.views.daily import DailyView
from ui.views.underlying import UnderlyingView
from ui.views.daily_underlying import DailyUnderlyingView
from ui.views.intraday import IntradayView
from ui.views.deltat import DeltaTView
from ui.views.strike_distance import StrikeDistanceView
from ui.views.call_put import CallPutView


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

        # --- Toggle: enable/disable heavy L3 pairing ---
        # OFF por defecto para ahorrar recursos
        self.var_enable_l3 = tk.BooleanVar(value=False)
        self.chk_enable_l3 = ttk.Checkbutton(
            self.topbar,
            text="Enable L3",
            variable=self.var_enable_l3,
        )
        self.chk_enable_l3.pack(side="left", padx=(12, 6), pady=pad_y)

        self.btn_pairs = ttk.Button(self.topbar, text="Run pairing", command=self.on_run_pairing, state="disabled")
        self.btn_pairs.pack(side="left", padx=6, pady=pad_y)
        
        self.btn_info = ttk.Button(self.topbar, text="ⓘ Info", command=self.on_show_pairing_info)
        self.btn_info.pack(side="left", padx=(6, 6), pady=pad_y)


        ttk.Label(self.topbar, text=" ", style="Muted.TLabel").pack(side="right", padx=10)

    def _build_body(self):
        self.sidebar = ttk.Frame(self.body, style="Sidebar.TFrame", width=150)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = ttk.Frame(self.body, style="TFrame")
        self.content.pack(side="left", fill="both", expand=True)

        self.views: dict[str, ttk.Frame] = {}
        self.views["overview"] = OverviewView(self.content)
        self.views["pairs"] = PairsView(self.content)
        self.views["counterparty"] = CounterpartyView(self.content)
        self.views["daily"] = DailyView(self.content)
        self.views["underlying"] = UnderlyingView(self.content)
        self.views["daily_und"] = DailyUnderlyingView(self.content)
        self.views["intraday"] = IntradayView(self.content)
        self.views["deltat"] = DeltaTView(self.content)
        self.views["strike_dist"] = StrikeDistanceView(self.content)
        self.views["callput"] = CallPutView(self.content)

        for v in self.views.values():
            v.place(relx=0, rely=0, relwidth=1, relheight=1)

        ttk.Label(self.sidebar, text="Navigation", style="Title.TLabel").pack(anchor="w", padx=12, pady=(12, 6))
        ttk.Button(self.sidebar, text="Overview", style="Sidebar.TButton", command=lambda: self.show("overview")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Pairs", style="Sidebar.TButton", command=lambda: self.show("pairs")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Counterparty", style="Sidebar.TButton", command=lambda: self.show("counterparty")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Daily", style="Sidebar.TButton", command=lambda: self.show("daily")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Underlying", style="Sidebar.TButton", command=lambda: self.show("underlying")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Daily-UND", style="Sidebar.TButton", command=lambda: self.show("daily_und")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Intraday", style="Sidebar.TButton", command=lambda: self.show("intraday")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Δt Pattern", style="Sidebar.TButton", command=lambda: self.show("deltat")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="StrikeDist", style="Sidebar.TButton", command=lambda: self.show("strike_dist")).pack(fill="x", padx=10, pady=4)
        ttk.Button(self.sidebar, text="Call / Put", style="Sidebar.TButton", command=lambda: self.show("callput")).pack(fill="x", padx=10, pady=4)

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

        # Nuevo: toggle de L3 (por defecto OFF para ahorrar recursos)
        # Si tu Filters dataclass es estricto y no tiene enable_l3, añade ese campo en controller/state.py
        self.state.filters.enable_L3 = bool(self.var_enable_l3.get())
        

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

    def on_show_pairing_info(self):
        msg = (
            "L1 – Simple pairing\n"
            "  Sell 100 contracts of ISIN X at 08:00:00\n"
            "  Buy  100 contracts of ISIN X at 08:00:10\n"
            "----------------------------------------\n\n"
            "L2 – Pairing with Split execution\n"
            "  Sell 100 contracts of ISIN X at 08:00:00\n"
            "  Buy   40 contracts of ISIN X at 08:00:04\n"
            "  Buy   60 contracts of ISIN X at 08:00:09\n"
            "----------------------------------------\n\n"
            "L3 – Pairing with quantity tolerance\n"
            "  Sell 100 contracts of ISIN X at 08:00:00\n"
            "  Buy   95 contracts of ISIN X at 08:00:05\n"
            "⚠️ Warning: L3 is computationally expensive\n"
            )
    
        messagebox.showinfo("Pairing logic – L1 / L2 / L3", msg)



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
        try:
            summary = actions.summarize(self.state)

            df_preview = None
            if self.state.df_master is not None:
                df_preview = self.state.df_master.head(200)
            elif self.state.df_clean is not None:
                df_preview = self.state.df_clean.head(200)

            df_input = self.state.df_clean.head(200) if self.state.df_clean is not None else None
            self.views["overview"].render(summary, df_input)

            self.views["pairs"].set_data(self.state.df_master)
            self.views["counterparty"].set_data(self.state.df_master)
            self.views["daily"].set_data(self.state.df_master)
            self.views["underlying"].set_data(self.state.df_master)
            self.views["daily_und"].set_data(self.state.df_master)
            self.views["intraday"].set_data(self.state.df_master)
            self.views["deltat"].set_data(self.state.df_master)
            self.views["strike_dist"].set_data(self.state.df_master)
            self.views["callput"].set_data(self.state.df_master)

        except Exception as e:
            # Prevent the UI loop from spamming exceptions
            self.status.set(f"Render error: {e}")
