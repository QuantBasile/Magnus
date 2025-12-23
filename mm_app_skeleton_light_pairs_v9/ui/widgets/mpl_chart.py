# Python 3.12
from __future__ import annotations
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MplChart(ttk.Frame):
    def __init__(self, parent, *, title: str = ""):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        if title:
            self.ax.set_title(title)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self):
        self.ax.clear()

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()

