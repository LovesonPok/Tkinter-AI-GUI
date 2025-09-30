# ui/_parts.py
import tkinter as tk
from tkinter import ttk

class ThemedScrolledText(ttk.Frame):
    """A Text with ttk Scrollbars that respects the current theme."""
    def __init__(self, master, **text_kwargs):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Text widget (classic tk) — remove bright borders
        tk_kwargs = dict(wrap="word", borderwidth=0, highlightthickness=0)
        tk_kwargs.update(text_kwargs or {})
        self.text = tk.Text(self, **tk_kwargs)
        self.text.grid(row=0, column=0, sticky="nsew")

        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=self.vsb.set)

        # optional horizontal bar
        # self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        # self.hsb.grid(row=1, column=0, sticky="ew")
        # self.text.configure(xscrollcommand=self.hsb.set)

    # convenience
    def insert(self, *a, **k): return self.text.insert(*a, **k)
    def delete(self, *a, **k): return self.text.delete(*a, **k)
    def get(self, *a, **k):    return self.text.get(*a, **k)
    def configure(self, *a, **k): return self.text.configure(*a, **k)
