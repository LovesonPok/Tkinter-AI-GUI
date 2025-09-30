# ui/info_frame.py
import tkinter as tk
from tkinter import ttk


class InfoFrame(ttk.Frame):
    """Panel to display model info and OOP explanations."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Left: Model information
        ttk.Label(self, text="Model Information & OOP Explanation", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        # Model Info
        self.model_text = tk.Text(self, wrap="word", height=6)
        self.model_text.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.model_text.insert("1.0", "Model details will appear here...")
        self.model_text.configure(state="disabled")

        # OOP Concepts
        self.oop_text = tk.Text(self, wrap="word", height=6)
        self.oop_text.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        self.oop_text.insert("1.0", "OOP concepts explanation will appear here...")
        self.oop_text.configure(state="disabled")

        self.rowconfigure(1, weight=1)

    def set_info(self, info):
        """Accepts dict or str and displays nicely."""
        self.model_text.configure(state="normal")
        self.model_text.delete("1.0", "end")
        if isinstance(info, dict):
            for k, v in info.items():
                self.model_text.insert("end", f"{k}: {v}\n")
        else:
            self.model_text.insert("end", str(info))
        self.model_text.configure(state="disabled")

        # Static OOP concepts explanation
        oop_explanation = (
            "• Encapsulation: Model pipelines are hidden behind adapter classes.\n"
            "• Polymorphism: GUI calls adapter.run(payload) regardless of model type.\n"
            "• Overriding: Each adapter overrides BaseModelAdapter.run().\n"
            "• Multiple decorators: e.g., @log_action, @timeit around run().\n"
            "• Multiple inheritance: BaseModelAdapter + SaveLoad mixins."
        )

        self.oop_text.configure(state="normal")
        self.oop_text.delete("1.0", "end")
        self.oop_text.insert("end", oop_explanation)
        self.oop_text.configure(state="disabled")
