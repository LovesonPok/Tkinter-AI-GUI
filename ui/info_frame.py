# ui/info_frame.py
import tkinter as tk
from tkinter import ttk

# Frame to display model information and OOP explanations in the GUI
class InfoFrame(ttk.Frame):
    """Panel to display model info and OOP explanations."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Configure two equal-width columns
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Header label spanning both columns
        ttk.Label(
            self, text="Model Information & OOP Explanation", style="Heading.TLabel"
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # Left text box: displays model information
        self.model_text = tk.Text(self, wrap="word", height=6)
        self.model_text.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.model_text.insert("1.0", "Model details will appear here...")
        self.model_text.configure(state="disabled")  # prevent editing

        # Right text box: displays OOP concepts explanation
        self.oop_text = tk.Text(self, wrap="word", height=6)
        self.oop_text.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        self.oop_text.insert("1.0", "OOP concepts explanation will appear here...")
        self.oop_text.configure(state="disabled")

        # Allow row 1 (the text boxes) to expand with the frame
        self.rowconfigure(1, weight=1)

    def set_info(self, info):
        """Accepts a dict or string and displays model info in the left text box."""
        self.model_text.configure(state="normal")
        self.model_text.delete("1.0", "end")

        # If dict, format nicely; otherwise insert as string
        if isinstance(info, dict):
            for k, v in info.items():
                self.model_text.insert("end", f"{k}: {v}\n")
        else:
            self.model_text.insert("end", str(info))
        self.model_text.configure(state="disabled")

        # Static OOP concepts explanation in the right text box
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
