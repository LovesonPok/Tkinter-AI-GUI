# ui/input_frame.py
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

class InputFrame(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="User Input")
        # grid weights (responsive)
        for c in range(3):
            self.columnconfigure(c, weight=1 if c != 2 else 0)
        for r in range(4):
            self.rowconfigure(r, weight=0)
        self.rowconfigure(3, weight=1)  # text area grows

        # mode (text / image)
        self.var_mode = tk.StringVar(value="text")
        ttk.Radiobutton(self, text="Text",  variable=self.var_mode, value="text").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(self, text="Image", variable=self.var_mode, value="image").grid(row=0, column=1, sticky="w", padx=6, pady=4)

        # image path + browse
        self.var_path = tk.StringVar()
        self.ent_path = ttk.Entry(self, textvariable=self.var_path)
        self.ent_path.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,6))
        ttk.Button(self, text="Browse", command=self._browse).grid(row=1, column=2, sticky="e", padx=(0,6), pady=(0,6))

        # prompt / text input (grows)
        self.txt = ScrolledText(self, height=8, wrap="word")
        self.txt.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0,6))

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files","*.*")])
        if path:
            self.var_mode.set("image")
            self.var_path.set(path)

    def get_payload(self):
        if self.var_mode.get() == "image":
            return {
                "mode": "image",
                "image_path": self.var_path.get().strip(),
                "prompt": self.txt.get("1.0", "end").strip(),  # keep text too for img-caption models
            }

        # text mode: prefer the multi-line text box, but fall back to the single-line
        # entry (users often type short prompts there as in the screenshot).
        prompt = self.txt.get("1.0", "end").strip()
        if not prompt:
            prompt = self.ent_path.get().strip()

        return {
            "mode": "text",
            "prompt": prompt,
        }

    def clear(self):
        """Reset input widgets to defaults."""
        self.var_mode.set("text")
        self.var_path.set("")
        self.ent_path.delete(0, "end")
        self.txt.delete("1.0", "end")
