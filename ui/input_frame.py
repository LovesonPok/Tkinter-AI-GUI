# ui/input_frame.py
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

# Frame for user input (text or image)
class InputFrame(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="User Input")

        # Configure grid columns and rows for responsive layout
        for c in range(3):
            self.columnconfigure(c, weight=1 if c != 2 else 0)
        for r in range(4):
            self.rowconfigure(r, weight=0)
        self.rowconfigure(3, weight=1)  # text area grows with window

        # Mode selector: Text or Image
        self.var_mode = tk.StringVar(value="text")
        ttk.Radiobutton(self, text="Text",  variable=self.var_mode, value="text").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(self, text="Image", variable=self.var_mode, value="image").grid(row=0, column=1, sticky="w", padx=6, pady=4)

        # Image path entry + Browse button
        self.var_path = tk.StringVar()
        self.ent_path = ttk.Entry(self, textvariable=self.var_path)
        self.ent_path.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,6))
        ttk.Button(self, text="Browse", command=self._browse).grid(row=1, column=2, sticky="e", padx=(0,6), pady=(0,6))

        # Multi-line text input (grows)
        self.txt = ScrolledText(self, height=8, wrap="word")
        self.txt.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0,6))

    # Open file dialog to select an image
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files","*.*")]
        )
        if path:
            self.var_mode.set("image")  # switch mode automatically
            self.var_path.set(path)

    # Gather user input as payload for model adapters
    def get_payload(self):
        if self.var_mode.get() == "image":
            return {
                "mode": "image",
                "image_path": self.var_path.get().strip(),
                "prompt": self.txt.get("1.0", "end").strip(),  # optional text for captioning
            }

        # Text mode: prefer multi-line text box, fallback to entry field
        prompt = self.txt.get("1.0", "end").strip()
        if not prompt:
            prompt = self.ent_path.get().strip()

        return {
            "mode": "text",
            "prompt": prompt,
        }

    # Reset input widgets to default state
    def clear(self):
        self.var_mode.set("text")
        self.var_path.set("")
        self.ent_path.delete(0, "end")
        self.txt.delete("1.0", "end")
