# ui/preferences.py
import tkinter as tk
from tkinter import ttk, colorchooser
from utils.config import load_config, save_config
from utils.theme import apply_theme

class PreferencesDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Preferences")
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)

        # ✅ Theme the dialog itself
        apply_theme(self)

        self.cfg = load_config()

        # Layout
        root = ttk.Frame(self)
        root.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Theme:").grid(row=0, column=0, sticky="w")
        self.var_theme = tk.StringVar(value=self.cfg["theme"])
        self.combo = ttk.Combobox(
            root, textvariable=self.var_theme,
            values=["Light", "Dark", "Blue", "Custom"],
            state="readonly", width=16
        )
        self.combo.grid(row=0, column=1, sticky="w", padx=6)
        self.combo.bind("<<ComboboxSelected>>", self._on_theme_change)

        self.custom_frame = ttk.LabelFrame(root, text="Custom Theme")
        self.custom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._build_custom_controls(self.custom_frame)

        btns = ttk.Frame(root)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=6)
        ttk.Button(btns, text="Apply", style="Accent.TButton", command=self._apply).pack(side="right", padx=6)

        # initial state
        self._toggle_custom(self.var_theme.get() == "Custom")

    def _build_custom_controls(self, parent):
        self.vars = {}
        def row(label, key, col=0, r=0):
            # ttk labels inherit themed colors; no manual fg/bg needed
            ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", pady=2)
            var = tk.StringVar(value=self.cfg["custom"][key])
            ent = ttk.Entry(parent, textvariable=var, width=14)
            ent.grid(row=r, column=col+1, sticky="w", padx=6)
            ttk.Button(parent, text="Pick…", command=lambda v=var: self._pick_color(v)).grid(row=r, column=col+2, sticky="w", padx=4)
            self.vars[key] = var

        row("Background", "bg", r=0)
        row("Foreground", "fg", r=1)
        row("Accent", "accent", r=2)
        row("Frame background", "frame_bg", r=3)
        row("Textbox bg", "textbox_bg", r=4)
        row("Textbox fg", "textbox_fg", r=5)
        row("Status bg", "status_bg", col=3, r=0)
        row("Status fg", "status_fg", col=3, r=1)

        ttk.Label(parent, text="Base font size").grid(row=2, column=3, sticky="w")
        self.var_font = tk.IntVar(value=int(self.cfg["custom"]["font_size"]))
        ttk.Spinbox(parent, from_=8, to=20, textvariable=self.var_font, width=6).grid(row=2, column=4, sticky="w", padx=6)

    def _on_theme_change(self, _):
        self._toggle_custom(self.var_theme.get() == "Custom")
        # ✅ live-preview the dialog’s colors when switching theme (without saving)
        #   only apply to this dialog so the main app doesn’t jump until Apply.
        apply_theme(self)

    def _toggle_custom(self, show: bool):
        state = "normal" if show else "disabled"
        for child in self.custom_frame.winfo_children():
            child.configure(state=state)

    def _pick_color(self, var):
        color = colorchooser.askcolor(initialcolor=var.get(), parent=self)[1]
        if color:
            var.set(color)

    def _apply(self):
        cfg = load_config()
        cfg["theme"] = self.var_theme.get()
        if cfg["theme"] == "Custom":
            for k, v in self.vars.items():
                cfg["custom"][k] = v.get()
            cfg["custom"]["font_size"] = int(self.var_font.get())
        save_config(cfg)

        # ✅ apply to the whole app AND this dialog
        apply_theme(self.master)
        apply_theme(self)
        self.destroy()
