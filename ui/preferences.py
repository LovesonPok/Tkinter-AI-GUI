# ui/preferences.py
import tkinter as tk
from tkinter import ttk, colorchooser
from utils.config import load_config, save_config
from utils.theme import apply_theme

# Dialog to configure theme preferences
class PreferencesDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Preferences")
        self.transient(master)  # stay on top of parent
        self.grab_set()          # modal behavior
        self.resizable(False, False)

        apply_theme(self)        # theme the dialog itself

        self.cfg = load_config()  # load current config

        # Root frame for layout
        root = ttk.Frame(self)
        root.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        root.columnconfigure(1, weight=1)

        # Theme selection combobox
        ttk.Label(root, text="Theme:").grid(row=0, column=0, sticky="w")
        self.var_theme = tk.StringVar(value=self.cfg["theme"])
        self.combo = ttk.Combobox(
            root, textvariable=self.var_theme,
            values=["Light", "Dark", "Blue", "Custom"],
            state="readonly", width=16
        )
        self.combo.grid(row=0, column=1, sticky="w", padx=6)
        self.combo.bind("<<ComboboxSelected>>", self._on_theme_change)

        # Custom theme settings frame
        self.custom_frame = ttk.LabelFrame(root, text="Custom Theme")
        self.custom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._build_custom_controls(self.custom_frame)

        # Buttons frame
        btns = ttk.Frame(root)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=6)
        ttk.Button(btns, text="Apply", style="Accent.TButton", command=self._apply).pack(side="right", padx=6)

        # Show/hide custom controls based on initial theme
        self._toggle_custom(self.var_theme.get() == "Custom")

    # Build controls for custom theme colors and font
    def _build_custom_controls(self, parent):
        self.vars = {}

        def row(label, key, col=0, r=0):
            ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", pady=2)
            var = tk.StringVar(value=self.cfg["custom"][key])
            ent = ttk.Entry(parent, textvariable=var, width=14)
            ent.grid(row=r, column=col+1, sticky="w", padx=6)
            ttk.Button(parent, text="Pick…", command=lambda v=var: self._pick_color(v)).grid(row=r, column=col+2, sticky="w", padx=4)
            self.vars[key] = var

        # Left-side color options
        row("Background", "bg", r=0)
        row("Foreground", "fg", r=1)
        row("Accent", "accent", r=2)
        row("Frame background", "frame_bg", r=3)
        row("Textbox bg", "textbox_bg", r=4)
        row("Textbox fg", "textbox_fg", r=5)

        # Right-side color options
        row("Status bg", "status_bg", col=3, r=0)
        row("Status fg", "status_fg", col=3, r=1)

        # Base font size
        ttk.Label(parent, text="Base font size").grid(row=2, column=3, sticky="w")
        self.var_font = tk.IntVar(value=int(self.cfg["custom"]["font_size"]))
        ttk.Spinbox(parent, from_=8, to=20, textvariable=self.var_font, width=6).grid(row=2, column=4, sticky="w", padx=6)

    # Handle theme selection changes
    def _on_theme_change(self, _):
        self._toggle_custom(self.var_theme.get() == "Custom")
        apply_theme(self)  # live-preview in the dialog only

    # Enable/disable custom theme controls
    def _toggle_custom(self, show: bool):
        state = "normal" if show else "disabled"
        for child in self.custom_frame.winfo_children():
            child.configure(state=state)

    # Open color picker dialog
    def _pick_color(self, var):
        color = colorchooser.askcolor(initialcolor=var.get(), parent=self)[1]
        if color:
            var.set(color)

    # Save and apply preferences
    def _apply(self):
        cfg = load_config()
        cfg["theme"] = self.var_theme.get()
        if cfg["theme"] == "Custom":
            for k, v in self.vars.items():
                cfg["custom"][k] = v.get()
            cfg["custom"]["font_size"] = int(self.var_font.get())
        save_config(cfg)

        # Apply theme to the whole app and dialog
        apply_theme(self.master)
        apply_theme(self)
        self.destroy()
