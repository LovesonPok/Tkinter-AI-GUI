# utils/theme.py
import tkinter as tk
from tkinter import ttk
from utils.config import load_config

_PRESETS = {
    "Light": dict(bg="#ffffff", fg="#111111", accent="#2563eb",
                  frame_bg="#f3f4f6", textbox_bg="#ffffff", textbox_fg="#111111",
                  status_bg="#f3f4f6", status_fg="#111111", font_size=10),
    "Dark":  dict(bg="#0f172a", fg="#e5e7eb", accent="#22d3ee",
                  frame_bg="#111827", textbox_bg="#0b1220", textbox_fg="#e5e7eb",
                  status_bg="#0b1220", status_fg="#e5e7eb", font_size=10),
    "Blue":  dict(bg="#0b1020", fg="#dbeafe", accent="#60a5fa",
                  frame_bg="#0f172a", textbox_bg="#0c1428", textbox_fg="#dbeafe",
                  status_bg="#0c1428", status_fg="#dbeafe", font_size=10),
}

def _palette(cfg):
    return cfg["custom"] if cfg.get("theme") == "Custom" else _PRESETS.get(cfg.get("theme"), _PRESETS["Light"])

def _retint_widget(w, p):
    import tkinter as tk
    try:
        if isinstance(w, (tk.Text, tk.Listbox, tk.Entry, tk.Spinbox)):
            w.configure(bg=p["textbox_bg"], fg=p["textbox_fg"], insertbackground=p["textbox_fg"],
                        highlightthickness=0, bd=0)
        elif isinstance(w, (tk.Label, tk.Button, tk.Radiobutton, tk.Checkbutton)):
            w.configure(bg=p["bg"], fg=p["fg"])
        elif isinstance(w, (tk.Frame, tk.Toplevel)):
            w.configure(bg=p["bg"])
    except tk.TclError:
        pass
    for c in w.winfo_children():
        _retint_widget(c, p)

def apply_theme(root: tk.Tk):
    cfg = load_config()
    p = _palette(cfg)
    style = ttk.Style(root)

    # Base
    root.configure(bg=p["bg"])
    try: style.theme_use("clam")
    except Exception: pass

    # Option DB
    root.option_add("*Font", ("Segoe UI", p["font_size"]))
    root.option_add("*Text.background",   p["textbox_bg"])
    root.option_add("*Text.foreground",   p["textbox_fg"])
    root.option_add("*Text.highlightThickness", 0)
    root.option_add("*Text.borderWidth", 0)
    root.option_add("*Entry.background",  p["textbox_bg"])
    root.option_add("*Entry.foreground",  p["textbox_fg"])
    root.option_add("*Entry.highlightThickness", 0)
    root.option_add("*Entry.borderWidth", 0)
    root.option_add("*Label.background",  p["bg"])
    root.option_add("*Label.foreground",  p["fg"])
    # Menus (tk menus, not ttk)
    root.option_add("*Menu.background", p["bg"])
    root.option_add("*Menu.foreground", p["fg"])
    root.option_add("*Menu.activeBackground", p["frame_bg"])
    root.option_add("*Menu.activeForeground", p["fg"])
    root.option_add("*Menu.relief", "flat")

    # ttk styles
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabelframe", background=p["bg"], bordercolor=p["frame_bg"])
    style.configure("TLabelframe.Label", background=p["bg"], foreground=p["fg"])

    style.configure("TLabel", background=p["bg"], foreground=p["fg"])
    style.configure("TRadiobutton", background=p["bg"], foreground=p["fg"])
    style.map("TRadiobutton",
              foreground=[("disabled", "#94a3b8")],
              background=[("active", p["bg"])])

    style.configure("TButton", foreground=p["fg"], background=p["frame_bg"])
    style.map("TButton",
              background=[("active", p["frame_bg"])],
              foreground=[("disabled", "#cbd5e1")])

    style.configure("Accent.TButton", foreground="white", background=p["accent"])
    style.map("Accent.TButton",
              background=[("active", p["accent"]), ("disabled", "#475569")],
              foreground=[("disabled", "#cbd5e1")])

    style.configure("TEntry", fieldbackground=p["textbox_bg"], foreground=p["textbox_fg"])
    style.configure("TCombobox",
                    fieldbackground=p["textbox_bg"],
                    background=p["bg"],
                    foreground=p["textbox_fg"])
    style.map("TCombobox",
              fieldbackground=[("readonly", p["textbox_bg"])],
              foreground=[("disabled", "#94a3b8")])

    style.configure("Vertical.TScrollbar", background=p["frame_bg"], troughcolor=p["frame_bg"])
    style.configure("Horizontal.TProgressbar", background=p["accent"])

    root._theme_palette = p
    _retint_widget(root, p)
