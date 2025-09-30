# utils/config.py
import json, os

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "app_config.json")
_CFG_PATH = os.path.abspath(_CFG_PATH)

_DEFAULTS = {
    "theme": "Light",   # Light | Dark | Blue | Custom
    "custom": {
        "bg": "#ffffff",
        "fg": "#111111",
        "accent": "#4f46e5",   # indigo-ish
        "frame_bg": "#f3f4f6",
        "textbox_bg": "#ffffff",
        "textbox_fg": "#111111",
        "status_bg": "#f3f4f6",
        "status_fg": "#111111",
        "font_size": 10
    }
}

def load_config():
    try:
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {**_DEFAULTS, **data, "custom": {**_DEFAULTS["custom"], **data.get("custom", {})}}
    except Exception:
        return _DEFAULTS.copy()

def save_config(cfg):
    try:
        with open(_CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass
