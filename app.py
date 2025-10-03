# app.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

from utils.theme import apply_theme
from ui.input_frame import InputFrame
from ui.output_frame import OutputFrame
from ui.info_frame import InfoFrame
from ui.preferences import PreferencesDialog  # Preferences dialog

# --- Adapters ---
from models.text_sentiment import TextSentimentAdapter
from models.image_classifier import ImageClassifierAdapter
from models.image_to_text import ImageToTextAdapter
from models.text_to_image import TextToImageAdapter
from models.text_to_video import TextToVideoAdapter  # comment out if not needed


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HIT137 • Tkinter AI GUI")
        self.geometry("1040x680")
        self.minsize(860, 560)
        apply_theme(self)

        # App state
        self._busy = False
        self._run_thread = None
        self._run_result = None
        self._run_error = None

        # Adapters in the dropdown
        self.adapters = {
            "Text Classification": TextSentimentAdapter(),
            "Image Classification": ImageClassifierAdapter(),
            "Image-to-Text": ImageToTextAdapter(),
            "Text-to-Image": TextToImageAdapter(),
           # "Text-to-Video": TextToVideoAdapter(),
        }
        self.var_model = tk.StringVar(value="Text Classification")

        # Base grid weights (responsive)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # row with panedwindow will stretch

        self._loaded_model_name = None
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()

    # ---------- Menu ----------
    def _build_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Preferences…", command=lambda: PreferencesDialog(self), accelerator="Ctrl+,")
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=filem)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                "Tkinter + Hugging Face Demo\n\n"
                "Models available:\n"
                "• Text Classification\n"
                "• Image Classification\n"
                "• Image-to-Text\n"
                "• Text-to-Image\n"
                "• Text-to-Video",
            ),
        )
        menubar.add_cascade(label="Help", menu=helpm)

    # ---------- Layout ----------
    def _build_layout(self):
        # Top bar
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        top.columnconfigure(2, weight=1)  # spacer

        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky="w")
        self.combo = ttk.Combobox(
            top, textvariable=self.var_model,
            values=list(self.adapters.keys()), width=28, state="readonly"
        )
        self.combo.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Button(top, text="Load Model", style="Accent.TButton",
                   command=self.load_selected).grid(row=0, column=3, sticky="w")

        # Progress bar (marquee)
        self.progress = ttk.Progressbar(top, mode="indeterminate", length=220)
        self.progress.grid(row=0, column=4, sticky="e")
        self.progress.stop()

        # Split panes
        mid = ttk.Panedwindow(self, orient="horizontal")
        mid.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)

        self.input_frame = InputFrame(mid)
        self.output_frame = OutputFrame(mid)
        mid.add(self.input_frame, weight=1)
        mid.add(self.output_frame, weight=2)

        # Action buttons
        btns = ttk.Frame(self)
        btns.grid(row=2, column=0, sticky="w", padx=12, pady=6)
        self.btn_run = ttk.Button(btns, text="Run Selected Model", command=self.run_selected)
        self.btn_run.pack(side="left", padx=5)
        ttk.Button(btns, text="Clear", command=lambda: self.output_frame.clear()).pack(side="left", padx=5)

        # Info + status
        self.info = InfoFrame(self)
        self.info.grid(row=3, column=0, sticky="ew", padx=12, pady=(6, 12))

        self.status = ttk.Label(self, anchor="w")
        self.status.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))

    # ---------- Shortcuts ----------
    def _bind_shortcuts(self):
        self.bind_all("<Control-r>", lambda e: self.run_selected())
        self.bind_all("<Control-l>", lambda e: self.load_selected())
        self.bind_all("<Control-q>", lambda e: self.destroy())
        self.bind_all("<Escape>",    lambda e: self._cancel_run())
        self.bind_all("<Control-comma>", lambda e: PreferencesDialog(self))

    # ---------- Busy helpers ----------
    def _set_busy(self, flag: bool, status_text: str = ""):
        self._busy = flag
        if flag:
            self.progress.start(12)  # faster = smoother
            self.combo.configure(state="disabled")
            self.btn_run.configure(state="disabled")
        else:
            self.progress.stop()
            self.combo.configure(state="readonly")
            self.btn_run.configure(state="normal")
        if status_text:
            self._status(status_text)

    # ---------- Actions ----------
    def load_selected(self):
        if self._busy:
            return
        name = self.var_model.get()
        # Clear previous output preview immediately so stale images don't remain
        try:
            self.output_frame.clear()
        except Exception:
            pass
        # If another model is currently loaded, ask for confirmation before switching
        if self._loaded_model_name and self._loaded_model_name != name:
            confirmed = messagebox.askyesno(
                "Switch model?",
                f"A different model ('{self._loaded_model_name}') is already loaded.\n"
                "Loading a new model will clear current inputs and outputs. Continue?",
            )
            if not confirmed:
                return

            # Clear UI and attempt to unload previous model to free resources
            try:
                # clear UI
                self.output_frame.clear()
                self.input_frame.clear()
                # unload previous adapter internals if possible
                prev = self.adapters.get(self._loaded_model_name)
                if prev is not None:
                    if hasattr(prev, "_pipe"):
                        try: prev._pipe = None
                        except Exception: pass
                    if hasattr(prev, "_model"):
                        try: prev._model = None
                        except Exception: pass
            except Exception:
                pass

        try:
            self._set_busy(True, f"Loading {name} …")
            self.update_idletasks()
            self.adapters[name].load()
            self.info.set_info(self.adapters[name].info())  # dict or str; InfoFrame handles both
            self._status(f"Loaded {name}")
            messagebox.showinfo("Loaded", f"{name} loaded successfully.")
            self._loaded_model_name = name
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self._status("Load error")
        finally:
            self._set_busy(False)

    def run_selected(self):
        if self._busy:
            return
        name = self.var_model.get()
        adapter = self.adapters[name]

        # Check either _pipe (diffusers/transformers) or _model (BLIP) exists
        if getattr(adapter, "_pipe", None) is None and not hasattr(adapter, "_model"):
            messagebox.showwarning("Not loaded", f"Please load '{name}' first.")
            return

        payload = self.input_frame.get_payload()
        # Normalize payload: InputFrame returns a dict  with mode/image_path/prompt.
        # Most adapters expect a plain string (prompt or path). Convert here so
        # adapters don't need to handle the UI-shaped dict.
        if isinstance(payload, dict):
            mode = payload.get("mode")
            if mode == "text":
                norm_payload = payload.get("prompt", "")
            elif mode == "image":
                # prefer explicit image_path, fallback to prompt
                norm_payload = payload.get("image_path") or payload.get("prompt", "")
            else:
                # unknown mode: pass prompt if available
                norm_payload = payload.get("prompt", "")
        else:
            norm_payload = payload
        self._run_result = None
        self._run_error = None

        # Background thread to avoid freezing UI
        def _worker():
            try:
                t0 = time.time()
                result = adapter.run(norm_payload)
                if not isinstance(result, dict):
                    result = {"result": str(result)}
                result.setdefault("_ms", (time.time() - t0) * 1000)
                self._run_result = result
            except Exception as ex:
                self._run_error = ex

        self._set_busy(True, f"Running {name} …")
        self._run_thread = threading.Thread(target=_worker, daemon=True)
        self._run_thread.start()
        self.after(80, self._poll_worker, name, adapter)

    def _poll_worker(self, name, adapter):
        if self._run_thread is not None and self._run_thread.is_alive():
            self.after(80, self._poll_worker, name, adapter)
            return

        self._set_busy(False)
        if self._run_error:
            self.output_frame.show({"result": f"Error: {self._run_error}"})
            self._status("Error")
            return

        out = self._run_result or {}
        #  pass the entire dict so OutputFrame can preview image/video
        self.output_frame.show(out)

        # Update info & status
        try:
            self.info.set_info(adapter.info())
        except Exception:
            pass
        ms = out.get("_ms")
        self._status(f"Ran {name} in {ms:.1f} ms" if ms else f"Ran {name}")

    def _cancel_run(self):
        # Soft cancel: can't kill thread, but re-enable UI & ignore result
        if self._busy:
            self._set_busy(False, "Cancelled (soft)")
            self._run_thread = None

    # ---------- Status ----------
    def _status(self, text: str):
        self.status.config(text=text)


if __name__ == "__main__":
    App().mainloop()
