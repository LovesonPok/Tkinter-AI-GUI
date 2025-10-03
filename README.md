HIT137 - Tkinter AI GUI

Overview

This repository contains a small desktop GUI built with Tkinter that demonstrates several Hugging Face model workflows. The app provides four main capabilities: text classification, image classification, image captioning (image-to-text), and text-to-image generation. It is intended as an educational demo and a lightweight local tool for experimentation.

This README explains how to prepare an environment, run the app, and work on the code. It also describes the layout of the repository and the responsibilities of the team members listed in the project.

Scope and important notes

- The app loads model pipelines from the Hugging Face ecosystem and runs them locally. Some models are GPU-friendly and will be slow or may not run on low-memory CPUs. The code includes CPU-friendly fallbacks for some pipelines, but performance varies.
- Several adapters in the project intentionally disable the diffusers safety checker to avoid blank outputs during development. Do not expose unfiltered outputs in publicly accessible services without proper moderation.

Requirements

This guide assumes a Linux environment and a zsh shell. Commands are copy-paste ready for that setup.

Core (required)

- Python 3.10 or newer (Python 3.12 recommended).
- Git (to clone and work with the repository).
- A working shell (zsh / bash) and a stable internet connection to download model weights.

Optional (only if you plan to use a GPU or extra accelerators)

- CUDA-capable NVIDIA GPU and matching drivers — only required if you want GPU acceleration for large models. Not required to run the app in CPU mode.
- Optional acceleration libraries such as `xformers` or vendor-specific builds of PyTorch. Install these only when you have compatible hardware and matching binaries.

Quick start

1. Clone the repository (if you do not already have it locally)

```bash
git clone <https://github.com/LovesonPok/Tkinter-AI-GUI> tk-hf-gui
cd tk-hf-gui
```

2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies

Follow the commands that match your hardware. These instructions assume a Linux system and the zsh shell. Run one of the PyTorch install lines below first, then install the remaining Python packages.

Upgrade pip and install wheel tools

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

Python packages

1. Upgrade pip and core build tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

2. Install PyTorch

- If you do not have a CUDA GPU, install the CPU-only wheels (recommended for machines without NVIDIA GPUs):

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- If you have an NVIDIA GPU and want GPU acceleration, install the PyTorch wheel that matches your CUDA driver (see https://pytorch.org/get-started/locally). This is optional — the app works without a GPU.

3. Install the remaining Python packages used by the project

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` contains the core Python packages the project needs (transformers, diffusers, pillow, etc.). Optional accelerators such as `xformers` should only be installed when your hardware and Python/CUDA versions are compatible.

System packages you may need

```bash
# ffmpeg is required for video encoding and decoding
sudo apt update
sudo apt install -y ffmpeg git-lfs
# after installing git-lfs, run once per user
git lfs install
```

If you will access private Hugging Face model weights, set your token

```bash
export HUGGINGFACE_HUB_TOKEN="<your-token-here>"
```

Disk and cache notes

Model weights are downloaded into your user cache by default. If you need to relocate or manage the cache, consult Hugging Face documentation.

Notes about optional packages

- `xformers` and other accelerator libraries can improve speed and memory usage on supported hardware. Install them only if you have a compatible GPU and matching Python/CUDA binaries.
- GPU usage is optional. If you have no GPU or do not want to install CUDA drivers, use the CPU-only PyTorch wheels above. Expect slower image generation on CPU.

Run the application

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt  # ensure deps installed
python3 app.py
```

The Tkinter window will appear. Use the top Model dropdown to choose a model, click Load Model and then use the input panel to supply a prompt or an image. Click Run Selected Model to execute the selected model. Generated images and media are saved into the `assets` folder.

Project layout and explanation

Top level files

- `app.py` - main application entry point and UI wiring. It builds the Tkinter window, the layout, and manages model loading and background runs.
- `requirements.txt` - Python dependencies used by the project.
- `README.md` - this file.

Folders

- `assets/` - generated images and example media are stored here. The app saves generated files to this folder.
- `docs/` - notes and helper files. See model-specific files inside for additional context.
- `models/` - adapter classes for each model category. Each adapter encapsulates loading and running a particular pipeline. Key files:
  - `text_to_image.py` - text-to-image adapter (Stable Diffusion variants). CPU-friendly defaults are provided; there are also higher-quality presets in the code comments.
  - `image_to_text.py` - image captioning adapter (BLIP).
  - `image_classifier.py` - image classification adapter.
  - `text_sentiment.py` - text classification adapter.
  - `text_to_video.py` - text-to-video adapter. This file includes a GPU path that uses SDXL and Stable Video Diffusion and a CPU fallback.
  - `base.py` - base adapter class shared by other adapters.
- `ui/` - Tkinter UI components split into files for better structure. Key files:
  - `input_frame.py` - user input widgets and logic to gather prompts and image paths.
  - `output_frame.py` - displays model output and previews generated images or video thumbnails.
  - `info_frame.py` - model information and static help text.
  - `preferences.py` - lightweight preferences dialog used by the app.
- `utils/` - reusable helper code
  - `decorators.py` - simple logging and timing decorators used by adapters.
  - `theme.py` - styling for the UI (colors and ttk styles).

How the adapters and UI work together

- The GUI calls `adapter.load()` to instantiate and prepare a pipeline. Loading may be slow on first run because model weights are fetched from the Hugging Face hub.
- The GUI gathers input from `InputFrame.get_payload()` and normalizes it to a simple string where appropriate. The adapter `run()` method receives the normalized input and returns a dictionary with at least a `result` string and optional preview keys such as `image_path` or `video_path`.
- The `OutputFrame` shows textual results and renders a preview of the media returned by adapters.

Team and roles

Project members

- Milan Sapkota (Dhan Prasad Sapkota)
- Loveson Pokharel
- Bishal Nayabha
- Suman Mahat

Suggested roles and responsibilities

- lovesonpok - Project lead and lead backend and performance engineer. He is the main point of contact for architecture decisions and backend implementation. Responsibilities: lead the overall design, implement and optimize model adapters, tune performance for CPU and GPU, manage optional acceleration packages, and coordinate integration testing on target hardware.
- milansapkota1600 - Developer and integrator. Responsibilities: work with the lead on adapter design, implement features across adapters, and help with integration and testing.
- bishalnayabha0 - UI and user experience lead. Responsibilities: Tkinter layout, input/output widgets, preview behavior, accessibility, and theme tuning.
- suman2120 - Documentation and testing lead. Responsibilities: README and docs, local setup instructions, troubleshooting guides, and writing small unit or smoke tests to catch regressions.

How the team works together

- Feature design and task assignment: team members discuss a feature or issue, then a single member implements the first version and opens a short description in the project issue tracker or a local task list.
- Review cycle: another team member reviews the change for correctness, style, and UI/UX. The reviewer runs the code locally and provides feedback. Small suggested edits are applied directly.
- Integration and testing: the integrator runs the app on a target machine, verifies behavior for CPU and GPU if available, and finalizes the change.
- Documentation and hand-off: the documentation lead updates the README or inline docs describing any configuration changes or new optional dependencies.

Development notes and tips

- If you want higher quality images, try using SDXL models or increase `num_inference_steps` and `guidance_scale` in `text_to_image.py`. For higher-quality results you may need a GPU with enough VRAM.
- If you use GPU acceleration, make sure the installed PyTorch build matches your CUDA drivers. Install a matching `torch` version before installing other packages where possible.
- The project uses simple decorators for timing and logging in `utils/decorators.py`. These are helpful for quick profiling and for showing how long a call took.

Common troubleshooting

- If you see errors about missing packages, check `requirements.txt` and install missing dependencies in the virtual environment.
- If a model load fails due to memory, try smaller image sizes in the adapter or use the CPU fallback where implemented.
- If image preview stays on screen after switching models, use the Clear button or restart the app. The app now clears the preview when you click Load Model, but behavior may vary if an in-progress operation was running.

Extending the project

- Add model options to the UI so users can control steps, guidance scale, negative prompts, seeds, and image size without editing code.
- Add a gallery view of previously generated assets in `assets`.
- Add automated tests for adapter output shapes and for small smoke tests of UI initialization.

License and ethical notice

This repository includes code that uses Hugging Face model weights. Respect the licenses associated with the models you are using. Some adapters in the code disable safety filters for development. 
Contact and support

If you need help reproducing the environment, contact the project authors listed in the Team and Roles section. Provide a short description of the issue, the output from running the app, and any tracebacks you see.

End of README
