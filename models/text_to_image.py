# models/text_to_image.py — High-quality CPU preset for Stable Diffusion 1.5
import os, re, datetime, torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from models.base import BaseModelAdapter

# Adapter for generating images from text using Stable Diffusion 1.5 on CPU
class TextToImageAdapter(BaseModelAdapter):
    model_name  = "runwayml/stable-diffusion-v1-5"
    category    = "Text-to-Image"
    description = "High-quality text-to-image on CPU (SD 1.5, DPM-Solver)."

    # Load the pipeline with CPU optimizations
    def load(self):
        self._device = "cpu"
        # Load the pretrained SD 1.5 model
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # float32 avoids black outputs on CPU
            use_safetensors=True,
        )

        # Use DPM-Solver scheduler for better quality with fewer steps
        try:
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        except Exception:
            pass

        # Disable safety checker for demo purposes (prevents blank images)
        if hasattr(self._pipe, "safety_checker"):
            def _noop(images, **kwargs): return images, [False] * len(images)
            self._pipe.safety_checker = _noop

        # Enable attention and VAE slicing to save memory on CPU
        for fn in ("enable_attention_slicing", "enable_vae_slicing"):
            try: getattr(self._pipe, fn)()
            except Exception: pass

        # Disable progress bar
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.to(self._device)

    # Run text-to-image generation
    def run(self, payload):
        # Get prompt from dict or raw string
        if isinstance(payload, dict):
            prompt = (payload.get("prompt") or payload.get("text") or "").strip()
        else:
            prompt = (payload or "").strip()

        if not prompt:
            return {"result": "Enter a text prompt."}

        # CPU-friendly generation settings
        steps = 30                 # Number of inference steps (higher = better quality)
        cfg   = 7.5                # Guidance scale
        h, w  = 384, 384           # Image height and width
        neg   = "blurry, lowres, bad anatomy, extra limbs, watermark, text, jpeg artifacts"

        # Generate image
        image = self._pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=(h//8)*8, width=(w//8)*8
        ).images[0]

        # Save image with timestamp and sanitized prompt in assets folder
        os.makedirs("assets", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snippet = re.sub(r"[^A-Za-z0-9_]+","_", "_".join(prompt.split()[:6]) or "image")[:48].strip("_")
        path = os.path.join("assets", f"generated_{snippet}_{ts}_{w}x{h}_s{steps}.png")
        image.save(path)

        # Return path and result message
        return {
            "result": f"Image generated → {path}",
            "image_path": path,
        }
