# models/text_to_image.py — HQ CPU preset (Stable Diffusion 1.5)
import os, re, datetime, torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from models.base import BaseModelAdapter

class TextToImageAdapter(BaseModelAdapter):
    model_name  = "runwayml/stable-diffusion-v1-5"
    category    = "Text-to-Image"
    description = "High-quality text-to-image on CPU (SD 1.5, DPM-Solver)."

    def load(self):
        # Force CPU on your laptop; float32 avoids black images on CPU.
        self._device = "cpu"
        # Use torch_dtype (works across diffusers versions; ignore deprecation warning if shown)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        # Better scheduler for quality with fewer steps
        try:
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        except Exception:
            pass

        # Safety checker → no-op for classroom demo (prevents blank outputs)
        if hasattr(self._pipe, "safety_checker"):
            def _noop(images, **kwargs): return images, [False] * len(images)
            self._pipe.safety_checker = _noop

        # Light memory tweaks on CPU
        for fn in ("enable_attention_slicing", "enable_vae_slicing"):
            try: getattr(self._pipe, fn)()
            except Exception: pass

        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.to(self._device)

    def run(self, payload):
        # Accept dict from InputFrame OR raw string
        if isinstance(payload, dict):
            prompt = (payload.get("prompt") or payload.get("text") or "").strip()
        else:
            prompt = (payload or "").strip()

        if not prompt:
            return {"result": "Enter a text prompt."}

        # Quality-oriented CPU defaults
        steps = 30         # 25–40: more steps = better (slower)
        cfg   = 7.5
        h, w  = 384, 384   # 512x512 looks better but is slower
        neg   = "blurry, lowres, bad anatomy, extra limbs, watermark, text, jpeg artifacts"

        image = self._pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=(h//8)*8, width=(w//8)*8
        ).images[0]

        os.makedirs("assets", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snippet = re.sub(r"[^A-Za-z0-9_]+","_", "_".join(prompt.split()[:6]) or "image")[:48].strip("_")
        path = os.path.join("assets", f"generated_{snippet}_{ts}_{w}x{h}_s{steps}.png")
        image.save(path)

        return {
            "result": f"Image generated → {path}",
            "image_path": path,
        }
