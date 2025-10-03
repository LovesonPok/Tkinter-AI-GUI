import os, re, datetime, numpy as np, torch
from PIL import Image
import imageio.v3 as iio

from models.base import BaseModelAdapter
from diffusers import (
    StableDiffusionXLPipeline,         # GPU/MPS text->image (HQ)
    StableVideoDiffusionPipeline,      # GPU/MPS img2vid
    StableDiffusionPipeline,           # CPU fallback text->image
    DPMSolverMultistepScheduler,
)


def _safe(s, n=64):
    return re.sub(r"[^A-Za-z0-9_]+", "_", "_".join(s.split()[:12]) or "video")[:n].strip("_")


class TextToVideoAdapter(BaseModelAdapter):
    category    = "Text-to-Video"
    description = "Fast mode: SDXL still + Stable Video Diffusion on CUDA/MPS; CPU fallback uses SD1.5 + Ken Burns."

    def __init__(self):
        super().__init__()
        self._device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )
        self._dtype  = torch.float16 if self._device in ("cuda", "mps") else torch.float32

    def load(self):
        os.makedirs("assets", exist_ok=True)

        if self._device in ("cuda", "mps"):
            # --- SDXL for still ---
            self._t2i = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=self._dtype,
                use_safetensors=True,
            )
            try:
                self._t2i.scheduler = DPMSolverMultistepScheduler.from_config(self._t2i.scheduler.config)
            except Exception:
                pass
            self._t2i.to(self._device)
            self._t2i.set_progress_bar_config(disable=True)

            # --- Stable Video Diffusion ---
            self._img2vid = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=self._dtype,
                use_safetensors=True,
            ).to(self._device)
            self._img2vid.set_progress_bar_config(disable=True)

            self._pipe = self._t2i

        else:
            # --- CPU fallback ---
            self._t2i_cpu = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            try:
                self._t2i_cpu.scheduler = DPMSolverMultistepScheduler.from_config(self._t2i_cpu.scheduler.config)
            except Exception:
                pass
            try: 
                self._t2i_cpu.enable_attention_slicing()
            except Exception: 
                pass
            self._t2i_cpu.to("cpu")
            self._t2i_cpu.set_progress_bar_config(disable=True)

            self._pipe = self._t2i_cpu

    def info(self):
        return {
            "Model": "SDXL + Stable Video Diffusion (fast mode)" if self._device in ("cuda","mps")
                     else "SD 1.5 (CPU fallback, fast) + Ken Burns",
            "Category": self.category,
            "Description": self.description,
            "Runtime": f"Device: {self._device}, DType: {str(self._dtype).split('.')[-1]}",
        }

    # --------- Helpers ---------
    def _generate_sdxl_still(self, prompt: str, steps=20, cfg=6.0, size=512):
        h = w = (size // 8) * 8
        if self._device == "cuda" and self._dtype == torch.float16:
            with torch.autocast("cuda"):
                img = self._t2i(prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, height=h, width=w).images[0]
        else:
            img = self._t2i(prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, height=h, width=w).images[0]
        return img.convert("RGB")

    def _svd_img2vid(self, img: Image.Image, num_frames=12, fps=10, motion_bucket_id=127, cond_aug=0.02):
        # Lower resolution for faster speed
        vid_w, vid_h = 512, 288
        base = img.resize((vid_w, vid_h), Image.LANCZOS)

        if self._device == "cuda" and self._dtype == torch.float16:
            with torch.autocast("cuda"):
                out = self._img2vid(
                    base, num_frames=num_frames, fps=fps,
                    motion_bucket_id=motion_bucket_id, noise_aug_strength=cond_aug
                )
        else:
            out = self._img2vid(
                base, num_frames=num_frames, fps=fps,
                motion_bucket_id=motion_bucket_id, noise_aug_strength=cond_aug
            )

        frames = getattr(out, "frames", None)
        if frames is None:
            frames = getattr(out, "images", None)
        if isinstance(frames, torch.Tensor):
            frames = (frames.clamp(0,1).mul(255).byte().cpu().numpy())
            frames = [Image.fromarray(f) for f in frames]
        elif isinstance(frames, list) and frames and isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(f) for f in frames]
        return frames, fps

    def _cpu_hq_still(self, prompt: str, steps=20, cfg=6.5, size=384):
        h = w = (size // 8) * 8
        img = self._t2i_cpu(
            prompt=prompt,
            negative_prompt="blurry, lowres, bad anatomy, watermark, text, jpeg artifacts, cartoon, illustration",
            num_inference_steps=steps, guidance_scale=cfg, height=h, width=w,
        ).images[0].convert("RGB")
        return img.resize((512, 512), Image.LANCZOS)

    def _ken_burns(self, img: Image.Image, seconds=2.0, fps=12, zoom_end=1.12, pan=(10, -6)):
        w, h = img.size
        n = int(seconds * fps)
        frames = []
        for t in range(n):
            a = t / max(1, n - 1)
            zoom = 1.0 * (1 - a) + zoom_end * a
            cw, ch = int(w / zoom), int(h / zoom)
            cx = w // 2 + int(pan[0] * (a - 0.5) * 2)
            cy = h // 2 + int(pan[1] * (a - 0.5) * 2)
            left = np.clip(cx - cw // 2, 0, w - cw)
            top  = np.clip(cy - ch // 2, 0, h - ch)
            frm = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)
            frames.append(frm)
        return frames, fps

    # --------- Main run ---------
    def run(self, payload, progress_callback=None):
        if isinstance(payload, dict):
            prompt = (payload.get("prompt") or payload.get("text") or "").strip()
        else:
            prompt = (payload or "").strip()

        if not prompt:
            return {"result": "Enter a text prompt."}

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = _safe(prompt)

        # ---- CUDA / MPS ----
        if self._device in ("cuda", "mps"):
            if progress_callback: progress_callback(10, "Generating still image...")
            still = self._generate_sdxl_still(prompt, steps=20, cfg=6.0, size=512)

            if progress_callback: progress_callback(50, "Animating into video...")
            frames, fps = self._svd_img2vid(still, num_frames=12, fps=10)

            still_path = os.path.join("assets", f"t2v_still_{safe}_{ts}.png")
            still.save(still_path)
            mp4_path   = os.path.join("assets", f"t2v_{safe}_{ts}_svd.mp4")

            if progress_callback: progress_callback(80, "Encoding video...")
            iio.imwrite(mp4_path, [f.convert("RGB") for f in frames], fps=fps, codec="h264", quality=6)

            if progress_callback: progress_callback(100, "Done ✅")

            return {
                "result": f"Video generated → {mp4_path}\nStill: {still_path}",
                "video_path": mp4_path,
                "still_path": still_path,
            }

        # ---- CPU fallback ----
        if progress_callback: progress_callback(10, "Generating still (CPU)...")
        still = self._cpu_hq_still(prompt, steps=20, cfg=6.5, size=384)
        still_path = os.path.join("assets", f"t2v_still_{safe}_{ts}.png")
        still.save(still_path)

        if progress_callback: progress_callback(50, "Applying Ken Burns...")
        frames, fps = self._ken_burns(still, seconds=2.0, fps=12)

        mp4_path = os.path.join("assets", f"t2v_{safe}_{ts}_kb.mp4")
        if progress_callback: progress_callback(80, "Encoding video...")
        iio.imwrite(mp4_path, [f.convert("RGB") for f in frames], fps=fps, codec="h264", quality=6)

        if progress_callback: progress_callback(100, "Done ✅")

        return {
            "result": f"Video generated → {mp4_path}\nStill: {still_path}",
            "video_path": mp4_path,
            "still_path": still_path,
        }
