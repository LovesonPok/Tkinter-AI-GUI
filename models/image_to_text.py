from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from utils.decorators import log_action, timeit
from models.base import BaseModelAdapter

class ImageToTextAdapter(BaseModelAdapter):
    model_name  = "Salesforce/blip-image-captioning-large"
    category    = "Image-to-Text"
    description = "Generates a descriptive caption for an image (BLIP-large)."

    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(device)

    @log_action
    @timeit
    def run(self, payload):
        # Accept either a raw path string or a UI dict
        if isinstance(payload, dict):
            path = (payload.get("image_path") or payload.get("prompt") or "").strip()
        else:
            path = (payload or "").strip()

        if not path:
            return {"result": "Choose an image file first."}

        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        out = self._model.generate(**inputs, max_new_tokens=40)
        caption = self._processor.decode(out[0], skip_special_tokens=True).strip()

        return {"result": f"Caption: {caption}", "image_path": path}
