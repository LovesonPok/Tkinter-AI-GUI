from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from utils.decorators import log_action, timeit
from models.base import BaseModelAdapter

# Adapter for generating captions from images using BLIP-large
class ImageToTextAdapter(BaseModelAdapter):
    model_name  = "Salesforce/blip-image-captioning-large"
    category    = "Image-to-Text"
    description = "Generates a descriptive caption for an image (BLIP-large)."

    # Load processor and model onto device (GPU if available)
    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(device)

    # Run the image-to-text generation
    @log_action
    @timeit
    def run(self, payload):
        # Get image path from payload
        if isinstance(payload, dict):
            path = (payload.get("image_path") or payload.get("prompt") or "").strip()
        else:
            path = (payload or "").strip()

        if not path:
            return {"result": "Choose an image file first."}

        # Open image and prepare inputs for the model
        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        # Generate caption
        out = self._model.generate(**inputs, max_new_tokens=40)
        caption = self._processor.decode(out[0], skip_special_tokens=True).strip()

        # Return the caption with the image path
        return {"result": f"Caption: {caption}", "image_path": path}
