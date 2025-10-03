from transformers import pipeline
from PIL import Image
from utils.decorators import log_action, timeit
from models.base import BaseModelAdapter

# Adapter class for image classification using Vision Transformer (ViT)
class ImageClassifierAdapter(BaseModelAdapter):
    # Model configuration
    model_name = "google/vit-base-patch16-224"  # Pretrained ViT model from Hugging Face
    category = "Image Classification"           # Defines the type of task
    description = "Classifies an image with ViT."  # Short description of the adapter

    # Method to load the model pipeline
    def load(self):
        # Initialize the image classification pipeline using the selected model
        self._pipe = pipeline("image-classification", model=self.model_name)

    # Main method to run classification on the input image
    @log_action
    @timeit
    def run(self, payload):
        # Determine the image path from payload, which can be a dict or string
        if isinstance(payload, dict):
            # Try to get image path from 'image_path' or fallback to 'prompt'
            path = (payload.get("image_path") or payload.get("prompt") or "").strip()
        else:
            # If payload is a plain string, use it directly
            path = (payload or "").strip()

        # Return message if no image path is provided
        if not path:
            return {"result": "Choose an image file first."}

        # Open the image and ensure it is in RGB format
        img = Image.open(path).convert("RGB")

        # Run the image through the classification pipeline
        pred = self._pipe(img)[0]

        # Return the top prediction label and confidence score along with the image path
        return {"result": f"{pred['label']} ({pred['score']:.2f})", "image_path": path}
