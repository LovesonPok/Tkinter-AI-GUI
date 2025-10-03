from transformers import pipeline
from utils.decorators import log_action, timeit
from models.base import BaseModelAdapter

# Adapter for sentiment analysis using DistilBERT
class TextSentimentAdapter(BaseModelAdapter):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    category = "Text Classification"
    description = "Sentiment (positive/negative) using DistilBERT."

    # Load the sentiment-analysis pipeline
    def load(self):
        self._pipe = pipeline("sentiment-analysis", model=self.model_name)

    # Run sentiment analysis on input text
    @log_action
    @timeit
    def run(self, payload):
        text = (payload or "").strip()  # Ensure text is not empty
        if not text:
            return {"result": "Enter text in the box."}

        # Get sentiment prediction
        out = self._pipe(text)[0]
        return {"result": f"{out['label']} ({out['score']:.2f})"}
