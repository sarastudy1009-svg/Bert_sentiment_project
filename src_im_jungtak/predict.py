import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import MODEL_SAVE_DIR, ID2LABEL, MAX_LEN
from .utils import get_device

class SentimentPredictor:
    def __init__(self, model_path: str = str(MODEL_SAVE_DIR)):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path

    def predict(self, text: str):
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_id = int(torch.argmax(probs))
        return {
            "label": ID2LABEL[pred_id],
            "positive_prob": float(probs[1]),
            "negative_prob": float(probs[0]),
            "model_path": self.model_path,
        }