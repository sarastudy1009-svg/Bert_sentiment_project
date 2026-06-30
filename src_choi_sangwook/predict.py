"""Prediction helper for the saved Korean BERT sentiment model."""

from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from src_choi_sangwook.config import DEFAULT_MODEL_NAME, ID_TO_LABEL, MAX_LEN, MODEL_DIR
from src_choi_sangwook.utils import get_device


class SentimentPredictor:
    """Load a trained local model if available, otherwise use the base Korean BERT."""

    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        fallback_model_name: str = DEFAULT_MODEL_NAME,
        max_len: int = MAX_LEN,
    ):
        self.model_dir = Path(model_dir)
        self.fallback_model_name = fallback_model_name
        self.max_len = max_len
        self.device = get_device()

        has_saved_model = (self.model_dir / "config.json").exists()
        self.load_path = str(self.model_dir) if has_saved_model else self.fallback_model_name

        self.tokenizer = BertTokenizerFast.from_pretrained(self.load_path)
        self.model = BertForSequenceClassification.from_pretrained(self.load_path, num_labels=2)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, float | str]:
        """Predict negative/positive sentiment for one Korean review sentence."""
        text = text.strip()
        if not text:
            raise ValueError("예측할 문장을 입력하세요.")

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            predicted_id = int(torch.argmax(probabilities).item())

        return {
            "label": ID_TO_LABEL[predicted_id],
            "negative_probability": float(probabilities[0].item()),
            "positive_probability": float(probabilities[1].item()),
            "model_path": self.load_path,
        }
