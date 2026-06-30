"""Korean review sentiment prediction with a Hugging Face BERT model."""

from pathlib import Path
from typing import Any, Dict, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src_hong_giback.config import DEFAULT_MODEL_NAME, ID_TO_LABEL, MAX_LEN, MODEL_DIR
from src_hong_giback.utils import get_device


PathLike = Union[str, Path]
PredictionResult = Dict[str, Union[float, str]]


class KoreanSentimentPredictor:
    """Predict positive or negative sentiment for Korean review text."""

    def __init__(
        self,
        model_dir: PathLike = MODEL_DIR,
        fallback_model_name: str = DEFAULT_MODEL_NAME,
        max_len: int = MAX_LEN,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.fallback_model_name = fallback_model_name
        self.max_len = max_len
        self.device = get_device()
        self.load_path = str(self.model_dir) if self.model_dir.exists() else self.fallback_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.load_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_text: str) -> PredictionResult:
        """Return Korean sentiment label and class probabilities for one sentence."""
        input_text = input_text.strip()
        if not input_text:
            raise ValueError("한국어 리뷰 문장을 입력하세요.")

        encoded = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            predicted_id = int(torch.argmax(probabilities).item())

        negative_probability = self._probability_for_class(probabilities, 0)
        positive_probability = self._probability_for_class(probabilities, 1)

        return {
            "label": self._label_for_id(predicted_id),
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
            "model_path": self.load_path,
        }

    def _label_for_id(self, label_id: int) -> str:
        label = self.model.config.id2label.get(label_id, ID_TO_LABEL.get(label_id, str(label_id)))
        normalized = str(label).lower()

        if normalized in {"label_0", "0", "negative", "neg"}:
            return "부정"
        if normalized in {"label_1", "1", "positive", "pos"}:
            return "긍정"
        if "negative" in normalized or "부정" in normalized:
            return "부정"
        if "positive" in normalized or "긍정" in normalized:
            return "긍정"

        return ID_TO_LABEL.get(label_id, str(label))

    def _probability_for_class(self, probabilities: torch.Tensor, class_id: int) -> float:
        if class_id < probabilities.numel():
            return float(probabilities[class_id].item())
        return 0.0


class SentimentPredictor(KoreanSentimentPredictor):
    """Backward-compatible class name for app imports."""


def predict_korean_sentiment(input_text: str) -> dict[str, Any]:
    """Convenience function for direct use from Streamlit or scripts."""
    predictor = KoreanSentimentPredictor()
    return predictor.predict(input_text)
