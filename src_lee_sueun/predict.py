"""한국어 리뷰 문장의 긍정/부정을 예측하는 파일입니다."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src_lee_sueun.config import DEFAULT_MODEL_NAME, ID_TO_LABEL, MAX_LEN, MODEL_DIR
from src_lee_sueun.utils import get_device


class KoreanSentimentPredictor:
    """Streamlit에서 재사용할 수 있는 한국어 감성분석 예측 클래스입니다."""

    def __init__(self, model_dir: str | Path = MODEL_DIR, fallback_model_name: str = DEFAULT_MODEL_NAME, max_len: int = MAX_LEN):
        self.model_dir = Path(model_dir)
        self.fallback_model_name = fallback_model_name
        self.max_len = max_len
        self.device = get_device()
        self.load_path = str(self.model_dir) if self.model_dir.exists() else self.fallback_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.load_path)
        self.model.to(self.device)
        self.model.eval()

    def _label_for_id(self, predicted_id: int) -> str:
        """모델 설정의 label 이름을 한국어 화면 표시용 라벨로 바꿉니다."""
        raw_label = str(self.model.config.id2label.get(predicted_id, ID_TO_LABEL.get(predicted_id, predicted_id))).lower()

        if raw_label in {"1", "label_1", "positive", "pos", "긍정"}:
            return "긍정"
        if raw_label in {"0", "label_0", "negative", "neg", "부정"}:
            return "부정"

        return ID_TO_LABEL.get(predicted_id, str(raw_label))

    def predict(self, text: str) -> dict[str, float | str]:
        """입력된 한국어 리뷰 하나에 대한 감성분석 결과와 확률을 반환합니다."""
        text = text.strip()

        if not text:
            raise ValueError("한국어 리뷰 문장을 입력하세요.")

        encoded = self.tokenizer(
            text,
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

        negative_probability = float(probabilities[0].item()) if len(probabilities) > 0 else 0.0
        positive_probability = float(probabilities[1].item()) if len(probabilities) > 1 else 0.0

        return {
            "label": self._label_for_id(predicted_id),
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
            "model_path": self.load_path,
        }
