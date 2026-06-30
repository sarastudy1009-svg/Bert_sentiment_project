"""한국어 감성분석 모델을 불러와 문장 긍정/부정을 예측하는 파일입니다."""

from pathlib import Path

import torch

# ▼▼▼ [변경] 임포트 클래스 변경 ▼▼▼
# 기존: from transformers import BertForSequenceClassification, BertTokenizerFast
#        → BERT 전용 클래스라 KoELECTRA 같은 비BERT 계열 모델을 불러올 수 없음
# 변경: AutoModelForSequenceClassification, AutoTokenizer 사용
#        → 모델명만 바꾸면 BERT/ELECTRA/RoBERTa 등 어떤 모델이든 자동으로 올바른 클래스를 선택해줌
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# ▲▲▲ [변경] 임포트 클래스 변경 ▲▲▲

# ▼▼▼ [변경] 임포트 경로 수정 ▼▼▼
# 기존: from src.config import ...   → 영어 패키지(src)의 설정을 참조하는 버그
# 변경: from src_oh_hanbin.config import ...  → 한국어 패키지 자체 설정을 참조하도록 수정
from src_oh_hanbin.config import DEFAULT_MODEL_NAME, ID_TO_LABEL, MAX_LEN, MODEL_DIR
# ▲▲▲ [변경] 임포트 경로 수정 ▲▲▲

from src.utils import get_device


# ▼▼▼ [변경] 클래스명 변경 ▼▼▼
# 기존: class SentimentPredictor  → 영어/한국어 구분이 없는 이름
# 변경: class KoreanSentimentPredictor  → 한국어 전용 클래스임을 명확히 표시
class KoreanSentimentPredictor:
# ▲▲▲ [변경] 클래스명 변경 ▲▲▲
    """한국어 감성분석 예측 클래스입니다. Streamlit과 일반 Python 코드에서 재사용 가능합니다."""

    def __init__(self, model_dir: str | Path = MODEL_DIR, fallback_model_name: str = DEFAULT_MODEL_NAME, max_len: int = MAX_LEN):
        self.model_dir = Path(model_dir)
        self.fallback_model_name = fallback_model_name
        self.max_len = max_len
        self.device = get_device()

        # 로컬 모델 폴더가 있으면 해당 경로에서, 없으면 Hugging Face Hub에서 모델을 불러옵니다.
        self.load_path = str(self.model_dir) if self.model_dir.exists() else self.fallback_model_name

        # ▼▼▼ [변경] 토크나이저 클래스 변경 ▼▼▼
        # 기존: BertTokenizerFast.from_pretrained(...)  → BERT 전용, KoELECTRA에 사용 불가
        # 변경: AutoTokenizer.from_pretrained(...)      → 모델에 맞는 토크나이저를 자동 선택
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_path)
        # ▲▲▲ [변경] 토크나이저 클래스 변경 ▲▲▲

        # ▼▼▼ [변경] 모델 클래스 변경 ▼▼▼
        # 기존: BertForSequenceClassification.from_pretrained(...)  → BERT 전용
        # 변경: AutoModelForSequenceClassification.from_pretrained(...) → 모델 아키텍처를 자동 감지해 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(self.load_path, num_labels=2)
        # ▲▲▲ [변경] 모델 클래스 변경 ▲▲▲

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, float | str]:
        """입력 한국어 문장에 대해 부정/긍정 예측 결과와 확률을 반환합니다."""
        text = text.strip()
        if not text:
            raise ValueError("예측할 문장을 입력하세요.")

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
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
