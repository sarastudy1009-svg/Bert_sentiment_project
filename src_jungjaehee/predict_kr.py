"""사전 학습된 한국어 감성분석(BERT) 모델을 불러와 문장 긍정/부정을 예측하는 파일입니다."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src_jungjaehee.utils import get_device

# NSMC(네이버 영화리뷰)로 파인튜닝된 다국어 BERT 모델입니다.
# 별도 학습 없이 바로 한국어 문장의 긍정/부정을 예측할 수 있습니다.
DEFAULT_KOREAN_MODEL_NAME = "sangrimlee/bert-base-multilingual-cased-nsmc"

# 한국어 모델의 숫자 라벨을 화면 표시용 문자열로 바꾸기 위한 딕셔너리입니다.
ID_TO_LABEL_KR = {0: "부정", 1: "긍정"}


class KoreanSentimentPredictor:
    """Streamlit과 다른 Python 코드에서 재사용할 수 있는 한국어 감성분석 예측 클래스입니다."""

    def __init__(self, model_name: str = DEFAULT_KOREAN_MODEL_NAME, max_len: int = 128):
        # 사용할 Hugging Face 모델명을 저장합니다.
        self.model_name = model_name

        # BERT 입력 최대 토큰 길이를 저장합니다.
        self.max_len = max_len

        # GPU 또는 CPU 실행 장치를 선택합니다. 영어 predict.py와 동일한 유틸 함수를 재사용합니다.
        self.device = get_device()

        # 한국어 NSMC 파인튜닝 모델에 맞는 Tokenizer를 불러옵니다.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 이진 분류(긍정/부정)용 모델을 불러옵니다.
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # 모델을 선택된 장치로 이동합니다.
        self.model.to(self.device)

        # 예측만 수행하므로 dropout 등을 비활성화하는 평가 모드로 전환합니다.
        self.model.eval()

    def predict(self, text: str) -> dict[str, float | str]:
        """입력된 한국어 문장 하나에 대해 부정/긍정 예측 결과와 확률을 반환합니다."""
        # 빈 문장을 예측하지 않도록 앞뒤 공백을 제거합니다.
        text = text.strip()

        # 사용자가 문장을 입력하지 않은 경우 명확한 오류를 발생시킵니다.
        if not text:
            raise ValueError("예측할 문장을 입력하세요.")

        # 입력 문장을 BERT 입력 텐서로 변환합니다.
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # 입력 텐서를 모델이 위치한 장치로 이동합니다.
        input_ids = encoded["input_ids"].to(self.device)

        # attention_mask도 같은 장치로 이동합니다.
        attention_mask = encoded["attention_mask"].to(self.device)

        # 예측 과정에서는 기울기 계산이 필요 없으므로 torch.no_grad()로 메모리 사용량을 줄입니다.
        with torch.no_grad():
            # 모델에 입력을 넣어 각 클래스에 대한 로짓을 계산합니다.
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # 로짓을 확률처럼 해석할 수 있도록 softmax를 적용합니다.
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)

            # 가장 확률이 높은 클래스 번호를 가져옵니다.
            predicted_id = int(torch.argmax(probabilities).item())

        # 예측 결과를 화면 출력에 편한 딕셔너리 형태로 반환합니다.
        return {
            "label": ID_TO_LABEL_KR[predicted_id],
            "negative_probability": float(probabilities[0].item()),
            "positive_probability": float(probabilities[1].item()),
            "model_path": self.model_name,
        }
