"""Streamlit app for English and Korean BERT sentiment analysis."""

import sys
from pathlib import Path
from typing import Dict, Union

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_DIR as ENGLISH_MODEL_DIR  # noqa: E402
from src.predict import SentimentPredictor as EnglishSentimentPredictor  # noqa: E402
from src_hong_giback.config import MODEL_DIR as KOREAN_MODEL_DIR  # noqa: E402
from src_hong_giback.predict import KoreanSentimentPredictor  # noqa: E402


@st.cache_resource
def load_english_predictor() -> EnglishSentimentPredictor:
    return EnglishSentimentPredictor(model_dir=ENGLISH_MODEL_DIR)


@st.cache_resource
def load_korean_predictor() -> KoreanSentimentPredictor:
    return KoreanSentimentPredictor(model_dir=KOREAN_MODEL_DIR)


PredictionResult = Dict[str, Union[float, str]]


def show_result(result: PredictionResult) -> None:
    st.subheader(f"분류 결과: {result['label']}")
    st.write(f"긍정 확률: {result['positive_probability']:.4f}")
    st.progress(float(result["positive_probability"]))
    st.write(f"부정 확률: {result['negative_probability']:.4f}")
    st.progress(float(result["negative_probability"]))
    st.caption(f"사용 모델: {result['model_path']}")


def main() -> None:
    st.set_page_config(page_title="BERT 감성분석", page_icon="💬", layout="centered")
    st.title("BERT 문장 감성분석")

    if not ENGLISH_MODEL_DIR.exists():
        st.warning("영어 학습 모델 폴더가 없습니다. 기본 영어 BERT 모델을 사용합니다.")

    st.header("영어 리뷰 감성 분석")
    english_text = st.text_area(
        "분석할 영어 문장 입력",
        value="This movie was wonderful and I loved it.",
        height=120,
        key="english_text",
    )

    if st.button("영어 감성분석 실행", type="primary"):
        try:
            result = load_english_predictor().predict(english_text)
            show_result(result)
        except Exception as error:
            st.error(f"영어 감성분석 중 오류가 발생했습니다: {error}")

    st.divider()

    st.header("한국어 리뷰 감성 분석")
    input_text = st.text_area(
        "한국어 리뷰 문장:",
        value="이 영화 정말 재미있고 감동적이었어요.",
        height=120,
        key="input_text",
    )

    if st.button("감성분석"):
        try:
            result = load_korean_predictor().predict(input_text)
            show_result(result)
        except Exception as error:
            st.error(f"한국어 감성분석 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
