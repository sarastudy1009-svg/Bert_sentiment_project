"""Streamlit으로 문장을 입력받고 BERT 감성분석 결과를 출력하는 앱입니다."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit을 프로젝트 루트 밖에서 실행해도 src 패키지를 찾을 수 있도록 프로젝트 루트를 Python 경로에 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 프로젝트 루트 경로가 sys.path에 없으면 추가합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 본인 브랜치 패키지(src_lee_taehyuk)의 한국어 감성분석 모델을 함께 불러옵니다.
from src_lee_taehyuk.config import MODEL_DIR as MODEL_DIR_KOR  # noqa: E402
from src_lee_taehyuk.predict import SentimentPredictor as SentimentPredictorKor  # noqa: E402

@st.cache_resource
def load_predictor_kor() -> SentimentPredictorKor:
    """한국어 감성분석 모델(klue/bert-base 기반)을 캐싱하여 불러옵니다."""
    # 저장된 한국어 모델 폴더가 있으면 그 모델을 사용하고, 없으면 기본 klue/bert-base를 사용합니다.
    return SentimentPredictorKor(model_dir=MODEL_DIR_KOR)


def main() -> None:
    """Streamlit 화면을 구성하고 사용자 입력에 대한 예측 결과를 출력합니다."""
    # 브라우저 탭 제목과 화면 레이아웃을 설정합니다.
    st.set_page_config(page_title="BERT 감성분석", page_icon="🤖", layout="centered")

    # 앱의 큰 제목을 출력합니다.
    st.title("BERT 문장 감성분석")

    # 앱 사용 목적을 짧게 설명합니다.
    st.write("문장을 입력하면 BERT 분류 모델이 긍정 또는 부정 결과를 예측합니다.")

    # ------------------------------------------------------------------
    # 한국 리뷰 입력 필드는 위에 그대로 두고, 아래에 한국어 리뷰 입력 섹션을 추가합니다.
    # ------------------------------------------------------------------
    st.divider()
    st.header("한국어 BERT 문장 감성분석")
    st.write("한국어 문장을 입력하면 klue/bert-base 기반 모델이 긍정 또는 부정 결과를 예측합니다.")

    # 한국어 학습 모델이 없는 경우 기본 사전 학습 모델(klue/bert-base)이 사용될 수 있음을 안내합니다.
    if not MODEL_DIR_KOR.exists():
        st.warning(
            "한국어로 학습된 모델 폴더가 없습니다. "
            "먼저 `python -m src_lee_taehyuk.train` 명령으로 모델을 학습하면 더 정확한 결과를 볼 수 있습니다."
        )

    # 한국어 리뷰 문장을 입력받는 텍스트 영역입니다. (요구사항: input_text로 입력받음)
    input_text = st.text_area(
        "한국어 리뷰 문장",
        value="이 영화 정말 감동적이었고 연기도 훌륭했어요.",
        height=120,
        key="input_text_kor",
    )

    # '감성분석' 버튼을 클릭하면 한국어 모델로 예측을 실행합니다.
    if st.button("감성분석", type="primary", key="run_kor_sentiment"):
        try:
            # 캐시된 한국어 예측 객체를 불러옵니다.
            predictor_kor = load_predictor_kor()

            # 입력 문장에 대한 예측 결과를 계산합니다.
            result_kor = predictor_kor.predict(input_text)

            # 최종 분류 결과를 크게 출력합니다.
            st.subheader(f"분류 결과: {result_kor['label']}")

            # 긍정 확률을 progress bar로 출력합니다.
            st.write(f"긍정 확률: {result_kor['positive_probability']:.4f}")
            st.progress(result_kor["positive_probability"])

            # 부정 확률을 progress bar로 출력합니다.
            st.write(f"부정 확률: {result_kor['negative_probability']:.4f}")
            st.progress(result_kor["negative_probability"])

            # 현재 어떤 모델 경로를 사용했는지 출력합니다.
            st.caption(f"사용 모델: {result_kor['model_path']}")

        except Exception as error:
            # 예측 중 발생한 오류를 화면에 표시하여 원인을 빠르게 확인할 수 있게 합니다.
            st.error(f"예측 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    # streamlit run app/streamlit_app.py로 실행할 때 main 함수를 호출합니다.
    main()
