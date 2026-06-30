"""Streamlit으로 문장을 입력받고 BERT 감성분석 결과를 출력하는 앱입니다."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit을 프로젝트 루트 밖에서 실행해도 src_moon_dongwon 패키지를 찾을 수 있도록 프로젝트 루트를 Python 경로에 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 프로젝트 루트 경로가 sys.path에 없으면 추가합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 패키지 경로를 src에서 src_moon_dongwon으로 수정합니다.
from src_moon_dongwon.config import MODEL_DIR  # noqa: E402
from src_moon_dongwon.predict import SentimentPredictor  # noqa: E402


@st.cache_resource
def load_predictor() -> SentimentPredictor:
    """Streamlit이 화면을 다시 그릴 때마다 모델을 다시 로드하지 않도록 캐싱합니다."""
    # 저장된 모델 폴더가 있으면 해당 모델을 사용하고, 없으면 기본 BERT 분류 모델을 사용합니다.
    return SentimentPredictor(model_dir=MODEL_DIR)


def main() -> None:
    """Streamlit 화면을 구성하고 사용자 입력에 대한 예측 결과를 출력합니다."""
    # 브라우저 탭 제목과 화면 레이아웃을 설정합니다.
    st.set_page_config(page_title="BERT 감성분석", page_icon="🤖", layout="centered")

    # 앱의 큰 제목을 출력합니다.
    st.title("BERT 문장 감성분석")

    # 앱 사용 목적을 짧게 설명합니다.
    st.write("문장을 입력하면 BERT 분류 모델이 긍정 또는 부정 결과를 예측합니다.")

    # 학습된 모델이 없는 경우 기본 사전 학습 모델이 사용될 수 있음을 안내합니다.
    if not MODEL_DIR.exists():
        st.warning("학습된 모델 폴더가 없습니다. 먼저 `python -m src_moon_dongwon.train` 명령으로 모델을 학습하면 더 정확한 결과를 볼 수 있습니다.")

    # 예측할 문장을 입력받는 텍스트 영역을 만듭니다. 한국어 데이터셋에 맞게 기본 예시 문장을 한국어로 변경합니다.
    text = st.text_area("분석할 문장 입력", value="이 영화 정말 시간 가는 줄 모르고 봤네요!", height=120)

    # 사용자가 버튼을 누르면 예측을 실행합니다.
    if st.button("감성분석 실행", type="primary"):
        try:
            # 캐시된 예측 객체를 불러옵니다.
            predictor = load_predictor()

            # 입력 문장에 대한 예측 결과를 계산합니다.
            result = predictor.predict(text)

            # 최종 분류 결과를 크게 출력합니다.
            st.subheader(f"분류 결과: {result['label']}")

            # 긍정 확률을 progress bar로 출력합니다.
            st.write(f"긍정 확률: {result['positive_probability']:.4f}")
            st.progress(result["positive_probability"])

            # 부정 확률을 progress bar로 출력합니다.
            st.write(f"부정 확률: {result['negative_probability']:.4f}")
            st.progress(result["negative_probability"])

            # 현재 어떤 모델 경로를 사용했는지 출력합니다.
            st.caption(f"사용 모델: {result['model_path']}")

        except Exception as error:
            # 예측 중 발생한 오류를 화면에 표시하여 원인을 빠르게 확인할 수 있게 합니다.
            st.error(f"예측 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    # streamlit run app/streamlit_dong_app.py로 실행할 때 main 함수를 호출합니다.
    main()