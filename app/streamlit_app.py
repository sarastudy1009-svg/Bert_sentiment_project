"""Streamlit으로 문장을 입력받고 BERT 감성분석 결과를 출력하는 앱입니다."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit을 프로젝트 루트 밖에서 실행해도 src 패키지를 찾을 수 있도록 프로젝트 루트를 Python 경로에 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 프로젝트 루트 경로가 sys.path에 없으면 추가합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_DIR  # noqa: E402
from src.predict import SentimentPredictor  # noqa: E402

# ▼▼▼ [추가] 한국어 패키지 임포트 ▼▼▼
# 기존: 영어 패키지(src)만 임포트
# 변경: 한국어 전용 패키지(src_oh_hanbin)의 설정과 예측 클래스를 추가로 임포트
#        KO_MODEL_DIR: 한국어 모델 저장 경로 (영어 MODEL_DIR과 이름 충돌 방지를 위해 별칭 사용)
from src_oh_hanbin.config import MODEL_DIR as KO_MODEL_DIR  # noqa: E402
from src_oh_hanbin.predict import KoreanSentimentPredictor  # noqa: E402
# ▲▲▲ [추가] 한국어 패키지 임포트 ▲▲▲


@st.cache_resource
def load_predictor() -> SentimentPredictor:
    """Streamlit이 화면을 다시 그릴 때마다 모델을 다시 로드하지 않도록 캐싱합니다."""
    return SentimentPredictor(model_dir=MODEL_DIR)


# ▼▼▼ [추가] 한국어 모델 로더 함수 ▼▼▼
# 기존: 영어 모델 로더(load_predictor)만 존재
# 변경: 한국어 모델을 위한 별도 로더 함수 추가
#        @st.cache_resource 로 캐싱하여 버튼 클릭마다 모델을 다시 불러오는 낭비를 방지
@st.cache_resource
def load_korean_predictor() -> KoreanSentimentPredictor:
    """한국어 감성분석 모델을 캐싱하여 재로드를 방지합니다."""
    return KoreanSentimentPredictor(model_dir=KO_MODEL_DIR)
# ▲▲▲ [추가] 한국어 모델 로더 함수 ▲▲▲


def main() -> None:
    """Streamlit 화면을 구성하고 사용자 입력에 대한 예측 결과를 출력합니다."""
    st.set_page_config(page_title="BERT 감성분석", page_icon="🤖", layout="centered")

    st.title("BERT 문장 감성분석")
    st.write("문장을 입력하면 BERT 분류 모델이 긍정 또는 부정 결과를 예측합니다.")

    # ════════════════════════════════════════════════════════════
    # ▼▼▼ [변경] 영어 섹션에 헤더 추가 ▼▼▼
    # 기존: 헤더 없이 바로 text_area 출력
    # 변경: st.header()로 영어/한국어 섹션을 시각적으로 구분
    st.header("영어 리뷰 문장")
    # ▲▲▲ [변경] 영어 섹션에 헤더 추가 ▲▲▲

    if not MODEL_DIR.exists():
        st.warning("학습된 모델 폴더가 없습니다. 먼저 `python -m src.train` 명령으로 모델을 학습하면 더 정확한 결과를 볼 수 있습니다.")

    text = st.text_area("분석할 문장 입력", value="This movie was wonderful and I loved it.", height=120)

    if st.button("감성분석 실행", type="primary"):
        try:
            predictor = load_predictor()
            result = predictor.predict(text)

            st.subheader(f"분류 결과: {result['label']}")
            st.write(f"긍정 확률: {result['positive_probability']:.4f}")
            st.progress(result["positive_probability"])
            st.write(f"부정 확률: {result['negative_probability']:.4f}")
            st.progress(result["negative_probability"])
            st.caption(f"사용 모델: {result['model_path']}")

        except Exception as error:
            st.error(f"예측 중 오류가 발생했습니다: {error}")

    # ▼▼▼ [추가] 섹션 구분선 ▼▼▼
    # 기존: 구분선 없음 (영어 섹션만 존재)
    # 변경: st.divider()로 영어 섹션과 한국어 섹션을 시각적으로 분리
    st.divider()
    # ▲▲▲ [추가] 섹션 구분선 ▲▲▲

    # ════════════════════════════════════════════════════════════
    # ▼▼▼ [추가] 한국어 감성분석 섹션 전체 신규 추가 ▼▼▼
    # 기존: 해당 섹션 없음
    # 변경: 한국어 리뷰 입력 → "감성분석" 버튼 클릭 → KoreanSentimentPredictor로 결과 출력
    st.header("한국어 리뷰 문장")
    st.write("한국어 문장을 입력하면 KoELECTRA 기반 모델이 긍정/부정을 예측합니다.")

    # 실습 요구사항: input_text 변수로 입력받고 key="ko_input"으로 영어 text_area와 충돌 방지
    input_text = st.text_area("한국어 분석할 문장 입력", value="이 영화는 정말 감동적이고 너무 좋았어요.", height=120, key="ko_input")

    # 실습 요구사항: 버튼 이름 "감성분석", key="ko_btn"으로 영어 버튼과 충돌 방지
    if st.button("감성분석", type="primary", key="ko_btn"):
        try:
            ko_predictor = load_korean_predictor()
            ko_result = ko_predictor.predict(input_text)

            st.subheader(f"분류 결과: {ko_result['label']}")
            st.write(f"긍정 확률: {ko_result['positive_probability']:.4f}")
            st.progress(ko_result["positive_probability"])
            st.write(f"부정 확률: {ko_result['negative_probability']:.4f}")
            st.progress(ko_result["negative_probability"])
            st.caption(f"사용 모델: {ko_result['model_path']}")

        except Exception as error:
            st.error(f"예측 중 오류가 발생했습니다: {error}")
    # ▲▲▲ [추가] 한국어 감성분석 섹션 전체 신규 추가 ▲▲▲


if __name__ == "__main__":
    # streamlit run app/streamlit_app.py로 실행할 때 main 함수를 호출합니다.
    main()
