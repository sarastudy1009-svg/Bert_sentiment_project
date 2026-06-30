import sys
import subprocess
from pathlib import Path
import streamlit as st

# 프로젝트 루트 경로를 시스템 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import SentimentPredictor as PredictorEn
from src.config import MODEL_DIR
import src_ha_jungwon.config as ConfigKr
import src_ha_jungwon.predict as PredictorKr


@st.cache_resource
def load_predictors():
    """모델들을 로드합니다."""
    return {
        "English (BERT)": PredictorEn(model_dir=MODEL_DIR),
        "Korean (Custom)": PredictorKr.SentimentPredictor(model_dir=ConfigKr.MODEL_DIR)
    }


def run_training(command: list[str]) -> tuple[bool, str]:
    """주어진 학습 명령어를 실행하고 성공 여부와 로그를 반환합니다."""
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    success = result.returncode == 0
    log = result.stdout + "\n" + result.stderr
    return success, log


def main() -> None:
    st.set_page_config(page_title="BERT 감성분석", page_icon="🤖", layout="centered")
    st.title("BERT 문장 감성분석")

    # ── 모델 학습 영역 ──────────────────────────────────────────
    st.subheader("모델 학습")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("영어 모델 학습하기"):
            command = [
                sys.executable, "-m", "src.train",
                "--data_path", "data/sample_sentiment.csv",
                "--epochs", "1",
                "--train_batch_size", "2",
                "--eval_batch_size", "2",
            ]
            with st.spinner("영어 모델 학습 중입니다... 시간이 걸릴 수 있습니다."):
                success, log = run_training(command)

            if success:
                st.success("영어 모델 학습이 완료되었습니다.")
                load_predictors.clear()
            else:
                st.error("영어 모델 학습 중 오류가 발생했습니다.")

            with st.expander("학습 로그 보기"):
                st.code(log)

    with col2:
        if st.button("한국어 모델 학습하기"):
            command = [
                sys.executable, "-m", "src_ha_jungwon.train",
                "--data_path", "data/ratings_train.txt",
                "--epochs", "1",
                "--train_batch_size", "2",
                "--eval_batch_size", "2",
            ]
            with st.spinner("한국어 모델 학습 중입니다... 시간이 걸릴 수 있습니다."):
                success, log = run_training(command)

            if success:
                st.success("한국어 모델 학습이 완료되었습니다.")
                load_predictors.clear()
            else:
                st.error("한국어 모델 학습 중 오류가 발생했습니다.")

            with st.expander("학습 로그 보기"):
                st.code(log)

    st.divider()

    # ── 감성 분석 영역 ──────────────────────────────────────────
    st.subheader("감성 분석")
    model_option = st.selectbox("사용할 모델을 선택하세요:", ["English (BERT)", "Korean (Custom)"])
    text = st.text_area("분석할 문장 입력", value="이 영화 정말 재미있어요!", height=120)

    if st.button("감성분석 실행", type="primary"):
        try:
            predictors = load_predictors()
            predictor = predictors[model_option]
            result = predictor.predict(text)

            st.subheader(f"분류 결과: {result['label']}")
            st.write(f"긍정 확률: {result['positive_probability']:.4f}")
            st.progress(result["positive_probability"])

            st.write(f"부정 확률: {result['negative_probability']:.4f}")
            st.progress(result["negative_probability"])

            st.caption(f"사용 모델: {model_option}")

        except Exception as error:
            st.error(f"예측 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()