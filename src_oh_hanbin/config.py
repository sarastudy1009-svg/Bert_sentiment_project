"""한국어 감성분석 패키지(src_oh_hanbin)에서 사용하는 경로와 기본 설정을 모아 둔 파일입니다."""

from pathlib import Path

# 현재 파일(config.py)의 위치를 기준으로 프로젝트 루트 경로를 계산합니다.
BASE_DIR = Path(__file__).resolve().parents[1]

# CSV 데이터 파일이 저장되는 기본 폴더 경로입니다.
DATA_DIR = BASE_DIR / "data"

# ▼▼▼ [변경] 모델 저장 경로 변경 ▼▼▼
# 기존: BASE_DIR / "models" / "bert_sentiment"  (영어 BERT 모델 경로)
# 변경: BASE_DIR / "models" / "ko_bert_sentiment" (한국어 모델 전용 경로로 분리)
MODEL_DIR = BASE_DIR / "models" / "ko_bert_sentiment"
# ▲▲▲ [변경] 모델 저장 경로 변경 ▲▲▲

# 예제 데이터 경로입니다.
DEFAULT_DATA_PATH = DATA_DIR / "sample_sentiment.csv"

# ▼▼▼ [변경] 기본 모델명 변경 ▼▼▼
# 기존: "bert-base-uncased"  → 영어 전용 BERT 모델
# 변경: "monologg/koelectra-base-finetuned-sentiment"
#        → 한국어 이진 감성분석(긍정/부정)에 특화된 KoELECTRA 기반 모델
#        → Hugging Face Hub에서 자동 다운로드되며 별도 학습 없이 바로 사용 가능
DEFAULT_MODEL_NAME = "monologg/koelectra-base-finetuned-sentiment"
# ▲▲▲ [변경] 기본 모델명 변경 ▲▲▲

# 모델 입력에 사용할 최대 토큰 길이입니다.
MAX_LEN = 128

# 학습과 데이터 분리에 사용할 난수 시드입니다.
SEED = 42

# 숫자 라벨을 화면 표시용 문자열로 바꾸기 위한 딕셔너리입니다.
ID_TO_LABEL = {0: "부정", 1: "긍정"}

# 문자열 라벨을 학습용 숫자 라벨로 바꾸기 위한 딕셔너리입니다.
LABEL_TO_ID = {"negative": 0, "부정": 0, "0": 0, 0: 0, "positive": 1, "긍정": 1, "1": 1, 1: 1}
