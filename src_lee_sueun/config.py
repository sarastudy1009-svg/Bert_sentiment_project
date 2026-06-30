"""한국어 감성분석에서 사용하는 경로와 기본 설정입니다."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "bert_korean_sentiment_lee_sueun"
DEFAULT_DATA_PATH = DATA_DIR / "sample_sentiment.csv"

# NSMC(네이버 영화 리뷰) 감성분석으로 fine-tuning 된 한국어/다국어 BERT 모델입니다.
DEFAULT_MODEL_NAME = "sangrimlee/bert-base-multilingual-cased-nsmc"

MAX_LEN = 128
SEED = 42

ID_TO_LABEL = {0: "부정", 1: "긍정"}
LABEL_TO_ID = {"negative": 0, "부정": 0, "0": 0, 0: 0, "positive": 1, "긍정": 1, "1": 1, 1: 1}
