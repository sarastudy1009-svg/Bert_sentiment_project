"""Configuration for the Korean sentiment analysis package."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Separate directory from the English model so both predictors can coexist.
MODEL_DIR = BASE_DIR / "models" / "bert_korean_sentiment"
DEFAULT_DATA_PATH = DATA_DIR / "ratings_train.txt"

# Fine-tuned BERT model for Korean customer-review sentiment analysis.
DEFAULT_MODEL_NAME = "WhitePeak/bert-base-cased-Korean-sentiment"

MAX_LEN = 128
SEED = 42

ID_TO_LABEL = {0: "부정", 1: "긍정"}
LABEL_TO_ID = {
    "negative": 0,
    "부정": 0,
    "0": 0,
    0: 0,
    "positive": 1,
    "긍정": 1,
    "1": 1,
    1: 1,
}
