"""Project paths and defaults for the Korean BERT sentiment package."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "bert-kor-base"

DEFAULT_DATA_PATH = DATA_DIR / "ratings_train.txt"
DEFAULT_MODEL_NAME = "kykim/bert-kor-base"

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
