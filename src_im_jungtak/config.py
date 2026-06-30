from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 한국어 BERT 사전학습 모델 (KLUE-BERT)
DEFAULT_MODEL_NAME = "klue/bert-base"

# ratings_sample.txt 형식: id \t document \t label  (label: 0=부정, 1=긍정)
DATA_PATH = BASE_DIR / "data" / "ratings_sample.txt"
MODEL_SAVE_DIR = BASE_DIR / "models" / "bert_sentiment_im_jungtak"

MAX_LEN = 128
LABEL2ID = {0: 0, 1: 1}   # label 컬럼이 이미 0/1 숫자이므로 그대로 사용
ID2LABEL = {0: "negative", 1: "positive"}