"""CSV 감성분석 데이터를 읽고 BERT 학습에 맞는 형태로 정리하는 파일입니다."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src_ha_jungwon.config import LABEL_TO_ID, SEED


def load_sentiment_txt(data_path: str | Path) -> pd.DataFrame:
    """id, document, label 컬럼을 가진 탭 구분 텍스트 파일을 읽어 정제합니다."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {data_path}")

    # sep='\t'를 사용하여 탭으로 구분된 데이터를 읽습니다.
    # 이미지에 컬럼명이 있으므로 header=0을 사용합니다.
    try:
        dataset = pd.read_csv(data_path, sep='\t')
    except UnicodeDecodeError:
        dataset = pd.read_csv(data_path, sep='\t', encoding="cp949")

    # 이미지상의 컬럼명이 'id', 'document', 'label' 이므로 이 컬럼들을 사용합니다.
    # 모델 학습을 위해 'document'를 'review'로, 'label'을 'sentiment'로 이름을 변경합니다.
    dataset = dataset.rename(columns={"document": "review", "label": "sentiment"})

    # 필수 컬럼 확인
    if not {"review", "sentiment"}.issubset(set(dataset.columns)):
        raise ValueError("파일에 review(document)와 sentiment(label) 컬럼이 있어야 합니다.")

    # 결측치 제거 및 타입 변환
    dataset = dataset.dropna(subset=["review", "sentiment"]).reset_index(drop=True)[:5000]
    dataset["sentiment"] = dataset["sentiment"].astype(int)
    dataset["review"] = dataset["review"].astype(str)

    return dataset


def split_dataset(dataset: pd.DataFrame):
    """전체 데이터를 train, validation, test 세트로 분리합니다."""
    # 전체 데이터를 먼저 학습용 80%, 테스트용 20%로 나눕니다.
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["sentiment"],
        random_state=SEED,
    )

    # 학습용 데이터에서 다시 검증용 30%를 분리합니다.
    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.3,
        stratify=train_df["sentiment"],
        random_state=SEED,
    )

    # 인덱스를 0부터 다시 정리하여 Dataset 접근을 단순하게 만듭니다.
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)
