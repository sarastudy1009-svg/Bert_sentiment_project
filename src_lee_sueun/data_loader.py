"""CSV 감성분석 데이터를 읽고 학습용 형태로 정리합니다."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src_lee_sueun.config import LABEL_TO_ID, SEED


def load_sentiment_csv(data_path: str | Path) -> pd.DataFrame:
    """review, sentiment 컬럼을 가진 CSV 파일을 정제해 반환합니다."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    try:
        dataset = pd.read_csv(data_path)
    except UnicodeDecodeError:
        dataset = pd.read_csv(data_path, encoding="cp949")

    dataset.columns = dataset.columns.str.strip()

    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(set(dataset.columns)):
        raise ValueError("CSV 파일에는 review, sentiment 컬럼이 반드시 있어야 합니다.")

    dataset = dataset.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    dataset["sentiment"] = dataset["sentiment"].map(lambda value: LABEL_TO_ID.get(value, LABEL_TO_ID.get(str(value).strip().lower())))

    if dataset["sentiment"].isna().any():
        raise ValueError("sentiment 컬럼은 positive/negative, 긍정/부정, 1/0 중 하나로 작성해야 합니다.")

    dataset["sentiment"] = dataset["sentiment"].astype(int)
    dataset["review"] = dataset["review"].astype(str)

    return dataset


def split_dataset(dataset: pd.DataFrame):
    """전체 데이터를 train, validation, test 세트로 분리합니다."""
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["sentiment"],
        random_state=SEED,
    )
    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.3,
        stratify=train_df["sentiment"],
        random_state=SEED,
    )

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)
