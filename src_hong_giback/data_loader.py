"""Load and split Korean sentiment datasets."""

from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src_hong_giback.config import LABEL_TO_ID, SEED


PathLike = Union[str, Path]


def load_sentiment_data(data_path: PathLike) -> pd.DataFrame:
    """Load either NSMC txt data or review/sentiment CSV data."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    if data_path.suffix.lower() == ".txt":
        dataset = pd.read_csv(data_path, sep="\t", encoding="utf-8")
    else:
        try:
            dataset = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            dataset = pd.read_csv(data_path, encoding="cp949")

    dataset.columns = dataset.columns.str.strip()

    if {"document", "label"}.issubset(dataset.columns):
        dataset = dataset.rename(columns={"document": "review", "label": "sentiment"})

    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError("데이터에는 review/sentiment 또는 document/label 컬럼이 있어야 합니다.")

    dataset = dataset.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    dataset["review"] = dataset["review"].astype(str)
    dataset["sentiment"] = dataset["sentiment"].map(_to_label_id)

    if dataset["sentiment"].isna().any():
        raise ValueError("sentiment 또는 label 값은 positive/negative, 긍정/부정, 1/0 중 하나여야 합니다.")

    dataset["sentiment"] = dataset["sentiment"].astype(int)
    return dataset[["review", "sentiment"]]


def load_sentiment_csv(data_path: PathLike) -> pd.DataFrame:
    """Backward-compatible loader name used by the training script."""
    return load_sentiment_data(data_path)


def split_dataset(dataset: pd.DataFrame):
    """Split data into train, validation, and test DataFrames."""
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


def _to_label_id(value):
    if value in LABEL_TO_ID:
        return LABEL_TO_ID[value]

    normalized = str(value).strip().lower()
    return LABEL_TO_ID.get(normalized)
