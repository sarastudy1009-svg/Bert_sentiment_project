"""Load and split Korean movie-review sentiment data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src_choi_sangwook.config import LABEL_TO_ID, SEED


def _normalize_label(value) -> int | None:
    if value in LABEL_TO_ID:
        return LABEL_TO_ID[value]
    return LABEL_TO_ID.get(str(value).strip().lower())


def load_sentiment_csv(data_path: str | Path) -> pd.DataFrame:
    """Read ratings_train.txt and return id/review/sentiment columns."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    dataset = pd.read_table(data_path)
    if len(dataset.columns) != 3:
        raise ValueError("데이터 파일은 id, review, sentiment 세 개의 컬럼을 가져야 합니다.")

    dataset.columns = ["id", "review", "sentiment"]
    dataset = dataset.dropna(subset=["review", "sentiment"]).reset_index(drop=True)

    dataset["sentiment"] = dataset["sentiment"].map(_normalize_label)
    if dataset["sentiment"].isna().any():
        raise ValueError("sentiment 컬럼은 0/1, positive/negative, 긍정/부정 중 하나여야 합니다.")

    dataset["review"] = dataset["review"].astype(str)
    dataset["sentiment"] = dataset["sentiment"].astype(int)
    return dataset


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, validation, and test sets like the notebook."""
    train_set, test_set = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["sentiment"],
        random_state=SEED,
    )

    train_set, valid_set = train_test_split(
        train_set,
        test_size=0.3,
        stratify=train_set["sentiment"],
        random_state=SEED,
    )

    return (
        train_set.reset_index(drop=True),
        valid_set.reset_index(drop=True),
        test_set.reset_index(drop=True),
    )
