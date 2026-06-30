import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(txt_path: str):
    # ratings_sample.txt: id \t document \t label
    df = pd.read_csv(txt_path, sep="\t")
    df = df.dropna(subset=["document"])               # 결측 리뷰 제거
    df = df[df["document"].str.strip() != ""]           # 공백 리뷰 제거
    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )
    return train_df, valid_df, test_df