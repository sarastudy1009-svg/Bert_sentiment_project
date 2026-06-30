"""한국어 리뷰 문장을 BERT 입력 텐서로 변환하는 Dataset입니다."""

import torch


class BertSentimentDataset(torch.utils.data.Dataset):
    """문장과 정답 라벨을 BERT 입력 형식으로 변환합니다."""

    def __init__(self, reviews, labels, tokenizer, max_len: int = 128):
        self.reviews = list(reviews)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        review = str(self.reviews[index])
        label = int(self.labels[index])

        encoded = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
