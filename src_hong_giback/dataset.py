"""PyTorch Dataset for Korean BERT sentiment classification."""

import torch


class BertSentimentDataset(torch.utils.data.Dataset):
    """Convert Korean review text and labels into BERT input tensors."""

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

        encoded = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
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
