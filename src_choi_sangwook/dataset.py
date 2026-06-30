"""PyTorch Dataset for Korean BERT movie-review sentiment classification."""

import torch


class BertMovieReviewDataset(torch.utils.data.Dataset):
    """Convert review text and labels into BERT input tensors."""

    def __init__(self, reviews, sentiments, tokenizer, max_len: int = 128):
        self.reviews = list(reviews)
        self.sentiments = list(sentiments)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        review = str(self.reviews[index])
        sentiment = int(self.sentiments[index])

        encoded = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(sentiment, dtype=torch.long),
        }


BertSentimentDataset = BertMovieReviewDataset
