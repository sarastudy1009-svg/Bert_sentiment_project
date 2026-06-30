"""Train and save the Korean BERT sentiment model."""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizerFast, Trainer, TrainingArguments

from src_choi_sangwook.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_NAME, MAX_LEN, MODEL_DIR, SEED
from src_choi_sangwook.data_loader import load_sentiment_csv, split_dataset
from src_choi_sangwook.dataset import BertMovieReviewDataset
from src_choi_sangwook.modeling import apply_fine_tuning_strategy, count_trainable_parameters, create_model
from src_choi_sangwook.utils import get_device, set_seed


def compute_metrics(pred) -> dict[str, float]:
    """Compute binary-classification metrics for Hugging Face Trainer."""
    labels = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(args: argparse.Namespace) -> None:
    """Run the full notebook-style train, validation, test, and save flow."""
    set_seed(SEED)
    device = get_device()

    dataset = load_sentiment_csv(args.data_path)
    train_df, valid_df, test_df = split_dataset(dataset)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_dataset = BertMovieReviewDataset(
        train_df["review"],
        train_df["sentiment"],
        tokenizer,
        max_len=args.max_len,
    )
    valid_dataset = BertMovieReviewDataset(
        valid_df["review"],
        valid_df["sentiment"],
        tokenizer,
        max_len=args.max_len,
    )
    test_dataset = BertMovieReviewDataset(
        test_df["review"],
        test_df["sentiment"],
        tokenizer,
        max_len=args.max_len,
    )

    model = create_model(args.model_name, num_labels=2)
    model = apply_fine_tuning_strategy(model, strategy=args.strategy)
    model.to(device)

    print("parameter info:", count_trainable_parameters(model))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_result = trainer.predict(test_dataset)
    print("test metrics:", test_result.metrics)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"model saved: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Korean BERT sentiment model")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to ratings_train.txt")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default=str(MODEL_DIR), help="Directory to save the trained model")
    parser.add_argument("--max_len", type=int, default=MAX_LEN, help="Maximum BERT input token length")
    parser.add_argument("--strategy", type=int, default=3, choices=[1, 2, 3], help="Fine-tuning strategy")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
