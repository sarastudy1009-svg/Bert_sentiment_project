import argparse
from transformers import Trainer, TrainingArguments

from .config import MODEL_SAVE_DIR, MAX_LEN, DEFAULT_MODEL_NAME, DATA_PATH
from .data_loader import load_dataset
from .dataset import SentimentDataset
from .modeling import build_model_and_tokenizer
from .utils import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()

    set_seed()
    train_df, valid_df, _ = load_dataset(args.data_path)
    model, tokenizer = build_model_and_tokenizer(DEFAULT_MODEL_NAME)

    train_ds = SentimentDataset(train_df["document"], train_df["label"], tokenizer, MAX_LEN)
    valid_ds = SentimentDataset(valid_df["document"], valid_df["label"], tokenizer, MAX_LEN)

    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
    )
    trainer.train()

    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

if __name__ == "__main__":
    main()