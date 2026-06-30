"""한국어 BERT 감성분석 모델을 학습하고 저장하는 실행 파일입니다."""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src_lee_sueun.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_NAME, MAX_LEN, MODEL_DIR, SEED
from src_lee_sueun.data_loader import load_sentiment_csv, split_dataset
from src_lee_sueun.dataset import BertSentimentDataset
from src_lee_sueun.modeling import apply_fine_tuning_strategy, count_trainable_parameters, create_model
from src_lee_sueun.utils import get_device, set_seed


def compute_metrics(pred) -> dict[str, float]:
    """Trainer 평가 결과에서 정확도, 정밀도, 재현율, F1-score를 계산합니다."""
    labels = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(args: argparse.Namespace) -> None:
    """명령행 인자를 받아 데이터 로드부터 모델 저장까지 전체 학습을 수행합니다."""
    set_seed(SEED)
    device = get_device()

    dataset = load_sentiment_csv(args.data_path)
    train_df, valid_df, test_df = split_dataset(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = BertSentimentDataset(train_df["review"], train_df["sentiment"], tokenizer, max_len=args.max_len)
    valid_dataset = BertSentimentDataset(valid_df["review"], valid_df["sentiment"], tokenizer, max_len=args.max_len)
    test_dataset = BertSentimentDataset(test_df["review"], test_df["sentiment"], tokenizer, max_len=args.max_len)

    model = create_model(args.model_name, num_labels=2)
    model = apply_fine_tuning_strategy(model, strategy=args.strategy)
    model.to(device)

    print("파라미터 정보:", count_trainable_parameters(model))

    output_dir = str(args.output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
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
    print("테스트 평가 결과:", test_result.metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"모델 저장 완료: {output_dir}")


def parse_args() -> argparse.Namespace:
    """명령행 인자를 정의하고 파싱합니다."""
    parser = argparse.ArgumentParser(description="한국어 BERT 감성분석 모델 학습 스크립트")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH), help="review,sentiment 컬럼을 가진 CSV 파일 경로")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face 모델명")
    parser.add_argument("--output_dir", type=str, default=str(MODEL_DIR), help="학습 모델 저장 폴더")
    parser.add_argument("--max_len", type=int, default=MAX_LEN, help="BERT 최대 입력 토큰 길이")
    parser.add_argument("--strategy", type=int, default=3, choices=[1, 2, 3], help="Fine-tuning 전략 번호")
    parser.add_argument("--epochs", type=int, default=1, help="학습 반복 횟수")
    parser.add_argument("--train_batch_size", type=int, default=8, help="학습 배치 크기")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="평가 배치 크기")
    parser.add_argument("--warmup_steps", type=int, default=0, help="학습률 warmup step 수")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="가중치 감쇠 정규화 강도")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
