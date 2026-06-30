"""한국어 감성분석 모델 생성과 fine-tuning 범위 설정입니다."""

from transformers import AutoModelForSequenceClassification


def create_model(model_name: str, num_labels: int = 2) -> AutoModelForSequenceClassification:
    """Hugging Face Hub 또는 로컬 경로에서 문장 분류 모델을 불러옵니다."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def apply_fine_tuning_strategy(model: AutoModelForSequenceClassification, strategy: int = 3) -> AutoModelForSequenceClassification:
    """전략 값에 따라 backbone 모델의 학습 가능 범위를 조정합니다."""
    for param in model.parameters():
        param.requires_grad = True

    backbone = getattr(model, "bert", None)
    if backbone is None:
        return model

    last_encoder_layer_name = "encoder.layer.11"

    if strategy == 1:
        for param in backbone.parameters():
            param.requires_grad = False
    elif strategy == 2:
        for name, param in backbone.named_parameters():
            if not name.startswith("pooler"):
                param.requires_grad = False
    elif strategy == 3:
        for name, param in backbone.named_parameters():
            if (not name.startswith("pooler")) and (last_encoder_layer_name not in name):
                param.requires_grad = False
    else:
        raise ValueError("strategy는 1, 2, 3 중 하나여야 합니다.")

    return model


def count_trainable_parameters(model: AutoModelForSequenceClassification) -> dict[str, int]:
    """전체 파라미터 수와 학습 가능한 파라미터 수를 계산합니다."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
