"""Model helpers for Korean sentiment classification."""

from transformers import AutoModelForSequenceClassification


def create_model(model_name: str, num_labels: int = 2) -> AutoModelForSequenceClassification:
    """Load a sequence classification model from Hugging Face Hub or a local path."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def apply_fine_tuning_strategy(model: AutoModelForSequenceClassification, strategy: int = 3) -> AutoModelForSequenceClassification:
    """Freeze part of the transformer encoder according to the selected strategy."""
    base_model = getattr(model, model.base_model_prefix)

    for param in model.parameters():
        param.requires_grad = True

    if strategy == 1:
        for param in base_model.parameters():
            param.requires_grad = False
    elif strategy == 2:
        for name, param in base_model.named_parameters():
            if not name.startswith("pooler"):
                param.requires_grad = False
    elif strategy == 3:
        encoder_marker = "encoder.layer.11"
        for name, param in base_model.named_parameters():
            if not name.startswith("pooler") and encoder_marker not in name:
                param.requires_grad = False
    else:
        raise ValueError("strategy는 1, 2, 3 중 하나여야 합니다.")

    return model


def count_trainable_parameters(model: AutoModelForSequenceClassification) -> dict[str, int]:
    """Count all model parameters and trainable parameters."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
