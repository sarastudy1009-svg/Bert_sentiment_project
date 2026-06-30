"""Create and configure a BERT sequence-classification model."""

from transformers import BertForSequenceClassification


def create_model(model_name: str, num_labels: int = 2) -> BertForSequenceClassification:
    """Load a pretrained BERT classifier from Hugging Face or a local path."""
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def apply_fine_tuning_strategy(
    model: BertForSequenceClassification,
    strategy: int = 3,
) -> BertForSequenceClassification:
    """Apply the same transfer-learning strategies used in the notebook."""
    for param in model.parameters():
        param.requires_grad = True

    last_encoder_layer_name = "encoder.layer.11"

    if strategy == 1:
        for _, param in model.bert.named_parameters():
            param.requires_grad = False
    elif strategy == 2:
        for name, param in model.bert.named_parameters():
            if not name.startswith("pooler"):
                param.requires_grad = False
    elif strategy == 3:
        for name, param in model.bert.named_parameters():
            if (not name.startswith("pooler")) and (last_encoder_layer_name not in name):
                param.requires_grad = False
    else:
        raise ValueError("strategy는 1, 2, 3 중 하나여야 합니다.")

    return model


def count_trainable_parameters(model: BertForSequenceClassification) -> dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
