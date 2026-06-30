from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import DEFAULT_MODEL_NAME

def build_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer