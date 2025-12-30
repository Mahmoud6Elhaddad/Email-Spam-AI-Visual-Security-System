import torch
import joblib
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent

BERT_MODEL_PATH = BASE_DIR / "transformer_model"
ML_MODEL_PATH = BASE_DIR / "ml_model.pkl"

LABEL_MAP = {0: "ham", 1: "spam"}

# Load BERT
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_PATH
).to(DEVICE)
bert_model.eval()

# Load ML pipeline (TF-IDF + SGD)
ml_pipeline = joblib.load(ML_MODEL_PATH)


def bert_predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return LABEL_MAP[pred], confidence


def ml_predict(text: str):
    prob = ml_pipeline.predict_proba([text])[0]
    pred = int(np.argmax(prob))
    confidence = float(prob[pred])

    return LABEL_MAP[pred], confidence
