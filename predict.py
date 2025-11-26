import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "model"   # 학습된 모델 폴더

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_grooming(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]

    nongroom, groom = probs.tolist()

    return {
        "grooming_probability": round(groom, 4),
        "nongrooming_probability": round(nongroom, 4),
        "label": int(groom > 0.5)
    }
