from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

MODEL_DIR = "./model_grooming"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

app = FastAPI()

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Input):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]

    return {
        "prob_non_grooming": round(probs[0], 4),
        "prob_grooming": round(probs[1], 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
