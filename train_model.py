import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("GPU device: NONE")

# 1) Cleaned CSV load
ds = load_dataset("csv", data_files={
    "train": "train_clean.csv",
    "test": "test_clean.csv",
})

# 2) Model / Tokenizer
MODEL = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 3) Safe preprocess
def preprocess(batch):
    txts = []
    for t in batch["text"]:
        if isinstance(t, str):
            txts.append(t)
        else:
            txts.append(str(t))
    return tokenizer(txts, padding="max_length", truncation=True, max_length=128)

ds = ds.map(preprocess, batched=True)
ds = ds.class_encode_column("label")
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4) Model load
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
).to(device)

# 5) Training settings
args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

trainer.train()
trainer.save_model("./model_output")

print("🎉 FINISHED TRAINING")
