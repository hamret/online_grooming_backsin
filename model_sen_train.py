# train_model.py
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

MODEL = "xlm-roberta-base"
TRAIN = "train.csv"
TEST = "test.csv"
OUTPUT = "./grooming-xlmroberta-sentence"

df_train = pd.read_csv(TRAIN)
df_test = pd.read_csv(TEST)

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

train_ds = Dataset.from_pandas(df_train)
test_ds = Dataset.from_pandas(df_test)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    # 텍스트가 None, float, list 등일 경우 대비
    clean_texts = []
    for t in batch["text"]:
        if t is None:
            clean_texts.append("")
        elif isinstance(t, str):
            clean_texts.append(t)
        else:
            clean_texts.append(str(t))  # 강제 문자열 변환

    return tokenizer(
        clean_texts,
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_ds = train_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

train_ds = train_ds.class_encode_column("label")
test_ds = test_ds.class_encode_column("label")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2
)

training_args = TrainingArguments(
    output_dir="./model_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,            # 🔥 LR 증가
    weight_decay=0.001,            # 🔥 더 안정적
    warmup_steps=6000,             # 🔥 warmup 추가
    num_train_epochs=2,
    logging_steps=200,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,  # ← 고친 부분
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

print("Training finished!")
