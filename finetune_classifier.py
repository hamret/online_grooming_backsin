import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_from_disk
import os

# -------------------------------------------------------------------------
# 1. 설정
# -------------------------------------------------------------------------

model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_path = r"C:\Users\user\Desktop\KoAlpaca-Polyglot-5.8B\final_data\grooming_classifier_dataset"
output_dir = "llama-3.1-instruct-grooming-classifier"

# -------------------------------------------------------------------------
# 2. 4-bit QLoRA 설정
# -------------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# -------------------------------------------------------------------------
# 3. 모델/토크나이저 로드
# -------------------------------------------------------------------------

print(f"Loading base model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# -------------------------------------------------------------------------
# 4. 데이터셋 로드
# -------------------------------------------------------------------------

print(f"Loading dataset from: {dataset_path}")
dataset = load_from_disk(dataset_path)

print("Dataset loaded successfully:")
print(dataset)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# -------------------------------------------------------------------------
# 5. LoRA 설정
# -------------------------------------------------------------------------

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)

# -------------------------------------------------------------------------
# 6. TrainingArguments 설정
# -------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",

    report_to=[],   # ← wandb/logging 모두 제거
)


# -------------------------------------------------------------------------
# 7. TRL 0.25.1 SFTTrainer용 formatting_func 생성
# -------------------------------------------------------------------------

def formatting_func(example):
    return f"""### Input:
{example['text']}

### Label:
{example['label']}
"""

# -------------------------------------------------------------------------
# 8. SFTTrainer 설정 (processing_class 사용 필수)
# -------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
)

# -------------------------------------------------------------------------
# 9. 학습 시작
# -------------------------------------------------------------------------

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning finished!")

# -------------------------------------------------------------------------
# 10. 모델 저장
# -------------------------------------------------------------------------

trainer.save_model(output_dir)
print(f"Model saved to: {output_dir}")
print("All done!")
