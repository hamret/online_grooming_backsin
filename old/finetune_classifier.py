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

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_path = r"C:\Users\user\Desktop\KoAlpaca-Polyglot-5.8B\final_data\grooming_classifier_dataset"
output_dir = "llama3.1-grooming-fast"

# ---------------------------------------------------------
# 2. Ultra-fast QLoRA config (Windows 최적)
# ---------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,      # ✔ 더 빠름
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # ✔ Win/NVIDIA 최적
)

# ---------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------
print(f"Loading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",     # ✔ FlashAttention 경고 제거 + 더 빠름
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------
# 4. Load dataset
# ---------------------------------------------------------
print("Loading dataset...")
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ---------------------------------------------------------
# 5. LoRA (더 빠른 최소 구성)
# ---------------------------------------------------------
peft_config = LoraConfig(
    r=16,                     # ✔ 64 → 16 (속도 급상승)
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    # ✔ 비필수 모듈 제거 → 속도 증가
    target_modules=[
        "q_proj",
        "v_proj",
    ]
)

# ---------------------------------------------------------
# 6. TrainingArguments (초고속)
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,       # ✔ 4 → 8 (속도 2배 증가)
    gradient_accumulation_steps=1,       # ✔ 2 → 1 (더 빠름)
    gradient_checkpointing=False,        # ✔ Windows GPU에서 성능 떨어짐
    fp16=True,
    logging_steps=20,

    optim="adamw_torch",                 # ✔ paged_adamw보다 더 빠름
    lr_scheduler_type="constant",
    learning_rate=2e-4,

    save_strategy="no",                  # ✔ 저장 때문에 느려짐 → 끔
    eval_strategy="no",            # ✔ eval 때문에 느려짐 → 끔
    report_to=[],                        # ✔ wandb 제거
)

# ---------------------------------------------------------
# 7. formatting func (최소 버전)
# ---------------------------------------------------------
def formatting_func(example):
    return f"{example['text']}\nLabel: {example['label']}"

# ---------------------------------------------------------
# 8. Trainer
# ---------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,           # ✔ eval 끔
    peft_config=peft_config,
    formatting_func=formatting_func,
)

# ---------------------------------------------------------
# 9. Train
# ---------------------------------------------------------
print("🚀 Starting FAST fine-tuning...")
trainer.train()
print("🔥 Training complete!")

trainer.save_model(output_dir)
print(f"Model saved to: {output_dir}")
