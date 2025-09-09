import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --------------------------
# Config
# --------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "/home/yesituser/yesitprojects/venv/isoTracker_requirement_AI/advanced_RAG/train_data_lora.json"
OUTPUT_DIR = "/home/yesituser/yesitprojects/venv/isoTracker_requirement_AI/advanced_RAG/lora-out"

BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4
MAX_SEQ_LEN = 512

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

# --------------------------
# Load model + tokenizer
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    device_map="auto",
)

# --------------------------
# LoRA config
# --------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # LLaMA/TinyLlama style
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# --------------------------
# Load dataset
# --------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# --------------------------
# Formatting function
# --------------------------
def formatting_func(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    messages = [
        {"role": "system", "content": "You are a senior requirements analyst. Copy names, dates, and numbers exactly. Ignore irrelevant bullets and briefly note conflicts if present."},
        {"role": "user", "content": f"QUESTION: {instruction}\n\nBULLETS:\n{input_text}"},
        {"role": "assistant", "content": output_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# --------------------------
# Trainer config (SFTConfig)
# --------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_length=MAX_SEQ_LEN,  # TRL uses max_length (not max_seq_length)
    packing=False,
    fp16=use_cuda,           # safe: only on CUDA
    bf16=use_bf16,           # safe: only if supported
)

# --------------------------
# Trainer
# --------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # TRL accepts tokenizer or processing_class
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=sft_config,
    peft_config=lora_config,
)

# --------------------------
# Train + Save
# --------------------------
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ LoRA fine-tuned model saved to {OUTPUT_DIR}")
