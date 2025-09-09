from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 1. Load dataset
dataset = load_dataset("json", data_files="sample_data.json")

# 2. Choose base model (small one for demo)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def format_example(example):
    # Turn each dataset entry into a training prompt
    prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer:"
    return {"input_text": prompt, "labels": example["output"]}

dataset = dataset.map(format_example)

# 3. Tokenize data
def tokenize(batch):
    tokens = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. Load base model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# 5. Configure LoRA
lora_config = LoraConfig(
    r=8,              # rank (small adapter size)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # parts of model to tune
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# 8. Train
trainer.train()

# 9. Save fine-tuned model
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")
