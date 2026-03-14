import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import os

# === Step 1: 加载和预处理数据 ===
def load_and_prepare_data(json_path):
    dataset = Dataset.from_json(json_path)
    
    def format(example):
        prompt = f"用户：{example['input']}\n助手：{example['output']}"
        return {"text": prompt}
    
    dataset = dataset.map(format)
    return dataset

# === Step 2: 加载模型和Tokenizer ===
def load_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True
    )
    return model, tokenizer

# === Step 3: 应用LoRA配置 ===
def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# === Step 4: Tokenize数据 ===
def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
    return dataset.map(tokenize_fn, remove_columns=["text"])

# === Step 5: 启动训练 ===
def train(model, tokenizer, tokenized_dataset):
    args = TrainingArguments(
        output_dir="./qwen-lora-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=50,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained("./qwen-lora-adapter")
    tokenizer.save_pretrained("./qwen-lora-adapter")

# === 主程序入口 ===
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    json_path = "train.json"

    dataset = load_and_prepare_data(json_path)
    model, tokenizer = load_model_tokenizer(model_name)
    model = apply_lora(model)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    train(model, tokenizer, tokenized_dataset)
