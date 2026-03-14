import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 基础模型名称
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_TOKEN = "/////"

# ✅ 确保 prompt 在这里定义
question = "温卡希格和安祖提出的MP模型借助了哪些学科的理论"
prompt = f"用户：{question}\n助手："

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, 
    trust_remote_code=True,
    token=HF_TOKEN
)

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 加载模型
print(">>> 加载原始模型")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    # load_in_4bit=True,
    token=HF_TOKEN
)
base_model.eval()

# ✅ 使用 prompt 变量
print("\n=== 原始模型输出 ===")
print(generate(base_model, prompt))  # 这里使用了 prompt

# # 加载 LoRA
# print("\n>>> 加载 LoRA 微调模型")
# lora_model = PeftModel.from_pretrained(
#     base_model, 
#     "./qwen-lora-adapter"
# )
# lora_model.eval()

# print("\n=== LoRA 微调模型输出 ===")
# print(generate(lora_model, prompt))  # 这里也使用了 prompt
