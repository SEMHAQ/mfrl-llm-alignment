"""检查模型生成质量：采样5条看实际输出"""
import sys, json, yaml, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

with open("configs/train.yaml") as f:
    config = yaml.safe_load(f)

model_name = config["model"]["name"]
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, trust_remote_code=True
).to("cuda")
model = PeftModel.from_pretrained(base, "outputs/mfrl_v3/final")

with open("data/lcsts_train.json") as f:
    data = json.load(f)

for i in range(5):
    item = data[-200 + i]
    inp = item["input"]
    ref = item.get("reference") or item.get("summary", "")

    prompt = f"请为以下文本生成简洁准确的摘要：\n{inp}\n摘要："
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs, max_new_tokens=50, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print(f"SOURCE: {inp[:80]}...")
    print(f"REF:    {ref}")
    print(f"GEN:    {gen}")
    print()
