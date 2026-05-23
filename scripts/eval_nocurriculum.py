"""快速评估 no_curriculum 消融实验"""
import sys, json, yaml, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.eval.evaluator import Evaluator

with open("configs/train.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tok = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"], dtype=torch.float16, trust_remote_code=True
).to("cuda")
model = PeftModel.from_pretrained(base, "outputs/mfrl_v3/ablation_no_curriculum/final")

with open("data/lcsts_train.json", encoding="utf-8") as f:
    data = json.load(f)

test_raw = data[-200:]
test_data = []
for item in test_raw:
    prompt = "请为以下文本生成简洁准确的摘要：\n" + item["input"] + "\n摘要："
    test_data.append({"prompt": prompt, "reference": item["reference"]})

evaluator = Evaluator(model, tok)
scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
print(scores)
