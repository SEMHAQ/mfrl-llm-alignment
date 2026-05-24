"""评估所有敏感性实验模型"""
import json, yaml, torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.eval.evaluator import Evaluator

with open("configs/train.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
tok.pad_token = tok.eos_token

with open("data/lcsts_train.json", encoding="utf-8") as f:
    data = json.load(f)
td = []
for item in data[-200:]:
    prompt = "请为以下文本生成简洁准确的摘要：\n" + item["input"] + "\n摘要："
    td.append({"prompt": prompt, "reference": item["reference"]})

os.makedirs("results", exist_ok=True)
models = [
    ("rank4", "outputs/ablation_rank4/final"),
    ("rank16", "outputs/ablation_rank16/final"),
    ("beta005", "outputs/ablation_beta005/final"),
    ("beta02", "outputs/ablation_beta02/final"),
    ("beta05", "outputs/ablation_beta05/final"),
]

print("=== Sensitivity Evaluation ===")
for label, path in models:
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["name"], dtype=torch.float16, trust_remote_code=True
        ).to("cuda")
        model = PeftModel.from_pretrained(base, path)
        evaluator = Evaluator(model, tok)
        scores = evaluator.evaluate_rouge(td, reference_key="reference")
        print(f"  {label}: ROUGE-L={scores['rougeL']:.4f}")
        json.dump(scores, open(f"results/{label}_eval.json", "w"))
        del model, base
        torch.cuda.empty_cache()
    else:
        print(f"  {label}: NOT FOUND")

print("\nDone. Run: git add results/ && git commit && git push")
