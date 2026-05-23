# Overnight: LoRA rank + DPO beta sensitivity
# 重用已有偏好数据，只改训练参数
# 总共约 5-6 小时
# Powershell -ExecutionPolicy Bypass -File scripts\run_overnight.ps1

Write-Host "=== Overnight Experiments ===" -ForegroundColor Green
Write-Host "Start: $(Get-Date)"

# Experiment 1+2: Rank sensitivity (rank=4, rank=16)
Write-Host "`n[1/5] Rank=4" -ForegroundColor Cyan
python scripts/quick_ablate_rank.py 4
if ($LASTEXITCODE -ne 0) { Write-Host "Rank=4 FAILED" -ForegroundColor Red }

Write-Host "`n[2/5] Rank=16" -ForegroundColor Cyan
python scripts/quick_ablate_rank.py 16
if ($LASTEXITCODE -ne 0) { Write-Host "Rank=16 FAILED" -ForegroundColor Red }

# Experiment 3+4+5: Beta sensitivity (beta=0.05, 0.2, 0.5)
Write-Host "`n[3/5] Beta=0.05" -ForegroundColor Cyan
python scripts/quick_ablate_beta.py 0.05
if ($LASTEXITCODE -ne 0) { Write-Host "Beta=0.05 FAILED" -ForegroundColor Red }

Write-Host "`n[4/5] Beta=0.2" -ForegroundColor Cyan
python scripts/quick_ablate_beta.py 0.2
if ($LASTEXITCODE -ne 0) { Write-Host "Beta=0.2 FAILED" -ForegroundColor Red }

Write-Host "`n[5/5] Beta=0.5" -ForegroundColor Cyan
python scripts/quick_ablate_beta.py 0.5
if ($LASTEXITCODE -ne 0) { Write-Host "Beta=0.5 FAILED" -ForegroundColor Red }

# Evaluate all
Write-Host "`n=== Evaluating all models ===" -ForegroundColor Cyan
python -c @'
import json, yaml, torch, os, sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.eval.evaluator import Evaluator

with open("configs/train.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
tok.pad_token = tok.eos_token

with open("data/lcsts_train.json", encoding="utf-8") as f:
    data = json.load(f)
test = data[-200:]
td = []
for item in test:
    prompt = "请为以下文本生成简洁准确的摘要：\n" + item["input"] + "\n摘要："
    td.append({"prompt": prompt, "reference": item["reference"]})

os.makedirs("results", exist_ok=True)
all_results = {}

for label, path in [
    ("rank4", "outputs/ablation_rank4/final"),
    ("rank16", "outputs/ablation_rank16/final"),
    ("beta0.05", "outputs/ablation_beta005/final"),
    ("beta0.2", "outputs/ablation_beta02/final"),
    ("beta0.5", "outputs/ablation_beta05/final"),
]:
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=torch.float16, trust_remote_code=True).to("cuda")
        model = PeftModel.from_pretrained(base, path)
        evaluator = Evaluator(model, tok)
        scores = evaluator.evaluate_rouge(td, reference_key="reference")
        print(f"{label}: ROUGE-L={scores['rougeL']:.4f}")
        all_results[label] = scores
        with open(f"results/{label}_eval.json", "w") as f:
            json.dump(scores, f)
        del model, base
        torch.cuda.empty_cache()

# Print summary table
print("\n=== Sensitivity Summary ===")
print("Rank sensitivity (base: rank=8, ROUGE-L=0.171):")
for k in all_results:
    if "rank" in k:
        print(f"  {k}: {all_results[k]['rougeL']:.4f}")
print("Beta sensitivity (base: beta=0.1, ROUGE-L=0.171):")
for k in all_results:
    if "beta" in k:
        print(f"  {k}: {all_results[k]['rougeL']:.4f}")

with open("results/sensitivity_summary.json", "w") as f:
    json.dump({k: v["rougeL"] for k, v in all_results.items()}, f, indent=2)
print("\nSaved results/sensitivity_summary.json")
'@

Write-Host "`n=== Complete: $(Get-Date) ===" -ForegroundColor Green
Write-Host "`nPush with:"
Write-Host "  git add -A"
Write-Host '  git commit -m \"results: rank + beta sensitivity\"'
Write-Host "  git push"
