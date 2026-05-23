# Overnight: LoRA rank sensitivity
# Powershell -ExecutionPolicy Bypass -File scripts\run_overnight.ps1

Write-Host "=== Overnight: Rank Sensitivity ===" -ForegroundColor Green
Write-Host "Start: $(Get-Date)"

# Run rank=4 ablation using existing preference data
Write-Host "`n[1/2] Rank=4 training..." -ForegroundColor Cyan
python scripts/quick_ablate_rank.py 4
if ($LASTEXITCODE -ne 0) { Write-Host "Rank=4 FAILED" -ForegroundColor Red }

# Run rank=16 ablation
Write-Host "`n[2/2] Rank=16 training..." -ForegroundColor Cyan
python scripts/quick_ablate_rank.py 16
if ($LASTEXITCODE -ne 0) { Write-Host "Rank=16 FAILED" -ForegroundColor Red }

# Evaluate both
Write-Host "`nEvaluating..." -ForegroundColor Cyan
python -c @'
import json, yaml, torch, os, sys
sys.path.insert(0, '.')
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
td = [{"prompt": "请为以下文本生成简洁准确的摘要：\n"+item["input"]+"\n摘要：",
       "reference": item["reference"]} for item in test]

os.makedirs("results", exist_ok=True)
for rank in [4, 8, 16]:
    path = f"outputs/ablation_rank{rank}/final"
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=torch.float16, trust_remote_code=True).to("cuda")
        model = PeftModel.from_pretrained(base, path)
        evaluator = Evaluator(model, tok)
        scores = evaluator.evaluate_rouge(td, reference_key="reference")
        print(f"rank={rank}: ROUGE-L={scores['rougeL']:.4f}")
        with open(f"results/rank{rank}_eval.json", "w") as f:
            json.dump(scores, f)
        del model, base
        torch.cuda.empty_cache()
print("Done")
'@

Write-Host "`n=== Complete: $(GetDate) ===" -ForegroundColor Green
Write-Host "Run: git add -A && git commit -m 'results: rank sensitivity' && git push"
