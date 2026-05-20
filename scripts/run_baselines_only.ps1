# 只跑baselines + ablations（MFRL和候选已跑完）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\run_baselines_only.ps1

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "===== Baselines + Ablations =====" -ForegroundColor Green
Write-Host "时间: $startTime" -ForegroundColor Cyan

# 复制MFRL结果
Copy-Item outputs\mfrl_v3\eval_results.json results\mfrl_v3_eval.json -ErrorAction SilentlyContinue

# Baselines
Write-Host "`n[1/5] Baseline: SFT" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines sft

Write-Host "`n[2/5] Baseline: DPO" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines dpo

Write-Host "`n[3/5] Baseline: KTO" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines kto
Copy-Item outputs\mfrl_v3\baseline_comparison.json results\baseline_eval.json -ErrorAction SilentlyContinue

# Ablations
Write-Host "`n[4/5] Ablation: no_model" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_model
Copy-Item outputs\mfrl_v3\ablation_no_model\eval_results.json results\ablation_no_model_eval.json -ErrorAction SilentlyContinue

Write-Host "`n[5/5] Ablation: no_curriculum" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_curriculum
Copy-Item outputs\mfrl_v3\ablation_no_curriculum\eval_results.json results\ablation_no_curriculum_eval.json -ErrorAction SilentlyContinue

# 打印结果
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n===== 结果汇总 =====" -ForegroundColor Green
Write-Host "总耗时: $duration" -ForegroundColor Cyan

$pySummary = @'
import json, os
results = {}
files = [
    ("MFRL", "results/mfrl_v3_eval.json"),
    ("no_model", "results/ablation_no_model_eval.json"),
    ("no_curriculum", "results/ablation_no_curriculum_eval.json"),
]
for name, path in files:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if "rouge" in data:
            results[name] = data["rouge"]
if os.path.exists("results/baseline_eval.json"):
    with open("results/baseline_eval.json") as f:
        results.update(json.load(f))
header = "{:<20} {:<12} {:<12} {:<12}".format("Method", "ROUGE-1", "ROUGE-2", "ROUGE-L")
print(header)
print("-" * 56)
for name, scores in results.items():
    line = "{:<20} {:.4f}       {:.4f}       {:.4f}".format(
        name, scores["rouge1"], scores["rouge2"], scores["rougeL"])
    print(line)
'@
$pySummary | python -

# 推送
Write-Host "`n===== 推送结果 =====" -ForegroundColor Green
git add results/
git commit -m "results: baselines + ablations (fixed SFT + single GPU)"
git push

Write-Host "`n===== 全部完成! =====" -ForegroundColor Green
