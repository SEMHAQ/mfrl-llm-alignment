# 只跑还没完成的部分（MFRL和候选已跑完，DPO/KTO旧结果可用）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\run_remaining.ps1

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "===== 补充实验 =====" -ForegroundColor Green
Write-Host "时间: $startTime" -ForegroundColor Cyan

# MFRL结果已就绪
Copy-Item outputs\mfrl_v3\eval_results.json results\mfrl_v3_eval.json -ErrorAction SilentlyContinue

# SFT baseline（之前beta=0.0没学到，用SFTTrainer重跑）
Write-Host "`n[1/4] Baseline: SFT (SFTTrainer)" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines sft

# 读取旧的DPO/KTO结果，合并SFT新结果
Write-Host "`n[2/4] 合并baseline结果" -ForegroundColor Yellow
$pyMerge = @'
import json, os
# 读旧结果
old = {}
if os.path.exists("results/baseline_eval.json"):
    with open("results/baseline_eval.json") as f:
        old = json.load(f)
# 读新SFT结果（从outputs目录）
sft_path = "outputs/mfrl_v3/sft/eval_results.json"
if os.path.exists(sft_path):
    with open(sft_path) as f:
        sft = json.load(f)
    old["sft"] = sft.get("rouge", sft)
# 保存合并结果
with open("results/baseline_eval.json", "w") as f:
    json.dump(old, f, ensure_ascii=False, indent=2)
print("Baselines:", list(old.keys()))
'@
$pyMerge | python -

# 消融实验
Write-Host "`n[3/4] Ablation: no_model" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_model
Copy-Item outputs\mfrl_v3\ablation_no_model\eval_results.json results\ablation_no_model_eval.json -ErrorAction SilentlyContinue

Write-Host "`n[4/4] Ablation: no_curriculum" -ForegroundColor Yellow
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
Write-Host "`n===== 推送 =====" -ForegroundColor Green
git add results/
git commit -m "results: SFT(SFTTrainer) + ablations, keep old DPO/KTO"
git push

Write-Host "`n===== 完成! =====" -ForegroundColor Green
