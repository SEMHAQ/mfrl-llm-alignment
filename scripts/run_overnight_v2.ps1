# 通宵实验脚本 v2 - 多温度候选 + 修复SFT
# 用法: powershell -ExecutionPolicy Bypass -File scripts\run_overnight_v2.ps1

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "===== 通宵实验 v2 开始 =====" -ForegroundColor Green
Write-Host "时间: $startTime" -ForegroundColor Cyan
Write-Host "预计: 4-5小时" -ForegroundColor Cyan

# Step 1: 更新配置
Write-Host "`n[配置] 多温度生成, min_score_diff=0.0, 3 epochs" -ForegroundColor Yellow
python -c "
import yaml
with open('configs/train.yaml','r',encoding='utf-8') as f: c=yaml.safe_load(f)
c['feedback']['min_score_diff']=0.0
c['dpo']['num_epochs']=3
c['generation']['multi_temperature']=True
with open('configs/train.yaml','w',encoding='utf-8') as f: yaml.dump(c,f,allow_unicode=True)
print('Config: multi_temperature=True, min_score_diff=0.0, epochs=3')
"

# Step 2: 生成多温度候选（最慢的部分）
Write-Host "`n[1/7] 生成多温度候选 (4 temps x 4 candidates = 16/input)" -ForegroundColor Yellow
python scripts/generate_candidates.py --config configs/train.yaml --batch_size 64

# Step 3: MFRL主实验
Write-Host "`n[2/7] MFRL (多温度候选, 3 epochs)" -ForegroundColor Yellow
Remove-Item outputs\mfrl_v3\preference_data.json -ErrorAction SilentlyContinue
python scripts/run_experiment.py --config configs/train.yaml
Copy-Item outputs\mfrl_v3\eval_results.json results\mfrl_v3_eval.json -ErrorAction SilentlyContinue

# Step 4: Baselines（逐个跑，避免OOM）
Write-Host "`n[3/7] Baseline: SFT" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines sft

Write-Host "`n[4/7] Baseline: DPO" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines dpo

Write-Host "`n[5/7] Baseline: KTO" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines kto
Copy-Item outputs\mfrl_v3\baseline_comparison.json results\baseline_eval.json -ErrorAction SilentlyContinue

# Step 5: 消融实验
Write-Host "`n[6/7] Ablation: no_model" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_model
Copy-Item outputs\mfrl_v3\ablation_no_model\eval_results.json results\ablation_no_model_eval.json -ErrorAction SilentlyContinue

Write-Host "`n[7/7] Ablation: no_curriculum" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_curriculum
Copy-Item outputs\mfrl_v3\ablation_no_curriculum\eval_results.json results\ablation_no_curriculum_eval.json -ErrorAction SilentlyContinue

# 恢复配置
python -c "
import yaml
with open('configs/train.yaml','r',encoding='utf-8') as f: c=yaml.safe_load(f)
c['feedback']['min_score_diff']=0.1
c['dpo']['num_epochs']=2
c['generation']['multi_temperature']=False
with open('configs/train.yaml','w',encoding='utf-8') as f: yaml.dump(c,f,allow_unicode=True)
print('Config restored')
"

# Step 6: 打印结果汇总
Write-Host "`n===== 结果汇总 =====" -ForegroundColor Green
python -c "
import json, os
results = {}
for name, path in [
    ('MFRL', 'results/mfrl_v3_eval.json'),
    ('SFT', 'results/baseline_eval.json'),
]:
    if os.path.exists(path):
        with open(path) as f: data = json.load(f)
        if 'rouge' in data:
            results[name] = data['rouge']
        else:
            results.update(data)
for name in ['SFT', 'DPO', 'KTO']:
    if name in results:
        pass  # already loaded
if os.path.exists('results/baseline_eval.json'):
    with open('results/baseline_eval.json') as f:
        bl = json.load(f)
        results.update(bl)
for name, path in [
    ('no_model', 'results/ablation_no_model_eval.json'),
    ('no_curriculum', 'results/ablation_no_curriculum_eval.json'),
]:
    if os.path.exists(path):
        with open(path) as f: data = json.load(f)
        if 'rouge' in data:
            results[name] = data['rouge']
print(f\"{'Method':<20} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}\")
print('-'*56)
for name, scores in results.items():
    print(f\"{name:<20} {scores['rouge1']:<12.4f} {scores['rouge2']:<12.4f} {scores['rougeL']:<12.4f}\")
"

# Step 7: 推送结果
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n===== 推送结果 =====" -ForegroundColor Green
Write-Host "总耗时: $duration" -ForegroundColor Cyan
git add results/
git commit -m "results: multi-temp MFRL + baselines + ablations (v2)"
git push

Write-Host "`n===== 全部完成! =====" -ForegroundColor Green
Write-Host "结束时间: $(Get-Date)" -ForegroundColor Cyan
