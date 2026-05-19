# 通宵实验脚本 - 跑完自动推送
# 用法: powershell -ExecutionPolicy Bypass -File scripts\run_overnight.ps1

Write-Host "===== 通宵实验开始 =====" -ForegroundColor Green
Write-Host "时间: $(Get-Date)" -ForegroundColor Cyan

# Step 1: 更新配置（去掉过滤，5 epochs）
Write-Host "`n[配置] 去掉min_score_diff过滤，5 epochs" -ForegroundColor Yellow
python -c "
import yaml
with open('configs/train.yaml','r',encoding='utf-8') as f: c=yaml.safe_load(f)
c['feedback']['min_score_diff']=0.0
c['dpo']['num_epochs']=5
with open('configs/train.yaml','w',encoding='utf-8') as f: yaml.dump(c,f,allow_unicode=True)
print('Config updated: min_score_diff=0.0, num_epochs=5')
"

# Step 2: MFRL主实验（5 epochs，无过滤）
Write-Host "`n[1/6] MFRL (5 epochs, no filter)" -ForegroundColor Yellow
Remove-Item outputs\mfrl_v3\preference_data.json -ErrorAction SilentlyContinue
python scripts/run_experiment.py --config configs/train.yaml --skip_generation
Copy-Item outputs\mfrl_v3\eval_results.json results\mfrl_v3_eval.json -ErrorAction SilentlyContinue

# Step 3: Baselines（逐个跑，避免OOM）
Write-Host "`n[2/6] Baseline: SFT" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines sft

Write-Host "`n[3/6] Baseline: DPO" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines dpo

Write-Host "`n[4/6] Baseline: KTO" -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines kto
Copy-Item outputs\mfrl_v3\baseline_comparison.json results\baseline_eval.json -ErrorAction SilentlyContinue

# Step 4: 消融实验
Write-Host "`n[5/6] Ablation: no_model" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_model --skip_generation
Copy-Item outputs\mfrl_v3\ablation_no_model\eval_results.json results\ablation_no_model_eval.json -ErrorAction SilentlyContinue

Write-Host "`n[6/6] Ablation: no_curriculum" -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_curriculum --skip_generation
Copy-Item outputs\mfrl_v3\ablation_no_curriculum\eval_results.json results\ablation_no_curriculum_eval.json -ErrorAction SilentlyContinue

# 恢复配置
python -c "
import yaml
with open('configs/train.yaml','r',encoding='utf-8') as f: c=yaml.safe_load(f)
c['dpo']['learning_rate']=5e-6
c['dpo']['num_epochs']=2
c['feedback']['min_score_diff']=0.1
with open('configs/train.yaml','w',encoding='utf-8') as f: yaml.dump(c,f,allow_unicode=True)
"

# Step 5: 推送结果
Write-Host "`n===== 推送结果 =====" -ForegroundColor Green
git add results/
git commit -m "results: overnight experiments (MFRL + baselines + ablations)"
git push

Write-Host "`n===== 全部完成! =====" -ForegroundColor Green
Write-Host "时间: $(Get-Date)" -ForegroundColor Cyan
