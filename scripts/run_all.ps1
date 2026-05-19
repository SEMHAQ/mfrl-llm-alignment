# 全量实验脚本（挂机跑）
# 在实验机上执行: powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1

Write-Host "===== Starting all experiments =====" -ForegroundColor Green

# Baselines
Write-Host "`n[1/3] Running baselines..." -ForegroundColor Yellow
python scripts/run_baselines.py --config configs/train.yaml --baselines sft dpo kto

# 消融实验
Write-Host "`n[2/3] Running ablation: no_model..." -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_model --skip_generation

Write-Host "`n[3/3] Running ablation: no_curriculum..." -ForegroundColor Yellow
python scripts/run_experiment.py --mode ablation --ablation_type no_curriculum --skip_generation

# 收集结果
Write-Host "`n===== Collecting results =====" -ForegroundColor Green
copy outputs\mfrl_v3\baseline_comparison.json results\baseline_eval.json -ErrorAction SilentlyContinue
copy outputs\mfrl_v3\ablation_no_model\eval_results.json results\ablation_no_model_eval.json -ErrorAction SilentlyContinue
copy outputs\mfrl_v3\ablation_no_curriculum\eval_results.json results\ablation_no_curriculum_eval.json -ErrorAction SilentlyContinue

# 推送
Write-Host "`n===== Pushing results =====" -ForegroundColor Green
git add results/
git commit -m "results: baselines + ablations (correct evaluation)"
git push

Write-Host "`n===== All done! =====" -ForegroundColor Green
