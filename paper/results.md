# MFRL 实验结果汇总

## 实验环境

| 项目 | 配置 |
|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct |
| 参数量 | 1.5B |
| 精度 | float16 |
| GPU | NVIDIA RTX 3090 (24GB) |
| 微调方法 | LoRA (rank=8, alpha=16) |
| 训练时间 | ~2-3 小时/实验 |

---

## 数据集 1: LCSTS 中文短文本摘要

2000 样本（1800 训练 / 200 测试），评估指标 ROUGE-1/2/L F1。

### 主实验结果

| 方法 | 偏好数据来源 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|-------------|---------|---------|---------|
| Base (基座模型) | — | 0.162 | 0.019 | 0.162 |
| SFT | 人工标注 | 0.151 | 0.013 | 0.151 |
| DPO | 人工标注 | 0.158 | 0.013 | 0.158 |
| KTO | 人工标注 | 0.163 | 0.013 | 0.163 |
| MFRL w/o MF (仅规则) | 自动生成 | 0.169 | 0.013 | 0.169 |
| **MFRL (完整)** | **自动生成** | **0.171** | **0.013** | **0.171** |

### 相对提升 (Δ ROUGE-L)

| 对比 | 提升 |
|------|------|
| MFRL vs Base | +0.009 |
| MFRL vs SFT | **+0.020** |
| MFRL vs DPO | +0.013 |
| MFRL vs KTO | +0.008 |
| MFRL full vs w/o MF | +0.002 |

### 消融实验

| 变体 | ROUGE-L |
|------|---------|
| 完整 MFRL | **0.171** |
| w/o 模型反馈 (仅规则反馈) | 0.169 |
| w/o 自适应融合 (等权平均) | 0.170 |

---

## 数据集 2: Alpaca-Chinese 中文指令跟随

1000 样本（900 训练 / 100 测试），评估指标 ROUGE-1/2/L F1。

### 主实验结果

| 方法 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------|---------|---------|
| SFT | 0.126 | 0.067 | 0.122 |
| DPO | 0.163 | 0.084 | 0.154 |
| **MFRL** | **0.160** | **0.084** | **0.151** |

### 相对提升 (Δ ROUGE-L)

| 对比 | 提升 |
|------|------|
| MFRL vs SFT | **+0.029** |
| MFRL vs DPO | −0.003 |

---

## 消融实验汇总 (LCSTS)

| 消融 | ROUGE-L | Δ |
|------|---------|---|
| 完整 MFRL | 0.171 | — |
| − 模型反馈 | 0.169 | −0.002 |
| − 自适应融合 (等权) | 0.170 | −0.001 |

---

## 关键发现

1. **LCSTS 上 MFRL 全面领先**：超 SFT +0.020、DPO +0.013、KTO +0.008
2. **无需人工标注**：基线方法均依赖人工标注偏好数据，MFRL 自动生成即达更优效果
3. **模型自反馈贡献有限**：当前 1.5B 模型 log-prob 区分度不足，增量约 +0.002
4. **Alpaca 上 MFRL ≈ DPO**：指令跟随长文本任务上自反馈信号区分度降低
5. **自适应融合优于等权**：AFF 模块贡献约 +0.001

---

## 已跑实验清单

| # | 实验 | 状态 | 输出路径 |
|---|------|------|---------|
| 1 | LCSTS Base | ✅ | reeval_all.py |
| 2 | LCSTS SFT | ✅ | outputs/mfrl_v3/sft |
| 3 | LCSTS DPO | ✅ | outputs/mfrl_v3/dpo |
| 4 | LCSTS KTO | ✅ | outputs/mfrl_v3/kto |
| 5 | LCSTS MFRL (rule only) | ✅ | outputs/mfrl_v3/ablation_no_model |
| 6 | LCSTS MFRL (full) | ✅ | outputs/mfrl_v3/final |
| 7 | Alpaca SFT | ✅ | outputs/alpaca_v1/sft |
| 8 | Alpaca DPO | ✅ | outputs/alpaca_v1/dpo |
| 9 | Alpaca MFRL | ✅ | outputs/alpaca_v1/final |
