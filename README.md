# MFRL: Multi-form Feedback Reinforced Learning for LLM Fine-tuning

基于多形式反馈融合的大模型直接偏好优化方法

## 项目概述

本项目实现了MFRL框架，用于电子学报论文《基于多形式反馈与强化学习的大模型微调方法研究》的实验。

### 核心创新

1. **多源反馈收集（MFC）**：从规则指标（ROUGE）和轻量模型两个维度获取异构反馈信号
2. **自适应反馈融合（AFF）**：通过注意力机制动态学习各反馈源权重
3. **课程学习策略**：按样本难度从易到难训练

### 为什么轻量

- 用DPO替代PPO，无需独立奖励模型
- LoRA微调Qwen2.5-1.5B，单卡3090可跑
- 反馈信号自动生成，无需人工标注
- 公开数据集，无需特殊数据

## 环境配置

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 运行MFRL主实验

```bash
python scripts/run_experiment.py --config configs/train.yaml
```

### 2. 运行Baseline对比

```bash
python scripts/run_baselines.py --config configs/train.yaml --data outputs/mfrl/preference_data.json --baselines sft dpo kto
```

### 3. 运行消融实验

```bash
# 去掉规则反馈
python scripts/run_experiment.py --config configs/train.yaml --mode ablation --ablation_type no_rule

# 去掉模型反馈
python scripts/run_experiment.py --config configs/train.yaml --mode ablation --ablation_type no_model

# 去掉课程学习
python scripts/run_experiment.py --config configs/train.yaml --mode ablation --ablation_type no_curriculum
```

## 项目结构

```
.
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── train.yaml              # 训练配置
│   └── eval.yaml               # 评估配置
├── src/
│   ├── feedback/               # 多形式反馈模块（核心创新）
│   │   ├── rule_feedback.py    # 规则反馈（ROUGE等）
│   │   ├── model_feedback.py   # 模型反馈（小模型评分）
│   │   └── feedback_fusion.py  # 自适应反馈融合
│   ├── trainer/                # 训练模块
│   │   ├── dpo_trainer.py      # DPO训练器 + LoRA
│   │   └── curriculum.py       # 课程学习调度
│   ├── data/                   # 数据处理
│   │   ├── dataset.py          # LCSTS / Alpaca-Chinese加载
│   │   └── preprocess.py       # 候选生成 + 偏好对构建
│   └── eval/                   # 评估模块
│       ├── metrics.py          # ROUGE + Win Rate
│       └── evaluator.py        # 评估器
├── scripts/
│   ├── run_experiment.py       # 主实验 + 消融实验
│   └── run_baselines.py        # SFT / DPO / KTO baseline
├── paper/
│   └── main.tex                # 电子学报 LaTeX模板
├── data/                       # 数据集（gitignore，实验机下载）
├── outputs/                    # 模型输出（gitignore）
└── journal_to_refer/           # 参考论文（gitignore）
```

## 预计实验时间（单卡3090）

| 实验 | 预计时间 |
|------|---------|
| 数据预处理（生成候选） | 1-2小时 |
| MFRL训练 | 2-3小时 |
| SFT Baseline | 1小时 |
| DPO Baseline | 1.5小时 |
| KTO Baseline | 1.5小时 |
| 消融实验（3组） | 6-9小时 |
| 评估 | 0.5小时 |
| **总计** | **约13-17小时** |

## 论文

LaTeX模板位于 `paper/main.tex`，按照电子学报格式编排。
实验完成后，将结果填入论文中的表格即可。
