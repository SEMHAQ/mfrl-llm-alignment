# 基于多形式反馈融合的大语言模型直接偏好优化方法

## 摘要

针对大语言模型微调过程中偏好数据依赖人工标注、反馈信号来源单一的问题，提出基于多形式反馈融合的直接偏好优化方法（Multi-form Feedback Reinforced Learning, MFRL）。该方法包含三个核心模块：

- **MFC（多源反馈收集）**：从 ROUGE 规则指标和策略模型对数概率两个维度获取异构反馈信号
- **AFF（自适应反馈融合）**：通过注意力机制动态学习各反馈源权重
- **POF（偏好优化微调）**：基于融合偏好对进行 DPO 训练

在 Qwen2.5-1.5B-Instruct 基座模型、LCSTS 中文摘要和 Alpaca-Chinese 指令跟随两个数据集上的实验结果表明：MFRL 在 LCSTS 上 ROUGE-L 达到 0.171，较 SFT、DPO 和 KTO 基线分别提升 0.020、0.013 和 0.008；在 Alpaca-Chinese 上 ROUGE-L 达到 0.151，较 SFT 提升 0.029。消融实验进一步验证了各模块的有效性。

**关键词：** 大语言模型，直接偏好优化，多形式反馈，反馈融合，强化学习，LoRA

---

## 1 引言

大语言模型（Large Language Model, LLM）在自然语言处理领域取得了显著进展，但预训练模型的输出往往与人类偏好存在偏差，需要通过微调技术进行对齐。

基于人类反馈的强化学习（RLHF）是当前主流的对齐范式，通过训练奖励模型量化人类偏好，再利用 PPO 对策略模型进行优化。然而，RLHF 需要维护独立的奖励模型，计算开销大，且偏好数据依赖人工标注，成本高昂。

直接偏好优化（DPO）将奖励建模和策略优化统一为单阶段监督学习，避免了显式奖励模型的训练。KTO 进一步简化标注要求，仅需二元标签即可训练。然而，上述方法的偏好数据构建仍依赖人工提供或单一反馈源。

Self-Reward 和 Constitutional AI 等方法探索了不依赖人工标注的自动反馈机制，但单一反馈源可能存在偏差。多源信息融合在多个领域已被证明优于单一信号源，但在偏好学习中的应用仍处于探索阶段。

针对上述挑战，本文提出基于多形式反馈融合的直接偏好优化方法（MFRL）。主要贡献如下：

1. 提出多源反馈收集模块（MFC），融合规则反馈与模型自反馈，自动生成高质量偏好对，无需人工标注。
2. 设计自适应反馈融合模块（AFF），通过注意力机制动态学习各反馈源权重，实现异构信号的自适应融合。
3. 在 LCSTS 数据集上，MFRL 的 ROUGE-L 达到 0.171，显著优于 SFT（+0.020）、DPO（+0.013）和 KTO（+0.008）基线。

---

## 2 相关工作

### 2.1 偏好优化方法

RLHF 通过人类偏好数据训练奖励模型并使用 PPO 优化策略。DPO 将奖励函数参数化为策略模型的隐式形式，绕过了显式奖励建模。KTO 仅需二元标注即可直接优化。SimPO 去除参考模型进一步简化。然而，上述方法均依赖人工标注构建偏好数据。

### 2.2 自动反馈与参数高效微调

Self-Reward 和 RLAIF 利用模型自身或 AI 生成反馈信号。LoRA 通过低秩矩阵分解近似参数更新，显著降低微调显存需求，使消费级 GPU 上进行偏好优化成为可能。

---

## 3 MFRL 方法

### 3.1 问题定义与总体架构

给定输入 x 和参考 y*，策略模型 π_θ 生成 K 个候选输出 {y_1, ..., y_K}。MFRL 框架包含三个核心模块：

```
输入：(x, y*)
  ↓ 多温度采样 + 拒绝滤波
MFC：ROUGE 规则得分 + 模型 Log-Probability 自评分
  ↓
AFF：注意力权重融合 → 候选排序 → 偏好对 (x, y+, y-)
  ↓
POF：DPO Loss + LoRA (rank=8)
  ↓
输出：优化后模型 π_θ
```

### 3.2 多源反馈收集模块 (MFC)

**候选生成：** 多温度采样 T ∈ {0.5, 0.8, 1.2, 1.8}，每输入 K=4 候选。拒绝滤波去除"我无法回答"等模式。

**规则反馈：** 基于 ROUGE-1/2/L，加权求和：
```
s_RF = 0.3·R1 + 0.3·R2 + 0.4·RL
```

**模型自反馈：** 策略模型对数概率作为质量信号：
```
s_MF(y|x) = (1/|y|) Σ log P(y_t | x, y_<t)
```
归一化至 [0,1] 与规则反馈对齐。

### 3.3 自适应反馈融合模块 (AFF)

通过可学习注意力权重动态融合异构反馈：
```
w = softmax(W·[s_RF, s_MF]^T + b)
s_fused = w_RF·s_RF + w_MF·s_MF
```

按 s_fused 排序，最高分→正例 y+，最低分→负例 y-，构成偏好对 (x, y+, y-)。

### 3.4 偏好优化微调模块 (POF)

DPO 损失函数，LoRA 微调（rank=8, alpha=16, dropout=0.05），目标模块为所有线性层。

超参数：学习率 5e-6，batch_size=4，gradient_accumulation=4，有效 batch=16，3 epochs，β=0.1。

---

## 4 实验与分析

### 4.1 实验设置

- **基座模型：** Qwen2.5-1.5B-Instruct，1.5B 参数，float16 精度
- **数据集：** LCSTS（2000 样本）、Alpaca-Chinese（1000 样本）
- **基线：** Base、SFT、DPO、KTO、MFRL w/o MF
- **评估：** ROUGE-1/2/L F1
- **硬件：** 单卡 RTX 3090 (24GB)，训练约 2-3 小时

详见 [实验结果](results.md)

---

## 5 结论

本文提出 MFRL 框架，通过多源反馈收集、自适应反馈融合和偏好优化微调三个模块协同工作，实现了从异构反馈信号到高质量偏好对的自动构建与高效训练。在 LCSTS 和 Alpaca-Chinese 两个数据集上的实验验证了多形式反馈融合相对于单一反馈信号的优势。

未来工作：引入过程奖励模型等更丰富的反馈信号；在更大规模模型上验证；在线反馈迭代优化。

---

## 参考文献

[1] Brown T, et al. Language models are few-shot learners. NeurIPS 2020.
[2] Touvron H, et al. LLaMA: Open and efficient foundation language models. arXiv:2302.13971, 2023.
[3] Wei J, et al. Finetuned language models are zero-shot learners. ICLR 2022.
[4] Ouyang L, et al. Training language models to follow instructions with human feedback. NeurIPS 2022.
[5] Schulman J, et al. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.
[6] Rafailov R, et al. Direct preference optimization: Your language model is secretly a reward model. NeurIPS 2023.
[7] Ethayarajh K, et al. KTO: Model alignment as prospect theoretic optimization. arXiv:2402.01306, 2024.
[8] Yuan W Z, et al. Self-rewarding language models. ICML 2024.
[9] Bai Y T, et al. Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073, 2022.
[10] 何友, 王国宏, 关欣, 等. 多传感器信息融合及应用. 电子工业出版社, 2000.
[11] Meng Y, et al. SimPO: Simple preference optimization with a reference-free reward. NeurIPS 2024.
[12] Hu E J, et al. LoRA: Low-rank adaptation of large language models. ICLR 2022.
[13] Lin C Y. ROUGE: A package for automatic evaluation of summaries. ACL Workshop 2004.
[14] Yang A, et al. Qwen2.5 technical report. arXiv:2412.15115, 2024.
[15] Hu B T, et al. LCSTS: A large scale Chinese short text summarization dataset. EMNLP 2015.
