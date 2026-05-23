# 论文大纲：基于多形式反馈融合的大模型直接偏好优化方法

## 1 引言（约1.5页）
### 1.1 研究背景
- 大语言模型（LLM）微调技术的发展：从 SFT 到 RLHF
- RLHF 的局限性：需要独立奖励模型、PPO 训练不稳定、计算开销大
- DPO 的提出与优势：无需独立奖励模型、更稳定

### 1.2 现有问题
- 偏好数据质量依赖人工标注，成本高
- 单一反馈信号（仅规则或仅人工）覆盖不全
- 已有工作（Self-Reward、PRO 等）的不足：同步训练依赖强、需多模型配合

### 1.3 本文贡献
- 提出 MFRL 框架，融合规则反馈与模型自反馈生成偏好对
- 自适应融合机制统一异构反馈信号
- DPO + LoRA 高效微调，单卡可训
- LCSTS 和 Alpaca-Chinese 两个数据集上验证有效性

## 2 相关工作（约2页）
### 2.1 偏好优化方法
- RLHF（PPO）：Stiennon et al. 2020, Ouyang et al. 2022
- DPO：Rafailov et al. 2023
- KTO：Ethayarajh et al. 2024
- 其他变体（IPO、SLiC 等）

### 2.2 自动反馈生成
- 规则反馈：ROUGE（Lin 2004）、BLEU（Papineni 2002）
- 模型反馈：GPT-4 作为评判者（Zheng et al. 2023）
- Self-Reward（Yuan et al. 2024）：策略模型自评分迭代训练
- 对比反馈：同一输入多候选排序

### 2.3 高效微调
- LoRA（Hu et al. 2021）
- QLoRA、AdaLoRA 等变体
- 参数高效微调在 LLM 对齐中的应用

## 3 MFRL 框架（约3-4页）
### 3.1 问题定义与总体架构
- 偏好优化问题形式化
- MFRL 三模块流水线：MFC → AFF → POF
- 总体架构图（流程：输入 → 候选生成 → 多源反馈 → 融合 → DPO 训练）

### 3.2 多源反馈收集（MFC）
#### 3.2.1 候选生成
- Multi-temperature 采样（T ∈ {0.5, 0.8, 1.2, 1.8}）
- 拒绝过滤（去除模型拒绝回答模式）

#### 3.2.2 规则反馈（Rule-based Feedback）
- ROUGE-1/2/L 计算候选与参考的相似度
- 加权融合为单一偏好分数
- 生成偏好对（chosen/rejected）

#### 3.2.3 模型自反馈（Self-Reward Model Feedback）
- 利用策略模型自身 log-probability 作为质量信号
- 候选的 log P(candidate|prompt) 计算
- 无需外部评判模型，零额外参数开销

### 3.3 自适应反馈融合（AFF）
- 异构反馈信号的归一化
- 注意力权重学习（每个反馈源自适应权重）
- 融合分数计算与偏好对筛选

### 3.4 偏好优化微调（POF）
- DPO 损失函数
- LoRA 配置（rank=8, target_modules=all linear）
- 训练细节：学习率 5e-6, batch size 16, 3 epochs

## 4 实验与分析（约4-5页）
### 4.1 实验设置
- 基座模型：Qwen2.5-1.5B-Instruct
- 数据集：LCSTS（中文短文本摘要，2000 样本）+ Alpaca-Chinese（中文指令跟随，1000 样本）
- 评估指标：ROUGE-1/2/L
- 基线：Base、SFT、DPO、KTO
- 实现细节：单卡 3090 (24GB), LoRA rank=8, float16, gradient checkpointing

### 4.2 LCSTS 结果（主实验）
- 主结果表（加粗最优）
- 分析与讨论：MFRL 在所有基线上的优势
- 消融实验：w/o model feedback 的效果
- 候选多样性分析（multi-temperature 的影响）

### 4.3 Alpaca-Chinese 结果
- 主结果表
- 分析与讨论：指令跟随任务上 MFRL 与 DPO 持平
- 原因分析：自反馈信号对开放生成任务的局限性

### 4.4 消融与敏感性分析
- 规则反馈贡献 vs 模型反馈贡献
- 融合策略对比（等权平均 vs 自适应）
- 候选数量影响（k=2,4,6）
- LoRA rank 敏感性

### 4.5 分析与讨论
- ROUGE-1 ≈ ROUGE-L 现象的说明
- 失败案例分析
- 小模型（1.5B）的总结能力上限
- 诚实讨论 self-reward 增量有限的可能原因

## 5 结论（约0.5页）
- 本文贡献总结
- 局限性
- 未来工作方向

## 参考格式说明
- 参考文献：25-40 篇
- 电子学报格式
- 中文文献占比 20-30%

## 图表清单
### 图
1. MFRL 总体架构流程图（MFC → AFF → POF）
2. Multi-temperature 候选生成示例
3. 消融实验柱状图
4. Alpaca-Chinese 生成示例对比

### 表
1. LCSTS 结果对比表
2. Alpaca-Chinese 结果对比表
3. 消融实验结果表
4. 实验配置表（超参数）

## 附录
- 详细超参数配置
- 候选生成示例
- 评估细节
