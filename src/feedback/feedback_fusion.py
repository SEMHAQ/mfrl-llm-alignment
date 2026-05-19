"""自适应反馈融合模块（Adaptive Feedback Fusion, AFF）

将来自不同反馈源（规则、模型等）的异构信号归一化并融合为统一的偏好分数。
使用可学习的注意力权重自适应调整各反馈源的重要性。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class AdaptiveFeedbackFusion(nn.Module):
    """自适应反馈融合模块。

    通过注意力机制学习各反馈源的权重，将异构反馈信号融合为统一偏好分数。
    核心思想：不同任务/场景下，各反馈源的可靠性不同，应自适应调整权重。
    """

    def __init__(self, num_feedback_sources: int = 3, hidden_dim: int = 64):
        """初始化融合模块。

        Args:
            num_feedback_sources: 反馈源数量（规则反馈、模型反馈、对比反馈）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.num_sources = num_feedback_sources

        # 轻量级注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(num_feedback_sources, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_feedback_sources),
            nn.Softmax(dim=-1),
        )

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        feedback_scores: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """融合多源反馈分数。

        Args:
            feedback_scores: 各反馈源的分数 [batch_size, num_sources]
            confidence: 各反馈源的置信度 [batch_size, num_sources]（可选）

        Returns:
            包含融合分数和权重的字典
        """
        # 计算自适应权重
        weights = self.attention_net(feedback_scores)  # [batch_size, num_sources]

        # 如果有置信度信息，加权调整
        if confidence is not None:
            weights = weights * confidence
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权融合
        fused_score = (feedback_scores * weights).sum(dim=-1)  # [batch_size]

        # 温度缩放
        fused_score = fused_score / self.temperature

        return {
            "fused_score": fused_score,
            "weights": weights,
            "temperature": self.temperature,
        }

    def fuse_preference_pairs(
        self,
        pairs_by_source: Dict[str, List[Dict]],
    ) -> List[Dict]:
        """融合来自不同反馈源的偏好对。

        将多个反馈源生成的偏好对按输入对齐，融合分数差异，
        生成最终的统一偏好对。

        Args:
            pairs_by_source: 各反馈源的偏好对字典
                {"rule": [...], "model": [...], ...}

        Returns:
            融合后的偏好对列表
        """
        # 按输入文本对齐
        aligned = {}
        source_names = list(pairs_by_source.keys())

        for source, pairs in pairs_by_source.items():
            for pair in pairs:
                key = pair["input"]
                if key not in aligned:
                    aligned[key] = {
                        "input": pair["input"],
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                        "reference": pair.get("reference", ""),
                        "scores": {},
                    }
                aligned[key]["scores"][source] = pair["score_diff"]

        # 融合
        fused_pairs = []
        for key, item in aligned.items():
            if len(item["scores"]) < 1:
                continue

            # 构造分数向量
            score_vec = []
            for source in source_names:
                score_vec.append(item["scores"].get(source, 0.0))

            score_tensor = torch.tensor([score_vec], dtype=torch.float32)

            with torch.no_grad():
                result = self.forward(score_tensor)

            fused_pairs.append({
                "input": item["input"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "reference": item.get("reference", ""),
                "fused_score_diff": result["fused_score"].item(),
                "source_weights": {
                    source: result["weights"][0][i].item()
                    for i, source in enumerate(source_names)
                },
            })

        # 按融合分数排序
        fused_pairs.sort(key=lambda x: x["fused_score_diff"], reverse=True)
        return fused_pairs


def normalize_scores(scores: List[float]) -> List[float]:
    """将分数归一化到 [0, 1] 区间。"""
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-8:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]
