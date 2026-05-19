"""规则反馈模块（Rule Feedback, RF）

基于自动评估指标（ROUGE、BLEU等）生成偏好对。
无须额外模型推理，计算开销极低。
"""

import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer


class RuleFeedback:
    """基于规则的反馈信号生成器。

    对同一输入的多个候选输出，使用ROUGE等指标与参考答案对比，
    生成偏好排序和偏好对。
    """

    def __init__(self, metrics: List[str] = None, weights: Dict[str, float] = None):
        """初始化规则反馈模块。

        Args:
            metrics: 使用的指标列表，默认 ["rouge1", "rouge2", "rougeL"]
            weights: 各指标权重，默认等权
        """
        self.metrics = metrics or ["rouge1", "rouge2", "rougeL"]
        self.weights = weights or {m: 1.0 / len(self.metrics) for m in self.metrics}
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=False)

    def score(self, reference: str, candidate: str) -> float:
        """计算单个候选相对于参考的综合得分。

        Args:
            reference: 参考文本
            candidate: 候选文本

        Returns:
            加权综合得分 [0, 1]
        """
        scores = self.scorer.score(reference, candidate)
        weighted_score = sum(
            self.weights[m] * scores[m].fmeasure for m in self.metrics
        )
        return weighted_score

    def score_batch(
        self, references: List[str], candidates: List[str]
    ) -> List[float]:
        """批量计算得分。"""
        return [
            self.score(ref, cand)
            for ref, cand in zip(references, candidates)
        ]

    def generate_preference_pairs(
        self,
        inputs: List[str],
        references: List[str],
        candidates_list: List[List[str]],
        top_k: int = 1,
    ) -> List[Dict]:
        """从多候选中生成偏好对。

        对每个输入的多个候选按得分排序，取top-k作为正例，bottom-k作为负例。

        Args:
            inputs: 输入文本列表
            references: 参考文本列表
            candidates_list: 每个输入对应的多个候选输出
            top_k: 取前k个作为正例

        Returns:
            偏好对列表，每个元素包含 input, chosen, rejected, score_diff
        """
        preference_pairs = []

        for inp, ref, candidates in zip(inputs, references, candidates_list):
            if len(candidates) < 2:
                continue

            # 计算每个候选的得分
            scores = [self.score(ref, c) for c in candidates]
            ranked = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )

            # 生成偏好对：top-k vs bottom-k
            for i in range(min(top_k, len(ranked) // 2)):
                chosen, chosen_score = ranked[i]
                rejected, rejected_score = ranked[-(i + 1)]
                preference_pairs.append({
                    "input": inp,
                    "chosen": chosen,
                    "rejected": rejected,
                    "reference": ref,
                    "score_diff": chosen_score - rejected_score,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "feedback_type": "rule",
                })

        return preference_pairs
