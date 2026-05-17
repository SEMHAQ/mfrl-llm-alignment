"""课程学习调度器（Curriculum Scheduler）

按样本难度从易到难排列训练数据，实现课程学习策略。
难度由反馈分数差异（score_diff）衡量：差异越大，区分度越高，越容易学习。
"""

import numpy as np
from typing import List, Dict, Optional


class CurriculumScheduler:
    """课程学习调度器。

    根据训练进度动态调整可见样本范围：
    - 训练初期：只使用容易的样本（score_diff大，区分明显）
    - 训练后期：逐步引入难样本（score_diff小，区分模糊）
    """

    def __init__(
        self,
        total_epochs: int,
        warmup_ratio: float = 0.3,
        difficulty_metric: str = "score_diff",
    ):
        """初始化课程调度器。

        Args:
            total_epochs: 总训练轮数
            warmup_ratio: 预热比例，前多少比例的epoch只用简单样本
            difficulty_metric: 难度衡量指标
        """
        self.total_epochs = total_epochs
        self.warmup_ratio = warmup_ratio
        self.difficulty_metric = difficulty_metric
        self.sorted_indices = None

    def prepare(self, dataset: List[Dict]) -> None:
        """按难度排序数据集。

        Args:
            dataset: 偏好对数据集，每个元素需包含 score_diff 字段
        """
        diffs = [item.get(self.difficulty_metric, 0.0) for item in dataset]
        # score_diff越大越容易（区分度高），按降序排列
        self.sorted_indices = np.argsort(diffs)[::-1].tolist()

    def get_epoch_dataset(
        self, dataset: List[Dict], epoch: int
    ) -> List[Dict]:
        """获取当前epoch应使用的数据子集。

        Args:
            dataset: 完整数据集
            epoch: 当前epoch（从0开始）

        Returns:
            当前epoch的数据子集
        """
        if self.sorted_indices is None:
            self.prepare(dataset)

        # 计算当前epoch应使用的数据比例
        progress = epoch / max(self.total_epochs - 1, 1)

        if progress < self.warmup_ratio:
            # 预热阶段：只用最简单的50%数据
            ratio = 0.5 + 0.5 * (progress / self.warmup_ratio)
        else:
            # 逐步引入所有数据
            ratio = 1.0

        n_samples = max(1, int(len(dataset) * ratio))
        selected_indices = self.sorted_indices[:n_samples]

        return [dataset[i] for i in selected_indices]

    def get_difficulty_distribution(self, dataset: List[Dict]) -> Dict:
        """获取数据集的难度分布统计。"""
        diffs = [item.get(self.difficulty_metric, 0.0) for item in dataset]
        return {
            "mean": float(np.mean(diffs)),
            "std": float(np.std(diffs)),
            "min": float(np.min(diffs)),
            "max": float(np.max(diffs)),
            "median": float(np.median(diffs)),
            "total": len(diffs),
        }
