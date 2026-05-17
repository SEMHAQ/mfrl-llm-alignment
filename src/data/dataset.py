"""数据集加载与处理

支持LCSTS（中文摘要）和Alpaca-Chinese（指令跟随）数据集。
"""

import os
import json
import random
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset, Dataset


# LCSTS数据集配置
LCSTS_CONFIG = {
    "source_column": "source",
    "summary_column": "summary",
}


def load_lcsts(
    subset: str = "default",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """加载LCSTS中文摘要数据集。

    Args:
        subset: 子集名称
        split: 数据分割（train/validation/test）
        max_samples: 最大样本数

    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset("lcsts", subset, split=split, trust_remote_code=True)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def load_alpaca_chinese(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """加载中文Alpaca指令跟随数据集。

    Args:
        split: 数据分割
        max_samples: 最大样本数

    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset(
        "silk-road/alpaca-data-gpt4-chinese",
        split=split,
        trust_remote_code=True,
    )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


class PreferenceDataset:
    """偏好数据集封装。

    将原始数据转换为DPO训练所需的格式：prompt, chosen, rejected。
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_lcsts_preferences(
        self,
        dataset: Dataset,
        preference_pairs: List[Dict],
    ) -> List[Dict]:
        """准备LCSTS数据集的偏好数据。

        Args:
            dataset: 原始LCSTS数据集
            preference_pairs: 融合后的偏好对

        Returns:
            DPO训练格式的数据列表
        """
        dpo_data = []
        for pair in preference_pairs:
            prompt = self._format_prompt(pair["input"], task="summarization")
            dpo_data.append({
                "prompt": prompt,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })
        return dpo_data

    def prepare_alpaca_preferences(
        self,
        dataset: Dataset,
        preference_pairs: List[Dict],
    ) -> List[Dict]:
        """准备Alpaca数据集的偏好数据。"""
        dpo_data = []
        for pair in preference_pairs:
            prompt = self._format_prompt(pair["input"], task="instruction")
            dpo_data.append({
                "prompt": prompt,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })
        return dpo_data

    def _format_prompt(self, input_text: str, task: str = "summarization") -> str:
        """格式化输入prompt。"""
        if task == "summarization":
            return f"请为以下文本生成简洁准确的摘要：\n{input_text}\n摘要："
        else:
            return f"{input_text}"

    def to_hf_dataset(self, data: List[Dict]) -> Dataset:
        """转换为HuggingFace Dataset格式。"""
        return Dataset.from_list(data)
