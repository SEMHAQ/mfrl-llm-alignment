"""评估指标

ROUGE指标计算和Win Rate评估。
"""

import numpy as np
from typing import List, Dict
from rouge_score import rouge_scorer


def compute_rouge(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = False,
) -> Dict[str, float]:
    """计算ROUGE指标。

    Args:
        predictions: 模型预测列表
        references: 参考文本列表
        use_stemmer: 是否使用词干提取（中文不需要）

    Returns:
        包含ROUGE-1/2/L的F1分数
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for metric in scores:
            scores[metric].append(result[metric].fmeasure)

    return {
        metric: float(np.mean(values))
        for metric, values in scores.items()
    }


def compute_win_rate(
    model_outputs: List[str],
    baseline_outputs: List[str],
    references: List[str],
) -> Dict[str, float]:
    """计算Win Rate（基于ROUGE-L的自动评估）。

    比较模型输出和baseline输出相对于参考的ROUGE-L分数。

    Args:
        model_outputs: 模型输出列表
        baseline_outputs: baseline输出列表
        references: 参考文本列表

    Returns:
        win/tie/lose的比例
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    wins, ties, losses = 0, 0, 0

    for mo, bo, ref in zip(model_outputs, baseline_outputs, references):
        model_score = scorer.score(ref, mo)["rougeL"].fmeasure
        baseline_score = scorer.score(ref, bo)["rougeL"].fmeasure

        if model_score > baseline_score + 0.01:
            wins += 1
        elif baseline_score > model_score + 0.01:
            losses += 1
        else:
            ties += 1

    total = len(model_outputs)
    return {
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "lose_rate": losses / total,
    }
