"""模型反馈模块（Model Feedback, MF）

使用轻量级小模型作为评判者，对候选输出进行评分。
相比人工标注，成本极低；相比纯规则，能捕捉语义质量。
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelFeedback:
    """基于轻量模型的反馈信号生成器。

    使用小模型（如Qwen2.5-0.5B）对候选输出进行评分，
    通过精心设计的prompt引导模型给出1-10分的评分。
    """

    JUDGE_PROMPT = """你是一个文本质量评估专家。请对以下输出进行评分（1-10分）。

评分标准：
- 信息完整性（是否涵盖了关键信息）
- 语言流畅性（是否通顺自然）
- 相关性（是否与输入相关）
- 准确性（是否存在错误信息）

输入：{input}
参考答案：{reference}
待评估输出：{candidate}

请只输出一个1-10的整数评分，不要输出其他内容。评分："""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        max_length: int = 512,
    ):
        """初始化模型反馈模块。

        Args:
            model_name: 评判模型名称
            device: 设备
            max_length: 最大生成长度
        """
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def score(self, input_text: str, reference: str, candidate: str) -> float:
        """对单个候选输出评分。

        Args:
            input_text: 输入文本
            reference: 参考答案
            candidate: 待评估候选

        Returns:
            归一化得分 [0, 1]
        """
        prompt = self.JUDGE_PROMPT.format(
            input=input_text, reference=reference, candidate=candidate
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=1.0,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # 解析评分
        try:
            score = float(response)
            score = max(1.0, min(10.0, score))
            return score / 10.0  # 归一化到 [0, 1]
        except ValueError:
            return 0.5  # 解析失败返回中性分

    def score_batch(
        self,
        inputs: List[str],
        references: List[str],
        candidates: List[str],
    ) -> List[float]:
        """批量评分。"""
        return [
            self.score(inp, ref, cand)
            for inp, ref, cand in zip(inputs, references, candidates)
        ]

    def generate_preference_pairs(
        self,
        inputs: List[str],
        references: List[str],
        candidates_list: List[List[str]],
    ) -> List[Dict]:
        """从多候选中生成基于模型评分的偏好对。

        Args:
            inputs: 输入文本列表
            references: 参考文本列表
            candidates_list: 每个输入对应的多个候选输出

        Returns:
            偏好对列表
        """
        preference_pairs = []

        for inp, ref, candidates in zip(inputs, references, candidates_list):
            if len(candidates) < 2:
                continue

            # 对每个候选评分
            scores = [self.score(inp, ref, c) for c in candidates]
            ranked = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )

            # 取最高分和最低分构成偏好对
            chosen, chosen_score = ranked[0]
            rejected, rejected_score = ranked[-1]

            if chosen_score > rejected_score:
                preference_pairs.append({
                    "input": inp,
                    "chosen": chosen,
                    "rejected": rejected,
                    "reference": ref,
                    "score_diff": chosen_score - rejected_score,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "feedback_type": "model",
                })

        return preference_pairs
