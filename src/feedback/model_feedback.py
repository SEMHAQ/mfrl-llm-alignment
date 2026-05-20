"""模型反馈模块（Model Feedback, MF）

使用模型对候选输出进行评分。
支持两种模式：
1. 外部评判模型（如Qwen2.5-0.5B）
2. Self-reward（使用策略模型自身的log-probability作为质量信号）
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .rule_feedback import is_valid_candidate


class ModelFeedback:
    """基于模型的反馈信号生成器。

    支持两种模式：
    1. 指定judge_model_name：加载外部评判模型
    2. 传入已有model/tokenizer：使用self-reward（log-probability）
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
        model_name: str = None,
        model=None,
        tokenizer=None,
        device: str = "cuda",
        max_length: int = 512,
    ):
        """初始化模型反馈模块。

        Args:
            model_name: 评判模型名称（如果model未提供）
            model: 已加载的模型（self-reward模式）
            tokenizer: 已加载的tokenizer（self-reward模式）
            device: 设备
            max_length: 最大生成长度
        """
        self.device = device
        self.max_length = max_length
        self.use_self_reward = model is not None

        if self.use_self_reward:
            self.model = model
            self.tokenizer = tokenizer
        else:
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
        if self.use_self_reward:
            return self._score_logprob(input_text, candidate)
        else:
            return self._score_judge(input_text, reference, candidate)

    def _score_logprob(self, prompt_text: str, candidate: str) -> float:
        """Self-reward: 用模型的log-probability作为质量信号。"""
        full_text = prompt_text + candidate
        messages = [{"role": "user", "content": prompt_text}]
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_formatted, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        full_ids = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = full_ids["input_ids"]

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        # 计算candidate部分的平均log-probability
        candidate_logits = logits[0, prompt_len - 1:-1, :]
        candidate_ids = input_ids[0, prompt_len:]
        log_probs = torch.log_softmax(candidate_logits, dim=-1)
        token_log_probs = log_probs.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)
        avg_log_prob = token_log_probs.mean().item()

        # 归一化到 [0, 1]（log-prob通常在-10到0之间）
        score = max(0.0, min(1.0, (avg_log_prob + 10.0) / 10.0))
        return score

    def _score_judge(self, input_text: str, reference: str, candidate: str) -> float:
        """外部评判模型：用prompt引导评分。"""
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

        try:
            score = float(response)
            score = max(1.0, min(10.0, score))
            return score / 10.0
        except ValueError:
            return 0.5

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
            # 过滤无效候选
            valid_candidates = [c for c in candidates if is_valid_candidate(c)]
            if len(valid_candidates) < 2:
                continue

            # 对每个候选评分
            scores = [self.score(inp, ref, c) for c in valid_candidates]
            ranked = sorted(
                zip(valid_candidates, scores), key=lambda x: x[1], reverse=True
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
