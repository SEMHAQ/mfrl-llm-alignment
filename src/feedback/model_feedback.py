"""模型反馈模块（Model Feedback, MF）

使用模型对候选输出进行评分。
支持两种模式：
1. 外部评判模型（如Qwen2.5-0.5B）
2. Self-reward（使用策略模型自身的log-probability作为质量信号）
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
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
        """对单个候选输出评分。"""
        if self.use_self_reward:
            return self._score_logprob_mega([input_text], [candidate])[0]
        else:
            return self._score_judge(input_text, reference, candidate)

    def _score_logprob_mega(
        self, prompts: List[str], candidates: List[str]
    ) -> List[float]:
        """批量评分：多个不同prompt的(输入,候选)对在一次forward中完成。

        Args:
            prompts: 每个候选对应的输入文本（长度 = candidates）
            candidates: 候选文本列表
        """
        # 构建所有序列并记录每个序列的prompt token长度
        full_texts = []
        prompt_lens = []
        for prompt, cand in zip(prompts, candidates):
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_texts.append(prompt + cand)
            p_ids = self.tokenizer(prompt_formatted, return_tensors="pt")
            prompt_lens.append(p_ids["input_ids"].shape[1])

        enc = self.tokenizer(full_texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        scores = []
        for i in range(len(candidates)):
            actual_len = attention_mask[i].sum().item()
            p_len = prompt_lens[i]
            cand_len = actual_len - p_len
            if cand_len <= 0:
                scores.append(0.0)
                continue

            cand_logits = logits[i, p_len - 1 : actual_len - 1, :]
            cand_ids = input_ids[i, p_len:actual_len]
            log_probs = torch.log_softmax(cand_logits, dim=-1)
            token_log_probs = log_probs.gather(1, cand_ids.unsqueeze(1)).squeeze(1)
            avg_log_prob = token_log_probs.mean().item()
            scores.append(max(0.0, min(1.0, (avg_log_prob + 10.0) / 10.0)))

        return scores

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
        """从多候选中生成基于模型评分的偏好对。"""
        preference_pairs = []
        MEGA_BATCH = 1  # 逐个处理，稳定优先

        # 预处理：过滤无效候选
        items = []
        for inp, ref, cands in zip(inputs, references, candidates_list):
            valid = [c for c in cands if is_valid_candidate(c)]
            if len(valid) >= 2:
                items.append((inp, ref, valid))

        for b_start in tqdm(range(0, len(items), MEGA_BATCH), desc="Model feedback"):
            batch_items = items[b_start : b_start + MEGA_BATCH]

            all_prompts = []
            all_refs = []
            all_candidates = []
            cand_counts = []
            for inp, ref, valid in batch_items:
                for c in valid:
                    all_prompts.append(inp)
                    all_refs.append(ref)
                    all_candidates.append(c)
                cand_counts.append(len(valid))

            if self.use_self_reward:
                all_scores = self._score_logprob_mega(all_prompts, all_candidates)
            else:
                all_scores = [
                    self._score_judge(p, r, c)
                    for p, r, c in zip(all_prompts, all_refs, all_candidates)
                ]

            # 分发分数回各输入
            offset = 0
            for (inp, ref, valid), n in zip(batch_items, cand_counts):
                scores = all_scores[offset : offset + n]
                offset += n

                ranked = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)
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
