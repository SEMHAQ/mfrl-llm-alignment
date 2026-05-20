"""评估器

封装完整的评估流程，支持自动指标和模型评估。
"""

import json
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .metrics import compute_rouge, compute_win_rate


class Evaluator:
    """MFRL评估器。

    支持ROUGE自动指标评估和基于Win Rate的对比评估。
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 50,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        """批量生成模型输出。"""
        self.model.eval()
        outputs = []

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_outputs = []

            for prompt in batch:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response = self.tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                batch_outputs.append(response)

            outputs.extend(batch_outputs)

        return outputs

    def evaluate_rouge(
        self,
        test_data: List[Dict],
        prompt_key: str = "prompt",
        reference_key: str = "chosen",
    ) -> Dict[str, float]:
        """评估ROUGE指标。

        Args:
            test_data: 测试数据列表
            prompt_key: prompt字段名
            reference_key: 参考文本字段名

        Returns:
            ROUGE指标字典
        """
        prompts = [item[prompt_key] for item in test_data]
        references = [item[reference_key] for item in test_data]

        predictions = self.generate(prompts)
        rouge_scores = compute_rouge(predictions, references)

        return rouge_scores

    def evaluate_win_rate(
        self,
        test_data: List[Dict],
        baseline_outputs: List[str],
        prompt_key: str = "prompt",
        reference_key: str = "chosen",
    ) -> Dict[str, float]:
        """评估Win Rate。

        Args:
            test_data: 测试数据列表
            baseline_outputs: baseline模型的输出
            prompt_key: prompt字段名
            reference_key: 参考文本字段名

        Returns:
            Win Rate指标字典
        """
        prompts = [item[prompt_key] for item in test_data]
        references = [item[reference_key] for item in test_data]

        model_outputs = self.generate(prompts)
        win_rate = compute_win_rate(model_outputs, baseline_outputs, references)

        return win_rate

    def full_evaluation(
        self,
        test_data: List[Dict],
        baseline_model=None,
        save_path: Optional[str] = None,
    ) -> Dict:
        """完整评估流程。

        Args:
            test_data: 测试数据
            baseline_model: baseline模型（可选，用于Win Rate对比）
            save_path: 结果保存路径

        Returns:
            完整评估结果
        """
        results = {}

        # ROUGE评估
        rouge_scores = self.evaluate_rouge(test_data)
        results["rouge"] = rouge_scores
        print(f"ROUGE scores: {rouge_scores}")

        # Win Rate评估（如果有baseline）
        if baseline_model is not None:
            baseline_evaluator = Evaluator(
                baseline_model, self.tokenizer,
                self.max_new_tokens, self.batch_size
            )
            prompts = [item["prompt"] for item in test_data]
            baseline_outputs = baseline_evaluator.generate(prompts)
            win_rate = self.evaluate_win_rate(test_data, baseline_outputs)
            results["win_rate"] = win_rate
            print(f"Win Rate: {win_rate}")

        # 保存结果
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {save_path}")

        return results
