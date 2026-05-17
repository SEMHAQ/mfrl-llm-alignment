"""数据预处理

生成候选输出、构建偏好对的完整流程。
"""

import torch
import random
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_candidates(
    model,
    tokenizer,
    prompts: List[str],
    num_candidates: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> List[List[str]]:
    """为每个prompt生成多个候选输出。

    使用不同的采样参数生成多样化候选，用于后续反馈评分和偏好对构建。

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompts: 输入prompt列表
        num_candidates: 每个prompt生成的候选数
        temperature: 采样温度
        top_p: nucleus sampling参数
        max_new_tokens: 最大生成长度
        batch_size: 批处理大小

    Returns:
        候选输出列表，shape [num_prompts, num_candidates]
    """
    model.eval()
    all_candidates = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        for prompt in batch_prompts:
            candidates = []
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            for _ in range(num_candidates):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                candidates.append(response)

            all_candidates.append(candidates)

    return all_candidates


def preprocess_for_dpo(
    model,
    tokenizer,
    raw_data: List[Dict],
    num_candidates: int = 4,
    max_samples: Optional[int] = None,
) -> Dict:
    """完整的DPO数据预处理流程。

    1. 提取输入和参考
    2. 生成多个候选输出
    3. 使用反馈模块评分和融合
    4. 构建偏好对

    Args:
        model: 用于生成候选的模型
        tokenizer: 分词器
        raw_data: 原始数据列表，每个元素包含 input/reference 字段
        num_candidates: 每个输入生成的候选数
        max_samples: 最大处理样本数

    Returns:
        包含偏好对和统计信息的字典
    """
    from src.feedback.rule_feedback import RuleFeedback
    from src.feedback.feedback_fusion import AdaptiveFeedbackFusion, normalize_scores

    if max_samples:
        raw_data = raw_data[:max_samples]

    inputs = [item["input"] for item in raw_data]
    references = [item["reference"] for item in raw_data]

    # Step 1: 生成候选
    print(f"Generating {num_candidates} candidates for {len(inputs)} inputs...")
    candidates_list = generate_candidates(
        model, tokenizer, inputs, num_candidates=num_candidates
    )

    # Step 2: 规则反馈评分
    print("Computing rule-based feedback scores...")
    rule_feedback = RuleFeedback()
    rule_pairs = rule_feedback.generate_preference_pairs(
        inputs, references, candidates_list
    )

    # Step 3: 模型反馈（可选，需要额外的小模型）
    # 这里先只用规则反馈，保持轻量
    pairs_by_source = {"rule": rule_pairs}

    # Step 4: 融合（单源时直接使用）
    if len(pairs_by_source) == 1:
        final_pairs = rule_pairs
    else:
        fusion = AdaptiveFeedbackFusion(num_feedback_sources=len(pairs_by_source))
        final_pairs = fusion.fuse_preference_pairs(pairs_by_source)

    # 格式化为DPO训练格式
    dpo_data = []
    for pair in final_pairs:
        dpo_data.append({
            "prompt": f"请为以下文本生成简洁准确的摘要：\n{pair['input']}\n摘要：",
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "score_diff": pair.get("score_diff", pair.get("fused_score_diff", 0.0)),
        })

    return {
        "dpo_data": dpo_data,
        "stats": {
            "total_inputs": len(inputs),
            "total_pairs": len(dpo_data),
            "avg_score_diff": sum(d["score_diff"] for d in dpo_data) / max(len(dpo_data), 1),
        },
    }
