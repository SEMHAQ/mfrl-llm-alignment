"""数据预处理

生成候选输出、构建偏好对的完整流程。
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_candidates(
    model,
    tokenizer,
    prompts: List[str],
    num_candidates: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 128,
    batch_size: int = 32,
) -> List[List[str]]:
    """为每个prompt生成多个候选输出。

    使用num_return_sequences批量生成，比逐个生成快数倍。

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompts: 输入prompt列表
        num_candidates: 每个prompt生成的候选数
        temperature: 采样温度
        top_p: nucleus sampling参数
        max_new_tokens: 最大生成长度（摘要任务128足够）
        batch_size: 每批处理的prompt数

    Returns:
        候选输出列表，shape [num_prompts, num_candidates]
    """
    model.eval()
    # decoder-only模型需要left-padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_candidates = []
    total = len(prompts)

    for i in range(0, total, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_size_actual = len(batch_prompts)

        # 为每个prompt构造chat格式
        batch_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        # tokenize整个batch
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=256
        ).to(model.device)

        # 一次性生成num_candidates个候选（num_return_sequences）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_candidates,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 解码并按prompt分组
        # outputs shape: [batch_size * num_candidates, seq_len]
        input_len = inputs["input_ids"].shape[1]
        for j in range(batch_size_actual):
            candidates = []
            for k in range(num_candidates):
                idx = j * num_candidates + k
                response = tokenizer.decode(
                    outputs[idx][input_len:], skip_special_tokens=True
                ).strip()
                candidates.append(response)
            all_candidates.append(candidates)

        if (i + batch_size) % 50 == 0 or i + batch_size >= total:
            print(f"  Generated candidates for {min(i + batch_size, total)}/{total} prompts")

    return all_candidates


def preprocess_for_dpo(
    model,
    tokenizer,
    raw_data: List[Dict],
    num_candidates: int = 4,
    max_samples: Optional[int] = None,
) -> Dict:
    """完整的DPO数据预处理流程。

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

    # 格式化为DPO训练格式
    dpo_data = []
    for pair in rule_pairs:
        dpo_data.append({
            "prompt": f"请为以下文本生成简洁准确的摘要：\n{pair['input']}\n摘要：",
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "score_diff": pair.get("score_diff", 0.0),
        })

    return {
        "dpo_data": dpo_data,
        "stats": {
            "total_inputs": len(inputs),
            "total_pairs": len(dpo_data),
            "avg_score_diff": sum(d["score_diff"] for d in dpo_data) / max(len(dpo_data), 1),
        },
    }
