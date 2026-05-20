"""修复偏好数据：过滤拒绝回复，提高score_diff阈值

用法：
    python scripts/fix_preference_data.py
"""

import json
import os
import re

REFUSAL_PATTERNS = [
    "我不能", "我无法", "作为AI", "作为语言模型", "作为人工智能",
    "请注意", "我不会", "我没有", "我不具备", "很抱歉",
    "I cannot", "I can't", "As an AI", "I'm sorry",
    "无法生成", "无法提供", "无法完成", "无法回答",
]

META_PATTERNS = [
    "在本任务中", "这段文本", "以下是", "摘要：", "总结如下",
    "根据上述", "以上是", "本文描述", "文本讲述", "该文本",
    "给定文本", "用户提供的", "以下是给定",
]


def is_refusal(text):
    """判断是否为拒绝回复。"""
    for p in REFUSAL_PATTERNS:
        if p in text:
            return True
    return False


def is_meta(text):
    """判断是否包含大量元描述（不是真正摘要）。"""
    for p in META_PATTERNS:
        if text.startswith(p):
            return True
    return False


def clean_preference_data(input_path, output_path, min_score_diff=0.05):
    """清理偏好数据。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Original: {len(data)} pairs")

    # Step 1: 过滤拒绝回复
    filtered = []
    refusal_count = 0
    meta_count = 0
    low_diff_count = 0

    for d in data:
        chosen = d["chosen"]
        rejected = d["rejected"]

        # 跳过包含拒绝回复的对
        if is_refusal(chosen) or is_refusal(rejected):
            refusal_count += 1
            continue

        # 跳过chosen是元描述的对
        if is_meta(chosen):
            meta_count += 1
            continue

        # 跳过score_diff太小的对
        if d.get("score_diff", 0) < min_score_diff:
            low_diff_count += 0  # 不计数，保留这些对
            pass

        filtered.append(d)

    print(f"Filtered refusals: {refusal_count}")
    print(f"Filtered meta: {meta_count}")
    print(f"After filtering: {len(filtered)} pairs")

    # Step 2: 按score_diff排序，统计分布
    filtered.sort(key=lambda x: x.get("score_diff", 0), reverse=True)
    diffs = [d.get("score_diff", 0) for d in filtered]
    if diffs:
        print(f"score_diff: mean={sum(diffs)/len(diffs):.4f}, max={max(diffs):.4f}")
        bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
        for lo, hi in bins:
            c = sum(1 for d in diffs if lo <= d < hi)
            print(f"  [{lo:.2f}, {hi:.2f}): {c}")

    # Step 3: 应用min_score_diff过滤
    if min_score_diff > 0:
        final = [d for d in filtered if d.get("score_diff", 0) >= min_score_diff]
        print(f"After min_score_diff={min_score_diff}: {len(final)} pairs")
    else:
        final = filtered

    # Step 4: 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

    return final


def main():
    # 处理所有preference_data.json
    configs = [
        ("outputs/mfrl_v3/preference_data.json", "outputs/mfrl_v3/preference_data_clean.json", 0.05),
        ("outputs/mfrl/preference_data.json", "outputs/mfrl/preference_data_clean.json", 0.05),
    ]

    for inp, out, min_diff in configs:
        if os.path.exists(inp):
            print(f"\n{'='*50}")
            print(f"Processing: {inp}")
            print(f"{'='*50}")
            clean_preference_data(inp, out, min_diff)


if __name__ == "__main__":
    main()
