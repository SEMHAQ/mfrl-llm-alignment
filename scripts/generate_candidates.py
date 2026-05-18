"""独立的候选生成脚本

生成候选输出并保存到缓存，进程退出后显存完全释放。
训练时直接读缓存，不会冲突。

用法：
    python scripts/generate_candidates.py --config configs/train.yaml
    python scripts/generate_candidates.py --config configs/train.yaml --max_samples 10000
"""

import os
import sys
import json
import argparse
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.preprocess import generate_candidates


def main():
    parser = argparse.ArgumentParser(description="Generate candidates and save to cache")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override max samples (default: use config value)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Generation batch size (default: 32)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载数据集
    dataset_name = config["data"]["dataset"]
    local_json = os.path.join("data", f"{dataset_name}_train.json")

    if os.path.exists(local_json):
        print(f"Loading from local file: {local_json}")
        with open(local_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        print(f"ERROR: {local_json} not found. Run prepare_data.py first.")
        return

    max_samples = args.max_samples or config["data"]["max_train_samples"]
    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]

    inputs = [item["input"] for item in raw_data]
    print(f"Total samples: {len(inputs)}")

    # 缓存路径
    num_candidates = config["feedback"]["num_candidates"]
    gen_config = config.get("generation", {})
    temp = gen_config.get("temperature", 0.8)
    cache_path = os.path.join("data", f"candidates_k{num_candidates}_t{temp}.json")

    # 检查已有缓存
    n_cached = 0
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        n_cached = len(cached)
        print(f"Cache exists: {n_cached} samples")

    if n_cached >= len(inputs):
        print(f"Cache already has enough samples ({n_cached} >= {len(inputs)}). Done.")
        return

    # 需要生成的部分
    inputs_to_gen = inputs[n_cached:]
    print(f"Need to generate: {len(inputs_to_gen)} samples")

    # 加载模型
    model_name = config["model"]["name"]
    dtype_str = config["model"]["dtype"]
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype_map.get(dtype_str, torch.float16),
        device_map="auto",
        trust_remote_code=True,
    )

    # 生成
    print(f"Generating {num_candidates} candidates per input, batch_size={args.batch_size}...")
    new_candidates = generate_candidates(
        model, tokenizer, inputs_to_gen,
        num_candidates=num_candidates,
        temperature=temp,
        top_p=gen_config.get("top_p", 0.95),
        max_new_tokens=gen_config.get("max_new_tokens", 128),
        batch_size=args.batch_size,
    )

    # 合并并保存
    if n_cached > 0:
        all_candidates = cached + new_candidates
    else:
        all_candidates = new_candidates

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(all_candidates)} candidates to {cache_path}")
    print(f"  Cached: {n_cached}, Generated: {len(new_candidates)}")


if __name__ == "__main__":
    main()
