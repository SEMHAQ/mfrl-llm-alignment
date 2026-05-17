"""数据集准备脚本

在能访问HuggingFace的机器上运行，下载数据集并保存为JSON。
然后git push，实验机git pull即可使用。

用法：
    # 本机运行（能访问HuggingFace）
    python scripts/prepare_data.py --dataset lcsts --max_samples 3000
    python scripts/prepare_data.py --dataset alpaca --max_samples 5000

    # 实验机直接运行实验（不需要下载）
    python scripts/run_experiment.py --config configs/train.yaml
"""

import argparse
import json
import os
from pathlib import Path


def download_lcsts(max_samples: int, output_dir: str):
    """下载LCSTS数据集并保存为JSON。"""
    from datasets import load_dataset

    lcsts_names = [
        "hfl/lcsts",
        "seamew/LCSTS",
        "IsmaelMousa/LCSTS",
        "xiaoda/LCSTS",
        "lcsts",
    ]
    dataset = None
    for name in lcsts_names:
        try:
            print(f"Trying: {name} ...")
            dataset = load_dataset(name, split="train")
            print(f"Success: {name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if dataset is None:
        raise RuntimeError("Failed to load LCSTS. Check dataset names on huggingface.co/datasets")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # 转换为标准格式
    data = []
    for item in dataset:
        # LCSTS字段可能是 source/summary 或 text/summary
        source = item.get("source", item.get("text", ""))
        summary = item.get("summary", "")
        if source and summary:
            data.append({"input": source, "reference": summary})

    output_path = os.path.join(output_dir, "lcsts_train.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")
    return output_path


def download_alpaca(max_samples: int, output_dir: str):
    """下载Alpaca-Chinese数据集并保存为JSON。"""
    from datasets import load_dataset

    dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    data = []
    for item in dataset:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        prompt = f"{instruction} {inp}".strip()
        if prompt and output:
            data.append({"input": prompt, "reference": output})

    output_path = os.path.join(output_dir, "alpaca_chinese_train.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--dataset", type=str, default="lcsts",
                        choices=["lcsts", "alpaca", "both"])
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output or str(Path(__file__).parent.parent / "data")
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset in ("lcsts", "both"):
        download_lcsts(args.max_samples, output_dir)
    if args.dataset in ("alpaca", "both"):
        download_alpaca(args.max_samples, output_dir)

    print("\nDone! Now git add + commit + push these files.")
    print(f"  git add data/*.json")
    print(f"  git commit -m 'data: add training dataset'")
    print(f"  git push")


if __name__ == "__main__":
    main()
