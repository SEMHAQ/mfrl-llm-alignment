"""准备 Alpaca-Chinese 数据集（流式加载，只取前2000条）"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

print("Downloading Alpaca-Chinese (streaming, first 2000 samples)...")
dataset = load_dataset(
    "silk-road/alpaca-data-gpt4-chinese",
    split="train[:2000]",
    streaming=True,
)

raw_data = []
for item in dataset:
    inp = item.get("instruction", "")
    inp2 = item.get("input", "")
    if inp2:
        inp = inp + " " + inp2
    raw_data.append({"input": inp, "reference": item["output"]})

print(f"Loaded {len(raw_data)} samples")

os.makedirs("data", exist_ok=True)
save_path = "data/alpaca_chinese_train.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=2)

print(f"Saved to {save_path}")
print(f"Sample: {raw_data[0]['input'][:80]}...")
