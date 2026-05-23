"""准备 Alpaca-Chinese 数据集"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import load_alpaca_chinese

dataset = load_alpaca_chinese(max_samples=2000)
raw_data = []
for item in dataset:
    inp = item.get("instruction", "")
    inp2 = item.get("input", "")
    if inp2:
        inp = inp + " " + inp2
    raw_data.append({"input": inp, "reference": item["output"]})

# 切分为训练/测试 (1800/200)
test_size = 200
test_data = raw_data[-test_size:]
train_data = raw_data[:-test_size]

os.makedirs("data", exist_ok=True)
with open("data/alpaca_chinese_train.json", "w", encoding="utf-8") as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=2)

print(f"Total: {len(raw_data)}, Train: {len(train_data)}, Test: {len(test_data)}")
print(f"Saved to data/alpaca_chinese_train.json")
print(f"Sample: {raw_data[0]['input'][:80]}... → {raw_data[0]['reference'][:80]}...")
