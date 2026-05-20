"""检查偏好数据的score_diff分布"""

import json
import glob

for p in glob.glob("outputs/**/preference_data.json", recursive=True):
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    diffs = [d.get("score_diff", 0) for d in data]
    mean_diff = sum(diffs) / len(diffs)
    print(f"{p}: {len(data)} pairs, score_diff mean={mean_diff:.4f}, max={max(diffs):.4f}")
    bins = [(0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)]
    for lo, hi in bins:
        c = sum(1 for d in diffs if lo <= d < hi)
        print(f"  [{lo:.2f}, {hi:.2f}): {c}")
    for i in range(3):
        chosen = data[i]["chosen"][:60]
        rejected = data[i]["rejected"][:60]
        print(f"  [{i}] chosen: {chosen}")
        print(f"  [{i}] rejected: {rejected}")
        print(f"  [{i}] diff: {diffs[i]:.4f}")
