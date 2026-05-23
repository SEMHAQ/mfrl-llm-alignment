"""从已有checkpoint提取训练指标，生成loss曲线数据"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 trainer_state.json 提取 metrics
checkpoints = [
    "outputs/mfrl_v3/checkpoint-102",
    "outputs/mfrl_v3/checkpoint-153",
    "outputs/mfrl_v3/checkpoint-204",
    "outputs/mfrl_v3/checkpoint-255",
    "outputs/mfrl_v3/checkpoint-306",
]

# Fallback: check all checkpoints in final dir
final = "outputs/mfrl_v3/final"
all_dirs = [final]
for root, dirs, files in os.walk("outputs/mfrl_v3"):
    for d in dirs:
        if d.startswith("checkpoint-"):
            all_dirs.append(os.path.join(root, d))

print("Found checkpoints:")
for d in all_dirs:
    state_file = os.path.join(d, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        print(f"  {d}: {len(log_history)} log entries")

# Extract loss curve data
metrics_data = []
for d in all_dirs:
    state_file = os.path.join(d, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        for entry in state.get("log_history", []):
            if "loss" in entry and "epoch" in entry:
                metrics_data.append({
                    "step": entry.get("step", 0),
                    "epoch": entry["epoch"],
                    "loss": entry["loss"],
                    "grad_norm": entry.get("grad_norm", 0),
                    "learning_rate": entry.get("learning_rate", 0),
                })

metrics_data.sort(key=lambda x: x["step"])

if metrics_data:
    out_path = "paper/figures/loss_curve.json"
    os.makedirs("paper/figures", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved {len(metrics_data)} data points to {out_path}")
    print(f"First: step={metrics_data[0]['step']}, loss={metrics_data[0]['loss']:.4f}")
    print(f"Last:  step={metrics_data[-1]['step']}, loss={metrics_data[-1]['loss']:.4f}")
else:
    print("No metrics found. Check trainer_state.json exists in checkpoint dirs.")
