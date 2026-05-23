"""生成论文所有 EPS 图。在实验机上运行。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = ['#4f81bd', '#9bbb59', '#c0504d', '#f4a940', '#8064a2', '#4bacc6']

# ====== 图2: Loss 曲线 ======
loss_path = os.path.join(OUT, "loss_curve.json")
if os.path.exists(loss_path):
    with open(loss_path) as f:
        data = json.load(f)
    steps = [d["step"] for d in data]
    losses = [d["loss"] for d in data]
    epochs = [d["epoch"] for d in data]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, losses, color=COLORS[0], linewidth=1.2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("DPO Loss")
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add epoch boundaries
    epoch_bounds = [0]
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_bounds.append(steps[i])
    for b in epoch_bounds[1:]:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUT, "fig2_loss_curve.eps")
    plt.savefig(path, format='eps')
    plt.close()
    print(f"Saved {path}")
else:
    print(f"Loss data not found at {loss_path}, run extract_metrics.py first")

# ====== 图3: LCSTS 柱状图 ======
methods = ['Base', 'SFT', 'DPO', 'KTO', 'MFRL\n(w/o MF)', 'MFRL']
rouge_l = [0.162, 0.151, 0.158, 0.163, 0.169, 0.171]
bar_colors = ['#b0c4de', '#4f81bd', '#9bbb59', '#f4a940', '#4bacc6', '#c0504d']

fig, ax = plt.subplots(figsize=(6, 3.8))
x = np.arange(len(methods))
bars = ax.bar(x, rouge_l, width=0.55, color=bar_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, rouge_l):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
bars[-1].set_edgecolor('#c0504d')
bars[-1].set_linewidth(2)
ax.set_ylim(0, 0.20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel('ROUGE-L', fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
path = os.path.join(OUT, "fig3_lcsts_bars.eps")
plt.savefig(path, format='eps')
plt.close()
print(f"Saved {path}")

# ====== 图4: Alpaca 柱状图 ======
methods2 = ['SFT', 'DPO', 'MFRL']
rouge_l2 = [0.122, 0.154, 0.151]
bar_colors2 = ['#4f81bd', '#9bbb59', '#c0504d']

fig, ax = plt.subplots(figsize=(4.5, 3.5))
x = np.arange(len(methods2))
bars = ax.bar(x, rouge_l2, width=0.45, color=bar_colors2, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, rouge_l2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.18)
ax.set_xticks(x)
ax.set_xticklabels(methods2, fontsize=10)
ax.set_ylabel('ROUGE-L', fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
path = os.path.join(OUT, "fig4_alpaca_bars.eps")
plt.savefig(path, format='eps')
plt.close()
print(f"Saved {path}")

# ====== 图5: 消融实验 ======
ablation_methods = ['等权融合\n(w/o AFF)', '仅规则\n(w/o MF)', '完整 MFRL']
ablation_values = [0.170, 0.169, 0.171]
ablation_colors = ['#9bbb59', '#f4a940', '#c0504d']

fig, ax = plt.subplots(figsize=(4.5, 3.5))
x = np.arange(len(ablation_methods))
bars = ax.bar(x, ablation_values, width=0.4, color=ablation_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, ablation_values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0.155, 0.180)
ax.set_xticks(x)
ax.set_xticklabels(ablation_methods, fontsize=9)
ax.set_ylabel('ROUGE-L', fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
path = os.path.join(OUT, "fig5_ablation.eps")
plt.savefig(path, format='eps')
plt.close()
print(f"Saved {path}")

print("\nAll figures generated in paper/figures/")
