"""生成论文 EPS 图（英文标签，dvipdfmx兼容）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

C = ['#4f81bd', '#9bbb59', '#c0504d', '#f4a940', '#8064a2', '#4bacc6']

# ====== Fig.2: Loss Curve ======
loss_path = os.path.join(OUT, "loss_curve.json")
if os.path.exists(loss_path):
    with open(loss_path) as f:
        data = json.load(f)

    # Filter: only training steps (skip eval/save entries)
    train_data = [d for d in data if d.get("loss", 0) > 0.05]
    steps = [d["step"] for d in train_data]
    losses = [d["loss"] for d in train_data]
    epochs = [d["epoch"] for d in train_data]

    # Smooth with moving average
    window = 10
    smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    smooth_steps = steps[window//2 : window//2 + len(smooth)]

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.plot(steps, losses, color='#cccccc', linewidth=0.5, alpha=0.6, label='Raw')
    ax.plot(smooth_steps, smooth, color=C[0], linewidth=1.2, label='Smoothed')
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("DPO Loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    epoch_bounds = [0]
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_bounds.append(steps[i])
    for b in epoch_bounds[1:]:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_loss_curve.eps"), format='eps')
    plt.close()
    print(f"Saved fig2_loss_curve.eps ({len(steps)} raw pts, {len(smooth)} smoothed)")

# ====== Fig.3: LCSTS ======
methods = ['Base', 'SFT', 'DPO', 'KTO', 'MFRL(w/o MF)', 'MFRL']
rl = [0.162, 0.151, 0.158, 0.163, 0.169, 0.171]
bc = ['#b0c4de', '#4f81bd', '#9bbb59', '#f4a940', '#4bacc6', '#c0504d']

fig, ax = plt.subplots(figsize=(4.5, 2.8))
x = np.arange(len(methods))
bars = ax.bar(x, rl, width=0.55, color=bc, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, rl):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
bars[-1].set_edgecolor('#c0504d')
bars[-1].set_linewidth(2)
ax.set_ylim(0, 0.21)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=7, rotation=15)
ax.set_ylabel('ROUGE-L', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_lcsts_bars.eps"), format='eps')
plt.close()
print("Saved fig3_lcsts_bars.eps")

# ====== Fig.4: Alpaca ======
methods2 = ['SFT', 'DPO', 'MFRL']
rl2 = [0.122, 0.154, 0.151]
bc2 = ['#4f81bd', '#9bbb59', '#c0504d']

fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(len(methods2))
bars = ax.bar(x, rl2, width=0.45, color=bc2, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, rl2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.18)
ax.set_xticks(x)
ax.set_xticklabels(methods2, fontsize=10)
ax.set_ylabel('ROUGE-L', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_alpaca_bars.eps"), format='eps')
plt.close()
print("Saved fig4_alpaca_bars.eps")

# ====== Fig.5: Ablation ======
alabels = ['Full MFRL', 'w/o Model\nFeedback', 'w/o Adaptive\nFusion']
avals = [0.171, 0.169, 0.170]
ac = ['#c0504d', '#f4a940', '#9bbb59']

fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(len(alabels))
bars = ax.bar(x, avals, width=0.4, color=ac, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, avals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0015,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0.155, 0.178)
ax.set_xticks(x)
ax.set_xticklabels(alabels, fontsize=8)
ax.set_ylabel('ROUGE-L', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_ablation.eps"), format='eps')
plt.close()
print("Saved fig5_ablation.eps")

print("Done.")
