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
loss_path = os.path.join(OUT, "raw_loss.json")
if os.path.exists(loss_path):
    with open(loss_path) as f:
        data = json.load(f)

    # Filter training entries (with loss field)
    train_data = [d for d in data if 'loss' in d and 'step' in d]
    steps = [d["step"] for d in train_data]
    losses = [d["loss"] for d in train_data]
    epochs = [d["epoch"] for d in train_data]

    # Light smooth (window=3 for 30 points)
    window = 3
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window//2 : window//2 + len(smooth)]
    else:
        smooth = losses
        smooth_steps = steps

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.scatter(steps, losses, color='#cccccc', s=15, alpha=0.7, label='Per-step')
    ax.plot(smooth_steps, smooth, color=C[0], linewidth=1.2, label='Smoothed')
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("DPO Loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    # Epoch boundaries
    epoch_bounds = [0]
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_bounds.append(steps[i])
    for b in epoch_bounds[1:]:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_loss_curve.eps"), format='eps')
    plt.close()
    print(f"Saved fig2_loss_curve.eps ({len(steps)} pts, window={window})")

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

# ====== Fig.5: Ablation (Chinese labels) ======
alabels = ['完整MFRL', '无模型反馈', '无自适应融合']
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
ax.set_xticklabels(alabels, fontsize=9)
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

# ====== Fig.1: Framework Diagram (Chinese) ======
fig, ax = plt.subplots(figsize=(5.5, 4.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

from matplotlib.patches import FancyBboxPatch
def dbox(x, y, w, h, title, sub='', color='#dce6f3', bold=False):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=5", facecolor=color, edgecolor='#4f81bd', linewidth=1.5)
    ax.add_patch(box)
    fs = 11 if bold else 9
    ax.text(x, y+0.012 if sub else y, title, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal')
    if sub:
        ax.text(x, y-0.028, sub, ha='center', va='center', fontsize=7, color='#4f81bd')

def darrow(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

def dlabel(x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5, color='#555',
            bbox=dict(boxstyle='round,pad=3', facecolor='#fafafa', edgecolor='#ddd', linewidth=0.5))

y = 0.94
dbox(0.5, y, 0.35, 0.06, '输入: (x, y*)', bold=True)
darrow(0.5, y-0.03, y-0.10)
dlabel(0.73, y-0.065, '多温度采样\n+ 拒绝滤波')

y2 = y - 0.14
dbox(0.5, y2, 0.4, 0.06, '候选: {y1, y2, y3, y4}', color='#f5f5f5')
darrow(0.5, y2-0.03, y2-0.10)

y3 = y2 - 0.14
dbox(0.5, y3, 0.48, 0.07, 'MFC', '多源反馈收集模块', bold=True)
dlabel(0.22, y3, 'ROUGE\n规则反馈')
dlabel(0.78, y3, 'LogP\n模型自反馈')
darrow(0.5, y3-0.035, y3-0.11)

y4 = y3 - 0.16
dbox(0.5, y4, 0.48, 0.07, 'AFF', '自适应反馈融合模块', bold=True)
dlabel(0.22, y4, '注意力\n权重学习')
dlabel(0.78, y4, '分数\n归一化')
darrow(0.5, y4-0.035, y4-0.11)

y5 = y4 - 0.15
dlabel(0.5, y5, '偏好对: (x, y+, y-)')
darrow(0.5, y5-0.04, y5-0.11)

y6 = y5 - 0.15
dbox(0.5, y6, 0.48, 0.07, 'POF', '偏好优化微调模块', bold=True)
dlabel(0.22, y6, 'DPO损失')
dlabel(0.78, y6, 'LoRA (rank=8)')
darrow(0.5, y6-0.035, y6-0.11)

y7 = y6 - 0.15
dbox(0.5, y7, 0.35, 0.06, '输出: 优化后模型',
     color='#4f81bd', bold=True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_framework.eps"), format='eps')
plt.close()
print("Saved fig1_framework.eps")

print("Done.")
