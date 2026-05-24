"""生成论文 EPS 图。依赖: pip install SciencePlots"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# SciencePlots style (no-latex for dvipdfmx compatibility)
try:
    plt.style.use(['science', 'no-latex', 'ieee'])
except:
    print("SciencePlots not found. Install: pip install SciencePlots")
    print("Falling back to manual style")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Muted professional palette
C0 = '#2c3e50'
C1 = '#3498db'
C2 = '#27ae60'
C3 = '#e74c3c'
C4 = '#f39c12'
C5 = '#8e44ad'
C6 = '#1abc9c'

# ====== Fig.2: Loss Curve ======
loss_path = os.path.join(OUT, "raw_loss.json")
if os.path.exists(loss_path):
    with open(loss_path) as f:
        data = json.load(f)
    train_data = [d for d in data if 'loss' in d and 'step' in d]
    steps = [d["step"] for d in train_data]
    losses = [d["loss"] for d in train_data]
    epochs = [d["epoch"] for d in train_data]

    window = 3
    smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    smooth_steps = steps[window//2 : window//2 + len(smooth)]

    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    ax.plot(steps, losses, 'o', markersize=2.5, color='#bdc3c7',
            alpha=0.7, markeredgewidth=0, label='Per-step')
    ax.plot(smooth_steps, smooth, '-', color=C1, linewidth=1.2, label='Smoothed')
    ax.set_xlabel("Training Step")
    ax.set_ylabel("DPO Loss")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fancybox=False, edgecolor='#ddd',
              fontsize=6.5, loc='upper right', handlelength=1.5, borderpad=0.4)
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.4)
    epoch_bounds = [0]
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_bounds.append(steps[i])
    for b in epoch_bounds[1:]:
        ax.axvline(x=b, color='#95a5a6', linestyle=':', alpha=0.4, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_loss_curve.eps"), format='eps')
    plt.close()
    print(f"Saved fig2_loss_curve.eps ({len(steps)} pts)")

# ====== Fig.3: LCSTS ======
methods = ['Base', 'SFT', 'DPO', 'KTO', 'MFRL(-MF)', 'MFRL']
rl = [0.162, 0.151, 0.158, 0.163, 0.169, 0.171]
colors = [C0, C1, C2, C4, C6, C3]

fig, ax = plt.subplots(figsize=(3.2, 2.0))
x = np.arange(len(methods))
bars = ax.bar(x, rl, width=0.55, color=colors, edgecolor='white', linewidth=0.3)
for bar, val in zip(bars, rl):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', fontsize=6.5, fontweight='bold', color='#2c3e50')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=7)
ax.set_ylabel("ROUGE-L")
ax.set_ylim(0, 0.21)
ax.grid(True, axis='y', linestyle='--', alpha=0.25, linewidth=0.4)
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
colors2 = [C1, C2, C3]

fig, ax = plt.subplots(figsize=(2.4, 1.6))
x = np.arange(len(methods2))
bars = ax.bar(x, rl2, width=0.45, color=colors2, edgecolor='white', linewidth=0.3)
for bar, val in zip(bars, rl2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
            f'{val:.3f}', ha='center', fontsize=8.5, fontweight='bold', color='#2c3e50')
ax.set_xticks(x)
ax.set_xticklabels(methods2, fontsize=9)
ax.set_ylabel("ROUGE-L")
ax.set_ylim(0, 0.18)
ax.grid(True, axis='y', linestyle='--', alpha=0.25, linewidth=0.4)
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
acolors = [C3, C4, C2]

fig, ax = plt.subplots(figsize=(2.4, 1.6))
x = np.arange(len(alabels))
bars = ax.bar(x, avals, width=0.4, color=acolors, edgecolor='white', linewidth=0.3)
for bar, val in zip(bars, avals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{val:.3f}', ha='center', fontsize=8.5, fontweight='bold', color='#2c3e50')
ax.set_xticks(x)
ax.set_xticklabels(alabels, fontsize=7)
ax.set_ylabel("ROUGE-L")
ax.set_ylim(0.155, 0.178)
ax.grid(True, axis='y', linestyle='--', alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_ablation.eps"), format='eps')
plt.close()
print("Saved fig5_ablation.eps")

# ====== Fig.1: Framework ======
fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

from matplotlib.patches import FancyBboxPatch
def dbox(x, y, w, h, title, sub='', color='#dce6f3', bold=False):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=4", facecolor=color, edgecolor='#4f81bd', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y+0.010 if sub else y, title, ha='center', va='center',
            fontsize=10 if bold else 8.5, fontweight='bold' if bold else 'normal', color='#1a3c6e')
    if sub:
        ax.text(x, y-0.025, sub, ha='center', va='center', fontsize=6.5, color='#4f81bd')

def darrow(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

def dlabel(x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=7, color='#555',
            bbox=dict(boxstyle='round,pad=2.5', facecolor='#fafafa', edgecolor='#ddd', linewidth=0.5))

y = 0.94
dbox(0.5, y, 0.32, 0.06, '输入: (x, y*)', bold=True)
darrow(0.5, y-0.03, y-0.09)
dlabel(0.72, y-0.06, '多温度采样\n拒绝滤波')

y2 = y - 0.13
dbox(0.5, y2, 0.38, 0.05, '候选集: {y1, y2, y3, y4}', color='#f5f5f5')
darrow(0.5, y2-0.025, y2-0.09)

y3 = y2 - 0.13
dbox(0.5, y3, 0.46, 0.07, 'MFC 多源反馈收集模块', bold=True)
dbox(0.22, y3, 0.16, 0.04, 'ROUGE规则反馈', color='#eef3fa')
dbox(0.78, y3, 0.16, 0.04, 'LogP模型自反馈', color='#eef3fa')
darrow(0.5, y3-0.035, y3-0.10)

y4 = y3 - 0.15
dbox(0.5, y4, 0.46, 0.07, 'AFF 自适应反馈融合模块', bold=True)
dbox(0.22, y4, 0.16, 0.04, '注意力权重', color='#eef3fa')
dbox(0.78, y4, 0.16, 0.04, '分数归一化', color='#eef3fa')
darrow(0.5, y4-0.035, y4-0.10)

y5 = y4 - 0.14
dlabel(0.5, y5, '偏好对: (x, y+, y-)')
darrow(0.5, y5-0.03, y5-0.10)

y6 = y5 - 0.14
dbox(0.5, y6, 0.46, 0.07, 'POF 偏好优化微调模块', bold=True)
dbox(0.22, y6, 0.16, 0.04, 'DPO损失', color='#eef3fa')
dbox(0.78, y6, 0.16, 0.04, 'LoRA (rank=8)', color='#eef3fa')
darrow(0.5, y6-0.035, y6-0.10)

y7 = y6 - 0.15
dbox(0.5, y7, 0.32, 0.06, '输出: 优化后模型', color='#4f81bd', bold=True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_framework.eps"), format='eps')
plt.close()
print("Saved fig1_framework.eps")

print("Done. pip install SciencePlots for prettier defaults.")
