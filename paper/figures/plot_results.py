"""生成MFRL论文数据图（柱状图）
使用 matplotlib，风格匹配电子学报。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 电子学报风格配色
COLORS = {
    'blue': '#4f81bd',
    'dark_blue': '#1a3c6e',
    'light_blue': '#b0c4de',
    'green': '#9bbb59',
    'dark_green': '#4bac6c',
    'orange': '#f4a940',
    'red': '#c0504d',
    'gray': '#d0d4dc',
    'dark_gray': '#555555',
    'text': '#333333',
}

# 通用设置 - 自动检测可用中文字体
import subprocess, re
try:
    # 检测系统可用中文字体
    result = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True, timeout=5)
    zh_fonts = re.findall(r'/([\w-]+\.\w+)', result.stdout)
    preferred = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']
    font_candidates = [f for f in preferred if any(f in s for s in zh_fonts)] + ['DejaVu Sans']
except:
    font_candidates = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': font_candidates,
    'font.size': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTPUT = os.path.dirname(os.path.abspath(__file__))


def fig2_lcsts_comparison():
    """图2：LCSTS数据集各方法ROUGE-L对比"""
    methods = ['Base', 'SFT', 'DPO', 'KTO', 'MFRL\n(w/o MF)', 'MFRL']
    values = [0.162, 0.151, 0.158, 0.163, 0.169, 0.171]
    bar_colors = [COLORS['light_blue'], COLORS['blue'], COLORS['green'],
                  COLORS['orange'], COLORS['dark_green'], COLORS['red']]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(methods))
    bars = ax.bar(x, values, width=0.55, color=bar_colors, edgecolor='white', linewidth=0.5)

    # 数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=COLORS['text'])

    # 最高柱子高亮
    bars[-1].set_edgecolor(COLORS['red'])
    bars[-1].set_linewidth(2)
    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2, bars[-1].get_height() + 0.008,
            '最佳', ha='center', va='bottom', fontsize=10, color=COLORS['red'], fontweight='bold')

    ax.set_ylabel('ROUGE-L', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # 网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    ax.set_axisbelow(True)

    # 边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#cccccc')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig2_lcsts_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig3_ablation():
    """图3：消融实验对比"""
    methods = ['等权融合\n(w/o AFF)', '仅规则反馈\n(w/o MF)', '完整 MFRL']
    values = [0.170, 0.169, 0.171]
    bar_colors = [COLORS['green'], COLORS['orange'], COLORS['red']]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    x = np.arange(len(methods))
    bars = ax.bar(x, values, width=0.45, color=bar_colors, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=COLORS['text'])

    bars[-1].set_edgecolor(COLORS['red'])
    bars[-1].set_linewidth(2)

    ax.set_ylabel('ROUGE-L', fontsize=13, fontweight='bold')
    ax.set_ylim(0.155, 0.178)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#cccccc')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig3_ablation.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


def fig4_alpaca_comparison():
    """图4：Alpaca-Chinese ROUGE-L对比"""
    methods = ['SFT', 'DPO', 'MFRL']
    values = [0.122, 0.154, 0.151]
    bar_colors = [COLORS['blue'], COLORS['green'], COLORS['red']]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    x = np.arange(len(methods))
    bars = ax.bar(x, values, width=0.45, color=bar_colors, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=COLORS['text'])

    bars[-1].set_edgecolor(COLORS['red'])
    bars[-1].set_linewidth(2)

    ax.set_ylabel('ROUGE-L', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#cccccc')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig4_alpaca_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    fig2_lcsts_comparison()
    fig3_ablation()
    fig4_alpaca_comparison()
    print('All figures generated.')
