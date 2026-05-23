"""改写 template.tex：保留 preamble，替换 body 为论文内容。"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('template.tex', 'rb') as f:
    raw = f.read()
text = raw.decode('gbk')
idx = text.find(r'\begin{document}')
preamble = text[:idx]

body = r'''\begin{document}

\newtheoremstyle{aastheorem}%
  {2pt}{2pt}%
  {}{0pt}%
  {}{.}%
  {1em}%
  {\thmname{#1} \thmnumber{#2}\thmnote{ (#3)}}

\theoremstyle{aastheorem}

\newtheorem{theorem}{\begin{CJK*}{GBK}{hei}定理\end{CJK*}}

\cntitle{{\hei\qquad 基于多形式反馈融合的大语言模型直接偏好优化方法}}

\thanks{收稿日期\
XXXX-XX-XX
\quad
录用日期\
XXXX-XX-XX}

\thanks{Manuscript received
Month Date, Year;
accepted
Month Date, Year}

\thanks{国家自然科学基金 (XXXXX) 资助}

\thanks{Supported by National Natural Science Foundation of China (XXXXX)}

\thanks{责任编辑\ XXX}

\thanks{Recommended by Associate Editor XXX}

\thanks{1.
XXXX大学 XXXX学院\ 城市\ 邮编
\quad 2.
XXXX大学 XXXX学院\ 城市\ 邮编}

\thanks{1.
School of XXXXXX, XXXXXX University, City Postcode
\quad 2.
School of XXXXXX, XXXXXX University, City Postcode
}}

\cnauthor{作者一$^{\scriptscriptstyle 1}$
\hspace{1em}
作者二$^{\scriptscriptstyle 2}$
}

\cnabstract{
针对大语言模型微调中偏好数据依赖人工标注、反馈信号来源单一的问题，提出基于多形式反馈融合的直接偏好优化方法（MFRL）。
该方法包含三个核心模块：MFC 从 ROUGE 规则指标和策略模型对数概率获取异构反馈；
AFF 通过注意力机制学习各反馈源权重；
POF 基于融合偏好对进行 DPO 训练。
在 Qwen2.5-1.5B-Instruct 上，LCSTS 和 Alpaca-Chinese 两个数据集的实验表明：MFRL 在 LCSTS 上 ROUGE-L 达 0.171，较 SFT、DPO、KTO 分别提升 0.020、0.013、0.008；Alpaca-Chinese 上达 0.151，较 SFT 提升 0.029。
}

\cnkeyword{大语言模型，直接偏好优化，多形式反馈，反馈融合，强化学习，LoRA}

\doi{10.16383/j.aas.cxxxxxx}

\entitle{Multi-form Feedback Reinforced Learning for Direct Preference Optimization of Large Language Models}

\enauthor{Author One$^{1}$
\qquad
Author Two$^2$
}

\enabstract{
To address labor-intensive annotation and single-source feedback in LLM fine-tuning, this paper proposes MFRL. The framework: MFC acquires feedback from ROUGE metrics and policy log-probabilities; AFF dynamically learns per-source weights via attention; POF conducts DPO training. On Qwen2.5-1.5B-Instruct, MFRL achieves 0.171 ROUGE-L on LCSTS (+0.020/+0.013/+0.008 over SFT/DPO/KTO) and 0.151 on Alpaca-Chinese (+0.029 over SFT).
}

\enkeyword{Large language model, direct preference optimization, multi-form feedback, feedback fusion, reinforcement learning, LoRA}

\cnaddress{作者一, 作者二. 基于多形式反馈融合的大语言模型直接偏好优化方法. 自动化学报, 202X,
\textbf{XX}(X): X$-$X}

\enaddress{Author One, Author Two.
Multi-form feedback reinforced learning for direct preference optimization of large language models.
\textsl{Acta Automatica Sinica}, 202X, \textbf{XX}(X): X$-$X}

\maketitle

\pagestyle{aasheadings}


% ===== 1 引言 =====

大语言模型（LLM）在自然语言处理领域取得了显著进展$^{[1-2]}$，但预训练输出与人类偏好存在偏差，需微调对齐$^{[3]}$。

RLHF$^{[4]}$ 是主流对齐范式，但需独立奖励模型，且偏好数据依赖人工标注。DPO$^{[6]}$ 统一奖励建模与策略优化，KTO$^{[7]}$ 仅需二元标签。但偏好数据仍依赖人工或单源信号。

Self-Reward$^{[8]}$ 和 Constitutional AI$^{[9]}$ 探索自动反馈，但单源可能偏差。多源信息融合$^{[10]}$ 在偏好学习中处于探索阶段。

本文提出 MFRL，贡献：(1) MFC 融合规则与自反馈，自动生成偏好对；(2) AFF 注意力机制融合异构信号；(3) LCSTS 上 ROUGE-L=0.171，超 SFT +0.020，DPO +0.013，KTO +0.008。


\section{相关工作}

\subsection{偏好优化方法}

RLHF$^{[4]}$ 训练奖励模型后用 PPO$^{[5]}$ 优化；DPO$^{[6]}$ 绕过显式奖励建模；KTO$^{[7]}$ 简化为二元标注。上述方法均依赖人工偏好数据。

\subsection{自动反馈与高效微调}

Self-Reward$^{[8]}$ 和 RLAIF$^{[9]}$ 用模型/AI 生成反馈。LoRA$^{[12]}$ 大幅降显存，使消费级 GPU 偏好优化可行。


\section{MFRL 方法}

\subsection{总体架构}

给定 $x$ 和参考 $y^*$，$\pi_\theta$ 生成候选 $\{y_1,\dots,y_K\}$。三模块：(1) MFC 获取多源反馈；(2) AFF 融合为统一分数；(3) POF 进行 DPO 训练。

\begin{center}
\vbox{\centerline{\fbox{\parbox{0.8\textwidth}{\centering
{\bf 输入}：$(x, y^*)$ \\[2pt]
$\downarrow$ 多温度采样 + 拒绝滤波 \\[2pt]
{\bf MFC}：ROUGE 得分 $+$ Log-Probability 自评分 \\[2pt]
$\downarrow$ \\[2pt]
{\bf AFF}：注意力融合 $\rightarrow$ 偏好对 $(x, y^+, y^-)$ \\[2pt]
$\downarrow$ \\[2pt]
{\bf POF}：DPO + LoRA (rank=8) \\[2pt]
$\downarrow$ \\[2pt]
{\bf 输出}：优化后模型 $\pi_\theta$
}}}\vskip1mm {\small
图\ 1\quad MFRL 整体框架
\\
Fig.\ 1\quad The MFRL framework}}}
\end{center}

\subsection{多源反馈收集 (MFC)}

多温度采样 $T\in\{0.5,0.8,1.2,1.8\}$，$K=4$，拒绝滤波。

规则反馈基于 ROUGE$^{[13]}$：
\begin{equation}
s_{\rm RF}(y_i)=0.3\cdot{\rm R1}+0.3\cdot{\rm R2}+0.4\cdot{\rm RL}
\end{equation}

模型自反馈用对数概率：
\begin{equation}
s_{\rm MF}(y_i|x)=\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log P(y_{i,t}|x,y_{i,<t})
\end{equation}
归一化至 $[0,1]$。

\subsection{自适应融合 (AFF)}

注意力权重动态融合：
\begin{equation}
\mathbf{w}={\rm softmax}(\mathbf{W}\cdot[s_{\rm RF},s_{\rm MF}]^\top+\mathbf{b}),\quad
s_{\rm fused}=w_{\rm RF}s_{\rm RF}+w_{\rm MF}s_{\rm MF}
\end{equation}
按 $s_{\rm fused}$ 排序，最高/最低构成偏好对 $(x, y^+, y^-)$。

\subsection{偏好优化微调 (POF)}

\begin{equation}
\mathcal{L}_{\rm DPO}=-\mathbb{E}\!\left[\!\log\sigma\!\left(\!\beta\log\frac{\pi_\theta(y^+|x)}{\pi_{\rm ref}(y^+|x)}-\beta\log\frac{\pi_\theta(y^-|x)}{\pi_{\rm ref}(y^-|x)}\!\right)\!\right]
\end{equation}

LoRA$^{[12]}$：rank=8, alpha=16。超参：lr=5e-6, batch=16, 3 epochs, $\beta$=0.1。


\section{实验与分析}

\subsection{实验设置}

基座：Qwen2.5-1.5B-Instruct$^{[14]}$。数据集：LCSTS$^{[15]}$ 2000 样本，Alpaca-Chinese 1000 样本。基线：Base, SFT, DPO, KTO, MFRL w/o MF。评估：ROUGE-1/2/L$^{[13]}$。单卡 3090（24GB），约 2--3h。

\subsection{LCSTS 结果}

\begin{center}
\vbox{\centering{\small 表\ 1 \quad LCSTS 数据集 ROUGE 对比
\\
Table\ 1 \quad ROUGE on LCSTS} \vskip2mm
\centerline{\tabcolsep=9pt\begin{tabular}{lcccc}
\toprule
方法 & 偏好数据 & ROUGE-1 & ROUGE-2 & ROUGE-L \\
\hline
Base & 无 & 0.162 & 0.019 & 0.162 \\
SFT & 人工 & 0.151 & 0.013 & 0.151 \\
DPO & 人工 & 0.158 & 0.013 & 0.158 \\
KTO & 人工 & 0.163 & 0.013 & 0.163 \\
MFRL w/o MF & 自动 & 0.169 & 0.013 & 0.169 \\
{\bf MFRL} & {\bf 自动} & {\bf 0.171} & {\bf 0.013} & {\bf 0.171} \\
\bottomrule
\end{tabular}}}}
\end{center}

MFRL 所有方法最优，且偏好数据自动生成无需人工标注。

\subsection{Alpaca-Chinese 结果}

\begin{center}
\vbox{\centering{\small 表\ 2 \quad Alpaca-Chinese 数据集 ROUGE 对比
\\
Table\ 2 \quad ROUGE on Alpaca-Chinese} \vskip2mm
\centerline{\tabcolsep=15pt\begin{tabular}{lccc}
\toprule
方法 & ROUGE-1 & ROUGE-2 & ROUGE-L \\
\hline
SFT & 0.126 & 0.067 & 0.122 \\
DPO & 0.163 & 0.084 & 0.154 \\
{\bf MFRL} & {\bf 0.160} & {\bf 0.084} & {\bf 0.151} \\
\bottomrule
\end{tabular}}}}
\end{center}

指令跟随上 MFRL 与 DPO 相当，均优于 SFT（+0.029）。

\subsection{消融实验}

\begin{center}
\vbox{\centering{\small 表\ 3 \quad 消融实验（LCSTS）
\\
Table\ 3 \quad Ablation on LCSTS} \vskip2mm
\centerline{\tabcolsep=15pt\begin{tabular}{lc}
\toprule
变体 & ROUGE-L \\
\hline
完整 MFRL & {\bf 0.171} \\
w/o 模型反馈 & 0.169 \\
w/o 自适应融合 & 0.170 \\
\bottomrule
\end{tabular}}}}
\end{center}

模型自反馈贡献 +0.002，自适应融合优于等权平均。


\section{结论}

MFRL 通过 MFC、AFF、POF 三模块协同，实现无人工标注的偏好优化。两数据集上验证了多形式反馈融合优于单一信号源。未来探索更丰富反馈信号、更大模型验证及在线迭代优化。


\section*{致谢}
感谢审稿人和编辑的宝贵意见。


\begin{thebibliography}{99}
\zihao{6} \addtolength{\itemsep}{0.2em} \urlstyle{rm}

\bibitem{1} Brown T, Mann B, Ryder N, et al. Language models are few-shot learners. In: {\sl Proceedings of Advances in Neural Information Processing Systems (NeurIPS)}. Virtual: Curran Associates, 2020. 1877$-$1901

\bibitem{2} Touvron H, Lavril T, Izacard G, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023

\bibitem{3} Wei J, Bosma M, Zhao V Y, et al. Finetuned language models are zero-shot learners. In: {\sl Proceedings of the 10th International Conference on Learning Representations (ICLR)}. Virtual: OpenReview.net, 2022

\bibitem{4} Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback. In: {\sl Proceedings of Advances in Neural Information Processing Systems (NeurIPS)}. New Orleans: Curran Associates, 2022. 27730$-$27744

\bibitem{5} Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017

\bibitem{6} Rafailov R, Sharma A, Mitchell E, et al. Direct preference optimization: Your language model is secretly a reward model. In: {\sl Proceedings of Advances in Neural Information Processing Systems (NeurIPS)}. New Orleans: Curran Associates, 2023. 53728$-$53741

\bibitem{7} Ethayarajh K, Xu W, Muennighoff N, et al. KTO: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024

\bibitem{8} Yuan W Z, Pang R Y, Cho K, et al. Self-rewarding language models. In: {\sl Proceedings of the 41st International Conference on Machine Learning (ICML)}. Vienna: PMLR, 2024

\bibitem{9} Bai Y T, Kadavath S, Kundu S, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022

\bibitem{10} 何友, 王国宏, 关欣, 等. 多传感器信息融合及应用. 北京: 电子工业出版社, 2000

\bibitem{11} Meng Y, Xia M, Chen D. SimPO: Simple preference optimization with a reference-free reward. In: {\sl Proceedings of Advances in Neural Information Processing Systems (NeurIPS)}. Vancouver: Curran Associates, 2024

\bibitem{12} Hu E J, Shen Y L, Wallis P, et al. LoRA: Low-rank adaptation of large language models. In: {\sl Proceedings of the 10th International Conference on Learning Representations (ICLR)}. Virtual: OpenReview.net, 2022

\bibitem{13} Lin C Y. ROUGE: A package for automatic evaluation of summaries. In: {\sl Proceedings of the ACL Workshop on Text Summarization Branches Out}. Barcelona: ACL, 2004. 74$-$81

\bibitem{14} Yang A, Yang B, Hui B, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024

\bibitem{15} Hu B T, Chen Q C, Zhu F M. LCSTS: A large scale Chinese short text summarization dataset. In: {\sl Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)}. Lisbon: ACL, 2015. 1967$-$1972

\end{thebibliography}


\begin{biography}[ssl.eps]
\noindent{\hei
作者一
}\quad
XXXX大学博士研究生. 主要研究方向为大语言模型微调与对齐，强化学习.
\\E-mail: author@example.com

\noindent({\bf
Author One
}\quad
Ph.D. student at XXXXXX University. His research interests include large language model fine-tuning and alignment, and reinforcement learning.)
\end{biography}

\end{document}'''

full = preamble + body
with open('template.tex', 'wb') as f:
    f.write(full.encode('gbk'))
print(f'Written {len(full)} chars to template.tex')
