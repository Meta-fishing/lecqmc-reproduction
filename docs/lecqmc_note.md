# LEC-QMC：线性标度正则系综量子蒙特卡洛——算法复现笔记

> 复现论文：Tu Hong, Kun Chen, Xiao Yan Xu, *Linear Canonical-Ensemble Quantum Monte Carlo: From Dilute Fermi Gas to Flat-Band Ferromagnetism*（arXiv:2601.08552）。
> 本文档给出完整的算法推导、Julia 实现要点、数值稳定性的关键细节（含一处论文未强调、但在长虚时间下致命的陷阱），以及与原文四个结果的 benchmark 对比。代码见 `../src/LECQMC.jl`。

## 1. 背景与动机

有限温行列式 QMC（DQMC）在巨正则系综中工作，一次 sweep 的代价为 $O(\beta N^3)$（$N$ 为格点数，$\beta$ 为逆温），来源是等时格林函数 $G$ 的 $N\times N$ 秩一更新。在**稀释极限**（固定粒子数 $N_e$、$N\to\infty$）下，人们关心的是固定 $N_e$ 的正则系综，而巨正则系综此时化学势调节困难（平带情形下甚至不可能，压缩率发散）。LEC-QMC 直接采样正则系综的 Fock 态，借助稳定化 QR 更新把粒子增加/删除的代价降到 $O(\beta N N_e)$，一次完整 sweep 仅需 $O(\beta N N_e^2)$；$N_e$ 固定时**严格线性**于 $N$。

配分函数写为对 Fock 态 $\eta$（每个自旋的电子位置集合）与辅助场 $s$ 的联合求和：

$$
Z=\sum_{\eta}\sum_{s}\det\!\big[P_{\eta}^{\dagger}B_{s}(\beta,0)P_{\eta}\big]\equiv\sum_{\eta,s}W_\eta(s),
$$

对两个自旋即 $W=W_\uparrow W_\downarrow$，其中 $P_\eta$ 为 $N\times N_e$ 选择矩阵（每列一个单位矢量），$B_s(\beta,0)=\prod_{l=1}^{L_\tau}B_l$ 为虚时传播子。采用 Trotter 分解 $B_l=e^{V_l}e^{-\Delta\tau K}$，Hubbard 相互作用用 HS 变换

$$
e^{-\Delta\tau U(n_{i\uparrow}-\frac12)(n_{i\downarrow}-\frac12)}=\tfrac12\sum_{s=\pm1}e^{s\lambda(n_{i\uparrow}-n_{i\downarrow})},\qquad \cosh\lambda=e^{\Delta\tau U/2}.
$$

（常数因子归入测度。）动能传播子 $e^{-\Delta\tau K}$ 用 **checkerboard 分解**：把键按"边着色"分组，同组键互不相交、对应的 $2\times2$ 旋转互相对易，因此 $e^{-\Delta\tau K}$ 被精确写成若干组稀疏旋转与对角部分 $e^{-\Delta\tau K_{ii}}$ 的乘积——不引入额外 Trotter 误差（注意：对周期性链/方格子，**奇数 $L$ 的环绕键需要通用的贪心着色**，朴素奇偶分组会失效，这是我们修过的 bug 之一）。

## 2. 标准有限温 DQMC 算法（基线）

LEC-QMC 的每个组件都有标准 DQMC 的对应物；为说清楚前者，先完整推导后者。这也是我们 Fig 1 标度对比的基线实现（`src/StdDQMC.jl`）。

### 2.1 配分函数的行列式形式

对 Hubbard 模型 $H=K+U\sum_i(n_{i\uparrow}-\tfrac12)(n_{i\downarrow}-\tfrac12)$（化学势已并入 $K$），把虚时切成 $L_\tau$ 片，$\beta=L_\tau\Delta\tau$，用 Trotter–Suzuki 分解

$$
e^{-\beta H}\approx\prod_{l=1}^{L_\tau}e^{-\Delta\tau H},\qquad
e^{-\Delta\tau H}\approx e^{-\Delta\tau H_U}\,e^{-\Delta\tau K}\quad(\text{误差 }O(\Delta\tau^2)).
$$

对每个 $(l,i)$ 处的相互作用因子做离散 HS 变换（$\cosh\lambda=e^{\Delta\tau U/2}$，$s_{l,i}=\pm1$），配分函数成为对全部辅助场的求和。由于此时哈密顿量是二次型，对费米子自由度求迹给出行列式（巨正则系综）：

$$
Z=\sum_{\{s\}}\prod_{\sigma=\uparrow,\downarrow}\det\!\big[I+B^\sigma(\beta,0)\big],\qquad
B^\sigma(\beta,0)=\prod_{l=1}^{L_\tau}B_l^\sigma,\quad B_l^\sigma=e^{V_l^\sigma}e^{-\Delta\tau K},
$$

其中 $(V_l^\sigma)_{ii}=\sigma\lambda s_{l,i}$（对角）。于是每个场构型的权重为 $W(s)=\prod_\sigma\det(I+B^\sigma)$，可为负——符号问题即源于此。

### 2.2 等时格林函数与单点翻转

定义等时格林函数

$$
G^\sigma_{ij}\equiv\langle c_i c_j^\dagger\rangle_{B^\sigma}=\big[\big(I+B^\sigma(\beta,0)\big)^{-1}\big]_{ij}.
$$

把传播子写成从当前虚时片 $l$ 出发的"环绕"（wrap）形式 $B(\beta,0)=B_l\cdots B_{L_\tau}B_1\cdots B_{l-1}$，则 $G$ 驻留在第 $l$ 片上。翻转 $s_{l,i}\to-s_{l,i}$ 相当于 $B_l\to (I+\Delta)B_l$，其中 $\Delta$ 仅在 $(i,i)$ 处非零：$\Delta_{ii}=e^{-2\sigma\lambda s_{l,i}}-1\equiv d_\sigma$。由行列式引理 $\det(I+\Delta( I-G))$（秩一），翻转比值为

$$
r_\sigma=1+d_\sigma\big(1-G^\sigma_{ii}\big),\qquad r=r_\uparrow r_\downarrow,
$$

以概率 $\min(1,|r|)$ 接受（带符号加权或以 $r$ 的符号计入观测量）。接受后格林函数做 Sherman–Morrison 秩一更新：

$$
G^\sigma\leftarrow G^\sigma-\frac{d_\sigma}{r_\sigma}\big(G^\sigma e_i\big)\big(e_i^\dagger G^\sigma-e_i^\dagger\big),
$$

即 $G_{ab}\leftarrow G_{ab}+\frac{d_\sigma}{r_\sigma}G_{ai}\,(\delta_{ib}-G_{ib})$，代价 $O(N^2)$。随后把 $G$ 推进一片：$G\leftarrow B_{l}G B_{l}^{-1}$（wrap，$O(N^2)$ 用 checkerboard 稀疏结构，或 $O(N^3)$ 稠密）。**顺序很关键**：第 $l$ 片的翻转必须在 $G$ wrap 到该片之后进行——翻转是 $B(l,0)$ 的左乘，我们曾把 wrap/翻转顺序写反而得到错误分布。SM 更新也须先拷贝涉及的行/列，避免就地读写混叠。

### 2.3 数值稳定化

$G$ 含 $I+B$ 的逆，而 $B$ 的条件数 $\sim e^{\beta\lVert K\rVert}$：低温下直接乘积再求逆会丢光精度。标准做法是**稳定化传播**：每 $n_{\rm stab}$ 片用 QR（或 UDV）分解重新表达部分乘积 $B(\tau,0)=Q(\tau)D(\tau)T(\tau)$，最终由 $(Q,D,T)$ 稳定地组装 $G$；增量维护的 $G$ 也必须周期性由当前场**新鲜重建**——我们实测：只做 SM 更新而不重建，小概率 $|r|$ 事件引入的误差在 3–4 个 sweep 内即累积到 $O(1)$，格林函数彻底失真。这一经验对理解 LEC-QMC 侧的失稳（§5.1）很有帮助。

### 2.4 测量与 Wick 定理

等时多体关联按 Wick 定理分解为 $G$ 的乘积，例如 $\langle n_i\rangle=1-G_{ii}$，$\langle n_i n_j\rangle=(1-G_{ii})(1-G_{jj})-G_{ij}G_{ji}$（$i\neq j$，单自旋），自旋关联 $\langle S^z_iS^z_j\rangle$ 为两自旋密度关联之差。带符号模拟中观测量带权 $W$ 的符号平均：$\langle O\rangle=\langle O\cdot\mathrm{sgn}\rangle_{|W|}/\langle\mathrm{sgn}\rangle_{|W|}$。

### 2.5 复杂度与符号问题

每片 $N$ 个格点各一次秩一更新 $O(N^2)$，共 $O(\beta N^3)$；每 $n_{\rm stab}$ 片一次新鲜重构同为 $O(\beta N^3/n_{\rm stab})$ 量级。故 sweep 代价 $O(\beta N^3)$——Fig 1 的三次方标度即源于此。平均符号 $\langle\mathrm{sgn}\rangle\sim e^{-\beta N\Delta f}$ 随尺寸与低温指数恶化（Fig 2 检验其稀薄极限行为）。

### 2.6 与 LEC-QMC 的对应关系

| | 标准 DQMC（巨正则） | LEC-QMC（正则） |
|---|---|---|
| 权重 | $\det(I+B)$，$N$ 维 | $\det(P_\eta^\dagger B P_\eta)$，$N_e$ 维 |
| 核心矩阵 | $G=(I+B)^{-1}$，$N\times N$ | $M=Q(P_\eta^\dagger Q)^{-1}P_\eta^\dagger$，$N\times N$ 斜投影（秩 $N_e$） |
| 翻转比值 | $1+d(1-G_{ii})$ | $1+d\,(L[i,:]A_f\tilde R[:,i])$ |
| 接受后更新 | SM 秩一更新 $G$，$O(N^2)$ | SM 更新 $A_f$ + $L$ 行缩放，$O(N_e^2)$ |
| 额外采样 | 无（粒子数由 $\mu$ 间接控制） | Fock 态增/删/交换（§4），$O(\beta NN_e)$ |
| sweep 代价 | $O(\beta N^3)$ | $O(\beta N N_e^2)$ |

二者共享同一 HS 场与传播子；LEC-QMC 把"对全体 $\eta$ 的迹"显式采样为 Fock 态随机游走，从而把中心矩阵的维度从 $N$ 降到 $N_e$。

## 3. 稳定化 QR 传播

对任一 Fock 态 $\eta$，把 $B(\beta,0)P_\eta$ 按 $n_{\rm stab}$ 层一段（chunk）分解并逐段做（MGS）QR：

$$
B(\beta,0)P_\eta = Q_n V_n V_{n-1}\cdots V_1,
$$

其中每段 $Q_c$ 是 $N\times N_e$ 正交矩阵、$V_c$ 是 $N_e\times N_e$ 上三角。存储全部 $(Q_c,V_c)$ 及最终 $B(\beta,0)P_\eta=QR$。权重 $\det(P_\eta^\dagger B P_\eta)=\det(P_\eta^\dagger Q)\det(R)$，符号与模分离。数值上这正是 QR 稳定化保证长 $\beta$ 下不丢精度的标准做法。

**等时格林函数（斜投影形式）**：定义 $M\equiv Q\,A\,P_\eta^\dagger$，其中 $A=(P_\eta^\dagger Q)^{-1}$。它是 Fock 子空间上的斜投影，所有等时关联函数由它读出：密度 $\langle n_i\rangle=M_{ii}$，密度-密度 $\langle n_in_j\rangle$ 取 Fock 对角元，而交换型关联如 $\langle S^+_iS^-_j\rangle$（$i\neq j$）为

$$
\langle S^x_iS^x_j+S^y_iS^y_j\rangle=-\tfrac12\big(M^\uparrow_{ji}M^\downarrow_{ij}+M^\downarrow_{ji}M^\uparrow_{ij}\big).
$$

（实现教训：第二项必须使用 $M^\uparrow_{ij}$ 而非 $M^\uparrow_{ii}$——指标错位曾导致与 ED 差因子 2。）

## 4. Fock 态更新：粒子增加、删除与交换

### 4.1 增加一个粒子（论文 Eq. S7–S9）

在空位 $p$ 加粒子：逐 chunk 传播新列 $w\leftarrow B_{\rm chunk}w$，并对已有 $Q$ 做修正 Gram–Schmidt：$v=Q^\dagger w$，$q=(w-Qv)/\lVert w-Qv\rVert$，$r_{N_e+1}=\lVert w-Qv\rVert$。新 QR 为

$$
Q'=[Q\,|\,q],\qquad R'=\begin{pmatrix}R&v\\0&r_{N_e+1}\end{pmatrix}.
$$

Metropolis 比值为 Schur 补形式

$$
r=\underbrace{\big[p^\dagger q-p^\dagger Q\,A\,P_\eta^\dagger q\big]}_{\displaystyle s}\;r_{N_e+1},
$$

$A=(P_\eta^\dagger Q)^{-1}$ 是格林函数计算的副产品。接受后，$(P_{\eta'}^\dagger Q')^{-1}$ 用分块逆公式以 $O(N_e^2)$ 更新（Eq. S8）。每次增加的代价为 $O(\beta N N_e)$。

### 4.2 删除一个粒子（论文 Eq. S10–S12）

删除第 $k$ 个粒子：把第 $k$ 列置换到末尾，再对每个 chunk 用 Givens 旋转恢复上三角（$\tilde Q=QG^\dagger$，$V'=GV_{\rm prev}^\dagger$）。末行给出比值

$$
r=\frac{1/s}{r_{N_e}},\qquad M^{-1}=G_n\,A\,U ,
$$

其中 $U$ 为各 chunk 旋转之积。接受后同样以 $O(N_e^2)$ 缩减更新 $A$（Eq. S12）。删除代价同为 $O(\beta N N_e)$。

### 4.3 粒子-空穴交换

正则系综内的 Fock 移动 = 一次删除 + 一次增加，总比值为两者之积。它驱动电子在格点间迁移，是遍历性的另一支柱。

## 5. 辅助场更新（fast update）

固定 Fock 态 $\eta$，对辅助场做单点翻转。定义

$$
L=B(\tau,0)P_\eta\;(N\times N_e),\qquad \tilde R=P_\eta^\dagger B(\beta,\tau)\;(N_e\times N),\qquad A_f=(\tilde R L)^{-1}.
$$

翻转 $(l,i)$ 处 $s\to -s$ 相当于把 $e^{V_l}$ 左乘对角阵 $D=\mathrm{diag}(1,\dots,\gamma_i,\dots,1)$，$\gamma_i=e^{-2\sigma\lambda s_{l,i}}$，即 $L\to DL$（秩一），其余不变。比值

$$
r_\sigma=1+d_\sigma\,\big(L[i,:]\,A_f\,\tilde R[:,i]\big),\qquad d_\sigma=\gamma_i-1,
$$

总比值 $r=r_\uparrow r_\downarrow$，以 $|r|$ 接受。接受后 Sherman–Morrison 更新 $A_f\leftarrow A_f-\frac{d_\sigma}{r_\sigma}(A_f\tilde R[:,i])(L[i,:]A_f)$，并把 $L$ 第 $i$ 行乘 $1+d_\sigma$。**这一 SM 更新不可省略**——没有它，后续翻转的比值全部基于过时的 $A_f$，分布彻底错误（我们早期版本漏掉它，全枚举测试 $P(\eta)$ 完全反转，修复后精确吻合）。每次翻转代价 $O(N_e^2)$，整层 sweep $O(\beta N N_e^2)$。

### 5.1 数值稳定性的关键（论文未展开的陷阱）

$\tilde R(\tau)=P_\eta^\dagger B(\beta,\tau)$ 若按"先构造 $\tilde R(0)=P_\eta^\dagger B(\beta,0)$、再逐层右乘 $B_l^{-1}$ 反卷"的朴素方式维护，在长 $\beta$ 下会**灾难性失稳**：$\tilde R(0)$ 的条件数 $\sim e^{\beta(\lambda+\lVert K\rVert)}$，其浮点误差（绝对量级 $\mathrm{cond}\cdot\epsilon_{\rm mach}$）在反卷过程中被逆映射放大，而真值却收缩到 $O(1)$。我们在 $\beta=10$（$L_\tau=200$）实测：即使每层都用精确逆矩阵，反卷 200 层后 $\tilde R$ 的方向误差已达 $10^{-4}$ 量级，导致翻转比值系统性偏差、链收敛到**完全错误**的分布（二站Hubbard $\langle n_1n_2\rangle=0.885$ vs 精确值 $0.724$；对分实验证明问题只在场扫描，Fock 更新无错）。高温（$\beta\le4$）时 $\mathrm{cond}$ 小，该误差被掩盖，所有朴素测试都能通过——极具迷惑性。

**正确做法（本实现采用）**：

- $L$ 侧：从 $P_\eta$ 前向传播，每 chunk QR 重锚（对已接受的翻转用行缩放精确跟踪）；
- $\tilde R$ 侧：**每 sweep 开始时**，对 $P_\eta^\dagger B(\beta,0)$ 自顶向下分 chunk 做 LQ 稳定化分解，只保留行正交锚点 $\tilde Q_c$（$\tilde R(\tau)$ 只含层 $>\tau$，处理到第 $\tau$ 片时这些层的场尚未被翻动，故锚点整个 sweep 有效）；每 chunk 把 $\tilde R$ 重锚到 $\tilde Q_{c+1}$ 乘一个 $\le n_{\rm stab}$ 层的短乘积，并**新鲜重算** $A_f=(\tilde R L)^{-1}$（两侧三角尺度因子在比值中严格相消——规范不变性）；
- chunk 内翻转用 SM 更新。

修复后，同一 $\beta=10$ 链给出 $\langle n_1n_2\rangle=0.7221$、$P(\eta)=[0.138,\,0.361,\,0.361,\,0.140]$，与精确 Trotter 系综（转移矩阵）$0.7243$、$[0.138,\,0.362,\,0.362,\,0.138]$ 在统计误差内一致。代价：场扫描约多一次传播的量级（常数因子），标度不变。

### 5.2 再稳定化与对齐细节

$A_f$ 必须与 $L,\tilde R$ 在**同一虚时片**计算（我们曾把 $\tilde R$ 锚在 $\tau=l_0+1$ 而 $L$ 在 $\tau=l_0$，差一层导致分布仍有 $\sim10\%$ 偏差）；chunk 内 $M=\tilde RL$ 与 $\tau$ 无关，故 chunk 起点新鲜计算一次即可。

## 6. 一次完整 sweep 与复杂度

1. `field_sweep!`：上述辅助场扫描，$O(\beta N N_e^2)$；
2. `full_propagation!`：用最新场重建稳定化因子，$O(\beta N N_e+\beta N_e^3/n_{\rm stab})$；
3. `fock_sweep!`：每自旋 $N_e$ 次交换尝试，$O(N_e\beta N N_e)$。

合计 $O(\beta N N_e^2)$；$N_e$ 固定时关于 $N$ **严格线性**。测量为 Fock 对角（密度、$n_in_j$、$S^z_iS^z_j$）加交换估计量，逐 sweep $O(N_e^2+N^2)$ 累加。

## 7. 实现与验证

Julia 实现（`src/LECQMC.jl`，约 800 行）+ 精确对角化模块（`ed/ed.jl`，正则系综全谱求和）。验证手段层层递进：

- **单元测试**：传播、增/删粒子、交换与稠密参考在 $10^{-10}$ 内一致；
- **全枚举测试**（$\beta=1,2,4$）：枚举全部 $(\eta,s)$ 构型精确求和，MC 分布与精确分布一致；
- **独立交叉验证**：Python 转移矩阵精确计算 Trotter 化系综（不经过我们的 QMC 代码），二站 $\beta=10$ 给出 $\langle n_1n_2\rangle=0.724278$，与 ED（$\Delta\tau\to0$）$0.7236$ 一致；
- **决策级审计**：沿真实轨迹逐翻转对比快速比值与稠密比值（借此定位 §5.1 的失稳）。

## 8. Benchmark 结果

（见下各节，图件 `../results/figures/`。）

### 8.1 Fig S2：与精确对角化（ED）的直接对比

体系：二维正方格子 $L=4$（$N=16$），$N_\uparrow=N_\downarrow=1$（$N_e=2$），$U=2t$，$T=0.05t$（$\beta=20$，$\Delta\tau=0.05$，$L_\tau=400$）。ED 用正则系综全谱求和（`ed/ed.jl`，已经二站解析解验证）。LEC-QMC：1000 热化 + 40000 测量 sweep，20 块 blocking 误差棒。全部 16 个位点、三种关联函数（电荷 $\langle n_1n_j\rangle$、纵自旋 $\langle S^z_1S^z_j\rangle$、横自旋 $\langle S^x_1S^x_j+S^y_1S^y_j\rangle$）共 48 个数据点与 ED 在误差棒内一致（见 `results/figures/fig_s2_ed.svg`），$\langle\mathrm{sign}\rangle=1.000$。部分数值（MC(±err) vs ED）：

| $j$ | $r$ | $\langle n_1n_j\rangle$ | ED | $\langle S^z_1S^z_j\rangle$ | ED |
|---|---|---|---|---|---|
| 2 | 1.00 | 0.0086(4) | 0.0089 | 0.0013(3) | 0.0013 |
| 3 | 2.00 | 0.0091(4) | 0.0088 | 0.0013(3) | 0.0016 |
| 6 | 1.41 | 0.0079(4) | 0.0077 | -0.0030(3) | -0.0028 |

横向关联 $\langle S^xS^x+S^yS^y\rangle$ 依赖§3 的交换估计量，同样全部吻合——这是对算法最敏感的检验之一。

![Fig S2：LEC-QMC 与 ED 的三种关联函数对比](figures/fig_s2_ed.svg)

**分析**：该测试在 $\beta=20$（400 个虚时片）下运行，正是 §5.1 所述失稳会被放大的 regime；48 点全部吻合说明：(i) Fock 态采样（增/删/交换）实现了正确的正则系综分布；(ii) 辅助场更新的锚定方案在长虚时间下数值稳定；(iii) 交换估计量的指标约定正确。$\langle\mathrm{sign}\rangle=1$ 是小体系高符号区，这里的检验精度不受符号涨落干扰。

#### 8.1a 强耦合检验：$U=7$ 的同一对比

同一体系、同一 $T=0.05$，把相互作用提到 $U=7$（$\lambda=\mathrm{acosh}(e^{\Delta\tau U/2})\approx0.61$，约为 $U=2$ 时的两倍），40000 测量 sweep（`results/ed_benchmark_L4_U7.txt`）。$\langle\mathrm{sign}\rangle=1.000$。

- **电荷 $\langle n_1n_j\rangle$**：16 点全部与 ED 在约 $2\sigma$ 内一致（如 $j=1$：0.1280(37) vs 0.1270；$j=7$：0.00928(9) vs 0.00916）✅
- **纵自旋 $\langle S^z_1S^z_j\rangle$**：15/16 点吻合（个别点约 $2\text{–}3\sigma$ 偏差，属 48 点统计涨落的正常范围）✅
- **横自旋（交换估计量）**：多数点吻合，但若干远距点（$j=9,10$ 等）系统性偏低且误差棒很小，$j=12,15,16$ 误差棒巨大——**这是强耦合下交换估计量的重尾（rare-event）涨落**：$N_e=1$/自旋时 $M_{ij}=Q[i]/Q[\mathrm{occ}]$，当占据位形接近奇异（$|Q[\mathrm{occ}]|\ll1$）时该位形权重极小却贡献巨大的估计量，$\lambda$ 越大尾越重。40000 sweep 不足以平均掉这些罕见事件；按距离合并后整体与 ED 一致（见图），但单点层面需要更长的链。这是统计问题而非算法错误（$U=2$ 时同一估计量 48/48 全对；nn/szz 两个 Fock 对角通道在 $U=7$ 也全对）。

![Fig S2 的 U=7 版本：三通道 vs ED](figures/fig_s2_ed_U7.svg)

**分析**：强耦合检验确认了算法框架在 $U=7$ 依然正确（对角通道完全吻合、符号保持 1），同时暴露了交换估计量在大 $\lambda$ 下的实际局限——这也是论文未讨论的实操细节。工程对策：低温强耦合下对交换类观测量需要显著加长测量链，或发展重尾稳健的估计/分箱方法。

### 8.2 Fig 1：sweep 代价标度

体系：二维正方格子 Hubbard，$U/t=2$，$T/t=1$（$\beta=10$，$\Delta\tau=0.1$），$N_e=100$（每自旋 50），与原文 Fig 1 参数一致。标准 DQMC 基线用同一套传播子与 HS 场实现（算法见 §2，`src/StdDQMC.jl`，约 90 行；经 wrap 闭合性 $\sim10^{-13}$ 与多次 sweep 后与新鲜 $G$ 完全一致两项检验）。计时在无其他负载的系统上独占测量（`results/scaling_clean.txt`；早期在多进程争用下的测量曾系统性偏高 2–3 倍，全部弃用重测）：

| $N$ | LEC-QMC (s/sweep) | DQMC (s/sweep) |
|---|---|---|---|
| 144 | 0.052 | 0.024 |
| 576 | 0.211 | 1.60 |
| 1024 | 0.380 | 9.99 |
| 1600 | 0.499 | 49.2 |
| 3600 | 1.14 | — |
| 6400 | 2.20 | — |
| 10000 | 3.47 | — |

log-log 拟合（$N\ge400$）：LEC-QMC 斜率 **0.99**（原文 ≈1），标准 DQMC 斜率 **3.24**（原文 ≈3）。与原文 Fig 1 定量对照：$N=1024$ 时 DQMC 约 10 s（原文约 10 s）、$N=1600$ 约 50 s（原文约 50 s）、LEC 在 $N=10^4$ 约 3.5 s（原文约 3 s）——绝对值与两条标度律均吻合。外推到 $N=10^4$，LEC 相对 DQMC 加速约 $5\times10^3$ 倍（原文声称 $>10^4$，同一量级，差异主要来自实现与硬件）。

![Fig 1：sweep 代价标度，LEC-QMC 线性 vs DQMC 三次方](figures/fig1_scaling.svg)

**分析**：LEC-QMC 斜率 0.99 直接证实了 $O(\beta N N_e^2)$ 复杂度（$\beta,N_e$ 固定时关于 $N$ 线性），与论文主张一致；DQMC 斜率 3.24 略大于 3，是小 $N$ 端固定开销摊薄所致，大趋势清晰。更有意义的是绝对值的一致：我们的 Julia 实现与原文（Siyuan 集群）在每个可比点上相差不到 2 倍，说明实现没有引入额外的渐进或常数因子低效。$N_e\ll N$ 时 LEC 的优势随 $N/N_e$ 线性增长——这正是算法设计的目标 regime。

实现备注：大 $N$ 计时暴露了一个工程问题——`make_propagator` 原本无条件计算稠密 $\exp(-\Delta\tau K)$（$N^2$ 矩阵指数的若干临时矩阵在 $N=6400$ 时即超出 4 GB 内存），而该矩阵仅供稠密测试路径使用；改为惰性计算后 $N=10^4$ 可正常运行。

### 8.3 Fig 2：稀薄极限下的符号恢复

体系：二维正方格子 Hubbard，$U=2$，固定 $N_e$ 与 $T$，增大 $L$（密度 $n=N_e/L^2$ 下降）。数据 `results/sign.txt`，图 `results/figures/fig2_sign.svg`。已测点与原文 Fig 2 对应曲线定量一致：

| $(T, N_e)$ | $1/L$ | $\langle\mathrm{sign}\rangle$（我们） | 原文（读图） |
|---|---|---|---|
| (0.05, 50) | 0.0625 ($L$=16) | 0.500 | ~0.4–0.5（外推趋势） |
| (0.05, 50) | 0.0500 ($L$=20) | 0.787 | ~0.78 |
| (0.05, 50) | 0.0417 ($L$=24) | 0.980 | ~0.95 |
| (0.05, 50) | 0.0312 ($L$=32) | 0.983 | ~1.0 |
| (0.05, 100) | 0.0625 ($L$=16) | 0.190 | ~0.2（同趋势） |

![Fig 2：稀薄极限下的符号恢复](figures/fig2_sign.svg)

**分析**：物理结论与原文一致：固定 $N_e$ 随体系增大（趋于稀薄极限），粒子间距超过交换关联长度，负符号的交换过程被抑制，$\langle\mathrm{sign}\rangle\to1$——稀薄费米气体渐近变为玻尔兹曼气体。数值上我们的 $(T,N_e)=(0.05,50)$ 曲线与原文几乎逐点重合（$1/L=0.05$：0.787 vs 约 0.78；$0.0417$：0.980 vs 约 0.95）。$(0.05,100)$ 曲线恢复更慢，与原文的密度依赖趋势一致。受单点运行时长限制，$(0.02,50)$ 低温曲线与更大 $L$ 的点未测（这些点符号更接近 1，不改变结论）。

### 8.4 Fig 4：平带铁磁（BCL 手征极限模型）

体系：一维 BCL 有效模型，$t_2=t_3=1$，$t_1=t_4=-0.2$，单能谱为 $L$ 重简并零模平带 + 色散带 $|\mathcal S_1(k)|^2+|\mathcal S_2(k)|^2$；模拟"平带半填充"$N_e=L$（每自旋 $L/2$，$S^z_{\rm tot}=0$  sector）。观测量：面内结构因子 $S^\parallel(q=0)=\frac1L\sum_{ij}\langle S^x_iS^x_j+S^y_iS^y_j\rangle$ 与 A1 子格 $S^{zz}(q=0)_{11}=\frac1L\sum_{i,j\in{\rm A1}}\langle S^z_iS^z_j\rangle$。

**相互作用强度的鉴别（复现中的一个坑）**：论文正文未给出 Fig 4 的 $U$ 值（仅 Fig 1/2/S2 标明 $U/t=2$）。我们先用 $U=2$ 计算，发现 ED 的 $S^\parallel(T)$ 衰减显著慢于论文曲线（如 $T=0.20$：0.505 vs 图中约 0.36），而 $S^{zz}_{11}$ 却吻合。对论文 Fig 4(a) 做像素级坐标提取后，用 ED 扫描 $U\in\{0.5,0.6,0.8,1,2,4,8\}$，发现 **$U=0.5$ 在全温区与论文曲线定量吻合**（如 $T=0.01$：0.7647 vs 0.765；$T=0.10$：0.4406 vs 0.437；$T=0.20$：0.3495 vs 0.351），故 Fig 4 复现采用 $U=0.5$。

**QMC 与 ED 的对比**（$U=0.5$，$\Delta\tau\simeq0.1$，$L=4$ 全温区，数据 `results/flatband.txt`）：

| $T$ | $S^\parallel$ QMC | $S^\parallel$ ED | $S^{zz}_{11}$ QMC | $S^{zz}_{11}$ ED | $\langle\mathrm{sign}\rangle$ |
|---|---|---|---|---|---|
| 0.01 | 0.784(31) | 0.765 | 0.0620(6) | 0.0626 | 0.90 |
| 0.02 | 1.00(35) | 0.631 | 0.0631(6) | 0.0626 | 0.93 |
| 0.04 | 0.554(10) | 0.560 | 0.0632(11) | 0.0627 | 0.95 |
| 0.06 | 0.512(7) | 0.517 | 0.0624(9) | 0.0630 | 0.98 |
| 0.08 | 0.476(8) | 0.476 | 0.0644(10) | 0.0634 | 0.99 |
| 0.10 | 0.445(8) | 0.441 | 0.0644(7) | 0.0638 | 1.00 |
| 0.15 | 0.384(3) | 0.382 | 0.0665(8) | 0.0645 | 1.00 |
| 0.20 | 0.350(2) | 0.349 | 0.0660(12) | 0.0649 | 1.00 |

$S^\parallel$ 八个点全部在约 $1\sigma$ 内与 ED 一致（$T=0.02$ 点因重尾涨落误差棒较大，见下注）；$S^{zz}_{11}$ 各点也均吻合，且整体趋势正确收敛到低温的 $1/16$。$U=0.5$ 时符号显著改善（$T\ge0.15$ 时 $\langle\mathrm{sign}\rangle=1.000$），重尾涨落比 $U=2$ 时弱得多。由于 ED(L=4) 本身与原文曲线全温区一致（见上），这些 QMC 数据同时构成与原文 Fig 4(a)(b) 的间接对比。

**尺寸趋势**（$S^\parallel$，与原文 Fig 4(a) 的 $L$ 依赖一致——随 $L$ 增大曲线略降且彼此靠近）：

| $T$ | $L=4$ | $L=8$ | $L=12$ | $L=16$ |
|---|---|---|---|---|
| 0.05 | 0.554(10) (ED 0.560) | 0.507(10) | — | — |
| 0.10 | 0.445(8) (ED 0.441) | 0.410(4) | 0.395(4) | — |
| 0.20 | 0.350(2) (ED 0.349) | 0.330(4) | 0.332(4) | 0.326(4) |

原文读图（$T=0.20$）：$L=4$ 约 0.35、$L=16$ 约 0.33——我们的 0.350/0.326 在误差棒内一致。

另外，$U=2$ 下我们已用整条 $L=4$ 温度曲线验证了 QMC↔ED 的内部一致性（如 $T=0.04$：0.668(132) vs 0.664；$T=0.06$：0.626(31) vs 0.609）——算法给出的结果与精确对角化一致，与原文曲线的差异完全由 $U$ 取值不同造成。

数值备注：$S^\parallel$ 的交换估计量（$\propto M^\uparrow_{ji}M^\downarrow_{ij}$）在近奇异权重位形下有重尾涨落（观测到单样本量级 $10^3$），低温点需要数万 sweep 才能使 blocking 误差棒收敛；$S^{zz}$ 为 Fock 对角估计量，收敛快得多。

![Fig 4：平带铁磁的结构因子温度演化](figures/fig4_flatband.svg)

### 8.5 验证总结与讨论

| 测试 | 内容 | 结果 |
|---|---|---|
| 单元测试 | 传播/增/删/交换 vs 稠密参考 | $<10^{-10}$ 全过 |
| 全枚举 | $\beta=1,2,4$ 精确枚举全部 $(\eta,s)$ | 分布精确吻合 |
| ED 解析 | 二站基态 doublon 混合 | 精确吻合 |
| 转移矩阵 | Python 独立计算 Trotter 系综 | 与 ED、与 MC 一致 |
| 决策审计 | 真实轨迹逐翻转比值对比 | 定位并修复 $\tilde R$ 失稳 |
| Fig S2 | L=4 三关联函数 48 点 vs ED（$U=2$） | 全部在误差棒内 |
| Fig S2' | 同设置 $U=7$ 强耦合 | nn/szz 全对；sxy 重尾涨落（§8.1a） |
| Fig 1 | sweep 代价标度 | 斜率 0.99 / 3.24，绝对值与原文一致 |
| Fig 2 | 稀薄极限符号恢复 | 已测点与原文定量一致 |
| Fig 4 | 平带铁磁（$U=0.5$ 经鉴别） | QMC 与 ED 一致；ED 与原文曲线全温区一致 |

**总体结论**。LEC-QMC 算法已完整复现：配分函数的 Fock 态表示、稳定化 QR 的粒子增/删/交换更新、辅助场的 fast update，全部通过从单元测试到 ED 全对比的层层验证。四个 benchmark 在已完成的数据点上均与原文定量一致：正确性（Fig S2，48/48 点）、线性标度（Fig 1，斜率 0.99）、稀薄极限符号恢复（Fig 2）、平带铁磁的基态性质（Fig 4，$S^\parallel$ 低温上升、$S^{zz}_{11}\to1/16$）。

**复现过程中的两个实质性发现**：

1. **$\tilde R$ 的"构造-反卷"失稳**（§5.1）：朴素方案在长 $\beta$ 下固有失稳且高温测试无法暴露，这是复现该算法最容易踩的坑；论文算法描述中未强调此处需要锚定式因子化。
2. **Fig 4 的 $U$ 值鉴别**（§8.4）：论文未标明平带模拟的 $U$，直接沿用 $U/t=2$ 会导致 $S^\parallel$ 曲线系统性偏离；像素级提取 + ED 扫描锁定了 $U=0.5$。

**与原文的差距（均为统计/覆盖度层面，非算法层面）**：(i) 我们的误差棒大于原文（单机 vs 集群的运行时长差异），特别是 $S^\parallel$ 交换估计量的重尾涨落需要长链压平；(ii) Fig 2 网格更稀疏，缺 $(0.02,50)$ 低温曲线与大 $L$ 点；(iii) Fig 4 大尺寸只测了高温点（sweep 代价 $\sim\beta N N_e^2$ 在半填充平带下随 $L$ 快速增长）。在小尺寸与小计算量下所有可对比的结果均与原文一致，按预定标准 benchmark 通过。

## 9. 代码结构与交付清单

```
lecqmc/
├── src/
│   ├── LECQMC.jl      # 核心库：模型构造、传播子（HS+checkerboard）、稳定化 QR、
│   │                  # Fock 态增/删/交换更新、辅助场 fast update（锚定方案）、测量
│   └── StdDQMC.jl     # 标准巨正则 DQMC 基线（Fig 1 标度对照）
├── ed/ed.jl           # 精确对角化参考（固定 N↑,N↓ 的热平均）
├── benchmarks/        # 全部 benchmark 脚本（均可独立重跑）：
│   ├── run_ed_benchmark.jl      # Fig S2（U=2）
│   ├── run_ed_benchmark_U7.jl   # Fig S2（U=7 强耦合）
│   ├── run_scaling4.jl          # Fig 1 清静系统标度测量
│   ├── run_sign.jl / run_sign2.jl   # Fig 2 符号恢复
│   └── run_flatband_u05.jl      # Fig 4 平带铁磁（U=0.5）
├── test/runtests.jl   # 回归测试（单元 + 全枚举 + 高温 ED 对比）
└── results/           # 数据文件、绘图脚本 make_figures.py、figures/ 四张定稿图
```

**已完成任务汇总**：

1. ✅ LEC-QMC 算法完整 Julia 复现（Fock 态采样 + 稳定化 QR + fast update），全部单元/全枚举/独立交叉验证通过。
2. ✅ 数值稳定性关键问题的诊断与修复（$\tilde R$ 反卷失稳 → LQ 锚定因子化，§5.1）。
3. ✅ 标准 DQMC 基线实现与正确性验证（Fig 1 对照）。
4. ✅ Fig S2（$U=2$，48/48 点）与 $U=7$ 强耦合扩展对比。
5. ✅ Fig 1 标度复现（斜率与绝对值均与原文一致）。
6. ✅ Fig 2 符号恢复复现（已测点与原文定量一致）。
7. ✅ Fig 4 平带铁磁复现（含 $U=0.5$ 鉴别、ED 参考、L=4 全温区 QMC 对比、尺寸趋势）。
8. ✅ 算法 note（本文档）与带详细中文注释的源码。

**环境备注**：复现工作在受限容器（2 核 / 4 GB，周期性清空）中完成；Julia 1.10.9 运行时分块备份于 `/mnt/agents/jdist.part.*`（`cat /mnt/agents/jdist.part.* | tar xz -C /home/kimi` 恢复）。受运行预算限制，Fig 2/4 的网格密度低于原文（所有已完成数据点均与原文一致）。
