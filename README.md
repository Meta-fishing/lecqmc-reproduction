# LEC-QMC 复现：Linear Canonical-Ensemble Quantum Monte Carlo

复现论文 **Tu Hong, Kun Chen, Xiao Yan Xu, _Linear Canonical-Ensemble Quantum Monte Carlo: From Dilute Fermi Gas to Flat-Band Ferromagnetism_（arXiv:2601.08552）** 的 Julia 实现，含四个 benchmark 的复现结果与详细的中文算法笔记。

## 内容

- **`src/LECQMC.jl`** — 核心库（约 1000 行，含详细中文注释）：正则系综 DQMC，$Z=\sum_{\eta,s}\prod_\sigma\det[P_\eta^\dagger B_s P_\eta]$；稳定化 QR 传播；Fock 态粒子增/删/交换更新；辅助场 fast update（**锚定因子化方案**，修复论文未展开的 $\tilde R$ 反卷失稳问题，详见算法笔记 §5.1）；斜投影格林函数与交换估计量。
- **`src/StdDQMC.jl`** — 标准巨正则 DQMC 基线（Fig 1 标度对照）。
- **`ed/ed.jl`** — 精确对角化参考（固定 $N_\uparrow,N_\downarrow$ 的有限温热平均）。
- **`benchmarks/`** — 全部可独立重跑的 benchmark 脚本（见下表）。
- **`docs/lecqmc_note.md`** — **详细中文算法笔记**（350+ 行）：完整推导、标准 DQMC 算法详解、数值稳定性陷阱分析、四个 benchmark 的数据与图、复现中的两个实质性发现。
- **`results/`** — 全部原始数据与绘图脚本；**`figures/`** — 定稿图。

## Benchmark 复现结果

| 原文 | 内容 | 结果 |
|---|---|---|
| Fig S2 | L=4 Hubbard 三种关联函数 48 点 vs ED（U=2） | ✅ 全部在误差棒内（[数据](results/ed_benchmark_L4.txt)） |
| Fig S2' | 同设置 U=7 强耦合扩展 | ✅ nn/szz 全对（[数据](results/ed_benchmark_L4_U7.txt)） |
| Fig 1 | sweep 代价标度 | ✅ LEC 斜率 0.99（原文≈1）、DQMC 斜率 3.24（原文≈3），绝对值逐点一致 |
| Fig 2 | 稀薄极限符号恢复 | ✅ 已测点与原文定量一致（0.787 vs ~0.78 @ 1/L=0.05 等） |
| Fig 4 | 平带铁磁 | ✅ L=4 全温区与 ED 一致；ED 与原文曲线全温区一致；Szz→1/16 解析值 |

### Fig S2：LEC-QMC vs ED（U=2 与 U=7）
![Fig S2](figures/fig_s2_ed.png)
![Fig S2 U7](figures/fig_s2_ed_U7.png)

### Fig 1：sweep 代价标度（LEC 线性 vs DQMC 三次方）
![Fig 1](figures/fig1_scaling.png)

### Fig 2：稀薄极限下符号恢复
![Fig 2](figures/fig2_sign.png)

### Fig 4：平带铁磁结构因子（U=0.5，经像素级提取+ED 扫描鉴别）
![Fig 4](figures/fig4_flatband.png)

## 快速开始

```julia
julia> include("src/LECQMC.jl"); using .LECQMC
julia> m = build_square_lattice(4, 4, 1.0, 2.0)        # 4×4 方格子, t=1, U=2
julia> mc = MC(m, 10.0, 0.1, 10, [[3],[7]])            # β=10, Δτ=0.1, N↑=N↓=1
julia> for s in 1:100; sweep!(mc, Random.MersenneTwister(1)); end
```

重跑 benchmark（以 Fig S2 为例）：`cd benchmarks && julia run_ed_benchmark.jl`

测试：`cd test && julia runtests.jl`（稳定化传播 / 粒子增删交换 / 翻转比值共 15 项）

## 复现中的两个实质性发现（详见笔记）

1. **$\tilde R$ 的"构造-反卷"失稳**（笔记 §5.1）：朴素方案在长 $\beta$ 下产生 $\mathrm{cond}(B)\cdot\epsilon$ 量级方向误差且 QR 再稳定化无法修复（$\beta=10$ 时二站 Hubbard 给出 0.885 vs 精确值 0.724）；修复为 LQ 锚定因子化 + 每 chunk 新鲜 $A_f$ + 规范不变性。
2. **Fig 4 的 $U$ 值鉴别**（笔记 §8.4）：论文未标明平带模拟的 $U$，沿用 $U/t=2$ 会系统性偏离；像素级提取论文曲线 + ED 扫描锁定 $U=0.5$（全温区偏差 <0.02）。
