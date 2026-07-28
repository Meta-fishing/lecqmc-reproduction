# ============================================================================
# LECQMC.jl — Linear Ensemble-Constrained Quantum Monte Carlo
# Reproduction of arXiv:2601.08552 (Hong, Chen, Xu)
#
# Canonical-ensemble (fixed Nup, Ndn) Fock-state DQMC with stabilized
# QR updates for particle addition/removal:  O(beta*N*Ne) per Fock update,
# O(beta*N*Ne^2) per sweep.
#
# Conventions:
#   H = sum_{ij} K_{ij} c^dag_i c_j + U sum_i n_iup n_idn
#   HS:  exp(-dt*U(nu-1/2)(nd-1/2)) = 1/2 sum_{s=±1} exp(s*lam*(nu-nd)),
#        cosh(lam) = exp(dt*U/2).  (linear/const terms drop out in canonical)
#   B_l^σ = exp(V_l^σ) exp(-dt*K),  exp(V_l^σ)_i = exp(σ*lam*s_{l,i})
#   Weight: W(η,s) = prod_σ det[P_σ^† B_σ(β,0) P_σ]
# ============================================================================
#
# 【中文总览】
# 本模块是 arXiv:2601.08552（Hong, Chen, Xu）"Linear Canonical-Ensemble QMC"
# 论文的核心复现库：在**正则系综**（固定粒子数 N↑、N↓）下对 Hubbard 类模型
# 做行列式量子蒙特卡洛（DQMC）。配分函数写成 Fock 态 η 与 HS 辅助场 s 的双重求和
#       Z = Σ_{η, s}  ∏_{σ=↑,↓} det[ P_η^† B_s^σ(β,0) P_η ] ,
# 其中 P_η 是 Fock 态对应的选择矩阵（N×Ne），B_s^σ(β,0) = B_{Lτ}⋯B_1 是虚时传播子。
# 与巨正则 DQMC 的两点区别：
#   (1) 除了 HS 场更新（field sweep），还要做 Fock 态更新（粒子增删/交换，
#       即 fock sweep），用稳定化的秩一 QR 技巧把单次 Fock 更新压到 O(β·N·Ne)；
#   (2) 所有量（Q、R、A=(P†Q)^{-1}）都按 chunk 存储（Qs、Vs），便于局部更新。
#
# 文件结构导览（按出现顺序）：
#   1. Model / build_square_lattice / build_flatband_chain —— 晶格与单粒子矩阵 K；
#   2. Propagator / make_propagator / apply_*!              —— HS 变换与稀疏传播子作用；
#   3. MCParams / SpinState / full_propagation!             —— 稳定化 QR 传播与状态维护；
#   4. Givens 旋转工具 + trial_add / trial_remove            —— Fock 更新（SM Sec. I）；
#   5. field_sweep!                                          —— HS 场 sweep（数值最微妙，见函数头注释）；
#   6. fock_sweep!                                           —— 粒子-空穴交换 sweep；
#   7. green_cols / measure! / Observables / weight_sign     —— 测量与符号加权；
#   8. MC / sweep!                                           —— 主驱动。
# ============================================================================
module LECQMC

using LinearAlgebra
using Random
using Printf

export Model, Propagator, SpinState, MCParams, MC, Observables,
       build_square_lattice, build_flatband_chain,
       make_propagator, apply_B!, apply_Binv_right!, apply_B_right!, apply_expK!,
       full_propagation!, trial_remove, trial_add, accept_trial!,
       field_sweep!, fock_sweep!, measure!, green_cols, selection_matrix,
       weight_sign, finalize_obs, sweep!

# 自旋指标常量：σ=1 为 ↑，σ=2 为 ↓（HS 场对两种自旋取相反符号）
const SPIN_UP = 1
const SPIN_DN = 2

# ---------------------------------------------------------------------------
# Model: lattice + hopping + interaction
# ---------------------------------------------------------------------------
# Model：模型容器。K 是单粒子（hopping + onsite）矩阵，即 H₀ = Σ_{ij} K_{ij} c†_i c_j；
# 哈密顿量为 H = Σ K_{ij} c†_i c_j + U Σ_i n_{i↑} n_{i↓}。
# bond_groups 是键的 checkerboard（棋盘）分组：同一组内的键共享的格点互不相交，
# 因此同组所有 2×2 指数因子可同时作用（可换序），这是稀疏传播子的基础。
struct Model
    N::Int                                  # number of sites
    K::Matrix{Float64}                      # one-body matrix (hopping + onsite)
    U::Float64
    bond_groups::Vector{Vector{Tuple{Int,Int,Float64}}}  # checkerboard groups (i,j,t_hop); sites disjoint within a group
end

# 【中文说明】贪心边着色：把键分成若干组，每组内任意两条键不共享格点（对任意 L、
# 周期边界均成立）。这样同组的 2×2 键指数块互相对易，exp(-Δτ K) 可精确分解为
# "对角 onsite 因子 × 逐组 2×2 旋转"，是 checkerboard 稀疏传播子的基础。
"""Greedy edge-coloring: partition bonds into groups with disjoint sites (valid for any L, PBC)."""
function color_bonds(bonds::Vector{Tuple{Int,Int,Float64}})
    groups = Vector{Tuple{Int,Int,Float64}}[]
    used = Vector{Set{Int}}()
    for (i, j, th) in bonds
        placed = false
        for (gi, g) in enumerate(groups)
            if !(i in used[gi]) && !(j in used[gi])
                push!(g, (i, j, th)); push!(used[gi], i, j)
                placed = true
                break
            end
        end
        if !placed
            push!(groups, [(i, j, th)])
            push!(used, Set([i, j]))
        end
    end
    return groups
end

# 【中文说明】二维正方格子 Hubbard 模型：最近邻跳跃 -t，周期边界条件。
# 格点一维编号 i = x + (y-1)*Lx（x=1..Lx 为行内指标）。只填 K 矩阵的
# 非对角元（±x、±y 方向各 -t），无化学势项；键列表交给 color_bonds 分组。
"""2D square lattice Hubbard, nearest-neighbor hopping -t, PBC. Site index: x + (y-1)*Lx."""
function build_square_lattice(Lx::Int, Ly::Int, t::Float64, U::Float64)
    N = Lx * Ly
    K = zeros(N, N)
    bonds = Tuple{Int,Int,Float64}[]
    idx(x, y) = mod(x - 1, Lx) + mod(y - 1, Ly) * Lx + 1
    for y in 1:Ly, x in 1:Lx
        i = idx(x, y)
        jx = idx(x + 1, y)
        jy = idx(x, y + 1)
        K[i, jx] = -t; K[jx, i] = -t
        K[i, jy] = -t; K[jy, i] = -t
        push!(bonds, (i, jx, -t))
        push!(bonds, (i, jy, -t))
    end
    return Model(N, K, U, color_bonds(bonds))
end

"""
1D flat-band effective model (BCL chiral limit), L unit cells, 2 sublattices (A1,A2).
H_eff(k) = S(k) S(k)^† with S1 = t1 + t2 e^{-ik}, S2 = t3 + t4 e^{-ik}.
Real space (cell R, sublattice a=1,2): site index = R + (a-1)*L.
  A1 onsite t1^2+t2^2, A1-A1(R±1) hop t1*t2
  A2 onsite t3^2+t4^2, A2-A2(R±1) hop t3*t4
  A1(R)-A2(R): t1*t3 + t2*t4 ;  A1(R)-A2(R+1): t1*t4 ;  A1(R)-A2(R-1): t2*t3
"""
# 【中文说明】一维平带有效模型（BCL 手征极限）：L 个原胞、两个子格 A1/A2。
# 动量空间单粒子哈密顿量取 H_eff(k) = S(k) S(k)†（正定、保证有严格零能平带），
# 其中 S 矩阵的两列分别为
#     S1(k) = t1 + t2 e^{-ik},   S2(k) = t3 + t4 e^{-ik}.
# 实空间跳跃（见下方英文 docstring 的逐项对应）：
#   子格内：A1  onsite 能 t1²+t2²、最近邻跳跃 t1·t2；A2  onsite 能 t3²+t4²、跳跃 t3·t4；
#   子格间：A1(R)-A2(R) 跳跃 t1·t3+t2·t4，A1(R)-A2(R+1) 跳跃 t1·t4，A1(R)-A2(R-1) 跳跃 t2·t3。
# 格点编号：site = R + (a-1)*L（a=1 为 A1，a=2 为 A2）。
function build_flatband_chain(L::Int, t1::Float64, t2::Float64, t3::Float64, t4::Float64, U::Float64)
    N = 2L
    K = zeros(N, N)
    A1(R) = mod(R - 1, L) + 1
    A2(R) = mod(R - 1, L) + 1 + L
    e1 = t1^2 + t2^2
    e2 = t3^2 + t4^2
    for R in 1:L
        K[A1(R), A1(R)] = e1
        K[A2(R), A2(R)] = e2
    end
    bonds = Tuple{Int,Int,Float64}[]
    for R in 1:L
        i, j = A1(R), A1(R + 1)
        K[i, j] += t1 * t2; K[j, i] += t1 * t2
        push!(bonds, (i, j, t1 * t2))
        i, j = A2(R), A2(R + 1)
        K[i, j] += t3 * t4; K[j, i] += t3 * t4
        push!(bonds, (i, j, t3 * t4))
        i, j = A1(R), A2(R)
        K[i, j] += t1 * t3 + t2 * t4; K[j, i] += t1 * t3 + t2 * t4
        push!(bonds, (i, j, t1 * t3 + t2 * t4))
        i, j = A1(R), A2(R + 1)
        K[i, j] += t1 * t4; K[j, i] += t1 * t4
        push!(bonds, (i, j, t1 * t4))
        i, j = A1(R), A2(R - 1)
        K[i, j] += t2 * t3; K[j, i] += t2 * t3
        push!(bonds, (i, j, t2 * t3))
    end
    return Model(N, K, U, color_bonds(bonds))
end

# ---------------------------------------------------------------------------
# Propagator: precomputed slice factors for a given Δτ and HS field sign
# ---------------------------------------------------------------------------
# Propagator：单个时间片 Δτ 的预计算因子。核心物理是离散 HS 变换
#     exp(-Δτ U (n↑-1/2)(n↓-1/2)) = (1/2) Σ_{s=±1} exp(s λ (n↑ - n↓)),
#     cosh λ = exp(Δτ U / 2),
# 于是单片传播子分解为 B_l^σ = exp(V_l^σ) exp(-Δτ K)，其中
# exp(V_l^σ) 是对角矩阵，第 i 个对角元为 exp(σ λ s_{l,i})（σ=±1 对应上/下自旋）。
# exp(-Δτ K) 不显式存储稠密矩阵，而是存成 checkerboard 形式：
#   diag_exp   —— onsite 对角因子 exp(-Δτ K_ii)；
#   bond_exp   —— 每组键的 2×2 块参数 (i,j,c,s)，作用在 (i,j) 平面上的矩阵
#                [[c,s],[s,c]] = exp(-Δτ·t_hop·σx)（c=cosh(Δτ t), s=-sinh(Δτ t)）。
# expK_dense 只在小体系/测试时计算：大 N 时 exp() 稠密矩阵指数会产生
# O(N²) 甚至更大的临时量导致 OOM（实践教训），因此默认惰性留空。
struct Propagator
    dt::Float64
    lam::Float64                            # HS coupling λ
    bond_exp::Vector{Vector{Tuple{Int,Int,Float64,Float64}}}   # (i,j,c,s): [[c,s],[s,c]] = exp(-dt*K_bond)
    diag_exp::Vector{Float64}               # exp(-dt*K_ii)
    expK_dense::Matrix{Float64}             # dense exp(-dt*K) (for tests / small systems)
    use_dense::Bool
end

# 【中文说明】构造 Propagator：由 cosh λ = exp(Δτ U/2) 解出 HS 耦合 λ，
# 对每个 checkerboard 组内的键 (i,j,t_hop) 预计算 2×2 指数块参数
# (c, s) = (cosh(Δτ·t_hop), -sinh(Δτ·t_hop))；onsite 对角因子单独存 diag_exp。
function make_propagator(m::Model, dt::Float64; use_dense::Bool=false)
    lam = acosh(exp(dt * m.U / 2))
    bond_exp = Vector{Tuple{Int,Int,Float64,Float64}}[]
    for g in m.bond_groups
        ge = Tuple{Int,Int,Float64,Float64}[]
        for (i, j, th) in g
            # 2x2 block of K: th*σx  ->  exp(-dt*th*σx) = cosh I - sinh σx
            push!(ge, (i, j, cosh(dt * th), -sinh(dt * th)))
        end
        push!(bond_exp, ge)
    end
    diag_exp = exp.(-dt .* diag(m.K))
    # dense exp(-dt*K) is only needed for the dense (test) path; skip it for
    # large N to avoid the O(N^2) temporaries of matrix exponential (OOM).
    # （中文）稠密 exp(-Δτ K) 仅测试路径需要；大 N 时跳过，避免矩阵指数的
    # O(N²) 临时数组造成内存溢出——生产路径一律走下面的 checkerboard 稀疏作用。
    expK_dense = (use_dense || m.N <= 2048) ? exp(-dt .* m.K) : zeros(0, 0)
    return Propagator(dt, lam, bond_exp, diag_exp, expK_dense, use_dense)
end

# 【中文说明】y .= exp(∓Δτ K) · x（x 为 N×k 的列集合，inv=false 为正传播，
# inv=true 为逆传播 exp(+Δτ K)）。checkerboard 路径复杂度 O(N·z·k)（z 为配位数），
# 远优于稠密 O(N²k)。顺序约定：expK = (逐组键块) · D，D 为对角 onsite 因子；
# 正传播先乘 D 再按组序乘各键块；逆传播用各因子的逆（键块 2×2 逆为 [[c,-s],[-s,c]]、
# 对角元取倒数），且顺序完全反转。
"""y .= expK * x  (x: N×k). Uses checkerboard (O(N·z·k)) or dense."""
function apply_expK!(y::AbstractMatrix{Float64}, x::AbstractMatrix{Float64}, p::Propagator, inv::Bool=false)
    if p.use_dense
        if inv
            mul!(y, inv(p.expK_dense), x)   # dense path (tests only)
        else
            mul!(y, p.expK_dense, x)
        end
        return y
    end
    copyto!(y, x)
    if !inv
        @inbounds for i in axes(y, 2)
            for r in axes(y, 1); y[r, i] *= p.diag_exp[r]; end
        end
        for g in p.bond_exp
            for (i, j, c, s) in g
                @inbounds for k in axes(y, 2)
                    xi = y[i, k]; xj = y[j, k]
                    y[i, k] = c * xi + s * xj
                    y[j, k] = s * xi + c * xj
                end
            end
        end
    else
        for g in reverse(p.bond_exp)
            for (i, j, c, s) in g
                @inbounds for k in axes(y, 2)
                    xi = y[i, k]; xj = y[j, k]
                    y[i, k] = c * xi - s * xj
                    y[j, k] = -s * xi + c * xj
                end
            end
        end
        @inbounds for i in axes(y, 2)
            for r in axes(y, 1); y[r, i] /= p.diag_exp[r]; end
        end
    end
    return y
end

# 【中文说明】左乘单片传播子：y .= B_l^σ · x。因 B_l^σ = exp(V_l^σ)·exp(-Δτ K)，
# 先做 expK 稀疏左乘（结果放 tmp），再逐行乘对角 HS 因子 exp(σλ s_{l,i})
# （σ=↑ 取 +1、σ=↓ 取 -1，体现 HS 场对两种自旋符号相反）。
"""y .= B_l^σ * x :  first expK, then interaction diagonal exp(σ λ s_l)."""
function apply_B!(y::AbstractMatrix{Float64}, x::AbstractMatrix{Float64},
                  p::Propagator, s_l::AbstractVector{Int8}, σ::Int,
                  tmp::AbstractMatrix{Float64})
    apply_expK!(tmp, x, p, false)
    sgn = σ == SPIN_UP ? 1.0 : -1.0
    @inbounds for i in axes(y, 1)
        f = exp(sgn * p.lam * s_l[i])
        for k in axes(y, 2)
            y[i, k] = f * tmp[i, k]
        end
    end
    return y
end

# 【中文说明】右乘单片传播子的逆：X .= X · (B_l^σ)^{-1}，X 为 Ne×N（行向量集合）。
# (B_l^σ)^{-1} = exp(+Δτ K) · exp(-V_l^σ)。右乘 X·M 时，M 的**最左**因子先作用于 X，
# 所以先施加 exp(+Δτ K)（对角逆 + 正组序、键块取逆 (c,-s)），再右乘 exp(-V)
# （逐列缩放 exp(-σλ s_{l,i})）。这是 field_sweep! 中向右回扫 R̃ 的核心操作。
"""X .= X * B_l^{-1}  (X is Ne×N, right multiplication): B_l^{-1} = exp(+dt K) exp(-V)."""
function apply_Binv_right!(X::AbstractMatrix{Float64}, p::Propagator, s_l::AbstractVector{Int8}, σ::Int,
                           tmp::AbstractMatrix{Float64})
    copyto!(tmp, X)
    # right multiply by exp(+dt K) = D^{-1} G_1^{-1} G_2^{-1} G_3^{-1} G_4^{-1}:
    # X M^{-1} applies leftmost factor first -> diag inverse first, groups forward with (c,-s)
    @inbounds for k in axes(tmp, 1)
        for r in axes(tmp, 2); tmp[k, r] /= p.diag_exp[r]; end
    end
    for g in p.bond_exp
        for (i, j, c, s) in g
            @inbounds for k in axes(tmp, 1)
                xi = tmp[k, i]; xj = tmp[k, j]
                tmp[k, i] = c * xi - s * xj
                tmp[k, j] = -s * xi + c * xj
            end
        end
    end
    # then right-multiply by exp(-V_l^σ): scale columns
    sgn = σ == SPIN_UP ? 1.0 : -1.0
    @inbounds for i in axes(tmp, 2)
        f = exp(-sgn * p.lam * s_l[i])
        for k in axes(tmp, 1)
            tmp[k, i] *= f
        end
    end
    copyto!(X, tmp)
    return X
end

# 【中文说明】右乘单片传播子：X .= X · B_l^σ（X 为 Ne×N）。B_l^σ = exp(V)·expK，
# 右乘时最左因子 exp(V) 先作用（逐列缩放 exp(σλ s_{l,i})），随后 expK 部分
# 按"反转组序、对角最后"施加。用于从 P† 出发自顶向下构造 P†B(β,τ)。
"""X .= X * B_l  (X Ne×N right multiplication): B_l = exp(V) expK, so X B_l = (X expV) expK."""
function apply_B_right!(X::AbstractMatrix{Float64}, p::Propagator, s_l::AbstractVector{Int8}, σ::Int,
                        tmp::AbstractMatrix{Float64})
    copyto!(tmp, X)
    # first right-multiply by exp(V_l^σ): scale columns
    sgn = σ == SPIN_UP ? 1.0 : -1.0
    @inbounds for i in axes(tmp, 2)
        f = exp(sgn * p.lam * s_l[i])
        for k in axes(tmp, 1)
            tmp[k, i] *= f
        end
    end
    # then right-multiply by exp(-dt K) = G_4 G_3 G_2 G_1 D (as matrix):
    # X M applies the leftmost factor first -> reverse group order, diag last
    for g in reverse(p.bond_exp)
        for (i, j, c, s) in g
            @inbounds for k in axes(tmp, 1)
                xi = tmp[k, i]; xj = tmp[k, j]
                tmp[k, i] = c * xi + s * xj
                tmp[k, j] = s * xi + c * xj
            end
        end
    end
    @inbounds for k in axes(tmp, 1)
        for r in axes(tmp, 2); tmp[k, r] *= p.diag_exp[r]; end
    end
    copyto!(X, tmp)
    return X
end

# ---------------------------------------------------------------------------
# Monte Carlo parameters & per-spin state
# ---------------------------------------------------------------------------
# MCParams：虚时离散参数。β 被分成 Lτ = β/Δτ 个时间片；每 nstab 片做一次
# 稳定化（QR/LQ 重正交），共 nchunks = Lτ/nstab 个 chunk（稳定化步）。
struct MCParams
    beta::Float64
    dt::Float64
    Ltau::Int                               # β/Δτ
    nstab::Int                              # stabilization interval (slices)
    nchunks::Int                            # Ltau/nstab stabilization steps
end

# 【中文说明】由 β、Δτ、nstab 推 Lτ 与 nchunks，并检查 β=Δτ·Lτ、nstab | Lτ。
function MCParams(beta::Float64, dt::Float64, nstab::Int)
    Ltau = round(Int, beta / dt)
    @assert abs(Ltau * dt - beta) < 1e-10
    @assert mod(Ltau, nstab) == 0
    return MCParams(beta, dt, Ltau, nstab, div(Ltau, nstab))
end

# 【中文说明】SpinState：单个自旋分量在当前 Fock 态 η 下的全部稳定化数据。
#   occ —— Fock 态的表示：占据格点列表（长度 Ne，即该自旋粒子数）。
#   整体分解 B(β,0) P = Q R（Q 为 N×Ne 列正交，R 为 Ne×Ne 上三角）；
#   按 chunk 存储分块因子 B_chunk·Q_{c-1} = Q_c·V_c（Qs、Vs），
#   这是 Fock 更新能"只传一列/只做局部 Givens"的关键。
#   A = (P†Q)^{-1} 维护用于 O(1) 计算行列式比值（正文 Eq. (4) 的副产品）。
"""
Per-spin stabilized data for the current Fock state:
B(β,0) P = Q R,  with per-chunk factors  B_i Q_{i-1} = Q_i V_i  stored in Qs, Vs.
A = (P† Q)^{-1} is maintained for fast ratios (it is a byproduct of the
Green's-function calculation, cf. main text Eq. (4)).
"""
mutable struct SpinState
    σ::Int
    occ::Vector{Int}                        # occupied sites (length Ne)
    Qs::Vector{Matrix{Float64}}             # per chunk: N×Ne orthonormal
    Vs::Vector{Matrix{Float64}}             # per chunk: Ne×Ne upper triangular
    Q::Matrix{Float64}                      # N×Ne final basis (= Qs[end])
    R::Matrix{Float64}                      # Ne×Ne accumulated (V_n···V_1)
    A::Matrix{Float64}                      # (P†Q)^{-1}
end

# 【中文说明】由占据列表 occ 构造选择矩阵 P（N×Ne）：第 a 列是第 occ[a] 个
# 坐标单位向量。P 把"Ne 个粒子的 Fock 态"嵌入 N 维单粒子 Hilbert 空间，
# P†M 即取 M 的 occ 行（如 P†Q = Q[occ,:]）。
"""Selection matrix P (N×Ne) from occupation list."""
function selection_matrix(N::Int, occ::Vector{Int})
    P = zeros(N, length(occ))
    for (a, i) in enumerate(occ)
        P[i, a] = 1.0
    end
    return P
end

# 【中文说明】自顶向下（τ=0→β）的稳定化 QR 全传播：从 X=P 出发，逐片左乘 B_l，
# 每传播完一个 chunk（nstab 片）对 X 做一次 QR：X = Q_c·V_c，把 Q_c 作为下一个
# chunk 的起点（防数值秩丢失/溢出）。最终
#   B(β,0)P = Q·R，R = V_n⋯V_1（逐 chunk 累乘），A = (P†Q)^{-1} = inv(Q[occ,:])。
# 返回的 SpinState 保存全部中间因子，供 Fock 更新复用。
# 复杂度：O(Lτ·z·N·Ne + nchunks·N·Ne²)。
"""
Full stabilized propagation B(β,0)P with current HS fields.
fields: Lτ×N Int8 matrix.  Returns SpinState with Qs,Vs,Q,R,A.
Cost: O(Lτ · z · N · Ne + nchunks · N · Ne^2).
"""
function full_propagation!(m::Model, p::Propagator, mp::MCParams,
                           fields::AbstractMatrix{Int8}, σ::Int, occ::Vector{Int})
    Ne = length(occ)
    X = selection_matrix(m.N, occ)
    tmp = similar(X)
    Qs = Matrix{Float64}[]
    Vs = Matrix{Float64}[]
    Racc = zeros(Ne, Ne)
    for c in 1:mp.nchunks
        for l in (c - 1) * mp.nstab + 1:c * mp.nstab
            apply_B!(X, X, p, @view(fields[l, :]), σ, tmp)   # in-place safe via tmp
        end
        if Ne > 0
            F = qr(X)
            Qc = Matrix(F.Q)
            Vc = Matrix(F.R)
        else
            Qc = zeros(m.N, 0)
            Vc = zeros(0, 0)
        end
        push!(Qs, Qc); push!(Vs, Vc)
        Racc = (c == 1) ? copy(Vc) : Vc * Racc
        X = copy(Qc)          # must copy: Qc is stored in Qs and reused as buffer
    end
    Q = copy(Qs[end])
    A = Ne > 0 ? inv(Q[occ, :]) : zeros(0, 0)
    return SpinState(σ, copy(occ), Qs, Vs, Q, Racc, A)
end

# ---------------------------------------------------------------------------
# Givens rotations (real): act on rows (j,j+1) from the left:
#   new_row_j   =  c*row_j - s*row_{j+1}
#   new_row_{j+1}= s*row_j + c*row_{j+1}
# chosen so that M[j+1,j] is annihilated.  det G = 1.
# ---------------------------------------------------------------------------
# 【中文说明】Givens 旋转工具集。实旋转 G 作用在相邻两行 (j, j+1)：
#   row_j   ← c·row_j - s·row_{j+1}
#   row_{j+1}← s·row_j + c·row_{j+1}
# 取 c = a/r, s = -b/r（r = hypot(a,b)）可消去目标矩阵的 M[j+1,j] 元；
# det G = 1，故不改变行列式（只可能经 R 的对角元改号，需要小心但此处安全）。
# 在粒子删除中用于把"列置换后变成上 Hessenberg"的 V 因子逐步恢复上三角。
function givens(a::Float64, b::Float64)
    r = hypot(a, b)
    if r == 0.0
        return 1.0, 0.0
    end
    return a / r, -b / r
end

"""Apply left rotations `rots` (list of (j,c,s)) to matrix M (in place)."""
function apply_rots_left!(M::AbstractMatrix{Float64}, rots)
    for (j, c, s) in rots
        @inbounds for col in axes(M, 2)
            x1 = M[j, col]; x2 = M[j + 1, col]
            M[j, col] = c * x1 - s * x2
            M[j + 1, col] = s * x1 + c * x2
        end
    end
    return M
end

"""Apply G† (inverse) on the right: M -> M G† for rotations list (in place)."""
function apply_rots_right_inv!(M::AbstractMatrix{Float64}, rots)
    # G† = G^T; applying to columns: new_col_j = c*col_j - s*col_{j+1}; new_col_{j+1} = s*col_j + c*col_{j+1}
    # G = G_{Ne-1}···G_k  =>  M G† = (M G_k†)···G_{Ne-1}† : forward order
    for (j, c, s) in rots
        @inbounds for row in axes(M, 1)
            x1 = M[row, j]; x2 = M[row, j + 1]
            M[row, j] = c * x1 - s * x2
            M[row, j + 1] = s * x1 + c * x2
        end
    end
    return M
end

"""
Triangularize upper-Hessenberg M (subdiagonal only at (j+1,j), j=kstart..Ne-1)
via left Givens rotations. Returns (rots, M_triangular).
"""
function triangularize!(M::Matrix{Float64}, kstart::Int)
    Ne = size(M, 1)
    rots = Tuple{Int,Float64,Float64}[]
    for j in kstart:Ne-1
        if M[j + 1, j] != 0.0
            c, s = givens(M[j, j], M[j + 1, j])
            push!(rots, (j, c, s))
            @inbounds for col in j:Ne
                x1 = M[j, col]; x2 = M[j + 1, col]
                M[j, col] = c * x1 - s * x2
                M[j + 1, col] = s * x1 + c * x2
            end
        end
    end
    return rots, M
end

# ---------------------------------------------------------------------------
# Particle ADDITION  (SM Sec. I A):  P' = [P, p],  p = e_j
# Propagate only the new column through the chunks; at each stabilization
# step orthogonalize against the stored basis Q_i (MGS) and update V', R'.
# Returns a named tuple with trial factors + r_add (= s * r_{Ne+1}).
# Cost: O(β · z · N + nchunks · N · Ne) = O(β N Ne).
#
# 【中文推导】在格点 j 增加一个粒子：P' = [P, p]，p = e_j。权重的行列式比为
#   r_add = det[P'† Q' R'] / det[P† Q R] = s · r_{Ne+1}，
# 其中 r_{Ne+1} 是新 R' 的最后一个对角元（来自传播新列时的逐步正交化），
# s 是 Schur 补：s = p†q − p†Q(P†Q)^{-1}P†q（正文 Eq. (4)/SM Eq. (S7)），
# q 是新列传播到 β 并正交化后的单位向量。
# 关键技巧：**只传播新列**。从 w=e_j 出发逐 chunk 施加 B_l；每到一个稳定化点，
# 用存储的基 Q_c 做一次 MGS 正交化：v = Q_c†w，res = w − Q_c v，q = res/‖res‖，
# 于是新基 Q'_c = [Q_c q]，新三角因子 V'_c = [[V_c, v],[0, ‖res‖]]。
# 最后新 A' = (P'†Q')^{-1} 用分块求逆公式（SM Eq. (S8)）O(Ne²) 更新。
# 总复杂度 O(β·N·Ne)，这就是论文标题 "Linear" 的来源。
# ---------------------------------------------------------------------------
function trial_add(m::Model, p::Propagator, mp::MCParams,
                   fields::AbstractMatrix{Int8}, st::SpinState, j::Int)
    N = m.N
    Ne = length(st.occ)
    w = zeros(N, 1)
    w[j, 1] = 1.0
    tmp = similar(w)
    Qs_new = Matrix{Float64}[]
    Vs_new = Matrix{Float64}[]
    Rlast = zeros(0)                        # last column of accumulated R'
    for c in 1:mp.nchunks
        for l in (c - 1) * mp.nstab + 1:c * mp.nstab
            apply_B!(w, w, p, @view(fields[l, :]), st.σ, tmp)
        end
        Qi = st.Qs[c]                       # N×Ne
        v = Qi' * w                         # Ne×1 projection
        res = w - Qi * v
        nrm = norm(res)
        q = res ./ nrm
        push!(Qs_new, [Qi q])
        Vp = zeros(Ne + 1, Ne + 1)
        Vp[1:Ne, 1:Ne] = st.Vs[c]
        Vp[1:Ne, end] = v
        Vp[end, end] = nrm
        push!(Vs_new, Vp)
        Rlast = (c == 1) ? [v; nrm] : Vp * Rlast
        w = q
    end
    q_final = vec(w)                        # new orthonormal column at τ=β
    rNe1 = Rlast[end]                       # new diagonal element of R'
    # Schur complement  s = p†q − p†Q (P†Q)^{-1} P†q   (Eq. 4 / S7)
    # （中文）下面四行计算 Schur 补 s：新行列式比中"规范不变"的部分。
    Pq = q_final[st.occ]                    # P†q
    pq = q_final[j]                         # p†q
    pQ = st.Q[j, :]                         # p†Q (row)
    s_schur = pq - dot(pQ, st.A * Pq)
    r_add = s_schur * rNe1
    # blockwise inverse update (Eq. S8), O(Ne^2)
    # （中文）分块求逆：(Ne+1)×(Ne+1) 矩阵 [[P†Q, P†q],[p†Q, p†q]] 的逆，
    # 由旧 A 和 Schur 补 s 显式写出四个块，复杂度 O(Ne²)。
    u = st.A * Pq
    wT = vec(st.A' * pQ)                    # (p†Q A) as column
    A_new = zeros(Ne + 1, Ne + 1)
    A_new[1:Ne, 1:Ne] = st.A + (u * wT') ./ s_schur
    A_new[1:Ne, end] = -u ./ s_schur
    A_new[end, 1:Ne] = -wT ./ s_schur
    A_new[end, end] = 1.0 / s_schur
    R_new = zeros(Ne + 1, Ne + 1)
    R_new[1:Ne, 1:Ne] = st.R
    R_new[:, end] = Rlast
    Q_new = [st.Q q_final]
    occ_new = [st.occ; j]
    return (r=r_add, Qs=Qs_new, Vs=Vs_new, Q=Q_new, R=R_new, A=A_new, occ=occ_new)
end

# ---------------------------------------------------------------------------
# Particle REMOVAL  (SM Sec. I B): remove k-th particle.
# Permute column k to the end (U), then at every stabilization step restore
# upper-triangular form with Givens rotations; truncate the last column/row.
# r_rem = 1/(s · r_Ne)   (Eq. S10).
#
# 【中文推导】删除第 k 个粒子（占据 occ[k]）。思路：
#   (1) 用置换 U 把第 k 列挪到最后一列：P → P U。置换后每个 chunk 的 V_c
#       不再是上三角，而变成上 Hessenberg（次对角元只出现在 (j+1,j)，j≥k）；
#   (2) 自第一个 chunk 起用一串左 Givens 旋转 G_c 恢复上三角：
#       V'_c = G_c·(V_c·G†_{c-1})，同时把逆旋转吸收进基：Q'_c = Q_c·G†_c。
#       因为 G 是正交的，Q'_c 仍列正交，分解结构 Q_c V_c 逐 chunk 保持；
#   (3) 最后一个 chunk 后，被删粒子恰好对应最后一列/行，直接截断即得
#       Ne-1 维的新 Qs/Vs/R。
# 行列式比 r_rem = 1/(s·r_Ne)（SM Eq. (S10)）：r_Ne 是截断前 R 的最后对角元，
# 1/s = [G_n (P†Q)^{-1} U] 的右下角元。新 A' 用 (Ne)×(Ne) 分块逆的 Schur 公式
# （SM Eq. (S12)）从截断前的 M^{-1} 得到。全程 O(nchunks·N·Ne) = O(β·N·Ne/nstab·Ne)。
# ---------------------------------------------------------------------------
function trial_remove(m::Model, mp::MCParams, st::SpinState, k::Int)
    Ne = length(st.occ)
    order = [setdiff(1:Ne, k); k]          # permutation moving column k to last
    Qs_new = Matrix{Float64}[]
    Vs_new = Matrix{Float64}[]
    rots_prev = Tuple{Int,Float64,Float64}[]
    Racc = zeros(Ne, Ne)
    Gacc = zeros(Ne, Ne)                       # product of left rotations at final step
    for c in 1:mp.nchunks
        M = copy(st.Vs[c])
        if c == 1
            M = M[:, order]                    # V1 U  (upper Hessenberg)
        else
            apply_rots_right_inv!(M, rots_prev)   # Vc G†_{c-1}
        end
        rots, Mtri = triangularize!(M, k)      # V'c = Gc (Vc G†_{c-1})
        Qc = copy(st.Qs[c])
        apply_rots_right_inv!(Qc, rots)        # Q'c = Qc G†_c
        push!(Qs_new, Qc)
        push!(Vs_new, Mtri)
        Racc = (c == 1) ? copy(Mtri) : Mtri * Racc
        rots_prev = rots
        if c == mp.nchunks
            Gacc = zeros(Ne, Ne) + I
            apply_rots_left!(Gacc, rots)       # G_n as dense (Ne×Ne)
        end
    end
    rNe = Racc[Ne, Ne]                         # last diagonal of permuted R (before truncation)
    # （中文）下面构造截断前的 [(PU)†Q̃]^{-1} = G_n·A·U：左乘累积旋转 G_n、
    # 右乘置换 U。其右下角元即 1/s（SM Eq. (S10) 下方说明）。
    # M^{-1} = [(PU)† Q̃]^{-1} = G_n (P†Q)^{-1} U      (SM below Eq. S10)
    Uperm = zeros(Ne, Ne)
    for (b, a) in enumerate(order)
        Uperm[a, b] = 1.0
    end
    Minv = Gacc * st.A * Uperm
    sinv = Minv[Ne, Ne]                        # 1/s = bottom-right element
    r_rem = sinv / rNe
    # inverse of the reduced system (Eq. S12)
    if Ne > 1
        A_new = Minv[1:Ne-1, 1:Ne-1] - (Minv[1:Ne-1, Ne:Ne] * Minv[Ne:Ne, 1:Ne-1]) ./ sinv
    else
        A_new = zeros(0, 0)
    end
    Q_new = Qs_new[end][:, 1:Ne-1]
    R_new = Racc[1:Ne-1, 1:Ne-1]
    Qs_t = [Qc[:, 1:Ne-1] for Qc in Qs_new]
    Vs_t = [Vc[1:Ne-1, 1:Ne-1] for Vc in Vs_new]
    occ_new = deleteat!(copy(st.occ), k)
    return (r=r_rem, Qs=Qs_t, Vs=Vs_t, Q=Q_new, R=R_new, A=A_new, occ=occ_new)
end

# ---------------------------------------------------------------------------
# Acceptance helpers: build new SpinState from a trial result
# 【中文说明】接受试探更新：把 trial_add/trial_remove 返回的新因子
# （occ、Qs、Vs、Q、R、A）整体写入 SpinState，完成一次 Fock 态跃迁。
# ---------------------------------------------------------------------------
function accept_trial!(st::SpinState, tr)
    st.occ = tr.occ
    st.Qs = tr.Qs
    st.Vs = tr.Vs
    st.Q = tr.Q
    st.R = tr.R
    st.A = tr.A
    return st
end

# ---------------------------------------------------------------------------
# Auxiliary-field sweep (fast update in the low-rank factor form)
# G(τ) = I - L A_f R̃,  L = B(τ,0)P (N×Ne),  R̃ = P†B(β,τ) (Ne×N),  A_f = (R̃ L)^{-1}.
# A HS flip at (ℓ,i) scales row i of L: ratio  rσ = 1 + Δσ (L[i,:] A_f R̃[:,i]).
# Per-flip cost O(Ne^2); per sweep O(β N Ne^2).
#
# Numerical stabilization (crucial at large β; the naive build-and-unwrap of
# R̃ loses ~cond·eps accuracy and biases the stationary distribution):
# * L side: forward propagation from P, QR re-anchored every chunk. Exact for
#   the *current* fields: accepted flips are folded into L by row rescaling.
# * R̃ side: once per sweep we build an LQ-stabilized factorization of P†B(β,0)
#   chunk-by-chunk from the top and keep only the row-orthonormal anchors Q̃_c.
#   (R̃(τ) involves only layers > τ, whose fields are untouched when slice τ is
#   processed, so the anchors stay valid throughout the sweep.) At each chunk,
#   R̃ is re-anchored to Q̃_{c+1} times a short (≤ nstab layers) product and
#   A_f is recomputed fresh as (R̃ L)^{-1}. The triangular scale factors of
#   both sides cancel exactly in the ratio (gauge invariance), so only these
#   well-conditioned gauge factors are ever used.
# ---------------------------------------------------------------------------
# 【中文详解：本函数是全代码数值上最微妙的部分】
#
# 目标：对 HS 场 s_{l,i} 做 Metropolis 单点翻转 sweep。等时格林函数写成
#   G(τ) = I − L·A_f·R̃，  L = B(τ,0)P (N×Ne)，R̃ = P†B(β,τ) (Ne×N)，A_f = (R̃L)^{-1}。
# 翻转 (ℓ,i) 处场只会让 L 的第 i 行缩放 (1+Δσ)、R̃ 的第 i 列缩放，行列式比
#   rσ = 1 + Δσ·(L[i,:]·A_f·R̃[:,i]) ≡ 1 + Δσ·gii，
# 接受后用 Sherman-Morrison 秩一公式 O(Ne²) 更新 A_f，L 行就地缩放。
#
# ★ 为什么不能朴素地"从头构造 R̃ 再逐片 unwrap（右乘 B_l^{-1}）维护"？
#   B(β,τ) 的条件数随 β 指数增长（cond ~ e^{const·β}）。朴素做法里 R̃ 直接携带
#   这个大动态范围，unwrap 时每步误差被放大到 ~cond(B)·eps 量级；这些误差沿
#   "几乎被抛弃的方向"积累，后续任何 QR 再稳定化都无法修复（QR 只能重新正交化
#   已有的列空间，无法恢复已丢失的尾部分量），导致行列式比有偏、稳态分布错误。
#
# ★ 解决方案（两侧都用"规范因子化"，只用条件良好的因子）：
#   (1) L 侧：从 P 出发向前传播，每个 chunk 边界做一次 QR 重锚定；
#       已被接受的场翻转通过对 L 的第 i 行就地缩放精确折叠进去，因此 L 始终
#       对应"当前"场构型。
#   (2) R̃ 侧：每个 sweep 开头自顶向下逐 chunk 构建 P†B(β,0) 的 **LQ 锚定因子化**，
#       只保留行正交的锚点 Q̃_c（三角部分全部丢弃）。因为处理时间片 τ 时 R̃(τ)
#       只涉及层号 > τ 的场，而这些场在本 sweep 中尚未被触及，所以锚点在整轮
#       sweep 内保持有效。
#   (3) 每进入一个 chunk，把 R̃ 重锚为 Q̃_{c+1}·(本 chunk 内 ≤ nstab 片的短乘积)，
#       并**新鲜地**计算 A_f = inv(R̃·L)。短乘积条件数可控，A_f 因此干净。
#   (4) L 与 R̃ 必须对齐**同一个 τ**：chunk c 内 L 传播到 τ=l0=(c-1)·nstab，
#       故 R̃ 锚定乘积是 l1:-1:(l0+1)（从本 chunk 顶端 l1 逐片右乘到 l0+1），
#       使 R̃_gauge(τ=l0) = Q̃_{c+1}·B_{l1}⋯B_{l0+1}，两侧同为 τ=l0 的因子。
#   (5) 规范不变性：行列式比 gii = L[i,:]·A_f·R̃[:,i] 中，L 侧的 QR 三角因子
#       与 R̃ 侧的 LQ 三角因子在 A_f=(R̃L)^{-1} 中精确相消，所以任意重锚定
#       （不同的规范选择）给出完全相同的物理比值——这正是可以随便换规范、
#       只保留良态因子的理论依据。
function field_sweep!(m::Model, p::Propagator, mp::MCParams,
                      fields::AbstractMatrix{Int8}, sts::Vector{SpinState}, rng::AbstractRNG)
    N = m.N
    Lτ = mp.Ltau
    nst = mp.nstab
    nc = cld(Lτ, nst)
    acc = 0

    # --- right-side LQ-stabilized anchors Q̃_c (per spin), from current fields.
    #     P†B(β,0) = (lower triangulars) · Q̃_1 with Q̃_c row-orthonormal.
    #     Qr[k][c] = Q̃_c (c = 1..nc);  Qr[k][nc+1] = P†.
    # （中文）构建右侧锚点：从 X=P† 出发，自 τ=β 向 τ=0 逐 chunk 右乘 B_l；
    # 每过一个 chunk 对 X† 做 QR（等价于对 X 做 LQ），只保留行正交因子 Q̃_c。
    # 注意 qr(X') 的 Q† 即 X 的行正交基。三角因子被丢弃——它们正是病态的来源。
    Qr = Vector{Vector{Matrix{Float64}}}()
    for st in sts
        P = selection_matrix(N, st.occ)
        qs = Matrix{Float64}[]
        X = Matrix(P')                       # Ne×N
        tmp = similar(X)
        for c in nc:-1:1
            ltop = min(c * nst, Lτ)
            lbot = (c - 1) * nst + 1
            for l in ltop:-1:lbot
                apply_B_right!(X, p, @view(fields[l, :]), st.σ, tmp)
            end
            F = qr(Matrix(X'))
            push!(qs, Matrix(F.Q)')          # Q̃_c (row-orthonormal)
            X = Matrix(F.Q)'
        end
        reverse!(qs)                         # qs[c] = Q̃_c
        push!(qs, Matrix(P'))                # Q̃_{nc+1} = P†
        push!(Qr, qs)
    end

    # --- per-spin working factors
    Ls   = Matrix{Float64}[copy(selection_matrix(N, st.occ)) for st in sts]  # N×Ne
    Rrs  = Matrix{Float64}[zeros(length(st.occ), N) for st in sts]           # Ne×N
    Afs  = Matrix{Float64}[zeros(length(st.occ), length(st.occ)) for st in sts]
    tmps = [similar(Ls[1]), similar(Ls[1])]
    tmpr = [similar(Rrs[1]), similar(Rrs[1])]

    for c in 1:nc
        l0 = (c - 1) * nst
        l1 = min(c * nst, Lτ)
        # --- chunk (re-)anchoring of both gauge factors + fresh A_f
        #     (both factors must correspond to the SAME slice τ = l0)
        # （中文）chunk 入口处的重锚定：
        #   R̃ 侧：从锚点 Q̃_{c+1} 出发，右乘本 chunk 内的短乘积 B_{l1}⋯B_{l0+1}
        #         （循环 l = l1:-1:(l0+1)），得到 τ=l0 处条件良好的 R̃ 规范因子；
        #   L 侧：对携带了上轮所有已接受翻转的 Ls 做一次 QR，丢弃三角因子；
        #   然后**新鲜**计算 A_f = inv(R̃·L)。两侧必须对应同一 τ=l0，否则
        #   A_f 失去 (R̃L)^{-1} 的含义，gii 比值就错了。
        for (k, st) in enumerate(sts)
            length(st.occ) == 0 && continue
            Rr = copy(Qr[k][c + 1])          # Q̃_{c+1}
            tmp = similar(Rr)
            for l in l1:-1:(l0 + 1)          # R̃_gauge(τ=l0) = Q̃_{c+1} B_{l1}⋯B_{l0+1}
                apply_B_right!(Rr, p, @view(fields[l, :]), st.σ, tmp)
            end
            Rrs[k] = Rr
            if c > 1
                F = qr(Ls[k])
                Ls[k] = Matrix(F.Q)
            end
            Afs[k] = inv(Rrs[k] * Ls[k])
        end
        # --- slices within the chunk
        # （中文）chunk 内逐时间片推进：L 左乘 B_l（τ: l→l+1 方向推进一格），
        # R̃ 右乘 B_l^{-1}（同步回退一格），保持两侧始终对齐同一 τ。
        # 短距离 unwrap（≤ nstab 片）误差可控，不会出现长 β 的 cond·eps 问题。
        for l in (l0 + 1):l1
            s_l = @view(fields[l, :])
            for (k, st) in enumerate(sts)
                length(st.occ) == 0 && continue
                apply_B!(Ls[k], Ls[k], p, s_l, st.σ, tmps[k])
                apply_Binv_right!(Rrs[k], p, s_l, st.σ, tmpr[k])
            end
            # flip attempts at all sites
            # （中文）对片上所有格点尝试翻转 s→-s。Δσ = exp(σλ(sp−s))−1；
            # 每个自旋分量的行列式比 rσ = 1+Δσ·gii，其中
            #   gii = L[i,:]·A_f·R̃[:,i]
            # 是规范不变的（L、R̃ 两侧的三角规范因子在 A_f 中相消）。
            # 总比值 rtot = ∏_σ rσ，Metropolis 按 |rtot| 接受（符号进入权重符号）。
            for i in 1:N
                s = fields[l, i]
                sp = -s
                rtot = 1.0
                ds = zeros(2)
                rs = ones(2)
                for (k, st) in enumerate(sts)
                    sgn = st.σ == SPIN_UP ? 1.0 : -1.0
                    d = exp(sgn * p.lam * (sp - s)) - 1.0
                    ds[k] = d
                    if length(st.occ) > 0
                        gii = dot(@view(Ls[k][i, :]), Afs[k] * @view(Rrs[k][:, i]))
                        rs[k] = 1.0 + d * gii
                        rtot *= rs[k]
                    end
                end
                if rand(rng) < abs(rtot)
                    # Sherman-Morrison update of A_f = (R̃ L)^{-1} and row scaling of L
                    # （中文）接受翻转后的秩一更新：翻转使 L 的第 i 行缩放
                    # L[i,:] → (1+Δσ)·L[i,:]，于是乘积发生秩一变化
                    # R̃L → R̃L + u·v'，u = Δσ·R̃[:,i]，v' = L[i,:]（旧行）。
                    # Sherman-Morrison：A_f → A_f − (A_f u)(v' A_f)/rσ，O(Ne²)；
                    # 同时把 L 的第 i 行就地缩放 (1+Δσ)，使 L 精确对应当前场。
                    for (k, st) in enumerate(sts)
                        if length(st.occ) > 0
                            u = ds[k] .* @view(Rrs[k][:, i])     # Δ R̃ col
                            v = copy(Ls[k][i, :])                # old L row
                            Afu = Afs[k] * u
                            Afs[k] .-= (Afu * (v' * Afs[k])) ./ rs[k]
                            Ls[k][i, :] .*= (1.0 + ds[k])
                        end
                    end
                    fields[l, i] = sp
                    acc += 1
                end
            end
        end
    end
    return acc
end

# ---------------------------------------------------------------------------
# Fock-state sweep: canonical sampling via particle-hole swaps
# One swap = removal (trial) + addition (trial); Metropolis on |r_rem r_add|.
# 【中文说明】Fock 态 sweep：正则系综内通过"粒子-空穴交换"遍历 Fock 空间
# （保持 Nσ 不变）。每次尝试 = 随机选一个占据轨道 k 做 trial_remove，
# 再随机选一个空格点 j 做 trial_add，接受率为 |r_rem·r_add|。
# 每个自旋分量各尝试 Ne 次交换。接受后用 accept_trial! 一次性替换全部因子。
# occflags 记录占据情况用于快速选空格点。
# ---------------------------------------------------------------------------
function fock_sweep!(m::Model, p::Propagator, mp::MCParams,
                     fields::AbstractMatrix{Int8}, sts::Vector{SpinState}, rng::AbstractRNG,
                     occflags::Vector{BitVector})
    acc = 0
    att = 0
    for st in sts
        Ne = length(st.occ)
        Ne == 0 && continue
        for _ in 1:Ne
            att += 1
            k = rand(rng, 1:Ne)
            # random empty site
            j = rand(rng, 1:m.N)
            tries = 0
            while occflags[st.σ][j]
                j = rand(rng, 1:m.N)
                tries += 1
                tries > 10 * m.N && break
            end
            occflags[st.σ][j] && continue
            tr_rem = trial_remove(m, mp, st, k)
            st_tmp = SpinState(st.σ, tr_rem.occ, tr_rem.Qs, tr_rem.Vs, tr_rem.Q, tr_rem.R, tr_rem.A)
            tr_add = trial_add(m, p, mp, fields, st_tmp, j)
            r = tr_rem.r * tr_add.r
            if rand(rng) < abs(r)
                oldsite = st.occ[k]
                accept_trial!(st, tr_add)
                occflags[st.σ][oldsite] = false
                occflags[st.σ][j] = true
                acc += 1
            end
        end
    end
    return acc, att
end

# ---------------------------------------------------------------------------
# Measurements at τ = β:  M = Q A P†,  ⟨c_j† c_i⟩ = M_{ij}
# QA = Q*A is N×Ne: column a corresponds to site occ[a].
# 【中文说明】τ=β 处的"斜投影"格林函数（oblique projected Green's function）：
#   M = Q·A·P† = Q(P†Q)^{-1}P†  （N×N），
# 物理内容为 ⟨c_j† c_i⟩ = M_{ij}。它是向 Q 列空间的斜投影（沿 P 的零空间），
# M² = M，本征值 0/1。由于 P 是选择矩阵，只需算 QA = Q·A（N×Ne）：
# QA 的第 a 列对应格点 occ[a]，M_{ij} = QA[i, pos[j]]。
# ---------------------------------------------------------------------------
green_cols(st::SpinState) = st.Q * st.A

# Observables：观测量累积器。所有量都以"符号加权"方式累加
# （Σ_config sign·O），最后在 finalize_obs 中除以 Σ sign。
mutable struct Observables
    nmeas::Int
    sign_acc::Float64
    density::Vector{Float64}          # ⟨n_i⟩ (Fock estimator, summed over configs)
    nn::Matrix{Float64}               # ⟨n_i n_j⟩ Fock-product estimator
    szz::Matrix{Float64}              # ⟨Sz_i Sz_j⟩
    sxy::Matrix{Float64}              # ⟨Sx_i Sx_j + Sy_i Sy_j⟩
end

function Observables(N::Int)
    return Observables(0, 0.0, zeros(N), zeros(N, N), zeros(N, N), zeros(N, N))
end

# 【中文说明】当前权重符号：W = ∏_σ det[P†QR] = ∏_σ det(P†Q)·det(R)，
# 取两自旋分量符号之积。DQMC 的符号问题即体现为平均符号 ⟨sign⟩ 的衰减。
"""Current sign of the weight W = prod_σ det[P†QR]."""
function weight_sign(sts::Vector{SpinState})
    s = 1.0
    for st in sts
        Ne = length(st.occ)
        if Ne > 0
            s *= sign(det(st.Q[st.occ, :]) * det(st.R))
        end
    end
    return s
end

# 【中文说明】一次测量（τ=β，等时关联）：
#   density、nn、szz 用 Fock 估计量——正则系综中 n_{iσ} 就是 0/1 占据数，
#   直接由 occ 列表读出，无需格林函数；
#   sxy（自旋交换）必须用格林函数：Wick 分解给出
#     ⟨S⁺_i S⁻_j + S⁻_i S⁺_j⟩/2 类项 = -½(M↑_{ji}M↓_{ij} + M↓_{ji}M↑_{ij})  (i≠j)，
#     对角项为 ½(n↑+n↓−2n↑n↓)。
#   所有累加都乘以 wsign（符号加权采样）。
function measure!(obs::Observables, sts::Vector{SpinState})
    obs.nmeas += 1
    wsign = weight_sign(sts)
    obs.sign_acc += wsign
    N = length(obs.density)
    nf = [zeros(N), zeros(N)]
    for st in sts
        nf[st.σ][st.occ] .= 1.0
    end
    n_tot = nf[1] + nf[2]
    sz = 0.5 .* (nf[1] - nf[2])
    obs.density .+= wsign .* n_tot
    obs.nn .+= wsign .* (n_tot * n_tot')
    obs.szz .+= wsign .* (sz * sz')
    # xy part: offsite exchange + onsite
    QA = [green_cols(st) for st in sts]
    pos = [zeros(Int, N), zeros(Int, N)]
    for st in sts
        for (a, i) in enumerate(st.occ)
            pos[st.σ][i] = a
        end
    end
    sxy = zeros(N, N)
    # onsite: (1/2)(n↑+n↓−2n↑n↓)
    @inbounds for i in 1:N
        sxy[i, i] = 0.5 * (nf[1][i] + nf[2][i] - 2 * nf[1][i] * nf[2][i])
    end
    # exchange i≠j: -(1/2)[ M↑_{ji} M↓_{ij} + M↓_{ji} M↑_{ij} ]
    # loop over i in occ↑, j in occ↓: term1 -> (i,j); term2 (spins swapped) -> (j,i)
    # （中文）交换估计量：双重循环 i∈occ↑、j∈occ↓，pij = M↑_{ji}·M↓_{ij}；
    # 由于 M↓ 与 M↑ 交换的另一项与此对称，直接对 (i,j) 与 (j,i) 各减 ½·pij，
    # 一次循环同时计入两项，避免重复遍历。
    for (ai, i) in enumerate(sts[1].occ)
        qai = @view QA[1][:, ai]
        for (aj, j) in enumerate(sts[2].occ)
            if i != j
                pij = qai[j] * QA[2][i, aj]      # M↑_{ji} M↓_{ij}
                sxy[i, j] -= 0.5 * pij
                sxy[j, i] -= 0.5 * pij
            end
        end
    end
    obs.sxy .+= wsign .* sxy
    return obs
end

# 【中文说明】收尾：符号加权归一化 ⟨O⟩ = (Σ sign·O)/(Σ sign)。
# 返回平均符号 sign = (Σ sign)/nmeas 及各观测量。
"""Finalize: normalize by sign-weighted counts (⟨O⟩ = Σ sign·O / Σ sign)."""
function finalize_obs(obs::Observables)
    s = obs.sign_acc
    return (sign=s / obs.nmeas,
            density=obs.density ./ s,
            nn=obs.nn ./ s,
            szz=obs.szz ./ s,
            sxy=obs.sxy ./ s)
end

# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------
# MC：蒙卡模拟总状态——模型、传播子、虚时参数、HS 场（Lτ×N，每片每格点 ±1）、
# 两个自旋分量的 SpinState（即当前 Fock 态 η 及其稳定化因子）、占据标记。
mutable struct MC
    m::Model
    p::Propagator
    mp::MCParams
    fields::Matrix{Int8}            # Lτ × N
    sts::Vector{SpinState}
    occflags::Vector{BitVector}
end

# 【中文说明】MC 构造器：给定 β、Δτ、稳定化间隔 nstab 与初始 Fock 态
# occs = [occ↑, occ↓]，随机初始化 HS 场（±1），并对两个自旋各做一次
# full_propagation! 建立初始稳定化分解。
function MC(m::Model, beta::Float64, dt::Float64, nstab::Int,
            occs::Vector{Vector{Int}}; use_dense::Bool=false, rng::AbstractRNG=Random.default_rng())
    mp = MCParams(beta, dt, nstab)
    p = make_propagator(m, dt; use_dense=use_dense)
    fields = rand(rng, Int8[-1, 1], mp.Ltau, m.N)
    sts = [full_propagation!(m, p, mp, fields, σ, occs[σ]) for σ in 1:2]
    occflags = [falses(m.N) for _ in 1:2]
    for σ in 1:2
        occflags[σ][occs[σ]] .= true
    end
    return MC(m, p, mp, fields, sts, occflags)
end

# 【中文说明】一次完整迭代（sweep）的三个步骤：
#   1. field_sweep!      —— 固定 Fock 态 η，扫描全部 HS 场（β·N 次翻转尝试，
#      每次 O(Ne²)），共 O(β·N·Ne²)；
#   2. full_propagation! —— 场已更新，对每个自旋重新做稳定化 QR 传播，
#      刷新 Qs/Vs/Q/R/A（O(β·N·Ne + nchunks·N·Ne²)），保证 Fock 更新所依赖
#      的因子精确对应当前场；
#   3. fock_sweep!       —— 固定场，做 Ne 次粒子-空穴交换尝试，每次 O(β·N·Ne)，
#      共 O(β·N·Ne²)。
# 总复杂度 O(β·N·Ne²)每 sweep（两个自旋分量同阶）。
"""One full iteration: field sweep -> repropagate -> Fock sweep."""
function sweep!(mc::MC, rng::AbstractRNG)
    field_sweep!(mc.m, mc.p, mc.mp, mc.fields, mc.sts, rng)
    for σ in 1:2
        mc.sts[σ] = full_propagation!(mc.m, mc.p, mc.mp, mc.fields, σ, mc.sts[σ].occ)
    end
    acc, att = fock_sweep!(mc.m, mc.p, mc.mp, mc.fields, mc.sts, rng, mc.occflags)
    return acc, att
end

end # module
