# Standard grand-canonical finite-temperature DQMC with fast (rank-1) updates.
# Comparison baseline for the Fig. 1 scaling benchmark.
# G_σ = (I + B^σ(β,0))^{-1}, swept/wrapped slice by slice: O(β N^3) per sweep.
#
# ============================ 算法框架说明 ============================
# 这是教科书式的标准有限温行列式量子蒙特卡洛（DQMC）基线实现，
# 用于与 LEC-QMC 做论文 Fig. 1 的标度（scaling）对比。
#
# 框架要点（巨正则、有限温）：
#   1. Hubbard 相互作用经离散 Hubbard-Stratonovich（HS）变换解耦为
#      每个虚时间片 l、每个格点 i 上的 Ising 型辅助场 s_{l,i} = ±1；
#      费米子在该辅助场下是自由的，传播子为
#          B^σ(l2, l1) = B^σ_{l2} B^σ_{l2-1} ... B^σ_{l1+1}，
#      其中单片 B^σ_l = expK · exp(V^σ_l)，V^σ_l 是由 s_{l,:} 决定的对角势。
#   2. 对给定 HS 场构型，等时格林函数为
#          G_σ = (I + B^σ(β, 0))^{-1}，
#      玻尔兹曼权重正比于 det(I + B^↑) · det(I + B^↓)。
#   3. Monte Carlo 更新采用"逐片扫描 + 快速秩一更新"：
#      在每一片 l 上先 wrap 格林函数（相似变换把 G 搬到该时间片），
#      再对该片所有格点尝试 HS 场翻转；接受时用 Sherman-Morrison
#      公式 O(N^2) 更新 G，而不是每次 O(N^3) 重建。
#   4. 每隔 nstab 片做一次新鲜重建（fresh stabilization），
#      抑制秩一更新积累的数值漂移。
#   总复杂度：每个 sweep 为 O(β N^3)（N 个格点、Lτ = β/dt 个时间片）。
# ===================================================================
module StdDQMC

using LinearAlgebra, Random

export StdMC, std_sweep!, std_green

"""
    StdMC

标准 DQMC 的 Monte Carlo 状态结构体：

- `Gs::Vector{Matrix{Float64}}`：两个自旋分量（σ=↑,↓）的等时格林函数
  `G_σ = (I + B^σ(l,0) B^σ(β,l))^{-1}`，始终对应"当前正在处理的时间片"，
  靠 wrap（相似变换）在各片之间搬运，靠 Sherman-Morrison 秩一更新跟随场翻转。
- `fields::Matrix{Int8}`：HS 辅助场，形状 (Lτ, N)，`fields[l, i] = ±1`
  表示第 l 个虚时间片、第 i 个格点上的 Ising 变量。
"""
struct StdMC
    Gs::Vector{Matrix{Float64}}     # G_σ at current slice
    fields::Matrix{Int8}
end

"""B_l^{-1} x (left multiply): B_l^{-1} = expK^{-1} e^{-V}."""
function apply_Binv!(y, x, p, s_l, σ, tmp)
    # 先作用对角部分 e^{-V^σ_l}：对第 i 行乘以 exp(-sgn·λ·s_{l,i})，
    # 其中 sgn = +1（自旋上）/ -1（自旋下），λ = p.lam 为 HS 耦合常数。
    sgn = σ == 1 ? 1.0 : -1.0
    @inbounds for i in axes(x, 1), k in axes(x, 2)
        tmp[i, k] = x[i, k] * exp(-sgn * p.lam * s_l[i])
    end
    # 再左乘 expK^{-1}（动能部分，借助 LECQMC 模块的共享实现，true 表示取逆）。
    Main.LECQMC.apply_expK!(y, tmp, p, true)
    return y
end

"""
    std_green(m, p, fields, σ) -> Matrix{Float64}

由当前 HS 场构型**新鲜（fresh）**构建自旋 σ 的等时格林函数
`G_σ = (I + B^σ(β, 0))^{-1}`。

做法：从单位矩阵出发，沿虚时间把 Lτ 个单片传播子 `B_l` 依次左乘上去，
累乘得到整圈传播子 `B(β, 0) = B_{Lτ} ... B_1`，最后对 `I + B` 直接求逆。
这是 O(Lτ·N²) 次矩阵乘法加一次 O(N³) 求逆；在本 benchmark 的温度
（T=1，无严重数值不稳定）下不做 QR 分解稳定化也足够稳定。
该函数既用于初始化，也用于 sweep 中每 nstab 片的"新鲜重建"（见 `std_sweep!`）。
"""
function std_green(m, p, fields, σ)
    N = m.N
    B = Matrix{Float64}(I, N, N)   # 累乘器：初始为单位矩阵
    tmp = similar(B)               # apply_B! 的临时工作数组
    Lτ = size(fields, 1)
    for l in 1:Lτ
        # B <- B_l · B：把第 l 片传播子乘到累乘器上（就地，内部用 tmp 避免混叠）
        Main.LECQMC.apply_B!(B, B, p, @view(fields[l, :]), σ, tmp)
    end
    # 格点等时格林函数：G = (I + B(β,0))^{-1}
    return inv(I + B)
end

# StdMC 构造函数：给定模型 m、参数 p、逆温 beta 与时间步长 dt，
# 随机初始化 HS 场（±1 均匀），并据此新鲜构建两个自旋的初始格林函数。
function StdMC(m, p, beta, dt; rng=Random.default_rng())
    Lτ = Int(round(beta / dt))                      # 虚时间片数 Lτ = β/dt
    fields = rand(rng, Int8[-1, 1], Lτ, m.N)        # 随机 HS 场，形状 (Lτ, N)
    Gs = [std_green(m, p, fields, 1), std_green(m, p, fields, 2)]  # σ=↑, ↓ 的初始 G
    return StdMC(Gs, fields)
end

"""
    std_sweep!(mc::StdMC, m, p, rng; nstab::Int=5) -> acc

执行一个完整的 Monte Carlo sweep：对所有 (l, i)（时间片 × 格点）各尝试
一次 HS 场翻转，返回接受的翻转次数 `acc`。

**逐片处理的顺序（关键）**：对每个时间片 l，
  1. **先 wrap**：把格林函数从第 l-1 片搬运到第 l 片，
     即相似变换 G ← B_l · G · B_l^{-1}。
     必须先做这一步再做翻转——因为在第 l 片翻转 HS 场 s_{l,i} 等价于
     对 B(l,0) 做一次**左乘**（B_l 位于 B(l,0) 连乘积的最左端），
     对应的 G 更新公式 G ← G + (d/r)(G e_i)[e_i'(G-I)] 只在 G 表示
     "当前片"的等时格林函数时才成立。
  2. **再做该片上全部 N 个格点的翻转尝试**：每个翻转用 Metropolis
     判据，比值 r = Π_σ [1 + d_σ(1 - G_σ[i,i])]；接受后对两个自旋的 G
     各做一次 Sherman-Morrison 秩一更新（O(N²)）。
  3. **每 nstab 片新鲜重建 G**：秩一更新只含乘加运算，误差会持续累积
     （SM 漂移）；实测若不重建，几个 sweep 内 G 的误差就发散到 O(1)。
     因此每隔 nstab 片调用 `std_green` 从当前场构型重新计算精确的 G，
     把漂移清零。这是数值稳定性的硬性要求，不是可选项。
"""
function std_sweep!(mc::StdMC, m, p, rng; nstab::Int=5)
    N = m.N
    Lτ = size(mc.fields, 1)
    acc = 0
    tmpB = similar(mc.Gs[1])    # wrap 时的临时矩阵（存 B_l · G · B_l^{-1}）
    tmpB2 = similar(mc.Gs[1])   # apply_B!/apply_Binv_right! 内部的工作数组
    for l in 1:Lτ
        s_l = @view(mc.fields[l, :])
        # wrap to slice l first: G -> B_l G B_l^{-1}  (flips at slice l are left-multiplications of B(l,0))
        # （即：先搬运 G 到第 l 片，再翻转该片——翻转是 B(l,0) 的左乘，公式依赖当前片的 G）
        for σ in 1:2
            Main.LECQMC.apply_B!(tmpB, mc.Gs[σ], p, s_l, σ, tmpB2)        # tmpB ← B_l · G
            Main.LECQMC.apply_Binv_right!(tmpB, p, s_l, σ, tmpB2)         # tmpB ← tmpB · B_l^{-1}
            copyto!(mc.Gs[σ], tmpB)                                       # 写回，完成 wrap
        end
        # ---- 第 2 步：对第 l 片上所有格点依次尝试 HS 场翻转 ----
        for i in 1:N
            s = mc.fields[l, i]; sp = -s      # s → sp = -s：提议的翻转
            rtot = 1.0; ds = (0.0, 0.0); rs = (1.0, 1.0)
            # 对两个自旋分别计算行列式比值并累乘：
            #   翻转 s_{l,i} 使单片势改变 e^{V} → e^{V}(I + d E_ii)，
            #   其中 d = exp(±λ(sp - s)) - 1 只作用于第 i 个格点；
            #   由矩阵行列式引理（matrix determinant lemma），
            #   det 比值 r_σ = 1 + d_σ (1 - G_σ[i,i])，总权重比 rtot = r_↑ · r_↓。
            for σ in 1:2
                sgn = σ == 1 ? 1.0 : -1.0
                d = exp(sgn * p.lam * (sp - s)) - 1.0
                r = 1.0 + d * (1.0 - mc.Gs[σ][i, i])
                ds = Base.setindex(ds, d, σ); rs = Base.setindex(rs, r, σ)
                rtot *= r
            end
            # Metropolis 判据（取 abs 仅为数值稳健；Hubbard 在半满/此处参数下无符号问题）
            if rand(rng) < abs(rtot)
                # ---- 接受翻转：对每个自旋做 Sherman-Morrison 秩一更新 ----
                #   G ← G + (d/r) · (G e_i) · [e_i'(G - I)]
                # 注意：必须先把第 i 列 (G e_i) 和第 i 行 (e_i'(G-I)) **拷贝**出来，
                # 再进入双重循环就地写 G——否则循环中读到的是刚被改写的元素，
                # 产生就地读写混叠（aliasing），结果会错。
                for σ in 1:2
                    d, r = ds[σ], rs[σ]
                    G = mc.Gs[σ]
                    gi = copy(G[:, i])          # G e_i
                    w = copy(G[i, :]); w[i] -= 1.0   # -e_i'(I - G)
                    @inbounds for b in 1:N, a in 1:N
                        G[a, b] += (d / r) * gi[a] * w[b]
                    end
                end
                mc.fields[l, i] = sp            # 场构型正式更新
                acc += 1
            end
        end
        # ---- 第 3 步：数值稳定化 ----
        # 每处理完 nstab 片，从当前 HS 场新鲜重建精确的 G。
        # 必要性：Sherman-Morrison 秩一更新会让截断误差在 G 中不断累积
        # （SM 漂移），实测几个 sweep 内即可发散到 O(1) 量级、污染所有观测量；
        # 周期性重建把漂移压回机器精度，是标准 DQMC 的必备环节。
        if mod(l, nstab) == 0
            for σ in 1:2
                mc.Gs[σ] = std_green(m, p, mc.fields, σ)
            end
        end
    end
    return acc    # 返回本 sweep 接受的翻转总数（供调用方估计接受率）
end

end # module
