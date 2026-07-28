# Exact diagonalization in the canonical (Nup, Ndn) sector for small Hubbard systems.
#
# ============================ 模块说明 ============================
# 小型 Hubbard 体系的精确对角化（ED）参考实现，用作 Monte Carlo（DQMC /
# LEC-QMC）结果的 benchmark 对照基准。
#
# 方法概要：
#   1. 在**固定粒子数**（正则系综，N↑ 个自旋上、N↓ 个自旋下粒子）的
#      Fock 空间中构建哈密顿矩阵。基矢取为 |U⟩⊗|D⟩ 的直积形式，
#      其中 U、D 分别是 N 个格点中自旋上/下粒子所占据格点的有序子集；
#      希尔伯特空间维数 dim = C(N, N↑) · C(N, N↓)，只适用于很小的体系。
#   2. 哈密顿 H = K + U·D：动能（跳跃）项由单粒子矩阵 K[a,b] 经
#      Jordan-Wigner 符号规则填入；Hubbard 相互作用 U·Σ_i n_i↑ n_i↓
#      在该占据数基下是对角的。
#   3. 对 H 做全谱对角化，用 Gibbs 权重 w_n = exp(-β(E_n - E_0))
#      计算**有限温**热平均：配分函数 Z、密度、密度-密度关联 nn、
#      自旋关联 szz 与 sxy 等，作为检验 QMC 正确性的精确答案。
# ==================================================================
module EDHubbard

using LinearAlgebra

export ed_thermal, subsets

"""
    subsets(N, k) -> Vector{Vector{Int}}

生成 {1,...,N} 的全部 k 元子集（每个子集内部升序排列），即组合数索引：
返回列表长度为 C(N, k)，每个元素是一个有序占据格点集合，用作
固定粒子数 Fock 空间中某一自旋分量的基矢标签。
例如 subsets(4, 2) = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]。
"""
function subsets(N::Int, k::Int)
    k == 0 && return [Int[]]          # k=0：空占据（该自旋分量无粒子）
    out = Vector{Int}[]
    # 回溯递归：从 start 开始逐个选格点加入 cur，选满 k 个即得一个基矢
    function rec(start, cur)
        if length(cur) == k
            push!(out, copy(cur)); return
        end
        for x in start:N
            push!(cur, x); rec(x + 1, cur); pop!(cur)
        end
    end
    rec(1, Int[])
    return out
end

"""Sign for c†_a c_b on a sorted occupied same-spin set (JW)."""
function jw_sign(set::Vector{Int}, b::Int, a::Int)
    # Jordan-Wigner 符号：算符 c†_a c_b 把粒子从 b 搬到 a 时，
    # 费米子反对易关系给出的符号为 (-1)^{a 与 b 之间已占据的同自旋模式数}。
    lo, hi = minmax(a, b)
    cnt = count(x -> lo < x < hi, set)   # 统计严格位于 (a,b) 之间的占据数
    return iseven(cnt) ? 1.0 : -1.0
end

"""
    hop_kinetic!(H, ups, dns, nu, nd, N, a, b, isup, t)

向哈密顿矩阵 H 中填入一个跳跃项 `t · c†_{aσ} c_{bσ}`（isup 指定 σ=↑ 或 ↓）。

对给定自旋分量的每个基矢（占据集合），若 b 被占据且 a 为空，则该算符
把粒子从 b 搬到 a，得到新占据集合；用 `findfirst` 在基矢表中查出其索引，
并在对应的矩阵元上累加 `t · (JW 符号)`。另一自旋分量在此过程中不变，
因此对它的全部基矢做循环（张量积结构）。

**基矢线性索引约定**：总基矢 |iu, id⟩ = |U_iu⟩⊗|D_id⟩ 按
`idx = iu + (id - 1) * nu` 编号（自旋上分量为快指标），
其中 nu = C(N, N↑)、nd = C(N, N↓)。
"""
function hop_kinetic!(H, ups, dns, nu, nd, N, a, b, isup::Bool, t::Float64)
    if isup
        # ---- 自旋上分量的跳跃 c†_{a↑} c_{b↑} ----
        for (iu, Uset) in enumerate(ups)
            (b in Uset && !(a in Uset)) || continue   # 要求 b 占据、a 为空
            Unew = sort([setdiff(Uset, [b]); a])      # 新占据集合：b→a
            iu2 = findfirst(isequal(Unew), ups)       # 查新基矢的索引
            iu2 === nothing && continue
            s = jw_sign(Uset, b, a)                   # Jordan-Wigner 符号
            for id in 1:nd                            # 自旋下分量不变，遍历其全部基矢
                H[iu2 + (id - 1) * nu, iu + (id - 1) * nu] += t * s
            end
        end
    else
        # ---- 自旋下分量的跳跃 c†_{a↓} c_{b↓}（结构同上，角色互换） ----
        for (id, Dset) in enumerate(dns)
            (b in Dset && !(a in Dset)) || continue
            Dnew = sort([setdiff(Dset, [b]); a])
            id2 = findfirst(isequal(Dnew), dns)
            id2 === nothing && continue
            s = jw_sign(Dset, b, a)
            for iu in 1:nu
                H[iu + (id2 - 1) * nu, iu + (id - 1) * nu] += t * s
            end
        end
    end
end

"""Exchange matrix for (1/2)(S+_i S-_j + S-_i S+_j), i≠j, with full JW signs.
S+_i S-_j = c†i↑ ci↓ c†j↓ cj↑ (rightmost applied first)."""
function exchange_matrix(ups, dns, nu, nd, N, i::Int, j::Int)
    dim = nu * nd
    M = zeros(dim, dim)
    Nup = length(ups[1])   # 自旋上粒子数（计算 JW 符号时要用到模式计数）
    # 遍历全部基矢 |Uset, Dset⟩，分别填入两个自旋翻转通道的矩阵元
    for (iu, Uset) in enumerate(ups), (id, Dset) in enumerate(dns)
        # ---- 通道 1：S⁺_i S⁻_j，要求 j 处有 ↑、i 处有 ↓，且两处单占据 ----
        # 作用后变为 i 处有 ↑、j 处有 ↓。四个费米子算符从右到左依次作用，
        # 每步按"其下方（模式编号更小）的占据数"给出 Jordan-Wigner 符号
        # （模式编号约定：1..N 为自旋上，N+1..2N 为自旋下）。
        if (j in Uset) && (i in Dset) && !(i in Uset) && !(j in Dset)
            # step1: c_{j↑}: sign (-1)^{#{u in U: u<j}}
            s1 = iseven(count(u -> u < j, Uset)) ? 1.0 : -1.0
            U1 = setdiff(Uset, [j])
            # step2: c†_{j↓} (mode N+j): modes below: all U1 + #{d in D: d<j}
            s2 = iseven(length(U1) + count(d -> d < j, Dset)) ? 1.0 : -1.0
            D1 = sort([Dset; j])
            # step3: c_{i↓} (mode N+i): modes below: U1 + #{d in D1: d<i}
            s3 = iseven(length(U1) + count(d -> d < i, D1)) ? 1.0 : -1.0
            D2 = setdiff(D1, [i])
            # step4: c†_{i↑} (mode i): modes below: #{u in U1: u<i}
            s4 = iseven(count(u -> u < i, U1)) ? 1.0 : -1.0
            Unew = sort([U1; i])
            Dnew = sort(D2)
            # 查末态基矢索引；四步符号相乘后与系数 1/2 一起累加进矩阵
            iu2 = findfirst(isequal(Unew), ups)
            id2 = findfirst(isequal(Dnew), dns)
            (iu2 === nothing || id2 === nothing) && continue
            s = s1 * s2 * s3 * s4
            M[iu2 + (id2 - 1) * nu, iu + (id - 1) * nu] += 0.5 * s
        end
        # S-_i S+_j term: requires i in U (up), j in D (down); maps to j-up, i-down
        if (i in Uset) && (j in Dset) && !(j in Uset) && !(i in Dset)
            # c_{j↓} (mode N+j), then c†_{j↑} (mode j), then c_{i↑}, then c†_{i↓} (mode N+i)
            r1 = iseven(Nup + count(d -> d < j, Dset)) ? 1.0 : -1.0
            D1 = setdiff(Dset, [j])
            r2 = iseven(count(u -> u < j, Uset)) ? 1.0 : -1.0
            U1 = sort([Uset; j])
            U2 = setdiff(U1, [i])
            r3 = iseven(count(u -> u < i, U1)) ? 1.0 : -1.0
            r4 = iseven(length(U2) + count(d -> d < i, D1)) ? 1.0 : -1.0
            Unew = sort(U2)
            Dnew = sort([D1; i])
            iu2 = findfirst(isequal(Unew), ups)
            id2 = findfirst(isequal(Dnew), dns)
            if !(iu2 === nothing || id2 === nothing)
                s = r1 * r2 * r3 * r4
                M[iu2 + (id2 - 1) * nu, iu + (id - 1) * nu] += 0.5 * s
            end
        end
    end
    return M
end

"""
    ed_thermal(K, U, Nup, Ndn, beta) -> Dict

在固定 (N↑, N↓) 粒子数 sector 内构建 Hubbard 哈密顿
H = Σ_{ab,σ} K[a,b] c†_{aσ} c_{bσ} + U Σ_i n_i↑ n_i↓，
精确对角化后计算逆温 beta 下的 Gibbs 热平均。

返回字典包含：配分函数 "Z"、基态能 "E0"、密度 "density"（⟨n_i⟩）、
密度-密度关联 "nn"（⟨n_i n_j⟩）、纵向自旋关联 "szz"（⟨S^z_i S^z_j⟩）、
横向自旋关联 "sxy"（⟨S^x_i S^x_j + S^y_i S^y_j⟩）。
"""
function ed_thermal(K::Matrix{Float64}, U::Float64, Nup::Int, Ndn::Int, beta::Float64)
    N = size(K, 1)
    # ---- 基矢生成：两个自旋分量各自的占据子集表（组合数索引） ----
    ups = subsets(N, Nup)
    dns = subsets(N, Ndn)
    nu, nd = length(ups), length(dns)
    dim = nu * nd                      # Fock 空间维数 C(N,N↑)·C(N,N↓)
    H = zeros(dim, dim)
    # ---- Hubbard 相互作用项：在占据数基下对角 ----
    # 对角元 = U × （同一格点上双占据的数目），即 U·|Uset ∩ Dset|
    for (a, Uset) in enumerate(ups), (b, Dset) in enumerate(dns)
        H[a + (b - 1) * nu, a + (b - 1) * nu] += U * length(intersect(Uset, Dset))
    end
    # ---- 动能（跳跃）项：对 K 的每个非零矩阵元填入两个自旋分量的 c†_a c_b ----
    for b in 1:N, a in 1:N
        K[a, b] == 0.0 && continue
        hop_kinetic!(H, ups, dns, nu, nd, N, a, b, true, K[a, b])    # σ=↑
        hop_kinetic!(H, ups, dns, nu, nd, N, a, b, false, K[a, b])   # σ=↓
    end
    # ---- 全谱对角化 + Gibbs 权重 ----
    F = eigen(Hermitian(H))
    E = F.values                       # 本征能量（升序）
    V = F.vectors                      # 对应本征矢（列为基）
    w = exp.(-beta .* (E .- E[1]))     # Gibbs 权重，减去 E0 防上溢
    Z = sum(w)                         # 配分函数（未归一权重和）
    # ---- 热平均关联函数的对角算符预备 ----
    # nn、szz 以及 sxy 的"在位"（i=j）部分在占据数基下都是对角的，
    # 预先把它们在每个基矢上的取值存成向量，后面热平均只需对 |ψ|² 加权求和。
    nn = zeros(N, N); szz = zeros(N, N); sxy = zeros(N, N); density = zeros(N)
    nvec = [zeros(dim) for _ in 1:N]      # nvec[i][x]   = ⟨x|n_i|x⟩ = n_i↑ + n_i↓
    szvec = [zeros(dim) for _ in 1:N]     # szvec[i][x]  = ⟨x|S^z_i|x⟩ = (n_i↑ - n_i↓)/2
    xyonsite = zeros(dim, N)              # sxy 在位项 = (n_i↑ + n_i↓ - 2 n_i↑ n_i↓)/2
                                          # （即单占据时为 1/2，空/双占据时为 0）
    for (a, Uset) in enumerate(ups), (b, Dset) in enumerate(dns)
        x = a + (b - 1) * nu              # 基矢 |Uset_a, Dset_b⟩ 的线性索引
        for i in 1:N
            niu = i in Uset ? 1.0 : 0.0   # 格点 i 上自旋上占据数
            nid = i in Dset ? 1.0 : 0.0   # 格点 i 上自旋下占据数
            nvec[i][x] = niu + nid
            szvec[i][x] = 0.5 * (niu - nid)
            xyonsite[x, i] = 0.5 * (niu + nid - 2 * niu * nid)
        end
    end
    # sxy 的非对角（i≠j）部分 = (1/2)(S⁺_i S⁻_j + S⁻_i S⁺_j)，
    # 是自旋翻转（exchange）算符，非对角，预生成全部 i,j 的矩阵。
    Mexc = [exchange_matrix(ups, dns, nu, nd, N, i, j) for i in 1:N, j in 1:N]
    # ---- Gibbs 热平均：对本征态逐个累加 w_n · ⟨n|O|n⟩ ----
    for n in 1:dim
        wn = w[n]
        wn < 1e-300 && continue           # 权重可忽略的态直接跳过
        ψ = view(V, :, n)                 # 第 n 个本征矢（占据数基下的振幅）
        for i in 1:N
            # 对角算符的期望值 = Σ_x |ψ_x|² · O_x
            density[i] += wn * sum(ψ .^ 2 .* nvec[i])
            sxy[i, i] += wn * sum(ψ .^ 2 .* xyonsite[:, i])
            for j in 1:N
                nn[i, j] += wn * sum(ψ .^ 2 .* (nvec[i] .* nvec[j]))    # ⟨n_i n_j⟩
                szz[i, j] += wn * sum(ψ .^ 2 .* (szvec[i] .* szvec[j])) # ⟨S^z_i S^z_j⟩
                if i != j
                    # 非对角算符的期望值 = ⟨ψ|M_exc|ψ⟩
                    sxy[i, j] += wn * dot(ψ, Mexc[i, j] * ψ)
                end
            end
        end
    end
    # 除以 Z 归一化，得到真正的热平均值后返回
    return Dict("Z" => Z, "E0" => E[1], "density" => density ./ Z,
                "nn" => nn ./ Z, "szz" => szz ./ Z, "sxy" => sxy ./ Z)
end

end # module
