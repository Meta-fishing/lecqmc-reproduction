# Fig 4 复现（U=0.5 版）：一维 BCL 平带铁磁模型。
# t1=t4=-0.2, t2=t3=1.0, U=0.5（该 U 值原文未标明，系像素级提取论文曲线+ED 扫描鉴别得出），
# 平带半填充：Ne=L（每自旋 L/2，Sz_tot=0）。dt≈0.1（L=4 与 ED 全温区对比验证）。
# 输出 ED(L=4) 参考曲线 + QMC 数据点到 results/flatband.txt。
# 用法: julia run_flatband_u05.jl
include("../src/LECQMC.jl")
include("../ed/ed.jl")
using .LECQMC, .EDHubbard
using LinearAlgebra, Random, Printf, Statistics

const U0 = 0.5

function blockerr(vals, nblocks=20)
    n = length(vals); bs = max(1, div(n, nblocks))
    bm = [mean(vals[(b-1)*bs+1:min(b*bs, n)]) for b in 1:div(n, bs)]
    return std(bm) / sqrt(length(bm))
end

function run_point(L, T, dt0, nstab; ntherm, nmeas, seed)
    m = build_flatband_chain(L, -0.2, 1.0, 1.0, -0.2, U0)
    N = 2L
    beta = 1.0 / T
    Ltau = nstab * max(1, round(Int, beta / dt0 / nstab))
    dt = beta / Ltau
    rng = MersenneTwister(seed)
    sites = shuffle(rng, 1:N)
    Neh = div(L, 2)                       # 平带半填充：每自旋 L/2
    occ = [sort(sites[1:Neh]), sort(sites[Neh+1:2Neh])]
    mc = MC(m, beta, dt, nstab, occ; rng=rng)
    for s in 1:ntherm; sweep!(mc, rng); end
    spn = zeros(nmeas); zzn = zeros(nmeas); sgn = zeros(nmeas)
    for s in 1:nmeas
        sweep!(mc, rng)
        obs = Observables(N)
        measure!(obs, mc.sts)
        r = finalize_obs(obs)
        ws = weight_sign(mc.sts)
        spn[s] = sum(r.sxy) / L * ws      # S∥(q=0) = (1/L)Σ_ij <SxSx+SySy>
        zzn[s] = sum(r.szz[1:L, 1:L]) / L * ws   # Szz(q=0)_11 (A1 子格)
        sgn[s] = ws
    end
    return spn, zzn, sgn
end

function main()
    OUT = joinpath(@__DIR__, "..", "results")
    Ts = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    f = open("$OUT/flatband.txt", "w")
    println(f, "# flat-band ferromagnet: t1=t4=-0.2 t2=t3=1.0 U=$U0 dt=0.1 flat-band half-filling (Ne=L)")
    println(f, "# L T Spar(q=0) err Szz11(q=0) err sign")
    m4 = build_flatband_chain(4, -0.2, 1.0, 1.0, -0.2, U0)
    for T in Ts
        ed = ed_thermal(m4.K, U0, 2, 2, 1.0 / T)
        sp = sum(ed["sxy"]) / 4
        zz = sum(ed["szz"][1:4, 1:4]) / 4
        @printf("ED  L=4 T=%.2f: Spar=%.5f Szz11=%.5f\n", T, sp, zz); flush(stdout)
        println(f, "ED4 $T $sp $zz")
    end
    flush(f)
    plan = Tuple{Int,Float64,Int,Int}[]
    for T in Ts
        push!(plan, (4, T, 2000, T == 0.01 ? 60000 : (T == 0.02 ? 50000 : (T <= 0.06 ? 25000 : 15000))))
    end
    for T in [0.05, 0.10, 0.20]
        push!(plan, (8, T, 1500, T == 0.05 ? 8000 : 6000))
    end
    push!(plan, (12, 0.10, 1200, 3000))
    push!(plan, (12, 0.20, 1200, 3000))
    push!(plan, (16, 0.20, 1000, 2000))
    for (L, T, ntherm, nmeas) in plan
        t0 = time()
        spn, zzn, sg = run_point(L, T, 0.1, 10; ntherm=ntherm, nmeas=nmeas, seed=100+L)
        msgn = mean(sg)
        sp = mean(spn) / msgn; sp_e = blockerr(spn) / msgn
        zz = mean(zzn) / msgn; zz_e = blockerr(zzn) / msgn
        @printf("QMC L=%2d T=%.2f: Spar=%.5f(+-%.5f) Szz11=%.5f(+-%.5f) sign=%.3f (%.0fs)\n",
                L, T, sp, sp_e, zz, zz_e, msgn, time() - t0)
        flush(stdout)
        println(f, "$L $T $sp $sp_e $zz $zz_e $msgn"); flush(f)
    end
    close(f); println("done")
end
main()
