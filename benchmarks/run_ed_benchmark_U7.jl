# Fig S2 式 ED 对比基准（U=7 版）：2D Hubbard L=4, Ne=2 (Nup=Ndn=1),
# t=1, T=0.05 (beta=20), dt=0.05。与 run_ed_benchmark.jl（U=2）相同设置，
# 用于检验强耦合区的算法正确性。测量 nn / szz / sxy 三种关联（相对位点 1），
# 全部样本带符号加权并做 blocking 误差分析。
# 用法: julia run_ed_benchmark_U7.jl
include("../src/LECQMC.jl")
include("../ed/ed.jl")
using .LECQMC, .EDHubbard
using LinearAlgebra, Random, Printf, Statistics

function block_analysis(vals::Vector{Float64}, nblocks::Int=20)
    n = length(vals); bs = div(n, nblocks)
    bs == 0 && return (mean(vals), std(vals))
    bm = [mean(vals[(b-1)*bs+1:b*bs]) for b in 1:nblocks]
    return (mean(bm), std(bm) / sqrt(nblocks))
end

function main()
    L = 4; N = L * L
    t, U = 1.0, 7.0
    T, dt, nstab = 0.05, 0.05, 10
    beta = 1.0 / T
    nsweep_therm = 1000
    nsweep_meas = 40000
    OUT = joinpath(@__DIR__, "..", "results")

    m = build_square_lattice(L, L, t, U)

    println("Running ED (Nup=Ndn=1, dim=$(N*N), U=$U)..."); flush(stdout)
    t0 = time()
    ed = ed_thermal(m.K, U, 1, 1, beta)
    @printf("ED done in %.1fs, E0 = %.8f, Z = %.6e\n", time() - t0, ed["E0"], ed["Z"]); flush(stdout)

    rng = MersenneTwister(12345)
    mc = MC(m, beta, dt, nstab, [[3], [7]]; rng=rng)
    println("Running LEC-QMC: $nsweep_therm therm + $nsweep_meas meas sweeps..."); flush(stdout)
    t0 = time()
    for s in 1:nsweep_therm; sweep!(mc, rng); end
    obs = Observables(N)
    nn_series = zeros(N, nsweep_meas)
    szz_series = zeros(N, nsweep_meas)
    sxy_series = zeros(N, nsweep_meas)
    sgn_series = zeros(nsweep_meas)
    for s in 1:nsweep_meas
        sweep!(mc, rng)
        measure!(obs, mc.sts)
        ws = weight_sign(mc.sts)
        sgn_series[s] = ws
        nf1 = zeros(N); nf2 = zeros(N)
        nf1[mc.sts[1].occ] .= 1.0; nf2[mc.sts[2].occ] .= 1.0
        ntot = nf1 .+ nf2; sz = 0.5 .* (nf1 .- nf2)
        for j in 1:N
            nn_series[j, s] = ws * ntot[1] * ntot[j]
            szz_series[j, s] = ws * sz[1] * sz[j]
        end
        QA = [green_cols(st) for st in mc.sts]
        i0 = 1
        for j in 1:N
            if j == i0
                sxy_series[j, s] = ws * 0.5 * (nf1[j] + nf2[j] - 2*nf1[j]*nf2[j])
            else
                v = 0.0
                ou = mc.sts[1].occ; od = mc.sts[2].occ
                ai = findfirst(==(i0), ou); aj = findfirst(==(j), od)
                if ai !== nothing && aj !== nothing
                    v -= 0.5 * QA[1][j, ai] * QA[2][i0, aj]
                end
                ai2 = findfirst(==(i0), od); aj2 = findfirst(==(j), ou)
                if ai2 !== nothing && aj2 !== nothing
                    v -= 0.5 * QA[2][j, ai2] * QA[1][i0, aj2]
                end
                sxy_series[j, s] = ws * v
            end
        end
    end
    @printf("LEC-QMC done in %.1fs\n", time() - t0); flush(stdout)
    msgn = mean(sgn_series)
    @printf("<sign> = %.4f\n", msgn); flush(stdout)

    err_nn = zeros(N); err_szz = zeros(N); err_sxy = zeros(N)
    mn_nn = zeros(N); mn_szz = zeros(N); mn_sxy = zeros(N)
    for j in 1:N
        a, e = block_analysis(vec(nn_series[j, :])); mn_nn[j], err_nn[j] = a / msgn, e / msgn
        a, e = block_analysis(vec(szz_series[j, :])); mn_szz[j], err_szz[j] = a / msgn, e / msgn
        a, e = block_analysis(vec(sxy_series[j, :])); mn_sxy[j], err_sxy[j] = a / msgn, e / msgn
    end

    println("\n j   r      ED_nn      MC_nn(±err)      ED_szz     MC_szz(±err)     ED_sxy     MC_sxy(±err)"); flush(stdout)
    open("$OUT/ed_benchmark_L4_U7.txt", "w") do f
        println(f, "# LEC-QMC vs ED, 2D Hubbard L=4 Ne=2 T=0.05 U=$U dt=0.05, beta=$beta, sign=$msgn")
        println(f, "# j  r  ED_nn  MC_nn  err  ED_szz  MC_szz  err  ED_sxy  MC_sxy  err")
        for j in 1:N
            x = mod(j-1, L); y = div(j-1, L)
            r = sqrt(min(x, L-x)^2 + min(y, L-y)^2)
            @printf("%2d  %.2f  % .6f  % .6f(%.4f)  % .6f  % .6f(%.4f)  % .6f  % .6f(%.4f)\n",
                    j, r, ed["nn"][1, j], mn_nn[j], err_nn[j],
                    ed["szz"][1, j], mn_szz[j], err_szz[j],
                    ed["sxy"][1, j], mn_sxy[j], err_sxy[j])
            println(f, "$j $r $(ed["nn"][1,j]) $(mn_nn[j]) $(err_nn[j]) $(ed["szz"][1,j]) $(mn_szz[j]) $(err_szz[j]) $(ed["sxy"][1,j]) $(mn_sxy[j]) $(err_sxy[j])")
        end
    end
    flush(stdout)
end
main()
