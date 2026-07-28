include("../src/LECQMC.jl")
include("../ed/ed.jl")
using .LECQMC, .EDHubbard
using LinearAlgebra, Random, Printf, Statistics

function block_analysis(vals, nblocks=25)
    n = length(vals); bs = div(n, nblocks)
    bm = [mean(vals[(b-1)*bs+1:b*bs]) for b in 1:nblocks]
    return (mean(bm), std(bm)/sqrt(nblocks))
end

function run_case(L, t, U, T, dt, nmeas, seed)
    N = L*L; beta = 1.0/T
    m = build_square_lattice(L, L, t, U)
    ed = ed_thermal(m.K, U, 1, 1, beta)
    rng = MersenneTwister(seed)
    mc = MC(m, beta, dt, 10, [[3],[7]]; rng=rng)
    for s in 1:2000; sweep!(mc, rng); end
    obs = Observables(N)
    nn13 = Float64[]; szz13 = Float64[]
    accf=0; attf=0
    for s in 1:nmeas
        a, at = sweep!(mc, rng); accf+=a; attf+=at
        measure!(obs, mc.sts)
        nf1 = zeros(N); nf2 = zeros(N)
        nf1[mc.sts[1].occ] .= 1.0; nf2[mc.sts[2].occ] .= 1.0
        push!(nn13, (nf1[1]+nf2[1])*(nf1[3]+nf2[3]))
        push!(szz13, 0.25*(nf1[1]-nf2[1])*(nf1[3]-nf2[3]))
    end
    res = finalize_obs(obs)
    m1, e1 = block_analysis(nn13); m2, e2 = block_analysis(szz13)
    @printf("T=%.3f dt=%.3f: sign=%.3f facc=%.2f\n", T, dt, res.sign, accf/attf)
    @printf("  nn_13  MC % .6f ± %.6f  ED % .6f  diff=%.2fσ\n", m1, e1, ed["nn"][1,3], (m1-ed["nn"][1,3])/e1)
    @printf("  szz_13 MC % .6f ± %.6f  ED % .6f  diff=%.2fσ\n", m2, e2, ed["szz"][1,3], (m2-ed["szz"][1,3])/e2)
    @printf("  sxy_13 MC % .6f          ED % .6f\n", res.sxy[1,3], ed["sxy"][1,3])
    @printf("  sxy_11 MC % .6f          ED % .6f\n", res.sxy[1,1], ed["sxy"][1,1])
    @printf("  nn_11  MC % .6f          ED % .6f\n", res.nn[1,1], ed["nn"][1,1])
    return nothing
end

run_case(4, 1.0, 2.0, 1.0, 0.05, 40000, 12345)
