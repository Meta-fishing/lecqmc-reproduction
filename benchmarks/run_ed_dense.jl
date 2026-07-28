# Convergence check with EXACT dense expK (no checkerboard Trotter error)
include("../src/LECQMC.jl")
include("../ed/ed.jl")
using .LECQMC, .EDHubbard
using LinearAlgebra, Random, Printf, Statistics

function block_analysis(vals, nblocks=25)
    n = length(vals); bs = div(n, nblocks)
    bm = [mean(vals[(b-1)*bs+1:b*bs]) for b in 1:nblocks]
    return (mean(bm), std(bm)/sqrt(nblocks))
end

function main()
    L = 4; N = L*L
    t, U = 1.0, 2.0
    T, dt, nstab = 0.05, 0.05, 10
    beta = 1.0/T
    m = build_square_lattice(L, L, t, U)
    ed = ed_thermal(m.K, U, 1, 1, beta)
    @printf("ED: E0=%.8f\n", ed["E0"])

    rng = MersenneTwister(999)
    mc = MC(m, beta, dt, nstab, [[3],[7]]; use_dense=true, rng=rng)
    ntherm, nmeas = 2000, 100000
    accf = 0; attf = 0
    for s in 1:ntherm
        a, at = sweep!(mc, rng); accf += a; attf += at
    end
    obs = Observables(N)
    nn13 = Float64[]; szz13 = Float64[]
    for s in 1:nmeas
        a, at = sweep!(mc, rng); accf += a; attf += at
        measure!(obs, mc.sts)
        nf1 = zeros(N); nf2 = zeros(N)
        nf1[mc.sts[1].occ] .= 1.0; nf2[mc.sts[2].occ] .= 1.0
        push!(nn13, (nf1[1]+nf2[1])*(nf1[3]+nf2[3]))
        push!(szz13, 0.25*(nf1[1]-nf2[1])*(nf1[3]-nf2[3]))
    end
    res = finalize_obs(obs)
    m1, e1 = block_analysis(nn13); m2, e2 = block_analysis(szz13)
    @printf("sign=%.4f  fock acc=%.3f\n", res.sign, accf/attf)
    @printf("nn_13 : MC %.6f ± %.6f   ED %.6f\n", m1, e1, ed["nn"][1,3])
    @printf("szz_13: MC %.6f ± %.6f   ED %.6f\n", m2, e2, ed["szz"][1,3])
    @printf("sxy_13: MC %.6f          ED %.6f\n", res.sxy[1,3], ed["sxy"][1,3])
    @printf("sxy_11: MC %.6f          ED %.6f\n", res.sxy[1,1], ed["sxy"][1,1])
    @printf("nn_11 : MC %.6f          ED %.6f\n", res.nn[1,1], ed["nn"][1,1])
    @printf("szz_11: MC %.6f          ED %.6f\n", res.szz[1,1], ed["szz"][1,1])
end
main()
