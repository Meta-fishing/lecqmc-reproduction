# Decisive test: full enumeration of the Trotterized ensemble for a tiny system
# (2 sites, Ltau=2 -> 4 HS spins, 4 Fock states) vs MC sampling.
include("../src/LECQMC.jl")
using .LECQMC
using LinearAlgebra, Random, Printf

function main()
    K = [0.0 -1.0; -1.0 0.0]
    m = LECQMC.Model(2, K, 2.0, LECQMC.color_bonds([(1,2,-1.0)]))
    beta, dt, nstab = 1.0, 0.5, 2
    mp = MCParams(beta, dt, nstab)
    p = make_propagator(m, dt)
    # enumerate
    focks = [[ [1],[1] ], [ [1],[2] ], [ [2],[1] ], [ [2],[2] ]]
    function denseB(fields, σ)
        B = Matrix{Float64}(I, 2, 2); tmp = similar(B)
        for l in 1:mp.Ltau
            apply_B!(B, B, p, @view(fields[l,:]), σ, tmp)
        end
        return B
    end
    Weta = zeros(4)
    tot = 0.0
    for fmask in 0:15
        fields = Int8[ (fmask >> 0) & 1 == 1 ? 1 : -1   (fmask >> 1) & 1 == 1 ? 1 : -1;
                       (fmask >> 2) & 1 == 1 ? 1 : -1   (fmask >> 3) & 1 == 1 ? 1 : -1]
        Bup = denseB(fields, 1); Bdn = denseB(fields, 2)
        for (fi, (ou, od)) in enumerate(focks)
            Pu = selection_matrix(2, ou); Pd = selection_matrix(2, od)
            w = det(Pu' * Bup * Pu) * det(Pd' * Bdn * Pd)
            Weta[fi] += w
            tot += w
        end
    end
    println("exact P(eta) (signed, normalized):")
    for fi in 1:4
        @printf("  eta=%s: % .6f\n", string(focks[fi]), Weta[fi]/tot)
    end
    # exact nn_12, szz_12
    nn12 = (Weta[2] + Weta[4]) / tot
    @printf("exact nn_12 = %.6f\n", nn12)
    # MC
    rng = MersenneTwister(5)
    mc = MC(m, beta, dt, nstab, [[1],[2]]; rng=rng)
    for s in 1:2000; sweep!(mc, rng); end
    cnt = zeros(4)
    obs = Observables(2)
    for s in 1:100000
        sweep!(mc, rng)
        ou = mc.sts[1].occ[1]; od = mc.sts[2].occ[1]
        fi = (ou==1 ? (od==1 ? 1 : 2) : (od==1 ? 3 : 4))
        cnt[fi] += 1
        measure!(obs, mc.sts)
    end
    println("MC P(eta):")
    for fi in 1:4
        @printf("  eta=%s: % .6f\n", string(focks[fi]), cnt[fi]/sum(cnt))
    end
    res = finalize_obs(obs)
    @printf("MC nn_12 = %.6f (sign %.4f)\n", res.nn[1,2], res.sign)
end
main()
