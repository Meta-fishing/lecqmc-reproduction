include("../src/LECQMC.jl")
using .LECQMC
using LinearAlgebra, Random, Printf

function main()
    K = [0.0 -1.0; -1.0 0.0]
    m = LECQMC.Model(2, K, 2.0, LECQMC.color_bonds([(1,2,-1.0)]))
    beta, dt, nstab = 10.0, 0.05, 5
    mp = MCParams(beta, dt, nstab)
    p = make_propagator(m, dt)
    rng = MersenneTwister(8)
    fields = rand(rng, Int8[-1,1], mp.Ltau, 2)
    function denseB(σ)
        B = Matrix{Float64}(I, 2, 2); tmp = similar(B)
        for l in 1:mp.Ltau
            apply_B!(B, B, p, @view(fields[l,:]), σ, tmp)
        end
        return B
    end
    maxrel = 0.0
    for trial in 1:20
        # random fock state
        ou = [rand(rng, 1:2)]; od = [rand(rng, 1:2)]
        stu = full_propagation!(m, p, mp, fields, 1, ou)
        # swap up particle to the other site
        j = ou[1] == 1 ? 2 : 1
        tr_rem = trial_remove(m, mp, stu, 1)
        st_tmp = SpinState(1, tr_rem.occ, tr_rem.Qs, tr_rem.Vs, tr_rem.Q, tr_rem.R, tr_rem.A)
        tr_add = trial_add(m, p, mp, fields, st_tmp, j)
        rf = tr_rem.r * tr_add.r
        B = denseB(1)
        P1 = selection_matrix(2, ou); P2 = selection_matrix(2, [j])
        rd = det(P2' * B * P2) / det(P1' * B * P1)
        rel = abs(rf-rd)/abs(rd)
        maxrel = max(maxrel, rel)
        rel > 1e-8 && @printf("trial %d: %s->%s formula=%.10f dense=%.10f rel=%.2e\n",
                              trial, ou, [j], rf, rd, rel)
        # perturb fields for next trial
        for _ in 1:20
            fields[rand(rng,1:mp.Ltau), rand(rng,1:2)] *= -1
        end
    end
    @printf("maxrel = %.2e\n", maxrel)
end
main()
