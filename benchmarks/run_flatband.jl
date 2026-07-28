# Fig 4 reproduction: 1D BCL flat-band ferromagnet, t1=t4=-0.2, t2=t3=1.0, U=2.
# Half-filling OF THE FLAT BAND: flat band has L orbitals -> Ne_total = L
# (Nup=Ndn=L/2), as in the paper ("half-filling of the lowest flat band").
# Measure S_parallel(q=0) and Szz(q=0)_11 (A1 sublattice) vs T for L=4,8,12,16.
include("../src/LECQMC.jl")
include("../ed/ed.jl")
using .LECQMC, .EDHubbard
using LinearAlgebra, Random, Printf, Statistics

function blockerr(vals, nblocks=20)
    n = length(vals); bs = div(n, nblocks)
    bs == 0 && return std(vals)
    bm = [mean(vals[(b-1)*bs+1:b*bs]) for b in 1:nblocks]
    return std(bm) / sqrt(nblocks)
end

function run_point(L, T, dt, nstab, U; ntherm, nmeas, seed)
    m = build_flatband_chain(L, -0.2, 1.0, 1.0, -0.2, U)
    N = 2L
    beta = 1.0 / T
    rng = MersenneTwister(seed)
    sites = shuffle(rng, 1:N)
    Neh = div(L, 2)                      # flat-band half-filling: L/2 per spin
    occ = [sort(sites[1:Neh]), sort(sites[Neh+1:2Neh])]
    mc = MC(m, beta, dt, nstab, occ; rng=rng)
    for s in 1:ntherm; sweep!(mc, rng); end
    sp_num = zeros(nmeas); zz_num = zeros(nmeas); sgnv = zeros(nmeas)
    for s in 1:nmeas
        sweep!(mc, rng)
        obs = Observables(N)
        measure!(obs, mc.sts)
        r = finalize_obs(obs)
        ws = weight_sign(mc.sts)
        sp_num[s] = sum(r.sxy) / L * ws
        zz_num[s] = sum(r.szz[1:L, 1:L]) / L * ws
        sgnv[s] = ws
    end
    return sp_num, zz_num, sgnv
end

function main()
    U, dt, nstab = 2.0, 0.05, 10
    OUT = "/mnt/agents/lecqmc/results"
    f = open("$OUT/flatband.txt", "w")
    println(f, "# flat-band ferromagnet: t1=t4=-0.2 t2=t3=1.0 U=$U dt=$dt half-filling")
    println(f, "# L T Spar(q=0) err Szz11(q=0) err sign")
    Ts = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    # ED reference for L=4
    println("computing ED reference (L=4)..."); flush(stdout)
    m4 = build_flatband_chain(4, -0.2, 1.0, 1.0, -0.2, U)
    for T in Ts
        beta = 1.0 / T
        ed = ed_thermal(m4.K, U, 2, 2, beta)
        sp = sum(ed["sxy"]) / 4
        zz = sum(ed["szz"][1:4, 1:4]) / 4
        @printf("ED  L=4 T=%.2f: Spar=%.5f Szz11=%.5f\n", T, sp, zz); flush(stdout)
        println(f, "ED4 $T $sp $zz")
    end
    flush(f)
    for L in [4, 8, 12, 16]
        for T in Ts
            ntherm = 500
            nmeas = 4000
            spn, zzn, sg = run_point(L, T, dt, nstab, U; ntherm=ntherm, nmeas=nmeas, seed=100+L)
            msgn = mean(sg)
            sp = mean(spn) / msgn; sp_e = blockerr(spn ./ msgn)
            zz = mean(zzn) / msgn; zz_e = blockerr(zzn ./ msgn)
            @printf("QMC L=%2d T=%.2f: Spar=%.5f(±%.5f) Szz11=%.5f(±%.5f) sign=%.3f\n",
                    L, T, sp, sp_e, zz, zz_e, msgn); flush(stdout)
            println(f, "$L $T $sp $sp_e $zz $zz_e $msgn"); flush(f)
        end
    end
    close(f)
    println("saved $OUT/flatband.txt")
end
main()
