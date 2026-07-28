# Quiet-system re-measure of all feasible scaling points -> scaling_clean.txt
include("../src/LECQMC.jl")
include("../src/StdDQMC.jl")
using .LECQMC, .StdDQMC
using LinearAlgebra, Random, Printf

function time_lec(L, U, beta, dt, nstab, Ne_half; ntherm=5, nmeas=5, seed=1)
    m = build_square_lattice(L, L, 1.0, U)
    rng = MersenneTwister(seed)
    N = L * L
    sites = shuffle(rng, 1:N)
    occ = [sort(sites[1:Ne_half]), sort(sites[Ne_half+1:2Ne_half])]
    mc = MC(m, beta, dt, nstab, occ; rng=rng)
    for s in 1:ntherm; sweep!(mc, rng); end
    GC.gc()
    t0 = time()
    for s in 1:nmeas; sweep!(mc, rng); end
    return (time() - t0) / nmeas
end

function time_dqmc(L, U, beta, dt; ntherm=3, nmeas=3, seed=1)
    m = build_square_lattice(L, L, 1.0, U)
    rng = MersenneTwister(seed)
    p = make_propagator(m, dt)
    mc = StdDQMC.StdMC(m, p, beta, dt; rng=rng)
    for s in 1:ntherm; StdDQMC.std_sweep!(mc, m, p, rng); end
    GC.gc()
    t0 = time()
    for s in 1:nmeas; StdDQMC.std_sweep!(mc, m, p, rng); end
    return (time() - t0) / nmeas
end

function main()
    U, T, dt, nstab = 2.0, 1.0, 0.1, 10
    beta = 1.0 / T
    OUT = "/mnt/agents/lecqmc/results"
    f = open("$OUT/scaling_clean.txt", "w")
    println(f, "# quiet re-measure; L  N  t_LEC  t_DQMC")
    for L in [12, 16, 20, 24, 32, 40, 50, 60, 80, 100]
        N = L * L
        GC.gc()
        tlec = time_lec(L, U, beta, dt, nstab, 50)
        tdq = NaN
        if N <= 1600
            GC.gc()
            tdq = time_dqmc(L, U, beta, dt)
        end
        @printf("%2d  %6d   %.5f   %.5f\n", L, N, tlec, tdq); flush(stdout)
        println(f, "$L $N $tlec $tdq"); flush(f)
    end
    close(f); println("done")
end
main()
