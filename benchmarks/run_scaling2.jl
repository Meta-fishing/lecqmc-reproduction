# scaling benchmark, remaining points: DQMC N=2500, LEC-only N=3600..10000
include("../src/LECQMC.jl")
include("../src/StdDQMC.jl")
using .LECQMC, .StdDQMC
using LinearAlgebra, Random, Printf

function time_lec(L, U, beta, dt, nstab, Ne_half; ntherm=10, nmeas=5, seed=1)
    m = build_square_lattice(L, L, 1.0, U)
    rng = MersenneTwister(seed)
    N = L * L
    sites = shuffle(rng, 1:N)
    occ = [sort(sites[1:Ne_half]), sort(sites[Ne_half+1:2Ne_half])]
    mc = MC(m, beta, dt, nstab, occ; rng=rng)
    for s in 1:ntherm; sweep!(mc, rng); end
    t0 = time()
    for s in 1:nmeas; sweep!(mc, rng); end
    return (time() - t0) / nmeas
end

function time_std(L, U, beta, dt; ntherm=2, nmeas=3, seed=1)
    m = build_square_lattice(L, L, 1.0, U)
    p = make_propagator(m, dt)
    rng = MersenneTwister(seed)
    mc = StdMC(m, p, beta, dt; rng=rng)
    for s in 1:ntherm; std_sweep!(mc, m, p, rng); end
    t0 = time()
    for s in 1:nmeas; std_sweep!(mc, m, p, rng); end
    return (time() - t0) / nmeas
end

function main()
    U, T, dt, nstab = 2.0, 1.0, 0.1, 10
    beta = 1.0 / T
    OUT = "/mnt/agents/lecqmc/results"
    f = open("$OUT/scaling.txt", "a")
    # DQMC N=2500
    tstd = time_std(50, U, beta, dt; ntherm=2, nmeas=3)
    @printf("50  2500   NaN    %.5f\n", tstd); flush(stdout)
    # LEC-only large points (and re-record 2500 LEC)
    for L in [50, 60, 80, 100]
        N = L * L
        nm = N <= 3600 ? 6 : 4
        tlec = time_lec(L, U, beta, dt, nstab, 50; ntherm=6, nmeas=nm)
        @printf("%2d  %6d   %.5f    NaN\n", L, N, tlec); flush(stdout)
        println(f, "$L $N $tlec NaN"); flush(f)
    end
    close(f)
    println("done")
end
main()
