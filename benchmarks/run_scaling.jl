# Fig 1 reproduction: CPU time per sweep vs N (log-log), LEC-QMC (Ne=100 fixed)
# vs standard fast-update DQMC.  Parameters: dt=0.1, U/t=2, T/t=1 (beta=1, Ltau=10).
include("../src/LECQMC.jl")
include("../src/StdDQMC.jl")
using .LECQMC, .StdDQMC
using LinearAlgebra, Random, Printf, Statistics

function time_lec(L, U, beta, dt, nstab, Ne_half; ntherm=20, nmeas=10, seed=1)
    m = build_square_lattice(L, L, 1.0, U)
    rng = MersenneTwister(seed)
    N = L * L
    # random initial occ, 50 per spin
    sites = shuffle(rng, 1:N)
    occ = [sort(sites[1:Ne_half]), sort(sites[Ne_half+1:2Ne_half])]
    mc = MC(m, beta, dt, nstab, occ; rng=rng)
    for s in 1:ntherm; sweep!(mc, rng); end
    t0 = time()
    for s in 1:nmeas; sweep!(mc, rng); end
    return (time() - t0) / nmeas
end

function time_std(L, U, beta, dt; ntherm=10, nmeas=5, seed=1)
    m = build_square_lattice(L, L, 1.0, U)
    p = make_propagator(m, dt)
    rng = MersenneTwister(seed)
    mc = StdMC(m, p, beta, dt; rng=rng)
    for s in 1:ntherm; std_sweep!(mc, m, p, rng); end
    t0 = time()
    for s in 1:nmeas; std_sweep!(mc, m, p, rng); end
    return (time() - t0) / nmeas
end

function selfcheck()
    println("self-check: StdDQMC wrap consistency...", ); flush(stdout)
    m = build_square_lattice(2, 2, 1.0, 2.0)
    p = make_propagator(m, 0.1)
    rng = MersenneTwister(3)
    mc = StdMC(m, p, 1.0, 0.1; rng=rng)
    for s in 1:5; std_sweep!(mc, m, p, rng); end
    G1 = std_green(m, p, mc.fields, 1)
    err = maximum(abs.(G1 - mc.Gs[1]))
    @printf("  wrap vs fresh: maxdiff = %.2e (should be < 1e-8)\n", err); flush(stdout)
end

function main()
    U, T, dt, nstab = 2.0, 1.0, 0.1, 10
    beta = 1.0 / T
    selfcheck()
    OUT = "/mnt/agents/lecqmc/results"
    f = open("$OUT/scaling.txt", "w")
    println(f, "# CPU time per sweep (s): LEC-QMC (Ne=100) vs standard DQMC; dt=$dt U=$U T=$T")
    println(f, "# L  N  t_LEC  t_DQMC")
    println(" L      N      t_LEC(s)     t_DQMC(s)"); flush(stdout)
    for L in [12, 16, 20, 24, 32, 40, 50, 60, 80, 100]
        N = L * L
        nm = N <= 400 ? 20 : (N <= 2500 ? 10 : 5)
        tlec = time_lec(L, U, beta, dt, nstab, 50; ntherm=10, nmeas=nm)
        tstd = NaN
        if N <= 2500
            nsd = N <= 400 ? 8 : (N <= 900 ? 5 : 3)
            tstd = time_std(L, U, beta, dt; ntherm=3, nmeas=nsd)
        end
        @printf("%2d  %6d   %.5f    %.5f\n", L, N, tlec, tstd); flush(stdout)
        println(f, "$L $N $tlec $tstd"); flush(f)
    end
    close(f)
    println("saved $OUT/scaling.txt")
end
main()
