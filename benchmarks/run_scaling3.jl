include("../src/LECQMC.jl")
using .LECQMC
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
function main()
    U, T, dt, nstab = 2.0, 1.0, 0.1, 10
    beta = 1.0 / T
    OUT = "/mnt/agents/lecqmc/results"
    f = open("$OUT/scaling.txt", "a")
    for L in [80, 100]
        N = L * L
        tlec = time_lec(L, U, beta, dt, nstab, 50; ntherm=4, nmeas=4)
        @printf("%2d  %6d   %.5f    NaN\n", L, N, tlec); flush(stdout)
        println(f, "$L $N $tlec NaN"); flush(f)
    end
    close(f); println("done")
end
main()
