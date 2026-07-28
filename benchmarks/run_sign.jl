# Fig 2 reproduction (leaner): <sign> vs 1/L, 2D square Hubbard U=2.
# Skips (T,Ne,L) already present in results/sign.txt (append mode).
include("../src/LECQMC.jl")
using .LECQMC
using LinearAlgebra, Random, Printf, Statistics

function measure_sign(L, U, T, dt, nstab, Ne_half; ntherm, nmeas, seed)
    m = build_square_lattice(L, L, 1.0, U)
    rng = MersenneTwister(seed)
    N = L * L
    beta = 1.0 / T
    sites = shuffle(rng, 1:N)
    occ = [sort(sites[1:Ne_half]), sort(sites[Ne_half+1:2Ne_half])]
    mc = MC(m, beta, dt, nstab, occ; rng=rng)
    for s in 1:ntherm; sweep!(mc, rng); end
    acc = 0.0
    t0 = time()
    for s in 1:nmeas
        sweep!(mc, rng)
        acc += weight_sign(mc.sts)
    end
    return acc / nmeas, (time() - t0) / nmeas
end

function main()
    U, dt, nstab = 2.0, 0.1, 10
    OUT = "/mnt/agents/lecqmc/results"
    done = Set{Tuple{Float64,Int,Int}}()
    isfile("$OUT/sign.txt") && for ln in eachline("$OUT/sign.txt")
        p = split(ln)
        length(p) >= 3 && push!(done, (parse(Float64, p[1]), parse(Int, p[2]), parse(Int, p[3])))
    end
    f = open("$OUT/sign.txt", "a")
    for (T, Ne) in [(0.05, 50), (0.05, 100), (0.02, 50)]
        Neh = div(Ne, 2)
        for L in [16, 20, 24, 32, 48, 64]
            N = L * L
            (T, Ne, L) in done && continue
            ntherm = 80
            nmeas = N <= 600 ? 300 : (N <= 1100 ? 200 : (N <= 2500 ? 120 : 80))
            sgn, tps = measure_sign(L, U, T, dt, nstab, Neh; ntherm=ntherm, nmeas=nmeas, seed=42+L)
            @printf("T=%.2f Ne=%d L=%d (1/L=%.4f, n=%.3f): <sign>=%.4f  (%.2fs/sweep)\n",
                    T, Ne, L, 1.0/L, Ne/N, sgn, tps)
            flush(stdout)
            println(f, "$T $Ne $L $(1.0/L) $sgn"); flush(f)
        end
    end
    close(f)
end
main()
