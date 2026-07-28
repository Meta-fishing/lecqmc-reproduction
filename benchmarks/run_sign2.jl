# Fig 2 reproduction (targeted, time-budgeted): <sign> vs 1/L, 2D square Hubbard U=2.
# Adds curve-1 L=32; curve-2 (Ne=100) L=16,20,24; curve-3 (T=0.02) L=16,20.
# Skips points already in sign.txt. Appends.
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
    for s in 1:nmeas
        sweep!(mc, rng)
        acc += weight_sign(mc.sts)
    end
    return acc / nmeas
end

function main()
    U, dt, nstab = 2.0, 0.1, 10
    OUT = "/mnt/agents/lecqmc/results"
    done = Set{Tuple{Float64,Int,Int}}()
    isfile("$OUT/sign.txt") && for ln in eachline("$OUT/sign.txt")
        p = split(ln)
        length(p) >= 3 && push!(done, (parse(Float64, p[1]), parse(Int, p[2]), parse(Int, p[3])))
    end
    # (T, Ne_total, L, ntherm, nmeas)
    plan = [
        (0.05, 50, 32, 60, 120),
        (0.05, 100, 16, 60, 200),
        (0.05, 100, 20, 60, 150),
        (0.05, 100, 24, 60, 120),
        (0.02, 50, 16, 60, 150),
        (0.02, 50, 20, 60, 120),
    ]
    f = open("$OUT/sign.txt", "a")
    for (T, Ne, L, ntherm, nmeas) in plan
        (T, Ne, L) in done && continue
        Neh = div(Ne, 2)
        t0 = time()
        sgn = measure_sign(L, U, T, dt, nstab, Neh; ntherm=ntherm, nmeas=nmeas, seed=42+L)
        @printf("T=%.2f Ne=%d L=%d (1/L=%.4f): <sign>=%.4f  (%.0fs total)\n",
                T, Ne, L, 1.0/L, sgn, time()-t0)
        flush(stdout)
        println(f, "$T $Ne $L $(1.0/L) $sgn"); flush(f)
    end
    close(f); println("done")
end
main()
