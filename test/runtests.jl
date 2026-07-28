# Unit tests: cross-check stabilized QR updates against dense brute force.
include("../src/LECQMC.jl")
using .LECQMC
using LinearAlgebra
using Random
using Test

rng = MersenneTwister(42)

# small 2D square lattice
m = build_square_lattice(3, 3, 1.0, 2.0)
beta, dt, nstab = 4.0, 0.1, 4
mp = MCParams(beta, dt, nstab)
p = make_propagator(m, dt; use_dense=false)   # checkerboard path under test
N = m.N

fields = rand(rng, Int8[-1, 1], mp.Ltau, N)
occ_up = [3, 7]
occ_dn = [2]

# ---- dense reference propagator: build B(β,0) explicitly with the SAME slices
function dense_B(m, p, mp, fields, σ)
    B = Matrix{Float64}(I, m.N, m.N)
    tmp = similar(B)
    for l in 1:mp.Ltau
        apply_B!(B, B, p, @view(fields[l, :]), σ, tmp)
    end
    return B
end

Bup = dense_B(m, p, mp, fields, 1)
Bdn = dense_B(m, p, mp, fields, 2)

st_up = full_propagation!(m, p, mp, fields, 1, occ_up)
st_dn = full_propagation!(m, p, mp, fields, 2, occ_dn)

Pup = selection_matrix(N, occ_up)
Pdn = selection_matrix(N, occ_dn)

@testset "stabilized propagation" begin
    @test norm(st_up.Q * st_up.R - Bup * Pup) / norm(Bup * Pup) < 1e-10
    @test det(st_up.Q[occ_up, :]) * det(st_up.R) ≈ det(Pup' * Bup * Pup) rtol=1e-8
    @test det(st_dn.Q[occ_dn, :]) * det(st_dn.R) ≈ det(Pdn' * Bdn * Pdn) rtol=1e-8
end

@testset "particle addition" begin
    j = 5
    tr = trial_add(m, p, mp, fields, st_up, j)
    P2 = selection_matrix(N, [occ_up; j])
    r_direct = det(P2' * Bup * P2) / det(Pup' * Bup * Pup)
    @test tr.r ≈ r_direct rtol=1e-8
    # QR consistency of updated factors
    @test norm(tr.Q * tr.R - Bup * P2) / norm(Bup * P2) < 1e-10
    @test tr.A ≈ inv(tr.Q[tr.occ, :]) rtol=1e-8
    # per-chunk relation B_c Q_{c-1} = Q_c V_c checked implicitly via final QR
end

@testset "particle removal" begin
    k = 1
    tr = trial_remove(m, mp, st_up, k)
    P2 = selection_matrix(N, [occ_up[2]])
    r_direct = det(P2' * Bup * P2) / det(Pup' * Bup * Pup)
    @test tr.r ≈ r_direct rtol=1e-8
    @test norm(tr.Q * tr.R - Bup * P2) / norm(Bup * P2) < 1e-10
    @test tr.A ≈ inv(tr.Q[tr.occ, :]) rtol=1e-8
end

@testset "removal+addition swap" begin
    k, j = 2, 4
    tr_rem = trial_remove(m, mp, st_up, k)
    st_tmp = SpinState(1, tr_rem.occ, tr_rem.Qs, tr_rem.Vs, tr_rem.Q, tr_rem.R, tr_rem.A)
    tr_add = trial_add(m, p, mp, fields, st_tmp, j)
    r = tr_rem.r * tr_add.r
    newocc = [occ_up[1], j]
    P2 = selection_matrix(N, newocc)
    r_direct = det(P2' * Bup * P2) / det(Pup' * Bup * Pup)
    @test r ≈ r_direct rtol=1e-8
    @test norm(tr_add.Q * tr_add.R - Bup * P2) / norm(Bup * P2) < 1e-9
end

@testset "single-particle sector edge cases" begin
    # remove the only down particle (Ne: 1 -> 0), then add (0 -> 1)
    tr_rem = trial_remove(m, mp, st_dn, 1)
    P0 = zeros(N, 0)
    @test tr_rem.r ≈ 1.0 / det(Pdn' * Bdn * Pdn) rtol=1e-8
    st_tmp = SpinState(2, tr_rem.occ, tr_rem.Qs, tr_rem.Vs, tr_rem.Q, tr_rem.R, tr_rem.A)
    tr_add = trial_add(m, p, mp, fields, st_tmp, 9)
    P1 = selection_matrix(N, [9])
    @test tr_add.r ≈ det(P1' * Bdn * P1) / 1.0 rtol=1e-8
    @test tr_rem.r * tr_add.r ≈ det(P1' * Bdn * P1) / det(Pdn' * Bdn * Pdn) rtol=1e-8
end

@testset "field flip ratio" begin
    # ratio for flipping field at (l0, i0), computed via factor formula
    l0, i0 = 5, 4
    # build factors at slice l0: L = B(l0,0)P, Rr = P†B(β,l0), Af = (P†B(β,0)P)^{-1}
    Lm = copy(Pup); tmp = similar(Lm)
    for l in 1:l0
        apply_B!(Lm, Lm, p, @view(fields[l, :]), 1, tmp)
    end
    Rr = Matrix(Pup'); tmpr = similar(Rr)
    for l in mp.Ltau:-1:l0+1
        apply_B_right!(Rr, p, @view(fields[l, :]), 1, tmpr)
    end
    Af = inv(Pup' * Bup * Pup)
    s = fields[l0, i0]; sp = -s
    d = exp(p.lam * (sp - s)) - 1.0
    r_factor = 1.0 + d * dot(Lm[i0, :], Af * Rr[:, i0])
    # direct: rebuild Bup with flipped field
    fields2 = copy(fields); fields2[l0, i0] = sp
    Bup2 = dense_B(m, p, mp, fields2, 1)
    r_direct = det(Pup' * Bup2 * Pup) / det(Pup' * Bup * Pup)
    @test r_factor ≈ r_direct rtol=1e-8
end

println("All unit tests passed.")
