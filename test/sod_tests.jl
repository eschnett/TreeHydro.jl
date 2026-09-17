# Sod's shock tube: the boundary, before the physics.
#
# The convergence sweep against the exact Riemann solution, the bit-for-bit
# direction independence, the closed-form boundary flux and the three-
# dimensional smoke run are in `test/long/sod_tests.jl` — they are `N` up to
# 128 and they belong to the long tier. What stays here are the two claims
# that cost almost nothing and that the rest of the case rests on:
#
#   * the Dirichlet hook is called, and with the right state. A run whose
#     hook was forgotten reflects, or reads conserved zeros and floors, and
#     still produces a profile that looks like a shock tube; the claim is
#     made *after* the run, because at `t = 0` every ghost holds the initial
#     state whether the hook ran or not;
#   * the arrival assertion refuses a run the boundary would reflect, which
#     is a mistake in the *experiment* rather than in the code and is why
#     the assertion is itself tested.
#
# `sod_ops` lives here rather than next door because both halves use it: the
# short tier is always included first (see `runtests.jl`), so the long file
# shares it and the two differ in the mesh and in nothing else.

# The conservative family with prolongation of order 3, as everywhere else
# in the suite. Nothing here refines — the two two-level Sod forests are
# measured next door, in `interface_tests.jl` — so only the family is
# exercised, and it is named so that those runs differ in the mesh and in
# nothing else.
sod_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

# The conserved state on each side of the diaphragm, which is what the
# Dirichlet ghosts must hold for all time.
function sod_outer_states(w::SodTube{T,D}) where {T,D}
    zeros_ = ntuple(_ -> zero(T), Val(D))
    return (prim2con(w.eos, (w.ρ_L, zeros_..., w.p_L)),
            prim2con(w.eos, (w.ρ_R, zeros_..., w.p_R)))
end

# The block whose origin has the given coordinate along `axis`.
function block_at(U::FieldSet{T,D}, axis, x) where {T,D}
    b = findfirst(b -> block_origin(T, U.forest, blockkey(U, b))[axis] == x,
                  1:nblocks(U))
    b === nothing && error("no block with origin $x along axis $axis")
    return b
end

@testset "The Dirichlet ghosts hold the initial state and the transverse \
          ghosts do not" begin
    # The first downstream use of TreeAMR's physical-boundary path, and the
    # claim is made *after* the run rather than before it, which is what
    # makes it sharp. At `t = 0` every ghost cell would hold the initial
    # state whether the hook ran or not, since the interior does too; after
    # seventy steps the interior near the diaphragm has moved, so a
    # transverse ghost filled from the initial data instead of from its
    # periodic neighbour, or an outer ghost filled by copying the interior
    # instead of by the hook, is a different number.
    #
    # Both halves of "Dirichlet in `x`, periodic transversally" are
    # asserted, because getting the periodicity wrong is the more insidious
    # mistake: a Dirichlet transverse boundary set to the *initial* state
    # would be wrong from the first step and would still produce a profile
    # that looks like a shock tube. And the outer ghosts are checked over
    # the whole transverse extent, corners included — TreeAMR fills the
    # edge and corner regions of a Dirichlet face unconditionally, which is
    # what the Sedov case of step 9 will lean on.
    T = Float64
    N = 8
    w = SodTube(T, Val(2))
    U_L, U_R = sod_outer_states(w)
    r = sod_errors(T, Val(2); N=N, ops=sod_ops(), roots=(4, 1), t_end=1 // 20)
    U = r.U
    cons = Array(U.work)
    G = U.G
    stored = size(cons, 1)
    @test stored == N + 2 * G[1]
    @test r.floor_hits == 0

    left = block_at(U, 1, 0.0)
    right = block_at(U, 1, 1.0 - 1 / 4)
    for j in 1:stored, v in 1:4
        for i in 1:G[1]                                   # x < 0
            @test cons[i, j, v, left] == U_L[v]
        end
        for i in (G[1] + N + 1):stored                    # x > 1
            @test cons[i, j, v, right] == U_R[v]
        end
    end

    # Transversally the block is its own periodic neighbour (one root
    # across), so ghost row `j` holds owned row `j + N` below and `j − N`
    # above. Checked on the block the contact and the shock are in, where
    # the evolved state differs from both initial states — which is the
    # point of checking after the run.
    mid = block_at(U, 1, 1 / 2)
    for i in (G[1] + 1):(G[1] + N), v in 1:4
        for j in 1:G[2]
            @test cons[i, j, v, mid] == cons[i, j + N, v, mid]
        end
        for j in (G[2] + N + 1):stored
            @test cons[i, j, v, mid] == cons[i, j - N, v, mid]
        end
    end
    evolved = [cons[i, G[2] + 1, 1, mid] for i in (G[1] + 1):(G[1] + N)]
    @test any(ρ -> ρ != w.ρ_L && ρ != w.ρ_R, evolved)
end

@testset "The arrival assertion refuses a run the boundary would reflect" begin
    # A Dirichlet boundary set to the initial state is exact until a wave
    # reaches it and reflects afterwards, so the condition is checked before
    # the run and not diagnosed after it. The failure it guards against is
    # quiet: the run completes, the profile is plausible, and the error
    # against the exact solution is simply larger than it should be.
    #
    # `t_end = 1` lets the fastest signal travel 2.19 against a half-box of
    # 0.5; `t_end = 1/5` lets it travel 0.438, which is the margin Sod's
    # standard end time carries.
    @test_throws "reaches the Dirichlet boundary" sod_errors(
        Val(1); N=8, ops=sod_ops(), roots=(4,), t_end=1)
    # And the assertion on its own, with the numbers spelled out, so that a
    # change to `sod_errors` cannot make the check unreachable without this
    # failing too.
    w = SodTube(Float64, Val(1))
    forest = sod_forest(Val(1), 8; roots=(4,), L=w.L)
    @test assert_no_arrival(w, forest, 1 // 5, 2.1916) === nothing
    @test_throws "reaches the Dirichlet boundary" assert_no_arrival(
        w, forest, 1 // 4, 2.1916)
end
