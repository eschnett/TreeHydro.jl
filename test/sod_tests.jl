# Sod's shock tube: the first physical boundary, the first discontinuous
# solution, and the first comparison against somebody else's answer.
#
# The exact Riemann solver is tested next door, on Toro's table; everything
# here is about the *scheme* run against it, and about the boundary that
# makes the run possible. Four things can go wrong that nothing measured so
# far would catch:
#
#   * the boundary hook is not called, or is called with the wrong state —
#     the run then reflects, or reads conserved zeros and floors, and the
#     profile still looks like a shock tube;
#   * the hook is called but the run outlives its validity, which is a
#     mistake in the *experiment* rather than in the code, and is why the
#     arrival assertion exists and is itself tested;
#   * the flux kernel has a preferred direction, which no convergence rate
#     would name and which a bit-for-bit transposition does;
#   * the conserved integrals drift — which here they *must*, because mass,
#     momentum and energy cross a physical boundary. The claim is therefore
#     the boundary flux and not conservation, and it is a sharper claim than
#     the entropy wave's for being an equality with a closed form.
#
# The `D = 1` runs carry the convergence sweep, `D = 2` the direction
# claims, and `D = 3` is one small smoke run, as `CODE.md` splits the case.

const SOD_NS = (16, 32, 64, 128)

# The conservative family with prolongation of order 3, as everywhere else
# in the suite. Nothing here refines — the two two-level Sod forests are
# measured next door, in `interface_tests.jl` — so only the family is
# exercised, and it is named so that those runs differ in the mesh and in
# nothing else.
sod_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

sod_study(::Type{T}; Ns, limiter=:minmod) where {T} =
    [sod_errors(T, Val(1); N=N, ops=sod_ops(), roots=(4,), limiter=limiter)
     for N in Ns]

# The two `D = 1` sweeps, run once and read by three testsets: every run is
# wall-clock, and splitting the claims into a testset apiece would triple
# the cost of saying the same things.
const SOD_MINMOD = sod_study(Float64; Ns=SOD_NS, limiter=:minmod)
const SOD_MC = sod_study(Float64; Ns=SOD_NS, limiter=:mc)

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

@testset "The scheme converges in L1 against the exact Riemann solution" begin
    # The claim H1 ends on, and the reason the exact solver exists. The norm
    # is L1 and not L∞: on a solution with a shock and a contact, L∞ is the
    # error in the one cell nearest the discontinuity and converges at no
    # rate at all — which the recorded numbers below show, since it does not
    # even decrease monotonically. A limited second-order scheme on such a
    # solution gives L1 rates a little under one, and `CODE.md` predicted
    # 0.8 to 1.
    #
    # The bracket is [0.7, 1.05] and it is not moved: a rate above one would
    # mean the reference is being compared against itself somewhere, and a
    # rate well below 0.7 means the limiter or the boundary is eating the
    # solution. The errors must also decrease at every refinement — a rate
    # fitted through a non-monotone sequence can be anything.
    T = Float64
    rs = SOD_MINMOD
    hs = [r.h for r in rs]
    l1s = [r.l1 for r in rs]
    rate = convergence_rate(hs, l1s)
    @info "Sod, D = 1, :minmod, N = $SOD_NS: L1 rate $(round(rate, digits=3)), " *
          "L1 $(l1s)"
    @test 0.7 ≤ rate ≤ 1.05
    @test issorted(l1s; rev=true)
    @test all(r -> r.nblocks == 4, rs)
    @test all(r -> r.floor_hits == 0, rs)
    @test all(r -> isfinite(r.linf), rs)

    # The time-step observation, measured here because this is where the
    # numbers are. `λ` is the supremum of `|v| + c_s` over the exact
    # solution for all time; `λ_initial` is what a driver measuring the
    # state it has at `t = 0` would use. The ratio is what sizes the
    # headroom factor the chunked driver of step 7 needs — see "Time
    # integration and the time step" in `CODE.md`.
    @info "Sod: λ = $(rs[1].λ), λ from the initial data $(rs[1].λ_initial), " *
          "ratio $(rs[1].λ_ratio); λ on the final state " *
          "$([r.λ_final / r.λ for r in rs])"
    @test all(r -> isapprox(r.λ_ratio, 1.8522; atol=1e-4), rs)
    @test all(r -> isapprox(r.λ_initial, sqrt(7 / 5); atol=1e-12), rs)
    # The discrete state sits a little *above* the exact supremum at a
    # discontinuity, and the overshoot falls with `h`. Measured rather than
    # assumed, because it is the second number the driver's headroom has to
    # cover.
    @test issorted([r.λ_final for r in rs]; rev=true)
    @test rs[end].λ_final / rs[end].λ < 1.001
end

@testset "The less diffusive limiter converges too, and overshoots by more" begin
    # `:mc` is what the Kelvin–Helmholtz rolls will want and what a GRMHD
    # code defaults to, so what it does on a shock is a number worth having
    # beside `:minmod`'s rather than discovered at step 10. Two things come
    # out of it, and the second is the one that matters here:
    #
    #   * the L1 rate survives — a little higher, in fact, since the
    #     steeper reconstruction sharpens the contact that dominates the
    #     error;
    #   * the *discrete* overshoot above the exact supremum of the signal
    #     speed is larger and falls more slowly, 0.78% at `N = 16` against
    #     `:minmod`'s 0.49% and still 0.44% at `N = 128` against 0.0056%.
    #
    # That second number is what sizes `sod_errors`'s `λ_headroom`, and it
    # is half of what the chunked driver of step 7 will have to cover; the
    # other half is the 1.85 growth the previous testset measures. Recorded
    # in "Measured results" in `CODE.md`.
    mcs, mms = SOD_MC, SOD_MINMOD
    rate = convergence_rate([r.h for r in mcs], [r.l1 for r in mcs])
    @info "Sod, D = 1, :mc, N = $SOD_NS: L1 rate $(round(rate, digits=3)), " *
          "λ on the final state $([r.λ_final / r.λ for r in mcs])"
    @test 0.7 ≤ rate ≤ 1.05
    @test issorted([r.l1 for r in mcs]; rev=true)
    @test all(r -> r.floor_hits == 0, mcs)
    # Sharper than `:minmod` at every resolution, which is the whole reason
    # to pay for it.
    @test all(p -> p[1].l1 < p[2].l1, zip(mcs, mms))
    # And above the exact supremum at every resolution, by less than the
    # 2% `sod_errors` allows and by more than `:minmod`'s worst.
    @test all(r -> 1.004 < r.λ_final / r.λ < 1.02, mcs)
    @test all(p -> p[1].λ_final > p[2].λ_final, zip(mcs, mms))
end

@testset "The tube is the same along every axis, bit for bit" begin
    # The flux kernel has no preferred direction, and the claim is exact
    # rather than approximate. Two statements, both owed:
    #
    #   * The `D = 2` tube along `y` is the tube along `x` with the two
    #     space axes transposed and the two momentum components swapped.
    #     Every direction-dependent line in the scheme — the `Base.setindex`
    #     that moves the pressure into slot `1 + d`, the stencil offsets in
    #     the flux kernel, the accumulation over `d` in the divergence —
    #     is a place a stray asymmetry would hide, and none of them would
    #     show in a convergence rate.
    #   * The `D = 2` planar tube's profile equals the `D = 1` run's, and
    #     `S_y` is exactly zero. **Bit-identity is owed here** and not merely
    #     agreement to roundoff: the two transverse faces of every cell see
    #     *identical* states, so their fluxes are the same floating-point
    #     numbers and their difference is an exact zero, and adding an exact
    #     zero to the `x` difference changes nothing. A scheme that produced
    #     a transverse difference of `1e-17` would be one whose transverse
    #     stencil was not symmetric, which is worth failing over.
    #
    # `nsteps` is passed explicitly because `hydro_dt` carries a factor `D`:
    # the two runs must take the same step, or the identity is hidden behind
    # a different `dt`. It comes from the `D = 1` run, so the step is the one
    # that run's own CFL condition chose.
    T = Float64
    N = SOD_NS[1]
    r1 = SOD_MINMOD[1]                          # D = 1, N = 16, roots = (4,)
    rx = sod_errors(T, Val(2); N=N, ops=sod_ops(), direction=1, roots=(4, 1),
                    nsteps=r1.nsteps)
    ry = sod_errors(T, Val(2); N=N, ops=sod_ops(), direction=2, roots=(1, 4),
                    nsteps=r1.nsteps)
    @test rx.nsteps == ry.nsteps == r1.nsteps
    @test rx.floor_hits == ry.floor_hits == 0

    # The whole stored array, ghosts included: the outer ghosts are the
    # boundary hook's output and the transverse ones the exchange's, and
    # both must transpose with everything else.
    ax, ay = Array(rx.U.work), Array(ry.U.work)
    fx, fy = rx.U.forest, ry.U.forest
    ndiff = 0
    for bx in 1:nblocks(rx.U)
        ox = block_origin(T, fx, blockkey(rx.U, bx))
        by = findfirst(b -> block_origin(T, fy, blockkey(ry.U, b)) == (ox[2], ox[1]),
                       1:nblocks(ry.U))
        @test by !== nothing
        for i in axes(ax, 1), j in axes(ax, 2)
            # (ρ, S_x, S_y, E) at (i, j) against (ρ, S_y, S_x, E) at (j, i).
            for (vx, vy) in ((1, 1), (2, 3), (3, 2), (4, 4))
                ax[i, j, vx, bx] == ay[j, i, vy, by] || (ndiff += 1)
            end
        end
    end
    @test ndiff == 0

    # And the planar tube against the line. The `D = 2` L1 is *not* the
    # `D = 1` L1 — the norm divides by the number of stored entries, and
    # there are four variables rather than three — so the comparison is on
    # the solution and not on the norm. (The same reason the two `D = 2`
    # runs' `l1` differ in their last ulp: identical numbers summed in a
    # different order.)
    a1 = Array(r1.U.work)
    n1diff = 0
    nonzero_Sy = 0
    for b in 1:nblocks(r1.U)
        o = block_origin(T, r1.U.forest, blockkey(r1.U, b))
        bx = findfirst(c -> block_origin(T, fx, blockkey(rx.U, c))[1] == o[1],
                       1:nblocks(rx.U))
        @test bx !== nothing
        for i in axes(a1, 1), j in axes(ax, 2)
            for (v1, v2) in ((1, 1), (2, 2), (3, 4))
                a1[i, v1, b] == ax[i, j, v2, bx] || (n1diff += 1)
            end
            iszero(ax[i, j, 3, bx]) || (nonzero_Sy += 1)
        end
    end
    @test n1diff == 0
    @test nonzero_Sy == 0
    @info "Sod direction independence: $(r1.nsteps) steps, " *
          "D = 1 L1 $(r1.l1), D = 2 L1 $(rx.l1)"
end

@testset "The drift is the boundary flux, not a leak" begin
    # The conservation claim, and it is a *different* claim from the entropy
    # wave's, which is why it is not made with the same helper. With a
    # physical boundary the domain integral is not constant: whatever the
    # Dirichlet faces let through leaves the box. So asserting that the
    # totals hold would be asserting something false, and asserting nothing
    # would let a run that forgot the hook pass unnoticed.
    #
    # The claim is the equality the exact solution predicts. Until a wave
    # arrives, both sides of each boundary face are in that face's own
    # initial state — a state at rest — so the only nonzero component of the
    # Euler flux there is the pressure, in the momentum row:
    #
    #     ΔS = (p_L − p_R) · t_end · A,     Δρ = ΔE = 0
    #
    # with `A` the tube's cross-section. That is an equality with a closed
    # form, which is a far sharper statement than "the totals moved", and it
    # fails for a forgotten hook (the ghosts would be conserved zeros, hence
    # the atmosphere, hence a nonzero `floor_hits` and a mass flux), for a
    # reflecting boundary, and for a run long enough for a wave to arrive.
    #
    # It is not exact to roundoff at the coarsest resolution and it is not
    # meant to be: the numerical rarefaction's foot has diffused a little
    # way toward the left boundary by `t = 1/5`, which lets `1e-11` of mass
    # across. The tolerance is `1e-8` of the scale, two orders above the
    # worst measured value (3.7e-11 at `N = 16`, roundoff from `N = 32` on).
    #
    # The conservation claim proper — all `D + 2` integrals to roundoff,
    # with the fixup and not without it — is made for Sod on a *refined*
    # mesh in `interface_tests.jl`, as the difference between two runs
    # sharing this same boundary flux.
    T = Float64
    t_end = 1 // 5
    for (D, roots, area) in ((1, (4,), 1.0), (2, (4, 1), 0.25))
        w = SodTube(T, Val(D))
        r = sod_errors(T, Val(D); N=16, ops=sod_ops(), roots=roots, t_end=t_end)
        expected = (w.p_L - w.p_R) * T(t_end) * area
        @test r.floor_hits == 0
        @test r.drift[1 + 1] ≈ expected rtol = 1e-8       # the tube's momentum
        @test r.drift[1] ≤ 1e-8 * r.scales[1]             # mass does not cross
        @test r.drift[D + 2] ≤ 1e-8 * r.scales[D + 2]     # nor does energy
        # The totals did move, which is the negative half of the claim: a
        # run whose boundary had been forgotten would have no pressure
        # difference across the box to integrate.
        @test r.drift[2] > 1e-2
        # In `D = 2` the transverse momentum is zero for all time and
        # exactly so — its flux difference is an exact zero on both faces of
        # every cell, as the direction testset explains.
        D == 2 && @test r.drift[3] == 0
        @info "Sod, D = $D: momentum drift $(r.drift[1 + 1]) against the " *
              "boundary flux $expected; mass $(r.drift[1]), energy " *
              "$(r.drift[D + 2]) of scales $(r.scales[1]), $(r.scales[D + 2])"
    end
end

@testset "The shock tube runs in three dimensions" begin
    # A smoke test and nothing more, as the entropy wave's 3D run is: the
    # planar tube in 3D exercises nothing the 2D one does not except the
    # dimension where a hard-wired 2 or a transverse index that forgot a
    # direction finally shows. One short run at the smallest legal block,
    # asserting that the answer is finite, close to the reference, and that
    # no floor fired.
    T = Float64
    r = sod_errors(T, Val(3); N=4, ops=sod_ops(), roots=(4, 1, 1), t_end=1 // 20)
    @info "Sod, D = 3, N = 4: L1 $(r.l1), L∞ $(r.linf), $(r.nsteps) steps"
    @test isfinite(r.l1) && isfinite(r.linf)
    @test length(r.drift) == 5
    @test r.nblocks == 4
    @test r.l1 < 1 // 10
    @test r.floor_hits == 0
    # The two transverse momenta are zero for all time, exactly.
    @test r.drift[3] == 0
    @test r.drift[4] == 0
end
