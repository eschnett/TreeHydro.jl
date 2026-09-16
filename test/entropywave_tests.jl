# The entropy wave: the first measured numbers in the package.
#
# Two claims, made on the **uniform** mesh, which is the control: every
# face is a same-level face, so the interface fixup has nothing to do and
# the conserved integrals must be constant whether it runs or not. The
# two-level hierarchy `hydro_forest` also builds — where the fixup is the
# difference between conservation and a leak — is step 5's.
#
#   * **Order.** The wave is an exact solution of the nonlinear system, the
#     initial data and the reference are exact cell averages, and the
#     limiter is `:none`, so the measured slope of the error against `h` is
#     the scheme's own order and nothing else. A limited run is measured
#     beside it because that is what the shock cases will use, and what it
#     costs is a number worth having written down.
#   * **Conservation.** All `D + 2` integrals, not just the mass: this is
#     TreeAMR's M8b claim repeated for a system, and the momentum
#     components are the new case.
#
# Every run is wall-clock, so each study is run **once** and every claim
# made on the same results; splitting them into a testset apiece would
# double the suite's runtime to say the same things.

const ENTROPY_NS = Dict(1 => (8, 16, 32, 64), 2 => (8, 16, 32))

# The conservative family with prolongation of order 3 — one more than the
# scheme's order, which is TreeAMR's interface-order rule. Nothing here
# refines, so only the family is exercised; it is named so that step 5's
# refined runs differ in the mesh and in nothing else.
entropy_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

entropy_study(::Type{T}, ::Val{D}; Ns, limiter, fixup=true) where {T,D} =
    [entropywave_errors(T, Val(D); N=N, ops=entropy_ops(), roots=4,
                        refined=false, limiter=limiter, fixup=fixup) for N in Ns]

# The drift of every conserved integral, in ulp of that variable's own
# scale per step — the unit the conservation claim is made in, since a
# momentum whose total is zero by symmetry has nothing else to be measured
# against. `Σ hᴰ |U_v|` is the scale; see `conserved_scales`.
drift_in_ulp(r, ::Type{T}) where {T} =
    ntuple(v -> r.drift[v] / (eps(T) * r.scales[v] * r.nsteps), length(r.drift))

# Every conserved integral constant to roundoff: a few ulp of the
# variable's own scale, and not growing with the step count. 8 is the round
# number above what a sum of `nsteps` roundoff errors of one ulp each can
# reach; the measured values are two orders of magnitude below it (recorded
# in "Measured results" in `CODE.md`).
function test_conservation(rs, ::Type{T}) where {T}
    for r in rs
        for v in 1:length(r.drift)
            @test r.drift[v] ≤ 8 * eps(T) * r.scales[v] * r.nsteps
        end
        @test r.floor_hits == 0
    end
    return nothing
end

@testset "The scheme is second order on the entropy wave: D=$D" for D in (1, 2)
    # The claim the whole package is built to make, and the one that fails
    # for almost any mistake in the scheme: a reconstruction that is
    # effectively first order, a flux evaluated half a cell off, a
    # divergence scaled by the wrong spacing. It is made in two norms
    # because they fail differently — L1 survives a localized defect that
    # L∞ does not, which is exactly the difference the limiter below shows.
    #
    # The errors must also *decrease* at every refinement: a rate averaged
    # over a non-monotone sequence can be 2 and mean nothing.
    T = Float64
    Ns = ENTROPY_NS[D]
    rs = entropy_study(T, Val(D); Ns=Ns, limiter=:none)
    hs = [r.h for r in rs]
    l1s, linfs = [r.l1 for r in rs], [r.linf for r in rs]
    l1rate, linfrate = convergence_rate(hs, l1s), convergence_rate(hs, linfs)
    @info "entropy wave, D = $D, :none, N = $Ns: L1 rate " *
          "$(round(l1rate, digits=3)), L∞ rate $(round(linfrate, digits=3))"
    @test l1rate ≥ 1.9
    @test linfrate ≥ 1.9
    @test issorted(l1s; rev=true)
    @test issorted(linfs; rev=true)
    # The mesh is the uniform control: `roots^D` blocks, no refinement.
    @test all(r -> r.nblocks == 4^D, rs)
    test_conservation(rs, T)
    @info "entropy wave, D = $D, :none: worst drift " *
          "$(round(maximum(maximum(drift_in_ulp(r, T)) for r in rs), digits=3)) " *
          "ulp of the scale per step"
end

@testset "A limiter clips the extrema and keeps the L1 order: D=$D" for D in (1, 2)
    # What `:mc` costs on smooth data, measured rather than assumed, since
    # it is the limiter the shock cases will run with. A TVD limiter is
    # first order at a smooth extremum by construction — the slope it
    # allows goes to zero where the sine turns over — so the clipped region
    # dominates L∞ while its shrinking width leaves L1 second order. That
    # is what the two numbers below say, and it is why the convergence
    # study above is run with `:none`: a rate measured under a limiter
    # would be measuring the limiter.
    #
    # The L∞ rate is asserted at 1.2 and not at the 1.5 the plan predicted:
    # the measured value is ≈ 1.35 in both dimensions and drifts *down* as
    # `N` grows, toward the 1 the theory gives. Recorded in "Measured
    # results" in `CODE.md`.
    T = Float64
    Ns = ENTROPY_NS[D]
    rs = entropy_study(T, Val(D); Ns=Ns, limiter=:mc)
    hs = [r.h for r in rs]
    l1rate = convergence_rate(hs, [r.l1 for r in rs])
    linfrate = convergence_rate(hs, [r.linf for r in rs])
    @info "entropy wave, D = $D, :mc, N = $Ns: L1 rate " *
          "$(round(l1rate, digits=3)), L∞ rate $(round(linfrate, digits=3))"
    @test l1rate ≥ 1.8
    @test linfrate ≥ 1.2
    @test issorted([r.l1 for r in rs]; rev=true)
    @test issorted([r.linf for r in rs]; rev=true)
    test_conservation(rs, T)
end

@testset "Conservation does not depend on the fixup on a uniform mesh: D=$D" for
        D in (1, 2)
    # The control that gives the coarse-fine claim of step 5 its meaning.
    # `restrict_interfaces!` replaces a coarse flux by the average of the
    # fine fluxes across a coarse-fine face; on a single-level mesh there
    # is no such face, so switching it off must change *nothing*. If a
    # uniform run leaked without the fixup, the fixup would be covering for
    # a bug elsewhere and the refined measurement would mean nothing.
    #
    # Stronger than "both conserve": the two runs are asserted to produce
    # the same numbers bit for bit, since they differ by a call that has no
    # work to do.
    T = Float64
    N = 16
    with = entropy_study(T, Val(D); Ns=(N,), limiter=:mc, fixup=true)
    without = entropy_study(T, Val(D); Ns=(N,), limiter=:mc, fixup=false)
    test_conservation(without, T)
    @test with[1].l1 == without[1].l1
    @test with[1].linf == without[1].linf
    @test with[1].drift == without[1].drift
end

@testset "The entropy wave runs in three dimensions" begin
    # A smoke test and nothing more: 3D is the dimension where an `ntuple`
    # with a hard-wired 2 or a transverse index that forgot a direction
    # finally shows, and it is too expensive to converge here. One short
    # run at the smallest legal block, asserting that the answer is finite,
    # of the right shape, close to the reference, and conserved.
    T = Float64
    r = entropywave_errors(T, Val(3); N=4, ops=entropy_ops(), roots=2,
                           refined=false, limiter=:none, t_end=1 // 8)
    @info "entropy wave, D = 3, :none, N = 4: L1 $(r.l1), L∞ $(r.linf), " *
          "$(r.nsteps) steps"
    @test isfinite(r.l1) && isfinite(r.linf)
    @test length(r.drift) == 5
    @test r.nblocks == 2^3
    @test r.l1 < 1 // 10
    test_conservation([r], T)
end
