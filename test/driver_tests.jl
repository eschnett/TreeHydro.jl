# The driver: what it refuses, what it reports, and the arithmetic of its
# CFL recheck.
#
# The tracked shock tube itself — the initial-data cycle, the tracking
# measure against the two uniform references, conservation through nine
# regrids, the buffer-width and prolongation-order tables — is fifteen
# evolutions and lives in `test/long/driver_tests.jl`. The short tier pins a
# reduced tracked run against its stored reference in `regression_tests.jl`
# instead, and keeps here the claims that need no evolution worth the name:
# the three refusals, the recheck's arithmetic, and the observer.
#
# The helpers below are shared with the long file, which the short tier is
# always included before (see `runtests.jl`), so the two halves run the same
# configuration and not two that merely look alike.

# The conservative family at the interface order the scheme wants, as
# `interface_tests.jl` and `refinement_tests.jl` run it. `p` is the
# prolongation order, which the `p = 1` table varies and nothing else does.
driver_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

# Step 6's calibrated thresholds, quoted rather than reinvented.
const DRIVER_REFINE_TOL = 2 // 25            # 0.08
const DRIVER_COARSEN_TOL = 1 // 50           # 0.02

# The two tracked configurations, stated once because a dozen runs below
# share them and a testset that quietly used its own would be comparing two
# different problems.
#
# `chunk` is the thing to understand here. The derived buffer covers
# `speed_headroom · λ · chunk` at the *cap's* spacing, and TreeAMR's
# recruitment reaches exactly one ring of neighbours, so the travel must
# stay under one finest-level block width — `(L/roots)/2^cap`, which is
# 1/32 in `D = 1` and 1/16 in `D = 2` here. With Sod's post-shock λ = 2.19
# and a headroom of 2 that caps the cadence at `chunk < 0.0071` and
# `chunk < 0.014`; `1/200` clears both, and `1/50` throws out of
# `refinement_buffer` naming the constraint. That is the buffer's own guard
# working, not a tuning knob.
const SOD1D = (roots=(8,), N=8, cap=2, chunk=1 // 200, t_end=1 // 5)
sod_case(::Val{D}, roots; kwargs...) where {D} =
    HydroCase(SodTube(Float64, Val(D)); roots=roots, kwargs...)

"""The tracked tube: the adaptive run every claim below is about."""
function tracked_sod(::Val{D}, cfg; p=3, fixup=true, buffer=nothing,
                     kwargs...) where {D}
    return evolve!(sod_case(Val(D), cfg.roots), Val(D); N=cfg.N, ops=driver_ops(p),
                   t_end=cfg.t_end, chunk=cfg.chunk, limiter=:minmod,
                   refine_tol=DRIVER_REFINE_TOL, coarsen_tol=DRIVER_COARSEN_TOL,
                   maxlevel_cap=cfg.cap, fixup=fixup, buffer=buffer, kwargs...)
end

@testset "The CFL recheck is arithmetic on five numbers" begin
    # Half of the step-4 amendment, and the half that needs no run: the
    # recheck is a pure function of the step taken, the finest spacing, the
    # dimension, the CFL number and the speed that turned up, and either the
    # step is inside the condition or it is not. The other half — that a
    # Riemann problem's fastest signal is not in its initial data, so a
    # driver believing a `λ` measured once per chunk runs the first chunk of
    # every shock case at nearly twice the CFL number it asked for — is the
    # measurement in `test/long/driver_tests.jl`.

    # Exactly at the bound, which is the case a naive `<` would fail on: the
    # step actually taken is `(stop − t)/steps` with an integer `steps` and
    # is therefore at most the step that was asked for.
    @test check_cfl(1 // 1000, 1 // 100, 1, 2 // 5, 4) == 2 // 5
    @test check_cfl(0.001, 0.01, 2, 0.4, 2.0) ≈ 0.4
    @test check_cfl(0.0005, 0.01, 1, 0.4, 4.0) ≈ 0.2
    # Past it, and the message names both speeds: the one the step was sized
    # for and the one that turned up.
    @test_throws "fastest signal of 4.0" check_cfl(0.001, 0.01, 1, 0.4, 5.0)
    @test_throws "λ_end = 5.0" check_cfl(0.001, 0.01, 1, 0.4, 5.0)
    @test_throws "in chunk 7" check_cfl(0.001, 0.01, 1, 0.4, 5.0; chunk=7)
    @test_throws "speed_headroom = 2" check_cfl(0.001, 0.01, 1, 0.4, 5.0; λ=1.0,
                                                headroom=2)
end

@testset "The driver refuses a cadence too slow for its cap" begin
    # The buffer's own guard, and it fires before a single step is taken.
    # The derived margin covers `speed_headroom · λ · chunk` at the cap's
    # spacing and TreeAMR's recruitment reaches exactly one ring of
    # neighbours, so a feature may not cross a whole finest-level block
    # between regrids. A cadence that would let it is refused by name rather
    # than truncated silently, which is the difference between a mesh that
    # cannot track and a mesh that quietly does not.
    @test_throws "exceeds the block width" tracked_sod(Val(1),
                                                       merge(SOD1D,
                                                             (chunk=1 // 50,)))
    # And a type the case was not built at, which is a mistake worth naming
    # rather than a `MethodError` from a signature nobody reads.
    @test_throws "carries its own working type" evolve!(
        Float32, sod_case(Val(1), SOD1D.roots), Val(1))
end

@testset "The driver refuses step 8's reset and reports every chunk to the observer" begin
    # The reset keyword exists now so that the signature does not change
    # when the atmosphere reset arrives; a run that silently ignored it
    # would be the worst of the three options.
    short = (roots=SOD1D.roots, N=SOD1D.N, cap=SOD1D.cap, chunk=1 // 200,
             t_end=1 // 50)
    @test_throws "arrives in step 8" tracked_sod(Val(1), short; reset=:stage)
    @test_throws "arrives in step 8" tracked_sod(Val(1), short; reset=:step)

    # The observer is what keeps `bin/` free of any time stepping of its
    # own, so it has to see the state *scattered* and `P` current, once per
    # chunk and once at t = 0, and always before the regrid that would
    # invalidate the field set it is handed.
    seen = Tuple{Float64,Int,Int,Float64}[]
    function watch(p, t, u)
        # `U` holds the state and `P` its primitives: reading the density
        # back out of the problem is the cheapest thing that fails if the
        # observer runs before the scatter.
        ρmax = maximum(block_mapreduce(identity, max, 0.0, p.P; vars=1))
        push!(seen, (t, length(u), nblocks(p.U), ρmax))
    end
    r = tracked_sod(Val(1), short; observer=watch)
    @test length(seen) == r.nchunks + 1
    @test [s[1] for s in seen] ≈ [0.0; [c * 0.005 for c in 1:(r.nchunks)]]
    @test all(s -> s[2] == s[3] * short.N * 3, seen)     # D + 2 = 3 variables
    @test all(s -> s[4] ≈ 1.0, seen)                     # ρ_L, never exceeded
end
