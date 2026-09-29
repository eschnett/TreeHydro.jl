# Precision (step 12, H6a): the drivers run in a caller-chosen float type,
# with no `Float64` left in the arithmetic, and a lower precision builds the
# same mesh and floors the same cells.
#
# The failure mode these guard is code that is generic in name only —
# computing in `Float64` and converting at the end, or reaching for a `Base`
# method only a hardware float has. Both are invisible in a `Float64` run,
# and the first is fatal on a device with no hardware fp64, which is what
# `Float32` exists here for.
#
# Each non-default type catches a different fault, as in TreeWave:
#
#   Float32    is the *leak detector*. A stray `Float64` operand widens the
#              result, so a returned `Float64` names the leak.
#   Float32x2  is the *off the beaten path* detector: a software type built
#              from two `Float32` limbs, which no hardware fast path can
#              serve. It cannot detect leaks — MultiFloats promotes `Float64`
#              *downward*, so a leak is absorbed silently — and tests instead
#              that nothing depends on a hardware float at all: no
#              `Int(::AbstractFloat)`, no `Float64(x)`, no `round(Int, x)`.
#              Step 12 found two of those, one here (`floor_hits` closed its
#              count with `round(Int, ·)`, now `roundint`) and one upstream
#              (IMEXRungeKutta's step count, fixed there at 1.3).
#
# The per-type claims `CODE.md`'s table under "Precision" makes, measured:
# the same mesh, step count and floor counts at every type, and the errors
# agreeing far inside the table's predicted 1%. What is *not* claimed at a
# lower precision is anything the host suite claims at `Float64` alone — a
# convergence rate, or a drift at `Float64`'s roundoff — and the
# Kelvin–Helmholtz half of the table is in `kelvinhelmholtz_tests.jl`,
# because it needs that file's `Float64` run.
#
# The entropy wave is absent from the MultiFloat runs on purpose: its
# initial data and its reference are `sin`, which MultiFloats does not
# implement. It runs at `Float32` below. Sod and Sedov carry `Float32x2`,
# their references being host `Float64` (the exact Riemann solver and the
# similarity quadrature), reached through `tofloat64` once per measurement.

using MultiFloats: Float32x2

const TYPE_OPS = Operators(family=Conservative, prolongation=3, restriction=2)
const TYPE_FLOATS = (Float64, Float32, Float32x2)
const TYPE_LOWER = (Float32, Float32x2)

# The agreement asked of a continuous result — an error norm, a shock radius,
# a peak — against the `Float64` run of the same case. At `Float32` it is the
# table's own prediction, "errors agreeing to ~1%", asserted as it was
# written; measured in step 12 it holds by 600× on Sod and the blast (worst
# 1.6e-5, the two-dimensional tube's L1) and by 8× on the entropy wave, whose
# L∞ error of 2.6e-4 is small enough that the *state's* roundoff shows in it
# at 1.2e-3 — an error norm is a difference of order-one states. At
# `Float32x2` a thousand times the worst measured, 9.9e-14. Mesh histories,
# step counts and floor counts get no tolerance at all.
type_rtol(::Type{Float32}) = 1e-2
type_rtol(::Type{Float32x2}) = 1e-10

type_close(x, ref, ::Type{T}) where {T} =
    isapprox(TreeHydro.tofloat64(x), TreeHydro.tofloat64(ref); rtol=type_rtol(T))

# `x / x` is one ulp away from 1 for a fair share of `Float32x2` values —
# its division is not correctly rounded — so a ratio that is exactly one at a
# hardware float is one to roundoff there; the exact claim is kept for the
# types that can carry it. TreeWave found the same.
type_is_one(x, ::Type{T}) where {T} = T <: Base.IEEEFloat ? x == 1 : x ≈ 1

type_tracked(::Type{T}, case, ::Val{D}; N=8, t_end, chunk, cap,
             observer=nothing) where {T,D} =
    evolve!(T, case, Val(D); N=N, ops=TYPE_OPS, t_end=t_end, chunk=chunk,
            limiter=:minmod, refine_tol=2 // 25, coarsen_tol=1 // 50,
            maxlevel_cap=cap, accounting=true, observer=observer)

# The four runs, at every type. The two-dimensional tube runs to `3//20` on
# purpose: at `Float32` its `t_end / chunk` is `30.000002f0`, which before
# step 12 gave it a one-step 31st chunk and a mesh history one entry longer
# than `Float64`'s (see `chunk_count`).
function type_runs(::Type{T}) where {T}
    # The one-dimensional tube also records the times its observer is handed,
    # which is where the end of the last chunk can be read.
    sod1_ts = T[]
    sod1 = type_tracked(T, HydroCase(SodTube(T, Val(1)); roots=(8,)), Val(1);
                        t_end=1 // 10, chunk=1 // 200, cap=2,
                        observer=(p, t, u) -> push!(sod1_ts, t))
    sod2 = type_tracked(T, HydroCase(SodTube(T, Val(2)); roots=(8, 1)), Val(2);
                        t_end=3 // 20, chunk=1 // 200, cap=1)
    blast = sedov_static(T, Val(2); N=8, ops=TYPE_OPS, roots=4, r₀=1 // 16,
                         t_end=1 // 10, refined=:center)
    tracked = type_tracked(T, HydroCase(SedovBlast(T, Val(2); r₀=1 // 16); roots=4),
                           Val(2); t_end=1 // 20, chunk=1 // 400, cap=2)
    return (; sod1, sod1_ts, sod2, blast, tracked)
end

const TYPE_RUNS = Dict(T => type_runs(T) for T in TYPE_FLOATS)

@testset "The chunk count is not given one more by rounding: T=$T" for T in TYPE_FLOATS
    # Guards `evolve!`'s cadence against the quotient `t_end / chunk` landing
    # an ulp above an integer — at `Float32` `3//20 / 1//200` is
    # `30.000002f0` — which `ceilint` turned into an extra chunk from
    # `30 · chunk` to `t_end`, one step long and with a regrid of its own.
    @test TreeHydro.chunk_count(T(3 // 20), T(1 // 200)) == 30
    @test TreeHydro.chunk_count(T(1 // 5), T(1 // 200)) == 40
    @test TreeHydro.chunk_count(T(3 // 2), T(1 // 200)) == 300
    @test TreeHydro.chunk_count(T(1 // 5), T(1 // 5)) == 1
    # And a chunk that does not divide `t_end` still gets its last, short
    # chunk: the tolerance is ulps, not a fraction of a chunk.
    @test TreeHydro.chunk_count(T(1 // 5), T(3 // 40)) == 3
    @test TreeHydro.chunk_count(T(1 // 10), T(1 // 3)) == 1
end

@testset "The last chunk ends at t_end exactly: T=$T" for T in TYPE_FLOATS
    # Guards the other half of the chunk-count fix: a run that stopped at
    # `nchunks · chunk` rather than at `t_end` would measure its error and
    # its totals at a time an ulp before the one it was asked for — last
    # digits only, which is why nothing else in the suite would notice.
    ts = TYPE_RUNS[T].sod1_ts
    @test length(ts) == 21                         # t = 0 and twenty chunks
    @test first(ts) === zero(T)
    @test last(ts) === T(1 // 10)
    @test issorted(ts) && allunique(ts)
    # The case is chosen so that the claim has teeth: at `Float32`
    # `20 · T(1//200)` is an ulp short of `T(1//10)`, so the old `min(c ·
    # chunk, t_end)` stopped there (measured after step 12).
    T === Float32 && @test 20 * T(1 // 200) != T(1 // 10)
end

@testset "Every number a run returns carries the run's type: T=$T" for T in TYPE_LOWER
    # At `Float32` a `Float64` here names a promotion inside the run; at
    # `Float32x2` a hardware-only method would have been a `MethodError` long
    # before, so reaching these lines is most of that claim.
    r = TYPE_RUNS[T]
    for x in (r.sod1, r.sod2, r.tracked)
        @test x.tracking isa T
        @test eltype(x.drift) === T && eltype(x.scales) === T
        @test eltype(x.injection) === T
        @test eltype(x.λ_history) === T && x.h isa T
        @test eltype(x.u) === T
    end
    for x in (r.sod1, r.sod2)
        @test x.l1 isa T && x.linf isa T
        @test isfinite(x.l1)
    end
    @test r.blast.r_s isa T && r.blast.peak isa T
    @test eltype(r.blast.injection) === T && eltype(r.blast.drift) === T
    @test eltype(r.blast.u) === T
end

@testset "A lower precision builds the Float64 mesh and floors the same cells: T=$T" for T in TYPE_LOWER
    # The claim that makes a reduced-precision run a rehearsal of the
    # `Float64` one rather than a different experiment: the refinement
    # criterion, the step count and the floors take the same decisions. A
    # criterion evaluated against a `Float64` literal, a chunk count that
    # rounds differently, or a floor comparison made at another precision
    # would each show here as a history of another length or a count off by
    # a cell.
    a, b = TYPE_RUNS[Float64], TYPE_RUNS[T]
    for k in (:sod1, :sod2, :tracked)
        x, y = getproperty(a, k), getproperty(b, k)
        @test y.nblocks_history == x.nblocks_history
        @test (y.nsteps, y.nchunks, y.nregrids, y.passes) ==
              (x.nsteps, x.nchunks, x.nregrids, x.passes)
        @test (y.floor_hits, y.reset_hits, y.ghost_hits) ==
              (x.floor_hits, x.reset_hits, x.ghost_hits)
        @test type_is_one(y.tracking, T) && x.tracking == 1
    end
    # The blast is where the floors fire — 4136 owned resets and 40 ghost
    # entries at `Float64`, on a static two-level mesh the shock crosses — so
    # it is the population a precision change would most plausibly move.
    x, y = a.blast, b.blast
    @test x.reset_hits > 0 && x.ghost_hits > 0
    @test (y.nsteps, y.reset_hits, y.ghost_hits, y.floor_hits) ==
          (x.nsteps, x.reset_hits, x.ghost_hits, x.floor_hits)
    @info "precision, $T against Float64: mesh histories of " *
          "$(length(b.sod1.nblocks_history)), $(length(b.sod2.nblocks_history)) " *
          "and $(length(b.tracked.nblocks_history)) chunks identical, steps " *
          "$((b.sod1.nsteps, b.sod2.nsteps, b.blast.nsteps, b.tracked.nsteps)); " *
          "the blast's $(y.reset_hits) resets and $(y.ghost_hits) ghost hits " *
          "against $(x.reset_hits) and $(x.ghost_hits)"
end

@testset "The errors are the discretization's, not the arithmetic's: T=$T" for T in TYPE_LOWER
    # Guards a lower precision that runs, builds the right mesh, and answers
    # with its roundoff rather than with the scheme: the L1 errors against
    # the host `Float64` references, the blast's shock radius and peak, all
    # against the `Float64` run of the same case. And the conserved
    # integrals hold to the run's *own* roundoff, which is `eps(T)`'s bound
    # and not `Float64`'s.
    a, b = TYPE_RUNS[Float64], TYPE_RUNS[T]
    rel(x, ref) = abs(TreeHydro.tofloat64(x) / TreeHydro.tofloat64(ref) - 1)
    for k in (:sod1, :sod2)
        x, y = getproperty(a, k), getproperty(b, k)
        @test type_close(y.l1, x.l1, T)
        @test type_close(y.linf, x.linf, T)
    end
    @test type_close(b.blast.r_s, a.blast.r_s, T)
    @test type_close(b.blast.peak, a.blast.peak, T)

    for x in (b.sod1, b.sod2, b.tracked, b.blast)
        # Sod's momentum and the tube's transverse momentum have no scale of
        # their own at rest, and Sod's boundary moves the momentum by its
        # flux; the claim is made on the totals that have a scale and no
        # boundary flux, mass and energy.
        D = length(x.drift) - 2
        for v in (1, D + 2)
            bound = 8 * eps(T) * x.scales[v] * x.nsteps
            if x === b.blast
                # The blast floors cells, so its drift is the injection.
                @test abs(x.drift[v] - abs(x.injection[v])) ≤ bound
            else
                @test x.drift[v] ≤ bound
            end
        end
    end
    @info "precision, $T against Float64: tracked Sod L1 relative difference " *
          "$(rel(b.sod1.l1, a.sod1.l1)) (D = 1) and $(rel(b.sod2.l1, a.sod2.l1)) " *
          "(D = 2), blast r_s $(rel(b.blast.r_s, a.blast.r_s)) and peak " *
          "$(rel(b.blast.peak, a.blast.peak))"
end

@testset "The entropy wave runs at Float32, and its error is the scheme's" begin
    # Not at `Float32x2`: its data and its reference are `sin`, which
    # MultiFloats does not implement (see the header). At `Float32` the
    # error is small enough — L1 1.35e-4, L∞ 2.6e-4 — that the state's
    # roundoff shows in it (3.1e-4 and 1.2e-3 relative, measured in step 12),
    # which is the worst agreement of the file and inside the table's 1% by
    # a factor of eight.
    a = entropywave_errors(Float64, Val(1); N=16, ops=TYPE_OPS, roots=4)
    b = entropywave_errors(Float32, Val(1); N=16, ops=TYPE_OPS, roots=4)
    @test b.l1 isa Float32 && b.linf isa Float32 && b.h isa Float32
    @test b.nsteps == a.nsteps && b.floor_hits == a.floor_hits == 0
    @test type_close(b.l1, a.l1, Float32)
    @test type_close(b.linf, a.linf, Float32)
    @test all(v -> b.drift[v] ≤ 8 * eps(Float32) * b.scales[v] * b.nsteps, 1:3)
    @info "precision, entropy wave at Float32: L1 $(b.l1) against $(a.l1), " *
          "relative difference $(abs(Float64(b.l1) / a.l1 - 1))"
end
