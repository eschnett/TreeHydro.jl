# The driver: **one** chunked evolve-and-regrid loop, and the case data it
# runs on.
#
# Regridding changes both the length and the meaning of the state vector, so
# a run cannot be one `solve` call: each chunk is a fresh `solve` on a fresh
# `HydroProblem`, and between chunks the mesh is rebuilt. That is TreeAMR's
# prescription and TreeWave's and Burgers' practice. What is different here
# is that there is exactly **one** such loop for every case — TreeWave has
# three near-identical ones and records that a fourth would be the one to
# drift — so a **case is data**: a small struct holding the primitive
# initial-data closure, the equation of state, the floors, the boundary hook
# or `nothing`, the periodicity, the extents, the root brick, the speed
# headroom, and the exact reference or `nothing`. Nothing in this file knows
# what a shock tube is. See "Regridding: one driver, restart per chunk" in
# `CODE.md`.
#
# Three things in the loop are easy to get wrong and are each written out
# where they happen:
#
#   * **The boundary hook goes to three places** — `fill_ghosts!` (through
#     `HydroProblem`), `regrid!` (which fills ghosts before its transfer)
#     and `adapt_to_initial_data!`. Step 7 is the first to exercise the
#     last two, and forgetting the second is the bug that arrives one chunk
#     late.
#   * **`λ_max` measured at the start of a chunk is not a bound within it.**
#     A Riemann problem's fastest signal is not in its initial data — Sod's
#     post-shock gas is 1.8522 times faster than anything at `t = 0` — so
#     the step is sized from `speed_headroom · λ` and the end-of-chunk
#     recheck is a *detector* that fires after the damage, not a guard.
#   * **Chunks are counted, not accumulated.** `nchunks = ceilint(t_end /
#     chunk)` and `t = min(c · chunk, t_end)`: a `while t < t_end − 1e-12`
#     guard compares a time against an absolute slack, and at `Float32`
#     that slack is far below one ulp of `t`. See "Precision" in `CODE.md`.

"""
    HydroCase(T, Val(D); initial, eos, floors, boundary = nothing, periodic,
              extents, roots, speed_headroom = 1, reference = nothing)

Everything [`evolve!`](@ref) needs to know about a *problem*, and nothing
about how time passes: the primitive initial data as a pure `x -> P`
closure, the equation of state and the floors, the physical-boundary hook
or `nothing`, the periodicity per dimension, the physical extents, the root
brick, the speed headroom, and an optional exact reference
`(U, t) -> state vector`.

**A case is data** (decided in `CODE.md`, "Regridding: one driver, restart
per chunk"). The four cases of this package differ in their initial data,
their boundaries, their equation-of-state parameters and their references,
and in *nothing* about how time passes; so the loop is written once and the
differences are a struct. This is a deliberate step past TreeWave's "no
abstraction over initial data", taken because the abstraction here is over
a loop that already exists three times upstream, not over physics.

`speed_headroom` is **the factor by which the fastest signal may grow
within one chunk**, and it is a case parameter because it is physics. It
is the amendment recorded under "Time integration and the time step" in
`CODE.md`: the per-chunk `λ_max` is measured from the state at the start of
the chunk, a Riemann problem's fastest signal is not present in its initial
data, and the end-of-chunk recheck fires only *after* the chunk that
violated the condition has been integrated. Sod's measured growth is
**1.8522**, which is why [`HydroCase`](@ref)`(::SodTube)` uses `2` — the
margin covers the discrete overshoot of up to 0.8% on top. A smooth case
whose speed cannot grow (the entropy wave) takes `1`, and a headroom below
`1` is refused: it would ask the driver to size a step for a signal slower
than the one it just measured.

`reference` is called as `reference(U, t)` and returns a state vector on
`U`'s own layout, which is what makes it comparable with the run after any
number of regrids; `nothing` means the case has no closed form and
[`evolve!`](@ref) reports no errors.

`initial` and `boundary` both become **kernel arguments** — the initial
data through `fill_by_coordinates!` inside the adaptation cycle, the
boundary hook at every right-hand-side evaluation — so both must capture
`isbits` only. The case parameter structs ([`SodTube`](@ref),
[`EntropyWave`](@ref)) are `isbits` for exactly this reason, and a closure
over one of them is a legal kernel argument on any backend.

The two case constructors live with their cases — `HydroCase(::SodTube)` in
`sod.jl` and `HydroCase(::EntropyWave)` in `entropywave.jl` — so that this
file stays free of anything case-specific.
"""
struct HydroCase{T,D,INI,EOS,FLR,BC,REF}
    initial::INI                 # x -> P, the primitive tuple
    eos::EOS
    floors::FLR
    boundary::BC                 # a CellBoundary, or nothing if fully periodic
    periodic::NTuple{D,Bool}
    extents::NTuple{D,Tuple{T,T}}
    roots::NTuple{D,Int}
    speed_headroom::T
    reference::REF               # (U, t) -> state vector, or nothing
    valD::Val{D}
end

function HydroCase(::Type{T}, ::Val{D}; initial, eos::EquationOfState,
                   floors::Floors, boundary=nothing, periodic, extents, roots,
                   speed_headroom=1, reference=nothing) where {T,D}
    per = periodic isa Tuple ? ntuple(d -> Bool(periodic[d]), D) :
          ntuple(_ -> Bool(periodic), D)
    length(per) == D || throw(ArgumentError(
        "a case needs one periodicity flag per dimension, got $(length(per)) " *
        "for D = $D: periodicity is the case's physics — the shock tube is " *
        "Dirichlet along its own axis and periodic across it — and not one " *
        "property of the box."))
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    length(rs) == D || throw(ArgumentError(
        "a case needs one root count per dimension, got $(length(rs)) for " *
        "D = $D: a thin tube wants several roots along its axis and one " *
        "across it."))
    all(>(0), rs) || throw(ArgumentError("root counts must be positive, got $rs."))
    length(extents) == D || throw(ArgumentError(
        "a case needs one extent per dimension, got $(length(extents)) for " *
        "D = $D."))
    ext = ntuple(d -> (T(extents[d][1]), T(extents[d][2])), D)
    all(d -> ext[d][2] > ext[d][1], 1:D) || throw(ArgumentError(
        "each extent must be nonempty and increasing, got $ext."))
    T(speed_headroom) ≥ 1 || throw(ArgumentError(
        "speed_headroom must be at least 1, got $speed_headroom: it is the " *
        "factor by which the fastest signal may grow within one chunk, and a " *
        "value below 1 would size the time step for a signal slower than the " *
        "one just measured. Sod's measured growth is 1.8522 and its case uses " *
        "2; a case whose speed cannot grow uses 1."))
    (boundary !== nothing || all(per)) || throw(ArgumentError(
        "the case is not periodic in every dimension ($per) but has no " *
        "boundary hook: the ghost regions facing outside the domain would " *
        "hold whatever the allocation left there. Pass `boundary`, or make " *
        "every dimension periodic."))
    return HydroCase{T,D,typeof(initial),typeof(eos),typeof(floors),
                     typeof(boundary),typeof(reference)}(
        initial, eos, floors, boundary, per, ext, rs, T(speed_headroom),
        reference, Val(D))
end

"""
    conserved_initial(case::HydroCase)

The case's initial data in **conserved** variables and wrapped in
`AllVariables` — what `fill_by_coordinates!` and
[`adapt_to_initial_data!`](@ref) fill a state with.

`AllVariables` is the once-per-cell form of TreeAMR's coordinate callbacks,
and `prim2con` needs all `D + 2` primitives of a point at once to form the
energy density, so the per-variable form would run the whole conversion
`D + 2` times per cell and throw all but one number away. Captures `case`,
which is `isbits` whenever its closures are.
"""
conserved_initial(case::HydroCase) =
    AllVariables(x -> prim2con(case.eos, case.initial(x)))

# A primitive set over the same forest and layout as `U`, with the
# `con2prim` kernel already run over every **stored** cell.
#
# The adaptation cycle has no `HydroProblem` to borrow a primitive set
# from: it regrids `U` alone (`transfer = false`) and re-evaluates the
# initial data, so a set built before the cycle would have the wrong
# number of blocks by the second pass. Building a scratch one per pass is
# the honest way to say that — at most `maxpasses` allocations of a mesh
# that is still small, against an evolution of hundreds of steps.
function scratch_primitives(U::FieldSet{T,D}, eos, floors) where {T,D}
    P = FieldSet{T}(U.forest, D + 4; G=U.G, centering=U.centering,
                    backend=get_backend(U.work))
    map_blocks!(con2prim_kernel!, P, P.work, U.work, eos, floors, Val(D);
                stored=true)
    return P
end

"""
    check_cfl(dt, h_min, D, cfl, λ_end; chunk = nothing, λ = nothing,
              headroom = nothing)

The end-of-chunk CFL recheck: throw an `ArgumentError` unless the step
actually taken satisfies the condition against the fastest signal now
present, `dt · D · λ_end / h_min ≤ cfl`. Returns the CFL number the step
achieved.

**It throws on purpose.** `λ_max` is measured once per chunk, and the step
is sized from `speed_headroom · λ` because a Riemann problem's fastest
signal is not in its initial data (Sod's post-shock gas is 1.8522 times
faster than anything at `t = 0`). This check is what turns "the per-chunk
value is a bound in practice" into a fact rather than a hope — and it is a
*detector*, not a guard: it fires after the chunk that violated the
condition has already been integrated. The remedies are a larger
`speed_headroom` or a shorter `chunk`, never deleting the check; an
instability discovered three chunks later is the thing it exists to
prevent.

The bound is compared with a few ulp of slack, because the step actually
taken is `(stop − t) / steps` with an integer `steps` and is therefore
*at most* the step that was asked for: the equality case is real and must
not fail on rounding.

A pure function of its five numbers, so the tests can exercise it on
synthetic ones. The keywords are the driver's context and only enter the
message: the chunk index, the `λ` the step was sized from, and the
headroom that multiplied it.
"""
function check_cfl(dt, h_min, D::Integer, cfl, λ_end; chunk=nothing, λ=nothing,
                   headroom=nothing)
    ν = dt * D * λ_end / h_min
    λ_allowed = cfl * h_min / (D * dt)
    ν ≤ cfl * (1 + 8 * eps(float(one(ν)))) && return ν
    where = chunk === nothing ? "" : " in chunk $chunk"
    sized = λ === nothing ? "" :
            (headroom === nothing ?
             " The step was sized from λ = $λ." :
             " The step was sized from λ = $λ with speed_headroom = " *
             "$headroom, so it allowed a growth to $(headroom * λ).")
    throw(ArgumentError(
        "the CFL condition was violated$where: the step actually taken, " *
        "dt = $dt at h_min = $h_min, is the step for a fastest signal of " *
        "$λ_allowed, and the fastest signal at the end of the chunk is " *
        "λ_end = $λ_end — a CFL number of $ν against the requested $cfl." *
        sized *
        " λ_max is measured once per chunk and this recheck is a detector " *
        "rather than a guard, so the chunk has already been integrated at " *
        "the wrong step: raise speed_headroom, or shorten chunk so that the " *
        "signal has less room to grow between measurements. Do not delete " *
        "the check."))
end

"""
    tracked_share(P::FieldSet; refine_tol, maxlevel_cap, ε, ε_g, scales)

What fraction of the cells whose Löhner indicator exceeds `refine_tol` sit
on blocks that are already at `maxlevel_cap` — the measure of whether the
refined region is actually *following* the feature, as `refined_share` is
in TreeAMR's Burgers test.

`1` means every strongly firing cell was at the finest level the run
allows, which is what "the mesh tracks the shock" has to mean for a
discontinuity: a captured shock's `τ` does not fall with `h`, so the
criterion can never stop refining it and the cap is what binds (measured in
step 6). A value below `1` says a feature outran its refined region between
one regrid and the next, and the remedy is a wider buffer or a shorter
chunk.

One `firing_boxes` sweep on the field set's own backend, with the per-block
counts combined in **block order**, so the answer does not move with the
thread count. A block with no firing cell contributes nothing to either
side, and a mesh where nothing fires at all scores `1` — there is no
feature to fail to track.

**`P` must be current, ghosts included**, as [`hydro_flags`](@ref) needs it
to be: the indicator's stencil reaches one cell past each block face.
"""
function tracked_share(P::FieldSet{T,D}; refine_tol, maxlevel_cap,
                       ε=T(1 // 100), ε_g=T(1 // 1000),
                       scales=indicator_scales(P)) where {T,D}
    R = float(real(T))
    rtol = R(refine_tol)
    refs = (R(scales[1]), R(scales[2]))
    εR, ε_gR = R(ε), R(ε_g)
    valD = Val(D)
    boxes = firing_boxes(P) do work, idx, b, x
        cell_tau(work, idx, b, refs, εR, ε_gR, valD) > rtol
    end
    total = 0
    inside = 0
    for b in 1:nblocks(P)
        n = boxes[b][1]
        total += n
        level(blockkey(P, b)) ≥ maxlevel_cap && (inside += n)
    end
    return total == 0 ? one(R) : R(inside) / total
end

"""
    reduce_to_grid(U::FieldSet, M, extents = U.forest.extents)

Every conserved variable reduced onto a uniform grid of `M` cells per
dimension by exact volume averaging — **the common ground two meshes can be
compared on at all**, which is what an adaptive run measured against a
uniform reference needs.

`M` is one count or one count per dimension, because the boxes here are not
cubes: a thin shock tube with `roots = (8, 1)` reduces onto `(8N, N)`. Each
cell of each mesh is a whole subdivision of one target cell provided the
target is no finer than the coarsest block, which the callers arrange and
this checks.

The result is an array of size `(M..., D + 2)`: the whole state and not one
variable, since the claim a tracked run makes is about the solution and not
about its density alone. [`l1_difference`](@ref) is the norm it is read in.

An oracle in the spirit of TreeAMR's `reduce_to_grid`, which it follows: it
knows positions and spacings and nothing about how the data got there.
"""
function reduce_to_grid(U::FieldSet{T,D}, M, extents=U.forest.extents) where {T,D}
    Ms = M isa Tuple ? ntuple(d -> Int(M[d]), D) : ntuple(_ -> Int(M), D)
    all(>(0), Ms) || throw(ArgumentError(
        "reduce_to_grid needs a positive cell count per dimension, got $Ms."))
    lo = ntuple(d -> T(extents[d][1]), D)
    H = ntuple(d -> (T(extents[d][2]) - lo[d]) / Ms[d], D)
    forest = U.forest
    R = float(real(T))
    out = zeros(R, Ms..., U.nvars)
    work = Array(U.work)
    for b in 1:nblocks(U)
        h = spacing(T, forest, blockkey(U, b))
        all(d -> h ≤ H[d] * (1 + 4096 * eps(R)), 1:D) || throw(ArgumentError(
            "reduce_to_grid needs cells no coarser than the target grid: " *
            "block $b has h = $h and the target cell is $H. Every cell of " *
            "every mesh compared has to be a whole subdivision of one target " *
            "cell, or the reduction is not a volume average."))
        w = prod(ntuple(d -> R(h) / R(H[d]), D))
        for idx in CartesianIndices(ntuple(d -> (U.G[d] + 1):(U.G[d] + forest.N), D))
            x = coordinates(T, U, b, Tuple(idx))
            cell = ntuple(d -> clamp(floorint((x[d] - lo[d]) / H[d]) + 1, 1, Ms[d]), D)
            for v in 1:(U.nvars)
                out[cell..., v] += w * R(work[Tuple(idx)..., v, b])
            end
        end
    end
    return out
end

"""
    l1_difference(a, b)

Mean absolute difference of two [`reduce_to_grid`](@ref) reductions onto the
same grid — TreeAMR's, copied rather than depended on, as `precision.jl` is.
"""
l1_difference(a, b) = sum(abs, a .- b) / length(a)

"""
    evolve!([T], case::HydroCase, Val(D); N, ops, t_end, chunk, limiter,
            refine_tol, coarsen_tol, maxlevel_cap, G = 2, cfl = 2//5,
            roots = case.roots, buffer = nothing, riemann = :hlle,
            fixup = true, reset = :stage, accounting = false,
            ε = T(1//100), ε_g = T(1//1000),
            maxpasses = 8, backend = CPU(), observer = nothing)

**The** time-stepping loop: adapt the mesh to the initial data, then evolve
in chunks of `chunk`, regridding between them so the refined region follows
the solution. Returns a named tuple of everything the claims are made of —
the drift of each conserved integral and the scale it is measured against,
the floor counts, the step, chunk and block counts, the tracking measure,
the errors against `case.reference`, and the final state, state vector and
forest.

There is **one** of these for every case, which is the decision recorded in
"Regridding: one driver, restart per chunk" in `CODE.md`. What differs
between cases is data — see [`HydroCase`](@ref) — and nothing here knows
what a shock tube or a blast wave is.

## Why a chunk is a restart

Regridding changes both the *length* and the *meaning* of the state vector:
block slots are compacted, new blocks are prolongated from their parents,
and every schedule is stale afterwards. So each chunk is a fresh fixed-step
`solve` on a freshly built [`HydroProblem`](@ref), with the primitive set
and the flux sets handed back in — `regrid!` resized them in place through
`fs => nothing`, and reallocating them would throw that away. This is
TreeAMR's prescription and TreeWave's and Burgers' practice.

The chunk count is computed in exact arithmetic, `nchunks = ceilint(t_end /
chunk)` with the last chunk shortened, and *not* as a `while t < t_end −
tiny` guard: an absolute slack is meaningless at a type whose ulp is larger
than it (see "Precision" in `CODE.md`).

## The headroom and the recheck

One global step for the whole hierarchy, from the finest spacing and the
fastest signal, `dt = cfl · h_min / (D · speed_headroom · λ)` with `λ`
measured once per chunk. The `speed_headroom` factor is the case's, and it
is there because **`λ_max` measured at the start of a chunk is not a bound
within it**: a Riemann problem's fastest signal is not present in its
initial data, and Sod's post-shock gas is 1.8522 times faster than anything
at `t = 0` (measured in step 4). At the end of the chunk the speed is
measured again and [`check_cfl`](@ref) **throws** if the step actually taken
violated the condition — a detector rather than a guard, since the chunk
has already been integrated. `speed_headroom = 1` on discontinuous initial
data throws in the first chunk, which is the measurement that sizes the
factor.

## The boundary hook goes to three places

`fill_ghosts!` (through `HydroProblem`'s `boundary`), `regrid!` — which
fills ghosts before its transfer, because a prolongation stencil reads its
parent's ghost layers — and `adapt_to_initial_data!`. All three are here,
and forgetting the second is the bug that arrives one chunk late.

## The initial-data cycle

`adapt_to_initial_data!` fills the data, flags, regrids with `transfer =
false` and **re-evaluates** the initial data on the new mesh rather than
interpolating it, until the hierarchy stops changing; the run refuses to
continue if it has not converged within `maxpasses`. Its criterion is the
same Löhner indicator the evolution uses, on a scratch primitive set built
per pass — there is no `HydroProblem` yet, since the forest is still
changing under it.

## Keywords

`N` cells per block, `ops` the operator family, `t_end`, `chunk`, `limiter`
and the three refinement parameters have **no defaults**, because each is
something the caller must think about: the limiter changes what is being
measured, and the tolerances and the cap are the case's (the values step 6
calibrated are `refine_tol = 0.08`, `coarsen_tol = 0.02`).

`roots` defaults to the case's own and exists so that [`uniform_run`](@ref)
can build the same case on a finer root brick — the uniform reference an
adaptive run is judged against — through this same loop rather than through
a second one.

`buffer` is the travelling margin in cells. Left at `nothing` it is derived
per chunk by [`refinement_buffer`](@ref) from `speed_headroom · λ · chunk`,
the fastest signal the next chunk may carry times the regrid cadence; the
derivation is conservative by the headroom factor, and if it exceeds a
finest-level block width it throws naming the constraint, which means the
chunk is too long for the cap. With `maxlevel_cap = 0` nothing can refine,
so the derivation is skipped and the margin is zero.

`reset` is where [`reset_atmosphere!`](@ref) acts — `:stage` after every
SSPRK stage, which is the default and GRMHD practice, `:step` once per step,
or `:none`. It runs in **two** places whichever hook is chosen: inside the
solve, and here on the freshly gathered state after a [`regrid!`](@ref),
because the `p = 3` prolongation into a new fine block is unlimited and can
leave an owned cell unphysical, which would otherwise wait for the first
stage of the next chunk to be caught.

`accounting` turns on the **injection measurement**: the per-variable totals
of the stage vector before and after every reset, accumulated over the run
and returned as `injection`. It costs two full reductions of the state per
reset, which is why it is off by default — it is a keyword the tests turn on
and the demos do not. The reset's *hit count* is always taken, being one
reduction over one diagnostic slot. See "Floors and the atmosphere" in
`CODE.md`.

`observer(P_problem, t, u)` is called with the state scattered into `U` and
`P` current, once after the initial-data cycle at `t = 0` and once per
chunk **before the regrid that would invalidate `U`** — which is what keeps
the viewers of step 11 free of any time stepping of their own.

## What comes back

`drift` and `scales` are per variable and are the *maximum over the run*,
as Burgers takes them: a leak that reversed sign would otherwise hide.

**Three floor numbers and an injection, and each answers a different
question.** `floor_hits` is what it has always been: owned cells that the
`con2prim` pass found unphysical at a chunk boundary, accumulated —
so with the stage reset in place it should be rare, the reset having
already handled the stage that produced them. `reset_hits` is the owned
cells [`reset_atmosphere!`](@ref) actually changed, over every stage of
every step and every post-regrid call. `ghost_hits` is
[`ghost_floor_hits`](@ref) accumulated over the chunk boundaries, and it is
the number that decides the upstream prolongation question. `injection` is
the per-variable `Σ hᴰ ΔU` the resets injected — and it is `nothing`, not a
tuple of zeros, when `accounting = false`, so that a caller cannot read "not
measured" as "measured and zero".

`tracking` is the *minimum* over chunks of [`tracked_share`](@ref), and
`nblocks_history`, `buffer_history` (the width actually used at each
regrid, which is the derivation's own record), `λ_history` and
`λ_end_history` are per chunk. The final
`forest`, `U` and `u` are the mesh the answer was computed on: the loop does
not regrid after the last chunk, so the state that comes back and the mesh
statistics beside it describe the same thing.
"""
evolve!(case::HydroCase{T,D}, valD::Val{D}=case.valD; kwargs...) where {T,D} =
    evolve!(T, case, valD; kwargs...)

# A leading `T` is this package's convention on every driver, and the case
# already carries one; saying a different one is a mistake worth naming
# rather than a `MethodError` from a signature nobody reads.
evolve!(::Type{S}, case::HydroCase{T}, ::Val) where {S,T} = throw(ArgumentError(
    "evolve! was asked for $S but the case is stated in $T: a case carries " *
    "its own working type — its extents, its states and its speed headroom " *
    "are all in it — so the type is chosen when the case is built and not " *
    "when it is run. Build the case at $S instead."))

function evolve!(::Type{T}, case::HydroCase{T,D}, ::Val{D}; N, ops, t_end, chunk,
                 limiter=nothing, refine_tol, coarsen_tol, maxlevel_cap, G=2,
                 cfl=2 // 5, roots=case.roots, buffer=nothing, riemann=:hlle,
                 fixup=true, reset=:stage, accounting::Bool=false,
                 ε=T(1 // 100), ε_g=T(1 // 1000),
                 maxpasses=8, backend=CPU(), observer=nothing) where {T,D}
    # Refused here rather than at the first chunk, so that a typo does not
    # cost an initial-data cycle before it is named.
    check_reset(reset)
    t_end, chunk, cfl = T(t_end), T(chunk), T(cfl)
    t_end > 0 || throw(ArgumentError("t_end must be positive, got $t_end."))
    chunk > 0 || throw(ArgumentError(
        "chunk must be positive, got $chunk: it is the regrid cadence, and " *
        "the number of chunks is counted as ceilint(t_end / chunk)."))
    maxlevel_cap ≥ 0 || throw(ArgumentError(
        "maxlevel_cap must be non-negative, got $maxlevel_cap."))

    forest = Forest{T}(roots; N=N, periodic=case.periodic, extents=case.extents)
    U = FieldSet{T}(forest, D + 2; G=G, backend=backend)
    initial_U = conserved_initial(case)

    # The whole flag vector at once rather than a mark per block, which is
    # the form that runs on a device. The cycle has already filled `U`'s
    # ghosts — the boundary hook included — when it calls this, which is
    # what the indicator's three-point stencil needs.
    criterion(fs) = hydro_flags(scratch_primitives(fs, case.eos, case.floors);
                                refine_tol=refine_tol, coarsen_tol=coarsen_tol,
                                maxlevel_cap=maxlevel_cap, ε=ε, ε_g=ε_g)

    # The margin the initial adaptation travels with, from the initial
    # data's own fastest signal. It is not a bound on the run — that is the
    # whole point of the headroom — but it is the only speed that exists
    # before the first chunk, and the loop re-derives the margin from the
    # measured `λ` at every chunk afterwards.
    fill_by_coordinates!(initial_U, U)
    λ_initial = max_signal_speed(scratch_primitives(U, case.eos, case.floors))
    derive_buffer(λ) = buffer !== nothing ? Int(buffer) :
                       maxlevel_cap == 0 ? 0 :
                       refinement_buffer(forest, maxlevel_cap,
                                         case.speed_headroom * λ * chunk)

    # The cycle's own schedule is dropped: `HydroProblem` builds one from
    # `U` below, together with the `D` interface schedules and the per-block
    # spacings, all of which are derived from the leaf array and none of
    # which the cycle returns.
    _, passes, converged = adapt_to_initial_data!(
        U, ops; initial=initial_U, flags=criterion, buffer=derive_buffer(λ_initial),
        maxpasses=maxpasses, boundary=case.boundary)
    converged || throw(ErrorException(
        "the initial-data cycle had not converged after $passes passes: the " *
        "hierarchy was still changing when maxpasses ran out. The cycle " *
        "re-evaluates the initial data on each new mesh rather than " *
        "interpolating it, so it terminates when the criterion stops asking " *
        "for anything new — and a criterion that never stops is either a " *
        "maxlevel_cap that is too high for the feature or a refine_tol below " *
        "what the data can reach. Raise maxpasses only if the passes were " *
        "still making progress."))

    # One record for the whole run, handed to every problem the loop builds:
    # a regrid rebuilds the problem, and the injection and the hit count have
    # to survive that rather than start again.
    acc = ResetAccounting{float(real(T))}(D + 2; measure=accounting)
    p = HydroProblem(U, ops; eos=case.eos, floors=case.floors, limiter=limiter,
                     riemann=riemann, fixup=fixup, boundary=case.boundary,
                     accounting=acc)
    u = statevector(U)
    gather!(u, U)
    update_primitives!(p, u)
    observer === nothing || observer(p, zero(T), u)

    # The reductions come back in `float(real(T))`, which is `T` for every
    # type this package runs at and is said once here rather than assumed.
    R = float(real(T))
    totals0 = conserved_totals(U)
    scales = conserved_scales(U)
    drift = ntuple(_ -> zero(R), Val(D + 2))
    hits = 0
    ghosts = 0
    nsteps = 0
    nregrids = 0
    tracking = one(R)
    nblocks_history = Int[]
    buffer_history = Int[]
    λ_history = R[]
    λ_end_history = R[]

    nchunks = ceilint(t_end / chunk)
    for c in 1:nchunks
        tstart = min((c - 1) * chunk, t_end)
        stop = min(c * chunk, t_end)
        stop > tstart || break

        # (1) the step, from the finest spacing and the fastest signal the
        # chunk is *allowed* to grow to.
        update_primitives!(p, u)
        λ = max_signal_speed(p)
        h_min = minimum_spacing(T, forest)
        dt = hydro_dt(forest, cfl, case.speed_headroom * λ, Val(D))
        steps = max(1, ceilint((stop - tstart) / dt))
        dt_used = (stop - tstart) / steps
        u = hydro_solve!(p, u, tstart, stop, steps; reset=reset)
        nsteps += steps

        # (2) the recheck. It throws, and it is meant to.
        update_primitives!(p, u)
        λ_end = max_signal_speed(p)
        check_cfl(dt_used, h_min, D, cfl, λ_end; chunk=c, λ=λ,
                  headroom=case.speed_headroom)

        # (3) the record. The drift is the worst over the run and not the
        # endpoint's, as Burgers takes it: a leak that reversed sign would
        # otherwise hide.
        totals = conserved_totals(U)
        chunkscales = conserved_scales(U)
        drift = ntuple(v -> max(drift[v], abs(totals[v] - totals0[v])), Val(D + 2))
        scales = ntuple(v -> max(scales[v], chunkscales[v]), Val(D + 2))
        hits += floor_hits(p)
        # The ghost population, from the same recovery: `update_primitives!`
        # above ran `con2prim` over every *stored* cell, so the flag slot is
        # current in the ghosts too — which it is not after a reset, that
        # one reaching owned cells only.
        ghosts += ghost_floor_hits(p)
        push!(nblocks_history, nleaves(forest))
        push!(λ_history, λ)
        push!(λ_end_history, λ_end)
        tracking = min(tracking,
                       tracked_share(p.P; refine_tol=refine_tol,
                                     maxlevel_cap=maxlevel_cap, ε=ε, ε_g=ε_g))

        # (4) whatever is watching, before the regrid invalidates `U`.
        observer === nothing || observer(p, stop, u)

        # (5) the regrid. Not after the last chunk: the mesh that comes back
        # is then the one the returned state was computed on.
        c < nchunks || break
        flags = hydro_flags(p; refine_tol=refine_tol, coarsen_tol=coarsen_tol,
                            maxlevel_cap=maxlevel_cap, ε=ε, ε_g=ε_g)
        bufferwidth = derive_buffer(λ_end)
        push!(buffer_history, bufferwidth)
        pairs = (U => p.schedule, p.P => nothing,
                 ntuple(d -> p.fluxes[d] => nothing, Val(D))...)
        if regrid!(forest, pairs; flags=flags, buffer=bufferwidth,
                   boundary=case.boundary)
            nregrids += 1
            # `prims` and `fluxes` are handed back in because `regrid!`
            # resized them in place; reallocating would throw that away.
            p = HydroProblem(U, ops; eos=case.eos, floors=case.floors,
                             limiter=limiter, riemann=riemann, fixup=fixup,
                             boundary=case.boundary, prims=p.P, fluxes=p.fluxes,
                             accounting=acc)
            u = statevector(U)
            gather!(u, U)
            # The reset's second call site, on the freshly gathered `u`: the
            # `p = 3` prolongation into a new fine block is unlimited and
            # acts on ρ, S and E separately, so it can leave an owned cell
            # below the floors — which would otherwise wait for the first
            # stage of the next chunk to be caught. `nothing` for the
            # integrator: there is none here, and the hook does not read it.
            reset === :none || reset_atmosphere!(u, nothing, p, stop)
        end
    end

    errs = case.reference === nothing ? (l1=nothing, linf=nothing) : begin
        err = u .- case.reference(U, t_end)
        (l1=volume_weighted_norm(U, err; p=1),
         linf=volume_weighted_norm(U, err; p=Inf))
    end
    # `nothing` rather than a tuple of zeros where the injection was not
    # measured: the two are different facts, and zero is the *answer* on
    # every case that floors nowhere.
    injection = accounting ? ntuple(v -> acc.injection[v], Val(D + 2)) : nothing
    return (drift=drift, scales=scales, totals0=totals0,
            totals=conserved_totals(U), floor_hits=hits,
            reset_hits=acc.hits, ghost_hits=ghosts, injection=injection,
            nsteps=nsteps,
            nchunks=nchunks, nregrids=nregrids, passes=passes,
            converged=converged, nblocks=nleaves(forest),
            nblocks_history=nblocks_history, buffer_history=buffer_history,
            levels=forest_levels(forest),
            cells=nleaves(forest) * N^D, tracking=tracking,
            λ_initial=λ_initial, λ_history=λ_history,
            λ_end_history=λ_end_history, h=minimum_spacing(T, forest),
            l1=errs.l1, linf=errs.linf, U=hostcopy(U), u=u, forest=forest)
end

"""
    uniform_run([T], case::HydroCase, Val(D); N, ops, t_end, chunk, limiter,
                roots = case.roots, …)

The same case on a **uniform** mesh with no regridding — the reference an
adaptive run is judged against, and the control that says the refinement
bought anything.

A thin wrapper over [`evolve!`](@ref) with `maxlevel_cap = 0`, and
deliberately not a second loop: TreeWave records that a fourth
near-identical loop would be the one to drift, and the whole point of "a
case is data" is that the uniform run and the tracked run differ in a
keyword. With the cap at zero the criterion can refine nothing and
`Coarsen` is never issued below level 0, so the mesh never changes and the
buffer derivation is skipped; the tolerances are still passed, and still
mean nothing.

`roots` is the knob: the **fine** reference is the case's root brick scaled
by `2^maxlevel_cap` of the tracked run, so that the two runs share a finest
spacing, and the **coarse** control is the case's own brick, so that the two
share a coarsest one. Scaling every root count by the same factor keeps the
blocks cubes and the extents exactly what the case says they are.

Pass the tracked run's `chunk` so that the two measure `λ_max` at the same
cadence and take comparable steps; the default is a single chunk.
"""
uniform_run(case::HydroCase{T,D}, valD::Val{D}=case.valD; kwargs...) where {T,D} =
    uniform_run(T, case, valD; kwargs...)

function uniform_run(::Type{T}, case::HydroCase, valD::Val; t_end, chunk=t_end,
                     refine_tol=1 // 10, coarsen_tol=1 // 100,
                     kwargs...) where {T}
    return evolve!(T, case, valD; t_end=t_end, chunk=chunk, maxlevel_cap=0,
                   refine_tol=refine_tol, coarsen_tol=coarsen_tol, kwargs...)
end
