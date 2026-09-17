# The refinement criterion: a per-cell Löhner indicator on the primitives,
# reduced to a per-block verdict through TreeAMR's `firing_boxes`.
#
# The split is the mesh's: TreeAMR takes one flag per block and says that
# "per-cell criteria are reduced to a block verdict inside the
# application's flag function". So the indicator lives here, and so does
# the reduction; the *buffering* that follows is the mesh's job, and this
# file only has to report **where** the criterion fired so that TreeAMR
# can dilate it.
#
# TreeWave's `src/refinement.jl` is the template and most of this file is
# its reasoning, transplanted. What differs, and why, is set out at length
# under "The refinement criterion" in `CODE.md`:
#
#   * **Two indicator variables, `ρ` and `p`, and never the velocity.**
#     `ρ` catches the contact and the shear layer, `p` catches shocks and
#     rarefactions, and both are positive — which is what makes the next
#     point work.
#   * **The noise floor is Löhner's local one *plus* a global term.**
#     TreeWave's fields cross zero and it had to throw the local floor
#     away; positive fields with a large dynamic range are the opposite
#     case, and need both.
#   * **One form of the criterion, not two.** TreeWave keeps a host loop
#     beside its `firing_boxes` form for historical reasons this package
#     does not have. Here there is one form and it runs on every backend.
#
# What refinement is for is **resolution**, not amplitude. The differences
# below are undivided, so the spacing enters implicitly: `τ` measures how
# well the *mesh* represents the data it holds, not the data's curvature.
# For smooth data `τ` therefore falls with `h` and a fixed threshold makes
# refinement terminate on its own — which is why the depth of the
# hierarchy is something a run measures rather than something the caller
# declares. A discontinuity is the case where it does *not* fall, and
# `maxlevel_cap` is what binds there.

"""
    lohner(um, u0, up, u_ref; ε = 1//100, ε_g = 1//1000)

The Löhner error indicator for three consecutive cell values of one
variable along one dimension: the second difference normalized by the
first differences and two noise floors,

    τ = |u₊ − 2u₀ + u₋| /
        (|u₊ − u₀| + |u₀ − u₋| + ε (|u₊| + 2|u₀| + |u₋|) + ε_g · u_ref)

Undivided differences, so `τ ∈ [0, 1]` and it measures the *mesh*: for
smooth data it falls as `h` shrinks, which is what makes refinement
terminate on its own. A zero denominator gives zero, not a `NaN`.

**Both floor terms are needed here, and that is the one place this
package's indicator differs from TreeWave's.** They guard different
things:

- `ε (|u₊| + 2|u₀| + |u₋|)` is **Löhner's own, local, floor, and it is
  right for a positive field with a large dynamic range.** Sedov's
  density spans orders of magnitude between the shell and the evacuated
  bubble, so a single global floor is simultaneously too high where the
  gas is thin — the bubble's own structure would score zero and never be
  refined — and too low where it is dense. A floor that scales with the
  local values is scale-free, which is exactly what that asks for.
  TreeWave replaced this term with a global amplitude because *its*
  fields cross zero: a tail of `3.9e-16, 3.6e-17, 3.6e-17` scored
  `τ = 0.986` against a floor that shrank along with it. `ρ` and `p` do
  not cross zero — the floors hold them above `ρ_atm` and `p_atm` — so
  that failure mode is not available here.
- `ε_g · u_ref` is **what stops the atmosphere from firing.** The
  evacuated region of a blast is not a feature: its cells sit at the
  floor, and the differences between them are numerical dust with `O(1)`
  *relative* variation at a *negligible absolute* level. The local floor
  cannot tell that from a feature, because it scales with the dust. An
  absolute term referred to the variable's global reference `u_ref` can:
  a region eight orders of magnitude below the data gets a negligible
  numerator against a fixed floor and scores `~0`. See "Floors and the
  atmosphere" in `CODE.md` for the region this guards.

So the first term guards dynamic range and the second guards the
atmosphere, and dropping either one breaks a case. `ε_g = 0` is the
negative control, and `test/refinement_tests.jl` runs it.

Both defaults are rationals converted to the value's own type with
`oftype` rather than decimal literals: a `Float64` literal in the
denominator would drag every `τ` into `Float64` however the field is
stored (see "Precision" in `CODE.md`). They are calibrated rather than
inherited — max `τ` on uniform meshes at successive `h`, tabulated under
"Step 6" in "Measured results".

!!! note "Not the canonical threshold"
    Löhner's usual `τ > 0.8` is a shock detector, and on a shock this
    indicator really does score ≈ 1 at every resolution. The thresholds
    that matter are the ones a *smooth* ramp crosses, and those are much
    smaller; see [`hydro_flags`](@ref) and `CODE.md`.
"""
@inline function lohner(um, u0, up, u_ref; ε=oftype(float(u0), 1 // 100),
                        ε_g=oftype(float(u0), 1 // 1000))
    num = abs(up - 2 * u0 + um)
    den = abs(up - u0) + abs(u0 - um) +
          ε * (abs(up) + 2 * abs(u0) + abs(um)) + ε_g * abs(u_ref)
    return iszero(den) ? zero(num) : num / den
end

"""
    cell_tau(work, idx, b, refs, ε, ε_g, ::Val{D})

The worst [`lohner`](@ref) indicator over the two indicator variables and
every dimension at one cell: `work` the ghost-inclusive working array of
the *primitive* set `P`, `idx` the cell's **stored** index, `b` its block,
`refs` the two global references from [`indicator_scales`](@ref).

The argument list is not this package's choice. It is exactly what
TreeAMR's `firing_boxes` hands a per-cell predicate — `(work, idx, b, x)`
with `idx` the stored index, so `Base.setindex(idx, idx[d] ± 1, d)` reaches
the neighbours through the ghosts — which is what makes each of
[`hydro_flags`](@ref)'s two predicates a single line.

**Which variables, and why not the velocity.** Slot `1` is `ρ` and slot
`D + 2` is `p`, the first and last of `P`'s `D + 2` primitives. The two
see different waves and neither sees both: `ρ` jumps across a contact and
across a shear layer, where `p` is flat; `p` jumps across a shock and
varies through a rarefaction, where `ρ` may be doing the same but need
not. A criterion on `ρ` alone misses a top-hat pressure deposition
entirely, which is Sedov's initial data.

The **velocity is deliberately absent**, and not as an economy. It
crosses zero — every Sedov velocity component does, on the far side of
the blast; Kelvin–Helmholtz's `v_y` does everywhere — and a field that
crosses zero is precisely the case Löhner's local floor fails on, since
the floor collapses with the values while the differences do not. `ρ` and
`p` are bounded away from zero by the floors, which is what licenses the
local term in [`lohner`](@ref) at all. See "The refinement criterion" in
`CODE.md`.

Everything it takes is `isbits` — a tuple, two scalars and a `Val` — so
it is a legal kernel argument. `ε` and `ε_g` are passed rather than
defaulted here for the same reason: a default written in terms of a
captured `T` is the documented way to fail to compile for a device.
"""
@inline function cell_tau(work, idx::NTuple{D,Int}, b::Integer, refs::NTuple{2},
                          ε, ε_g, ::Val{D}) where {D}
    # `ρ` and `p`: the first and last of the `D + 2` primitives, a tuple
    # of two compile-time constants because `D` arrives as a `Val`.
    slots = (1, D + 2)
    τ = zero(refs[1])
    for j in 1:2
        v = slots[j]
        u_ref = refs[j]
        u0 = work[idx..., v, b]
        for d in 1:D
            up = work[Base.setindex(idx, idx[d] + 1, d)..., v, b]
            um = work[Base.setindex(idx, idx[d] - 1, d)..., v, b]
            τ = max(τ, lohner(um, u0, up, u_ref; ε=ε, ε_g=ε_g))
        end
    end
    return τ
end

"""
    indicator_scales(P::FieldSet) -> (ρ_ref, p_ref)

The two global references the `ε_g` term of [`lohner`](@ref) is measured
against: the largest `ρ` and the largest `p` over every owned cell of the
primitive set.

Plain maxima rather than maxima of absolute values, because both
variables are positive by construction — the floors hold them above
`ρ_atm` and `p_atm`, and a negative one would be a bug the criterion is
the wrong place to hide.

Two `block_mapreduce` reductions, one per variable, because a device
reduction is a kernel launch and a launch takes a single variable range.
The per-block maxima are combined in block order, so the answer is
bit-identical whatever the thread count.

**Refreshed once per flagging pass.** This package's solutions change
amplitude by orders of magnitude — Sedov's peak density, the pressure
behind a spreading shell — and TreeWave measured what a reference frozen
at `t = 0` does to a blast: the criterion refines the whole domain.
[`hydro_flags`](@ref) therefore recomputes it by default, and a caller
that wants to freeze it has to say so.

!!! warning "Never from inside the predicate"
    `block_mapreduce` is itself a threaded reduction, and `firing_boxes`
    evaluates its predicate concurrently over blocks. Computing the
    references inside the predicate would nest one over the other, once
    per cell. [`hydro_flags`](@ref) evaluates them at its own call site,
    which is where the `scales` keyword exists to be hoisted from.
"""
function indicator_scales(P::FieldSet{T,D}) where {T,D}
    R = float(real(T))
    ρ_ref = maximum(block_mapreduce(identity, max, zero(R), P; vars=1))
    p_ref = maximum(block_mapreduce(identity, max, zero(R), P; vars=D + 2))
    return (ρ_ref, p_ref)
end

"""
    hydro_flags(P::FieldSet; refine_tol, coarsen_tol, maxlevel_cap,
                ε = T(1//100), ε_g = T(1//1000), scales = indicator_scales(P))
    hydro_flags(p::HydroProblem; …)

The flag vector [`regrid!`](@ref) takes, one entry per leaf, from the
Löhner indicator on a set of current primitives.

**Two entry points, one implementation** (the split arrived in step 7).
During an evolution the primitives are the ones a [`HydroProblem`](@ref)
holds and the problem is the natural argument; during
[`adapt_to_initial_data!`](@ref) there is no problem to hold them — the
forest is still changing under the cycle, so a primitive set built before
it would have the wrong number of blocks by the second pass — and the
criterion is handed a scratch set instead. The `HydroProblem` method simply
forwards `p.P`.

**`P` must be current, ghosts included.** The stencil reads one cell past
each block face, and those cells are ghosts; a driver makes them right by
calling [`update_primitives!`](@ref)`(p, u)` — steps (0) to (2) of the
right-hand side, which scatters the state, fills the conserved ghosts and
recovers the primitives in every *stored* cell. Stale ghosts do not raise
anything; they corrupt the verdict silently.

The four marks are TreeWave's, and each of the four means something:

    any cell > refine_tol, level < cap  ->  (Refine, box)
    any cell > coarsen_tol              ->  (Keep, box)    travelling margin
    nothing fired, level > 0            ->  Coarsen        (bare)
    otherwise                           ->  Keep           (bare)

- `refine_tol` is **"under-resolved here"** — the mesh is not
  representing what it holds, so go finer.
- `coarsen_tol` is **"there is something here at all"** — the feature is
  present even where it is adequately resolved. The gap between the two
  is the hysteresis dead band, which is why `coarsen_tol ≥ refine_tol` is
  refused rather than accepted as a degenerate case: with one threshold
  instead of two, a block sitting at it flips on alternate regrids and
  the hierarchy never settles.

**The box is the `coarsen_tol` sweep's, not the `refine_tol` sweep's**,
and that is load-bearing. A block refined to the cap has by construction
stopped being under-resolved — its `τ` fell below `refine_tol`, which is
precisely why refinement stopped there — so keying the box on
`refine_tol` would make a feature-holding block at the cap report
nothing, and the `(Keep, box)` margin would be unreachable in the one
case it exists for. Reporting a box is what makes a block a dilation
source, and a source asks for `level + 1` when it says `Refine` and its
own `level` when it says `Keep`: a block holding the feature at the
finest level it may reach still asks for an **equal-level margin that
travels with the feature**. A quiet `Keep` stays bare for the mirror
reason — a box on a block whose criterion did not fire would recruit its
neighbours, and since most blocks are quiet most of the time, coarsening
would die everywhere at once.

**Two sweeps, because there are two thresholds.** A firing count answers
one yes-or-no question per block and the criterion asks two: is any cell
above `refine_tol` (a count of zero is exactly `τmax ≤ refine_tol`, since
the count is over cells and `τmax` is their maximum), and *where* are the
cells above `coarsen_tol`. The second cannot be recovered from the first,
so the cells are walked twice. That is the cost of having one form of the
criterion instead of two, and it is paid once per regrid against an
evolution of many steps.

**Neither tolerance nor the cap has a default**, because each is
something the caller must think about. The values step 6 calibrated are
recorded under "Step 6" in "Measured results" in `CODE.md`, so a case can
quote them; `maxlevel_cap` is named a cap rather than a `maxlevel`
because it is one — and because `maxlevel` is TreeAMR's own exported
query, which a keyword of that name would shadow inside this body.

Everything the two predicates close over is `isbits`: the two references
as a tuple, four scalars, and a `Val`. See [`cell_tau`](@ref).
"""
hydro_flags(p::HydroProblem; kwargs...) = hydro_flags(p.P; kwargs...)

function hydro_flags(P::FieldSet{T,D}; refine_tol, coarsen_tol, maxlevel_cap,
                     ε=T(1 // 100), ε_g=T(1 // 1000),
                     scales=indicator_scales(P)) where {T,D}
    R = float(real(T))
    rtol, ctol = R(refine_tol), R(coarsen_tol)
    ctol < rtol || throw(ArgumentError(
        "coarsen_tol ($coarsen_tol) must lie strictly below refine_tol " *
        "($refine_tol): the gap between them is the hysteresis dead band. A " *
        "block refines above refine_tol and coarsens only once every cell has " *
        "dropped below coarsen_tol, so with the two equal — or inverted — a " *
        "block sitting at the threshold refines and coarsens on alternate " *
        "regrids and the hierarchy never settles."))
    length(scales) == 2 || throw(ArgumentError(
        "the indicator reads two variables, ρ and p, so it needs two global " *
        "references; got $(length(scales)). `indicator_scales(P)` is the " *
        "pair, and it is computed once per flagging pass rather than inside " *
        "the predicate."))
    P.nvars ≥ D + 2 || throw(ArgumentError(
        "the indicator reads ρ from slot 1 and p from slot $(D + 2) of the " *
        "primitive set, so it needs at least $(D + 2) variables; got " *
        "nvars=$(P.nvars). The primitive set of a HydroProblem has $(D + 4) " *
        "— the two diagnostic slots come after the primitives."))

    # Tuples and scalars in the working type, bound once here: this is what
    # the two predicates capture, and a kernel argument must be `isbits`.
    refs = (R(scales[1]), R(scales[2]))
    εR, ε_gR = R(ε), R(ε_g)
    valD = Val(D)

    refires = firing_boxes(P) do work, idx, b, x
        cell_tau(work, idx, b, refs, εR, ε_gR, valD) > rtol
    end
    boxfires = firing_boxes(P) do work, idx, b, x
        cell_tau(work, idx, b, refs, εR, ε_gR, valD) > ctol
    end

    return map(1:nblocks(P)) do b
        k = blockkey(P, b)
        nrefine, _ = refires[b]
        nbox, box = boxfires[b]
        if nrefine > 0 && level(k) < maxlevel_cap
            return (Refine, box)
        elseif nbox > 0
            return (Keep, box)
        elseif level(k) > 0
            return Coarsen
        else
            return Keep
        end
    end
end

"""
    refinement_buffer(forest, maxlevel_cap, travel)

The buffer width in cells that covers a feature moving `travel` in
physical units between one regrid and the next, measured at the spacing
of level `maxlevel_cap`:

    buffer = ceil(travel / spacing(forest, maxlevel_cap)) + 1

The width is the application's to choose because it is physics — feature
speed times regrid cadence, `travel = λ_max · chunk`, which the driver of
step 7 supplies — and the mesh cannot know it. What TreeAMR measured is
that the margin must *exceed* the motion it covers: a margin narrower
than the travel per interval came out slightly worse than no margin at
all, since the feature leaves the refined region either way and the
narrow buffer only adds cells. Hence the `+ 1` rather than a bare `ceil`.

The spacing is taken at `maxlevel_cap` rather than from
`minimum_spacing`, which reports the *current* finest spacing — coarse
while the hierarchy is still being built, and so would derive a uselessly
narrow margin on the first adaptation pass, which is the one that builds
the hierarchy.

Recruitment reaches exactly one ring of neighbours, so TreeAMR caps the
buffer at `N`. That cap is really a statement about cadence — the feature
may not cross a whole finest-level *block* between regrids — and this
throws naming the constraint rather than letting the caller discover it
as an opaque rejection from inside [`regrid!`](@ref). `λ_max` on a
hydrodynamic problem is not the constant 1 that TreeWave's wave speed is,
and it grows: Sod's post-shock gas is 1.85 times faster than anything in
its initial data (see "Time integration and the time step" in `CODE.md`),
so a cadence that fits at `t = 0` need not fit later.
"""
function refinement_buffer(forest::Forest, maxlevel_cap::Integer, travel::Real)
    h = spacing(forest, maxlevel_cap)
    cells = ceilint(travel / h) + 1
    cells <= forest.N || throw(ArgumentError(
        "a feature travelling $travel between regrids needs a $cells-cell " *
        "margin at level $maxlevel_cap (h = $h), which exceeds the block " *
        "width N = $(forest.N). Recruitment reaches one ring of neighbours, " *
        "so the travel must stay under one finest-level block width, " *
        "$(forest.N * h): regrid more often, or lower the cap."))
    return cells
end
