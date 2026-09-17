# Sod's shock tube: the discontinuous case, the first physical boundary,
# and the first comparison against an answer that is not this code's.
#
#     (ρ, v, p) = (1, 0, 1)      left of x₀
#                 (⅛, 0, ⅒)      right of it
#
# with `γ = 7/5` on `[0, 1]` to `t = 1/5`. Three waves come out of the
# discontinuity — a left-going rarefaction fan, a contact and a right-going
# shock — so one run exercises every part of the scheme that the smooth
# entropy wave cannot: the limiter where it is meant to limit, the Riemann
# solver away from the linear regime, and the recovery on states that move.
# See "Sod shock tube" in `CODE.md`.
#
# What is new here beyond the physics is the **boundary**. The entropy wave
# is periodic in every direction and hands `fill_ghosts!` a `nothing`; the
# tube is Dirichlet along its own axis and periodic across it, which is the
# first downstream use of TreeAMR's physical-boundary path. The hook is
# `boundary_by_coordinates(AllVariables(x -> prim2con(eos, initial(x))))` —
# the same callback that fills the initial data, which is what makes the
# boundary *exact*: until a wave arrives, the gas outside the box really is
# in its initial state. Once one arrives it is not, and a fixed state
# reflects, so [`assert_no_arrival`](@ref) refuses the run in advance rather
# than letting the reflection be discovered in a plot. See "Boundaries" in
# `CODE.md`.
#
# Two smaller things this file settles, both recorded in `CODE.md`:
#
#   * **Point samples at cell centers**, not cell averages. The data is
#     discontinuous, where a cell average is no better defined than a
#     sample and the closed form does not exist for a mesh whose `h` the
#     initial-data cycle will change under it. So the fill is
#     `fill_by_coordinates!` and not a host loop, unlike the entropy wave's.
#   * **`λ` comes from the exact solution, not from the initial data.** A
#     Riemann problem's fastest signal is not present at `t = 0`; see
#     [`max_signal_speed`](@ref)`(::ExactRiemann)` and the amendment under
#     "Time integration and the time step" in `CODE.md`.

"""
    SodTube(T, Val(D); ρ_L = 1, v_L = 0, p_L = 1, ρ_R = 1//8, v_R = 0,
            p_R = 1//10, γ = 7//5, x₀ = 1//2, L = 1, direction = 1,
            floors = …)

The parameters of the shock tube in `D` dimensions at working type `T`: the
two states, the position of the diaphragm, the box side, the axis the tube
runs along, the equation of state and the floors.

The defaults are Sod's own, and they are the states every finite-volume
paper of the last fifty years reports: a pressure ratio of ten and a
density ratio of eight across a stationary discontinuity in a diatomic gas.
Every one of them is a rational converted to `T` rather than a decimal
literal, so that a measured number stays put when the study is run at
another type (see "Precision" in `CODE.md`).

`direction` is the axis the tube runs along, and it is the whole of the
`D`-dimensionality of this case: the velocity has that one component, the
initial data depends on that one coordinate, and the solution is uniform
across the other `D − 1`. It is a type parameter of the struct rather than
a field, so the coordinate the initial data reads is a compile-time index.
Running the same tube along each axis in turn and comparing the results bit
for bit is the direction-independence claim, and it is cheap and sharp
precisely because the transverse directions are doing exact nothing.

**The floors sit eight orders of magnitude below the data** and are
expected never to fire — the lowest pressure anywhere in the exact solution
is `1/10` — so [`sod_errors`](@ref) reports the count and the tests assert
it is zero. A floor firing on Sod's problem would mean the scheme produced
a state the initial data does not contain.

`isbits`, like everything a kernel argument may hold: the initial-data
callback and the boundary hook both capture one of these, and the boundary
hook becomes a kernel argument at every right-hand-side evaluation.
"""
struct SodTube{T,D,DIR,E<:EquationOfState}
    ρ_L::T
    v_L::T
    p_L::T
    ρ_R::T
    v_R::T
    p_R::T
    x₀::T
    L::T
    eos::E
    floors::Floors{T}
    valD::Val{D}
    valdir::Val{DIR}
end

function SodTube(::Type{T}, ::Val{D}; ρ_L=1, v_L=0, p_L=1, ρ_R=1 // 8, v_R=0,
                 p_R=1 // 10, γ=7 // 5, x₀=1 // 2, L=1, direction=1,
                 floors::Floors{T}=Floors{T}(; ρ_atm=T(1 // 10^8),
                                             p_atm=T(1 // 10^8),
                                             p_floor=T(1 // 10^8))) where {T,D}
    dir = Int(direction)
    1 ≤ dir ≤ D || throw(ArgumentError(
        "the tube's direction must be one of the $D axes, got $dir: it is the " *
        "axis the diaphragm is normal to, the one component the velocity has, " *
        "and the one direction the mesh is not periodic in."))
    (T(ρ_L) > 0 && T(p_L) > 0 && T(ρ_R) > 0 && T(p_R) > 0) || throw(ArgumentError(
        "the shock tube needs both states positive, got (ρ_L, p_L) = " *
        "($ρ_L, $p_L) and (ρ_R, p_R) = ($ρ_R, $p_R): the exact Riemann " *
        "solution this case is measured against does not exist otherwise, and " *
        "the floors are a repair for what the scheme produces, not a licence " *
        "for what it is given."))
    0 < T(x₀) < T(L) || throw(ArgumentError(
        "the diaphragm must be inside the box, got x₀ = $x₀ and L = $L: both " *
        "physical boundaries are set from the initial data, so a diaphragm on " *
        "or outside one of them would make the whole box a single state."))
    return SodTube{T,D,dir,IdealGas{T}}(T(ρ_L), T(v_L), T(p_L), T(ρ_R), T(v_R),
                                        T(p_R), T(x₀), T(L), IdealGas(T(γ)),
                                        floors, Val(D), Val(dir))
end

"""The axis the tube runs along, as an `Int` read off the type."""
@inline tube_axis(::SodTube{T,D,DIR}) where {T,D,DIR} = DIR

"""
    sod_state(w::SodTube, x) -> P

The initial **primitive** state at position `x`: the left state where
`x[direction] < x₀` and the right state elsewhere, with the velocity in the
tube's own component and zero across it.

A plain function of `(w, x)` rather than the closure
[`sod_initial`](@ref) returns, so that the direction is a compile-time
index: the closure captures only `w`, whose type carries the axis, and this
method specializes on it.
"""
@inline function sod_state(w::SodTube{T,D,DIR}, x) where {T,D,DIR}
    left = x[DIR] < w.x₀
    ρ = left ? w.ρ_L : w.ρ_R
    v = left ? w.v_L : w.v_R
    p = left ? w.p_L : w.p_R
    return (ρ, ntuple(d -> d == DIR ? v : zero(v), Val(D))..., p)
end

"""
    sod_initial(w::SodTube)

The initial data as a closure `x -> P`, in **primitive** variables, which
is how `CODE.md` says a case states its data ("The cases": a pure `x -> P`
per case, converted with `prim2con` once per cell).

[`sod_conserved`](@ref) is the form the mesh actually takes.
"""
sod_initial(w::SodTube) = x -> sod_state(w, x)

"""
    sod_conserved(w::SodTube)

The same initial data in **conserved** variables and wrapped in
`AllVariables`: the object `fill_by_coordinates!` fills the state with, and
the object the Dirichlet hook is built from.

`AllVariables` is the once-per-cell form of TreeAMR's coordinate callbacks
— `f(x) -> NTuple{nvars}` instead of `f(x, v) -> value` — and it exists for
exactly this. `prim2con` needs all `D + 2` primitives of a point at once to
form the energy density, so the per-variable form would run the whole
conversion `D + 2` times per cell and throw all but one number away: once
per cell at setup, and once per cell of every outward-facing ghost region
at *every right-hand-side evaluation* through the boundary hook. It is one
of the two prerequisites this package took upstream (see "Upstream
prerequisites" in `CODE.md`).

Captures `w` and nothing else, and `w` is `isbits`, so it is a legal kernel
argument on any backend.
"""
sod_conserved(w::SodTube) = AllVariables(x -> prim2con(w.eos, sod_state(w, x)))

"""
    sod_boundary(w::SodTube)

The Dirichlet boundary hook: every outer ghost cell holds the conserved
initial state at its own position, for all time.

`boundary_by_coordinates(AllVariables(…))`, which is a `CellBoundary` — the
per-cell form TreeAMR launches as a kernel on every backend, as opposed to
the region form, which reads the interior and is CPU-only. Nothing here
needs the interior: a Dirichlet condition is a function of position, which
is what makes it the boundary this package uses (see "Boundaries" in
`CODE.md`).

**It is exact until a wave arrives and wrong afterwards.** The gas outside
the box is in its initial state only while no wave has reached the
boundary; after that a fixed state reflects, and a shock hitting it behaves
as if it hit a wall. That is a property of the condition and not a defect
of it — periodic boundaries would be no better, the shock merely
re-entering from the far side — so the rule is enforced ahead of the run by
[`assert_no_arrival`](@ref) rather than worked around.

The hook goes to three places, and forgetting the second is the bug that
arrives one chunk late: [`fill_ghosts!`](@ref) (through `HydroProblem`'s
`boundary` keyword, which is what step 3 put in the signature for this
step), `regrid!`, which fills ghosts before its transfer, and
`adapt_to_initial_data!`. Only the first exists yet; the other two arrive
with the driver.
"""
sod_boundary(w::SodTube) = boundary_by_coordinates(sod_conserved(w))

"""
    exact_riemann(w::SodTube) -> ExactRiemann

The exact solution of the case's own Riemann problem, in host `Float64`.

The two states and `γ` converted once through [`tofloat64`](@ref) — which
is a conversion and not a `Float64(…)` because a software float may not
offer one — and handed to the solver in `exact_riemann.jl`. The
self-similar solution is the reference at every time and the source of the
`λ` the time step is built from.
"""
exact_riemann(w::SodTube) =
    exact_riemann(tofloat64(w.eos.γ), tofloat64(w.ρ_L), tofloat64(w.v_L),
                  tofloat64(w.p_L), tofloat64(w.ρ_R), tofloat64(w.v_R),
                  tofloat64(w.p_R))

"""
    sod_forest(Val(D), N; direction = 1, roots = …, L = 1, refined = false,
               x₀ = L/2, T = Float64)

The tube's mesh: **non-periodic along `direction`, periodic across it**, a
box of side `L` along the tube and one root block thick in every other
dimension by default.

The periodicity is the case's physics and not a convenience (see
"Boundaries" in `CODE.md`). Along the tube the two `x` faces hold the
initial states, which is exact until a wave arrives. Across it the planar
solution is translation invariant, so periodic is the *only* correct
choice: a Dirichlet boundary set to the initial state would be wrong there
from the first step, because the transverse ghost cells of a cell near the
diaphragm must hold that cell's own state and not the state the initial
data assigns to a point beyond the box.

`roots` is a tuple, one root count per dimension, because the two
directions are not alike. The tube wants several roots; transversally the
solution is uniform, so one root is enough and a planar tube wastes nothing
by being thin. TreeAMR's blocks are cubes, so the transverse extents follow
from the tube's root spacing `L / roots[direction]` rather than being
given: a box of side `L` across would be `roots[direction]` times too much
mesh for a solution that does not vary in it.

**`refined` picks one of three static meshes** (added in step 5), all of
them 2:1 balanced and all refining whole root blocks, selected by where
they put the coarse-fine face relative to the two things the tube has that
the entropy wave does not — a travelling shock and a physical boundary:

- `false` (or `:none`) leaves the box uniform. The control, and what step
  4 measured on.
- `:middle` refines the root blocks whose center **along the tube** lies in
  `(L/4, 3L/4)`, as [`hydro_forest`](@ref) does for the periodic box. The
  refined region is then the middle half of the tube, the diaphragm sits
  inside it, and the shock — which travels at 1.752 from `x₀` — *leaves* it
  at `t ≈ 0.143`, before the standard `t_end = 0.2`. That the shock crosses
  a coarse-fine face during the run is the whole point of the
  configuration: a refined region the solution never leaves would make the
  conservation claim about a mesh nothing interesting happened on.
- `:left` refines every root block whose center along the tube is below
  `x₀`. The **refined region then touches the Dirichlet face** at the low
  end, and the single coarse-fine face sits at the diaphragm. This is the
  configuration "Boundaries" in `CODE.md` asks for: the physical-boundary
  hook has to fill the outer ghosts of *fine* blocks, and the prolongation
  sweep runs beside a face the hook wrote. What it does not exercise is the
  M2 ordering case in full — a prolongation reaching *tangentially* into
  hook-filled ghosts needs two physical faces meeting, and the tube is
  periodic across itself; that is Sedov's corner in step 9.

The criterion is on the tube's axis alone, unlike `hydro_forest`'s: across
the tube there is one root block by default and the solution does not vary,
so a criterion that also asked about the transverse center would refine
nothing or everything depending on the box's thickness.
"""
function sod_forest(::Val{D}, N; direction=1,
                    roots=ntuple(d -> d == direction ? 4 : 1, D), L=1,
                    refined=false, x₀=nothing, T::Type=Float64) where {D}
    dir = Int(direction)
    1 ≤ dir ≤ D || throw(ArgumentError(
        "the tube's direction must be one of the $D axes, got $dir."))
    rs = roots isa Tuple ? roots : ntuple(_ -> roots, D)
    length(rs) == D || throw(ArgumentError(
        "sod_forest needs one root count per dimension, got $(length(rs)) for " *
        "D = $D: the tube's axis and the directions across it want different " *
        "numbers, which is why this is a tuple and not a single count."))
    L = T(L)
    # Blocks are cubes, so every dimension shares the tube's root spacing.
    h = L / rs[dir]
    extents = ntuple(d -> d == dir ? (zero(T), L) : (zero(T), h * rs[d]), D)
    forest = Forest(rs; N=N, periodic=ntuple(d -> d != dir, D), extents=extents)

    (refined === false || refined === :none) && return forest
    xd = x₀ === nothing ? L / 2 : T(x₀)
    inside = if refined === :middle
        x -> L / 4 < x < 3 * L / 4
    elseif refined === :left
        x -> x < xd
    else
        throw(ArgumentError(
            "sod_forest's refined must be false, :none, :middle or :left, got " *
            "$(repr(refined)): :middle puts the coarse-fine face where the " *
            "shock crosses it and :left puts the refined region against the " *
            "Dirichlet face, and those are the two static configurations step " *
            "5 measures. A mesh that follows the solution is the driver's, in " *
            "step 7."))
    end
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        inside((ext[dir][1] + ext[dir][2]) / 2)
    end
    # A refinement that refined nothing would leave a single-level mesh, on
    # which every conservation claim below passes with no coarse-fine face
    # to make it about — the way such a test passes for the wrong reason.
    isempty(targets) && throw(ArgumentError(
        "sod_forest(refined = $(repr(refined))) selected no root block to " *
        "refine out of $(length(forest.leaves)) with roots = $rs along axis " *
        "$dir: the criterion reads the *center* of a root block along the " *
        "tube, so too few roots leaves it with nothing to pick and the mesh " *
        "single-level. Use at least four roots along the tube."))
    refine!(forest, targets)
    balance!(forest)
    return forest
end

"""
    sod_reference(U, w::SodTube, t) -> state vector

The exact solution at time `t`, sampled at every owned cell center and
converted to conserved variables, in the state vector's layout — what the
evolved state is judged against.

A Riemann problem is self-similar, so the reference at any `t > 0` is one
[`ExactRiemann`](@ref) sampled at `ξ = (x_dir − x₀)/t`; there is no
quadrature, no table and no second solve. The sampling is host `Float64`
and the result converted to `T` once per cell, which is the discipline
every reference in this package follows (see "Precision" in `CODE.md`).

**Point samples, not cell averages**, for the reason "The cases" in
`CODE.md` gives: the solution has a shock and a contact in it, and across a
discontinuity a cell average is a different `O(h)` quantity from a sample
but no more the right answer. The comparison is in the volume-weighted L1
norm, which is the norm a discontinuous solution has an order in at all.

`t = 0` works and returns the initial data: `ξ` is then `±Inf`, which
[`sample`](@ref) resolves to the two initial states exactly. A cell center
never lands on `x₀` for an even block size, so the `0/0` that would make it
a `NaN` does not arise.
"""
function sod_reference(U::FieldSet{T,D}, w::SodTube{T,D,DIR}, t) where {T,D,DIR}
    forest = U.forest
    N = forest.N
    sol = exact_riemann(w)
    t64 = tofloat64(T(t))
    t64 ≥ 0 || throw(ArgumentError(
        "the shock tube's reference is defined for t ≥ 0, got $t: the " *
        "similarity variable x/t reverses the solution for a negative time."))
    x₀ = tofloat64(w.x₀)
    host = zeros(T, statelength(U))
    arr = reshape(host, ntuple(_ -> N, D)..., U.nvars, nblocks(U))
    for b in 1:nblocks(U)
        for idx in CartesianIndices(ntuple(_ -> N, D))
            x = coordinates(T, U, b, ntuple(d -> Tuple(idx)[d] + U.G[d], D))
            ρ, u, p = sample(sol, (tofloat64(x[DIR]) - x₀) / t64)
            P = (T(ρ), ntuple(d -> d == DIR ? T(u) : zero(T), Val(D))..., T(p))
            Ucell = prim2con(w.eos, P)
            for v in 1:(D + 2)
                arr[Tuple(idx)..., v, b] = Ucell[v]
            end
        end
    end
    u = statevector(U)
    copyto!(u, host)
    return u
end

"""
    assert_no_arrival(w::SodTube, forest, t_end, λ)

Throw an `ArgumentError` naming the numbers unless `t_end · λ` is strictly
less than the distance from the diaphragm to the nearer physical boundary.

**Dirichlet-from-initial-data reflects once a wave arrives.** The boundary
holds the initial state for all time, which is the exact exterior solution
only while the interior has not reached it; afterwards the fixed state acts
as a wall. The failure is not loud — the run continues and produces a
plausible profile — so the condition is asserted *before* the run rather
than diagnosed after it. If it fires, shorten `t_end` or enlarge the box;
do not remove it. See "Boundaries" in `CODE.md`.

`λ` is the fastest signal in the exact solution, from
[`max_signal_speed`](@ref)`(::ExactRiemann)`, so the bound is the
characteristic distance and not the shock's own travel — conservative by
about 25% on Sod's data, and conservative in the direction that matters.
The boundary positions come from `forest`'s extents along the tube's axis,
so the check is about the mesh that will actually be run and not about the
case's nominal box.
"""
function assert_no_arrival(w::SodTube{T,D,DIR}, forest, t_end, λ) where {T,D,DIR}
    lo, hi = forest.extents[DIR]
    reach = tofloat64(T(t_end)) * tofloat64(T(λ))
    distance = min(tofloat64(w.x₀) - tofloat64(lo), tofloat64(hi) - tofloat64(w.x₀))
    reach < distance || throw(ArgumentError(
        "a wave reaches the Dirichlet boundary before t_end: the fastest " *
        "signal λ = $λ travels $reach by t_end = $t_end, and the diaphragm " *
        "x₀ = $(w.x₀) is only $distance from the nearer physical boundary of " *
        "[$lo, $hi] along axis $DIR. A boundary set to the initial state is " *
        "exact until a wave arrives and reflects afterwards, so the run would " *
        "measure the reflection. Shorten t_end or enlarge the box."))
    return nothing
end

"""
    sod_errors([T = Float64], Val(D); N, ops, …)

Run the shock tube to `t_end` with the Dirichlet boundary in place, and
return the whole claim: the volume-weighted `l1` and `linf`
errors of the state vector against [`sod_reference`](@ref), the
per-variable `drift` of the `D + 2` conserved integrals and the `scales`
they are measured against, the owned-cell `floor_hits`, the finest spacing
`h`, the step count, the block count, the `levels` the mesh occupies, the
three signal speeds below, and the final state — the field set `U` and the state vector `u` — so that two
runs can be compared cell by cell.

Keywords: `N` cells per block and `ops` the operator family are required;
`direction = 1`, `roots` one count per dimension (four along the tube, one
across, as [`sod_forest`](@ref) has it), `G = 2`, `limiter = :minmod`,
`riemann = :hlle`, `fixup = true`, `refined = false`, `cfl = 2//5`,
`t_end = 1//5`, `nsteps = nothing`, `λ_headroom = 1//50`,
`backend = CPU()`, and anything else goes to [`SodTube`](@ref).

`refined` is [`sod_forest`](@ref)'s, and it is what step 5 measures: the
uniform mesh is the control, `:middle` puts a coarse-fine face where the
shock crosses it, and `:left` puts the refined region against the Dirichlet
face. The time step follows the *finest* spacing, so a refined run takes
twice the steps of the uniform run it is named after and half the steps of
the uniform run at its own finest spacing — which is the run it should be
compared against.

`limiter = :minmod` is the default *here* and `:none` is the entropy wave's:
this solution has a shock in it, and an unlimited centered slope across a
shock oscillates. The norm to read is `l1` — the primary one for a
discontinuous solution, where L∞ is dominated by the one cell nearest the
shock and converges at no rate at all.

**`λ` is the exact solution's, not the initial data's.** `λ` in the result
is `max_signal_speed(exact_riemann(w))`, the supremum over all time;
`λ_initial` is what a driver measuring `max_signal_speed(p)` at `t = 0`
would have used, and `λ_ratio` is the first over the second. The ratio is
the point: a step sized from the initial data would run the first steps at
`λ_ratio` times the intended CFL number, because a Riemann problem's
fastest signal is not in its initial data. The measured number is recorded
under "Time integration and the time step" in `CODE.md`, where it sizes the
headroom factor the chunked driver of step 7 will need beside its
end-of-chunk recheck.

`λ_final` is `max_signal_speed(p)` on the final state, and the run refuses
to return if it exceeded `λ` by more than the relative `λ_headroom`.
**That slack is a second measurement and not a tolerance chosen to pass**
(measured in step 4, recorded in `CODE.md`): the *discrete* state overshoots
the exact supremum slightly at a discontinuity, because a second-order
reconstruction of a jump produces a face state the exact solution does not
contain. The overshoot on Sod is 0.49% at `N = 16` under `:minmod` and
falls as `h` does — 0.18%, 0.035%, 0.0056% at `N = 32, 64, 128` — and
0.78% falling to 0.44% under the less diffusive `:mc`. So the exact `λ` is
a bound on the *solution* and very nearly one on the *run*, and the
chunked driver's headroom has to cover this as well as the growth the
ratio above measures.

`nsteps` may be given explicitly, which is what makes a `D = 1` run and a
`D = 2` planar run comparable bit for bit: [`hydro_dt`](@ref) carries a
factor `D`, so the two would otherwise take different steps and the
identity being claimed would be hidden behind a different `dt`.
"""
sod_errors(valD::Val; kwargs...) = sod_errors(Float64, valD; kwargs...)

function sod_errors(::Type{T}, ::Val{D}; N, ops, direction=1,
                    roots=ntuple(d -> d == direction ? 4 : 1, D), G=2,
                    limiter=:minmod, riemann=:hlle, fixup=true, refined=false,
                    cfl=2 // 5, t_end=1 // 5, nsteps=nothing,
                    λ_headroom=1 // 50, backend=CPU(), params...) where {T,D}
    w = SodTube(T, Val(D); direction=direction, params...)
    forest = sod_forest(Val(D), N; direction=direction, roots=roots, L=w.L,
                        refined=refined, x₀=w.x₀, T=T)
    U = FieldSet{T}(forest, D + 2; G=G, backend=backend)
    p = HydroProblem(U, ops; eos=w.eos, floors=w.floors, limiter=limiter,
                     riemann=riemann, fixup=fixup, boundary=sod_boundary(w))

    # One callback, two jobs: the interior at setup and the exterior
    # forever. That they are the same object is what makes the boundary
    # exact rather than merely consistent.
    fill_by_coordinates!(sod_conserved(w), U)
    u = statevector(U)
    gather!(u, U)
    totals0 = conserved_totals(U)
    scales = conserved_scales(U)

    update_primitives!(p, u)
    λ_initial = max_signal_speed(p)
    λ = T(max_signal_speed(exact_riemann(w)))
    t_end = T(t_end)
    dt = hydro_dt(forest, T(cfl), λ, Val(D))
    nsteps = nsteps === nothing ? ceilint(t_end / dt) : Int(nsteps)
    assert_no_arrival(w, forest, t_end, λ)

    u = hydro_solve!(p, u, zero(T), t_end, nsteps)
    update_primitives!(p, u)
    λ_final = max_signal_speed(p)
    λ_final ≤ λ * (1 + T(λ_headroom)) || throw(ErrorException(
        "the run's fastest signal, $λ_final, exceeded the exact solution's " *
        "supremum $λ by more than the $(Float64(λ_headroom)) headroom, so the " *
        "step this run took was not the CFL number it was asked for. The exact " *
        "λ bounds every state the exact solution contains, and the discrete " *
        "state may sit a little above it — a reconstruction of a jump produces " *
        "a face state the solution does not contain — but that overshoot falls " *
        "with h (0.49% at N = 16 under :minmod, 0.0056% at N = 128) and a large " *
        "one is a different thing entirely."))

    err = u .- sod_reference(U, w, t_end)
    totals1 = conserved_totals(U)
    return (l1=volume_weighted_norm(U, err; p=1),
            linf=volume_weighted_norm(U, err; p=Inf),
            drift=ntuple(v -> abs(totals1[v] - totals0[v]), Val(D + 2)),
            scales=scales, floor_hits=floor_hits(p),
            h=minimum_spacing(T, forest), nsteps=nsteps, nblocks=nleaves(forest),
            levels=forest_levels(forest), λ=λ, λ_initial=λ_initial,
            λ_ratio=λ / λ_initial, λ_final=λ_final, U=hostcopy(U), u=u)
end
