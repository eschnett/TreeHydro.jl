# The Sedov blast wave: the strong shock, the first case in which a floor
# actually fires, and the first mesh problem the package was built for.
#
#     (ρ, v, p) = (ρ₀, 0, (γ−1) E₀ / V_D(r₀))     inside |x| < r₀
#                 (ρ₀, 0, p_amb)                  outside
#
# with `γ = 7/5` on `[−L/2, L/2]^D` and **Dirichlet ambient on every face**.
# One run exercises what neither the entropy wave nor the shock tube can:
#
#   * **The floors and the atmosphere reset, for real.** Step 8 built the
#     reset and measured it on cases where nothing fires; this is the case
#     where the measurement has a nonzero answer to give — and the answer
#     (measured in step 9) is not the one the design expected. The similarity
#     solution's interior density falls like `λ^{D/(γ−1)}`, but the discrete
#     bubble never gets near `ρ_atm`: numerical diffusion holds it at
#     `ρ ≈ 6.7e-2`, so the *atmosphere* rule never fires. What fires is the
#     *pressure* floor, where a strong shock crosses a coarse-fine face and the
#     interface flux restriction acts on gas whose internal energy is
#     `p_amb/(γ−1)`. See "Floors and the atmosphere" in `CODE.md`.
#   * **A refined region that follows a closed expanding surface.** The
#     feature is a shock travelling outward in every direction, not a front
#     crossing a box. `CODE.md` predicted a shell whose interior coarsens and a
#     block count that rises and then falls; measured in step 9, the region is
#     a growing *disk* and the count rises monotonically, because the Sedov
#     interior is a steep density ramp the Löhner indicator correctly fires on.
#     It still tracks the blast and still saves cells. TreeWave could only
#     imitate this problem with the wave equation.
#   * **The boundary hook on edges and corners.** Every face is Dirichlet, so
#     two physical faces meet — which is the M2 ordering case "Boundaries" in
#     `CODE.md` has been waiting for since step 4, and which a shock tube,
#     periodic across itself, cannot produce.
#   * **The 3D coarse-fine face under a real shock**, where the fixup averages
#     `2 × 2` fine faces.
#
# Three things about this case are easy to get wrong, and each is written out
# where it happens:
#
#   * **Deposition is a function of position, not "one cell".** The
#     initial-data cycle re-evaluates the data on every mesh it produces, so
#     the top hat has to be defined in physical space; `r₀` spans several
#     cells at the refinement cap — eight in the measured `D = 1, 2`
#     configurations, four in `D = 3` — so that it is resolved on the mesh the
#     cycle converges to.
#   * **The `E₀` that enters the similarity law is the measured one.** Which
#     cell centres fall inside `r₀` is a property of the mesh, so the nominal
#     `E₀ = 1` is not what the blast carries; [`measured_E₀`](@ref) is.
#   * **The fastest signal grows in the first chunk**, as it does on Sod and
#     for the same reason — the jump at `r₀` is a Riemann problem, and the gas
#     on the hot side of its contact moves. `CODE.md` said `λ_max` "only
#     decreases" here; step 9 measured the first chunk and amended it.
#
# See "Sedov blast wave" in `CODE.md` for the case as decided, and "Step 9" in
# "Measured results" for every number below.

"""
    SedovBlast(T, Val(D); ρ₀ = 1, p_amb = 1//10^5, E₀ = 1, r₀ = 1//16,
               γ = 7//5, L = 1, floors = …)

The parameters of the blast in `D` dimensions at working type `T`: the ambient
state, the energy and the radius of the top hat it is deposited in, the box
side, the equation of state and the floors. The box is
`[−L/2, L/2]^D` and the explosion sits at the origin.

`p_hot = (γ−1) E₀ / V_D(r₀)` is computed once here and carried as a field, with
`V_1 = 2r₀`, `V_2 = π r₀²`, `V_3 = 4π r₀³/3`. That keeps `π` and the volume
formula out of [`sedov_state`](@ref), which is a kernel-callable function
evaluated at every cell of every mesh the initial-data cycle produces and at
every outward-facing ghost at every right-hand-side evaluation.

**`r₀` is a resolution parameter as much as a physical one.** The top hat has
to span several cells at the refinement cap, or the deposition is a single
cell on the finest mesh and the blast starts from a delta function the scheme
cannot represent; the default `1//16` is eight cells at the cap of the
two-dimensional configuration step 9 measures (`roots = 4`, `N = 8`,
`cap = 2`, so `h_cap = 1/128`), and eight rather than three or four because
the hot spot's sound speed goes as `r₀^{-D/2}` and sets the regrid cadence —
see `test/sedov_tests.jl` and "Step 9" in `CODE.md`. The energy the mesh actually
receives is [`measured_E₀`](@ref) and not `E₀`, because which cell centres fall
inside `r₀` depends on the mesh.

**Why these floors** (and there are no defaults for them anywhere else in the
package, for the reason [`Floors`](@ref) gives):

- `ρ_atm = 1//10^6` is **six orders below `ρ₀`**, which is the depth the
  refinement criterion needs and the depth the solution reaches. `CODE.md`
  measures the `ε_g` term silencing an atmosphere six orders below the data at
  `τ = 0.0020` and five orders at a marginal 0.0196, so a shallower atmosphere
  would make the evacuated bubble fire the criterion; and the similarity
  solution's own `G(λ) ∼ λ^{D/(γ−1)}` passes `10⁻⁶` at `λ ≈ 0.06` in `D = 2`,
  so it is also where the gas genuinely runs out.
- `p_atm = 1//10^6` is the pressure a vacuum cell becomes. It sits **below
  `p_amb`** on purpose: an atmosphere pressure above the ambient would make the
  reset *raise* the energy of undisturbed gas if it ever fired there.
- `p_floor = 1//10^8` is two orders below `p_amb`, so the pressure floor can
  only fire on a recovery that produced a state the initial data does not
  contain — which is what a floor count means as a measurement.

Every parameter is a rational converted to `T` rather than a decimal literal,
so a measured number stays put when the study is run at another type (see
"Precision" in `CODE.md`). `isbits`, like everything a kernel argument may
hold: the initial-data callback and the boundary hook both capture one of
these.
"""
struct SedovBlast{T,D,E<:EquationOfState}
    ρ₀::T
    p_amb::T
    E₀::T
    r₀::T
    p_hot::T
    L::T
    eos::E
    floors::Floors{T}
    valD::Val{D}
end

"""
    deposition_volume(T, r₀, ::Val{D})

The volume of the top hat: `2r₀`, `π r₀²`, `4π r₀³/3` in `D = 1, 2, 3`.

The planar value counts **both** sides, which is the convention
[`sedov_alpha`](@ref) matches and the reason this package's planar `α` is twice
the literature's. A `D` outside `1:3` is refused rather than guessed at: the
similarity reference is stated for the three geometries and nothing here has a
fourth.
"""
function deposition_volume(::Type{T}, r₀, ::Val{D}) where {T,D}
    r = T(r₀)
    D == 1 && return 2 * r
    D == 2 && return T(π) * r^2
    D == 3 && return 4 * T(π) * r^3 / 3
    throw(ArgumentError(
        "the Sedov blast is stated for D = 1, 2 or 3 — planar, cylindrical " *
        "and spherical — got D = $D, for which neither the deposition volume " *
        "nor the similarity law has a form here."))
end

function SedovBlast(::Type{T}, ::Val{D}; ρ₀=1, p_amb=1 // 10^5, E₀=1,
                    r₀=1 // 16, γ=7 // 5, L=1,
                    floors::Floors{T}=Floors{T}(; ρ_atm=T(1 // 10^6),
                                                p_atm=T(1 // 10^6),
                                                p_floor=T(1 // 10^8))) where {T,D}
    ρ₀, p_amb, E₀, r₀, L = T(ρ₀), T(p_amb), T(E₀), T(r₀), T(L)
    (ρ₀ > 0 && p_amb > 0 && E₀ > 0) || throw(ArgumentError(
        "the blast needs a positive ambient state and a positive energy, got " *
        "ρ₀ = $ρ₀, p_amb = $p_amb and E₀ = $E₀: the similarity law it is " *
        "measured against exists for no other, and the floors are a repair " *
        "for what the scheme produces, not a licence for what it is given."))
    0 < r₀ < L / 2 || throw(ArgumentError(
        "the deposition radius must be inside the box, got r₀ = $r₀ and " *
        "L = $L: the top hat is where the energy goes, and one that reached " *
        "the Dirichlet boundary would make the whole run a reflection."))
    ρ₀ > floors.ρ_atm || throw(ArgumentError(
        "the ambient density $ρ₀ is at or below ρ_atm = $(floors.ρ_atm): the " *
        "atmosphere reset would replace the undisturbed gas with the " *
        "atmosphere on the first stage, so the case would never have an " *
        "ambient at all. The atmosphere belongs orders of magnitude below " *
        "the ambient — six, for the refinement criterion's global floor term."))
    p_amb > floors.p_floor || throw(ArgumentError(
        "the ambient pressure $p_amb is at or below p_floor = " *
        "$(floors.p_floor): the pressure floor would fire on undisturbed gas, " *
        "and the floor counts this case exists to measure would count the " *
        "initial data rather than the scheme."))
    p_hot = (T(γ) - 1) * E₀ / deposition_volume(T, r₀, Val(D))
    p_hot > p_amb || throw(ArgumentError(
        "the deposited pressure $p_hot does not exceed the ambient $p_amb: " *
        "there is no blast. Raise E₀ or shrink r₀."))
    return SedovBlast{T,D,IdealGas{T}}(ρ₀, p_amb, E₀, r₀, p_hot, L,
                                       IdealGas(T(γ)), floors, Val(D))
end

"""
    sedov_state(w::SedovBlast, x) -> P

The initial **primitive** state at position `x`: gas at rest at density `ρ₀`
everywhere, at pressure `p_hot` inside the top hat `|x| < r₀` and `p_amb`
outside.

**A function of position, which is the whole point** (decided in `CODE.md`).
The initial-data cycle regrids and then *re-evaluates* the data on each new
mesh rather than interpolating it, so "the energy goes in one cell" is not
something this case can say: it would mean a different physical problem on
every pass of the cycle. A top hat of a fixed physical radius is the same
problem on every mesh, and the energy it actually deposits — which does depend
on the mesh, through which cell centres land inside — is what
[`measured_E₀`](@ref) reports and what the similarity law is then applied with.

A plain function of `(w, x)` rather than the closure [`sedov_initial`](@ref)
returns, so that the closure captures only `w`, which is `isbits`.

The comparison is on `r²` against `r₀²` and takes no square root: a cell centre
never lands exactly on the top hat's edge for the radii used here, and squaring
keeps the whole of it in `T`.
"""
@inline function sedov_state(w::SedovBlast{T,D}, x) where {T,D}
    r² = sum(ntuple(d -> x[d] * x[d], Val(D)))
    p = r² < w.r₀ * w.r₀ ? w.p_hot : w.p_amb
    return (w.ρ₀, ntuple(_ -> zero(T), Val(D))..., p)
end

"""
    ambient_state(w::SedovBlast) -> P

The undisturbed gas, `(ρ₀, 0, …, 0, p_amb)` — the state the Dirichlet
boundary holds on every face for all time, and the state the corner and edge
tests assert the outward-facing ghosts against.
"""
@inline ambient_state(w::SedovBlast{T,D}) where {T,D} =
    (w.ρ₀, ntuple(_ -> zero(T), Val(D))..., w.p_amb)

"""
    sedov_initial(w::SedovBlast)

The initial data as a closure `x -> P`, in **primitive** variables, which is
how `CODE.md` says a case states its data.
"""
sedov_initial(w::SedovBlast) = x -> sedov_state(w, x)

"""
    sedov_conserved(w::SedovBlast)

The same initial data in **conserved** variables and wrapped in `AllVariables`:
the object `fill_by_coordinates!` fills the state with, and the object the
Dirichlet hook is built from. See [`sod_conserved`](@ref) for why the
all-variables form is a prerequisite rather than a convenience.
"""
sedov_conserved(w::SedovBlast) =
    AllVariables(x -> prim2con(w.eos, sedov_state(w, x)))

"""
    sedov_boundary(w::SedovBlast)

The Dirichlet hook: every outer ghost cell holds the conserved ambient state at
its own position, on **every** face, for all time.

This is the case "Boundaries" in `CODE.md` names as the one that runs the hook
on edges and corners. A shock tube's two physical faces never meet — it is
periodic across itself — so the tangential half of TreeAMR's M2 ordering rule,
a prolongation reaching sideways into hook-filled ghosts, has no configuration
in this package until here. With six faces (three in `D = 2`) meeting at edges
and corners, a fine block in a corner of the box has ghost regions that only
two physical faces together can fill, and `test/sedov_tests.jl` asserts every
stored entry of them.

Exact until the shock arrives and a reflecting wall afterwards, as every
Dirichlet-from-initial-data boundary is; [`assert_no_arrival`](@ref) refuses the
run in advance rather than letting the reflection turn up in a plot. Here the
condition is sharper than the tube's, because the Sedov law says in advance
where the shock will be.
"""
sedov_boundary(w::SedovBlast) = boundary_by_coordinates(sedov_conserved(w))

"""
    sedov_similarity(w::SedovBlast) -> SedovSimilarity

The similarity law for this case's `γ` and dimension, in host `Float64` — the
reference the exponent, the radius and the arrival check are read from.
"""
sedov_similarity(w::SedovBlast{T,D}) where {T,D} =
    SedovSimilarity(tofloat64(w.eos.γ), D)

"""
    HydroCase(w::SedovBlast; roots = 4, speed_headroom = 2)

The blast as a case the driver can run: **Dirichlet on every face**, with
[`sedov_boundary`](@ref) as the hook and **no reference**, since the closed-form
radial profile is an extension rather than a milestone (`CODE.md`) and the
acceptance is the exponent, the jump and a uniform fine run.

`roots` must be the same in every dimension. The box is a cube and TreeAMR's
blocks are cubes, so anything else would describe a different box than
`[−L/2, L/2]^D`; and [`uniform_run`](@ref) builds its fine reference by scaling
every root count by `2^cap`, which leaves a cubic brick cubic.

**`speed_headroom = 2`, and the number is measured rather than inherited**
(step 9). `CODE.md` said the hot spot's sound speed at `t = 0` is the maximum
and `λ_max` only decreases afterwards; that is right about the *blast* and
wrong about the *first chunk*, for exactly the reason it was wrong on Sod. The
jump at `r₀` is a Riemann problem, and the gas on the hot side of its contact
moves at `u★` with its own sound speed, so `|v| + c_s` there exceeds the hot
spot's `c_s` before the discontinuity has resolved. The measured first-chunk
growth is recorded in `CODE.md` under "Step 9"; `2` covers it, as it does
Sod's 1.8522.

Beware what the headroom costs at the *other* end: the travelling margin is
derived from `speed_headroom · λ · chunk` at the cap's spacing and must stay
under one finest-level block width, so doubling the headroom halves the
admissible chunk — and Sedov's early `λ` is the hot spot's sound speed,
`sqrt(γ p_hot/ρ₀)`, which is large. That product is what sets the regrid
cadence for this case; see "Things that will bite" in `CLAUDE.md`.
"""
function HydroCase(w::SedovBlast{T,D}; roots=4, speed_headroom=2) where {T,D}
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    allequal(rs) || throw(ArgumentError(
        "the Sedov blast needs the same root count in every dimension, got " *
        "$rs: the box is the cube [−L/2, L/2]^$D and TreeAMR's blocks are " *
        "cubes, so unequal counts would describe a box of another shape — and " *
        "the blast is isotropic, so there is no direction to spend the " *
        "asymmetry on."))
    half = w.L / 2
    extents = ntuple(_ -> (-half, half), D)
    return HydroCase(T, Val(D); initial=x -> sedov_state(w, x), eos=w.eos,
                     floors=w.floors, boundary=sedov_boundary(w),
                     periodic=ntuple(_ -> false, D), extents=extents, roots=rs,
                     speed_headroom=speed_headroom, reference=nothing)
end

"""
    sedov_forest(Val(D), N; roots = 4, L = 1, refined = false, T = Float64)

The blast's mesh: **non-periodic in every direction**, the cube
`[−L/2, L/2]^D` on a `roots^D` brick of `N`-cell blocks.

`refined` picks one of four **static** meshes — the configurations step 9
measures with [`hydro_solve!`](@ref) directly, as `sod_forest`'s `:middle` and
`:left` are for the tube. Two put a coarse-fine face where the blast crosses
it, and two put fine blocks where Dirichlet faces meet:

- `false` (or `:none`) leaves the box uniform. The control.
- `:center` refines the `2^D` root blocks around the origin — those whose
  centre lies within `L/4` of it in every dimension — so the blast **starts
  inside the refined region and leaves it**, at `|x_d| = L/4`. This is
  `sod_forest(:middle)`'s role for the blast, and it is the only
  configuration in this package in which a *strong shock* crosses a
  coarse-fine face: the tracked mesh of [`evolve!`](@ref) keeps the shock at
  the cap by construction, so its coarse-fine faces stand in undisturbed gas
  and carry no flux to conserve (measured in step 9). Every claim about the
  interface flux restriction, the prolongation order and the ghost floor count
  on this case is made here. It needs `roots ≥ 4`; with fewer, every root
  block touches the origin and there is nothing left coarse.
- `:corner` refines the one root block in the **low corner of the box**, where
  every dimension is at its own physical face. In `D = 2` that is a corner in
  the ordinary sense, two Dirichlet faces meeting; in `D = 3` it is three.
- `:edge` refines the root blocks at the low face of the **first two**
  dimensions, spanning the last — an edge of the box in `D = 3`, where two
  Dirichlet faces meet along a line of blocks. It is refused in `D < 3`,
  where it would select exactly the blocks `:corner` does and the two names
  would be one configuration under two claims.

The last two put *fine* blocks against two physical faces at once, which is the
configuration "Boundaries" in `CODE.md` has been asking for since step 4: the
prolongation sweep runs after the boundary hook and reaches **tangentially**
into the ghosts the hook wrote, and only two faces meeting can produce a ghost
region that one face alone does not fill. The blast stays at the centre and
never reaches them within the few steps these meshes are run for, which is what
makes "the corner block's interior is bit-identical to the ambient afterwards"
a claim about the exchange and not about the physics.
"""
function sedov_forest(::Val{D}, N; roots=4, L=1, refined=false,
                      T::Type=Float64) where {D}
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    allequal(rs) || throw(ArgumentError(
        "sedov_forest needs the same root count in every dimension, got $rs: " *
        "the box is a cube and so are TreeAMR's blocks."))
    L = T(L)
    half = L / 2
    extents = ntuple(_ -> (-half, half), D)
    forest = Forest(rs; N=N, periodic=ntuple(_ -> false, D), extents=extents)

    (refined === false || refined === :none) && return forest
    selector = if refined === :center
        k -> begin
            ext = block_extent(forest, k)
            all(d -> abs((ext[d][1] + ext[d][2]) / 2) < half / 2, 1:D)
        end
    elseif refined === :corner
        k -> begin
            ext = block_extent(forest, k)
            all(d -> ext[d][1] ≈ -half, 1:D)
        end
    elseif refined === :edge
        D ≥ 3 || throw(ArgumentError(
            "sedov_forest(refined = :edge) needs D ≥ 3, got D = $D: an edge " *
            "is where two faces meet along a line, and in $(D) dimension" *
            "$(D == 1 ? "" : "s") that line is the corner :corner already " *
            "selects. Use :corner, and keep :edge for the three-dimensional " *
            "mesh it names."))
        k -> begin
            ext = block_extent(forest, k)
            all(d -> ext[d][1] ≈ -half, 1:2)
        end
    else
        throw(ArgumentError(
            "sedov_forest's refined must be false, :none, :center, :corner or " *
            ":edge, got $(repr(refined)): :center puts a coarse-fine face " *
            "where the blast crosses it, :corner puts a fine block where " *
            "every Dirichlet face meets and :edge puts a line of them where " *
            "two do, and those are the static configurations step 9 measures. " *
            "A mesh that follows the blast is the driver's."))
    end
    targets = filter(selector, forest.leaves)
    # A refinement that refined nothing would leave a single-level mesh on
    # which every claim below passes with no coarse-fine face and no fine
    # boundary block to make it about.
    isempty(targets) && throw(ArgumentError(
        "sedov_forest(refined = $(repr(refined))) selected no root block out " *
        "of $(length(forest.leaves)) with roots = $rs: :center reads each " *
        "root block's centre against L/4 and needs at least four roots per " *
        "dimension — with two, every root block touches the origin and there " *
        "is nothing left coarse — while :corner and :edge read the low " *
        "extent against the box's."))
    refine!(forest, targets)
    balance!(forest)
    return forest
end

"""
    measured_E₀(r, w::SedovBlast) -> Float64

The energy the blast actually carries: the total energy integral at `t = 0` on
the mesh the initial-data cycle produced, less the ambient thermal energy of
the whole box.

**This and not the nominal `E₀` is what enters the similarity law** (decided in
`CODE.md`). The deposition is a top hat in space and the mesh samples it at cell
centres, so the energy the mesh receives is
`(p_hot − p_amb)/(γ−1)` times the volume of the cells whose centres fall inside
`r₀` — which is `V_D(r₀)` only in the limit, and differs from it by a percent
or so at the resolutions the tests run. Reading the law with the nominal value
would put a systematic error into the radius and therefore into the intercept
of the exponent fit.

`r` is an [`evolve!`](@ref) result, read for `totals0` — the conserved integrals
before the first step — and `forest`, read for the box the ambient is
subtracted over. The gas is at rest at `t = 0`, so the whole of the energy
integral is thermal and no kinetic term has to be separated out.
"""
function measured_E₀(r, w::SedovBlast{T,D}) where {T,D}
    volume = prod(ntuple(d -> tofloat64(T(r.forest.extents[d][2])) -
                              tofloat64(T(r.forest.extents[d][1])), D))
    ambient = tofloat64(w.p_amb) / (tofloat64(w.eos.γ) - 1) * volume
    return tofloat64(T(r.totals0[D + 2])) - ambient
end

"""
    shock_radius(P::FieldSet, w::SedovBlast; threshold = 3//2) -> T

How far the blast has got: the greatest distance from the origin to the centre
of an owned cell whose density exceeds `threshold · ρ₀`.

**A density threshold, because the shock is where the density is.** The
post-shock compression of a strong shock is `(γ+1)/(γ−1) = 6` and the ambient
is `ρ₀`, so a threshold anywhere between the two finds the shell and nothing
else; `3//2` is low enough that a captured shock — which reaches only part of
the ideal jump on a coarse mesh — is still found, and high enough that the
precursor a limiter leaves ahead of the front is not. The measured peak is
reported beside the radius by [`peak_compression`](@ref), so a threshold that
had stopped being crossed would be visible rather than silent.

The value is **quantized at the cell spacing** and lands on the outermost
*firing cell centre*, which is up to half a cell inside the true front and at
least half a cell outside the last unshocked cell. That is the tolerance every
claim made on it has to carry, and it is why the exponent is fitted over many
chunks rather than read off two.

**A host loop, and deliberately not `firing_boxes`.** The natural device form —
one `firing_boxes` sweep on `ρ > threshold·ρ₀`, the per-block bounding box
converted to coordinates, the radius taken at its outermost corner — is
*biased*, because a bounding box loses the correlation between dimensions: the
shell crosses a block diagonally, so the box's corner sticks out beyond the
outermost firing cell by about `w²/(2 r_s)` for a block of width `w`. At the
sizes step 9 runs that is 8% at the end of the run and 22% at the start, and a
bias that shrinks as the shock grows lands directly on the *slope* being
measured — it cost about `0.13` of the exponent in `D = 2`. So this is an
oracle in the spirit of [`reduce_to_grid`](@ref): it reads positions and
values on the host, in block order, and knows nothing about how the data got
there. It runs once per chunk, against hundreds of steps.

**`P` must be current**, as the refinement criterion needs it to be:
[`update_primitives!`](@ref) is what leaves it so, and [`evolve!`](@ref) calls
the observer with it already done.
"""
function shock_radius(P::FieldSet{T,D}, w::SedovBlast{T,D};
                      threshold=3 // 2) where {T,D}
    R = float(real(T))
    thr = R(threshold) * R(w.ρ₀)
    thr > R(w.ρ₀) || throw(ArgumentError(
        "shock_radius needs a threshold above 1, got $threshold: the ambient " *
        "density is ρ₀ itself, so a threshold at or below it fires in every " *
        "cell of the box and reports the corner of the domain."))
    work = Array(P.work)
    N = P.forest.N
    best = zero(R)
    for b in 1:nblocks(P)
        for idx in CartesianIndices(ntuple(d -> (P.G[d] + 1):(P.G[d] + N), D))
            work[Tuple(idx)..., 1, b] > thr || continue
            x = coordinates(T, P, b, Tuple(idx))
            best = max(best, sum(ntuple(d -> R(x[d]) * R(x[d]), Val(D))))
        end
    end
    return sqrt(best)
end

"""
    peak_compression(P::FieldSet, w::SedovBlast) -> T

The largest density anywhere, divided by `ρ₀` — the **post-shock density jump**
the strong-shock limit puts at `(γ+1)/(γ−1) = 6`.

The second of the two checks that need no similarity constant (the exponent is
the first). A captured shock does not reach `6`: the scheme spreads the jump
over three or four cells and the peak is the average over the cell the front
sits in, so the measured value approaches `6` from below as `h` falls and
*exceeding* it by more than a little would mean an overshoot the limiter was
supposed to prevent. Both directions are asserted in `test/sedov_tests.jl`.

One `block_mapreduce` over the density slot of the primitive set, combined in
block order, so the answer does not move with the thread count.
"""
function peak_compression(P::FieldSet{T,D}, w::SedovBlast{T,D}) where {T,D}
    R = float(real(T))
    return maximum(block_mapreduce(identity, max, zero(R), P; vars=1)) / R(w.ρ₀)
end

"""
    sedov_static([T = Float64], Val(D); N, ops, r₀, t_end, …)

Run the blast to `t_end` on a mesh **fixed before the first step**, and return
what the static claims are made of: the per-variable `drift` of the `D + 2`
conserved integrals and the `scales` they are measured against, the three floor
counts and the reset's measured `injection`, the shock radius and the peak
compression at the end, the mesh statistics, and the final state — the field set
`U` and the state vector `u` — so that two runs can be compared cell by cell.

**This is where the coarse-fine claims of this case are made, and
[`evolve!`](@ref) is where they cannot be.** A tracked mesh keeps the shock at
the refinement cap by construction — that is what `tracking == 1` means — so
every coarse-fine face of a tracked run stands in gas the blast has not reached,
carries no flux worth restricting, and leaves the interface flux restriction
with nothing to do. Measured in step 9: the tracked run, the same run with
`fixup = false`, and the same run at `p = 1` agree in every column. So the fixup,
the prolongation order and the ghost floor count are measured on
[`sedov_forest`](@ref)`(refined = :center)`, where the mesh is fixed, the blast
starts inside the refined region and leaves it, and a strong shock really does
cross a coarse-fine face. `sod_errors` plays the same role for the tube and
`refined = :middle` is the same idea.

Keywords: `N` cells per block and `ops` the operator family are required;
`roots = 4`, `r₀`, `t_end`, `refined = false` ([`sedov_forest`](@ref)'s), `G = 2`,
`limiter = :minmod`, `riemann = :hlle`, `fixup = true`, `reset = :stage`,
`cfl = 2//5`, `nsteps = nothing`, `speed_headroom = 2`, `accounting = true`,
`backend = CPU()`, and anything else goes to [`SedovBlast`](@ref).

**The step count is sized with a headroom and rechecked, exactly as the driver
does it.** `λ_max` from the initial data is the hot spot's sound speed, and it is
not a bound on the run for the reason `CODE.md`'s "Time integration and the time
step" gives: the jump at `r₀` is a Riemann problem and its star region is not
present at `t = 0`. So the step is sized from `speed_headroom · λ` and
[`check_cfl`](@ref) is called on the state that comes out — which **throws** if
the step taken was not the step the CFL number asked for, here as there.
`nsteps` may be given explicitly, which is what makes a run and its
`fixup = false` or `p = 1` control comparable: they must take the same steps or
the difference between them is not the thing being measured.

`accounting = true` by default, unlike [`evolve!`](@ref): this is a measurement
driver and the injection is one of the numbers it exists to report.
"""
sedov_static(valD::Val; kwargs...) = sedov_static(Float64, valD; kwargs...)

function sedov_static(::Type{T}, ::Val{D}; N, ops, roots=4, r₀=1 // 16,
                      t_end=1 // 8, refined=false, G=2, limiter=:minmod,
                      riemann=:hlle, fixup=true, reset=:stage, cfl=2 // 5,
                      nsteps=nothing, speed_headroom=2, accounting::Bool=true,
                      backend=CPU(), params...) where {T,D}
    w = SedovBlast(T, Val(D); r₀=r₀, params...)
    forest = sedov_forest(Val(D), N; roots=roots, L=w.L, refined=refined, T=T)
    U = FieldSet{T}(forest, D + 2; G=G, backend=backend)
    acc = ResetAccounting{float(real(T))}(D + 2; measure=accounting)
    p = HydroProblem(U, ops; eos=w.eos, floors=w.floors, limiter=limiter,
                     riemann=riemann, fixup=fixup, boundary=sedov_boundary(w),
                     accounting=acc)

    # One callback, two jobs: the interior at setup and the exterior forever.
    fill_by_coordinates!(sedov_conserved(w), U)
    u = statevector(U)
    gather!(u, U)
    totals0 = conserved_totals(U)
    scales = conserved_scales(U)

    update_primitives!(p, u)
    λ = max_signal_speed(p)
    t_end = T(t_end)
    h_min = minimum_spacing(T, forest)
    assert_no_arrival(w, forest, t_end, w.E₀)
    steps = nsteps === nothing ?
            max(1, ceilint(t_end / hydro_dt(forest, T(cfl),
                                            T(speed_headroom) * λ, Val(D)))) :
            Int(nsteps)
    u = hydro_solve!(p, u, zero(T), t_end, steps; reset=reset)

    update_primitives!(p, u)
    λ_end = max_signal_speed(p)
    check_cfl(t_end / steps, h_min, D, T(cfl), λ_end; λ=λ,
              headroom=T(speed_headroom))
    totals1 = conserved_totals(U)
    # The scale is the larger of the two endpoints', as `evolve!` takes it as
    # the largest over chunks. At `t = 0` the gas is at rest, so every
    # momentum scale `Σ hᴰ |S_d|` is **exactly zero** and a drift measured
    # against it would have no yardstick at all — the same trap Sod's
    # momentum sets, and here it is the initial scale that is empty rather
    # than the final one.
    endscales = conserved_scales(U)
    scales = ntuple(v -> max(scales[v], endscales[v]), Val(D + 2))
    return (drift=ntuple(v -> abs(totals1[v] - totals0[v]), Val(D + 2)),
            scales=scales, totals0=totals0, totals=totals1,
            floor_hits=floor_hits(p), reset_hits=acc.hits,
            ghost_hits=ghost_floor_hits(p),
            injection=accounting ?
                      ntuple(v -> acc.injection[v], Val(D + 2)) : nothing,
            r_s=shock_radius(p.P, w), peak=peak_compression(p.P, w),
            h=h_min, nsteps=steps, nblocks=nleaves(forest),
            levels=forest_levels(forest), cells=nleaves(forest) * N^D,
            λ=λ, λ_end=λ_end, w=w, forest=forest, U=hostcopy(U), u=u)
end

"""
    assert_no_arrival(w::SedovBlast, forest, t_end, E₀; margin = 6//5)

Throw an `ArgumentError` naming the numbers unless `margin · r_s(t_end)` is
strictly less than the distance from the origin to the nearest physical face.

**The case owns this check and the driver cannot make it** — settled in step 7
and unchanged here. The driver knows neither where the feature is nor the
supremum of `λ` over all time; this case knows both, and knows them in closed
form, which is more than the shock tube did. Sod's version bounds the travel by
the *characteristic* speed because a tube has three waves to keep track of; the
blast has one feature and the similarity law says exactly where it is, so the
bound is the law's own radius with a margin on top.

`margin = 6//5` covers two things and is not a tolerance chosen to pass. The
law is the *asymptotic* solution, and at the finite `r_s/r₀` a test can afford
the real shock leads it slightly; and the discrete front sits up to half a cell
outside the exact one. Twenty percent is comfortably more than either, and the
remedy if it fires is to shorten `t_end` or enlarge the box — not to lower it.

`E₀` is the measured energy from [`measured_E₀`](@ref) where one is available,
and the nominal one before the first run exists.
"""
function assert_no_arrival(w::SedovBlast{T,D}, forest, t_end, E₀;
                           margin=6 // 5) where {T,D}
    sim = sedov_similarity(w)
    reach = Float64(margin) *
            sedov_radius(sim, tofloat64(T(t_end)), tofloat64(T(E₀)),
                         tofloat64(w.ρ₀))
    distance = minimum(ntuple(D) do d
                           lo, hi = forest.extents[d]
                           min(-tofloat64(T(lo)), tofloat64(T(hi)))
                       end)
    reach < distance || throw(ArgumentError(
        "the blast reaches the Dirichlet boundary before t_end: the " *
        "similarity law puts the shock at " *
        "$(sedov_radius(sim, tofloat64(T(t_end)), tofloat64(T(E₀)), tofloat64(w.ρ₀))) " *
        "at t_end = $t_end with E₀ = $E₀, which with the $(Float64(margin))× " *
        "margin reaches $reach, and the nearest physical face is only " *
        "$distance from the origin. A boundary set to the initial state is " *
        "exact until a wave arrives and reflects afterwards, so the run would " *
        "measure the reflection. Shorten t_end or enlarge the box."))
    return nothing
end
