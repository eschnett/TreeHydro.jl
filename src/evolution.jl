# The right-hand side: the three field sets, the three kernels that write
# them, and the six steps that call the kernels in order.
#
# This is the first file that touches the mesh. Everything before it is
# pointwise arithmetic on tuples — an equation of state, two conversions,
# two floor rules, a reconstruction and three Riemann solvers — and
# everything here is about *where* those numbers live and in what order
# they are computed. TreeAMR's `test/burgers.jl` is the worked example and
# this is that file with two steps added: a `con2prim` pass over every
# stored cell and the primitive field set it writes. See "The right-hand
# side" and "Field sets" in `CODE.md`.
#
# The three ghost widths are the thing to keep straight, and they are why
# every kernel below takes its `G`s as `Val`s:
#
#   * `U`, conserved, cell-centered, `G = 2` — the only evolved set, so the
#     state vector is `statevector(U)`;
#   * `P`, primitive, cell-centered, `G = 2` — the same layout as `U`, so
#     that one stored index serves both;
#   * `F_d`, `d = 1 … D`, `facecentered(D, d)`, `G = 0` — computed over the
#     closed range and never exchanged.
#
# Face `i` of a block (in `1 … N+1`) lies between cells `i−1` and `i`; cell
# `i` is stored at `i + G_P` in `P` and the face at `i + G_F` in `F_d`.
# Under `stored = true` — the `con2prim` launch, and nothing else here —
# the kernel's index *is* the stored index and adds nothing.

"""
    HydroProblem(U, ops; eos, floors, limiter, riemann = :hlle, fixup = true,
                 boundary = nothing, prims = nothing, fluxes = nothing)

Everything a hydrodynamic right-hand side needs, built once per mesh: the
conserved state and its ghost schedule, the primitive set the
reconstruction reads, the `D` flux sets and their interface schedules, the
per-block spacings on whatever backend the data lives on, the equation of
state and the floors, and — as `Val`s, so that the kernels specialize on
them and no branch survives into the innermost loop — the dimension, the
three ghost widths, the limiter and the Riemann solver.

`U` is the only evolved set, so it is the one argument: `P` and the fluxes
are scratch, rewritten at every evaluation, and this builds them unless
they are handed in. After a [`regrid!`](@ref) the problem is rebuilt
rather than mutated, since both kinds of schedule and the spacings are
derived from the leaf array; pass the existing `prims` and `fluxes` when
doing so, because `regrid!` resized them in place (`fs => nothing`) and
reallocating them would throw that away.

**`limiter` has no default.** The cases choose differently and the choice
changes what is being measured: `:none` — the plain centered slope — is
what a convergence study wants, since a limiter clips at smooth extrema
and would hide the scheme's own order behind its footprint, while
`:minmod` or `:mc` is what a shock wants. `riemann` defaults to `:hlle`,
which is the decision recorded in "Riemann solver" in `CODE.md`: `:hllc`
is the comparison, measured on the Kelvin–Helmholtz instability, not the
baseline.

`fixup = false` skips [`restrict_interfaces!`](@ref) and nothing else —
the negative control for the conservation claim. `boundary` is the hook
[`fill_ghosts!`](@ref) calls on ghost regions facing outside a
non-periodic domain, or `nothing` for a fully periodic one.

The layout requirements are checked here rather than discovered later: `U`
cell-centered with `G ≥ 2` in every dimension (the reconstruction reads
cells `i−2 … i+1`, and both sides of a same-level face must compute the
flux from the same four values — see "Conservation at coarse-fine faces"
in `CODE.md`), `P` with the same forest, ghost width and centering as `U`
so that one stored index serves both, and fluxes with `G = 0`.
"""
struct HydroProblem{T,D,GU,GP,GF,LIM,RS,FU,FP,FL,SC,IS,SP,EOS,FLR,BC}
    U::FU                        # conserved, cell-centered, G = 2
    P::FP                        # primitive + 2 diagnostic slots, same layout
    fluxes::FL                   # NTuple{D,FieldSet}, facecentered(D, d), G = 0
    schedule::SC
    ischeds::IS                  # NTuple{D,InterfaceSchedule}
    spacings::SP                 # per block, wherever the kernels run
    eos::EOS
    floors::FLR
    boundary::BC
    fixup::Bool
    valD::Val{D}
    valGU::Val{GU}
    valGP::Val{GP}
    valGF::Val{GF}
    limiter::Val{LIM}
    solver::Val{RS}
end

# The three admissible values of each, in the order `CODE.md` introduces
# them. Named here so that the constructor's message can list them and the
# list cannot drift from the methods that exist.
const HYDRO_LIMITERS = (:none, :minmod, :mc)
const HYDRO_SOLVERS = (:llf, :hlle, :hllc)

function HydroProblem(U::FieldSet{T,D}, ops::Operators; eos::EquationOfState,
                      floors::Floors, limiter::Union{Symbol,Nothing}=nothing,
                      riemann::Symbol=:hlle, fixup::Bool=true, boundary=nothing,
                      prims=nothing, fluxes=nothing) where {T,D}
    forest = U.forest
    backend = get_backend(U.work)

    limiter === nothing && throw(ArgumentError(
        "HydroProblem has no default limiter: pass `limiter` explicitly, one " *
        "of $(HYDRO_LIMITERS). The choice is the case's, not the scheme's — " *
        ":none is the plain centered slope, which keeps the truncation error " *
        "a clean O(h²) and is what a convergence study measures, while a " *
        "limited slope is what keeps a shock monotone and clips at a smooth " *
        "extremum while doing it."))
    limiter in HYDRO_LIMITERS || throw(ArgumentError(
        "limiter must be one of $(HYDRO_LIMITERS), got :$limiter: the limiter " *
        "travels into the flux kernel as a Val and `slope` has a method for " *
        "each of those three and no others, so an unknown symbol would be a " *
        "MethodError inside a kernel launch rather than here."))
    riemann in HYDRO_SOLVERS || throw(ArgumentError(
        "riemann must be one of $(HYDRO_SOLVERS), got :$riemann: the solver " *
        "travels into the flux kernel as a Val and `riemann_flux` has a " *
        "method for each of those three and no others."))

    all(==(:cell), U.centering) || throw(ArgumentError(
        "the conserved state is cell-centered; got $(U.centering). A finite " *
        "volume scheme evolves cell averages, and the staggered sets here are " *
        "the fluxes, which this builds itself."))
    all(>=(2), U.G) || throw(ArgumentError(
        "the reconstruction reads cells i−2 … i+1, so the conserved state " *
        "needs G >= 2 in every dimension for the two sides of a same-level " *
        "face to compute the same flux from the same four values; got " *
        "G=$(U.G). It is also what the Conservative family's p = 3 " *
        "prolongation needs."))
    U.nvars == D + 2 || throw(ArgumentError(
        "the conserved state holds (ρ, S₁…S_D, E), which is $(D + 2) " *
        "variables in $D dimensions; got nvars=$(U.nvars)."))

    P = prims === nothing ?
        FieldSet{T}(forest, D + 4; G=U.G, centering=U.centering, backend=backend) :
        prims
    P.forest === forest || throw(ArgumentError(
        "the primitive set must be over the same forest as the conserved " *
        "state: block indices come from the leaf array, and two forests with " *
        "the same leaves today would still be a different mesh tomorrow."))
    (P.G == U.G && P.centering == U.centering) || throw(ArgumentError(
        "the primitive set must have the conserved state's layout, G=$(U.G) " *
        "and centering $(U.centering); got G=$(P.G) and $(P.centering). The " *
        "con2prim pass runs over every stored cell of both and uses one index " *
        "for both, so a different ghost width would read the wrong cell."))
    P.nvars == D + 4 || throw(ArgumentError(
        "the primitive set holds the $(D + 2) primitives (ρ, v₁…v_D, p) and " *
        "two diagnostic slots — the cell's signal speed and its floor-hit " *
        "flag, both written by the con2prim kernel because block_mapreduce " *
        "maps a scalar function over one variable and cannot form |v| + c_s " *
        "from three; got nvars=$(P.nvars), expected $(D + 4)."))

    fluxes = fluxes === nothing ?
             ntuple(d -> FieldSet{T}(forest, D + 2; G=0,
                                     centering=facecentered(D, d), backend=backend),
                    D) : fluxes
    all(f -> all(==(0), f.G), fluxes) || throw(ArgumentError(
        "a flux set carries no ghosts: it is computed over the closed range " *
        "and never exchanged, and the interface fixup reads and writes " *
        "closed-range values only; got G=$(map(f -> f.G, fluxes))."))

    schedule = GhostSchedule(U, ops)
    ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)
    spacings = to_backend(backend, block_spacings(forest, T))
    GU, GP, GF = U.G, P.G, first(fluxes).G
    return HydroProblem{T,D,GU,GP,GF,limiter,riemann,typeof(U),typeof(P),
                        typeof(fluxes),typeof(schedule),typeof(ischeds),
                        typeof(spacings),typeof(eos),typeof(floors),
                        typeof(boundary)}(
        U, P, fluxes, schedule, ischeds, spacings, eos, floors, boundary, fixup,
        Val(D), Val(GU), Val(GP), Val(GF), Val(limiter), Val(riemann))
end

# --- the three kernels ----------------------------------------------------

# Step (2): the primitives, in **every stored cell** of every block, ghosts
# included. That is the whole point of the pass — the reconstruction reads
# primitives two cells into the neighbours, so a recovery over the owned
# range alone would leave the flux at a block's own boundary face reading
# zeros. The launch is `map_blocks!(…; stored = true)` and so the global
# index *is* the stored index: nothing is added to it anywhere below, and a
# kernel written for the owned range would be silently wrong here, since
# both are in bounds.
#
# It also writes the two diagnostic slots, because it is the only pass that
# holds all `D + 2` primitives of a cell at once: slot `D+3` the signal
# speed `max_d(|v_d| + c_s)`, which sets the time step, and slot `D+4` the
# floor-hit flag as a `1` or a `0`. `block_mapreduce` maps a scalar
# function over *one* variable's values, so it could form neither from the
# primitives themselves; with the numbers written down, `λ_max` and the
# floor count are plain reductions over one slot. See "Sharp edges" in
# `PLAN.md` and "Field sets" in `CODE.md`.
@kernel function con2prim_kernel!(prim, @Const(cons), eos, floors,
                                  ::Val{D}) where {D}
    I = @index(Global, NTuple)                     # already a stored index
    b = I[D + 1]
    c = ntuple(d -> I[d], Val(D))
    Ucell = ntuple(v -> cons[c..., v, b], Val(D + 2))
    Pcell, hit = con2prim(eos, floors, Ucell)
    for v in 1:(D + 2)
        prim[c..., v, b] = Pcell[v]
    end
    prim[c..., D + 3, b] = signal_speed(eos, Pcell)
    # `one`/`zero` of a value rather than of a captured type: a `Type` in a
    # kernel closure is the leak "Running on a device" in `CODE.md` warns
    # about.
    prim[c..., D + 4, b] = hit ? one(first(Pcell)) : zero(first(Pcell))
end

# Step (3): reconstruction and Riemann solve fused into one kernel per
# direction, launched over the flux set's **closed** range, so `I[d]` runs
# over `1 … N+1` and the transverse indices over `1 … N` — a block computes
# the flux on both of its own faces, and making the two sides of a
# coarse-fine face agree afterwards is `restrict_interfaces!`'s job.
#
# Face `I[d]` lies between cells `I[d]−1` and `I[d]`; cell `i` of `P` is
# stored at `i + GP[e]` per dimension, the face at `I[e] + GF[e]`.
# Transversally a face index *is* a cell index, in both field sets. This is
# the Burgers flux kernel's index arithmetic unchanged; what differs is
# that each of the four stencil points is a `D + 2`-tuple rather than one
# number.
@kernel function flux_kernel!(flux, @Const(prim), eos, floors, ::Val{D}, ::Val{GP},
                              ::Val{GF}, ::Val{d}, lim::Val,
                              solver::Val) where {D,GP,GF,d}
    I = @index(Global, NTuple)                     # (i1..iD, block)
    b = I[D + 1]
    c = ntuple(e -> I[e] + GP[e], Val(D))
    m1 = Base.setindex(c, c[d] - 1, d)
    m2 = Base.setindex(c, c[d] - 2, d)
    p1 = Base.setindex(c, c[d] + 1, d)

    P₋₂ = ntuple(v -> prim[m2..., v, b], Val(D + 2))
    P₋₁ = ntuple(v -> prim[m1..., v, b], Val(D + 2))
    P₀ = ntuple(v -> prim[c..., v, b], Val(D + 2))
    P₊₁ = ntuple(v -> prim[p1..., v, b], Val(D + 2))

    P_L, P_R = face_states(lim, eos, floors, P₋₂, P₋₁, P₀, P₊₁)
    F = riemann_flux(solver, eos, P_L, P_R, Val(d))

    f = ntuple(e -> I[e] + GF[e], Val(D))
    for v in 1:(D + 2)
        flux[f..., v, b] = F[v]
    end
end

# Step (5): `du = −Σ_d (F_d[i+1] − F_d[i]) / h` for every one of the
# `D + 2` conserved variables, over the state's owned cells and written
# straight into the state layout.
#
# `fluxes` is an `NTuple{D}` of identically typed arrays, so indexing it
# with the loop variable is type stable; all `D` flux sets share one `GF`,
# which is what lets one stored index serve every direction.
@kernel function divergence_kernel!(du, fluxes, @Const(spacings), ::Val{D},
                                    ::Val{GF}) where {D,GF}
    I = @index(Global, NTuple)                     # (i1..iD, block)
    b = I[D + 1]
    c = ntuple(e -> I[e] + GF[e], Val(D))
    o = ntuple(e -> I[e], Val(D))
    h = spacings[b]
    for v in 1:(D + 2)
        acc = zero(eltype(du))
        for d in 1:D
            hi = Base.setindex(c, c[d] + 1, d)
            acc += fluxes[d][hi..., v, b] - fluxes[d][c..., v, b]
        end
        # A source term would enter here, as `du = −ΔF/h + S_v(P)`: gravity,
        # a geometric source in curvilinear coordinates, the GRMHD
        # connection terms. The Euler system on a Cartesian mesh has none,
        # so the slot is empty and named rather than absent — it is the one
        # line that would change (see "The right-hand side" in `CODE.md`).
        du[o..., v, b] = -acc / h
    end
end

# --- the right-hand side --------------------------------------------------

"""
    hydro_rhs!(du, u, p::HydroProblem, t)

The six-step conservative right-hand side, written out by the application
as "The right-hand side" in `CODE.md` sets it out:

    scatter!(U, u)                          # (0) state → working array
    fill_ghosts!(U, schedule; boundary)     # (1) conserved ghosts
    con2prim over every stored cell         # (2) primitives, ghosts included
    for d in 1:D                            # (3) reconstruct + Riemann
        restrict_interfaces!(F_d, …)        # (4) the fixup, unless fixup = false
    end
    du = −Σ_d ΔF_d / h                      # (5) the divergence

There is no `semidiscretize`-style wrapper, here as everywhere: the mesh
supplies the launches and the exchange, the application supplies the
physics and the order.

**Pure in `(u, t)`, and it never mutates `u`.** `P` and the flux sets are
scratch, fully rewritten at every evaluation, and nothing reads them from a
previous one; the atmosphere reset that does touch the state acts on the
*stage* vector from inside the integrator's limiter hook, which is where
the method defines that to happen (see "Floors and the atmosphere" in
`CODE.md`).

The `ntuple` over `Val(D)` is not decoration: it unrolls the direction loop
so that each launch gets a *constant* `Val(d)`, which is what the flux
kernel specializes its face dimension on.
"""
function hydro_rhs!(du, u, p::HydroProblem{T,D}, t) where {T,D}
    scatter!(p.U, u)
    fill_ghosts!(p.U, p.schedule; boundary=p.boundary)
    map_blocks!(con2prim_kernel!, p.P, p.P.work, p.U.work, p.eos, p.floors, p.valD;
                stored=true)
    ntuple(Val(D)) do d
        map_blocks!(flux_kernel!, p.fluxes[d], p.fluxes[d].work, p.P.work, p.eos,
                    p.floors, p.valD, p.valGP, p.valGF, Val(d), p.limiter, p.solver;
                    closed=true)
        p.fixup && restrict_interfaces!(p.fluxes[d], p.ischeds[d])
        nothing
    end
    map_blocks!(divergence_kernel!, p.U, statearray(du, p.U),
                map(f -> f.work, p.fluxes), p.spacings, p.valD, p.valGF)
    return nothing
end

"""
    update_primitives!(p::HydroProblem, u)

Steps (0) to (2) of [`hydro_rhs!`](@ref) and no further: scatter `u` into
the conserved set, fill its ghosts, and recover the primitives in every
stored cell. Returns the primitive field set.

A driver needs this outside a right-hand-side evaluation — before the
first step, to measure `λ_max` for the time step; after a chunk, to feed
the refinement criterion, which reads `ρ` and `p`; after a regrid, before
anything asks `P` a question. Computing the fluxes as well would be the
expensive two thirds of an evaluation thrown away.

`P` is scratch, so this leaves the problem in exactly the state a
subsequent [`hydro_rhs!`](@ref) would overwrite anyway.
"""
function update_primitives!(p::HydroProblem{T,D}, u) where {T,D}
    scatter!(p.U, u)
    fill_ghosts!(p.U, p.schedule; boundary=p.boundary)
    map_blocks!(con2prim_kernel!, p.P, p.P.work, p.U.work, p.eos, p.floors, p.valD;
                stored=true)
    return p.P
end

"""
    max_signal_speed(P::FieldSet)
    max_signal_speed(p::HydroProblem)

`λ_max = max over owned cells of max_d (|v_d| + c_s)`, the fastest signal
anywhere on the mesh — the number the global time step is built from (see
[`hydro_dt`](@ref) and "Time integration and the time step" in `CODE.md`).

**Two entry points, one implementation** (the split arrived in step 7, as
[`hydro_flags`](@ref)'s did and for the same reason): the driver's
initial-data cycle has no [`HydroProblem`](@ref) to hold a primitive set,
because the forest is still changing under it, and measures the speed on a
scratch set instead.

**`P` must be current**: this reads diagnostic slot `D + 3`, which the
`con2prim` kernel wrote at the last [`hydro_rhs!`](@ref) or
[`update_primitives!`](@ref) call. It is a plain
[`block_mapreduce`](@ref) over that one slot precisely because the slot
exists — a reduction cannot form `|v| + c_s` from three variables, so the
kernel that held all three wrote the number down.

The per-block maxima are combined in block order, so the answer does not
depend on the thread count.
"""
max_signal_speed(p::HydroProblem) = max_signal_speed(p.P)

function max_signal_speed(P::FieldSet{T,D}) where {T,D}
    R = float(real(T))
    P.nvars == D + 4 || throw(ArgumentError(
        "the signal speed lives in diagnostic slot $(D + 3) of the primitive " *
        "set, which the con2prim kernel writes; got nvars=$(P.nvars) where " *
        "$(D + 4) was expected."))
    return maximum(block_mapreduce(identity, max, zero(R), P; vars=D + 3))
end

"""
    floor_hits(p::HydroProblem)

How many **owned** cells had a floor fire in them when the primitives were
last recovered, as an `Int`: the sum of diagnostic slot `D + 4`, which the
`con2prim` kernel wrote as a `1` or a `0` per cell.

The count is a measurement the design depends on and not a diagnostic (see
"Floors and the atmosphere" in `CODE.md`): where it is zero the
conservation claim is the plain one, and where it is not, the drift is
claimed net of a measured injection. This is the *owned* population;
the ghost-cell count — the one that decides whether to ask upstream for a
limited, positivity-preserving prolongation — needs the stored extent,
which `block_mapreduce` does not offer, and arrives with the atmosphere
reset.

Requires `P` to be current, as [`max_signal_speed`](@ref) does.
"""
function floor_hits(p::HydroProblem{T,D}) where {T,D}
    R = float(real(T))
    return round(Int, sum(block_mapreduce(identity, +, zero(R), p.P; vars=D + 4)))
end

"""
    hydro_dt(forest, cfl, λ_max, ::Val{D})

The global time step, `cfl · h_min / (D · λ_max)`.

One step for the whole hierarchy, from the finest spacing and the fastest
signal: TreeAMR has no subcycling, ever. The `D` is the sum over
directions of an unsplit scheme's CFL condition, bounded above by
`D · λ_max` — slightly conservative, and simpler than a per-cell sum. See
"Time integration and the time step" in `CODE.md`, which is also where the
driver's obligation to re-measure `λ_max` at the end of each chunk is set
out.
"""
hydro_dt(forest, cfl::T, λ::T, ::Val{D}) where {T,D} =
    cfl * minimum_spacing(T, forest) / (D * λ)

"""
    conserved_totals(U::FieldSet)

The domain integral `Σ hᴰ U_v` of each of the `D + 2` conserved variables,
as a tuple.

The quantity the conservation claim is about, and there are `D + 2` of
them here where Burgers had one: total mass, each momentum component and
the total energy, each with its own integral and its own scale (see
[`conserved_scales`](@ref) and "Conservation at coarse-fine faces" in
`CODE.md`). Per block through [`total_mass`](@ref) and summed in block
order, so the value does not move with the thread count.
"""
conserved_totals(U::FieldSet{T,D}) where {T,D} =
    ntuple(v -> total_mass(U, v), Val(D + 2))

"""
    conserved_scales(U::FieldSet)

`Σ hᴰ |U_v|` for each of the `D + 2` conserved variables, as a tuple — the
scale each drift in [`conserved_totals`](@ref) is roundoff against.

The absolute value matters: a momentum component whose total is zero by
symmetry — the Kelvin–Helmholtz `S_y`, every Sedov `S_d` — would otherwise
be compared against nothing at all, and the mass integral itself can be
small through cancellation for a field that changes sign.
"""
function conserved_scales(U::FieldSet{T,D}) where {T,D}
    R = float(real(T))
    forest = U.forest
    return ntuple(Val(D + 2)) do v
        partials = block_mapreduce(abs, +, zero(R), U; vars=v)
        for b in 1:nblocks(U)
            partials[b] *= spacing(R, forest, blockkey(U, b))^D
        end
        sum(partials)
    end
end

"""
    hydro_solve!(p::HydroProblem, u, t0, t1, nsteps)

One fixed-step `SSPRK33` solve of `nsteps` steps from `t0` to `t1`,
returning the final state vector.

Strong-stability-preserving rather than plain Runge–Kutta because a
limited scheme's shocks stay monotone only under one: conservation holds
for *any* Runge–Kutta method, since every stage's `du` already sums to
zero. Fixed step because `λ_max` is measured once per chunk and a
step-adaptive `dt` would be a callback fighting a fixed-step `solve`; the
driver's CFL recheck at the end of a chunk is what makes that safe. Its
`stage_limiter!` hook is where the atmosphere reset will act. See "Time
integration and the time step" in `CODE.md`.
"""
function hydro_solve!(p::HydroProblem{T}, u, t0::T, t1::T, nsteps::Int) where {T}
    prob = ODEProblem(hydro_rhs!, u, (t0, t1), p)
    sol = solve(prob, SSPRK33(); dt=(t1 - t0) / nsteps, adaptive=false,
                save_everystep=false)
    return sol.u[end]
end

"""
    forest_levels(forest) -> sorted Vector{Int}

The refinement levels the forest's leaves actually occupy, sorted and
without duplicates.

A one-line mesh query, here because every conservation claim this package
makes across a coarse-fine face rests on the mesh *having* one, and
`nblocks > roots^D` does not say that on its own — a forest could have been
refined and coarsened back. `[0, 1]` is what the static two-level
hierarchies of [`hydro_forest`](@ref) and [`sod_forest`](@ref) must report,
and a run that quietly produced `[0]` would pass every drift assertion for
the wrong reason (see "Conservation at coarse-fine faces" in `CODE.md`).
"""
forest_levels(forest) = sort(unique(level.(forest.leaves)))

"""
    convergence_rate(hs, errs)

Least-squares slope of `log(err)` against `log(h)` — the measured order of
a convergence study, from more than two resolutions at once so that one
noisy point does not become the answer.

TreeWave's and TreeAMR's, copied rather than depended on, as
`precision.jl` and `device.jl` are.
"""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
