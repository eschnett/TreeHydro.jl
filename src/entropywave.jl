# The entropy wave: the smooth case, and the one that pins the order.
#
#     ρ = ρ₀ + a sin(2π (Σ_d x_d − Σ_d v_d t) / L),   v = const,   p = const
#
# An exact solution of the *nonlinear* Euler equations in any `D`: with a
# uniform velocity and a uniform pressure the momentum and energy equations
# reduce to advection of `ρ` at the velocity `v`, and nothing steepens.
# It is the system's counterpart of the Burgers sine — smooth, exact,
# periodic, and dependent on `Σ_d x_d` so that every flux direction does
# real work — and the slope of its error against `h` is 2 or the scheme is
# not second order. See "Entropy wave" in `CODE.md`.
#
# It is also a *contact*: the density jumps across it and the pressure does
# not, which is exactly the wave an HLL-family flux diffuses most. The rate
# is unaffected and the constant is not, which is why the HLLE/HLLC
# comparison has its first number here.
#
# Everything below fills **exact cell averages**, on the host, in one loop
# per field set with one `copyto!` at the end. That is what a finite-volume
# scheme's stored numbers mean, and here the closed form exists: the
# average of `sin(k Σ_d x_d)` over a cube of side `h` is
# `((2/(kh)) sin(kh/2))^D · sin(k Σ_d x_d)`. The damping factor depends on
# the *block's* own `h`, which a coordinate callback cannot see — the same
# reason `fill_burgers_averages!` is a host loop. It is not the way the
# other cases initialize: they are discontinuous, where a cell average is
# no better defined than a point sample, and their meshes change under
# them.

"""
    EntropyWave(T, Val(D); ρ₀ = 1, a = 1//5, v = 1, p₀ = 1, γ = 7//5, L = 1,
                floors = …)

The parameters of the entropy wave in `D` dimensions at working type `T`:
the mean density, the amplitude, the `D` velocity components, the
pressure, the box side, the equation of state and the floors.

`v` is either one number, used for every component, or a `D`-tuple. The
defaults are the ones the convergence study is run at: an amplitude of a
fifth of the mean, so the wave is well resolved and nowhere near vacuum;
`v_d = 1` in every direction, so that every flux direction carries
something and no component of the momentum is zero by accident; and a
diatomic `γ = 7/5`, matching Sod's gas.

Every default is a rational converted to `T`, never a decimal literal: at
`Float64` the two are bit-identical, and that is what lets a measured
number stay put when the study is run at another type (see "Precision" in
`CODE.md`).

**The floors default eight orders of magnitude below the data** and are
expected never to fire — `entropywave_errors` reports the count, and the
tests assert it is zero. A floor firing on a smooth wave whose density
never leaves `[4/5, 6/5]` would mean something is wrong upstream of the
floors, not that the floors are doing their job.

`isbits`, like everything a kernel argument may hold.
"""
struct EntropyWave{T,D,E<:EquationOfState}
    ρ₀::T
    a::T
    v::NTuple{D,T}
    p₀::T
    L::T
    eos::E
    floors::Floors{T}
end

function EntropyWave(::Type{T}, ::Val{D}; ρ₀=1, a=1 // 5, v=1, p₀=1, γ=7 // 5,
                     L=1,
                     floors::Floors{T}=Floors{T}(; ρ_atm=T(1 // 10^8),
                                                 p_atm=T(1 // 10^8),
                                                 p_floor=T(1 // 10^8))) where {T,D}
    vs = v isa Tuple ? ntuple(d -> T(v[d]), Val(D)) : ntuple(_ -> T(v), Val(D))
    length(vs) == D || throw(ArgumentError(
        "the entropy wave needs one velocity component per dimension: got " *
        "$(length(vs)) for D = $D."))
    T(ρ₀) > T(a) ≥ 0 || throw(ArgumentError(
        "the entropy wave needs 0 ≤ a < ρ₀, got a = $a and ρ₀ = $ρ₀: the " *
        "density ρ₀ + a sin(…) is a smooth exact solution only while it stays " *
        "positive, and an amplitude at or above the mean puts a vacuum — and " *
        "then the floors — into what is meant to be the clean case."))
    return EntropyWave{T,D,IdealGas{T}}(T(ρ₀), T(a), vs, T(p₀), T(L),
                                        IdealGas(T(γ)), floors)
end

"""The wave number `k = 2π/L`."""
@inline wavenumber(w::EntropyWave{T}) where {T} = 2 * T(π) / w.L

"""
The factor by which averaging `sin(k Σ_d x_d)` over a cube of side `h`
damps it: `((2/(kh)) sin(kh/2))^D`, one factor per dimension, exactly.

`1 − (kh)²/24` to leading order, so it is an `O(h²)` correction — the same
order as the error the study measures, which is why the reference is the
average and not the point value.
"""
@inline function average_damping(w::EntropyWave{T,D}, h::T) where {T,D}
    k = wavenumber(w)
    return (2 * sin(k * h / 2) / (k * h))^D
end

"""
    entropywave_state(w, damp, x, t) -> U

The **conserved** cell average at position `x` and time `t`, given the
damping factor of that cell's size.

`v` and `p` are constant in space and time, so the averages of
`S_d = ρ v_d` and of `E = p/(γ−1) + ½ ρ v²` are exact in terms of the
averaged density and no quadrature enters anywhere: one analytic
expression serves the initial data and the reference at any later time,
with `x_d → x_d − v_d t` the whole of the time dependence.
"""
@inline function entropywave_state(w::EntropyWave{T,D}, damp::T, x, t::T) where {T,D}
    k = wavenumber(w)
    s = sum(ntuple(d -> x[d] - w.v[d] * t, Val(D)))
    ρ = w.ρ₀ + w.a * damp * sin(k * s)
    E = w.p₀ / (w.eos.γ - 1) + ρ * squarednorm(w.v) / 2
    return (ρ, ntuple(d -> ρ * w.v[d], Val(D))..., E)
end

"""
    hydro_forest(Val(D), N; roots = 4, L = 1, refined = true, T = Float64)

TreeAMR's M3 two-level hierarchy, as `burgers_forest` and `wave_forest`
build it and for the same reason: a `roots^D` periodic box of side `L`
with the middle sub-box refined once and the result 2:1 balanced, held
fixed in physical space as `N` varies, so that a convergence study really
does just shrink `h`.

`refined = false` leaves the box uniform. That is the **control**, and it
is what steps 3 and 4 run on: on a single-level mesh every face is a
same-level face, so the conserved integrals are constant to roundoff with
the interface fixup and without it, and any difference between the two
would be a bug in the fixup rather than a property of the mesh. The
refined path is built here and **measured in step 5**: on the two-level
mesh the conserved integrals hold to roundoff with the fixup and leak by
ten orders of magnitude without it, and the prolongation order decides the
L∞ rate (see "Measured results" in `CODE.md`).
"""
function hydro_forest(::Val{D}, N; roots=4, L=1, refined=true,
                      T::Type=Float64) where {D}
    L = T(L)
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), L), D))
    refined || return forest
    quarter, threequarters = L / 4, 3 * L / 4
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> quarter < (ext[d][1] + ext[d][2]) / 2 < threequarters, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

"""
    fill_entropywave_averages!(U, w::EntropyWave)

Fill every owned cell of the conserved set with the **exact cell average**
of the initial state — what a finite-volume scheme's stored numbers mean,
and what makes an `O(h²)` measurement about the scheme rather than about
the initial data.

A host loop and one `copyto!` rather than `fill_by_coordinates!`: the
damping factor depends on the block's own `h`, which a coordinate callback
cannot see. The other cases, whose meshes change under them and whose data
is discontinuous, initialize from point samples instead.
"""
function fill_entropywave_averages!(U::FieldSet{T,D},
                                    w::EntropyWave{T,D}) where {T,D}
    forest = U.forest
    host = zeros(T, size(U.work))
    for b in 1:nblocks(U)
        damp = average_damping(w, spacing(T, forest, blockkey(U, b)))
        for idx in CartesianIndices(ntuple(d -> (U.G[d] + 1):(U.G[d] + forest.N), D))
            x = coordinates(T, U, b, Tuple(idx))
            Ucell = entropywave_state(w, damp, x, zero(T))
            for v in 1:(D + 2)
                host[Tuple(idx)..., v, b] = Ucell[v]
            end
        end
    end
    copyto!(U.work, host)
    return U
end

"""
    entropywave_reference(U, w::EntropyWave, t) -> state vector

The exact solution at time `t` as cell averages, in the state vector's
layout — what the evolved state is judged against.

The same closed-form averages as [`fill_entropywave_averages!`](@ref) with
`x_d → x_d − v_d t`, because the wave is an exact solution for all time
and the average of an advected profile is the advected average. No
quadrature and no Newton iteration: unlike Burgers' sine, this solution
does not become implicit after `t = 0`.
"""
function entropywave_reference(U::FieldSet{T,D}, w::EntropyWave{T,D}, t) where {T,D}
    forest = U.forest
    N = forest.N
    t = T(t)
    host = zeros(T, statelength(U))
    arr = reshape(host, ntuple(_ -> N, D)..., U.nvars, nblocks(U))
    for b in 1:nblocks(U)
        damp = average_damping(w, spacing(T, forest, blockkey(U, b)))
        for idx in CartesianIndices(ntuple(_ -> N, D))
            x = coordinates(T, U, b, ntuple(d -> Tuple(idx)[d] + U.G[d], D))
            Ucell = entropywave_state(w, damp, x, t)
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
    entropywave_errors([T = Float64], Val(D); N, ops, …)

Run the entropy wave to `t_end` and return what the convergence and
conservation claims are made of: the volume-weighted `l1` and `linf`
errors of the whole state vector against the exact averages, the
per-variable `drift` of the `D + 2` conserved integrals and the `scales`
they are roundoff against, the owned-cell `floor_hits`, the finest
spacing `h`, the step count, the block count and the `levels` the mesh
actually occupies.

Keywords: `N` cells per block and `ops` the operator family are required;
`G = 2`, `roots = 4`, `limiter = :none`, `riemann = :hlle`, `fixup = true`,
`refined = false`, `cfl = 2//5`, `t_end = 1//4`, `backend = CPU()`, and
anything else is passed to [`EntropyWave`](@ref).

`limiter = :none` is the default *here* and nowhere else: this is the
convergence study, and a limiter clips at the smooth extrema of a sine and
would measure its own footprint instead of the scheme's order. `refined =
false` is the uniform control; `refined = true` is the two-level mesh
measured in step 5, where the fixup is the difference between conservation
and a leak and the prolongation order is the difference between second
order and first in L∞.

The time step is `cfl · h_min / (D λ_max)` with `λ_max` measured from the
initial data through [`max_signal_speed`](@ref), which is exact for all
time here — the velocity and the sound speed are constant along the
characteristic and the density's range does not grow.
"""
entropywave_errors(valD::Val; kwargs...) =
    entropywave_errors(Float64, valD; kwargs...)

function entropywave_errors(::Type{T}, ::Val{D}; N, ops, G=2, roots=4,
                            limiter=:none, riemann=:hlle, fixup=true,
                            refined=false, cfl=2 // 5, t_end=1 // 4,
                            backend=CPU(), params...) where {T,D}
    w = EntropyWave(T, Val(D); params...)
    forest = hydro_forest(Val(D), N; roots=roots, L=w.L, refined=refined, T=T)
    U = FieldSet{T}(forest, D + 2; G=G, backend=backend)
    p = HydroProblem(U, ops; eos=w.eos, floors=w.floors, limiter=limiter,
                     riemann=riemann, fixup=fixup)

    fill_entropywave_averages!(U, w)
    u = statevector(U)
    gather!(u, U)
    totals0 = conserved_totals(U)
    scales = conserved_scales(U)

    # `P` current before the run, for `λ_max`; and again after it, so that
    # the floor count describes the state the errors are measured on.
    update_primitives!(p, u)
    t_end = T(t_end)
    dt = hydro_dt(forest, T(cfl), max_signal_speed(p), Val(D))
    nsteps = ceil(Int, t_end / dt)
    u = hydro_solve!(p, u, zero(T), t_end, nsteps)
    update_primitives!(p, u)

    err = u .- entropywave_reference(U, w, t_end)
    totals1 = conserved_totals(U)
    return (l1=volume_weighted_norm(U, err; p=1),
            linf=volume_weighted_norm(U, err; p=Inf),
            drift=ntuple(v -> abs(totals1[v] - totals0[v]), Val(D + 2)),
            scales=scales, floor_hits=floor_hits(p),
            h=minimum_spacing(T, forest), nsteps=nsteps, nblocks=nleaves(forest),
            levels=forest_levels(forest))
end
