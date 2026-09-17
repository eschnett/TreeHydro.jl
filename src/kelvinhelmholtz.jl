# The Kelvin–Helmholtz instability: the case that exists for the picture, the
# only one with no exact solution, and the one that puts a *contact* under the
# refinement criterion instead of a shock.
#
# The setup is McNally, Lyra & Passy (2012), ApJS 201:18 (arXiv:1111.1764) —
# the smooth-ramp shear layer designed as a *converged* code-comparison test,
# rather than the classic sharp-interface setup whose small-scale structure
# never converges. On the periodic unit square with `γ = 5/3` and `p = 5/2`,
# their equations (1)–(5):
#
#     ρ = ρ₁ − ρ_m e^{(y−¼)/L}   v_x = v₁ − v_m e^{(y−¼)/L}    ¼ > y ≥ 0
#         ρ₂ + ρ_m e^{(¼−y)/L}         v₂ + v_m e^{(¼−y)/L}    ½ > y ≥ ¼
#         ρ₂ + ρ_m e^{(y−¾)/L}         v₂ + v_m e^{(y−¾)/L}    ¾ > y ≥ ½
#         ρ₁ − ρ_m e^{(¾−y)/L}         v₁ − v_m e^{(¾−y)/L}    1 > y ≥ ¾
#     v_y = 0.01 sin(4πx)
#
# with `ρ_m = (ρ₁−ρ₂)/2`, `v_m = (v₁−v₂)/2`, `ρ₁ = 1`, `ρ₂ = 2`, `v₁ = ½`,
# `v₂ = −½` and `L = 1/40`, run to `t = 1.5`.
#
# **Checked against the paper in step 10.** `CODE.md`'s transcription of the
# profiles, the parameters, the perturbation, `γ`, `p` and `t_end` is the
# paper's in every term. One thing in it was wrong and is corrected here and
# there: the weighting of the mode amplitude `M(t)` is *not* "the lower
# interface alone". The paper's equations (6)–(8) weight by
# `e^{−4π|y − ¼|}` for `y < ½` and by `e^{−4π|(1−y) − ¼|}` for `y ≥ ½`, so
# **both** interfaces are read, mirrored onto one another. On a mesh whose
# cells are not all the same size the sums are area-weighted, equations
# (14)–(17), which is the form [`mode_amplitude`](@ref) uses — an adaptive
# mesh has cells of two sizes by construction, and the uniform form would
# weight a coarse cell as though it were a fine one.
#
# What this case measures that no other does:
#
#   * **Refinement following a structure that grows rather than travels.** Two
#     strips at `t = 0`, then rolls. The pulse and the shell do not test that.
#   * **Conservation under a flow with no shock**, where the leak without the
#     interface fixup is small in absolute terms and the roundoff claim is
#     correspondingly sharper — and, unlike the blast, on the *tracked* mesh
#     itself: the whole domain is in motion here, so a coarse-fine face
#     anywhere in the box carries real flux.
#   * **HLLE against HLLC**, which `CODE.md`'s "Riemann solver" defers to this
#     milestone. The shear layer *is* a contact, which is the wave the two
#     solvers differ in and the only one they differ in.
#
# Three things about this case are easy to get wrong:
#
#   * **The growth rate is a fit over a window, and the window is the claim.**
#     `M(t)` is not a pure exponential for all time — it starts from a seeded
#     perturbation that has to shed its transient and it bends over as the
#     rolls saturate. [`growth_rate`](@ref) fits `log M` against `t` over an
#     explicit range of `M`, and the range is recorded beside the number.
#   * **A `Float32` run diverges from the `Float64` one late, by design.** The
#     instability amplifies roundoff exponentially. The claim at `Float32` is
#     the mesh statistics of the first chunks and `M(t)` through the linear
#     phase, never the final state; `CODE.md` says so and `CLAUDE.md` repeats
#     it.
#   * **MultiFloats cannot run this case at all** — `sin` and `exp` are not
#     implemented there — so the precision study of step 12 takes Sod and
#     Sedov and leaves this one at `Float64` and `Float32`.
#
# See "Kelvin–Helmholtz instability" in `CODE.md` for the case as decided, and
# "Step 10" in "Measured results" for every number below.

"""
    KelvinHelmholtz(T, Val(D); ρ₁ = 1, ρ₂ = 2, v₁ = 1//2, v₂ = -1//2,
                    p₀ = 5//2, L = 1//40, a = 1//100, γ = 5//3, Lbox = 1,
                    floors = …)

The parameters of McNally, Lyra & Passy's shear layer at working type `T`: the
two densities and the two velocities, the uniform pressure, the ramp width,
the amplitude of the seeded `v_y` mode, the box side, the equation of state
and the floors.

**`D = 2` only.** The setup is two-dimensional by definition — a shear layer
in `y` with a single seeded mode in `x`, and the diagnostics of Section 3 of
the paper are stated on the `(x, y)` plane — so any other `D` is refused
rather than guessed at. There is no planar smoke test here as there is for the
blast: a one-dimensional Kelvin–Helmholtz problem is a contact discontinuity
and nothing else.

`ρ_m = (ρ₁ − ρ₂)/2` and `v_m = (v₁ − v₂)/2` are computed once here and carried
as fields, so that [`kh_state`](@ref) — which is evaluated at every cell of
every mesh the initial-data cycle produces — does no arithmetic it can avoid.
Note their **signs**: with `ρ₁ < ρ₂` the paper's `ρ_m` is negative, and the
profile is written `ρ₁ − ρ_m e^{…}` rather than `ρ₁ + |ρ_m| e^{…}` because
that is the paper's form and a sign flipped in transcription would be
invisible in a plot.

`Lbox` scales the whole setup: the profile is evaluated at `x/Lbox` and
`y/Lbox`, so `Lbox = 1` is the paper's unit square and anything else is the
same dimensionless problem on a differently sized box. The ramp width `L` and
the amplitude `a` are in box units for the same reason.

**The floors sit eight orders of magnitude below the data** and are expected
never to fire, exactly as the entropy wave's and Sod's are: the density stays
in `[1, 2]` and the pressure never leaves the neighbourhood of `5/2`, so a
floor firing here would mean something is wrong upstream of the floors.
`test/kelvinhelmholtz_tests.jl` asserts all three counts are zero and that the
measured injection is exactly zero.

Every parameter is a rational converted to `T` rather than a decimal literal,
so a measured number stays put when the study is run at another type (see
"Precision" in `CODE.md`). `isbits`, like everything a kernel argument may
hold: the initial-data callback captures one of these.
"""
struct KelvinHelmholtz{T,E<:EquationOfState}
    ρ₁::T
    ρ₂::T
    v₁::T
    v₂::T
    ρ_m::T
    v_m::T
    p₀::T
    L::T
    a::T
    Lbox::T
    eos::E
    floors::Floors{T}
end

function KelvinHelmholtz(::Type{T}, ::Val{D}; ρ₁=1, ρ₂=2, v₁=1 // 2,
                         v₂=-1 // 2, p₀=5 // 2, L=1 // 40, a=1 // 100,
                         γ=5 // 3, Lbox=1,
                         floors::Floors{T}=Floors{T}(; ρ_atm=T(1 // 10^8),
                                                     p_atm=T(1 // 10^8),
                                                     p_floor=T(1 // 10^8))) where {T,D}
    D == 2 || throw(ArgumentError(
        "the Kelvin–Helmholtz test is stated for D = 2 and got D = $D: it is " *
        "a shear layer in y with a single seeded mode sin(4πx), and both of " *
        "McNally's diagnostics — the amplitude of that mode and the maximum " *
        "y-kinetic energy — are defined on the (x, y) plane. In one dimension " *
        "there is no shear and in three the setup would need a third profile " *
        "this package has no reference for."))
    ρ₁, ρ₂, v₁, v₂ = T(ρ₁), T(ρ₂), T(v₁), T(v₂)
    p₀, L, a, Lbox = T(p₀), T(L), T(a), T(Lbox)
    (ρ₁ > 0 && ρ₂ > 0 && p₀ > 0) || throw(ArgumentError(
        "the shear layer needs two positive densities and a positive " *
        "pressure, got ρ₁ = $ρ₁, ρ₂ = $ρ₂ and p₀ = $p₀: the case is a smooth " *
        "ramp between two states of a gas, and the floors are a repair for " *
        "what the scheme produces rather than a licence for what it is given."))
    ρ₁ != ρ₂ || throw(ArgumentError(
        "the shear layer needs ρ₁ ≠ ρ₂, got both $ρ₁: with equal densities " *
        "the Atwood number is zero, the instability is the pure vortex-sheet " *
        "one, and the mode amplitude the case is measured by has nothing to " *
        "grow on the density contrast."))
    v₁ != v₂ || throw(ArgumentError(
        "the shear layer needs v₁ ≠ v₂, got both $v₁: with no velocity jump " *
        "there is no shear and no instability."))
    L > 0 || throw(ArgumentError(
        "the ramp width must be positive, got L = $L: the whole point of " *
        "McNally's setup is that the interface is resolved, and L = 0 is the " *
        "sharp-interface problem whose small-scale structure never converges."))
    (min(ρ₁, ρ₂) > floors.ρ_atm && p₀ > floors.p_atm) || throw(ArgumentError(
        "the data sits at or below the floors (ρ ≥ $(min(ρ₁, ρ₂)), " *
        "p = $p₀ against ρ_atm = $(floors.ρ_atm) and p_atm = $(floors.p_atm)): " *
        "the atmosphere reset would replace the flow itself, and the floor " *
        "counts this case asserts to be zero would count the initial data."))
    return KelvinHelmholtz{T,IdealGas{T}}(ρ₁, ρ₂, v₁, v₂, (ρ₁ - ρ₂) / 2,
                                          (v₁ - v₂) / 2, p₀, L, a, Lbox,
                                          IdealGas(T(γ)), floors)
end

"""
    kh_state(w::KelvinHelmholtz, x) -> P

The initial **primitive** state `(ρ, v_x, v_y, p)` at position `x`: McNally,
Lyra & Passy's equations (1)–(5), a pure function of position.

Four branches in `y` and one seeded mode in `x`. The branch boundaries are at
`y/Lbox = ¼, ½, ¾`, and the profile is continuous at all three — at `¼` and
`¾` it is continuous in its *derivative* too, exactly, and at `½` the two
branches are the same expression, `ρ₂ + ρ_m e^{−1/(4L)}`, and agree to the
last bit. That value sits `4.5·10⁻⁵ ρ_m` away from `ρ₂` rather than at it,
because the exponential ramps of the two interfaces overlap in the middle of
each slab; it is the paper's own setup and not a transcription slip.

The pressure is uniform, which is what makes the layer a **contact**: the
density jumps across it and the pressure does not, so it is the wave an
HLL-family flux diffuses most and the one HLLC restores. That is the whole
reason this case is where the two solvers are compared.

A plain function of `(w, x)` rather than the closure [`kh_initial`](@ref)
returns, so that the closure captures only `w`, which is `isbits`. `π`, `sin`
and `exp` are taken at `T`.
"""
@inline function kh_state(w::KelvinHelmholtz{T}, x) where {T}
    ξ = x[1] / w.Lbox
    η = x[2] / w.Lbox
    quarter, half, threequarters = T(1 // 4), T(1 // 2), T(3 // 4)
    if η < quarter
        e = exp((η - quarter) / w.L)
        ρ = w.ρ₁ - w.ρ_m * e
        vx = w.v₁ - w.v_m * e
    elseif η < half
        e = exp((quarter - η) / w.L)
        ρ = w.ρ₂ + w.ρ_m * e
        vx = w.v₂ + w.v_m * e
    elseif η < threequarters
        e = exp((η - threequarters) / w.L)
        ρ = w.ρ₂ + w.ρ_m * e
        vx = w.v₂ + w.v_m * e
    else
        e = exp((threequarters - η) / w.L)
        ρ = w.ρ₁ - w.ρ_m * e
        vx = w.v₁ - w.v_m * e
    end
    vy = w.a * sin(4 * T(π) * ξ)
    return (ρ, vx, vy, w.p₀)
end

"""
    kh_initial(w::KelvinHelmholtz)

The initial data as a closure `x -> P`, in **primitive** variables, which is
how `CODE.md` says a case states its data.
"""
kh_initial(w::KelvinHelmholtz) = x -> kh_state(w, x)

"""
    kh_conserved(w::KelvinHelmholtz)

The same initial data in **conserved** variables and wrapped in `AllVariables`:
the object `fill_by_coordinates!` fills a state with. There is no boundary hook
for this case — it is periodic in both directions — so unlike the tube's and
the blast's, this object has exactly one job.
"""
kh_conserved(w::KelvinHelmholtz) =
    AllVariables(x -> prim2con(w.eos, kh_state(w, x)))

"""
    HydroCase(w::KelvinHelmholtz; roots = 4, speed_headroom = 1)

The shear layer as a case the driver can run: **periodic in both directions**,
no boundary hook, and **no reference**, since the case has no closed form and
the quantitative reference is a uniform fine run of this code.

Periodicity here is intrinsic and not an economy (`CODE.md`, "Boundaries"):
the shear flow and its `sin(4πx)` seed are periodic in `x` by construction, and
McNally's two-interface profile is what makes the setup periodic in `y` — that
is *why* there are two interfaces rather than one. A Dirichlet boundary
anywhere would be wrong from the first step, since the gas crosses every face
of the box at `t = 0`.

**`speed_headroom = 1`, and the number is measured** (step 10). The flow is
smooth and subsonic — `c_s = sqrt(γ p/ρ)` is 2.0412 in the light gas and
1.4434 in the heavy one against `|v_x| ≤ ½` — so there is no Riemann problem
in the initial data whose star region could outrun it, which is the mechanism
that forces `2` on Sod and on Sedov. The entropy wave takes `1` for the same
reason and the end-of-chunk recheck in [`evolve!`](@ref) is what makes either
a measurement rather than a hope; the worst growth this case actually shows is
recorded in `CODE.md` under "Step 10".
"""
function HydroCase(w::KelvinHelmholtz{T}; roots=4, speed_headroom=1) where {T}
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), 2) : ntuple(_ -> Int(roots), 2)
    allequal(rs) || throw(ArgumentError(
        "the Kelvin–Helmholtz box is the square [0, Lbox]² and TreeAMR's " *
        "blocks are cubes, so it needs the same root count in both " *
        "dimensions, got $rs. The two directions are not interchangeable — " *
        "the shear is along x and the ramps are across y — but the box is " *
        "square in the paper and the diagnostics are stated on a square."))
    return HydroCase(T, Val(2); initial=x -> kh_state(w, x), eos=w.eos,
                     floors=w.floors, boundary=nothing, periodic=(true, true),
                     extents=((zero(T), w.Lbox), (zero(T), w.Lbox)), roots=rs,
                     speed_headroom=speed_headroom, reference=nothing)
end

"""
    mode_amplitude(P::FieldSet, w::KelvinHelmholtz) -> Float64

The amplitude `M(t)` of the seeded `v_y` mode: McNally, Lyra & Passy's
equations **(14)–(17)**, the area-weighted form,

    s_i = V_y w_i sin(4πx_i) e_i      c_i = V_y w_i cos(4πx_i) e_i
    d_i = w_i e_i                     M   = 2 √((Σs/Σd)² + (Σc/Σd)²)

with `e_i = e^{−4π|y_i − ¼|}` for `y_i < ½` and `e^{−4π|(1−y_i) − ¼|}` for
`y_i ≥ ½`, and `w_i` the cell's **area** `h_b²`.

**Both interfaces are read, mirrored** — the weighting is the paper's, and
`CODE.md`'s first draft said "the lower interface alone", which is wrong and is
corrected there (step 10). The two interfaces of a periodic two-slab setup
carry the same mode with opposite sign of `∂_y v_x`, and the mirrored weight is
what lets them add rather than cancel.

**The area-weighted form and not the uniform-grid one** (equations (6)–(9)),
because an adaptive mesh has cells of two sizes by construction: on a mesh
where half the layer sits at the cap and half one level below it, the uniform
sums would weight a coarse cell as though it were a fine one and `M` would jump
at every regrid. The two forms agree exactly on a uniform mesh, where `w_i` is
a constant that cancels between the numerator and the denominator.

A **host loop in block order**, in the spirit of [`shock_radius`](@ref) and
[`reduce_to_grid`](@ref): it reads positions and values and knows nothing about
how the data got there, its answer does not move with the thread count, and it
runs once per chunk against tens of steps. `block_mapreduce` could not form it
in any case — the summand needs the cell's `v_y`, its position and its size at
once, and a reduction over one variable's values has none of the last two.

**`P` must be current**: [`update_primitives!`](@ref) is what leaves it so, and
[`evolve!`](@ref) calls the observer with it already done.
"""
function mode_amplitude(P::FieldSet{T,2}, w::KelvinHelmholtz{T}) where {T}
    work = Array(P.work)
    N = P.forest.N
    four_π = 4 * π
    Σs, Σc, Σd = 0.0, 0.0, 0.0
    for b in 1:nblocks(P)
        h = tofloat64(spacing(T, P.forest, blockkey(P, b)))
        area = h * h
        for idx in CartesianIndices(ntuple(d -> (P.G[d] + 1):(P.G[d] + N), 2))
            x = coordinates(T, P, b, Tuple(idx))
            ξ = tofloat64(T(x[1])) / tofloat64(w.Lbox)
            η = tofloat64(T(x[2])) / tofloat64(w.Lbox)
            # Equations (14)–(16): the upper half is read through its mirror
            # image `1 − y`, so that both interfaces contribute with the same
            # sign rather than cancelling.
            mirrored = η < 0.5 ? η : 1 - η
            e = exp(-four_π * abs(mirrored - 0.25))
            vy = tofloat64(T(work[Tuple(idx)..., 3, b]))
            Σs += vy * area * sin(four_π * ξ) * e
            Σc += vy * area * cos(four_π * ξ) * e
            Σd += area * e
        end
    end
    Σd > 0 || throw(ArgumentError(
        "mode_amplitude found no cell at all: the weight Σ w_i e_i came out " *
        "$Σd on a field set with $(nblocks(P)) blocks."))
    return 2 * sqrt((Σs / Σd)^2 + (Σc / Σd)^2)
end

"""
    max_y_kinetic_energy(P::FieldSet) -> Float64

The maximum of `½ ρ v_y²` over the owned cells — the second of McNally, Lyra &
Passy's two diagnostics, and the sensitive one.

The pair is deliberate and is the paper's: `M(t)` is a smoothed quantity that a
diffusive scheme still resolves, and the maximum `y`-kinetic energy density is
"very sensitive to noise in the computed velocity field" (their Section 3). A
scheme that smeared the layer would keep the first and lose the second, so a
claim made on one alone is half a claim.

Its growth rate is **twice** the mode's, since it is quadratic in `v_y`: the
paper's loose guide for the infinite-domain incompressible flow is
`M ∝ e^{4.384 t}` and `max ½ρv_y² ∝ e^{2·4.384 t}` (Wang et al. 2010, Eq. 18).

A host loop in block order for the reason [`mode_amplitude`](@ref) gives — the
summand needs two variables of the same cell, and `block_mapreduce` maps a
scalar function over one variable's values. It takes no case parameters: `ρ`
and `v_y` are slots 1 and 3 of the primitive set in `D = 2` and nothing else
about the problem enters.
"""
function max_y_kinetic_energy(P::FieldSet{T,2}) where {T}
    work = Array(P.work)
    N = P.forest.N
    best = 0.0
    for b in 1:nblocks(P)
        for idx in CartesianIndices(ntuple(d -> (P.G[d] + 1):(P.G[d] + N), 2))
            ρ = tofloat64(T(work[Tuple(idx)..., 1, b]))
            vy = tofloat64(T(work[Tuple(idx)..., 3, b]))
            best = max(best, ρ * vy * vy / 2)
        end
    end
    return best
end

"""
    growth_rate(ts, Ms; from, to = Inf) -> Float64

Least-squares slope of `log M` against `t` over the samples with
`from ≤ M ≤ to` — the measured exponential growth rate of the instability's
linear phase.

**The window is half the claim, and it is stated rather than discovered.**
`M(t)` is not an exponential for all time: it begins at the seeded
perturbation's own amplitude and has to shed a transient while the ramp
adjusts, and it bends over as the rolls saturate. A fit over the whole run
would measure those two ends rather than the instability. The window is given
as a range of `M` and not of `t` so that it means the same thing at every
resolution and under either flux — the phase a run is in is a property of how
far the mode has grown, not of the clock.

This is [`convergence_rate`](@ref)'s least squares with the abscissa left
alone: there the slope is `d log(err)/d log(h)` and here it is `d log M/dt`,
so the two cannot share a call the way [`exponent_fit`](@ref) shares one.

The rate is to be compared with **two** upper bounds, and it must stay below
both. The paper's loose guide is `M ∝ e^{4.384 t}` for the infinite-domain
incompressible flow (Wang et al. 2010, Eq. 18), and the sharp-interface
incompressible result is `k Δv √(ρ₁ρ₂)/(ρ₁+ρ₂) ≈ 5.92` for `k = 4π`. The ramp
and the compressibility both slow the real thing.
"""
function growth_rate(ts, Ms; from, to=Inf)
    length(ts) == length(Ms) || throw(ArgumentError(
        "growth_rate needs one amplitude per time, got $(length(ts)) times " *
        "and $(length(Ms)) amplitudes."))
    keep = [i for i in eachindex(ts)
            if Ms[i] > 0 && from ≤ Ms[i] ≤ to]
    length(keep) ≥ 2 || throw(ArgumentError(
        "growth_rate needs at least two samples with $from ≤ M ≤ $to, got " *
        "$(length(keep)) out of $(length(ts)): the fit runs over the linear " *
        "phase, which is a window in M and not in t, so either the run is too " *
        "short to have entered it or the window was set for another " *
        "resolution. Widen the window and say in the record that it was " *
        "widened."))
    x = [tofloat64(ts[i]) for i in keep]
    y = [log(tofloat64(Ms[i])) for i in keep]
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

"""
    kh_run([T = Float64], Val(2); N, ops, chunk, maxlevel_cap, refine_tol,
           coarsen_tol, t_end = 3//2, roots = 4, scale = 1, riemann = :hllc, …)

Run the shear layer through [`evolve!`](@ref) and record **both** of McNally's
diagnostics once per chunk through the observer, returning the driver's own
result beside the curves: `ts`, `Ms` ([`mode_amplitude`](@ref)), `Ks`
([`max_y_kinetic_energy`](@ref)) and `nbs` (the block count at each sample).

The observer is the only place either diagnostic can be taken, and that is why
this wrapper exists rather than the tests calling `evolve!` themselves: the
state is scattered into `U` and `P` is current exactly there and nowhere a
caller can reach, and the next thing the loop does is regrid, which invalidates
both. The viewer of step 11 reads the same curves through the same hook.

`scale` multiplies the case's root brick, which is how the **uniform fine
reference** is built: `maxlevel_cap = 0` with `scale = 2^cap` gives a uniform
mesh at the tracked run's own finest spacing, and `scale = 1` gives the uniform
coarse control at its coarsest. [`kh_uniform`](@ref) is that call under a name.

`riemann = :hllc` is this case's default and **the one place in the package
where it is** (decided in step 10; see "Riemann solver" in `CODE.md`). The
shear layer is a contact, HLLE's two-wave average smears exactly that wave, and
the measurement recorded under "Step 10" is what chose it. The package-wide
default in [`HydroProblem`](@ref) and [`evolve!`](@ref) stays `:hlle`, which is
*the* GRMHD flux.

`accounting = true` by default, unlike [`evolve!`](@ref): this is a measurement
driver, the injection is one of the numbers it exists to report, and on this
case it is expected to be exactly zero.

`refine_tol`, `coarsen_tol`, `chunk`, `maxlevel_cap`, `N` and `ops` have no
defaults, for the reason [`evolve!`](@ref) gives; anything not listed goes to
[`KelvinHelmholtz`](@ref).
"""
kh_run(valD::Val; kwargs...) = kh_run(Float64, valD; kwargs...)

function kh_run(::Type{T}, ::Val{D}; N, ops, chunk, maxlevel_cap, refine_tol,
                coarsen_tol, t_end=3 // 2, roots=4, scale=1, limiter=:minmod,
                riemann=:hllc, fixup=true, reset=:stage, cfl=2 // 5,
                speed_headroom=1, accounting::Bool=true, backend=CPU(),
                params...) where {T,D}
    w = KelvinHelmholtz(T, Val(D); params...)
    case = HydroCase(w; roots=roots, speed_headroom=speed_headroom)
    ts, Ms, Ks, nbs = Float64[], Float64[], Float64[], Int[]
    function watch(pr, t, u)
        push!(ts, tofloat64(T(t)))
        push!(Ms, mode_amplitude(pr.P, w))
        push!(Ks, max_y_kinetic_energy(pr.P))
        push!(nbs, nblocks(pr.P))
    end
    r = evolve!(case, Val(D); N=N, ops=ops, t_end=t_end, chunk=chunk,
                limiter=limiter, refine_tol=refine_tol,
                coarsen_tol=coarsen_tol, maxlevel_cap=maxlevel_cap,
                roots=case.roots .* scale, cfl=cfl, riemann=riemann,
                fixup=fixup, reset=reset, accounting=accounting,
                backend=backend, observer=watch)
    return (; r, w, ts, Ms, Ks, nbs)
end

"""
    kh_uniform([T = Float64], Val(2); scale = 1, …)

The same shear layer on a **uniform** mesh with no regridding — the quantitative
reference this case has instead of a closed form, and the control that says the
refinement bought anything.

[`kh_run`](@ref) with `maxlevel_cap = 0`; `scale = 2^cap` is the *fine*
reference, which shares the tracked run's finest spacing, and `scale = 1` is
the *coarse* control, which shares its coarsest. Pass the tracked run's `chunk`
so that the two measure `λ_max` at the same cadence, take comparable steps, and
sample `M(t)` at the same times — the last of which is what makes the two
curves subtractable.

With the cap at zero the criterion can refine nothing, so the tolerances are
still passed and still mean nothing; they are given the calibrated values here
so that a caller cannot read a different number into the reference than into
the run it references.
"""
kh_uniform(valD::Val; kwargs...) = kh_uniform(Float64, valD; kwargs...)

kh_uniform(::Type{T}, valD::Val; refine_tol=2 // 25, coarsen_tol=1 // 50,
           kwargs...) where {T} =
    kh_run(T, valD; maxlevel_cap=0, refine_tol=refine_tol,
           coarsen_tol=coarsen_tol, kwargs...)
