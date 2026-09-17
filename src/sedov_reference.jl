# The Sedov–Taylor similarity law: the blast wave's *reference solution*,
# and — like the exact Riemann solver beside it — deliberately not a method.
#
# A point explosion in a uniform gas at rest has no length scale and no time
# scale of its own, so dimensional analysis alone fixes the shock's radius up
# to one dimensionless number:
#
#     r_s(t) = ξ₀ (E₀ t² / ρ₀)^{1/(D+2)}
#
# The **exponent** `2/(D+2)` is free: it follows from the dimensions and needs
# no solution at all, which is what makes it the acceptance check a code can be
# held to without agreeing on a constant. `ξ₀(γ, D)` is not free — it is
# `α^{-1/(D+2)}` with `α` the dimensionless energy integral of the similarity
# solution — and this file computes it.
#
# **Host `Float64` throughout, at every working type**, as `exact_riemann.jl`
# is and for the same reason: a reference that shared the run's precision would
# stop being a reference where it matters most. This is therefore the other
# file in the package where a decimal literal is not a leak, because there is
# no `T` for it to leak into. See "Precision" in `CODE.md`.
#
# ## Where the closed form comes from
#
# Kamm & Timmes (2007), LA-UR-07-2849, "On efficient generation of numerically
# robust Sedov solutions" is the reference `CODE.md` names, and it gives Sedov's
# closed-form parametrization. This file derives the same parametrization from
# the similarity equations rather than transcribing its coefficients, because
# `CLAUDE.md` records what transcription from memory costs (the
# Kelvin–Helmholtz formulas), and because a derivation can be *checked*: see
# `test/sedov_tests.jl`, which asserts the momentum equation's residual on the
# result. The derivation, in the variables `λ = r/r_s`, `V = v t / r`,
# `G = ρ/ρ₀` and `Z = c² t²/r²`:
#
#   * The **energy integral**. The energy inside a similarity surface
#     `λ = const` is constant in time, which is one algebraic relation between
#     `Z` and `V`:
#
#         Z = γ(γ−1) V² (V−ν) / (2(ν−γV)),        ν = 2/(D+2)
#
#     It reproduces the strong-shock value `Z(1) = 2γ(γ−1)ν²/(γ+1)²` exactly,
#     which is the first check that it is the right integral.
#   * **Continuity and entropy** then give `dlnλ/dV` and `dlnG/dV` as *rational*
#     functions of `V`, with simple poles at `V = 0, ν/γ, 2/A, ν` where
#     `A = D(γ−1) + 2`. Partial fractions integrate them in closed form, so `λ`
#     and `G` are products of powers — Sedov's parametrization, with the
#     residues computed numerically here rather than written out.
#   * The parameter runs from `V₂ = 2ν/(γ+1)` at the shock (`λ = 1`) down to
#     `V₀ = ν/γ` at the centre (`λ = 0`), where `Z → ∞` and `G → 0`.
#
# ## Two things that bite, both measured
#
#   * **The centre endpoint cancels.** `V` and `V₀` agree to the last bit long
#     before `λ` is small, so every formula here is written in terms of
#     `u = V − V₀` and `V` is only ever *formed*, never differenced. Writing
#     `V - V₀` instead produces a `NaN` at `λ ≈ 10⁻³` and an integral that does
#     not terminate.
#   * **The planar `α` is twice the literature's**, and the factor is a
#     convention rather than an error. `α` here is `σ_D ∫ …` with
#     `σ_1 = 2, σ_2 = 2π, σ_3 = 4π` — the surface factor belonging to the
#     deposition volumes `V_1 = 2r₀`, `V_2 = πr₀²`, `V_3 = 4πr₀³/3` that
#     [`sedov_state`](@ref) uses — so `E₀` is the energy in `|x| < r`, *both*
#     sides of a planar blast. Kamm & Timmes count one side. See
#     [`sedov_alpha`](@ref).

"""
    adaptive_simpson(f, a, b; atol, rtol, maxdepth) -> Float64

Adaptive Simpson quadrature with Richardson extrapolation: the whole numerical
method this file needs, written here rather than taken as a dependency, in the
spirit of `precision.jl` and `convergence_rate`.

Both an absolute and a relative tolerance, and a depth cap, because the
integrand this exists for vanishes at one endpoint: a purely *relative*
criterion never terminates on a panel whose exact value is zero, and a
recursion that halves a panel sixty times is a hang rather than an error.

Host `Float64`, like everything in this file.
"""
function adaptive_simpson(f, a::Float64, b::Float64; atol::Float64=1e-15,
                          rtol::Float64=1e-13, maxdepth::Int=30)
    fa, fm, fb = f(a), f((a + b) / 2), f(b)
    whole = (b - a) / 6 * (fa + 4 * fm + fb)
    return simpson_panel(f, a, b, fa, fm, fb, whole, atol, rtol, maxdepth)
end

# One panel: halve it, compare the two halves against the whole, and either
# accept with Richardson's `/15` correction or recurse. The acceptance test is
# written as `!(err > tol)` so that a `NaN` accepts rather than recursing to
# the depth cap — a `NaN` panel is a bug to be found in the integrand, not one
# to be subdivided.
function simpson_panel(f, a, b, fa, fm, fb, whole, atol, rtol, depth)
    m = (a + b) / 2
    flm, frm = f((a + m) / 2), f((m + b) / 2)
    left = (m - a) / 6 * (fa + 4 * flm + fm)
    right = (b - m) / 6 * (fm + 4 * frm + fb)
    err = abs(left + right - whole)
    if depth ≤ 0 || !(err > 15 * (atol + rtol * abs(left + right)))
        return left + right + (left + right - whole) / 15
    end
    return simpson_panel(f, a, m, fa, flm, fm, left, atol, rtol, depth - 1) +
           simpson_panel(f, m, b, fm, frm, fb, right, atol, rtol, depth - 1)
end

"""
    SedovSimilarity(γ, D)

The Sedov–Taylor similarity solution for an adiabatic index `γ` in `D`
dimensions: the constant `ξ₀` of the law
`r_s(t) = ξ₀ (E₀ t²/ρ₀)^{1/(D+2)}`, the dimensionless energy integral `α` it
comes from, and the coefficients of the closed-form profile.

Built once and read many times, as [`ExactRiemann`](@ref) is: the quadrature
that produces `α` runs in the constructor and nothing iterates again. Host
`Float64` at every working type.

The fields, in the order the solution is built: `γ` and the geometry index `D`
(1 planar, 2 cylindrical, 3 spherical); the similarity exponent `ν = 2/(D+2)`
and `A = D(γ−1)+2`; the three values of the parameter `V` that bound and
puncture its range — `V₀ = ν/γ` at the centre, `V₂ = 2ν/(γ+1)` at the shock,
and the pole `V_A = 2/A` outside it — with `u₂ = V₂−V₀`, `dA = V₀−V_A` and
`dν = V₀−ν` the offsets every formula uses so that nothing is ever differenced
against `V₀`; the three residues `r0, r1, r2` of `dlnλ/dV` and the three
`q1, q2, q3` of `dlnG/dV`; the post-shock compression `G₂ = (γ+1)/(γ−1)`; and
finally `α` and `ξ₀`.

`γ > 1` is required and says why: the solution is a strong shock in a polytrope
and there is none at `γ ≤ 1`.
"""
struct SedovSimilarity
    γ::Float64
    D::Int
    ν::Float64
    A::Float64
    V₀::Float64
    V₂::Float64
    V_A::Float64
    u₂::Float64
    dA::Float64
    dν::Float64
    r0::Float64
    r1::Float64
    r2::Float64
    q1::Float64
    q2::Float64
    q3::Float64
    G₂::Float64
    α::Float64
    ξ₀::Float64
end

# The surface factor belonging to this package's deposition volumes: the
# "volume" of |x| < r is `2r`, `πr²`, `4πr³/3`, so its derivative is `2`, `2πr`,
# `4πr²`. See the note on the planar convention in the file header.
sedov_surface(D::Integer) = D == 1 ? 2.0 : D == 2 ? 2π : 4π

# `P` and `Q`, the numerators of `dlnλ/dV` and `dlnG/dV`:
#
#     dlnλ/dV = P(V) / (−γ A V (V−V₀)(V−V_A))
#     dlnG/dV = Q(V) / ( γ A   (V−V₀)(V−V_A)(V−ν))
#
# with `P(V) = γ(1+γ)V² − 2ν(1+γ)V + 2ν²` and
# `Q(V) = D·P(V) − (γV−ν)(AV−2)`. Both are quadratics over cubics, so the
# partial-fraction expansions below are exact.
sedov_P(γ, ν, V) = γ * (1 + γ) * V^2 - 2ν * (1 + γ) * V + 2ν^2
sedov_Q(γ, ν, A, D, V) = D * sedov_P(γ, ν, V) - (γ * V - ν) * (A * V - 2)

function SedovSimilarity(γ::Real, D::Integer)
    γ = Float64(γ)
    γ > 1 || throw(ArgumentError(
        "the Sedov similarity solution needs γ > 1, got $γ: the solution is a " *
        "strong shock in a polytrope, its compression is (γ+1)/(γ−1), and " *
        "neither exists at γ ≤ 1."))
    1 ≤ D ≤ 3 || throw(ArgumentError(
        "the Sedov similarity solution is stated for D = 1, 2 or 3 — planar, " *
        "cylindrical and spherical — got D = $D."))
    ν = 2 / (D + 2)
    A = D * (γ - 1) + 2
    V₀, V₂, V_A = ν / γ, 2ν / (γ + 1), 2 / A
    P(V) = sedov_P(γ, ν, V)
    Q(V) = sedov_Q(γ, ν, A, D, V)
    # The residues of the two logarithmic derivatives, at the poles
    # 0, V₀, V_A and V₀, V_A, ν respectively.
    r0 = P(0.0) / (-γ * A * (0 - V₀) * (0 - V_A))
    r1 = P(V₀) / (-γ * A * V₀ * (V₀ - V_A))
    r2 = P(V_A) / (-γ * A * V_A * (V_A - V₀))
    q1 = Q(V₀) / (γ * A * (V₀ - V_A) * (V₀ - ν))
    q2 = Q(V_A) / (γ * A * (V_A - V₀) * (V_A - ν))
    q3 = Q(ν) / (γ * A * (ν - V₀) * (ν - V_A))
    # Built twice on purpose: the quadrature that produces `α` reads the
    # profile, which is this struct, so the parametrization exists before the
    # constant does. The first one carries `NaN` in the two fields nothing has
    # computed yet, which is the honest value for them and which
    # [`sedov_alpha`](@ref) never reads.
    build(α, ξ₀) = SedovSimilarity(γ, Int(D), ν, A, V₀, V₂, V_A, V₂ - V₀,
                                   V₀ - V_A, V₀ - ν, r0, r1, r2, q1, q2, q3,
                                   (γ + 1) / (γ - 1), α, ξ₀)
    α = sedov_alpha(build(NaN, NaN))
    return build(α, α^(-1 / (D + 2)))
end

"""
    sedov_profile(sim::SedovSimilarity, u) -> (λ, G, V, Z)

The similarity solution at the parameter value `u = V − V₀`: the radius
fraction `λ = r/r_s`, the compression `G = ρ/ρ₀`, the velocity function
`V = v t / r`, and `Z = c² t²/r²`, from which
`p = ρ₀ G Z (r/t)²/γ`.

**Parametric in `u`, not in `λ`**, and that is the whole of what it is: `u`
runs from `0` at the centre to `u₂ = V₂ − V₀` at the shock, and the map
`u ↦ λ` is the closed form. Inverting it for a profile at a *given* radius is a
root find this package does not need — the exponent, the jump and the
comparison against a uniform fine run are the acceptance for the Sedov case,
and `CODE.md` records the full radial profile as an extension rather than a
milestone.

It is here because it **came for free** with the energy integral: `α` is a
quadrature over exactly these three functions, so having them is not a cost,
and the radial-scatter panel of step 11 will want them. Nothing in the package
asserts it beyond what `α` and the momentum-equation residual assert, which is
what "untested" means for it.

The argument is `u` rather than `V` because `V` and `V₀` agree to the last bit
long before `λ` is small; see the file header.
"""
function sedov_profile(sim::SedovSimilarity, u::Float64)
    V = sim.V₀ + u
    λ = (V / sim.V₂)^sim.r0 * (u / sim.u₂)^sim.r1 *
        ((u + sim.dA) / (sim.u₂ + sim.dA))^sim.r2
    G = sim.G₂ * (u / sim.u₂)^sim.q1 *
        ((u + sim.dA) / (sim.u₂ + sim.dA))^sim.q2 *
        ((u + sim.dν) / (sim.u₂ + sim.dν))^sim.q3
    # `ν − γV` is exactly `−γu`, which is the difference that must not be
    # formed from `V` and `V₀`.
    Z = sim.γ * (sim.γ - 1) * V^2 * (u + sim.dν) / (2 * (-sim.γ * u))
    return (λ, G, V, Z)
end

"""The logarithmic derivatives of `λ` and `G` at `u = V − V₀`."""
sedov_dlnλ(sim::SedovSimilarity, u::Float64) =
    sedov_P(sim.γ, sim.ν, sim.V₀ + u) /
    (-sim.γ * sim.A * (sim.V₀ + u) * u * (u + sim.dA))

sedov_dlnG(sim::SedovSimilarity, u::Float64) =
    sedov_Q(sim.γ, sim.ν, sim.A, sim.D, sim.V₀ + u) /
    (sim.γ * sim.A * u * (u + sim.dA) * (u + sim.dν))

"""
    sedov_alpha(sim::SedovSimilarity) -> Float64

The **dimensionless energy integral** of the similarity solution,

    α = σ_D ∫₀¹ λ^{D+1} G (V²/2 + Z/(γ(γ−1))) dλ

with `σ_1 = 2`, `σ_2 = 2π`, `σ_3 = 4π`. It is the whole content of `ξ₀`:
setting the integral equal to `E₀` gives `1 = α ξ₀^{D+2}`, so
`ξ₀ = α^{-1/(D+2)}`.

**The planar value is twice the literature's, and the factor is the
convention.** `σ_1 = 2` counts the energy on *both* sides of a planar blast,
because that is the energy [`SedovBlast`](@ref) deposits: its top hat is
`|x| < r₀`, of "volume" `2r₀`, and the `E₀` that enters the law is the
measured `Σ hᴰ E` of the whole box. Kamm & Timmes count one side, so their
planar `α` is half of this one; the cylindrical and spherical values agree.
The measured numbers are in `CODE.md` under "Step 9".

**How the endpoint is handled.** The integrand has an integrable singularity at
the centre: in the parameter `u = V − V₀` it behaves as `u^{s−1}` with
`s = D(γ−1)/(2γ+D−2)`, which is `0.32` for `γ = 7/5` in 3D — small enough that
no quadrature on `u` would converge usefully. The substitution
`u = u₂ w^{3/s}` turns it into `w²` times a function of `w^{3/s}`, which is
smooth, vanishes at `w = 0`, and integrates to full `Float64` precision; the
exponent `3` is free and the answer is unchanged at `2`, `4` and `5`, which is
the check that the substitution is not doing the work.
"""
function sedov_alpha(sim::SedovSimilarity)
    γ, D = sim.γ, sim.D
    # `s = D · r1` is the endpoint exponent; `r1` is the residue of `dlnλ/dV`
    # at the centre, so the relation is read off the parametrization rather
    # than quoted, and the closed form `D(γ−1)/(2γ+D−2)` is the test's.
    s = D * sim.r1
    m = 3 / s
    B = sim.u₂
    function integrand(w::Float64)
        w > 0 || return 0.0
        u = B * w^m
        u > 0 || return 0.0                    # underflow: the limit is zero
        λ, G, V, Z = sedov_profile(sim, u)
        val = λ^(D + 2) * sedov_dlnλ(sim, u) * G *
              (V^2 / 2 + Z / (γ * (γ - 1))) * (B * m * w^(m - 1))
        return isfinite(val) ? val : 0.0
    end
    return sedov_surface(D) * adaptive_simpson(integrand, 0.0, 1.0)
end

"""
    sedov_exponent(D) -> Float64
    sedov_exponent(sim::SedovSimilarity) -> Float64

The similarity exponent `2/(D+2)`: `2/3`, `1/2`, `2/5` in `D = 1, 2, 3`.

**The check that needs no constant.** `log r_s` against `log t` has this slope
whatever `γ` is and whatever `ξ₀` is, because it follows from the dimensions of
`E₀`, `ρ₀` and `t` alone. It is therefore the acceptance a code can be held to
without agreeing on a quadrature, and it is what
[`exponent_fit`](@ref) measures on a run.
"""
sedov_exponent(D::Integer) = 2 / (D + 2)
sedov_exponent(sim::SedovSimilarity) = sedov_exponent(sim.D)

"""
    sedov_radius(sim::SedovSimilarity, t, E₀, ρ₀ = 1) -> Float64

The shock radius `r_s(t) = ξ₀ (E₀ t²/ρ₀)^{1/(D+2)}`.

`E₀` is the energy actually deposited — `CODE.md` is explicit that it is the
**measured** `Σ hᴰ E` at `t = 0` on the adapted mesh less the ambient, not the
nominal value, because which cell centres fall inside `r₀` is a property of the
mesh (see [`measured_E₀`](@ref)). `t = 0` gives `0`, which is the law's own
answer and not a special case.
"""
function sedov_radius(sim::SedovSimilarity, t::Real, E₀::Real, ρ₀::Real=1.0)
    t = Float64(t)
    t ≥ 0 || throw(ArgumentError(
        "the Sedov law is stated for t ≥ 0, got $t: the blast begins at the " *
        "explosion and r_s(t) ∝ t^{2/(D+2)} has no real value before it."))
    Float64(E₀) > 0 && Float64(ρ₀) > 0 || throw(ArgumentError(
        "the Sedov law needs a positive energy and ambient density, got " *
        "E₀ = $E₀ and ρ₀ = $ρ₀."))
    return sim.ξ₀ * (Float64(E₀) * t^2 / Float64(ρ₀))^(1 / (sim.D + 2))
end

"""
    exponent_fit(ts, rs; from) -> Float64

Least-squares slope of `log r_s` against `log t` over the samples with
`r_s ≥ from` — the measured similarity exponent of a run, to be compared with
`2/(D+2)`.

**`from` is the whole of the claim's honesty.** The similarity solution is the
asymptotic one: it describes the blast only once it has forgotten the top hat
it started from, so a fit that included the early chunks would measure the
deposition rather than the law. The Sedov tests set `from` to a few `r₀` and
say so; `CODE.md` records the criterion beside the number.

[`convergence_rate`](@ref) is the same least-squares slope under another name —
it is what the convergence studies use — so this filters and forwards rather
than fitting a second line of its own.
"""
function exponent_fit(ts, rs; from)
    length(ts) == length(rs) || throw(ArgumentError(
        "exponent_fit needs one radius per time, got $(length(ts)) times and " *
        "$(length(rs)) radii."))
    keep = [i for i in eachindex(ts) if rs[i] ≥ from && ts[i] > 0 && rs[i] > 0]
    length(keep) ≥ 2 || throw(ArgumentError(
        "exponent_fit needs at least two samples with r_s ≥ $from and t > 0, " *
        "got $(length(keep)) out of $(length(ts)): the similarity law is the " *
        "asymptotic solution, so the fit runs over the chunks in which the " *
        "shock has forgotten its top hat. Either run longer, deposit into a " *
        "smaller r₀, or lower `from` and say in the record that it was " *
        "lowered."))
    return convergence_rate([Float64(ts[i]) for i in keep],
                            [Float64(rs[i]) for i in keep])
end
