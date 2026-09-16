# Toro's exact Riemann solver for the ideal gas: the shock tube's
# *reference solution*, and deliberately not a flux.
#
# Every other numerical method in this package was chosen because it has a
# GRMHD counterpart (see "What has a GRMHD counterpart" in `CODE.md`). This
# one has none — there is no closed-form Riemann solution for relativistic
# magnetohydrodynamics, and even in Newtonian hydrodynamics an iterative
# solver in every cell face of every stage is not what anybody uses. It is
# here for one job: to say what the right answer is, so that the L1 error
# of the scheme against it is a measurement and not a comparison of two
# approximations. `riemann_flux` in `riemann.jl` is what the scheme
# actually calls; nothing in this file is reachable from a kernel.
#
# **Host `Float64` throughout, at every working type.** The run may be at
# `Float32` or at `Float32x2`; the reference is computed in `Float64` and
# converted once, at the comparison, exactly as TreeWave's Hankel table is
# and for the same reason — a reference that shared the run's precision
# would stop being a reference where it matters most. This is therefore the
# one file in the package where a decimal literal is not a leak, because
# there is no `T` for it to leak into. See "Precision" in `CODE.md`.
#
# The transcription is Toro, *Riemann Solvers and Numerical Methods for
# Fluid Dynamics*, 3rd ed., chapter 4: the pressure function (4.5)–(4.7),
# the Newton–Raphson iteration of §4.3.2 with the adaptive initial guess of
# §4.3.3, the star densities (4.50)/(4.53), the wave speeds (4.52)/(4.55),
# and the sampling of §4.5.

"""
    ExactRiemann

The exact solution of one Riemann problem for the ideal gas: the two
initial states, the star region between the two nonlinear waves, and the
speeds of everything. Built by [`exact_riemann`](@ref) and read by
[`sample`](@ref).

A Riemann problem is self-similar — its solution depends on `x` and `t`
only through `ξ = x/t` — so *this one struct is the whole solution for all
time*, and sampling it is a search over five numbers rather than an
evolution. The struct is what makes that explicit: once it exists, nothing
iterates again.

The fields, in the order the solution is built: the adiabatic index `γ`;
the left state `ρ_L, u_L, p_L` and its sound speed `c_L`, then the right
one; the star pressure `p★` and velocity `u★`, which are common to both
sides of the contact; the two star densities `ρ★_L`, `ρ★_R` and their
sound speeds, which are not; `shock_L` and `shock_R`, whether each
nonlinear wave is a shock or a rarefaction fan; and the four speeds
`head_L ≤ tail_L ≤ u★ ≤ tail_R ≤ head_R` that bound the waves — the head
is the outer edge of a fan and the tail its inner one, and for a shock the
two coincide at the shock speed, which is what lets [`sample`](@ref) treat
both cases with one pair of comparisons. `iterations` is how many
Newton steps the pressure took, kept because a reference that quietly
stopped converging would be the worst kind of wrong.

Host `Float64`, as the file header says.
"""
struct ExactRiemann
    γ::Float64
    ρ_L::Float64
    u_L::Float64
    p_L::Float64
    c_L::Float64
    ρ_R::Float64
    u_R::Float64
    p_R::Float64
    c_R::Float64
    p★::Float64
    u★::Float64
    ρ★_L::Float64
    c★_L::Float64
    ρ★_R::Float64
    c★_R::Float64
    shock_L::Bool
    shock_R::Bool
    head_L::Float64
    tail_L::Float64
    tail_R::Float64
    head_R::Float64
    iterations::Int
end

# The lowest pressure the iteration is allowed to propose. Toro's TOL: the
# pressure function's rarefaction branch raises `p/p_K` to a fractional
# power, so a Newton step that overshot into negative pressure would take
# the whole solve out of the reals rather than merely too far.
const RIEMANN_PMIN = 1e-12

# `f_K(p)` and `f′_K(p)` for one side, Toro (4.6)–(4.7): how much the
# velocity changes across the wave that connects the state `K` to a star
# region at pressure `p`, and the derivative the Newton step needs.
#
# The two branches are the two kinds of wave, and which one applies is
# decided by the pressure alone: compression (`p > p_K`) is a shock and
# follows the Rankine–Hugoniot relation, expansion is an isentropic
# rarefaction. `f` and its derivative are continuous where the branches
# meet, at `p = p_K`, which is why a Newton iteration can cross between
# them without noticing.
@inline function wave_function(γ::Float64, ρ_K::Float64, p_K::Float64,
                               c_K::Float64, p::Float64)
    if p > p_K                                  # shock
        A = 2 / ((γ + 1) * ρ_K)
        B = (γ - 1) / (γ + 1) * p_K
        q = sqrt(A / (B + p))
        return (p - p_K) * q, q * (1 - (p - p_K) / (2 * (B + p)))
    end
    ratio = p / p_K                             # rarefaction
    f = 2 * c_K / (γ - 1) * (ratio^((γ - 1) / (2γ)) - 1)
    return f, ratio^(-(γ + 1) / (2γ)) / (ρ_K * c_K)
end

# The density behind a wave of either kind: the Rankine–Hugoniot jump
# (4.50) across a shock, the isentropic law `ρ ∝ p^{1/γ}` (4.53) through a
# fan. The pressure and the velocity are continuous across the contact and
# the density is not, which is why there are two of these and one `p★`.
@inline function star_density(γ::Float64, ρ_K::Float64, p_K::Float64,
                              p★::Float64)
    ratio = p★ / p_K
    p★ > p_K || return ρ_K * ratio^(1 / γ)
    g = (γ - 1) / (γ + 1)
    return ρ_K * (ratio + g) / (g * ratio + 1)
end

# Toro's adaptive initial guess, §4.3.3. The linearized (PVRS) estimate is
# used where the two pressures are within a factor of two of each other and
# it lands between them; otherwise the guess is the exact solution of the
# two-rarefaction or the two-shock problem, whichever the linearized value
# points at. Newton converges from almost anything positive here — the
# pressure function is monotone and convex — so the guess buys iterations
# rather than correctness, and a bad one on Toro's test 3 (a pressure ratio
# of 10⁵) is the difference between four steps and twenty.
function initial_pressure(γ::Float64, ρ_L, u_L, p_L, c_L, ρ_R, u_R, p_R, c_R)
    ppv = (p_L + p_R) / 2 + (u_L - u_R) * (ρ_L + ρ_R) * (c_L + c_R) / 8
    ppv = max(RIEMANN_PMIN, ppv)
    pmin, pmax = min(p_L, p_R), max(p_L, p_R)
    if pmax / pmin ≤ 2 && pmin ≤ ppv ≤ pmax
        return ppv                              # PVRS
    elseif ppv < pmin                           # two rarefactions, (4.46)
        e = (γ - 1) / (2γ)
        pq = (p_L / p_R)^e
        um = (pq * u_L / c_L + u_R / c_R + 2 * (pq - 1) / (γ - 1)) /
             (pq / c_L + 1 / c_R)
        return max(RIEMANN_PMIN,
                   (p_L * (1 + (γ - 1) * (u_L - um) / (2 * c_L))^(1 / e) +
                    p_R * (1 + (γ - 1) * (um - u_R) / (2 * c_R))^(1 / e)) / 2)
    end
    g_L = sqrt(2 / ((γ + 1) * ρ_L) / ((γ - 1) / (γ + 1) * p_L + ppv))
    g_R = sqrt(2 / ((γ + 1) * ρ_R) / ((γ - 1) / (γ + 1) * p_R + ppv))
    return max(RIEMANN_PMIN,                    # two shocks, (4.48)
               (g_L * p_L + g_R * p_R - (u_R - u_L)) / (g_L + g_R))
end

"""
    exact_riemann(γ, ρ_L, u_L, p_L, ρ_R, u_R, p_R; tol = 1e-12, maxiter = 100)

The exact solution of the Riemann problem with those two states, as an
[`ExactRiemann`](@ref).

The whole solution is one number, the star pressure `p★`, and everything
else follows from it algebraically. `p★` is the root of

    f(p) = f_L(p) + f_R(p) + (u_R − u_L)

where `f_K` is the velocity change across the nonlinear wave joining state
`K` to a star region at pressure `p` — the Rankine–Hugoniot jump where the
wave is a shock (`p > p_K`), the isentropic Riemann invariant where it is a
rarefaction (`p ≤ p_K`). `f` is monotone increasing and convex, so
Newton–Raphson from a positive guess converges monotonically from below;
the guess is Toro's adaptive one, and the iteration stops on a relative
change of `tol`. Then `u★ = ½(u_L + u_R) + ½(f_R(p★) − f_L(p★))`, the two
star densities, and the wave speeds.

**Vacuum is refused.** The solution above exists only while the two states
are not pulling apart faster than the gas can follow: the *pressure
positivity condition*

    2 (c_L + c_R) / (γ − 1) > u_R − u_L

Where it fails, the exact solution contains a vacuum region between two
rarefactions, `p★ = 0`, and the pressure function has no positive root at
all — a Newton iteration would wander rather than fail. So it is checked
first and refused with an [`ArgumentError`](@ref) naming both sides of the
inequality. Sod's problem is far from it (`u_R − u_L = 0` against a bound
of `11.2`).

Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
chapter 4. Host `Float64` at every working type; see the file header and
"Precision" in `CODE.md`.
"""
function exact_riemann(γ, ρ_L, u_L, p_L, ρ_R, u_R, p_R;
                       tol::Float64=1e-12, maxiter::Int=100)
    γ, ρ_L, u_L, p_L = Float64(γ), Float64(ρ_L), Float64(u_L), Float64(p_L)
    ρ_R, u_R, p_R = Float64(ρ_R), Float64(u_R), Float64(p_R)

    γ > 1 || throw(ArgumentError(
        "the exact Riemann solver needs γ > 1, got $γ: every exponent below " *
        "closes through γ − 1, and a sound speed sqrt(γ p / ρ) is not a real " *
        "number for a gas that is not one."))
    (ρ_L > 0 && p_L > 0 && ρ_R > 0 && p_R > 0) || throw(ArgumentError(
        "the exact Riemann solver needs both states positive, got " *
        "(ρ_L, p_L) = ($ρ_L, $p_L) and (ρ_R, p_R) = ($ρ_R, $p_R): this is the " *
        "reference solution, so a state the floors would have repaired is a " *
        "question about the initial data and not about the solver."))

    c_L = sqrt(γ * p_L / ρ_L)
    c_R = sqrt(γ * p_R / ρ_R)
    Δu = u_R - u_L
    bound = 2 * (c_L + c_R) / (γ - 1)
    bound > Δu || throw(ArgumentError(
        "these states generate a vacuum: the pressure positivity condition " *
        "2 (c_L + c_R)/(γ − 1) > u_R − u_L reads $bound > $Δu, which is " *
        "false, so the two gases separate faster than either can expand and " *
        "the exact solution has a vacuum region between two rarefactions " *
        "rather than a star region. There is no positive root of the " *
        "pressure function to iterate toward."))

    p★ = initial_pressure(γ, ρ_L, u_L, p_L, c_L, ρ_R, u_R, p_R, c_R)
    iterations = 0
    converged = false
    f_L = f_R = 0.0
    for _ in 1:maxiter
        iterations += 1
        f_L, df_L = wave_function(γ, ρ_L, p_L, c_L, p★)
        f_R, df_R = wave_function(γ, ρ_R, p_R, c_R, p★)
        pnew = max(RIEMANN_PMIN, p★ - (f_L + f_R + Δu) / (df_L + df_R))
        change = 2 * abs(pnew - p★) / (pnew + p★)
        p★ = pnew
        if change ≤ tol
            # One more evaluation, so that `u★` is built from `f_K` at the
            # pressure that is actually stored rather than at the previous
            # iterate.
            f_L, _ = wave_function(γ, ρ_L, p_L, c_L, p★)
            f_R, _ = wave_function(γ, ρ_R, p_R, c_R, p★)
            converged = true
            break
        end
    end
    converged || throw(ErrorException(
        "the exact Riemann solver's pressure iteration did not converge to a " *
        "relative $tol in $maxiter Newton steps, reaching p★ = $p★ with " *
        "residual $(f_L + f_R + Δu). The pressure function is monotone and " *
        "convex, so this is a sign of an initial state the checks above let " *
        "through rather than of a step size."))

    u★ = (u_L + u_R) / 2 + (f_R - f_L) / 2
    ρ★_L = star_density(γ, ρ_L, p_L, p★)
    ρ★_R = star_density(γ, ρ_R, p_R, p★)
    c★_L = sqrt(γ * p★ / ρ★_L)
    c★_R = sqrt(γ * p★ / ρ★_R)
    shock_L, shock_R = p★ > p_L, p★ > p_R

    # A shock has one speed and a fan has two; storing the shock speed as
    # both the head and the tail is what lets `sample` ask the same two
    # questions either way. (4.52) for the shocks, (4.55) for the fans.
    head_L, tail_L = if shock_L
        s = u_L - c_L * sqrt((γ + 1) / (2γ) * p★ / p_L + (γ - 1) / (2γ))
        s, s
    else
        u_L - c_L, u★ - c★_L
    end
    tail_R, head_R = if shock_R
        s = u_R + c_R * sqrt((γ + 1) / (2γ) * p★ / p_R + (γ - 1) / (2γ))
        s, s
    else
        u★ + c★_R, u_R + c_R
    end

    return ExactRiemann(γ, ρ_L, u_L, p_L, c_L, ρ_R, u_R, p_R, c_R,
                        p★, u★, ρ★_L, c★_L, ρ★_R, c★_R, shock_L, shock_R,
                        head_L, tail_L, tail_R, head_R, iterations)
end

"""
    sample(sol::ExactRiemann, ξ) -> (ρ, u, p)

The primitive state of the exact solution at `ξ = x/t`.

The solution is self-similar, so this one function is the solution
everywhere and at every time: the state at position `x` and time `t > 0` is
`sample(sol, x/t)`, and the initial data is the limit `t → 0`, which
`ξ = ±Inf` returns exactly.

Six regions, in the order `ξ` crosses them: the left state, the left
rarefaction fan, the left star state, the right star state, the right fan,
the right state. Which of the six applies is decided by four comparisons
against the wave speeds the constructor stored — and by the *same* four
whether a wave is a shock or a fan, since a shock is stored with its head
and tail at one speed, so the fan branch is unreachable for it rather than
guarded against.

Inside a fan the flow is an isentropic expansion whose Riemann invariant is
constant, which makes `u` and `c` linear in `ξ` (Toro §4.5); `ρ` and `p`
then follow from the isentropic law. That is the reason
[`max_signal_speed`](@ref) can bound the whole solution by looking at the
four constant states alone: inside a fan `|u| + c` is a convex piecewise
linear function of `ξ` and so attains its maximum at an edge, where it
equals the neighbouring constant state's value.
"""
function sample(sol::ExactRiemann, ξ)
    ξ = Float64(ξ)
    γ = sol.γ
    if ξ ≤ sol.u★
        ξ ≤ sol.head_L && return (sol.ρ_L, sol.u_L, sol.p_L)
        ξ ≥ sol.tail_L && return (sol.ρ★_L, sol.u★, sol.p★)
        u = 2 / (γ + 1) * (sol.c_L + (γ - 1) / 2 * sol.u_L + ξ)
        c = 2 / (γ + 1) * (sol.c_L + (γ - 1) / 2 * (sol.u_L - ξ))
        r = c / sol.c_L
        return (sol.ρ_L * r^(2 / (γ - 1)), u, sol.p_L * r^(2γ / (γ - 1)))
    end
    ξ ≥ sol.head_R && return (sol.ρ_R, sol.u_R, sol.p_R)
    ξ ≤ sol.tail_R && return (sol.ρ★_R, sol.u★, sol.p★)
    u = 2 / (γ + 1) * (-sol.c_R + (γ - 1) / 2 * sol.u_R + ξ)
    c = 2 / (γ + 1) * (sol.c_R - (γ - 1) / 2 * (sol.u_R - ξ))
    r = c / sol.c_R
    return (sol.ρ_R * r^(2 / (γ - 1)), u, sol.p_R * r^(2γ / (γ - 1)))
end

"""
    max_signal_speed(sol::ExactRiemann)

The largest `|u| + c_s` anywhere in the exact solution, at any time — the
true supremum of the signal speed a run of this Riemann problem will ever
see.

The reason it exists is the time step. [`hydro_dt`](@ref) takes a `λ_max`,
and a driver measures that from the *state it has*; but a Riemann problem's
fastest signal is not in its initial data. Sod's initial data carries
`c_L = sqrt(7/5) ≈ 1.18` and nothing faster, while the state behind its
shock carries `u★ + c★_R ≈ 2.19`, so a step sized from `t = 0` runs the
first steps at nearly twice the intended CFL number. The shock tube takes
its `λ` from here instead, which is a bound for all time by construction.
See "Time integration and the time step" in `CODE.md`, where the measured
ratio of the two is recorded and what the driver of step 7 owes it is
written down.

Four numbers suffice: the two initial states and the two star states. A
rarefaction fan interpolates between one of each, and `|u| + c` is convex
and piecewise linear in `ξ` there (see [`sample`](@ref)), so the fan can
carry nothing its own edges do not. The shock speeds are not candidates —
a shock is subsonic relative to the state behind it, so `|s|` is below that
state's own `|u| + c`.
"""
max_signal_speed(sol::ExactRiemann) =
    max(abs(sol.u_L) + sol.c_L, abs(sol.u★) + sol.c★_L,
        abs(sol.u★) + sol.c★_R, abs(sol.u_R) + sol.c_R)
