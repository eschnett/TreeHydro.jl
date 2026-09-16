# The equation of state, the two state vectors, and the conversions
# between them.
#
# This is the smallest file in the scheme and the one every other part of
# it calls. The initial data is primitive and becomes conserved through
# `prim2con`; the evolved state is conserved and becomes primitive through
# `con2prim` at every right-hand-side evaluation, in every stored cell
# including the ghosts; the reconstruction, the Riemann solver and the
# refinement criterion all read primitives; the atmosphere reset goes
# `con2prim`, `apply_floors`, `prim2con` and writes back. See "The
# equations" in `CODE.md`.
#
# Two things about the shape of what is here, both of which are about
# GRMHD rather than about Newtonian hydrodynamics:
#
#   - The EOS is a *struct behind three functions*, not a `γ` passed
#     around, so that a hybrid or tabulated EOS is a new struct and
#     nothing else changes.
#   - `con2prim` is *fallible and flooring*, and says so in its return
#     value, although the Newtonian recovery is closed-form algebra that
#     could not fail on a physical state. In GRMHD the same signature
#     hides a one-dimensional root find, and it is the place where most of
#     a code's robustness lives. Keeping the signature is the point of
#     writing it this way here.
#
# Everything is arithmetic on `isbits` values in the working type `T`,
# written with integer literals only, so that it is callable from inside a
# KernelAbstractions kernel on any backend. See "Precision" in `CODE.md`.

"""
    EquationOfState

The gas law the scheme is closed with, as a type.

An equation of state provides three functions and nothing else:

    pressure(eos, ρ, ε)          # the closure itself
    internal_energy(eos, ρ, p)   # its inverse, for prim2con
    soundspeed(eos, ρ, p)        # the characteristic speed, for the fluxes

[`IdealGas`](@ref) is the only one here. A hybrid or tabulated equation of
state — the reason this is an interface rather than a `γ` threaded through
the signatures — is a new subtype implementing those three, after which
`prim2con`, `con2prim`, the reconstruction and the Riemann solvers are
unchanged. See "The equations" in `CODE.md`.

A subtype must be `isbits`: it is a kernel argument at every
right-hand-side evaluation and a captured value in the initial-data and
boundary callbacks.
"""
abstract type EquationOfState end

"""
    IdealGas(γ)
    IdealGas{T}(γ)

The ideal gas, `p = (γ − 1) ρ ε`, with adiabatic index `γ`.

The one equation of state in this package, and a case parameter rather
than a constant: Sod's tube uses `γ = 7/5` and the Kelvin–Helmholtz
instability `γ = 5/3`, following the setups they are compared against.

`γ` is stored as a float in the working type of the run. `IdealGas(7//5)`
is therefore a `Float64` gas; a run at another type writes
`IdealGas(T(7//5))`, which is also what keeps a `Float32` run from
promoting to `Float64` through the adiabatic index alone — `T(7//5)`
rather than `1.4` is the package's rule for exactly this (see "Precision"
in `CODE.md`).

`γ ≤ 1` is refused. It is not a gas: both `p = (γ − 1) ρ ε` and
`c_s = sqrt(γ p / ρ)` need `γ − 1 > 0` to return a positive pressure and a
real sound speed from a positive internal energy.

`isbits`, so that it travels into a kernel as an argument.
"""
struct IdealGas{T} <: EquationOfState
    γ::T

    function IdealGas{T}(γ) where {T}
        γ = convert(T, γ)
        γ > 1 || throw(ArgumentError(
            "IdealGas needs γ > 1, got $γ: the pressure (γ − 1) ρ ε and the " *
            "sound speed sqrt(γ p / ρ) are a positive pressure and a real " *
            "speed only where γ − 1 > 0, so an adiabatic index at or below " *
            "1 is not a gas this equation of state can describe."))
        return new{T}(γ)
    end
end

IdealGas(γ::Real) = IdealGas{float(typeof(γ))}(γ)

"""
    pressure(eos::IdealGas, ρ, ε)

The pressure of a gas of density `ρ` and specific internal energy `ε`:
`p = (γ − 1) ρ ε`.

One of the three functions an equation of state provides; see
[`EquationOfState`](@ref). It is the closure of the Euler system — the
equation that makes `(ρ, v, ε)` enough to write a flux — and the place a
different gas law would differ.
"""
@inline pressure(eos::IdealGas, ρ, ε) = (eos.γ - 1) * ρ * ε

"""
    internal_energy(eos::IdealGas, ρ, p)

The specific internal energy of a gas of density `ρ` at pressure `p`:
`ε = p / ((γ − 1) ρ)`, the inverse of [`pressure`](@ref).

[`prim2con`](@ref) is the caller: the conserved energy density is
`E = ρ ε + ½ ρ v²`, and the primitive state carries `p` rather than `ε`.
Both directions of the closure are part of the interface because both are
needed, and a tabulated equation of state inverts its table here.
"""
@inline internal_energy(eos::IdealGas, ρ, p) = p / ((eos.γ - 1) * ρ)

"""
    soundspeed(eos::IdealGas, ρ, p)

The adiabatic sound speed, `c_s = sqrt(γ p / ρ)`.

The third of the interface's functions, and the one the rest of the
scheme uses most: `|v_d| + c_s` is the signal speed that sets the time
step, and the wave-speed estimates of every HLL-family flux are built from
it. In GRMHD the same function returns the fast magnetosonic speed and
nothing above it changes.

It takes `p` rather than `ε` because both of its callers — the flux kernel
and the signal speed — hold a primitive state. Positivity of `ρ` and `p`
is the floors' business, not this function's: it is called on states that
have already been through [`apply_floors`](@ref), and would return a `NaN`
on one that had not.
"""
@inline soundspeed(eos::IdealGas, ρ, p) = sqrt(eos.γ * p / ρ)

# --- states ---------------------------------------------------------------
#
# A state is an `NTuple{D+2,T}` and carries its own `D` in its length, so
# every function below is generic in the number of dimensions without
# being told it. Index 1 is the density, indices `2 … D+1` are the `D`
# vector components and index `D+2` is the pressure or the total energy —
# the same positions in `P` and in `U` (decided; see "The equations" in
# `CODE.md`), so that a kernel reading "variable `1+d`" reads the
# `d`-component of whichever set it was handed.
#
# The accessors exist so that nothing downstream of here indexes a state
# by a literal number. A `P[3]` that means the pressure in 1D and the
# second velocity component in 2D is the kind of mistake that produces
# plots which look almost right, and it is invisible at the call site.

"""
    statedims(state)

The number of spatial dimensions a state tuple describes, as a
`Val{D}` — a state has `D + 2` entries, so this is `Val(length - 2)`.

A `Val` rather than an `Int` because its callers pass it straight to
`ntuple`, which needs the length at compile time to unroll; and it is
read off the tuple's *type*, so it costs nothing at run time.
"""
@inline statedims(::NTuple{M,Any}) where {M} = Val(M - 2)

"""
    density(state)

The density of a primitive or conserved state: index 1 of either.
"""
@inline density(state::NTuple) = state[1]

"""
    velocity(P)

The `D` velocity components of a primitive state, as an `NTuple{D}`:
indices `2 … D+1`.
"""
@inline velocity(P::NTuple{M,Any}) where {M} = ntuple(d -> P[1 + d], Val(M - 2))

"""
    momentum(U)

The `D` momentum density components of a conserved state, as an
`NTuple{D}`: indices `2 … D+1`.

The same indices [`velocity`](@ref) reads, which is the variable order's
whole purpose; the two names exist so that the call site says which set it
holds.
"""
@inline momentum(U::NTuple{M,Any}) where {M} = ntuple(d -> U[1 + d], Val(M - 2))

"""
    pressure_of(P)

The pressure of a primitive state: index `D + 2`, the last.

Named apart from [`pressure`](@ref), which is the equation of state's
function of `(ρ, ε)`. This one reads a slot; that one computes a closure.
"""
@inline pressure_of(P::NTuple{M,Any}) where {M} = P[M]

"""
    energy(U)

The total energy density of a conserved state: index `D + 2`, the last —
the slot [`pressure_of`](@ref) reads in a primitive one.
"""
@inline energy(U::NTuple{M,Any}) where {M} = U[M]

# `u ⋅ u` for a velocity or a momentum. `ntuple` over a `Val` and `sum`
# over the result, so it unrolls and holds no intermediate array.
@inline squarednorm(u::NTuple{D,Any}) where {D} =
    sum(ntuple(d -> u[d] * u[d], Val(D)))

"""
    prim2con(eos, P) -> U

The conserved state of a primitive one: `(ρ, v₁…v_D, p)` becomes
`(ρ, S₁…S_D, E)` with

    S_d = ρ v_d,     E = ρ ε + ½ ρ v²,     ε = internal_energy(eos, ρ, p)

Algebra, here and behind every equation of state; [`con2prim`](@ref) is
the direction that is hard, and only in GRMHD. This is
called wherever data enters the evolved state: once per cell on the
initial data, at every right-hand-side evaluation on the Dirichlet
boundary state, and on the way out of the atmosphere reset.

It cannot fail and does not floor, so it returns a state and not a pair.
A `P` that is not a physical state produces a `U` that is not one either;
[`apply_floors`](@ref) is what stands in front of it.
"""
@inline function prim2con(eos::EquationOfState, P::NTuple{M,Any}) where {M}
    ρ = density(P)
    v = velocity(P)
    p = pressure_of(P)
    ε = internal_energy(eos, ρ, p)
    S = ntuple(d -> ρ * v[d], statedims(P))
    E = ρ * ε + ρ * squarednorm(v) / 2
    return (ρ, S..., E)
end

"""
    con2prim(eos, floors, U) -> (P, hit)

The primitive state recovered from a conserved one, and whether a floor
fired doing it:

    v_d = S_d / ρ,     ε = (E − ½ S·S / ρ) / ρ,     p = pressure(eos, ρ, ε)

followed by [`apply_floors`](@ref).

**Why the signature is this shape.** In Newtonian hydrodynamics the
recovery is the closed-form algebra above and could be a one-line
function of `U` alone. It is written instead as a *fallible, flooring,
reporting* step — taking the floors, returning a flag — because that is
what it is in the code this package rehearses. In GRMHD the same three
arguments and the same two return values hide a one-dimensional root find
on the pressure, which does not always converge, and the recovery is
where most of a relativistic code's robustness lives. Keeping the
signature means the callers here — the `con2prim` pass over every stored
cell, the reconstruction's face states, the atmosphere reset — are already
written against the hard version. See "The equations" and "What has a
GRMHD counterpart" in `CODE.md`.

`hit` is `true` when either floor rule fired. It is not a failure: the
caller records it, per chunk and by population, and carries on. The
counts are a measurement the design depends on, not a diagnostic.

**A non-positive or `NaN` density never reaches the division.** The
atmosphere test comes first, before `S/ρ` is formed, so
`U = (0, S…, E)` returns the atmosphere state — finite, with a zero
velocity — rather than an `Inf` velocity or a `NaN` that would have to be
cleaned up afterwards. Every comparison in the floors is written as the
negation of the healthy condition for the same reason, so that a `NaN`
takes the flooring branch instead of passing through it; see
[`apply_floors`](@ref) for what that leaves and what it does not.
"""
@inline function con2prim(eos::EquationOfState, floors::Floors,
                          U::NTuple{M,Any}) where {M}
    ρ = density(U)
    # Before the division, not after it. This is the same rule
    # `apply_floors` applies below — and applies again, harmlessly, since
    # the recovered ρ is unchanged — but it has to be asked here, because
    # the recovery divides by ρ and the atmosphere is the guarantee that
    # it may.
    if in_atmosphere(floors, ρ)
        return atmosphere_state(floors, statedims(U)), true
    end
    S = momentum(U)
    E = energy(U)
    v = ntuple(d -> S[d] / ρ, statedims(U))
    ε = (E - squarednorm(S) / ρ / 2) / ρ
    p = pressure(eos, ρ, ε)
    return apply_floors(eos, floors, (ρ, v..., p))
end
