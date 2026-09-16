# The two rules that keep a cell inside the states the scheme can
# represent, and the three numbers they need.
#
# Every GRMHD code has these two rules and every one of them writes them
# down once, because they are the last line between an unphysical cell and
# a run that stops: a density that has fallen through zero is divided by
# in `con2prim`, and a non-positive pressure takes the sound speed
# `sqrt(γ p / ρ)` out of the reals. They are not a numerical method with a
# better alternative on offer; they are the definition of what the code
# does where the gas runs out. See "Floors and the atmosphere" in
# `CODE.md`.
#
# `apply_floors` is the whole of them. Three callers reach it: `con2prim`
# below, the reconstruction's face states (step 2), and the atmosphere
# reset of the conserved state inside the integrator's stage hook (step
# 8). One place the rules are written, three places they are applied.
#
# This file is included *before* `eos.jl` because `con2prim` takes a
# `Floors` and says so in its signature, and a signature is evaluated
# where the method is defined.

"""
    Floors{T}(ρ_atm, p_atm, p_floor)
    Floors{T}(; ρ_atm, p_atm, p_floor)
    Floors(; ρ_atm, p_atm, p_floor)

The density and pressure of the atmosphere, and the lowest pressure any
cell may hold: the case parameters of [`apply_floors`](@ref).

`ρ_atm` and `p_atm` are the state a cell *becomes* where the gas has run
out. They describe an atmosphere rather than a clamp, because the
velocity is zeroed along with them — a cell below `ρ_atm` is not a thin
cell, it is vacuum, and what it held is discarded. `p_floor` is the
clamp: where there *is* gas but the recovered pressure is not positive
enough to be a pressure, only the pressure changes.

**There is no default for any of the three.** What counts as vacuum is a
property of the problem — Sedov's ambient density is `1` and its
atmosphere sits orders of magnitude below it, while a star in a large box
wants another number entirely — so a default would be a physics decision
taken on the caller's behalf, which `CLAUDE.md` rules out for anything the
caller must think about.

All three must be positive, and `p_atm` must not fall below `p_floor`;
the constructor says why when they do not. The type is fixed rather than
promoted at each use: a `Floors` holds the working type `T` of the run,
the same `T` as the state it is applied to, so that a `Float32` run stays
at `Float32` (see "Precision" in `CODE.md`). The keyword form that does
not name `T` takes it from the arguments and floats it, so
`Floors(; ρ_atm = 1//10^6, …)` is a `Float64` set of floors and a run at
another type writes `T(1//10^6)` or names `T` itself.

`isbits`, so that it travels into a KernelAbstractions kernel as an
argument and a callback may capture it — the reason the floors are a
struct and not three keyword arguments threaded through every signature.
"""
struct Floors{T}
    ρ_atm::T
    p_atm::T
    p_floor::T

    function Floors{T}(ρ_atm, p_atm, p_floor) where {T}
        ρ_atm, p_atm, p_floor = convert(T, ρ_atm), convert(T, p_atm),
                                convert(T, p_floor)
        ρ_atm > 0 || throw(ArgumentError(
            "Floors needs a positive ρ_atm, got $ρ_atm: the atmosphere is " *
            "the density con2prim is allowed to divide by, and a " *
            "non-positive one would put a division by zero exactly where " *
            "the floor exists to prevent one."))
        p_atm > 0 || throw(ArgumentError(
            "Floors needs a positive p_atm, got $p_atm: the atmosphere is a " *
            "state of the gas like any other, and the sound speed " *
            "sqrt(γ p / ρ) is not a real number where p ≤ 0."))
        p_floor > 0 || throw(ArgumentError(
            "Floors needs a positive p_floor, got $p_floor: the floor's job " *
            "is to keep the internal energy p / ((γ − 1) ρ) positive, which " *
            "a floor at or below zero does not do."))
        p_atm ≥ p_floor || throw(ArgumentError(
            "Floors needs p_atm ≥ p_floor, got p_atm = $p_atm and p_floor " *
            "= $p_floor: the atmosphere is a state the floors must leave " *
            "alone, and an atmosphere below the pressure floor would be " *
            "floored again the next time apply_floors saw it — so applying " *
            "the floors twice would not equal applying them once."))
        return new{T}(ρ_atm, p_atm, p_floor)
    end
end

Floors{T}(; ρ_atm, p_atm, p_floor) where {T} = Floors{T}(ρ_atm, p_atm, p_floor)

function Floors(; ρ_atm, p_atm, p_floor)
    T = float(promote_type(typeof(ρ_atm), typeof(p_atm), typeof(p_floor)))
    return Floors{T}(ρ_atm, p_atm, p_floor)
end

"""
    in_atmosphere(floors, ρ)

Whether a cell of density `ρ` is vacuum as far as [`apply_floors`](@ref)
is concerned.

Written as `!(ρ ≥ ρ_atm)` and not as `ρ < ρ_atm`, which is the same
question for every number and a different one for a `NaN`: a `NaN` fails
*both* comparisons, so the first form sends it to the atmosphere and the
second would let it through as healthy gas. The floors are the last line;
a `NaN` is precisely what they exist to catch, and a comparison that
quietly says "not below the floor" is how it would get past them.

It is also the test that has to come *before* the division in
[`con2prim`](@ref), which is why it is a function of its own rather than
a line inside `apply_floors`: a non-positive `ρ` must reach the
atmosphere without `S/ρ` ever being formed.
"""
@inline in_atmosphere(floors::Floors, ρ) = !(ρ ≥ floors.ρ_atm)

"""
    atmosphere_state(floors, ::Val{D})

The primitive state of a vacuum cell in `D` dimensions:
`(ρ_atm, 0, …, 0, p_atm)`.

The zero velocity is the whole point of the atmosphere rule rather than a
detail of it. In a star's exterior — the configuration this package
rehearses — a cell that keeps whatever velocity its unphysical state
happened to carry sets `λ_max = max(|v| + c_s)` and therefore the time
step of the entire hierarchy, and it does so with a number that means
nothing. Discarding the momentum is what bounds the signal speed where
there is no gas to carry one.
"""
@inline atmosphere_state(floors::Floors{T}, ::Val{D}) where {T,D} =
    (floors.ρ_atm, ntuple(_ -> zero(T), Val(D))..., floors.p_atm)

"""
    apply_floors(eos, floors, P) -> (P′, hit)

The two floor rules applied to one primitive state `P = (ρ, v₁…v_D, p)`,
returning the state that came out and whether either rule fired.

    ρ < ρ_atm         →  (ρ_atm, 0, …, 0, p_atm)        the atmosphere
    p < p_floor       →  (ρ, v₁, …, v_D, p_floor)       the pressure floor
    otherwise         →  P unchanged

The **atmosphere** rule replaces the whole state, velocity included: below
`ρ_atm` the cell is vacuum, and what it held is numerical dust. Zeroing
the velocity is what keeps `|v| + c_s` — and with it the time step of the
whole hierarchy — bounded in a region where there is no gas to carry a
signal. It is a reset, not a clamp.

The **pressure floor** keeps `ρ` and `v` and changes only `p`, so in the
conserved variables only `E` moves: there *is* gas here, and its momentum
is meaningful, but the internal energy that came out of the recovery is
not. For an ideal gas `p = (γ − 1) ρ ε` with `γ > 1` and `ρ > 0`, so a
non-positive internal energy is a non-positive pressure and this one rule
covers both — which is why the rule is stated on `p` and not on `ε`.

`hit` is what the floor counts are made of. The driver counts hits per
chunk in two populations, owned cells and ghost cells, because the ghost
count is the measurement that decides an open upstream question about the
prolongation (see "Floors and the atmosphere" in `CODE.md`). A rule that
fired silently would make that question unanswerable.

Both comparisons are written as the negation of the healthy condition
(`!(ρ ≥ ρ_atm)`, `!(p ≥ p_floor)`), so a `NaN` takes the flooring branch
instead of escaping through it; see [`in_atmosphere`](@ref). A `NaN` in
`ρ` therefore yields the atmosphere state exactly, which is finite. A
`NaN` that reaches only `p` yields `hit = true` and `p = p_floor` while
`ρ` and `v` are returned as they came: the two rules are rules about `ρ`
and `p`, and a `NaN` *velocity* with a healthy density means the state
was already broken before the floors saw it. The flag reports that; it is
not silently repaired, because repairing it would make a broken state
look like a floored one.

`eos` is not read for an ideal gas, and is in the signature because a
tabulated or hybrid EOS floors against the bounds of its own table and
needs it. The floors are a property of the case, the EOS a property of
the gas, and both travel together everywhere else in the package.

Pure, pointwise and `isbits` in and out, so it is callable from inside a
kernel and identical on every backend and at every thread count. The
*reset* of the conserved state `U` that uses this — `con2prim`,
`apply_floors`, `prim2con`, written back from the integrator's stage hook
— is step 8; this function is only the rules.
"""
@inline function apply_floors(eos, floors::Floors, P::NTuple{M}) where {M}
    ρ = density(P)
    if in_atmosphere(floors, ρ)
        return atmosphere_state(floors, statedims(P)), true
    end
    p = pressure_of(P)
    # Negated for the reason `in_atmosphere` is: a NaN pressure — which is
    # what a NaN anywhere in a conserved state with a healthy density
    # arrives as — must be floored and flagged, not passed through.
    if !(p ≥ floors.p_floor)
        return (ρ, velocity(P)..., floors.p_floor), true
    end
    return P, false
end
