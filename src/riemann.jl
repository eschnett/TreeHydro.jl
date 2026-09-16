# The physical flux, the signal speed, and the three approximate Riemann
# solvers that turn a pair of face states into the flux through the face
# between them.
#
# This is the second half of the flux kernel of step 3: `face_states` in
# `reconstruction.jl` builds `P_L` and `P_R`, `riemann_flux` here returns
# the `D + 2` numbers written into the face-centered flux set. Nothing in
# this file knows about the mesh either; it is arithmetic on `isbits`
# tuples. See "Riemann solver" in `CODE.md`.
#
# Three things govern the shape of everything below:
#
#   - **The direction is a `Val{d}`**, and the velocity an `NTuple{D}`, so
#     one method serves every direction and every number of dimensions.
#     The only place `d` appears is as an index into that tuple and as the
#     slot the pressure is added to; there is no special case for `D = 1`,
#     and `Base.setindex` is how a single component of a tuple is
#     replaced.
#   - **The solver is a `Val` of a symbol**, like the limiter, so the
#     kernel specializes on it and the branch between the three
#     disappears at compile time.
#   - **Integer literals only**, for the reason `reconstruction.jl` gives.
#
# The solvers call `prim2con` and `physical_flux` on the same face state,
# and `physical_flux` calls `prim2con` itself. That is one repeated
# conversion on paper and none in the generated code: both are `@inline`
# and pure arithmetic, so the common subexpression survives exactly once.
# Writing it out by hand would save nothing and would put the variable
# order in a second place.

"""
    physical_flux(eos, P, ::Val{d}) -> F

The exact Euler flux in direction `d` of the primitive state `P`, in the
order of the conserved variables:

    F = (ρ v_d,  S_i v_d + p δ_id,  (E + p) v_d)

with `S` and `E` taken from [`prim2con`](@ref), so that the flux and the
state it is a flux *of* cannot disagree about the variable order.

This is what every approximate Riemann solver below is built from, and it
is what they all reduce to when the two face states are equal — the
consistency property, which is the sharpest single test of a flux
function. It is also the exact answer on either side of a supersonic
face, which is what the upwinding branches of HLLE and HLLC return.

The pressure enters one slot, `1 + d`, and the tangential momenta are
merely advected; that is the whole of the direction dependence. See
"Riemann solver" in `CODE.md`.
"""
@inline function physical_flux(eos::EquationOfState, P::NTuple{M,Any},
                               ::Val{d}) where {M,d}
    p = pressure_of(P)
    U = prim2con(eos, P)
    S = momentum(U)
    vd = velocity(P)[d]
    z = zero(p)
    F_S = ntuple(i -> S[i] * vd + (i == d ? p : z), statedims(P))
    return (density(U) * vd, F_S..., (energy(U) + p) * vd)
end

"""
    signal_speed(eos, P)

The fastest signal the cell in state `P` can carry, `max_d (|v_d| + c_s)`.

The per-cell number that sets the time step of the entire hierarchy:
`λ_max` is its maximum over every cell, and `dt = cfl · h / (D λ_max)`.
The `con2prim` kernel of step 3 writes it into a diagnostic slot of `P`
as it recovers each cell, because `block_mapreduce` maps a scalar function
over *one* variable's values and cannot form `|v| + c_s` from three of
them; the kernel that already holds all of them writes the number once,
and the reduction is then a plain one over that slot. See "Time
integration and the time step" in `CODE.md`.

It is the maximum over directions rather than `|v| + c_s`, because an
unsplit scheme's stability condition is a sum over directions of
per-direction speeds and this bounds each of them.
"""
@inline function signal_speed(eos::EquationOfState, P::NTuple)
    v = velocity(P)
    c = soundspeed(eos, density(P), pressure_of(P))
    return maximum(ntuple(d -> abs(v[d]) + c, statedims(P)))
end

# The two numbers every wave-speed estimate below is built from: the
# velocity component normal to the face and the sound speed. The
# tangential components never enter a speed — they are advected, not
# propagated — which is exactly why one function serves every direction.
@inline function normal_waves(eos::EquationOfState, P::NTuple, ::Val{d}) where {d}
    c = soundspeed(eos, density(P), pressure_of(P))
    return velocity(P)[d], c
end

# Davis's two-wave estimate, shared by HLLE and HLLC: the slowest and the
# fastest signal either state can carry across the face.
#
# `s_R − s_L > 0` always, and that is a fact about the floors rather than
# about this function: `s_R − s_L ≥ (v_L + c_L) − (v_L − c_L) = 2 c_L > 0`
# because `c_s = sqrt(γ p / ρ)` is strictly positive wherever `ρ ≥ ρ_atm`
# and `p ≥ p_floor`, which is what `apply_floors` guarantees of every state
# that reaches here. So the HLL average below never divides by zero.
@inline function davis_speeds(eos::EquationOfState, P_L::NTuple, P_R::NTuple,
                              ::Val{d}) where {d}
    v_L, c_L = normal_waves(eos, P_L, Val(d))
    v_R, c_R = normal_waves(eos, P_R, Val(d))
    return min(v_L - c_L, v_R - c_R), max(v_L + c_L, v_R + c_R)
end

"""
    riemann_flux(::Val{:llf},  eos, P_L, P_R, ::Val{d}) -> F
    riemann_flux(::Val{:hlle}, eos, P_L, P_R, ::Val{d}) -> F
    riemann_flux(::Val{:hllc}, eos, P_L, P_R, ::Val{d}) -> F

The numerical flux in direction `d` through a face with primitive states
`P_L` on its left and `P_R` on its right, as an `NTuple{D+2}` in the order
of the conserved variables.

All three are *consistent*: with `P_L == P_R` each returns
[`physical_flux`](@ref) of that state, to roundoff. All three need nothing
but the two primitive states and the equation of state — no eigenvectors,
no iteration — which is what makes them the fluxes a relativistic code can
use. The solver travels as a `Val` so the flux kernel specializes on it.

**`:llf`, local Lax–Friedrichs (Rusanov).**

    F = (F_L + F_R)/2 − λ (U_R − U_L)/2,   λ = max(|v_L| + c_L, |v_R| + c_R)

One wave speed, the fastest either side can carry, applied to every
characteristic field. The simplest and the most diffusive of the three,
and the fallback every GRMHD code keeps for the cells where something
better fails.

**`:hlle`** (the default). Two waves, with Davis's speed estimates
`s_L = min(v_L − c_L, v_R − c_R)` and `s_R = max(v_L + c_L, v_R + c_R)`:

    F = F_L                                           if s_L ≥ 0
    F = F_R                                           if s_R ≤ 0
    F = (s_R F_L − s_L F_R + s_L s_R (U_R − U_L)) / (s_R − s_L)   otherwise

This is *the* GRMHD flux, and the reason is the middle line of that
estimate: HLLE asks the physics for the fastest and the slowest signal
speed and for nothing else. In GRMHD those two are the fast magnetosonic
speeds, which a code computes anyway; every solver that needs more — the
intermediate eigenvalues, the eigenvectors, a decomposition — is a
solver that gets harder as the equations do. The division is safe:
`s_R − s_L > 0` follows from `c_s > 0`, which the floors guarantee (see
[`apply_floors`](@ref)). The price is the contact wave, which the
two-wave average smears: with `s_L < 0 < s_R` a stationary contact still
gets a nonzero mass flux.

**`:hllc`.** Toro's three-wave solver (*Riemann Solvers and Numerical
Methods for Fluid Dynamics*, 3rd ed., §10.4), with the same Davis speeds
and a middle wave at

    s★ = (p_R − p_L + ρ_L v_L (s_L − v_L) − ρ_R v_R (s_R − v_R))
         / (ρ_L (s_L − v_L) − ρ_R (s_R − v_R))

For each side `K ∈ {L, R}` the star state is Toro's (10.39),

    U★_K = ρ_K (s_K − v_K)/(s_K − s★) ·
           (1, s★ in the normal slot and v_K in the tangential ones,
            E_K/ρ_K + (s★ − v_K)(s★ + p_K/(ρ_K (s_K − v_K))))

and the flux through it is `F★_K = F_K + s_K (U★_K − U_K)`; the face flux
is `F_L`, `F★_L`, `F★_R` or `F_R` according to which of `s_L`, `s★`, `s_R`
is the first that is not negative. It restores the contact HLLE smears —
`s★` *is* the contact speed, and a stationary contact comes out exactly —
which is why it exists at all. It has a GRHD counterpart (Mignone & Bodo
2005) and an MHD one (HLLD), so it is admissible here; but it is **the
comparison, not the baseline** (decided): HLLE stays the default, and
what HLLC buys is measured on the Kelvin–Helmholtz instability in step 10,
where the shear layer is a contact and the diffusion shows.

See "Riemann solver" in `CODE.md` for why these three and no others.
"""
@inline function riemann_flux(::Val{:llf}, eos::EquationOfState,
                              P_L::NTuple{M,Any}, P_R::NTuple{M,Any},
                              ::Val{d}) where {M,d}
    v_L, c_L = normal_waves(eos, P_L, Val(d))
    v_R, c_R = normal_waves(eos, P_R, Val(d))
    λ = max(abs(v_L) + c_L, abs(v_R) + c_R)
    F_L = physical_flux(eos, P_L, Val(d))
    F_R = physical_flux(eos, P_R, Val(d))
    U_L = prim2con(eos, P_L)
    U_R = prim2con(eos, P_R)
    return ntuple(v -> (F_L[v] + F_R[v] - λ * (U_R[v] - U_L[v])) / 2, Val(M))
end

@inline function riemann_flux(::Val{:hlle}, eos::EquationOfState,
                              P_L::NTuple{M,Any}, P_R::NTuple{M,Any},
                              ::Val{d}) where {M,d}
    s_L, s_R = davis_speeds(eos, P_L, P_R, Val(d))
    z = zero(s_L)
    F_L = physical_flux(eos, P_L, Val(d))
    # Supersonic toward the right: the face sees the left state alone, and
    # the answer is its exact flux — not an average that happens to equal
    # it.
    s_L ≥ z && return F_L
    F_R = physical_flux(eos, P_R, Val(d))
    s_R ≤ z && return F_R
    U_L = prim2con(eos, P_L)
    U_R = prim2con(eos, P_R)
    return ntuple(v -> (s_R * F_L[v] - s_L * F_R[v] +
                        s_L * s_R * (U_R[v] - U_L[v])) / (s_R - s_L), Val(M))
end

# One side's star state and the flux through it: Toro's (10.39) and
# (10.38). `w = ρ (s − v_d)` is the mass flux through the outer wave as
# seen in its own frame, and it appears three times — as the numerator of
# the star density, as the `p/(ρ(s − v))` of the star energy, and (with the
# other side's) in `s★` itself — so it is formed once.
#
# `w` is never zero: `s_L ≤ v_L − c_L` and `s_R ≥ v_R + c_R`, so
# `s − v_d` is at most `−c_s` on the left and at least `c_s` on the right,
# and `c_s > 0`. Neither is `s − s★`, in the branches that call this: the
# cascade in `riemann_flux` reaches the left star state only when
# `s_L < 0 ≤ s★` and the right one only when `s★ < 0 ≤ s_R`, so the
# difference it divides by straddles zero in both cases.
@inline function hllc_star_flux(P::NTuple{M,Any}, U::NTuple{M,Any},
                                F::NTuple{M,Any}, s, s★, ::Val{d}) where {M,d}
    ρ = density(P)
    v = velocity(P)
    p = pressure_of(P)
    w = ρ * (s - v[d])
    ρ★ = w / (s - s★)                       # the star state's density
    v★ = Base.setindex(v, s★, d)            # normal slot moves, tangential stay
    e★ = energy(U) / ρ + (s★ - v[d]) * (s★ + p / w)
    U★ = (ρ★, ntuple(i -> ρ★ * v★[i], statedims(P))..., ρ★ * e★)
    return ntuple(i -> F[i] + s * (U★[i] - U[i]), Val(M))
end

@inline function riemann_flux(::Val{:hllc}, eos::EquationOfState,
                              P_L::NTuple{M,Any}, P_R::NTuple{M,Any},
                              ::Val{d}) where {M,d}
    v_L = velocity(P_L)[d]
    v_R = velocity(P_R)[d]
    s_L, s_R = davis_speeds(eos, P_L, P_R, Val(d))
    z = zero(s_L)
    F_L = physical_flux(eos, P_L, Val(d))
    s_L ≥ z && return F_L

    w_L = density(P_L) * (s_L - v_L)
    w_R = density(P_R) * (s_R - v_R)
    # `w_L < 0 < w_R`, so the denominator is strictly negative and the
    # contact speed is always defined.
    s★ = (pressure_of(P_R) - pressure_of(P_L) + w_L * v_L - w_R * v_R) /
         (w_L - w_R)

    U_L = prim2con(eos, P_L)
    s★ ≥ z && return hllc_star_flux(P_L, U_L, F_L, s_L, s★, Val(d))

    F_R = physical_flux(eos, P_R, Val(d))
    U_R = prim2con(eos, P_R)
    s_R ≥ z && return hllc_star_flux(P_R, U_R, F_R, s_R, s★, Val(d))
    return F_R
end
