# Piecewise-linear reconstruction of the primitive variables: the three
# slope limiters and the pair of face states they build.
#
# This is the first half of the flux kernel of step 3. At face `i` of a
# block — the face between cells `i−1` and `i` — that kernel reads the
# four primitive states `P_{i−2} … P_{i+1}`, calls `face_states` on them
# and hands the pair to a Riemann solver from `riemann.jl`. Nothing here
# knows about the mesh; everything is arithmetic on four `isbits` tuples,
# so the same functions serve the host loop, a CPU kernel and a device.
# See "Reconstruction" in `CODE.md`.
#
# Two properties this file is written to keep, both of which the flux
# kernel inherits:
#
#   - **The limiter is a `Val`, never a `Symbol`.** A `Val{:mc}` is a
#     singleton type, so the kernel specializes on it, the branch between
#     the limiters disappears at compile time, and the argument is
#     `isbits` and may be captured or passed to a device. A `Symbol`
#     argument would be neither.
#   - **Integer literals only.** `(a + b) / 2`, never `0.5 * (a + b)`: a
#     decimal literal is an `Float64` operand in the innermost loop of the
#     scheme, which widens a `Float32` run and needs hardware `Float64` on
#     a device. See "Precision" in `CODE.md`.

"""
    slope(::Val{:none},   a, b)
    slope(::Val{:minmod}, a, b)
    slope(::Val{:mc},     a, b)

The limited slope of one variable in one cell, from that cell's two
one-sided differences `a = P_i − P_{i−1}` and `b = P_{i+1} − P_i`. The
result is the *full* rise across the cell, so the face values are
`P_i ± σ/2`.

Three limiters exist, and each is here for a reason the others cannot
serve:

  - **`:none`** — the plain centered slope `(a + b)/2`. Not a limiter at
    all: the reconstruction stays linear everywhere, which keeps the
    truncation error a clean `O(h²)`, and that is what a convergence
    study needs. A limiter clips at a smooth extremum and would hide the
    scheme's order — and, at a coarse-fine face, the interface's own
    contribution — behind its own first-order footprint. It does not
    preserve monotonicity, so a face state it builds can leave the range
    of the two cells it came from; [`face_states`](@ref) floors them.
  - **`:minmod`** — zero where the two differences disagree in sign, and
    otherwise the one of smaller magnitude. The most diffusive of the
    total-variation-diminishing limiters and the robust default for
    shocks: it is the one that will not overshoot.
  - **`:mc`** — monotonized central, `minmod(2a, 2b, (a + b)/2)`: zero
    where `a b ≤ 0`, and otherwise the smallest magnitude of the three,
    which all share the sign of `a`. It is the centered slope wherever
    that is not steeper than twice either one-sided difference, so it is
    markedly less diffusive than `:minmod` while still TVD. The usual
    default of a GRMHD code, and what the Kelvin–Helmholtz rolls of H5
    want: their shear layer is a contact, and a contact is what a
    diffusive limiter smears fastest.

**There is no fourth.** A symbol that is not one of these three has no
method and raises a `MethodError` where the kernel specializes, which is
at compile time rather than in the middle of a run — the cheapest form of
"refuse it" for an argument whose whole purpose is to be a compile-time
constant.

All three are symmetric in their arguments and odd under negation:
`slope(a, b) = slope(b, a)` and `slope(−a, −b) = −slope(a, b)`. The
second is why a reconstruction of a profile and of its negative agree, and
the first is why the face between two cells does not depend on which side
asked for it.

See "Reconstruction" in `CODE.md`.
"""
@inline slope(::Val{:none}, a, b) = (a + b) / 2

@inline function slope(::Val{:minmod}, a, b)
    z = zero(a)
    # `a * b ≤ 0` rather than a comparison of signs: it is one
    # multiplication, it is exact about which side of zero the product is
    # on for the purpose of this test, and it needs no `sign` — which, for
    # a software float, is a function that may not exist.
    a * b ≤ z && return z
    return abs(a) < abs(b) ? a : b
end

# Nested rather than written out, because `minmod` of three arguments *is*
# the two-argument one applied twice: where `a b > 0` all three candidates
# carry the sign of `a`, so the smaller-magnitude rule composes; and where
# `a b ≤ 0` the inner call already returns zero, which the outer one then
# passes through. The `:mc` limiter's "zero at an extremum" rule therefore
# comes from the same line that gives `:minmod` its own.
@inline slope(::Val{:mc}, a, b) =
    slope(Val(:minmod), slope(Val(:minmod), 2 * a, 2 * b), (a + b) / 2)

"""
    face_states(lim, eos, floors, P₋₂, P₋₁, P₀, P₊₁) -> (P_L, P_R)

The two primitive states on either side of the face between cells `i−1`
and `i`, reconstructed from the four cell states that face reads:

    σ₋₁ = slope(lim, P₋₁ − P₋₂, P₀ − P₋₁)     the slope in cell i−1
    σ₀  = slope(lim, P₀ − P₋₁, P₊₁ − P₀)      the slope in cell i
    P_L = P₋₁ + σ₋₁/2                          extrapolated forward
    P_R = P₀  − σ₀/2                           extrapolated backward

componentwise over all `D + 2` primitives, followed by
[`apply_floors`](@ref) on each. `lim` is a [`slope`](@ref) limiter as a
`Val`; the four states are `NTuple{D+2}`s in the order `(ρ, v₁…v_D, p)`
and `D` is read off their length, so one method serves every dimension.

**Primitives, not conserved variables** (decided; see "Reconstruction" in
`CODE.md`). Reconstructing `ρ`, `S` and `E` separately produces pressure
oscillations at a contact, where `ρ` jumps and `p` does not: the recovered
pressure is a difference of two reconstructions that were limited
independently. In GRMHD the same choice also decides where the primitive
recovery runs — per cell, as here, or per *face state*, which is twice the
root finds and puts their failures in the hardest place to handle them.
Characteristic variables are rejected for the GRMHD reason: the
eigenvectors are expensive there and nobody uses them.

**The floor hits are discarded here, and that is deliberate.** Only the
floored states are returned; no count is kept, and nothing downstream adds
one. Under a TVD limiter (`:minmod`, `:mc`) they cannot fire on physical
cell states at all — the face value lies between the two neighbouring cell
values, so a positive `ρ` and `p` on both sides give a positive `ρ` and
`p` at the face — and under `:none` they can, because the centered slope
overshoots. The floors are there for that case, and for a stencil that was
already unphysical before the reconstruction saw it. What the driver
counts, per chunk and in two populations, are the hits in *cells*: owned
cells from the atmosphere reset, ghost cells from the `con2prim` pass.
Those two are a measurement the design depends on — the ghost count
decides an open upstream question about the prolongation — and a third
count mixing in face states, which are not cells and are recomputed at
every stage, would only blur it. See "Floors and the atmosphere" in
`CODE.md`.

Pure, pointwise, `isbits` in and out, and the argument order puts `lim`
first so that the kernel's `Val` is the value the method specializes on.
"""
@inline function face_states(lim::Val, eos::EquationOfState, floors::Floors,
                             P₋₂::NTuple{M,Any}, P₋₁::NTuple{M,Any},
                             P₀::NTuple{M,Any}, P₊₁::NTuple{M,Any}) where {M}
    P_L = ntuple(v -> P₋₁[v] + slope(lim, P₋₁[v] - P₋₂[v], P₀[v] - P₋₁[v]) / 2,
                 Val(M))
    P_R = ntuple(v -> P₀[v] - slope(lim, P₀[v] - P₋₁[v], P₊₁[v] - P₀[v]) / 2,
                 Val(M))
    # `first` and not the pair: the flags are not counted anywhere, for the
    # reason the docstring gives.
    return first(apply_floors(eos, floors, P_L)),
           first(apply_floors(eos, floors, P_R))
end
