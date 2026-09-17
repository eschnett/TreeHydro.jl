# TreeHydro.jl — Design

TreeHydro.jl solves the equations of Newtonian ideal hydrodynamics with a
high-resolution shock-capturing (HRSC) finite-volume scheme on an
adaptively refined mesh. It is the second worked example for
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl), and the
*conservative* one. [TreeWave](https://github.com/eschnett/TreeWave.jl)
shows the mesh under a second-order finite-difference scheme that needs
nothing special at coarse-fine faces; this package shows it under a
finite-volume scheme that needs everything TreeAMR's M8 milestone added —
the conservative operator family, per-field-set ghost widths, ghost-free
face-centered flux fields, and the interface flux restriction that makes
the scheme conserve across refinement boundaries.

*Status: milestones H0 (scaffolding), H1 (the scheme on a uniform mesh) and
H2 (coarse-fine faces on a static mesh) done — the equation of state, the
two state conversions and the floors (step 1), the reconstruction and the
three Riemann fluxes (step 2), the six-step right-hand side, the time
integration and the entropy wave (step 3), the exact Riemann solver, Sod's
shock tube and the Dirichlet boundary hook (step 4), and the static
two-level mesh (step 5).* The measured numbers are in
[Measured results](#measured-results): the scheme's order on smooth data,
its L1 rate against the exact Riemann solution, the conservation of all
`D + 2` integrals with the interface fixup and the ten-order leak without
it, the interface-order table for the system, and the boundary flux that
replaces the conservation claim where a boundary is physical. Everything
from the refinement criterion on is still unmeasured.
Markers: **(decided)** is a decision taken in review; **(proposed)** is
one this document makes and still wants confirmed; **(predicted)** is a
number a milestone will measure and the "Measured results" section will
then record, as TreeAMR's and TreeWave's `CODE.md` do; **(open)** points
at [Open questions](#open-questions). Two things this package needs from
TreeAMR before its first milestone are under
[Upstream prerequisites](#upstream-prerequisites).

## Goals

- **Exercise M8 end to end**, from a real downstream package using the
  public API only: a cell-centered state with `G = 2` and `Conservative`
  operators; `D` face-centered flux sets with `G = 0`;
  `InterfaceSchedule` / `restrict_interfaces!`; `regrid!` over
  `fs => schedule` pairs, including `fs => nothing` for the sets that are
  merely resized; and the physical-boundary hook, which no downstream
  code has used yet (TreeWave is periodic throughout). The particular
  things to stress, from `TODO.md`: regridding, and conservation across
  refinement boundaries.
- **Rehearse a relativistic MHD code.** The eventual goal (not of this
  package) is GRMHD, so every numerical method here is chosen to have a
  direct GRMHD counterpart, and methods that only work for Newtonian
  hydro are avoided even where they would be better here. The table under
  [What has a GRMHD counterpart](#what-has-a-grmhd-counterpart-and-what-is-deliberately-not-used)
  is the checklist.
- **Be small.** One equation system, one scheme with a handful of
  switches, four initial conditions, one time-stepping driver.
- **Run in the caller's floating-point type, on the caller's backend, on
  any thread count with bit-identical results**, as TreeWave does and
  for the same reasons. Nothing here is new in that respect; it is
  inherited discipline.
- **Produce a picture worth looking at.** The 2D Kelvin–Helmholtz
  instability is the case that exists for this, alongside what it
  measures.

## Scope and non-goals

- **Newtonian Euler equations, ideal-gas equation of state.** No
  gravity, no source terms, no viscosity, no cooling. The RHS keeps the
  *slot* where source terms go, because GRMHD has geometric sources and
  the structure `∂ₜU = −∇·F + S` should not need rearranging later; the
  slot is empty here.
- **No MHD.** Constrained-transport MHD is the next package, not a
  later milestone of this one. It needs two things TreeAMR has specified
  but not implemented — a state vector spanning several field sets (the
  cell-centered hydro state *and* the face-centered `B`), and a
  divergence-preserving prolongation of face fields, which is not a
  tensor product and is an application operator. TreeHydro is deliberately
  the package that does *not* need either: one evolved field set, one
  schedule, so that the M8 features are exercised without those two open
  items in the way. What TreeHydro settles — primitive recovery as an
  explicit fallible step, the atmosphere reset, HLL-family fluxes,
  primitive reconstruction, the chunked regrid driver — is what the MHD
  package inherits unchanged.
- **Second-order reconstruction.** Piecewise-linear MUSCL with a
  limiter, which is `G = 2`. Higher order (PPM, WENO) is a listed
  extension with a known price (`G = 3` and the interface-order rule
  re-measured), not a milestone.
- **No mesh machinery.** Anything about trees, ghosts, or interpolation
  belongs upstream. The one place this package is likely to *want*
  something from the mesh — a limited, positivity-preserving
  prolongation of hydro data — is recorded as an upstream question
  rather than built here (see [Operator order](#operator-order)).
- **Not a production code.** No checkpointing, no HDF5, no unit system.
  Output is what the viewer in `bin/` needs.
- **Dirichlet physical boundaries only.** They are what the
  device-capable `CellBoundary` hook can express, and — set to the
  initial state — they are exact for a shock tube or a blast wave until a
  wave reaches the boundary, which the driver checks in advance.
  Reflecting and outflow conditions read the interior and are CPU-only
  upstream; nothing here needs them.

## Upstream prerequisites

Two things this package needs from TreeAMR before H1, both decided in
review and both about the storage rather than the physics, which is why
they go upstream instead of being spelled out here:

1. **`map_blocks!(kernel!, fs, …; stored = true)`** — a launch over every
   *stored* point of every block, ghosts included, beside the existing
   owned and closed ranges. The consumer is the `con2prim` pass of step
   (2) of the right-hand side: the reconstruction reads primitives two
   cells into the neighbours, and with the exchange in conserved
   variables those primitives have to be recovered in the ghost cells. A
   pointwise kernel over the stored extent is a statement about the
   layout, not about hydrodynamics.
2. **An all-variables form of the coordinate callbacks** —
   `fill_by_coordinates!`, `adapt_to_initial_data!`'s `initial`, and
   `CellBoundary` / `boundary_by_coordinates` — called once per point and
   returning every variable at once, beside the present
   `(x, v) -> value` form. The consumer is `prim2con`: an initial or
   boundary state is a primitive state converted to a conserved one, and
   the conversion needs all the primitives at once. Per variable it would
   be evaluated `D + 2` times per cell, and in the boundary hook's case
   at every RHS evaluation.

`m8` has since been merged, so both land on TreeAMR's `main`, which is
what the `[sources]` entry pins.

## The equations

The Euler equations in `D` dimensions, in conservation form, for the
**conserved variables** `U = (ρ, S₁, …, S_D, E)` — mass density,
momentum density, total energy density — with `nvars = D + 2`:

    ∂ₜ ρ   + ∂_d (ρ v_d)             = 0
    ∂ₜ S_i + ∂_d (S_i v_d + p δ_id)  = 0
    ∂ₜ E   + ∂_d ((E + p) v_d)       = 0

The **primitive variables** are `P = (ρ, v₁, …, v_D, p)`. The two are
related through the equation of state, here the ideal gas with adiabatic
index `γ`:

    p = (γ − 1) ρ ε,     E = ρ ε + ½ ρ v²,     c_s² = γ p / ρ

**Variable order** (decided): index 1 is `ρ`, indices `2 … D+1` are the
`D` momentum (velocity) components, index `D+2` is `E` (`p`). The same
positions in `U` and `P`, so that a kernel reading "variable `1+d`" reads
the `d`-component of whichever set it was handed.

**`prim2con`** is algebraic. **`con2prim`** is algebraic here too —
`v = S/ρ`, `ε = (E − ½ S²/ρ)/ρ`, `p = (γ−1) ρ ε` — but it is written
as what it is in GRMHD: a function `con2prim(eos, floors, U) → (P, ok)`
that may *fail* (negative density or internal energy) and may *floor*
(see [Floors and the atmosphere](#floors-and-the-atmosphere)), returning
whether it did. The Newtonian case
never iterates; the GRMHD case runs a one-dimensional root find behind
the same signature. Keeping the signature is the point.

**The equation of state is a struct**, `IdealGas(γ)`, behind a small
interface — `pressure(eos, ρ, ε)`, `internal_energy(eos, ρ, p)`,
`soundspeed(eos, ρ, p)` — so that a hybrid or tabulated EOS is a new
struct and nothing else. `γ` is stored in the working type `T`, and is a
case parameter: Sod uses `7/5`, Kelvin–Helmholtz uses `5/3` (following
the setups they are compared against).

**(Implemented in step 1.)** What writing `src/eos.jl` and
`src/floors.jl` settled, beyond the above:

- **A state is an `NTuple{D+2,T}` and nothing indexes one by a literal.**
  `density`, `velocity`, `momentum`, `pressure_of` and `energy` are the
  accessors, and `statedims` is the `Val{D}` read off the tuple's length,
  so every function is generic in `D` without being told it. A `P[3]`
  that means the pressure in 1D and the second velocity component in 2D
  is the mistake this removes; it is invisible at the call site.
- **A non-positive or `NaN` density never reaches the division.** The
  atmosphere condition is tested in `con2prim` *before* `S/ρ` is formed,
  so `ρ ≤ 0` returns the atmosphere state rather than an `Inf` velocity
  that a later comparison could not undo.
- **Every floor comparison is the negation of the healthy condition** —
  `!(ρ ≥ ρ_atm)`, `!(p ≥ p_floor)` — because a `NaN` fails `<` and `≥`
  alike, and the natural spelling would report a state full of `NaN`s as
  needing no floor. The consequence is a claim worth having: `hit` is
  true for *any* `U` that is not a finite physical state. A `NaN` in `ρ`
  yields the atmosphere state, which is finite; a `NaN` that reaches only
  `p` yields `hit = true` and `p = p_floor` with `ρ` and `v` as they
  came, since the two rules are rules about `ρ` and `p` and a repaired
  velocity would make a broken state indistinguishable from a floored
  one.
- **`Floors` refuses `p_atm < p_floor`**, beside the three positivity
  checks, because the atmosphere state would otherwise trip the pressure
  floor and `apply_floors` would not be idempotent — the property the
  stage reset of H4 rests on. With the check, idempotence holds *bit for
  bit* rather than to roundoff: both rules select a state rather than
  computing one.
- **The recovery's cancellation is real, and it is not the floors'.**
  `ε = (E − ½ S·S/ρ)/ρ` loses relative accuracy in proportion to
  `½ρv²/(ρε)`, so a pressure floor orders of magnitude below the kinetic
  energy survives a round trip through `U` only to that ratio, and
  whether a state sitting exactly *on* `p_floor` trips it again is a
  question about the last ulp. Idempotence is therefore claimed on
  `apply_floors`, where it is exact, and the round trip is claimed on the
  values and not on the flag.

## The scheme

A dimensionally unsplit, second-order, method-of-lines finite-volume
scheme: MUSCL reconstruction of primitives, an HLL-family approximate
Riemann solver, strong-stability-preserving Runge–Kutta in time. This is
the standard skeleton of every production GRMHD code, and that, not its
merits for Sod's problem, is why it is the choice.

### Field sets

Three kinds of field set over one forest (decided):

| set | centering | `nvars` | `G` | in the state vector | at a regrid |
|---|---|---|---|---|---|
| `U`, conserved | cell | `D+2` | `2` | yes | `U => schedule` (Conservative, `p = 3`) |
| `P`, primitive | cell | `D+4` | `2` | no | `P => nothing` (resized; recomputed by the next RHS) |
| `F_d`, fluxes, `d = 1 … D` | `facecentered(D, d)` | `D+2` | `0` | no | `F_d => nothing` |

**`P` carries two diagnostic slots beyond the `D + 2` primitives**
(amended in step 3; it was `nvars = D+2` when this table was first
written). Slot `D+3` is the cell's signal speed `max_d (|v_d| + c_s)` and
slot `D+4` its floor-hit flag, `1` or `0`; the `con2prim` kernel writes
both as it recovers each cell, over the stored extent like everything else
it writes. They are there because `block_mapreduce` maps a *scalar*
function over *one* variable's values and so cannot form `|v_d| + c_s`
from three of them — the reduction would need the velocity and the sound
speed of the same cell at once, which its signature does not offer. The
kernel that already holds all `D + 2` primitives writes the number once,
after which `λ_max` and the owned-cell floor count are plain
`block_mapreduce` calls over one slot each, deterministic upstream and
with no second pass over the data. The alternative — a reduction kernel of
this package's own over the stored extent — would be mesh machinery
written downstream, which `CLAUDE.md` rules out. (The *ghost*-cell floor
count does need the stored extent, which `block_mapreduce` does not cover;
that one is a one-item-per-block kernel summing slot `D+4`, and it arrives
with the atmosphere reset.)

`U` is the **only evolved set**, so the state vector is `statevector(U)`
and the several-set form TreeAMR has specified is not needed. `G = 2` on
`U` is the same-level conservation obligation from TreeAMR's
"Conservation at coarse-fine faces": the reconstruction at a block's own
boundary face reads cells `i−2 … i+1`, and both sides of that face must
compute the flux from the same four values. It is also what the
`Conservative` family's `p = 3` prolongation needs (`G ≥ (p−1)/2 = 1`)
and what `p = 5` would need (`G ≥ 2`), so the order is free to move
within the layout. `N ≥ 2G` puts `N ≥ 4`; tests use `N = 8`, demos
`N = 16 … 32`.

**`P` has `U`'s ghost width and centering, and holds primitives in its
ghost cells too** (decided). The reconstruction reads primitives at
`i−2 … i+1`, so a block needs primitives two cells into its neighbours. There are two
ways to get them, and the choice is a real design decision because the
two prolongate different quantities across a coarse-fine face:

- **(a) Exchange `U`, recover `P` everywhere.** `fill_ghosts!` moves
  conserved variables (conservatively, with the operators of the
  schedule); then one pointwise `con2prim` pass over *every stored cell*
  of `U`, ghosts included, writes `P`. `P` is never exchanged and needs no
  schedule.
- **(b) Recover `P` on owned cells, exchange `P`.** One `con2prim` pass
  over the owned range, then `fill_ghosts!(P, …)` with operators of the
  application's choice — this is the "primitive-variable-based
  prolongation" TreeAMR's design lists as an application operator.

**(a) is decided**, with the understanding that it may change. It keeps
the state vector and everything the mesh interpolates in conserved
variables, so the ghost exchange, the regrid transfer and the
conservation argument are one story; it is one schedule rather than two;
and it is what the Burgers example already does, with `con2prim` as the
identity. Its known weakness is that an unlimited linear prolongation of
`ρ`, `S`, `E` *separately* into a fine ghost cell across a strong shock
can produce a negative internal energy there, which `con2prim` then
floors. That is a measurement, not an argument: the driver counts floor
hits in ghost cells separately from owned cells (see
[Floors and the atmosphere](#floors-and-the-atmosphere)), and if the
ghost count dominates on the Sedov blast, that is the concrete case to
take upstream as a request for a limited prolongation — and the point at
which (b) would be reconsidered.

The `con2prim` pass over the stored extent is one thing `map_blocks!`
cannot launch today: it covers the owned or the closed range, never the
ghosts. **`map_blocks!(…; stored = true)` is an upstream prerequisite**
(decided) rather than a direct launch here: a launch over every stored
point of every block is a statement about the storage layout, which is
the mesh's to make, and a second application spelling out `size(fs.work)`
would be a second copy of that statement. See
[Upstream prerequisites](#upstream-prerequisites).

### Reconstruction

Piecewise-linear reconstruction of **primitive variables**, component by
component, with a slope limiter chosen per run through a `Val` so that
the flux kernel specializes and the branch disappears (as the Burgers
kernel does):

| limiter | slope from `a = P_i − P_{i−1}`, `b = P_{i+1} − P_i` | use |
|---|---|---|
| `:none` | `(a + b)/2` | convergence studies on smooth data — a limiter clips at smooth extrema and would hide the interface behind its own footprint |
| `:minmod` | `minmod(a, b)` | the robust default for shocks |
| `:mc` | `minmod(2a, 2b, (a+b)/2)` | monotonized central — less diffusive, the usual GRMHD default, and what the Kelvin–Helmholtz rolls want |

Face states at the face between cells `i−1` and `i`:
`P_L = P_{i−1} + ½ σ_{i−1}`, `P_R = P_i − ½ σ_i`. With a TVD limiter the
reconstructed `ρ` and `p` lie between neighbouring cell values and so
stay positive; with `:none` they may not, and the face states go through
the same floor function `con2prim` uses before the flux is formed.

Primitives rather than conserved variables (decided), because
reconstructing `U` produces pressure oscillations at contacts and, in
GRMHD, would cost a root find per *face state* instead of per cell and
put its failures where they are hardest to handle. Characteristic-variable
reconstruction is rejected for the GRMHD reason: the eigenvectors are
expensive and nobody uses them there.

**(Implemented in step 2.)** `src/reconstruction.jl` is `slope(lim, a, b)`
for the three limiters and
`face_states(lim, eos, floors, P₋₂, P₋₁, P₀, P₊₁) -> (P_L, P_R)`. What
writing it settled:

- **The face-state floor hits are not counted, anywhere.**
  `face_states` returns the two floored states and discards both flags.
  Under a TVD limiter they cannot fire on physical cell states at all —
  the face value lies between the two neighbouring cell values, so a
  positive `ρ` and `p` on both sides give a positive `ρ` and `p` at the
  face, which is a testset of its own — and under `:none` they can,
  which is why the call is there. The counts the design depends on are
  the ones in *cells*, owned and ghost (see
  [Floors and the atmosphere](#floors-and-the-atmosphere)); a third count
  over face states, which are not cells and are rebuilt at every stage of
  every step, would only blur the ghost count that decides the upstream
  prolongation question.
- **`:mc` is `minmod` applied twice**, not a third formula. Where
  `a b > 0` all three candidates carry the sign of `a`, so the
  smaller-magnitude rule composes: `minmod(minmod(2a, 2b), (a+b)/2)`; and
  where `a b ≤ 0` the inner call already returns zero, so `:mc`'s
  vanishing at an extremum comes from the same line that gives `:minmod`
  its own. All three slopes are symmetric in their arguments and odd
  under negation.
- **A fourth limiter is refused at compile time** by not existing: a
  symbol with no method is a `MethodError` where the kernel specializes,
  not a branch taken in the middle of a run.

### Riemann solver

Three fluxes, selected by a `Val` like the limiter, all needing only
the primitive face states and the equation of state:

- **LLF / Rusanov.** `F = ½(F_L + F_R) − ½ λ (U_R − U_L)`, with
  `λ = max(|v_L| + c_L, |v_R| + c_R)`. The simplest and most diffusive;
  the fallback every GRMHD code keeps for cells that fail otherwise.
- **HLLE** (decided default). Two-wave HLL with Davis's speed
  estimates, `s_L = min(v_L − c_L, v_R − c_R)`,
  `s_R = max(v_L + c_L, v_R + c_R)`. Needs the fastest and slowest
  signal speeds and nothing else, which in GRMHD are the fast
  magnetosonic speeds — this is *the* GRMHD flux.
- **HLLC.** Restores the contact wave HLLE smears. It has a GRHD
  counterpart (Mignone & Bodo 2005) and HLLD is the MHD one, so it is
  admissible. It is *the comparison, not the baseline* (decided): the
  Kelvin–Helmholtz case is where HLLE's contact diffusion shows, so HLLC
  is added at that milestone and measured against HLLE there.

`v` here is the component normal to the face; the tangential momenta are
advected. The flux function takes the direction `d` as a `Val` and the
velocity as an `NTuple{D}`, so one function serves every direction and
every `D`.

Deliberately **not** the flux: the exact Riemann solver. It exists in
this package — as the *reference solution* for the shock tube — but it
has no GRMHD counterpart as a flux and is not used as one.

**(Implemented in step 2.)** `src/riemann.jl` is
`physical_flux(eos, P, ::Val{d})`, `signal_speed(eos, P)` and
`riemann_flux(::Val{:llf|:hlle|:hllc}, eos, P_L, P_R, ::Val{d})`. HLLC is
written here beside the other two, and measured in H5. What writing them
settled:

- **`s_R − s_L > 0` is a fact about the floors, not about Davis's
  estimate.** `s_R − s_L ≥ (v_L + c_L) − (v_L − c_L) = 2 c_L > 0` because
  `c_s = sqrt(γ p / ρ)` is strictly positive wherever `ρ ≥ ρ_atm` and
  `p ≥ p_floor`, which `apply_floors` guarantees of every state that
  reaches a flux. So the HLL average never divides by zero, and the
  guarantee is the floors' rather than the solver's.
- **The HLLC transcription is Toro's**, *Riemann Solvers and Numerical
  Methods for Fluid Dynamics*, 3rd ed., §10.4: the contact speed is
  (10.37), the star state (10.39) with `E_K` the total energy per unit
  *volume* so that `E_K/ρ_K` is the specific total energy, and
  `F★_K = F_K + s_K (U★_K − U_K)` is (10.38). Davis's speeds are shared
  with HLLE, so the two differ in the middle wave and in nothing else,
  which is what makes the H5 comparison a comparison of that wave.
- **HLLC's four-way cascade also removes the divisions that could
  vanish.** `F_L` if `s_L ≥ 0`, else `F★_L` if `s★ ≥ 0`, else `F★_R` if
  `s_R ≥ 0`, else `F_R`: the left star state is reached only when
  `s_L < 0 ≤ s★` and the right one only when `s★ < 0 ≤ s_R`, so the
  `s_K − s★` each divides by straddles zero. `s★`'s own denominator,
  `ρ_L (s_L − v_L) − ρ_R (s_R − v_R)`, is a sum of two strictly negative
  terms for the same reason `s_R − s_L` is positive.
- **A stationary contact is exact under HLLC and wrong under HLLE by a
  closed form.** With both states at rest and at equal pressure, Davis
  gives `s_R = −s_L = max(c_L, c_R)` and the HLLE average collapses to
  its dissipative term: its mass flux is `c (ρ_L − ρ_R)/2` where the
  exact flux is zero. HLLC's `s★` is zero there and its star state is the
  cell state, so it returns the exact flux `(0, p δ_id, 0)`. Both halves
  are asserted in the tests; the second is what the H5 measurement is
  about.
- **LLF is at least as diffusive as HLLE**, and Sod's states *at rest*
  are the degenerate pair where the two coincide exactly (`s_R = −s_L`
  makes the two central terms equal and `|s_L s_R|/(s_R − s_L) = λ/2`).
  Worth recording because "LLF is the diffusive one" is otherwise checked
  on the one pair where it is not.
- **The direction enters in two places only**: as an index into the
  velocity tuple, and as the slot the pressure — and, in HLLC, the
  contact speed — is written to with `Base.setindex`. One method serves
  every `d` and every `D`, which a cyclic-permutation test asserts.

### Time integration and the time step

`SSPRK33` from `OrdinaryDiffEqSSPRK`, fixed step, the way TreeAMR's
Burgers test uses it: conservation holds for any Runge–Kutta method
(every stage's `du` sums to zero), but only a strong-stability-preserving
one keeps a limited scheme's shocks monotone. Its `stage_limiter!` hook
is also where the atmosphere reset acts (see
[Floors and the atmosphere](#floors-and-the-atmosphere)) — a second
reason for an SSPRK method, since a positivity-preserving correction
after each stage is what those hooks exist for.

One global time step for the whole hierarchy — TreeAMR has no
subcycling, ever — from the finest spacing and the fastest signal:

    λ_max = max over cells and d of (|v_d| + c_s)         # a block_mapreduce over P
    dt    = cfl · minimum_spacing / (D · λ_max)           # cfl ≈ 0.4

The `D` is the sum over directions of an unsplit scheme's CFL condition,
bounded above by `D · λ_max`; slightly conservative, and simpler than a
per-cell sum. `λ_max` is measured **once per chunk** (proposed), because
the integrator owns the steps within a chunk and a step-adaptive `dt`
would mean a callback fighting a fixed-step `solve`. The driver
re-measures `λ_max` at the end of the chunk and *throws* if the step it
used violated the condition — a loud failure naming the chunk rather than
an instability a few chunks later. For the three test problems the speed
is constant (Sod), nearly constant (Kelvin–Helmholtz) or decreasing
(Sedov, after the deposition), so the per-chunk value is a bound in
practice; the check is what makes that a fact rather than a hope.

**(Amended in step 4: "constant for Sod" is wrong, and the driver needs a
headroom factor beside the recheck.)** A Riemann problem's fastest signal
is *not in its initial data*. Sod's initial data carries nothing above
`c_L = sqrt(7/5) ≈ 1.18322`; the gas behind the shock, which does not
exist until the discontinuity resolves, carries
`u★ + c★_R ≈ 2.19157`. The ratio is **1.8522** (measured in step 4; see
[Measured results](#measured-results)), so a step sized from
`max_signal_speed(p)` at the start of a chunk would run the first steps of
that chunk at nearly twice the CFL number it asked for. The speed is
constant in the sense that matters — it stops growing once the waves are
formed — and that is exactly the wrong sense for a `λ_max` measured at
`t = 0`.

Two consequences, one taken now and one recorded for step 7:

- **The shock tube takes its `λ` from the exact solution.**
  `max_signal_speed(::ExactRiemann)` is the supremum over all time, by
  construction, so `sod_errors` needs no headroom at all. That is
  available because Sod has a closed-form solution; the driver will not
  have one.
- **The per-chunk `λ_max` needs a headroom factor, as a case parameter,
  beside the end-of-chunk recheck.** The recheck is a detector, not a
  guard: it fires *after* the chunk that violated the condition has
  already been integrated. Sod's 1.8522 is the number that sizes the
  factor — it is the largest growth within a chunk that any case here
  produces, and it is produced by the first chunk of a shock tube, which
  is the configuration the driver will meet on Sod, on Sedov's deposition
  and on any restart from discontinuous initial data. A second, much
  smaller contribution is measured beside it: the *discrete* state sits
  above the exact supremum at a discontinuity by 0.49% at `N = 16`, falling
  to 0.0056% at `N = 128`, because a reconstruction of a jump produces a
  face state the exact solution does not contain. The driver is not
  designed here; the fact and the number are recorded so that it is not
  designed without them.

**(Implemented in step 7; the amendment is closed.)** The driver sizes each
chunk's step from `cfl · h_min / (D · speed_headroom · λ)` with `λ`
measured once at the start of the chunk, re-measures at the end, and calls
`check_cfl(dt_used, h_min, D, cfl, λ_end)`, which **throws** an
`ArgumentError` naming the chunk, both speeds, the headroom and the two
remedies. The comparison carries a few ulp of slack, because the step
actually taken is `(stop − t)/steps` with an integer `steps` and is
therefore at most the step that was asked for; the equality case is real.

The measurement that justifies the parameter existing: `speed_headroom = 1`
on Sod throws in the **first chunk**, with `λ_end = 1.9486` against a step
sized for `1.25` — a CFL number of **0.6236** against the requested 0.4. At
`2` the same run completes, and `λ_end` never exceeds `2λ` at any chunk,
with `λ` rising from 1.1832 to 2.2047 over the run. So the recheck is not
decoration: it fires on the first thing a driver does with discontinuous
initial data, and the headroom is what stops it firing.

The headroom feeds one other thing, and that is the constraint which sets
the regrid cadence. The travelling margin is derived from
`speed_headroom · λ · chunk` at the cap's spacing, and TreeAMR's
recruitment reaches one ring of neighbours, so the travel must stay under
one finest-level block width `(L/roots)/2^cap`. On the tracked tube that
caps the cadence at `chunk < 0.0071` in `D = 1` at a cap of 2, and
`chunk = 1/50` is refused by `refinement_buffer` naming the constraint —
the guard working, not a tuning knob. Doubling the travel through the
headroom is conservative by exactly the factor the headroom is, and in the
right direction: a margin that is too wide costs cells, and one that is too
narrow costs the feature.

The Amdahl term TreeWave measured — the integrator's serial stage
arithmetic capping a threaded step at 3.6× — applies here unchanged and
is not TreeHydro's to fix. A hand-written SSPRK33 with its stage updates
as kernels is the fix, and it is listed under extensions in both
packages; if the MHD package needs it, it goes there first.

### Floors and the atmosphere

Two regimes, two rules, as in every GRMHD code (decided):

- **Atmosphere.** Where `ρ < ρ_atm`, the cell *is* vacuum and its whole
  state is replaced: `ρ = ρ_atm`, `v = 0`, `p = p_atm`. Zeroing the
  velocity is the point in astrophysics — a star's exterior is where
  `|v| + c_s` would otherwise run away and set the time step — and it is
  what makes this a *reset* rather than a clamp.
- **Pressure floor.** Where `ρ ≥ ρ_atm` but the internal energy
  `E − ½ S²/ρ` is non-positive or below `p_floor/(γ − 1)`: keep `ρ` and
  `v`, set `p = p_floor`. Only `E` changes.

`ρ_atm`, `p_atm` and `p_floor` are case parameters in `T`, held in a
`Floors` struct beside the EOS. One function,
`apply_floors(eos, floors, P) → (P′, hit)`, implements both rules on a
primitive state and is the only place they are written down.

**Where the reset acts.** The first draft floored `P` only and left `U`
untouched, which conserves exactly and is enough for the cases here. It
is not enough for what the package rehearses: a star evolved in a large
vacuum region, where the atmosphere must be *imposed on the state* or
its velocities run away. So `U` is reset — and the question is where,
given that TreeAMR's RHS contract forbids the RHS to mutate `u` and an
external integrator owns the stages. The ways it can be done
consistently, and what each costs:

1. **In the integrator's own stage hook** (decided). Every SSPRK method
   in `OrdinaryDiffEqSSPRK` takes `stage_limiter!(u, integrator, p, t)`
   and `step_limiter!(u, integrator, p, t)`, called on each stage vector
   after it is formed and on the step's result — the hooks that exist
   for positivity-preserving limiters, which is what this is. The reset
   is a pointwise kernel over `statearray(u, U)`: `con2prim`,
   `apply_floors`, `prim2con`, written back where anything changed. It
   needs no ghosts, no spacings and no scatter, only the EOS and the
   floors, which the hook reaches through `p`. This is GRMHD practice —
   reset after every substep — spelled in the integrator's vocabulary,
   and the RHS contract is untouched: the RHS never mutates `u`; the
   *method* does, where the method defines it. `reset = :stage` is the
   default; `:step` (the cheaper hook, once per step) and `:none` (the
   first draft's behaviour) are switches, so that what the per-stage
   reset buys over the per-step one is a measurement on Sedov rather
   than an inheritance.
2. **A `DiscreteCallback`** with an always-true condition, modifying
   `integrator.u` and calling `u_modified!`. Equivalent to the
   `step_limiter!` and less direct; not adopted, recorded because it is
   what works with a method that has no limiter hook.
3. **At chunk boundaries only**, in the driver, where the state is
   scattered anyway. Free, and useless for a star: an atmosphere
   velocity runs away within a few steps, not a few chunks. Rejected as
   the only reset; kept as one of the two places the reset also runs
   (below).
4. **A conservative alternative that is not a reset:**
   positivity-preserving flux limiting (Zhang & Shu 2010; the
   flux-correction form of Hu, Adams & Shu 2013), blending each
   high-order flux toward the first-order LLF flux by the factor that
   keeps every forward-Euler update above the floor; SSPRK is a convex
   combination of forward-Euler steps, so positivity carries over. It
   lives inside the RHS, conserves to roundoff, and guarantees
   positivity — but not an atmosphere *value*, and it cannot zero a
   velocity, so it does not do the astrophysical job. It also interacts
   with the coarse-fine fixup: an average of limited fine fluxes need
   not keep the coarse cell positive. Listed under extensions as the
   complement, not the replacement.

**What "consistent" has to mean**, and how each part is checked:

- **`U` and `P` agree after a reset**: `con2prim` of the reset `U`
  reproduces the floored `P` to roundoff, and applying the reset twice
  equals applying it once to roundoff (an idempotence test; bit-for-bit
  would need the floor comparison to absorb the `prim2con`/`con2prim`
  round trip, and is not claimed).
- **Conservation is accounted for, not assumed.** A reset injects mass,
  momentum and energy; the injection is *measured* — the per-variable
  totals of the stage vector before and after the reset, through
  `block_mapreduce` over `(U, u)`, accumulated over the chunk — and
  reported beside the drift. Where no cell is floored the two totals are
  bit-identical and the injection is exactly zero, so the roundoff
  conservation claim stands unchanged on the entropy wave, Sod and
  Kelvin–Helmholtz; on Sedov the drift is reported as fixup roundoff
  plus measured injection, and the negative control compares the two
  runs on the drift *net of* injection. The accounting doubles the
  reductions per stage and is a keyword the tests turn on and the demos
  do not.
- **The reset is a pure pointwise map**, so it is bit-identical across
  thread counts and identical on every backend, and it composes with
  the RHS's purity: the RHS reads the reset `u` and nothing else.
- **The RHS-level floor on `P` stays, as the second line.** The reset
  reaches owned cells only, because that is what the state vector holds;
  ghost cells are refilled from owned data at every evaluation, and a
  *prolongated* fine ghost across a strong shock can still be
  unphysical, as can a reconstructed face state under `:none`. So step
  (2) of the RHS still applies `apply_floors` to `P` everywhere, never
  touching `U`. With the stage reset in place, hits in owned cells
  should be rare there — the reset already handled the stage that
  produced them — and the ghost count is the measurement of the
  ghost-exchange decision above.
- **After a regrid, before the next `solve`**, the driver runs the same
  reset on the gathered state: the `p = 3` prolongation into a fresh
  fine block is unlimited and can produce an unphysical owned cell,
  which would otherwise wait for the first stage of the next chunk to
  be caught.

**Floor hits are counted, per chunk, in two populations:** owned cells
(the stage reset's count, accumulated over the chunk) and ghost cells (a
per-block count over the stored extent of `U` at the chunk boundary,
combined in block order; `firing_boxes` covers owned cells only). The
ghost count is what the Sedov milestone reports against the
ghost-exchange decision.

### What has a GRMHD counterpart, and what is deliberately not used

| here | GRMHD counterpart |
|---|---|
| conserved `(ρ, S_i, E)` evolved, primitives `(ρ, v_i, p)` recovered | `(D, S_i, τ)` densitized by `√γ`; primitives by a 1D root find |
| `con2prim` as a fallible, flooring step | the same, and where most of the code's robustness lives |
| atmosphere reset of `U` after every stage; pressure floor on `P` in the RHS | the same two rules in the same two places |
| `IdealGas` behind an EOS interface | hybrid or tabulated EOS behind the same |
| primitive reconstruction, MC or minmod | the same |
| HLLE from `|v| ± c_s` | HLLE from the fast magnetosonic speeds |
| HLLC (planned) | HLLC (Mignone & Bodo 2005), HLLD |
| LLF fallback | LLF fallback |
| MOL, SSPRK33, unsplit flux divergence, one global `dt` | the same |
| empty source-term slot in the divergence kernel | geometric source terms |
| Löhner indicator on `ρ` and `p` | the same, plus `B` |
| Dirichlet boundary from the initial state | an analytic exterior |

**Not used, on purpose:** the exact Riemann solver as a flux; Roe's
solver and any characteristic decomposition; operator (Strang) splitting;
the MUSCL–Hancock predictor; the dual-energy formalism; artificial
viscosity; Lagrange-plus-remap; anything that needs the Newtonian
eigenvectors in closed form.

## The right-hand side

The conservative RHS of TreeAMR's "Application interface", with two
steps added for the primitives — six in all, in this order:

    scatter!(U, u)                                   # (0) state → working array
    fill_ghosts!(U, schedule; boundary)              # (1) conserved ghosts
    con2prim!(P, U)                                  # (2) every stored cell, ghosts included
    for d in 1:D
        map_blocks!(flux_kernel!, F_d, …; closed = true)   # (3) reconstruct + Riemann, N+1 faces
        restrict_interfaces!(F_d, isched[d])         # (4) the fixup at coarse-fine faces
    end
    map_blocks!(divergence_kernel!, U, statearray(du, U), …)  # (5) du = −Σ_d ΔF_d / h  (+ S = 0)

Step (3) fuses reconstruction and flux into one kernel per direction, as
the Burgers flux kernel does: at face `i` of the closed range it reads
primitives `i−2 … i+1`, forms `P_L`, `P_R`, and writes the flux. Storing
per-cell left and right face states as a separate pass — the other common
arrangement — costs two more field sets for nothing at second order.

Two properties inherited from TreeWave and TreeAMR, restated because a
hydro code has more places to break them: **the RHS never mutates `u`**
— the atmosphere reset mutates the *stage vector*, from inside the
integrator's limiter hook, which is where the method defines that to
happen (see [Floors and the atmosphere](#floors-and-the-atmosphere)) —
and **the RHS is a pure function of `(u, t)`**: `P` and `F_d` are
scratch, fully rewritten at every evaluation, and nothing reads them
from a previous one.

`HydroProblem` is the container the integrator carries, in the shape of
`BurgersProblem`: the three kinds of field set, the ghost schedule, the
`D` interface schedules, the per-block spacings on the field set's
backend, the EOS and floors, and — as `Val`s — `D`, the three ghost
widths, the limiter and the Riemann solver. It is rebuilt after every
regrid and takes the existing `P` and `F_d` when it is, since `regrid!`
already resized them.

**(Implemented in step 3.)** `src/evolution.jl`, in `hydro_rhs!`, is the
six steps above with nothing between them. What the writing settled:

- **`HydroProblem(U, ops; eos, floors, limiter, riemann = :hlle, fixup =
  true, boundary = nothing, prims = nothing, fluxes = nothing)`.** `U` is
  the only field set the caller builds, because it is the only evolved
  one; `P` and the `D` flux sets are scratch and the constructor allocates
  them, or takes the ones a `regrid!` has already resized. `limiter` has
  **no default** — the cases choose differently, and a default would pick
  one of them silently — while `riemann` defaults to `:hlle`, which is
  this document's decision and not the caller's. The layout obligations
  are checked here rather than discovered later: `U` cell-centered with
  `G ≥ 2` everywhere, `P` with `U`'s forest, ghost width and centering and
  `nvars = D + 4`, the fluxes with `G = 0`.
- **Three kernels, three ranges.** `con2prim_kernel!` runs under
  `stored = true` and its index *is* the stored index; `flux_kernel!` runs
  under `closed = true`, one launch per direction with a constant
  `Val(d)`, and reads `P` at `c[d]−2 … c[d]+1` around face `I[d]`;
  `divergence_kernel!` runs over the owned range and writes straight into
  the state layout. The empty source slot is a comment on the line that
  would carry `S(P)`.
- **The boundary hook is passed through** `fill_ghosts!(U, schedule;
  boundary)` from step 3 on, although the only case here is periodic and
  hands it `nothing`: the hook is the part of the signature a Dirichlet
  case needs, and adding it in step 4 would mean touching the right-hand
  side again for it.
- **Four helpers the driver will want** live beside the right-hand side
  rather than inside it: `update_primitives!(p, u)` is steps (0)–(2)
  alone, so a driver can refresh `P` for `λ_max` or for the refinement
  criterion without computing fluxes; `max_signal_speed(p)` and
  `floor_hits(p)` are the two reductions over the diagnostic slots, and
  both require `P` to be current; `hydro_dt(forest, cfl, λ, Val(D))` is
  the time step above.

## Conservation at coarse-fine faces

The mechanism is TreeAMR's and needs nothing from this package but the
obligation: with `G = 2` on `U`, same-level faces agree bit for bit, and
step (4) replaces every coarse flux on a coarse-fine face by the
area-weighted average of the fine ones. What this package adds is that
there are now **`D + 2` conserved quantities**, each with its own domain
integral, and the claim is made for all of them:

- **(measured in step 5)** Total mass, each momentum component and total
  energy are conserved to a few ulp of their own scale `Σ hᴰ |U_v|` over a
  run on the static two-level mesh in `D = 1, 2, 3`; the drift does not
  grow with the step count — the worst of the `D + 2` integrals moves by
  **0.003 to 0.011 ulp of its own scale per step**. The negative control —
  `fixup = false`, the single difference, same mesh and same step count —
  leaks by `1e-4 … 3e-5` of the scale, which is **1e8 to 1e9 times** the
  bound the fixup run meets. The **uniform control** is
  **(measured in step 3)**: on a single-level mesh the two runs agree bit
  for bit, since `restrict_interfaces!` has no coarse-fine face to act on.
  All three are in [Measured results](#measured-results). This is TreeAMR's
  M8b table, repeated for a system. What step 5 does *not* cover, because
  it has no driver yet, is a refined region that *follows* a shock with
  regrids in between; that is step 7's, and the static mesh is the sharper
  measurement of the two because nothing but the fixup differs between the
  runs.
- **(measured in step 7)** The moving mesh holds as well as the static one.
  On the tracked Sod tube — the mesh rebuilt under the solution **9** times
  in `D = 1` over 588 steps and **3** times in `D = 2` over 431 — the mass
  and energy drifts are `3.3e-16` and `1.6e-15` in `D = 1` and `4.2e-17`
  and `1.9e-16` in `D = 2`, and the momentum equals its closed-form
  boundary flux to `8.6e-16` and `1.0e-16`. The negative control leaks
  `4.2e9`, `1.2e9` and `7.4e9` times more in `D = 1` and `1.0e9`, `7.8e8`
  and `4.9e8` in `D = 2`, on **the same mesh history and the same step
  count** — the leak does not move the criterion, which is worth knowing
  because it means the two runs really do differ in one line. The static
  measurement is still the sharper one, and this is the one that says
  prolongating a fresh fine block and restricting a coarsened one preserve
  the integrals too.
- The **momentum** is the new case: for a momentum component whose total
  is zero by symmetry (the Kelvin–Helmholtz `S_y`, the Sedov `S_d`), the
  drift is measured against the maximum of that component's `Σ hᴰ |S_d|`
  over the run, not against its total, which may be zero. **(Amended in
  step 5: the scale can be zero too.)** On Sod the initial state is at
  rest, so `Σ hᴰ |S|` is *exactly* zero at `t = 0` and the momentum has no
  scale of its own at all. There the drift is measured against the
  closed-form boundary flux `(p_L − p_R)·t_end·A` it is claimed to equal,
  which is a better yardstick than any norm of the state — see
  [Sod shock tube](#sod-shock-tube).
- The **atmosphere reset** does touch `U`, and what it injects is
  measured rather than lost in the drift (see
  [Floors and the atmosphere](#floors-and-the-atmosphere)): the claim is
  made on the drift net of the measured injection, and where no cell is
  floored — every case but Sedov — the injection is exactly zero and the
  claim is the plain one. The floor-hit counts are reported beside the
  drift so that a run that conserved *because* nothing interesting
  happened is distinguishable from one that conserved through a strong
  shock.

## Operator order

The state's ghosts and its regrid transfer use the `Conservative` family:
restriction the exact average (no order), prolongation of odd order `p`.
TreeAMR's interface-order rule for a flux divergence — measured on
Burgers in M8b — says `p` must exceed the scheme's order by *one*, so a
second-order scheme wants **`p = 3`**, and `p = 5` buys nothing further.
Two things this package expects to add to that finding:

- **(measured in step 5)** The rule holds for the system as it did for the
  scalar, and the prediction — L∞ rates 1, 2, 2 for `p = 1, 3, 5` and L1
  rates 2, 2, 2, with the refined `p = 3` run landing on the unrefined
  control's rate — is met without amendment in both dimensions. The table
  is in [Measured results](#measured-results): L∞ **0.963 / 2.034 / 2.037**
  in `D = 1` against a control of 2.024, and **0.925 / 2.041 / 2.037** in
  `D = 2` against 2.029, with every L1 rate between 1.90 and 2.02. The
  norm is part of the result here as it was on Burgers, and for the same
  reason: the defect an order-`p` prolongation leaves sits on the
  coarse-fine face and nowhere else, so a volume-weighted norm multiplies
  its `O(h^p)` by the shrinking measure of the region it occupies and sees
  the scheme's own order whatever `p` is. The **negative control on the
  rate** reproduces TreeAMR's finding closely — with `fixup = false` at
  `p = 1` the L1 rate falls from 1.966 to **1.111** in `D = 1` and from
  1.900 to **1.192** in `D = 2`, against Burgers' 1.98 → 1.12 and
  1.80 → 1.25, while L∞ stays at 1.0 either way — which is what pins the
  locality of the defect on *conservation* and not on the flux-divergence
  form: without the fixup the residual has net mass and the equation
  carries it downstream as an `O(h)` plateau, which an integral norm does
  see.
- **The `p = 1` question is a real one here, and open.** Piecewise-constant
  prolongation is the only linear prolongation that is
  positivity-preserving for every field, and Burgers measured that an
  integral norm never sees the interface defect it leaves. For a limited
  shock-capturing scheme judged in L1 on discontinuous solutions — the
  Sod and Sedov cases — `p = 1` may be indistinguishable from `p = 3` in
  the error and strictly better in the floor count. The Kelvin–Helmholtz
  rolls, smooth and L∞-sensitive, are where `p = 3` should win. **Default
  `p = 3` (decided), with `p = 1` measured beside it on all three shock
  cases and the table recorded here.** A *limited* linear prolongation
  — `p = 3` accuracy with `p = 1` positivity — is what a hydro code would
  actually want, is not a fixed-weight tensor-product stencil, and is
  therefore the one upstream request this package is most likely to
  make. It is not made in advance of the measurement.

  **What the entropy wave says about it (measured in step 5), and why the
  question stays open.** On a smooth solution `p = 1` costs exactly what
  the rule says: a full order in L∞ (0.963 and 0.925 against a control of
  2.024 and 2.029) and *nothing at all* in L1 (1.966 and 1.900, inside the
  spread of the `p = 3` and `p = 5` runs). So the smooth case reproduces
  the Burgers finding and decides nothing: it confirms that an integral
  norm cannot see the interface defect, which is the premise of the
  question rather than its answer. The question is what happens in L1 on a
  *discontinuous* solution, where the scheme's own rate is about 0.9 rather
  than 2 and where positivity is a live concern — and that is Sod in step 7
  and Sedov in step 9, run at `p = 1` beside `p = 3` with the floor counts
  beside the errors. Until then `p = 3` remains the default and the
  upstream request remains unmade.

  **The first discontinuous row (measured in step 7), and it goes against
  `p = 1`.** On the tracked Sod tube in `D = 1`:

  | `p` | L1 against the exact solution | tracking | cells | floor hits |
  |---|---|---|---|---|
  | 1 | 4.701364e-3 | **0.9091** | 216 | 0 |
  | 3 | **4.540016e-3** | 1.0000 | 200 | 0 |
  | 5 | 4.539988e-3 | 1.0000 | 200 | 0 |

  `p = 1` is 3.6% worse in L1, uses 8% more cells, and — the interesting
  column — **loses tracking**: the defect a piecewise-constant prolongation
  leaves on a coarse-fine face is itself a second difference, so the
  indicator fires on it, on a block that is not at the cap. So on this case
  the lower order costs accuracy *and* mesh, and buys nothing in the floor
  count because Sod fires no floor at all. That last clause is why the
  question is not closed here: positivity is the argument for `p = 1`, and
  Sod cannot speak to it. Sedov, whose density spans orders of magnitude
  and whose bubble sits at the atmosphere, is where the floor count becomes
  a real column, and it completes the table in step 9.

## Boundaries

**Dirichlet wherever the physics does not require periodicity; periodic
only where it does** (decided, reversing the first draft, which reached
for periodic boundaries wherever the problem allowed them because they
cost no application code). Two arguments overturned that default:

- **Cost.** A periodic boundary face is an ordinary face of the exchange
  — a copy, or a prolongation and a restriction where the levels differ,
  and under MPI (TreeAMR's M7) a message. A Dirichlet face is a kernel
  over the outward-facing ghost regions with no transfer behind it and,
  later, no communication. The saving per RHS evaluation is modest on
  one node and real across nodes, and it is free.
- **Coverage.** The physical-boundary hook is the one exchange path no
  downstream code exercises. A tube's two `x` faces run it on faces
  only; a case that is Dirichlet on *every* face runs it on the edge and
  corner regions too, which TreeAMR fills unconditionally and which
  TreeWave never touches.

What that gives per case:

| case | boundaries | why |
|---|---|---|
| entropy wave | periodic in all directions | the exact solution passes through the boundary; nothing else is exact |
| Sod | Dirichlet in `x`, periodic transversally | the `x` states are the initial ones until a wave arrives; transversally the planar solution is translation-invariant, and a Dirichlet boundary set to the *initial* state there would be wrong from the first step |
| Sedov | Dirichlet (the ambient state) on every face | the ambient gas is uniform and at rest until the shock arrives; this is the case that runs the hook on edges and corners |
| Kelvin–Helmholtz | periodic in all directions | intrinsic: the shear flow and its `sin(4πx)` seed in `x`, McNally's two-interface setup in `y` |

**Dirichlet from the initial data** is
`boundary = boundary_by_coordinates(initial_U)`, the `CellBoundary` form,
which runs on every backend. Set to the initial state, it is exact as
long as no wave has reached the boundary — and *not transparent* once one
has: a fixed state reflects, so a shock hitting it behaves as if it hit a
wall. Periodic boundaries would be no better there (the shock would
re-enter from the other side), so the rule is the same in either case
and the driver *asserts* it before the run: `t_end · λ_max` against the
distance from the feature to the nearest physical boundary. The hook is
passed to `fill_ghosts!`, to `regrid!` (which fills ghosts before its
transfer) and to `adapt_to_initial_data!` alike; forgetting the second is
the bug that would arrive one chunk late.

The hook runs *between* the copy/restriction phase and the prolongation
sweep, as TreeAMR's M2 amendment describes, because prolongation at a
domain edge reads tangentially into the source's outer ghosts. A two-level
mesh whose refined region touches a Dirichlet boundary is the
configuration that would show a mistake there; the Sod tests include one
on a face **(implemented in step 5)**, and the Sedov tests include one on
an edge and a corner.

**(Implemented in step 4.)** The shock tube is the first downstream use of
the physical-boundary path, and `src/sod.jl` uses it through
`boundary_by_coordinates(AllVariables(x -> prim2con(eos, sod_state(w, x))))`
handed to `HydroProblem`'s `boundary` keyword, which step 3 put in the
signature for this. What the exercise showed:

- **One callback serves the interior at setup and the exterior forever.**
  The same `AllVariables` object goes to `fill_by_coordinates!` and, wrapped
  in `boundary_by_coordinates`, to `fill_ghosts!`. That they are the same
  object is what makes the boundary exact rather than merely consistent, and
  it is the shape the other three cases will copy.
- **The claim has to be made after the run, not before it.** At `t = 0`
  every ghost cell holds the initial state whether the hook ran or not,
  because the interior does too. After seventy steps the interior near the
  diaphragm has moved, and then the two halves separate: the outer `x`
  ghosts still hold the left and right conserved states *exactly*, while the
  transverse ghosts hold the evolved interior of their periodic neighbour.
  A test written at `t = 0` would pass with no hook at all.
- **The corner and edge regions of a Dirichlet face are filled.** With
  `D = 2`, the outer `x` ghosts hold the boundary state over the whole
  transverse extent, the transverse ghost rows included — TreeAMR fills
  them unconditionally, so the tube's two faces already exercise a little
  of what Sedov's six will exercise in full.
- **Transverse periodicity is not a detail.** A Dirichlet condition applied
  across the tube instead of along it would be wrong from the first step and
  would still produce a profile that looks like a shock tube, which is why
  both halves are asserted rather than only the interesting one.
- **Direction independence came out bit for bit**, over the whole stored
  array including the ghosts the hook wrote: see
  [Measured results](#measured-results).

**(Implemented in step 5: a refined region touching the Dirichlet face.)**
`sod_forest(…; refined = :left)` refines every root block whose center
along the tube is below `x₀`, so the low physical face is covered by
*fine* blocks and the single coarse-fine face sits at the diaphragm. What
that measured:

- **The hook reaches a fine block's outer ghosts, and reaches all of
  them.** After the run every stored entry of the outward-facing ghost
  region of every boundary block holds the conserved left or right state
  *exactly* — the level-1 blocks on the low face included, and the ghost
  rows *across* the tube included, which TreeAMR fills unconditionally.
  Asserted after the run and not before it, for the reason step 4 gives.
- **A coarse-fine face beside a physical one conserves.** Mass and energy
  drift by no more than the uniform mesh of the same coarse spacing does,
  and the momentum by the boundary flux; without the fixup all three leak
  by four to seven orders of magnitude more. At `N = 32` in `D = 1`, where
  the boundary's own numerical flux has itself reached roundoff, the drift
  is `1.1e-16`, `6.7e-16` and `2.8e-17` against bounds of `2.8e-13`,
  `6.9e-13` and `9.0e-14`.
- **What it does *not* exercise, and it is worth being explicit.** The M2
  ordering case in full is a prolongation reaching *tangentially* into
  hook-filled ghosts, and that needs two physical faces meeting at an edge
  or a corner. The tube is periodic across itself, so its only physical
  faces are its two ends and they never meet. That case is Sedov's, in
  step 9, where every face is Dirichlet. What `:left` exercises is the
  hook on a fine block and a coarse-fine face with the boundary state on
  one side of the refined region.

**(Measured in step 7.)** The hook's other two call sites, `regrid!` and
`adapt_to_initial_data!`, are wired in `evolve!` and exercised on the
tracked tube: after the adaptation cycle every stored entry of the
outward-facing ghost regions along the tube holds its conserved boundary
state exactly — ghost rows across the tube included — on blocks that did
not exist when the run started, and it stays that way through nine
regrids. What is still not exercised is the M2 ordering case in full,
which needs two physical faces meeting; that is Sedov's corner, in step 9.

## The cases

Four initial conditions. As in TreeWave, none is a variation on another:
each measures something the others cannot.

| case | `D` | boundaries | reference | what it measures |
|---|---|---|---|---|
| entropy wave | 1, 2, 3 | periodic | exact | the scheme's order; the interface-order rule for a system; conservation on a static mesh |
| Sod shock tube | 1, 2, (3) | Dirichlet in `x`, periodic else | exact Riemann solver | shock capturing against a known answer; conservation through regrids; direction independence; the boundary hook |
| Sedov blast | 1, 2, **3** | Dirichlet (ambient) on every face | similarity law; uniform-fine run | a strong shock through the floors and the atmosphere reset; a refined *shell* that grows while its interior coarsens; the 3D coarse-fine face; the boundary hook on edges and corners |
| Kelvin–Helmholtz | 2 | periodic | published diagnostics; uniform-fine run | a contact-dominated, vortical flow; refinement following a growing structure; the picture |

**Initial data is given as primitives** — a pure `x -> P` closure per
case, returning all `D + 2` values as a tuple — and converted with
`prim2con` once per cell. TreeAMR's coordinate callbacks are today called
once *per variable*, `(x, v) -> value`, which would evaluate `P(x)`
`D + 2` times per cell and, worse, would make the boundary hook do the
same at every RHS evaluation; an all-variables form of
`fill_by_coordinates!`, of `adapt_to_initial_data!`'s `initial` and of
`CellBoundary` is therefore an **upstream prerequisite** (decided; see
[Upstream prerequisites](#upstream-prerequisites)), and the cases are
written against it.
**Point samples at cell centers**, not cell averages (decided in step 4,
which is the first case to use them): every
case but the entropy wave is discontinuous or nearly so, where a cell
average is no better defined than a sample, and the initial-data cycle
re-evaluates the data on every mesh it produces, which a closed-form
average could not do for a mesh-dependent `h`. The entropy wave's
convergence study fills exact cell averages on the host, as the Burgers
study does, because there the `O(h²)` difference is exactly what is
being measured.

### Entropy wave

    ρ = ρ₀ + a sin(2π (Σ_d x_d − Σ_d v_d t) / L),   v = const,   p = const

An exact solution of the *nonlinear* Euler equations in any `D`: with
uniform velocity and pressure the momentum and energy equations reduce
to advection of `ρ`. It is the system's counterpart of the Burgers sine
— smooth, exact, periodic, and dependent on `Σ x_d` so that every flux
direction does real work — and it is what pins the **order**: run at
several `N` on the M3 two-level hierarchy held fixed in physical space
(`hydro_forest`, a copy of `burgers_forest`), with `:none` as the
limiter, and the slope of the error against `h` is 2 or it is not.

It is also the smooth case for the interface-order table above and for
the static-mesh conservation control. It is a contact wave, which is
exactly what HLLE diffuses most; the rate is unaffected, the constant is
not, and the HLLE/HLLC comparison has its first number here.

Not a demo: it lives in the tests and the viewer does not draw it.

**(Implemented in step 3; the two-level runs measured in step 5.)**
`src/entropywave.jl`. Step 3 measured it on the *uniform* mesh
(`refined = false`), which is the control; step 5 ran the same study on the
M3 two-level hierarchy and it is where the package's two central numbers
come from — conservation of all `D + 2` integrals with the fixup against a
ten-order leak without it, and the interface-order table at `p = 1, 3, 5`.
The rates and the drifts are in [Measured results](#measured-results). What
the writing settled:

- **The parameters are a struct**, `EntropyWave(T, Val(D); ρ₀ = 1,
  a = 1//5, v = 1, p₀ = 1, γ = 7//5, L = 1, floors)`, `isbits` and
  carrying its own EOS and floors, so that one object is the whole case
  and every default is a rational converted to `T`. `v_d = 1` in every
  direction, so no momentum component is zero by accident; the floors sit
  eight orders of magnitude below the data and never fire, which the tests
  assert rather than assume.
- **Everything is an exact cell average and there is no quadrature
  anywhere.** The average of `sin(k Σ_d x_d)` over a cube of side `h` is
  `((2/(kh)) sin(kh/2))^D` times its value at the center; `v` and `p` are
  constant, so the averages of `S = ρ v` and `E = p/(γ−1) + ½ ρ v²` follow
  from the averaged `ρ` exactly, and the reference at time `t` is the same
  expression with `x_d → x_d − v_d t`. Unlike Burgers' sine, this solution
  never becomes implicit, so the reference needs neither a Newton solve
  nor a Gauss–Legendre oracle. The damping factor depends on the *block's*
  own `h`, which is why the fill is a host loop and one `copyto!`.
- **The `O(h²)` damping factor is not optional.** It is the same order as
  the error being measured, so a reference built from point samples would
  put an `O(h²)` floor under every number in the table and the study would
  measure the initial data.
- **`entropywave_errors` returns the whole claim**, not just an error:
  the two norms, the per-variable drift *and* the per-variable scale it is
  roundoff against, the floor count, `h`, the step count and the block
  count. A run that conserved because nothing happened is then
  distinguishable from one that conserved through something.

### Sod shock tube

The standard states, `(ρ, v, p) = (1, 0, 1)` for `x < ½` and
`(⅛, 0, ⅒)` for `x > ½`, `γ = 7/5`, on `[0, 1]`, to `t = 0.2`; the
fastest wave reaches a boundary at `t ≈ 0.29`, so the Dirichlet boundary
is exact for the whole run with margin, and the driver's assertion says
so.

**Reference:** the exact Riemann solution (Toro's pressure iteration and
sampling), host `Float64` code in the role TreeWave's Hankel table
plays, evaluated at cell centers and compared in the volume-weighted L1
norm. **(measured in step 4: 0.903)** L1 convergence at a rate between 0.8
and 1 on a uniform mesh, as a limited second-order scheme gives on a
solution with a contact and a shock — this is a check that the scheme is
right, not a claim of order.

**What it measures beyond that:**

- **Conservation through regrids** in `D = 1` and `2`: the refined region
  follows the shock (and the contact, and the rarefaction — the Löhner
  indicator on `ρ` fires on all three), the mesh is rebuilt every chunk,
  and all `D + 2` integrals hold to roundoff with the fixup and leak
  without it. The M8b acceptance test, for a system. (Measured in step 7:
  they do, and the leak is `4.9e8` to `7.4e9` times the bound.)
- **A tracked shock matches the uniformly fine reference at fewer
  cells**, reduced onto a common grid as `track_shock` does, with the
  uniform coarse mesh as the control that says refinement bought
  something (measured in step 7: the error ratio is 1.0004 and 1.0000
  against a control of 3.678 and 1.850).
- **Direction independence, bit for bit** (measured in step 4: both
  halves hold, with no differing entry anywhere). On a uniform mesh
  the tube along `y` is the tube along `x` transposed, and the `D = 2`
  planar tube's profile equals the `D = 1` run's: the transverse flux
  differences are *exactly* zero (both faces see identical states), and
  adding an exact zero is exact. A cheap, sharp claim about the flux
  kernel having no preferred direction, worth asserting because the
  `ntuple` loops over `d` are the place a stray asymmetry would hide.
- **The boundary hook**, as above.

`D = 3` is a smoke test at small size; the planar tube in 3D exercises
nothing the 2D one does not, except cost.

**(Implemented in step 4.)** `src/exact_riemann.jl` and `src/sod.jl`,
measured on the *uniform* mesh. The numbers are in
[Measured results](#measured-results). What the writing settled:

- **The `t ≈ 0.29` above is the shock's travel time, and the assertion is
  made on the characteristic speed instead.** `assert_no_arrival` compares
  `t_end · λ` against the distance from the diaphragm to the nearer
  physical boundary, with `λ` the supremum of `|v| + c_s` over the exact
  solution, so it permits `t_end < 0.228` rather than `0.285`. That is
  conservative by the right amount and in the right direction: the
  boundary state is exact only while *nothing* has reached it, and a
  characteristic carries information a shock front does not.
- **The parameters are a struct**, `SodTube(T, Val(D); ρ_L = 1, v_L = 0,
  p_L = 1, ρ_R = 1//8, v_R = 0, p_R = 1//10, γ = 7//5, x₀ = 1//2, L = 1,
  direction = 1, floors)`, `isbits` and carrying its own EOS and floors, as
  `EntropyWave` is. **The tube's axis is a type parameter**, not a field,
  so that the coordinate the initial data reads is a compile-time index
  rather than a runtime index into a tuple — which matters because the
  initial-data callback is also the boundary hook and runs as a kernel at
  every right-hand-side evaluation.
- **The forest is a tuple of root counts**, not one count. The two kinds of
  direction are not alike: the tube wants several roots and the transverse
  directions want one, since the solution is uniform across the tube and a
  planar tube wastes nothing by being thin. TreeAMR's blocks are cubes, so
  the transverse *extents* follow from the tube's root spacing rather than
  being given — a box of side `L` across would be `roots[direction]` times
  too much mesh.
- **`λ` comes from the exact solution**, which is the amendment recorded
  under [Time integration and the time
  step](#time-integration-and-the-time-step) and the one thing about this
  case that generalizes badly: a driver without a closed-form solution
  cannot do the same and needs a headroom factor instead.
- **The conservation claim is a different claim here.** With a physical
  boundary the domain integral is not constant, and the entropy wave's
  "constant to roundoff" is simply false. What replaces it is sharper: until
  a wave arrives, both sides of each boundary face are in that face's own
  initial state, which is at rest, so the only nonzero component of the
  Euler flux there is the pressure, and the momentum total must move by
  exactly `(p_L − p_R) · t_end · A` while the mass and the energy totals do
  not move at all. That is an equality with a closed form, and it fails for
  a forgotten hook, for a reflecting boundary and for a run long enough for
  a wave to arrive. The roundoff conservation claim proper is made for Sod
  on a refined mesh in step 5, as the *difference* between two runs sharing
  this same boundary flux.

**(Implemented in step 5: the two static two-level configurations.)**
`sod_forest` gains `refined`, which is `false`/`:none`, `:middle` or
`:left`. Both refined meshes are still non-periodic along the tube and
periodic across it, both are 2:1 balanced, and both refine whole root
blocks on a criterion that reads the tube's axis *alone* — across the tube
there is one root block and the solution does not vary, so a criterion that
also asked about the transverse center would refine nothing or everything
depending on the box's thickness. What step 5 settled about the case:

- **`:middle` exists so that the shock crosses a coarse-fine face.** It
  refines the root blocks whose center along the tube lies in
  `(L/4, 3L/4)`, as `hydro_forest` does for the periodic box, so the
  diaphragm starts *inside* the refined region and the shock — travelling
  at 1.752156 — leaves it at `t = 0.14268`, before the standard
  `t_end = 0.2`, ending at `x = 0.85043`. A refined region the solution
  never left would make every conservation assertion pass for the wrong
  reason, so the crossing is asserted from the exact solution rather than
  assumed.
- **`:left` puts the refined region against the Dirichlet face**, with the
  single coarse-fine face at `x₀`; see the step-5 note under
  [Boundaries](#boundaries) for what it does and does not exercise.
- **The mass and energy drift here is the boundary's, not the coarse-fine
  face's, and it is a discretization error rather than roundoff.** The
  numerical foot of the rarefaction and of the shock reaches the Dirichlet
  faces and lets a little mass and energy across: `2.1e-11` of the mass at
  `N = 16` in `D = 1`, falling by about four orders of magnitude per
  halving of `h` and gone by `N = 32` (`D = 2` needs `N = 32` too). So the
  claim made with the fixup is the larger of roundoff and *ten times what
  the uniform mesh at the same coarse spacing drifts by* — the refined runs
  come in at 0.15 to 2.8 times the uniform ones, so a coarse-fine face adds
  nothing — and at the resolution where the boundary is itself clean the
  plain roundoff bound holds on a mesh with a coarse-fine face in it. The
  negative control is compared against the run *with* the fixup and not
  against that bound, since the leak is what is being measured and
  inflating the yardstick would measure the yardstick.
- **The momentum's yardstick is the boundary flux and not a norm.** Sod's
  initial state is at rest, so `Σ hᴰ |S|` is *exactly* zero and
  `conserved_scales` gives the momentum no scale at all. The closed form
  `(p_L − p_R)·t_end·A` is the better yardstick anyway: it is an equality
  rather than a bound. The refined runs meet it to `5e-11` relative and
  the runs without the fixup miss it by `2e-3` to `4e-3`.
- **The two-level tube's error is 1.338 times the uniform run's at the same
  finest spacing**, in `D = 1` and in `D = 2` alike, and below the uniform
  run's at the coarse spacing. That is a sanity bound and not the
  tracked-shock claim, which needs a mesh that follows the shock and is
  step 7's.

**(Implemented in step 7: the tracked tube.)** `HydroCase(::SodTube)` and
`evolve!`; the numbers are in
[Measured results](#step-7--the-driver-and-the-tracked-shock-tube). What
step 7 settled about the case:

- **A tracked shock does match the uniformly fine reference, and the
  match is much better than the static mesh's 1.338.** The tracked run's
  L1 error against the exact Riemann solution is **1.0004** times the
  uniform run's at the same finest spacing in `D = 1` and **1.0000** times
  it in `D = 2`, against a uniform *coarse* control at 3.678 and 1.850.
  Reduced onto the grid the meshes have in common, the tracked and fine
  runs differ by `1.745e-5` in `D = 1` where the coarse and fine runs
  differ by `1.073e-2` — three orders of magnitude, and four in `D = 2`.
  The static two-level mesh cost 1.338 because its refined region did not
  move; a mesh that follows the waves costs essentially nothing.
- **`tracking == 1.0`**: every cell whose indicator exceeded `refine_tol`
  sat on a block already at the cap, at every one of the 40 chunks in
  `D = 1` and 30 in `D = 2`. This is the claim the case exists for, and it
  is a claim about the *criterion plus the buffer plus the cadence*
  together — the buffer table below is what it costs.
- **The cell saving is real but modest, because Sod's waves are most of
  the tube.** 200 cells against 256 in `D = 1` (22%) and 1472 against 2048
  in `D = 2` (28%). By `t = 0.2` the rarefaction head has reached
  `x = 0.263` and the shock `x = 0.850`, so 59% of the box holds something
  the criterion fires on; the `D = 2` run stops at `t = 0.15` for cost,
  where it is 44%. A case whose feature is a thin shell — Sedov, in step 9
  — is where the saving becomes the headline.
- **`speed_headroom = 1` throws in the first chunk**, which is the
  measurement that turns the step-4 amendment into a parameter. See
  [Time integration and the time
  step](#time-integration-and-the-time-step).
- **The boundary hook's other two call sites hold.** After the adaptation
  cycle the two root blocks meeting at the diaphragm are at the cap, and
  every stored entry of the outward-facing ghost regions along the tube —
  ghost rows across the tube included — holds its conserved boundary state
  exactly, on blocks that did not exist when the run started.

### Sedov blast wave

Uniform ambient gas at rest, `ρ₀ = 1`, small pressure `p_amb`, with
energy `E₀` deposited as thermal energy in a top-hat of radius `r₀`
around the center: `p = (γ − 1) E₀ / V_D(r₀)` inside, `p_amb` outside,
`γ = 7/5`. The classic test of a strong shock (density jump
`(γ+1)/(γ−1) = 6`, a near-vacuum interior where floors are exercised for
real), and the case `TODO.md` names for 3D.

**Deposition is defined as a function of position**, not as "one cell",
so that the initial-data cycle can re-evaluate it on each mesh; `r₀` is
chosen to span several cells at the refinement cap. The energy actually
deposited depends slightly on which cell centers fall inside `r₀`, so
**the `E₀` that enters the similarity law is the measured
`Σ hᴰ E` at `t = 0` on the adapted mesh**, less the ambient thermal
energy, not the nominal value.

**Reference, in two parts.** The similarity solution
`r_s(t) = ξ₀ (E₀ t² / ρ₀)^{1/(D+2)}` gives two checks that need no
constant: the **exponent** — the slope of `log r_s` against `log t` is
`2/(D+2)`, i.e. `2/3`, `1/2`, `2/5` in `D = 1, 2, 3`, once `r_s ≫ r₀` —
and the **post-shock density jump** of 6. `ξ₀(γ, D)` and the full radial
profile come from the standard quadrature of the similarity ODE (Kamm &
Timmes 2007), as host `Float64` reference code; the classical value
`ξ₀ ≈ 1.033` for `γ = 7/5` in 3D (Taylor) is the check on that code.
The profile is the second part and is **an extension, not a milestone**
(decided): the exponent, the jump, and the comparison against a uniform
fine run are the acceptance, and the profile is what makes the
radial-scatter figure (below) quantitative rather than qualitative.

**What it measures:**

- **Floors and the atmosphere reset**: how many cells are floored and
  where (the interior bubble against fine ghosts at the shock), what the
  reset injects, that the drift net of the injection is roundoff, and
  what `:stage` buys over `:step`.
- **A refined shell that grows while the interior coarsens** — the
  Sedov mesh problem TreeWave could only imitate with the wave equation.
  Block count and coverage of the shell against time.
- **The 3D coarse-fine face**, where the fixup averages `2 × 2` fine
  faces, under a real shock. The 3D adaptive run is expensive (TreeWave
  measured minutes for a wave-equation shell); the test runs it small and
  short (`N = 8`, few roots, cap 1, a few chunks), and `bin/` runs it at
  demo size.
- **The early phase is where the time step is smallest** — `c_s` inside
  the hot spot scales as `(E₀/V_D(r₀))^{1/2}` — and it is *also* where
  `λ_max` only decreases, so the per-chunk `dt` is a bound. Both facts
  are recorded here so that the first chunk being slow is expected rather
  than investigated.

Dirichlet boundaries holding the ambient state on every face — the case
that runs the boundary hook on edges and corners, see
[Boundaries](#boundaries); the run stops before `r_s` reaches `L/2`,
which the similarity law predicts in advance and the driver's arrival
check enforces.

### Kelvin–Helmholtz instability

The case that exists for the picture, and the only one without an exact
solution. **Setup (decided): McNally, Lyra & Passy (2012), ApJS 201:18**
— a smooth-ramp shear layer designed as a *converged* code-comparison
test, rather than the classic sharp-interface setup whose small-scale
structure never converges and whose secondary rolls are grid noise. On
the periodic unit square with `γ = 5/3`, `p = 5/2`, `ρ₁ = 1`, `ρ₂ = 2`,
`v₁ = ½`, `v₂ = −½`, ramp width `L = 1/40`, and
`ρ_m = (ρ₁ − ρ₂)/2`, `v_m = (v₁ − v₂)/2`:

    y ∈ [0, ¼):   ρ = ρ₁ − ρ_m e^{(y − ¼)/L}       v_x = v₁ − v_m e^{(y − ¼)/L}
    y ∈ [¼, ½):   ρ = ρ₂ + ρ_m e^{(¼ − y)/L}       v_x = v₂ + v_m e^{(¼ − y)/L}
    y ∈ [½, ¾):   ρ = ρ₂ + ρ_m e^{(y − ¾)/L}       v_x = v₂ + v_m e^{(y − ¾)/L}
    y ∈ [¾, 1):   ρ = ρ₁ − ρ_m e^{(¾ − y)/L}       v_x = v₁ − v_m e^{(¾ − y)/L}
    v_y = 0.01 sin(4πx)

run to `t = 1.5`. *Transcribed from memory; the implementation checks
every formula against the paper before any number is recorded.* Step 6
uses the density ramp's **shape** — the four branches above, in one
dimension, at uniform pressure — as the smooth calibration profile for
the refinement criterion, and nothing else of the setup. Nothing there
depends on the transcription being faithful; if step 10 finds it is not,
the calibration table is still a table of `max τ` against `h` for an
exponential ramp of width `1/40`, which is what picked the thresholds. Its
diagnostics are the paper's: the **amplitude of the seeded mode**
`M(t)`, from the projections of `v_y` onto `sin(4πx)` and `cos(4πx)`
weighted by `e^{−4π|y − ¼|}` so that the lower interface alone is read,
and the **maximum `y`-kinetic energy** `max ½ ρ v_y²` over the domain
against time. `M(t)` grows exponentially through the linear phase and
saturates; the incompressible sharp-interface growth rate
`k Δv √(ρ₁ρ₂)/(ρ₁+ρ₂) ≈ 5.9` for `k = 4π` is an upper bound the measured
rate must stay below, since the ramp and compressibility both slow it.
The quantitative reference is a **uniform fine run of this code**, as
for the shock cases; the paper's curves are the sanity check on their
shape, since the setup is theirs.

**What it measures:**

- **Refinement following a structure that grows** rather than travels:
  two strips at `t = 0`, then rolls, then the whole layer — the
  criterion's behaviour under a feature whose footprint changes shape,
  which neither the pulse nor the shell tests.
- **Conservation of `S_x`, `S_y` and `E`** under a flow with no shocks
  to speak of, where the leak without the fixup is small in absolute
  terms and the roundoff claim is correspondingly sharper.
- **HLLE against HLLC** (predicted): at the same resolution HLLC's rolls
  are visibly sharper and its `M(t)` closer to the fine reference,
  because the shear layer *is* a contact. This is the measurement that
  decides whether HLLC becomes the default or stays a switch.
- **The picture**: a filmstrip of `ρ` at four times, one heatmap per
  block with the block boundaries drawn and coloured by level, in the
  manner of TreeWave's `visualize2d.jl`; `M(t)` and the maximum
  `y`-kinetic energy against time with the uniform reference over them;
  block count against time.

A note on what it is *not*: a `Float32` Kelvin–Helmholtz run will not
reproduce the `Float64` one late in the run. The instability amplifies
roundoff exponentially, so the two diverge in detail while agreeing in
mesh statistics early and in `M(t)` through the linear phase. TreeWave's
"the same mesh at both precisions" claim is made here for Sod and Sedov
only; for Kelvin–Helmholtz the claim is the growth rate and the first
chunks, and it is recorded as such rather than asserted where it cannot
hold.

## The refinement criterion

A per-cell **Löhner indicator on the primitives `ρ` and `p`**, reduced
to a per-block verdict through `firing_boxes`, with TreeWave's
two-threshold scheme and travelling margin. What is the same as TreeWave
and what is different:

**Same:** two thresholds meaning two things — `refine_tol`
("under-resolved here") and `coarsen_tol` ("something is here at all"),
their gap the hysteresis band; four marks — `(Refine, box)` below the
cap, `(Keep, box)` at it (the equal-level margin that travels with the
feature), bare `Coarsen` where nothing fired, bare `Keep` otherwise; the
box keyed on `coarsen_tol` for the reason TreeWave records (a block at the
cap has stopped being under-resolved, so a box keyed on `refine_tol`
would make the margin unreachable exactly where it exists for);
`maxlevel_cap` named a cap; and `refinement_buffer(forest, cap, travel)`
with `travel = λ_max · chunk`, the fastest signal times the regrid
cadence, `+1` because the margin must exceed the motion.

**Different, on purpose:**

- **Which variables.** `ρ` and `p`, and not the velocity. `ρ` catches
  the contact and the shear layer, `p` catches shocks and rarefactions
  (it is flat across a contact), and both are **positive** — bounded
  away from zero by the floors — which is what makes the next point work.
- **The noise floor is Löhner's local one, plus a global term.**
  TreeWave found the classic local floor `ε(|u₊| + 2|u₀| + |u₋|)` fatal
  and replaced it with a global amplitude, because its fields cross zero
  and a tail of `1e−16` scored `τ ≈ 1` against a floor that shrank with
  it. Positive fields with a large dynamic range — Sedov's density spans
  orders of magnitude between shell and bubble — are the opposite case:
  a single global floor is too high where the gas is thin and too low
  where it is dense, and the local form is right. But the Sedov bubble
  is also where `ρ` sits near the atmosphere and is numerical dust, and
  the local form alone would fire on it. So the denominator carries both,
  `|Δ₊| + |Δ₋| + ε (|u₊| + 2|u₀| + |u₋|) + ε_g · u_ref`, with `u_ref` a
  per-variable global reference (the initial maximum) and `ε_g ≪ ε`. The
  first term guards dynamic range, the second guards the atmosphere.
  Both `ε` are `T(1//100)`-style rationals, and both are calibrated the
  way TreeWave calibrated its thresholds: max `τ` on uniform meshes at
  successive `h`, tabulated, thresholds chosen mid-plateau.
- **One form of the criterion, not two.** TreeWave keeps a host loop
  beside its `firing_boxes` form because the host loop returns `τmax`,
  which its tests and viewer use, and because it is what its numbers were
  measured with. TreeHydro has no such history and starts with the form
  that runs everywhere, following TreeAMR's Burgers test: two
  `firing_boxes` sweeps (one per threshold) and the verdict on the host.
  The viewer computes `τ` on a host copy for its own panel and does not
  need the package to.

**Ghosts must be filled and `P` current** when the criterion runs: the
stencil reaches one cell past the block face. The driver runs
`fill_ghosts!` and the `con2prim` pass before flagging, and flags on `P`.

**(Implemented in step 6.)** `src/refinement.jl`: `lohner`, `cell_tau`,
`indicator_scales`, `hydro_flags` and `refinement_buffer`, with the
calibration in [Measured results](#step-6--the-refinement-criterion).
What the implementation settled, and where the prediction above needed
amending:

- **The denominator, exactly as written above**, with `ε = 1//100` — the
  literature's value, not calibrated, since it is the local floor's
  relative weight and nothing here asked it to move — and
  **`ε_g = 1//1000`**, which is calibrated from both sides. It is small
  enough to mask nothing: it lowers the shear layer's `τ` by 0.8% and
  Sod's rarefaction fan by 0.5–2.6%. It is large enough to silence an
  atmosphere six orders below the data, which is where Sedov's sits: `O(1)`
  relative noise there scores `τ = 0.0020` with it and **0.971** without
  it. Five orders would be marginal (0.0196 against a `coarsen_tol` of
  0.02) and four would fire outright, which is the statement of how far
  the atmosphere has to be below the data for this to work.
- **The thresholds are `refine_tol = 0.08` and `coarsen_tol = 0.02`**,
  chosen mid-plateau in TreeWave's manner and recorded in
  [Measured results](#step-6--the-refinement-criterion) with the table
  that picks them. The shear layer is what decides them; the
  discontinuities cannot, since they exceed any usable threshold forever.
  They are *not* defaults of `hydro_flags`, which has none for either
  threshold or for the cap.
- **A captured shock scores ≈ 0.57, not ≈ 1, and does not fall with `h`**
  (0.5413, 0.5975, 0.5722, 0.5858 over a factor of eight). The
  non-falling half is the important one and was predicted: the criterion
  never resolves a discontinuity, and `maxlevel_cap` is what binds there
  — on every shock case, not as a safety net. The *value* is the
  amendment. `τ ≈ 1` is what a **sharp** jump scores — Sod's initial data
  gives 0.9657 and 0.9847 in the two cells straddling the diaphragm — and
  capture over three or four cells takes it down to 0.57 and leaves it
  there. So Löhner's canonical `τ > 0.8` detects a jump in the *data* and
  would miss a shock the scheme is actually carrying.
- **A kink does not score `O(1)` forever** (amended in step 6). The
  expectation was that the ratio of slope jump to slope sum is
  `h`-independent, so that a rarefaction's head — a corner against a flat
  state — would fire at every resolution as a shock does. It does not:
  the measured head falls almost exactly like `h` (0.1736, 0.1050,
  0.0569, 0.0286). Two reasons, both worth keeping. The scheme rounds the
  corner over a nearly fixed *physical* width, so what the indicator sees
  on a fine mesh is smooth data and not a kink at all; and even for a
  true kink the local `ε` term takes over the denominator once the first
  differences `h·s` fall below `ε·4|u|`, which for `O(1)` data is around
  `h·s ≈ 0.04`. The consequence is the good one: a rarefaction is
  refinable to a finite depth, like the shear layer and unlike the shock.
- **The contact sits in between**: 0.19, 0.22, 0.18, 0.13 — above
  `refine_tol` at every resolution measured, and falling only slowly, as
  the HLL diffusion spreads it over more cells the longer the run goes
  on. It will hold the cap in practice.

**(Measured in step 7: the criterion driving a real `regrid!`, and the
buffer against a real `λ_max · chunk`.)** The calibrated thresholds do what
they were calibrated to do on a mesh that moves: on the tracked Sod tube
the initial-data cycle converges in 3 passes in `D = 1` and 2 in `D = 2`,
the mesh holds the diaphragm at the cap, and every strongly firing cell
stays on a block at the cap for the whole run. The buffer table is in
[Measured results](#step-7--the-driver-and-the-tracked-shock-tube), and it
says something neither upstream package's did: TreeAMR measured a margin
narrower than the motion coming out *slightly worse* than none at all, and
TreeWave measured it coming out no worse; here the widths are **strictly
ordered**, every cell of margin buying both tracking and accuracy. The
difference is what the margin is around. TreeAMR's and TreeWave's narrow
margins were measured on features their criteria could resolve away, where
losing the feature and half-holding it cost the same; a shock fires
forever, so a partially covered shock is partially resolved and the
partial credit is real.

## Regridding: one driver, restart per chunk

Regridding changes the length and the meaning of the state vector, so
each chunk is a fresh `solve` and a fresh `HydroProblem` — TreeAMR's
prescription, TreeWave's and Burgers' practice. What is different here is
that **there is exactly one time-stepping loop** (decided):

    adapt_to_initial_data!(U, ops; initial, flags, buffer, boundary)
    p = HydroProblem(U, ops; eos, floors, limiter, riemann, fixup)
    while t < t_end
        λ  = max_signal_speed(p)                     # block_mapreduce over P
        dt = cfl · minimum_spacing(forest) / (D · λ)
        u  = solve(SSPRK33(stage_limiter! = reset_atmosphere!), hydro_rhs!,
                   u, t → stop; dt)                  # the reset, per stage
        scatter!(U, u); fill_ghosts!(U, …; boundary); con2prim!(P, U)
        assert dt ≤ cfl · h / (D · max_signal_speed(p))      # the CFL check
        record totals of every conserved variable, the injection, floor
            counts by population, block count
        observer(p, t, u)                            # before the regrid invalidates U
        flags = hydro_flags(P; refine_tol, coarsen_tol, cap)
        if regrid!(forest, (U => p.schedule, P => nothing, F_1 => nothing, …);
                   flags, buffer, boundary)
            p = HydroProblem(U, ops; …, prims = p.prims, fluxes = p.fluxes)
            u = statevector(U); gather!(u, U); reset_atmosphere!(u, p)
        end
    end

TreeWave has three near-identical loops and records that a fourth would
be the one to drift; the four cases here differ in their initial data,
boundaries, EOS parameters and references, and in nothing about how time
passes. So a **case is data** — a small struct holding the primitive
initial-data closure, the boundary hook or `nothing`, the periodicity,
the extents, `γ`, the floors, and the reference solution — and
`evolve!(case; …)` is the loop. This is a deliberate step past TreeWave's
"no abstraction over initial data", taken because the abstraction here
is over a *loop that already exists three times upstream*, not over
physics.

The loop returns what the tests assert on: the drift of each conserved
integral against its scale, the floor counts, the mesh statistics per
chunk, the tracking measure, the final state and forest, and — through
`observer(p, t, u)`, called with `U` scattered and `P` current — whatever
the viewer wants, so that `bin/` contains no time-stepping of its own.

**(Implemented in step 7.)** `src/driver.jl`: `HydroCase`, `evolve!`,
`uniform_run`, and the three measurements around them — `check_cfl`,
`tracked_share`, and `reduce_to_grid` / `l1_difference` after TreeAMR's
Burgers test. The numbers are in
[Measured results](#step-7--the-driver-and-the-tracked-shock-tube). What
the implementation settled:

- **The shape of the case.** `HydroCase` holds the primitive initial data
  as a pure `x -> P` closure, the EOS, the floors, the boundary hook or
  `nothing`, the periodicity per dimension, the physical extents, the root
  brick, `speed_headroom`, and an optional reference `(U, t) -> state
  vector`. The two case constructors, `HydroCase(::SodTube)` and
  `HydroCase(::EntropyWave)`, live *with their cases* in `sod.jl` and
  `entropywave.jl` rather than in the driver, which is what keeps the
  driver free of anything case-specific; the dependency runs from the case
  to the loop and never back. The struct carries its own working type, so
  `evolve!` takes `T` as a leading positional argument only to check it
  against the case's, and refuses a mismatch by name.
- **The headroom is a case parameter, and Sod's is 2.** This is the step-4
  amendment closed; see [Time integration and the time
  step](#time-integration-and-the-time-step) for what the driver does with
  it and for what `speed_headroom = 1` measures.
- **The criterion the initial-data cycle runs is the evolution's, on a
  scratch primitive set.** The cycle regrids `U` alone with
  `transfer = false` and re-evaluates the initial data, so a primitive set
  built before it would have the wrong number of blocks by the second pass,
  and there is no `HydroProblem` to borrow one from. `hydro_flags` and
  `max_signal_speed` therefore each gained a `FieldSet` method with the
  `HydroProblem` method forwarding — one implementation, two entry points —
  and the cycle's `flags` callback allocates a set per pass, at most
  `maxpasses` allocations of a mesh that is still small. The cycle
  converges in **3** passes on Sod in `D = 1` and **2** in `D = 2`.
- **`uniform_run` is `evolve!` with the cap at zero**, and not a second
  loop. With `maxlevel_cap = 0` nothing can refine and `Coarsen` is never
  issued below level 0, so the mesh never changes; the *fine* reference is
  the case's root brick scaled by `2^cap`, which is why `HydroCase` holds
  the brick and the extents separately — scaling every root count by the
  same factor leaves the box exactly where it was and the blocks cubes.
  The buffer derivation is skipped at a cap of zero, since a mesh that
  cannot refine has no margin to travel.
- **The loop does not regrid after the last chunk**, so the mesh, the block
  count and the state that come back all describe the same thing.
- **The observer is called once at `t = 0` and once per chunk**, with the
  state scattered into `U`, `P` current, and before the regrid that would
  invalidate either — TreeWave's contract, so that the viewers of step 11
  get the initial frame without a loop of their own.
- **The cell saving on Sod is modest, and that is the case's doing rather
  than the driver's.** Sod's three waves occupy 59% of the tube by
  `t = 0.2`; the tracked mesh saves 22% of the cells in `D = 1` at
  `t = 0.2` and 28% in `D = 2` at `t = 0.15`. What the tracked run claims
  is the fine reference's *error*, which it matches to four decimal places,
  and the cell count is the price of that match.

## Precision

Inherited from TreeWave wholesale: every driver takes `T` as a leading
positional argument defaulting to `Float64`; physical quantities carry
`T`, counts do not; **no floating-point literal in an expression where
`T` is in play** — `T(1//2)`, `T(7//5)`, `oftype(x[1], 2)` inside
closures; `wrap`/`ceilint`/`floorint`/`tofloat64` from a copied
`precision.jl` where `Base` does not serve a software float.

What differs is which cases each type can run:

| type | Sod, Sedov, entropy wave | Kelvin–Helmholtz |
|---|---|---|
| `Float64` | everything measured here | everything |
| `Float32` | same mesh, same floor counts, errors agreeing to ~1% (predicted, TreeWave's pattern) | same mesh and `M(t)` through the linear phase; then divergence by design |
| `Float32x2` | Sod and Sedov run (arithmetic and `sqrt` only; the references are host `Float64`) | does not run: `sin` and `exp` in the initial data, which MultiFloats lacks |

The two host reference codes — the exact Riemann solver and the Sedov
quadrature — stay `Float64` at every `T`, converted once at the
comparison, exactly as TreeWave's Hankel table does.

## Multi-threading

Nothing to configure and nothing to write: every kernel goes through
`map_blocks!` or is a KernelAbstractions launch of its own, every
reduction — `λ_max`, the conserved totals, the floor counts, the mode
amplitude — goes through `block_mapreduce` or `firing_boxes`, and the
per-cell callbacks (initial data, boundary, flagging) are pure. So the
package inherits TreeAMR's bit-identity across thread counts, and holds
itself to it the way TreeWave does: `test/thread_workload.jl` runs a
tracked Sod tube and a short Kelvin–Helmholtz and prints digests per
chunk, and `test/threading_tests.jl` compares a subprocess at another
thread count character for character. The Kelvin–Helmholtz is in the
workload on purpose: an instability is the case where a summation-order
difference would be *amplified* into a visible one, so it is the sharpest
place to guard the invariant.

## Running on a device

Also inherited: `backend` is a keyword beside `T` on every driver; the
spacings are uploaded once per `HydroProblem` with a copied
`to_backend`; every closure captures `isbits` only — the EOS struct, the
floors, tuples of thresholds, never a `Type` or a `Vector`; the viewer
takes a `hostcopy`. The two new device-side pieces are the `con2prim`
pass over the stored extent, which `map_blocks!(…; stored = true)`
launches, and the atmosphere reset, a pointwise kernel over the stage
vector the integrator hands the limiter hook — a device array when the
state lives on a device. Device tests
are opt-in behind `TREEHYDRO_TEST_BACKEND` as in TreeWave, and the
assertion worth having without a device — that the whole RHS reproduces
the host bit for bit on the CPU backend of a second run — is TreeAMR's
Burgers claim, which this package should reproduce for the system at
`Float32` when a device is present.

**(predicted)** The per-cell arithmetic of a hydro RHS — a `con2prim`,
two reconstructions and a Riemann solve per face — is an order of
magnitude more work per byte than the wave equation's Laplacian, so on
unified memory (Apple silicon) the RHS should show a device speedup where
TreeWave's showed none, and on an H200 it should beat the bandwidth
ratio. Measured in H6; the number is the first thing anyone will ask.

## File layout

| file | contents |
|---|---|
| `src/TreeHydro.jl` | module shell: `using`s, exports, includes |
| `src/precision.jl`, `src/device.jl` | copied from TreeWave (not a dependency on it): `Base` bridges for software floats; `to_backend`, `hostcopy` |
| `src/eos.jl` | `IdealGas`, `prim2con`, `con2prim`, `soundspeed` |
| `src/floors.jl` | `Floors`, `apply_floors`, the `reset_atmosphere!` stage limiter and its injection accounting |
| `src/reconstruction.jl` | the three slopes, face states |
| `src/riemann.jl` | LLF, HLLE, HLLC fluxes, direction-generic |
| `src/evolution.jl` | the three kernels (`con2prim_kernel!`, `flux_kernel!`, `divergence_kernel!`), `HydroProblem`, `hydro_rhs!`, `update_primitives!`, `max_signal_speed`, `floor_hits`, `hydro_dt`, the conserved totals and scales, `hydro_solve!`, `convergence_rate` |
| `src/refinement.jl` | the Löhner indicator on primitives, `hydro_flags`, `refinement_buffer` |
| `src/driver.jl` | `HydroCase`, `evolve!` — the one loop — `uniform_run`, and its diagnostics: `check_cfl`, `tracked_share`, `reduce_to_grid`, `l1_difference` |
| `src/exact_riemann.jl` | Toro's exact Riemann solver, host `Float64`, the shock-tube reference |
| `src/sedov_reference.jl` | the similarity law; later the Kamm–Timmes profile |
| `src/entropywave.jl`, `src/sod.jl`, `src/sedov.jl`, `src/kelvinhelmholtz.jl` | the four cases: initial data, parameters, references, per-case diagnostics |
| `src/benchmark.jl` | per-phase timings, TreeWave's format |
| `test/` | one `*_tests.jl` per case holding its unit, structural and physics claims together, plus `type_tests.jl`, `threading_tests.jl`, `device_tests.jl` and the standalone `thread_workload.jl` |
| `.github/workflows/CI.yml` | the one workflow: the whole suite on every push, over the Julia × OS matrix, at one thread and at four |
| `bin/visualize1d.jl` | the shock tube against the exact solution, per block, coloured by level, with `τ` and the conserved totals against time |
| `bin/visualize2d.jl` | the Kelvin–Helmholtz filmstrip and diagnostics; the Sedov filmstrip and radial scatter (`--case=`) |
| `bin/backend.jl`, `bin/benchmark.jl`, `bin/Project.toml` | as in TreeWave |

`Project.toml` depends on `TreeAMR`, `KernelAbstractions`,
`OrdinaryDiffEqSSPRK` and `SciMLBase`; tests add `MultiFloats`; `bin/`
adds `CairoMakie` and `SixelTerm` in its own environment. TreeAMR is
unregistered and is located through a `[sources]` entry, which puts the
Julia floor at 1.11 as it does for TreeWave. **The entry pins
`rev = "main"`**, as TreeWave's does (decided; the design was written
against the `m8` branch, which has since been merged).

Testset names are claims, each opening with the failure mode it guards;
measured numbers are recorded in this file when they change. Conventions
follow TreeAMR's `CLAUDE.md`, since the three packages are read together.

## Testing

*(Added in step 7b, rewritten in step 7c.)* **There is one suite and it
runs whole.** Every claim in "Measured results" comes from a test that
runs on every push, at one thread and at four, over the Julia × OS matrix
of `.github/workflows/CI.yml`: the convergence sweeps and their rates, the
interface-order tables and the negative control on the rate, the `D = 3`
runs, the refinement calibration, the tracked shock tube and its buffer
and prolongation-order tables. A number recorded in this file is a number
CI recomputes, and a regression shows up as a changed number rather than
as a test that merely still passes. The tests take under two minutes
locally at either thread count; a CI entry takes a few minutes, a shared
runner being slower and the rest of it precompilation.

**The one thing that must stay off is code coverage**, and the reason is
the whole of the history below. `julia-actions/julia-runtest` turns
coverage on by default, and nothing here consumes it — no upload step, no
badge — so it buys nothing. What it costs is a factor of a hundred.

Julia compiles a coverage hit into an atomic read-modify-write on one
global 64-bit counter per source line (`visitLine` in `src/codegen.cpp`
emits an `AtomicRMW` add at monotonic ordering), and `src/coverage.cpp`
packs the counters of 32 neighbouring lines into one 256-byte block. A
KernelAbstractions CPU launch is one task per thread over chunks of the
*same* kernel, so every thread executing a given kernel line does an
atomic add on the same cache line, and the cost scales with the parallel
work in the kernel. That is why it fell on the `D ≥ 2` runs and left the
one-dimensional ones alone.

Measured locally, on twelve cores with no oversubscription, on the
`D = 2` entropy-wave sweep at `N = (8, 16, 32)` with `--check-bounds=yes`:

| | 1 thread | 4 threads |
|---|---|---|
| coverage off | 1.83 s | 0.79 s |
| coverage on | 10.35 s | 79.13 s |

So coverage alone costs **5.7×** at one thread on kernel-heavy code and
**100×** at four, and it inverts the sign of threading: without it four
threads are 2.3× faster than one, with it 7.7× slower.

On CI the same thing, at the same shape. The step 6 push ran both jobs
under the default `--code-coverage=@<package path>` and
`--check-bounds=yes`, and took **5 m 14 in the test phase serially against
52 m 53 at four threads**. Per segment, from the interval between
consecutive `@info` lines in the two logs:

| segment | 1 thread | 4 threads | ratio |
|---|---|---|---|
| `D = 2` interface-order sweep | 1 m 53 | 31 m 06 | 16.5 |
| `D = 2` `p = 1`, `fixup = false` | 34 s | 9 m 27 | 16.6 |
| `D = 2` entropy wave, `:mc` | 13 s | 3 m 33 | 16.0 |
| `D = 2` entropy wave, `:none` | 13 s | 2 m 57 | 14.0 |
| `D = 3` entropy wave, `N = 4` | 6 s | 47 s | 8.0 |
| `D = 3` two-level | 14 s | 76 s | 5.3 |
| `D = 1` Sod sweep, `:minmod` | 8 s | 10 s | 1.3 |
| first segment (compile-dominated) | 68 s | 46 s | 0.7 |

With coverage off the whole suite runs in **1 m 45 at four threads and
1 m 58 at one** on the development machine — four threads faster than
one, as it was before any of this.

The rule that follows is one line of YAML: `coverage: false` in CI.yml,
and if coverage is ever wanted it goes on a *separate one-thread entry*
and leaves the threaded one alone. Nothing about the suite's contents has
to change; a `D ≥ 2` sweep is seconds of arithmetic and belongs wherever
the claim it makes belongs.

**Step 7b's diagnosis was wrong and is corrected here** (amended in step
7c). It read the same segment table as evidence that "on GitHub's 4-vCPU
shared runners the two-dimensional sweeps are 10–17× slower at four
threads than at one", blamed the runner's oversubscription, and split the
suite into a short tier and a long one to keep the sweeps off the threaded
job; it said in as many words that coverage "cost nothing measurable". It
was reading the right table and attributing it to the wrong cause — both
jobs had coverage on, so the runner was never the variable — and it set
`coverage: false` in the *same push* as the split, which is why the fast
CI that followed did not distinguish the two changes. The split, the
reduced configurations in `test/regression_tests.jl`, the stored
`test/references/*.toml` and `.github/workflows/Long.yml` were all undone
in step 7c; what step 7b is kept for is the paragraph below, which is a
real measurement and cost a red CI run to get.

**Across machines the numbers are *not* bit-identical**, which is a trap
for any future test that compares against stored numbers. Measured when
7b's reference files were first taken to CI: on all four runners — macOS
arm64 on the same Julia 1.13 as the generating machine included, with the
one differing dependency tested and cleared — the conserved totals of
order one came back **1 to 4 ulp** from the numbers this machine produces.
The mechanism is Base's `sum` and `mapreduce`, whose inner loops run under
`@simd` and may therefore reassociate: the arrangement of vectorized
partial sums follows the CPU target the code was compiled for, and
TreeAMR's `block_mapreduce` — hence every conserved total and every
volume-weighted norm here — rests on them. TreeAMR's bit-identity across
*thread counts* is untouched by this and is asserted; it is bit-identity
across *microarchitectures* that Base's reductions never offered. So a
claim of the form "this run produces this number" needs a tolerance of a
few hundred ulp of the quantity's own scale *and* an absolute floor, and a
claim about a *difference* of two order-one totals — every drift in this
package is one — cannot be made relatively at all. The claims in the suite
are made against bounds instead, which is why they survive the move.

## Milestones

Each has an acceptance test; serial `Float64` correctness first.

- **H0 — Scaffolding and prerequisites.** *(Done.)* The two TreeAMR
  prerequisites landed on TreeAMR's `main`; `Project.toml` with the
  `[sources]` pin to `main`, CI on 1.11 and release at one and four
  threads, `CLAUDE.md`, this document. *Accept:* a clean clone
  instantiates, an empty test suite passes, and the pinned TreeAMR
  provides both prerequisites.

  What H0 settled, beyond what was written above:

  - **The suite is not empty.** `test/prerequisite_tests.jl` asserts the
    two prerequisites rather than assuming them — that
    `fill_by_coordinates!(AllVariables(f), fs)` fills a field set with
    *bit-for-bit* the numbers the per-variable form does (the convergence
    studies compare against numbers the two forms must agree on, so
    agreement to roundoff would put a floor under every error this
    package measures), and that `map_blocks!(…; stored = true)` reaches
    every stored point with the kernel's index *as* the stored index,
    checked on a face-centered set with `G = (0, 1)` so that a
    per-dimension ghost width with a zero in it is what is tested. The
    rest of the M8 surface — `InterfaceSchedule`, `restrict_interfaces!`,
    `CellBoundary`, `boundary_by_coordinates`, `firing_boxes`,
    `block_mapreduce` — is checked as a list of exported names, which is
    the cheapest thing that fails when the pin moves under a rename.
  - **`hostcopy` is split.** `hostcopy(fs)` returns `fs` *itself* on the
    CPU, so on a machine with no device its copying path would be dead
    code — which is every machine CI runs on. The copy is therefore
    `hostcopy!(dst, src)`, callable host to host and tested that way; it
    checks that the two layouts agree, because a `copyto!` between arrays
    of equal length and unequal shape transposes the data instead of
    failing, and `G` and the centering are exactly what a destination
    built carelessly would get wrong. Only `hostcopy` is exported, as in
    TreeWave; `hostcopy!` is reached as `TreeHydro.hostcopy!`.
  - **The compat bounds.** `KernelAbstractions = "0.9.42, 1"` and
    `SciMLBase = "3.50.1"` follow TreeWave's, `OrdinaryDiffEqSSPRK =
    "2.3.2"` the version TreeAMR's test environment resolves,
    `TreeAMR = "0.1.0"`, `julia = "1.11"` for the `[sources]` entry.
    `OrdinaryDiffEqSSPRK` and `SciMLBase` are dependencies from H0 and
    unused until H1c, so that the floor is fixed before anything relies
    on it.
- **H1 — The scheme on a uniform mesh.** *(Done.)* EOS, `con2prim`, MUSCL
  with the three limiters, LLF, HLLE and HLLC, SSPRK33, the six-step RHS
  with `D` flux sets, the exact Riemann solver and the Dirichlet boundary
  hook — on a single-level forest, `D = 1, 2` (3D smoke).
  *Accept:* the entropy wave converges at second order in L1 and L∞ with
  `:none`, and in L1 with `:mc` (amended in step 3: the first draft said
  "near it" in both norms under a limiter, and the measurement says
  otherwise — `:mc` clips the smooth extrema and its L∞ rate is 1.35, see
  [Measured results](#measured-results)); Sod against the exact Riemann
  solution at an L1 rate in `[0.8, 1.0]`; direction independence bit for
  bit; every conserved integral constant to roundoff (every face is a
  same-level face, with or without the fixup — the control).

  What H1 measured, across steps 1–4, all of it in
  [Measured results](#measured-results) and all of it identical at one and
  at four threads:

  - **Second order on smooth data**: the entropy wave at L1 and L∞ rates
    2.015 and 2.024 in `D = 1`, 2.021 and 2.029 in `D = 2`, with `:none`;
    and the semi-discrete residual — one right-hand-side evaluation against
    the exact time derivative of the exact cell averages, with no time
    stepping in it — at 2.033.
  - **What a limiter costs on smooth data**: `:mc` keeps the L1 rate (2.006)
    and loses the L∞ one (1.354, falling toward the 1 the theory gives),
    which is why the convergence study runs with `:none`. The prediction was
    wrong and is amended rather than met.
  - **First order, near enough, on a discontinuous one**: Sod at an L1 rate
    of 0.903 under `:minmod` and 0.945 under `:mc` over `N = 16 … 128`,
    inside the predicted 0.8 to 1.
  - **Conservation on the uniform mesh**: all `D + 2` integrals to a few ulp
    of their own scale in `D = 1, 2, 3`, with the drift *per step* falling
    as `N` rises, and **bit-identical with the fixup and without it** —
    which is the control that gives H2's coarse-fine claim its meaning. No
    floor fired in any run of any case.
  - **The boundary flux where a boundary is physical**: Sod's momentum total
    moves by exactly `(p_L − p_R) · t_end · A` while its mass and energy
    totals do not move, which is the form the conservation claim takes when
    the domain integral is not constant.
  - **Direction independence, bit for bit, both halves**, over the whole
    stored array including the ghosts the boundary hook wrote.
  - **`λ_max` from the initial data is not a bound.** Sod's post-shock gas
    is 1.8522 times faster than anything at `t = 0`, which is the amendment
    under [Time integration and the time
    step](#time-integration-and-the-time-step) and the number that will size
    the driver's headroom factor in H3.

  The pieces H1 wrote but did not measure: HLLC, which is step 10's
  comparison; the two-level `hydro_forest`, which is H2's; and the boundary
  hook's other two call sites, `regrid!` and `adapt_to_initial_data!`, which
  are H3's.
- **H2 — Coarse-fine faces, static mesh.** *(Done.)* The two-level
  `hydro_forest`, the two two-level `sod_forest` configurations, the
  fixup, the boundary hook on a fine block. *Accept:* conservation of all
  `D + 2` integrals to roundoff with the fixup and a leak without, in
  `D = 1, 2, 3`, on the entropy wave and on Sod; the interface-order
  table for the system (predicted L∞ 1, 2, 2 at `p = 1, 3, 5`; L1 2, 2,
  2); a two-level Sod whose refined region touches the Dirichlet
  boundary.

  What H2 measured, all of it in
  [Measured results](#measured-results) and all of it identical at one and
  at four threads. Step 5 added almost no code — `sod_forest`'s `refined`
  option and a one-line `forest_levels` — and the whole of it is
  `test/interface_tests.jl`:

  - **The fixup is what conserves.** On the static two-level mesh, with
    the fixup every one of the `D + 2` integrals holds to a *hundredth* of
    one ulp of its own scale per step in `D = 1, 2, 3`; without it — the
    same mesh, the same block count, the same step count, one line
    different — every one of them leaks by `1e8` to `1e9` times that
    bound. This is TreeAMR's M8b claim for a system, and the 3D entry is
    where a coarse-fine face carries four fine faces.
  - **The interface-order rule carries over to the system unamended.** L∞
    rates **0.963 / 2.034 / 2.037** in `D = 1` and **0.925 / 2.041 /
    2.037** in `D = 2` for `p = 1, 3, 5`, against unrefined controls of
    2.024 and 2.029, with `p = 3` landing within 0.013 of the control and
    `p = 5` buying nothing further; every L1 rate is the scheme's own,
    `p = 1` included. Predicted before it was run.
  - **The norm is part of the result, and the reason is conservation.**
    With `fixup = false` at `p = 1` the L1 rate falls from 1.966 to
    **1.111** in `D = 1` and from 1.900 to **1.192** in `D = 2`, while L∞
    is 1.0 either way. TreeAMR measured 1.98 → 1.12 and 1.80 → 1.25 on
    Burgers.
  - **Sod conserves across a coarse-fine face its shock crosses.** The
    shock leaves the refined box at `t = 0.14268` of a `t_end = 0.2` run.
    With the fixup the mass and energy drift is 0.15 to 2.8 times the
    uniform mesh's at the same coarse spacing — the coarse-fine face adds
    nothing to the boundary's own numerical flux — and the momentum total
    moves by the closed-form boundary flux to `5e-11` relative; without it
    all three are `6.3e3` to `7.7e7` times worse. At `N = 32` in `D = 1`,
    where the boundary's own flux has itself reached roundoff, the claim
    is the plain one: `0.0`, `2.2e-16` and `2.8e-17` against bounds of
    `2.8e-13`, `6.9e-13` and `9.0e-14`.
  - **The Dirichlet hook fills a fine block's outer ghosts**, all of them,
    including the ghost rows across the tube, with zero mismatches after
    the run.
  - **The two-level tube costs 1.338 times the uniform fine run's L1
    error** at the same finest spacing, in both dimensions — a sanity
    bound, since the refined region here is static.

  What H2 wrote but did not decide: the `p = 1` question, which the smooth
  wave cannot answer (it confirms the premise — an integral norm does not
  see the interface defect — and leaves the answer to the discontinuous
  cases of steps 7 and 9).
- **H3 — Regridding.** *(Done.)* The criterion, the buffer, `evolve!`, the
  initial-data cycle. *Accept:* the cycle converges to a fixed hierarchy
  on the initial data it is given — Sod's and the entropy wave's here;
  Sedov's and Kelvin–Helmholtz's arrive with their cases in H4 and H5, and
  run through this same cycle; the tracked Sod tube in `D = 1, 2` matches
  the uniformly fine reference at fewer cells with the uniform coarse
  mesh as control, conserves to roundoff through the regrids, and leaks
  without the fixup; the `p = 1` against `p = 3` table for Sod; the
  buffer-width table.

  What step 6 measured, all of it in
  [Measured results](#step-6--the-refinement-criterion) and all of it
  identical at one and at four threads:

  - **The indicator fires where the physics is and nowhere else.** Sod's
    initial data fires in exactly the two cells straddling the diaphragm
    and at exactly zero everywhere else; a top-hat pressure in a uniform
    density fires on `p` with the density's indicator exactly zero, which
    is Sedov's initial data and the case a criterion on `ρ` alone would
    miss entirely; a pure contact fires with the pressure's indicator
    exactly zero.
  - **The global term is what silences the atmosphere.** Six orders below
    the data, with `O(1)` relative noise: `τ = 0.0020` with
    `ε_g = 1/1000` and **0.971** with `ε_g = 0`, every cell of it firing
    in the second case. The negative control is the measurement.
  - **The thresholds, calibrated against `h`**: `refine_tol = 0.08` and
    `coarsen_tol = 0.02`, mid-plateau of the depth the shear layer's ramp
    reaches. The ramp is what picks them because the discontinuities
    cannot — they exceed any usable threshold at every resolution.
  - **Two predictions amended.** A captured shock scores 0.57 rather than
    the ≈ 1 of a sharp jump, while keeping the property that matters (it
    does not fall with `h`, so `maxlevel_cap` binds there); and a kink
    does *not* fire forever — a numerically computed rarefaction head
    falls at first order in `h`.

  What step 7 measured, all of it in
  [Measured results](#step-7--the-driver-and-the-tracked-shock-tube) and
  all of it identical at one and at four threads. Step 7 is the first step
  whose mesh *moves*:

  - **A tracked shock matches the uniformly fine reference at fewer
    cells.** The tracked Sod tube's L1 error against the exact Riemann
    solution is **1.0004** times the uniform fine run's in `D = 1` and
    **1.0000** in `D = 2`, at 200 cells against 256 and 1472 against 2048,
    with the uniform coarse control at **3.678** and **1.850** — and
    reduced onto the grid the meshes share, the tracked and fine runs
    differ by three to four orders of magnitude less than the coarse and
    fine runs do. The static two-level mesh of H2 cost 1.338 for the same
    finest spacing; a mesh that follows the waves costs essentially
    nothing.
  - **The mesh really does follow them.** `tracking == 1.0` in both
    dimensions: at every chunk, every cell whose indicator exceeded
    `refine_tol` sat on a block already at the cap. The initial-data cycle
    converges in 3 passes and 2, and puts the diaphragm's two blocks at the
    cap.
  - **Conservation survives a mesh rebuilt under the solution.** Mass
    `3.3e-16` and energy `1.6e-15` in `D = 1` over 588 steps and 9 mesh
    changes, `4.2e-17` and `1.9e-16` in `D = 2`, the momentum equal to its
    closed-form boundary flux to `8.6e-16` and `1.0e-16`; `fixup = false`
    leaks `4.9e8` to `7.4e9` times more, on the same step count and the
    same mesh history, and the single-level control is bit-identical either
    way.
  - **The headroom is what the recheck is for.** `speed_headroom = 1` on
    Sod throws in the **first chunk** — a CFL number of 0.6236 against the
    requested 0.4 — and at 2 the run completes with `λ_end ≤ 2λ` at every
    chunk. That closes the step-4 amendment.
  - **The buffer table is new**, and reproduces neither upstream finding:
    the widths are strictly ordered in both tracking and error, because a
    shock fires forever and a partially covered one is partially resolved.
  - **The first discontinuous row of the `p = 1` question**, and it goes
    against `p = 1`: 3.6% worse in L1, 8% more cells, tracking lost to
    0.9091, and no floor count to buy — which is exactly why Sod cannot
    close the question and Sedov must.

  What step 7 wrote but did not exercise: the atmosphere reset, whose
  place in the loop is a comment and whose keyword is already in the
  signature (step 8), and every case but Sod and the entropy wave.
- **H4 — Sedov.** Floors and the atmosphere reset, the `D = 2` and
  `D = 3` blasts, the similarity checks, the hook on edges and corners.
  *Accept:* the exponent and the jump; the reset idempotent and `U`/`P`
  consistent after it; the injection measured and the drift net of it
  at roundoff, `:stage` against `:step`; floor counts by population; the
  shell tracked to the
  uniform fine reference at fewer cells; block count rising then
  falling behind the shock; the `p = 1` / `p = 3` comparison and the
  ghost-floor count that decides the upstream question. The 3D adaptive
  run at test size.
- **H5 — Kelvin–Helmholtz.** The McNally setup, `M(t)` and the kinetic
  energy diagnostic, HLLC, the viewer. *Accept:* `M(t)` grows below the
  incompressible bound and converges toward the uniform fine run as the
  cap rises; conservation through regrids; HLLE against HLLC measured and
  the default chosen; the filmstrip rendered in CI.
- **H6 — Precision, threads, device.** `T` and `backend` on every
  driver, the type table above, the thread workload, device tests, the
  benchmark. *Accept:* `Float32` reproduces the Sod and Sedov meshes and
  floor counts; `Float32x2` runs Sod and Sedov; digests identical across
  thread counts; the per-phase device table with the RHS speedup on
  unified memory measured.
- **H7 — Higher-order reconstruction** *(optional)*. PPM or WENO-Z at
  `G = 3`, and the interface-order rule re-measured against it.

## Measured results

This section takes the numbers as the milestones produce them, each beside
the prediction it confirms or corrects. Everything here is `Float64` on the
CPU backend, and every number is identical at one and at four threads.

### Step 3 — the entropy wave on the uniform mesh

The M3 box with `roots = 4` left unrefined (`roots = 2` in `D = 3`),
`ρ₀ = 1`, `a = 1/5`, `v_d = 1`, `p₀ = 1`, `γ = 7/5`, `L = 1`, HLLE,
`cfl = 2/5`, to `t = 1/4` (`t = 1/8` in `D = 3`). Rates are the
least-squares slope of the volume-weighted error of the whole `D + 2`
component state vector against `h`, over the `N` listed.

| `D` | limiter | `N` | L1 rate | L∞ rate |
|---|---|---|---|---|
| 1 | `:none` | 8, 16, 32, 64 | **2.015** | **2.024** |
| 2 | `:none` | 8, 16, 32 | **2.021** | **2.029** |
| 1 | `:mc` | 8, 16, 32, 64 | **2.006** | **1.354** |
| 2 | `:mc` | 8, 16, 32 | **1.894** | **1.351** |

Second order with `:none` in both norms and both dimensions, as predicted.
With `:mc` the L1 rate survives and **the L∞ rate does not**: 1.35, and
falling as `N` grows (1.49, 1.25, 1.36 between consecutive `D = 1` pairs;
1.55 then 1.15 in `D = 2`). This is the TVD limiter clipping the sine's
two smooth extrema, where it is first order by construction, so L∞ tends
to 1 while the shrinking width of the clipped region leaves L1 at 2. The
plan predicted "≥ 1.5 with `:mc`"; that prediction was wrong in L∞ and the
tests assert 1.2, with the measured value recorded here rather than the
threshold moved to meet it. It is also the reason the convergence study is
run with `:none`: under a limiter the study measures the limiter.

**Conservation, all `D + 2` integrals.** Worst over the `D + 2` variables
of `|Σ hᴰ U_v(t_end) − Σ hᴰ U_v(0)|`, in ulp of that variable's own scale
`Σ hᴰ |U_v|`:

| `D` | limiter | steps | drift, ulp of the scale | per step |
|---|---|---|---|---|
| 1 | `:none` | 47 … 372 | 0.67 … 2.0 | ≤ 0.015 |
| 2 | `:none` | 93 … 372 | 1.0 … 17.1 | ≤ 0.047 |
| 1 | `:mc` | 47 … 372 | 0.67 … 1.33 | ≤ 0.015 |
| 2 | `:mc` | 93 … 372 | 1.71 … 4.0 | ≤ 0.019 |
| 3 | `:none` | 18 | 1.5 | 0.084 |

Roundoff, and not growing with the step count — the drift per step *falls*
as `N` rises. **With `fixup = false` every number above is bit-identical**,
which is the uniform control: `restrict_interfaces!` has no coarse-fine
face to act on, so switching it off must change nothing, and the tests
assert the two runs agree bit for bit rather than merely both conserving.
No floor fired anywhere (`floor_hits == 0` in every run), so the
conservation claim here is the plain one and not one net of an injection.

**The semi-discrete residual** — one right-hand-side evaluation on the
exact initial averages against the exact time derivative of those
averages, `D = 1`, `:none`, `N = 8, 16, 32` — converges at **2.033** in
the volume-weighted L∞ norm. It is the sharpest and cheapest statement
that the space discretization is second order, since no time stepping and
no approximate reference enter it.

### Step 4 — Sod on the uniform mesh

Sod's states on `[0, 1]`, `γ = 7/5`, `:minmod`, HLLE, `cfl = 2/5`, to
`t = 1/5`, Dirichlet along the tube and periodic across it, `roots = 4`
along the tube and one across. Errors are the volume-weighted norms of the
whole `D + 2` component state vector against the exact Riemann solution
sampled at cell centers.

**The exact solver against Toro's Table 4.3**, tests 1, 2 and 3, each to
one unit in the last decimal the table prints:

| test | `p★` | `u★` | `ρ★_L` | `ρ★_R` | Newton steps |
|---|---|---|---|---|---|
| 1, Sod | 0.3031301781 | 0.9274526200 | 0.4263194282 | 0.2655737117 | 4 |
| 2, the 123 problem | 0.0018938734 | 0.0 | 0.0218521182 | 0.0218521182 | 1 |
| 3, the left blast | 460.89378749 | 19.597451389 | 0.5750622985 | 5.9992407048 | 4 |

Every entry also agrees to `1e-4` *relative* except test 2's `p★`: the
table prints 0.00189, which is five decimal places but only three
significant digits, and the agreement is 3.9e-6 absolute, 2.0e-3 relative.
That is the table's rounding, recorded rather than absorbed into a looser
tolerance.

**L1 convergence**, `D = 1`:

| `N` | cells | steps | L1 | L∞ |
|---|---|---|---|---|
| 16 | 64 | 71 | 1.668e-2 | 3.344e-1 |
| 32 | 128 | 141 | 8.501e-3 | 1.997e-1 |
| 64 | 256 | 281 | 4.537e-3 | 2.226e-1 |
| 128 | 512 | 562 | 2.553e-3 | 3.398e-1 |

**L1 rate 0.903**, inside the predicted 0.8 to 1, with the errors falling
monotonically. L∞ is recorded and nothing is asserted on it: on a solution
with a shock in it, L∞ is the error in the one cell nearest the
discontinuity and does not even decrease with `h`. Under `:mc` the same
sweep gives 0.945, which is the number to beat when HLLC is measured.

**The signal speed, and why the driver will need headroom.**

| | value |
|---|---|
| `λ` from the exact solution, `u★ + c★_R` | **2.191566** |
| `λ` from the initial data, `c_L = sqrt(7/5)` | 1.183216 |
| **ratio** | **1.85221** |
| `u★ + c★_L`, the gas behind the contact | 1.925175 |
| shock speed | 1.752156 |

The fastest signal is in the post-shock gas, which does not exist at
`t = 0`. The *discrete* state overshoots the exact supremum slightly at the
discontinuity — `λ_final/λ` is 1.00491, 1.00175, 1.00035, 1.000056 at
`N = 16 … 128` under `:minmod` and 1.00775 … 1.00439 under `:mc` — because
a reconstruction of a jump produces a face state the exact solution does
not contain. Both numbers are recorded against
[Time integration and the time step](#time-integration-and-the-time-step).

**Direction independence, bit for bit, both halves.** With `N = 16`,
`roots = 4` along the tube and the same 71 steps in every run:

- the `D = 2` tube along `y`, transposed and with its two momentum
  components swapped, equals the tube along `x` in **every entry of the
  whole stored array** — ghosts included, blocks matched by their
  transposed origins, 0 differences out of 25 600 comparisons;
- the `D = 2` planar tube's profile equals the `D = 1` run's in every
  entry, and `S_y` is exactly zero everywhere.

The two `D = 2` runs' `l1` differ in the last ulp, which is the same
numbers summed in a different order; the claim is made on the solution and
not on the norm, and the `D = 1` and `D = 2` `l1` differ by exactly 3/4
because the norm divides by the number of stored entries and there are
four variables rather than three.

**The drift is the boundary flux.** With a physical boundary the domain
integral is not constant, so the entropy wave's claim does not apply. Until
a wave arrives both sides of each boundary face are in that face's own
initial state — at rest — so the only nonzero flux there is the pressure,
in the momentum row. At `N = 16`:

| `D` | momentum drift | `(p_L − p_R)·t_end·A` | mass | energy | `S_y` |
|---|---|---|---|---|---|
| 1 | 0.17999999994 | 0.18 | 2.1e-11 | 4.7e-11 | — |
| 2 | 0.04499999998 | 0.045 | 5.8e-12 | 1.3e-11 | **0** |

Mass and energy would cross exactly nothing; the `1e-11` is the numerical
rarefaction's foot having diffused a little way toward the left boundary by
`t = 1/5`, and it is roundoff from `N = 32` on. The transverse momentum
drifts by *exactly* zero. **No floor fired in any run**, in any dimension,
at any resolution.

### Step 5 — the static two-level mesh

The M3 two-level box for the entropy wave (`roots = 4`, the middle
sub-box refined once, 2:1 balanced, held fixed in physical space) and the
two static two-level tubes for Sod. `Float64`, HLLE, `cfl = 2/5`;
everything below is identical at one and at four threads.

**Conservation, all `D + 2` integrals, and the negative control.** The
entropy wave with `:none` and `p = 3` to `t = 1/4` (`t = 1/8` in `D = 3`).
The two runs differ in the single line `p.fixup && restrict_interfaces!(…)`
and agree on the mesh, the block count and the step count:

| `D` | `N` | blocks | steps | worst drift with the fixup | worst drift without |
|---|---|---|---|---|---|
| 1 | 8 | 6 | 93 | **0.011** ulp of the scale per step | **1.02e-4** of the scale |
| 2 | 8 | 28 | 186 | **0.0031** | **3.21e-5** |
| 3 | 4 | 120 | 70 | **0.0071** | **2.98e-5** |

"Worst" is over the `D + 2` integrals, each against its own scale
`Σ hᴰ |U_v|`. With the fixup every one of them sits at a *hundredth* of
one ulp of its scale per step; without it every one of them leaks by
`1e8` to `1e9` times the `8 eps · scale · nsteps` bound the fixup run
meets, and by `1e-5` to `1e-4` of the scale. No floor fired in any run.
The 3D entry is the one where a coarse-fine face carries four fine faces
and the fixup's tangential average is a 2×2 rather than a single cell.

**The interface-order rule for the system.** The same wave, `:none`, over
`N = 8 … 64` in `D = 1` and `N = 8 … 32` in `D = 2`, conservative
restriction (exact, and therefore never entering), prolongation order
varied:

| prolongation | L∞, `D = 1` | L∞, `D = 2` | L1, `D = 1` | L1, `D = 2` |
|---|---|---|---|---|
| 1 | **0.963** | **0.925** | 1.966 | 1.900 |
| 3 | 2.034 | 2.041 | 1.971 | 2.009 |
| 5 | 2.037 | 2.037 | 1.976 | 2.009 |
| *unrefined control, `p = 3`* | 2.024 | 2.029 | 2.015 | 2.021 |

The prediction — 1, 2, 2 in L∞ and 2, 2, 2 in L1 — holds, in both
dimensions, without amendment. Order 1 costs a full order in L∞; order 3
recovers the scheme's own rate, landing within **0.013** of the unrefined
control in `D = 1` and **0.012** in `D = 2`, which is the sharper statement
that the interface has stopped being what limits it; order 5 buys nothing
further, and in `D = 2` is fractionally *worse* than order 3 in the third
digit. Every L1 rate is the scheme's own, `p = 1` included.

**The negative control on the rate**, which is what pins the locality of
the interface defect on conservation rather than on the flux-divergence
form:

| | L1, `D = 1` | L1, `D = 2` | L∞, `D = 1` | L∞, `D = 2` |
|---|---|---|---|---|
| `p = 1`, fixup | 1.966 | 1.900 | 0.963 | 0.925 |
| `p = 1`, **no fixup** | **1.111** | **1.192** | 1.021 | 1.042 |

TreeAMR measured 1.98 → 1.12 and 1.80 → 1.25 on Burgers; the system
reproduces it. With the fixup the defect is a dipole — the fine cell loses
exactly what the coarse cell gains — and a first-order hyperbolic operator
carries a zero-mean residual nowhere; without it the residual has net mass
`O(h)` per unit time and is transported downstream as an `O(h)` plateau,
which an integral norm does see. L∞ never depended on the fixup at all.

**Sod across a coarse-fine face.** `:minmod`, `p = 3`, `N = 16`,
`roots = 4` along the tube and one across, to `t = 1/5`, with the
Dirichlet boundary in place. The shock leaves the `:middle` box at
`t = 0.14268` and ends at `x = 0.85043`, so it crosses the coarse-fine
face at `x = 3/4` during the run. Drifts, against the closed-form
boundary flux `(p_L − p_R)·t_end·A` for the momentum (0.18 in `D = 1`,
0.045 in `D = 2`, since `Σ hᴰ |S|` is exactly zero at `t = 0`):

| run | blocks | steps | mass | energy | \|ΔS − flux\| |
|---|---|---|---|---|---|
| `D = 1` `:middle` | 6 | 141 | 8.6e-12 | 2.4e-11 | 9.1e-12 |
| `D = 1` `:left` | 6 | 141 | 4.7e-11 | 1.3e-10 | 5.0e-11 |
| `D = 1` uniform, same coarse `h` | 4 | 71 | 2.1e-11 | 4.7e-11 | 6.0e-11 |
| `D = 1` `:middle`, **no fixup** | 6 | 141 | **3.3e-4** | **1.5e-3** | **7.0e-4** |
| `D = 1` `:left`, **no fixup** | 6 | 141 | **2.2e-3** | **5.8e-3** | **3.1e-4** |
| `D = 2` `:middle` | 10 | 281 | 2.2e-12 | 6.1e-12 | 2.3e-12 |
| `D = 2` `:left` | 10 | 281 | 1.2e-11 | 3.3e-11 | 1.3e-11 |
| `D = 2` uniform, same coarse `h` | 4 | 141 | 5.8e-12 | 1.3e-11 | 1.6e-11 |
| `D = 2` `:middle`, **no fixup** | 10 | 281 | **8.3e-5** | **3.8e-4** | **1.8e-4** |
| `D = 2` `:left`, **no fixup** | 10 | 281 | **5.4e-4** | **1.4e-3** | **7.9e-5** |

The refined runs with the fixup drift by **0.15 to 2.8 times** what the
uniform mesh of the same coarse spacing drifts by, so the coarse-fine face
adds nothing to the boundary's own numerical flux; the runs without it
leak by **6.3e3 to 7.7e7 times** more, and miss the momentum's closed form
by `1.7e-3` to `3.9e-3` relative where the runs with it meet it to
`5.2e-11` … `2.8e-10`. No floor fired in any of them, and the transverse
momentum drifts by *exactly* zero in `D = 2` with the fixup and without it.

**And it is roundoff where the boundary's own flux is.** The residual
above is not roundoff: it is the numerical foot of the rarefaction and the
shock reaching the Dirichlet faces, and it falls by four orders of
magnitude per halving of `h` — `2.1e-11`, `1.1e-16`, `0.0` for the mass on
the uniform `D = 1` mesh at `N = 16, 32, 64`. At `N = 32` in `D = 1` the
two-level tube therefore obeys the entropy wave's plain bound on a mesh
with a coarse-fine face in it:

| run | mass | energy | \|ΔS − flux\| |
|---|---|---|---|
| `:middle`, `N = 32`, 281 steps | **0.0** | **2.2e-16** | **2.8e-17** |
| `:left`, `N = 32`, 281 steps | **1.1e-16** | **6.7e-16** | **2.8e-17** |
| `8 eps · scale · nsteps` | 2.8e-13 | 6.9e-13 | 9.0e-14 |
| the same runs **without the fixup** | 1.8e-4 … 1.1e-3 | 8.0e-4 … 2.9e-3 | 1.5e-4 … 3.4e-4 |

`D = 2` reaches the same point at `N = 32` and costs eight times as much
to say it, so it is recorded here rather than run in the suite.

**The boundary hook on a fine block.** With `refined = :left` the low
physical face is covered by level-1 blocks (one in `D = 1`, two in
`D = 2`), and after the run **every** stored entry of the outward-facing
ghost regions along the tube holds the conserved left or right state
exactly — the rows across the tube included — with zero mismatches out of
the whole slab.

**The error the refinement buys.** The two-level `:middle` tube against the
exact Riemann solution, in the volume-weighted L1 norm, beside the two
uniform runs that bracket it:

| `D` | two-level `:middle` | uniform, same coarse `h` | uniform, same finest `h` | ratio to the fine one |
|---|---|---|---|---|
| 1 | 1.1372e-2 | 1.6678e-2 | 8.5013e-3 | **1.338** |
| 2 | 8.5295e-3 | 1.2517e-2 | 6.3769e-3 | **1.338** |

The same ratio in both dimensions, to four digits. It is a sanity bound and
not a tracking claim: the refined region here is static and half the tube,
so the shock spends most of the run outside it. The claim that refinement
following the shock matches the uniformly fine run at fewer cells is step
7's.

### Step 6 — the refinement criterion

`ε = 1/100`, `ε_g = 1/1000`, `Float64`, `D = 1` unless said otherwise, and
every number identical at one and at four threads. Nothing here is
regridded: these are statements about the indicator on a given mesh.

**What a discontinuity scores, and what the atmosphere scores.** Sod's
initial data on a uniform mesh, `roots = 4`, `N = 8`: exactly the two
cells straddling the diaphragm fire, at **`τ = 0.9657`** and **`0.9847`**,
and every other cell scores **exactly zero**. The two blocks that meet at
`x₀` report exactly those cells as their boxes (`N:N` and `1:1`) and every
other block is a bare `Keep`. In `D = 2` the same holds with a whole
column of cells firing in each. Against that, a synthetic atmosphere six
orders below the data, alternating cell by cell between `ρ_atm` and
`2ρ_atm` and likewise in `p`:

| | max τ in the noisy region |
|---|---|
| `ε_g = 1/1000` | **0.0020** |
| `ε_g = 0` (the negative control) | **0.971** |

Every cell of the noisy region fires with `ε_g = 0`; none of them reaches
even `coarsen_tol` with it. This is the whole justification of the global
term, and the two rows are one test.

**Sod after a short evolution.** `t = 0.1`, `:minmod`, HLLE, `D = 1`,
`roots = 4`, uniform, `cfl = 2/5`. Max `τ` in a ±0.03 window around each
feature, located from the exact solution (`head` at `x = 0.38168`, `tail`
at `0.49297`, contact at `0.59275`, shock at `0.67522`); "inside the fan"
is the middle 40% of the interval between head and tail.

| `h` | shock | contact | fan head | fan tail | inside the fan |
|---|---|---|---|---|---|
| 1/64  | 0.5413 | 0.1871 | 0.1736 | 0.3301 | 0.09010 |
| 1/128 | 0.5975 | 0.2233 | 0.1050 | 0.3713 | 0.03999 |
| 1/256 | 0.5722 | 0.1806 | 0.0569 | 0.1750 | 0.01442 |
| 1/512 | 0.5858 | 0.1279 | 0.0286 | 0.0786 | 0.00476 |

Three of the four expectations held and one did not; the amendments are
recorded under [The refinement
criterion](#the-refinement-criterion). **The shock does not resolve** —
flat to within 10% over a factor of eight in `h` — which is what makes
`maxlevel_cap` load-bearing rather than a safety net, but it scores 0.57
and not the ≈ 1 of a sharp jump, because capture spreads it over three or
four cells. **The fan's interior falls faster than first order and
approaches second** (ratios 2.25, 2.77, 3.03), as the local `ε` term takes
over the denominator. **The fan's kinks fall too**, the head at almost
exactly first order — the prediction that a kink fires forever is wrong
for a numerically computed rarefaction. **The contact** stays above
`refine_tol` throughout and falls slowly.

**The McNally density ramp.** `ρ` from 1 to 2 over an exponential ramp of
width `L = 1/40` at uniform `p = 5/2`, one dimension, no evolution — the
shape of the Kelvin–Helmholtz shear layer, which is the feature the
criterion exists to resolve *and stop*:

| `h` | max τ | ratio |
|---|---|---|
| 1/64  | 0.25354 | |
| 1/128 | 0.12148 | 2.09 |
| 1/256 | 0.05316 | 2.29 |
| 1/512 | 0.02099 | 2.53 |
| 1/1024 | 0.00736 | 2.85 |

(The last row is measured but not asserted in the suite.) The depth this
reaches from a `1/64` base, which is the table the thresholds are read
off, in TreeWave's manner:

| `refine_tol` | depth reached | finest `h` |
|---|---|---|
| 0.03, 0.05 | 3 | 1/512 |
| 0.075, 0.08, 0.10, 0.12 | **2** | 1/256 |
| 0.15, 0.20 | 1 | 1/128 |
| 0.30 | 0 | 1/64 |

Any `refine_tol` in `(0.053, 0.121)` terminates at two levels, so the
defaults for the cases are **`refine_tol = 0.08`** — the geometric middle
of that plateau — and **`coarsen_tol = 0.02`**, a quarter of it as
TreeWave's is a quarter of its own, and ten times the atmosphere's score.
Note what the plateau is *not* about: on Sod the shock and the contact
exceed 0.08 at every resolution and refine to the cap, which is intended.
The plateau is about the smooth feature, because that is the only one
whose refinement the indicator can terminate.

### Step 7 — the driver and the tracked shock tube

`Float64`, the CPU backend, `:minmod`, HLLE, `cfl = 2/5`, the conservative
family at `p = 3` unless said otherwise, `refine_tol = 0.08`,
`coarsen_tol = 0.02`, `ε = 1/100`, `ε_g = 1/1000`, and every number
identical at one and at four threads. Two configurations:

| | `roots` | `N` | cap | `chunk` | `t_end` | headroom |
|---|---|---|---|---|---|---|
| `D = 1` | `(8,)` | 8 | 2 | 1/200 | 1/5 | 2 |
| `D = 2` | `(8, 1)` | 8 | 1 | 1/200 | 3/20 | 2 |

The cadence is not free: the derived margin covers
`speed_headroom · λ · chunk` at the cap's spacing, and TreeAMR's
recruitment reaches one ring of neighbours, so the travel must stay under
one finest-level block width — 1/32 in `D = 1` and 1/16 in `D = 2`. With
`λ = 2.20` and a headroom of 2 that means `chunk < 0.0071` and
`chunk < 0.014`; `chunk = 1/50` is **refused** by `refinement_buffer`,
naming the constraint. The derived width comes out **6** cells on the first
regrid and **7** on every later one in `D = 1`, and **4** throughout in
`D = 2`.

**The tracked tube against its two uniform references.** L1 of the whole
`D + 2`-component state against the exact Riemann solution, volume
weighted; "fine" is the uniform mesh at the tracked run's *finest* spacing
and "coarse" the uniform mesh at its *coarsest*, both through the same
loop with the cap at zero and the same chunk.

| | L1 | ratio to fine | cells | blocks | steps |
|---|---|---|---|---|---|
| `D = 1` tracked | **4.540016e-3** | **1.0004** | 200 | 25 | 588 |
| `D = 1` fine | 4.538238e-3 | 1 | 256 | 32 | 588 |
| `D = 1` coarse | 1.669060e-2 | **3.6778** | 64 | 8 | 156 |
| `D = 2` tracked | **6.215185e-3** | **1.0000** | 1472 | 23 | 431 |
| `D = 2` fine | 6.215186e-3 | 1 | 2048 | 32 | 431 |
| `D = 2` coarse | 1.149851e-2 | **1.8501** | 512 | 8 | 216 |

The same claim without the exact solution in it, through
`reduce_to_grid` onto the grid the meshes have in common (`64` cells in
`D = 1`, `64 × 8` in `D = 2`) and `l1_difference`:

| | tracked − fine | coarse − fine |
|---|---|---|
| `D = 1` | **1.745e-5** | 1.073e-2 |
| `D = 2` | **2.219e-7** | 5.292e-3 |

`tracking == 1.0` in both dimensions: at every one of the 40 chunks in
`D = 1` and 30 in `D = 2`, every cell whose indicator exceeded
`refine_tol` sat on a block already at the cap. The initial-data cycle
converged in **3** passes in `D = 1` and **2** in `D = 2`; the mesh grew
from 12 to 25 blocks in `D = 1` and 14 to 23 in `D = 2`, changing 9 and 3
times; no floor fired in any run.

**Conservation through the regrids, and the negative control.** The bound
on mass and energy is step 5's — the larger of `8 eps · scale · nsteps` and
ten times the uniform mesh's drift at the same coarse spacing, since a
physical boundary's own numerical flux is what remains there — and the
momentum's yardstick is the closed form `(p_L − p_R)·t_end·A` it must
equal (0.18 in `D = 1`, 0.016875 in `D = 2`), because Sod starts at rest
and `Σ hᴰ |S|` gives the momentum no scale at all.

| | mass | energy | \|Δmomentum − flux\| |
|---|---|---|---|
| `D = 1` tracked | 3.331e-16 | 1.554e-15 | 8.604e-16 |
| `D = 1` `fixup = false` | 1.385e-6 | 1.793e-6 | 6.350e-6 |
| ratio | **4.2e9** | **1.2e9** | **7.4e9** |
| `D = 2` tracked | 4.163e-17 | 1.943e-16 | 1.041e-16 |
| `D = 2` `fixup = false` | 4.309e-8 | 1.508e-7 | 5.116e-8 |
| ratio | **1.0e9** | **7.8e8** | **4.9e8** |

The two runs of each pair take the **same number of steps and regrid at
the same chunks with the same block counts**, so they differ in one line
and nothing else — the leak does not move the criterion. The transverse
momentum in `D = 2` is *exactly* zero either way. And the uniform control
is bit-identical with the fixup and without it, which is what says the
leak belongs to the coarse-fine face and not to the driver.

**The buffer-width table** (`D = 1`, everything else as above):

| buffer | L1 | tracking | cells | mesh changes |
|---|---|---|---|---|
| derived, 6–7 | **4.540016e-3** | **1.0000** | 200 | 9 |
| 2 | 4.540964e-3 | 1.0000 | 176 | 14 |
| 1 | 4.546217e-3 | 0.9444 | 168 | 13 |
| 0 | 4.552096e-3 | 0.9048 | 168 | 12 |

Strictly ordered in both columns, which is **neither** upstream finding:
TreeAMR measured a margin narrower than the motion coming out slightly
*worse* than no margin, TreeWave measured it coming out no worse, and here
every cell of margin buys something. See [The refinement
criterion](#the-refinement-criterion) for why. Note also what a narrow
margin costs in the other currency: the mesh changes 14 times instead of 9,
because the refined region has to be rebuilt as the feature walks out of
it.

**`p = 1` against `p = 3` and `p = 5`** (`D = 1`, tracked): the table is
under [Operator order](#operator-order), where it is the first
*discontinuous* row of the open question — 4.701364e-3 at tracking 0.9091
and 216 cells for `p = 1`, against 4.540016e-3 at 1.0000 and 200 cells for
`p = 3`, with `p = 5` indistinguishable from `p = 3`, and zero floor hits
throughout.

**The CFL recheck.** `speed_headroom = 1` on Sod throws in the **first**
chunk: `λ_end = 1.9486` against a step sized for `1.25`, a CFL number of
**0.6236** against the requested 0.4. At the case's own headroom of 2 the
run completes, `λ` rises from **1.183216** at `t = 0` to **2.204737**, and
`λ_end ≤ 2λ` at every chunk.

**The entropy wave through the driver**, on the periodic, hook-free path
with the cap at zero, against `entropywave_errors` at the same `N`:

| | driver L1 | study L1 | ratio | steps |
|---|---|---|---|---|
| `D = 1`, `N = 8` | 5.219843e-4 | 5.538618e-4 | 0.9424 | 50 / 47 |
| `D = 1`, `N = 16` | 1.343281e-4 | 1.350740e-4 | **0.9945** | 95 / 93 |
| `D = 2`, `N = 8` | 1.249213e-3 | 1.325157e-3 | 0.9427 | 95 / 93 |

Not bit for bit, and for two reasons that both close like `h²`: the driver
chunks its steps, so it takes a slightly different number of them; and a
case states its initial data as a pure `x -> P`, which is a **point
sample**, where the convergence study fills the exact cell average — a
relative difference of `(kh)²/24` in the amplitude. All `D + 2` integrals
hold to roundoff, which here is the plain claim, the box being periodic.

## Possible extensions

Not planned, listed because they are the obvious next questions:

- The Sedov radial profile from the Kamm–Timmes quadrature, making the
  radial-scatter figure quantitative.
- A limited, positivity-preserving prolongation — upstream, if the ghost
  floor count asks for it.
- Positivity-preserving flux limiting (Zhang–Shu, Hu–Adams–Shu), the
  conservative complement to the atmosphere reset; see
  [Floors and the atmosphere](#floors-and-the-atmosphere) for why it is
  a complement and not a replacement, and for its interaction with the
  coarse-fine fixup.
- A hand-written SSPRK33 with kernel stage updates, shared with TreeWave.
- Reflecting boundaries, once the region-form hook has a device form
  upstream; the 2D Sedov in a quadrant is the case that would want them.
- The isentropic vortex, a second exact smooth solution in 2D that is
  stationary and would isolate the limiter's clipping from advection.
- The next package: constrained-transport MHD, which is where the
  several-set state vector and the divergence-preserving prolongation
  become necessary and where every method chosen here is meant to carry
  over.

## Open questions

Decided in review: the ghost exchange in conserved variables with
`con2prim` over the stored extent (may change, on the ghost floor count);
primitive reconstruction; HLLE as the baseline flux with HLLC as the
Kelvin–Helmholtz comparison; the atmosphere reset of `U` in the
integrator's stage hook, with the floor on `P` as the second line;
Dirichlet boundaries wherever the physics does not require periodicity;
one `evolve!` driver with cases as data; the two
[upstream prerequisites](#upstream-prerequisites); default prolongation
order `p = 3`, with `p = 1` measured beside it; the McNally smooth-ramp
Kelvin–Helmholtz setup; Sedov acceptance on the exponent, the jump and a
uniform reference, the full profile an extension; and the `[sources]`
pin to TreeAMR's `main` (`m8` having been merged).

Still proposed:

1. **Whether the entropy wave is a case** or only a test fixture; it is
   written above as a case so that the interface-order table has a
   driver, and it does not appear in `bin/`.
2. **The reset cadence**, `:stage` against `:step`, is a measurement and
   not a question; it is listed so that it is not mistaken for a
   decision already made.

Smaller defaults marked (proposed) in the text — the variable order,
point samples rather than cell averages for discontinuous initial data,
`λ_max` measured once per chunk — are implementation choices that the
first milestones will either confirm or amend in place. All three are now
settled: the variable order in step 1, the point samples in step 4 (and
generalized in step 7, where a case states its data pointwise because the
adaptation cycle changes `h` under it), and `λ_max` once per chunk in
steps 4 and 7 — confirmed, but only beside a `speed_headroom` factor and
a recheck that throws.
