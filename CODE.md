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

*Status: milestone H0 (scaffolding) done, and H1 begun — the equation of
state, the two state conversions and the floors are written (step 1), and
so are the reconstruction and the three Riemann fluxes (step 2); the
right-hand side that calls them is not.* Nothing below is measured.
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
| `P`, primitive | cell | `D+2` | `2` | no | `P => nothing` (resized; recomputed by the next RHS) |
| `F_d`, fluxes, `d = 1 … D` | `facecentered(D, d)` | `D+2` | `0` | no | `F_d => nothing` |

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

**`P` has the same layout as `U` and holds primitives in its ghost cells
too** (decided). The reconstruction reads primitives at `i−2 … i+1`, so
a block needs primitives two cells into its neighbours. There are two
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

## Conservation at coarse-fine faces

The mechanism is TreeAMR's and needs nothing from this package but the
obligation: with `G = 2` on `U`, same-level faces agree bit for bit, and
step (4) replaces every coarse flux on a coarse-fine face by the
area-weighted average of the fine ones. What this package adds is that
there are now **`D + 2` conserved quantities**, each with its own domain
integral, and the claim is made for all of them:

- **(predicted)** Total mass, each momentum component and total energy
  are conserved to a few ulp of their own scale `Σ hᴰ |U_v|` over a run
  with a shock crossing a refined region that follows it, regrids in
  between, in `D = 1, 2, 3`; the drift does not grow with the step count.
  The negative control — `fixup = false`, the single difference — leaks
  by orders of magnitude more, and a uniform mesh conserves either way.
  This is TreeAMR's M8b table, repeated for a system.
- The **momentum** is the new case: for a momentum component whose total
  is zero by symmetry (the Kelvin–Helmholtz `S_y`, the Sedov `S_d`), the
  drift is measured against the maximum of that component's `Σ hᴰ |S_d|`
  over the run, not against its total, which may be zero.
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

- **(predicted)** The rule holds for the system as it did for the scalar:
  on the smooth entropy wave (below) over the static two-level mesh,
  L∞ rates 1, 2, 2 for `p = 1, 3, 5` and L1 rates 2, 2, 2, with the
  `p = 3` refined run landing on the unrefined control's rate. Predicted
  before it is run, as the Burgers rates were.
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
on a face, and the Sedov tests include one on an edge and a corner.

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
**Point samples at cell centers**, not cell averages (proposed): every
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

### Sod shock tube

The standard states, `(ρ, v, p) = (1, 0, 1)` for `x < ½` and
`(⅛, 0, ⅒)` for `x > ½`, `γ = 7/5`, on `[0, 1]`, to `t = 0.2`; the
fastest wave reaches a boundary at `t ≈ 0.29`, so the Dirichlet boundary
is exact for the whole run with margin, and the driver's assertion says
so.

**Reference:** the exact Riemann solution (Toro's pressure iteration and
sampling), host `Float64` code in the role TreeWave's Hankel table
plays, evaluated at cell centers and compared in the volume-weighted L1
norm. **(predicted)** L1 convergence at a rate between 0.8 and 1 on a
uniform mesh, as a limited second-order scheme gives on a solution with a
contact and a shock — this is a check that the scheme is right, not a
claim of order.

**What it measures beyond that:**

- **Conservation through regrids** in `D = 1` and `2`: the refined region
  follows the shock (and the contact, and the rarefaction — the Löhner
  indicator on `ρ` fires on all three), the mesh is rebuilt every chunk,
  and all `D + 2` integrals hold to roundoff with the fixup and leak
  without it. The M8b acceptance test, for a system.
- **A tracked shock matches the uniformly fine reference at fewer
  cells**, reduced onto a common grid as `track_shock` does, with the
  uniform coarse mesh as the control that says refinement bought
  something.
- **Direction independence, bit for bit** (predicted). On a uniform mesh
  the tube along `y` is the tube along `x` transposed, and the `D = 2`
  planar tube's profile equals the `D = 1` run's: the transverse flux
  differences are *exactly* zero (both faces see identical states), and
  adding an exact zero is exact. A cheap, sharp claim about the flux
  kernel having no preferred direction, worth asserting because the
  `ntuple` loops over `d` are the place a stray asymmetry would hide.
- **The boundary hook**, as above.

`D = 3` is a smoke test at small size; the planar tube in 3D exercises
nothing the 2D one does not, except cost.

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
every formula against the paper before any number is recorded.* Its
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
| `src/evolution.jl` | the kernels (`con2prim!`, flux, divergence), `HydroProblem`, `hydro_rhs!`, `max_signal_speed`, `hydro_dt` |
| `src/refinement.jl` | the Löhner indicator on primitives, `hydro_flags`, `refinement_buffer` |
| `src/driver.jl` | `HydroCase`, `evolve!` — the one loop — and its diagnostics |
| `src/exact_riemann.jl` | Toro's exact Riemann solver, host `Float64`, the shock-tube reference |
| `src/sedov_reference.jl` | the similarity law; later the Kamm–Timmes profile |
| `src/entropywave.jl`, `src/sod.jl`, `src/sedov.jl`, `src/kelvinhelmholtz.jl` | the four cases: initial data, parameters, references, per-case diagnostics |
| `src/benchmark.jl` | per-phase timings, TreeWave's format |
| `test/` | one `*_tests.jl` per case, plus `conservation_tests.jl`, `type_tests.jl`, `threading_tests.jl`, `device_tests.jl`, and the standalone `thread_workload.jl` |
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
- **H1 — The scheme on a uniform mesh.** EOS, `con2prim`, MUSCL with the
  three limiters, LLF and HLLE, SSPRK33, the six-step RHS with `D` flux
  sets — on a single-level periodic forest, `D = 1, 2` (3D smoke).
  *Accept:* the entropy wave converges at second order in L1 and L∞ with
  `:none` and near it with `:mc`; Sod against the exact Riemann solution
  at an L1 rate in `[0.8, 1.0]`; direction independence bit for bit;
  every conserved integral constant to roundoff (every face is a
  same-level face, with or without the fixup — the control).
- **H2 — Coarse-fine faces, static mesh.** The two-level `hydro_forest`,
  the fixup, the boundary hook. *Accept:* conservation of all `D + 2`
  integrals to roundoff with the fixup and a leak without, in
  `D = 1, 2, 3`, on the entropy wave and on Sod; the interface-order
  table for the system (predicted L∞ 1, 2, 2 at `p = 1, 3, 5`; L1 2, 2,
  2); a two-level Sod whose refined region touches the Dirichlet
  boundary.
- **H3 — Regridding.** The criterion, the buffer, `evolve!`, the
  initial-data cycle. *Accept:* the cycle converges to a fixed hierarchy
  on all four initial data; the tracked Sod tube in `D = 1, 2` matches
  the uniformly fine reference at fewer cells with the uniform coarse
  mesh as control, conserves to roundoff through the regrids, and leaks
  without the fixup; the `p = 1` against `p = 3` table for Sod; the
  buffer-width table.
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

None yet. This section takes the numbers as the milestones produce them,
each beside the prediction it confirms or corrects.

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
first milestones will either confirm or amend in place.
