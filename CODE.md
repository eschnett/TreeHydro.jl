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

**(The ghost floor count arrived in step 8.)** `ghost_floor_hits(p)` is the
one-item-per-block kernel this section promised: it sums slot `D + 4` over
each block's stored cells, skips the owned range, and combines the per-block
values on the host in block order. It counts ghost *entries* and not cells,
so a physical cell that is a ghost of two blocks counts twice — what is
being measured is how often the recovery meets an unphysical ghost. It reads
**zero** on the tracked shock tube and on the entropy wave (measured in step
8), which is what says the open question above is about the Sedov blast in
particular and not about the exchange in general.

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

**(Measured in step 10: HLLC becomes the Kelvin–Helmholtz case's default
and the package-wide default stays HLLE.)** The comparison was made on the
shear layer, which is where it belongs, and it is not close. `M(t = 1.5)`
on uniform meshes:

| | 32² | 64² | 128² |
|---|---|---|---|
| HLLC | 0.011608 | 0.075489 | 0.123981 |
| HLLE | 0.000662 | 0.006599 | 0.035717 |
| ratio | 17.53 | 11.44 | 3.471 |

**HLLC at half the linear resolution is further along than HLLE at full
resolution** — 0.075489 at 64² against 0.035717 at 128², and 0.011608 at
32² against 0.006599 at 64². On this flow HLLE therefore costs at least a
factor of two in linear resolution, which is four in cells and eight in
work; and it costs it in the *growth rate* and not merely in the amplitude
(2.5803 against 1.1712 over the same window). The ratio narrows with
resolution, as two consistent fluxes must, which is what says this is
diffusion and not a defect. The shear layer *is* a contact, HLLE's
two-wave average is what smears it, and that is the whole of the
difference between the two solvers — the stationary-contact unit test says
the same thing in closed form on one pair of states, and this says it on a
flow.

What is **not** measured by any of this is the shock cases. HLLE remains
the package-wide default of `HydroProblem` and `evolve!` because it is
*the* GRMHD flux — it needs only the fastest and slowest signal speeds,
which in GRMHD are the fast magnetosonic ones — and because the baseline a
comparison is made against should be the one every result before this
milestone was taken with. `kh_run`'s `riemann` keyword defaults to `:hllc`
and nothing else in the package does.

The criterion this decision was made by was written down **before** the
HLLE runs were made, and two of its three clauses failed to decide;
recording that is the point of writing it first. It asked (1) that the two
uniform fine references agree in `M(t_end)` to within 5%, and they differ
by 247%; (2) that HLLC's *tracked* run be closer to its own fine reference
than HLLE's is, and both are within a percent — 0.42% against 0.29%, HLLE
nominally closer — because a tracked run measured against a fine run *with
the same flux* measures the **mesh** and not the flux, and the mesh is
equally good under either. Clause (3), the difference between the two fine
references, was overwhelming, and the resolution sweep above is the
amendment that turns it into a decision.

### Time integration and the time step

`SSPRK33` from `OrdinaryDiffEqSSPRK`, fixed step, the way TreeAMR's
Burgers test uses it: conservation holds for any Runge–Kutta method
(every stage's `du` sums to zero), but only a strong-stability-preserving
one keeps a limited scheme's shocks monotone. Its `stage_limiter!` hook
is also where the atmosphere reset acts — a `solve` keyword since step 8,
the constructor form having been deprecated upstream, with the hook, its
signature and its cadence unchanged (see
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
**(Amended in step 9: "decreasing after the deposition" is right and the
first chunk is not covered by it.** Sedov's jump at `r₀` is a Riemann
problem like Sod's, and its star region is not present at `t = 0`, so
`λ_max` grows within the first chunk by **1.16537, 1.25164 and 1.10396** in
`D = 1, 2, 3` — smaller than Sod's 1.8522 because the blast's initial sound
speed is already large, and still a growth. `HydroCase(::SedovBlast)` uses
`speed_headroom = 2`, and those are also the largest growths over each
run.)

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

   **(Amended in step 8: the hooks are `solve` keywords now, not
   constructor arguments.)** The signature and the semantics are exactly
   as above, but `OrdinaryDiffEqCore` has moved the two limiters into the
   solver options: `SSPRK33(; stage_limiter! = f)` is deprecated in
   favour of `solve(prob, SSPRK33(); stage_limiter = f)`, warns on every
   run under `--depwarn=yes` — which is what `Pkg.test` passes — and
   would, once the deprecation is completed, leave the constructor's
   field silently unread. A positivity correction that is installed
   nowhere and reports nothing is the one failure mode this must not
   have, so `hydro_solve!` passes the keywords to `solve` and
   `test/reset_tests.jl` asserts that a step really does come out floored
   under `:stage` and under `:step` and does not under `:none`. Read
   against the installed `OrdinaryDiffEqSSPRK`: `stage_limiter!` is
   applied to every stage of `SSPRK33` *including the last*, and
   `step_limiter!` once to the accepted step's result, which is what
   makes `:step` the cheaper cadence of the same thing rather than a
   different correction.
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

  **(Measured in step 8; the prediction is corrected in both
  directions.)** The *state* is idempotent **bit for bit** — at `Float64`
  and at `Float32`, in `D = 1, 2, 3`, on a synthetic state carrying all
  six populations (healthy gas, `0 < ρ < ρ_atm`, `ρ = 0`, `ρ < 0`,
  `ρ = NaN`, and a healthy density with a negative internal energy). The
  *flag* is not, and the reason is the round trip the prediction named: a
  pressure-floored cell recovers its internal energy through the
  cancellation `E − ½S²/ρ`, so where the kinetic energy dominates, the
  recovered pressure lands a fraction of an ulp below `p_floor` and the
  rule fires again — writing `prim2con(eos, (ρ, v, p_floor))` from the
  same `ρ` and the same `v`, which is the same arithmetic on the same
  numbers and therefore the same bits. Second-pass counts on that state:
  2 of 13 floored cells at `Float64` in `D = 1`, none in `D = 2`, 85 of
  426 in `D = 3`, none at `Float32` in any dimension. So the reset is a
  fixed point of the state and not of its own report, which is why the
  counts are taken per call rather than read back off the flag slot.
  `con2prim` of the reset `U` reproduces the floored `P` to roundoff of
  the *data's own scale* — not of `p_floor`, for the same cancellation
  reason — and every cell in which no floor fired is bit-identical to
  what it was.
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

  **(Measured in step 8, and "exactly" is the word.)** On the tracked
  shock tube under `:stage` and under `:step`, and on the entropy wave
  through the driver, the injection comes back `(0.0, 0.0, 0.0)`, the
  reset hit count is `0`, the ghost floor count is `0` — and the final
  state, the drift, the step count and the mesh history are
  **bit-identical** to the `reset = :none` run's. That is what the
  "written back only where `hit` is true" rule buys, and it is why every
  roundoff drift bound recorded in this file is unchanged now that
  `:stage` is the default and every run in the suite calls the reset three
  times per step. The accounting is `accounting = false` by default and
  then returns `nothing` rather than a tuple of zeros, so that "not
  measured" cannot be read as "measured and zero". One honest exception it
  reports rather than hides: a state carrying a `NaN` has no total, so the
  injection into it is a `NaN` in the variable the `NaN` lived in and a
  finite number in the others — the state is repaired all the same.
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

**(Implemented in step 8.)** `src/floors.jl` holds `ResetAccounting` and
`reset_atmosphere!(u, integrator, p, t)` — a `map_blocks!` launch over the
owned cells of `statearray(u, U)`, writing back only where `hit` came back
true and writing the flag into `P`'s diagnostic slot `D + 4` as it goes;
`src/evolution.jl` holds `ghost_floor_hits`, `check_reset`, the state-vector
form of `conserved_totals`, and `hydro_solve!`'s `reset` keyword;
`src/driver.jl` wires `reset` and `accounting` through `evolve!` and calls
the reset a second time after every `regrid!`. The numbers are in
[Measured results](#step-8--the-atmosphere-reset). What the implementation
settled:

- **The hit count is always taken and the injection is not.** The count is
  one `block_mapreduce` over one diagnostic slot, which `floor_hits`
  already was; the injection is two full reductions of the state per call.
  So the count rides along with every reset and the injection is behind
  `accounting`.
- **`P`'s flag slot is shared between the recovery and the reset, and the
  sharing is safe because `P` is scratch.** After `hydro_rhs!` or
  `update_primitives!` the slot holds the `con2prim` pass's flags over
  every stored cell, which is what `floor_hits` and `ghost_floor_hits` read
  at a chunk boundary; immediately after a reset it holds the reset's own,
  over owned cells only, which is what the reset reads. The two never mix
  within one measurement, and no second field set is needed — which is the
  alternative this rejected, since a `FieldSet` of flags handed back in
  after every regrid would be mesh bookkeeping written downstream.
- **The ghost count is the one reduction in the package written as a launch
  of its own**: one work item per block, summing slot `D + 4` over that
  block's stored cells and skipping the owned range, with the per-block
  values combined on the host in block order. `block_mapreduce` reduces a
  block's *interior*, which is exactly the range this count must not use.
- **The record is host-side, mutable, and held by the problem.** A run
  rebuilds its `HydroProblem` after every regrid, so a record the problem
  made for itself would start again at every mesh change; `evolve!` makes
  one per run and hands the same one to every problem it builds. No kernel
  ever receives it.
- **`reset = :none` skips the post-regrid reset too**, so that it is the
  negative control it is meant to be rather than "the reset, minus the
  hooks".

**(Measured in step 9, on the case where the floors finally fire — and the
design was wrong about which rule fires and about where.)** The numbers are
in [Measured results](#step-9--the-sedov-blast).

- **The atmosphere rule never fires on Sedov, at any size the tests can
  afford.** The prediction above put the firing in the evacuated interior,
  where the similarity solution's density falls six orders of magnitude:
  `G(λ) ∼ λ^{D/(γ−1)}` passes `10⁻⁶` at `λ ≈ 0.06` in `D = 2`. The
  *discrete* bubble never gets there — numerical diffusion refills it, and
  the measured minimum density is **6.7e-2** on the tracked `D = 2` run at
  `h = 1/128` — so `ρ < ρ_atm` is not reached, no velocity is ever zeroed,
  and every mass injection in this package is still **exactly zero**. The
  remedy is not a shallower atmosphere: `ρ_atm` is set six orders below the
  data by the refinement criterion's global floor term, and raising it to
  make the rule fire would be tuning a measurement into existence.
- **What fires is the pressure floor, and what makes it fire is the
  coarse-fine face.** The interface flux restriction replaces a coarse
  cell's flux with the average of its fine neighbours', and in gas whose
  internal energy is `p_amb/(γ−1) = 2.5e-5` that correction can take it
  below `p_floor`. On the static two-level mesh: **4096** owned cells and
  **40** ghost entries in `D = 2`, **24504** and **4703** in `D = 3`. On
  every *uniform* mesh, and on the tracked mesh — whose coarse-fine faces
  stand in undisturbed gas — nothing fires at all. This is the interaction
  the fourth option above named in the abstract ("an average of limited
  fine fluxes need not keep the coarse cell positive"), met in the concrete
  and from the other direction.
- **The injection is energy only, and it is exactly energy only.** The
  pressure floor keeps `ρ` and `v`, so `prim2con` writes `ρ` back bit for
  bit and the mass injection is exactly zero; the momentum injection is
  *roundoff* rather than zero, because the same round trip recomputes
  `S = ρ (S/ρ)`, which is not the bits it started from — measured
  `-1.65e-24` in `D = 3` against a bound of `2.2e-14`.
- **The accumulated injection is an upper bound under `:stage` and an
  equality under `:step`** (amended). The accounting adds the raw
  `Σ hᴰ ΔU` of every call, but `SSPRK33`'s three stage vectors enter the
  step's result with weights `1/6`, `2/3` and `1`, so an injection into a
  stage reaches the state scaled by that stage's weight. Measured ratio of
  drift to accumulated injection: **0.51984** in `D = 2` and **0.52203** in
  `D = 3`. Under `:step`, which resets once on the step's own result, the
  drift **equals** the injection — to `4.4e-16` and `2.2e-16` against
  roundoff bounds of `8.0e-13` and `2.5e-13`. Step 8 could not see this,
  because on Sod and the entropy wave the injection was exactly zero and
  every weighting of zero is zero. The fix is not to weight the
  accounting — the hook is not told which stage it is in, and a
  `DiscreteCallback` would not be either — but to say what the number is:
  a bound with `:stage`, an equality with `:step`.
- **So what does `:stage` buy over `:step`?** On this case: three times the
  repairs (4096 against 1384 owned cells in `D = 2`, 24504 against 8232 in
  `D = 3`) for the same final state to roundoff, and a strictly less
  informative injection. What it buys is what it was adopted for and this
  case cannot show — a stage vector that is never seen in an unphysical
  state by the *next* stage's right-hand side — and the case where that
  matters is a star in a vacuum, not a blast in an ambient. `:stage` stays
  the default, GRMHD practice being the reason, and the measurement
  recorded here is that on a blast the cheaper cadence is indistinguishable
  in the answer and better in the bookkeeping.
- **The ghost floor count answers the upstream question, and the answer is
  "not on the exchange in general".** `ghost_floor_hits` is zero on the
  entropy wave, on Sod, on every uniform mesh and on the tracked Sedov
  mesh; it is nonzero exactly where a `p = 3` prolongation spans a strong
  shock, and there a `p = 1` prolongation takes it to zero as well (see
  [Operator order](#operator-order)). So a limited, positivity-preserving
  prolongation is not needed to make the ghost exchange work; it is what
  one would ask for to keep `p = 3` accuracy *and* the `p = 1` floor count,
  and the trade is now measured rather than assumed.

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

  **The question is closed in step 9, and the answer is a trade with a
  price on it.** It took two measurements, and the first was that the
  *tracked* blast cannot answer it either: `tracking == 1` puts the refined
  region's boundary ahead of the shock by construction, so a tracked run's
  prolongation only ever acts on undisturbed ambient gas, and `p = 1` there
  is indistinguishable from `p = 3` in every column — same exponent, same
  peak, same cells, same mesh history, no floor hit either way. The
  question needs a prolongation acting **across a strong shock**, which is
  the static two-level mesh `sedov_forest(refined = :center)`, where the
  blast leaves the refined region. There, in `D = 2` over 433 steps:

  | `p` | L1 against the uniform fine run | reset hits | ghost hits | energy drift |
  |---|---|---|---|---|
  | 1 | 4.427249e-2 | **0** | **0** | 2.0e-15 |
  | 3 | **4.406604e-2** | 4096 | 40 | 1.24659e-5 |

  So `p = 1` buys **exact positivity** — nothing floored, nothing injected,
  conservation at roundoff — for **0.47%** of L1. That is the argument
  `p = 1` was proposed for, measured, and it is the opposite sign from
  Sod's row, where `p = 1` was 3.6% worse and bought nothing. The two rows
  do not disagree: they measure different things, and which one a case is
  in depends on whether its prolongation ever spans a shock.

  **`p = 3` stays the default (decided, unchanged).** The interface-order
  rule is what it is there for, it is better in L1 on both discontinuous
  cases that can tell the difference, and it is strictly better on the
  smooth ones. What the table changes is the *upstream request*: a limited,
  positivity-preserving prolongation would buy `p = 3`'s 0.47% and `p = 1`'s
  zero floor count together, and that is now a request with a number on it
  rather than a guess. It is still not made, because the floor count it
  would remove is a count of repairs the reset already makes correctly and
  accounts for exactly.

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

**(Implemented in step 9: the M2 ordering case in full.)** The blast is
Dirichlet on every face, so `sedov_forest(refined = :corner)` puts a *fine*
block where every physical face meets — two in `D = 2`, three in `D = 3` —
and `refined = :edge` puts a line of them along a 3D edge, where two do.
The blast sits at the centre and these meshes are run for a few tens of
steps, so the corner is undisturbed throughout and the claim is about the
exchange and nothing else. What it showed:

- **The hook fills the corner and edge regions, and fills all of them.**
  Zero mismatched entries out of **1664**, **72000** and **59360** stored
  outward-facing entries on the 2D corner, the 3D edge and the 3D corner —
  counting an entry as outward-facing if it lies in the outer ghost range
  of *any* dimension whose block sits against a physical face, so the
  regions reachable by neither face alone are included.
- **The prolongation sweep that runs after the hook reads what the hook
  wrote.** The fine corner block's interior is untouched: its density and
  its energy are **bit-identical** to the ambient, and its momenta — which
  start at exactly zero, so any nonzero value is a change — move by at most
  `1.5e-54`, which is `1e-33` of one ulp of the ambient energy density. A
  tangential prolongation reading unfilled ghosts would put the *blast's*
  numbers there, not a `1e-54`.
- **The momenta are exactly zero in `D = 2` and are not in `D = 3`**, and
  the difference is the pressure flux failing to cancel in the last bit
  through one more tensor factor. It is recorded because "bit-identical"
  was the natural claim to write and is true only of `ρ` and `E`.
- **A coarse-fine face beside two physical ones conserves.** Every drift is
  at or below its roundoff bound on all three meshes.

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
radial-scatter figure (below) quantitative rather than qualitative
**(drawn in step 11: `bin/visualize2d.jl --case=sedov` sweeps the
profile parametrically in `u` and draws it against the *measured* `E₀`,
stopping at the shock, the law being the strong-shock limit and
describing nothing outside it)**.

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

**(Implemented in step 9.)** `src/sedov_reference.jl` holds the similarity
law and `src/sedov.jl` the case; the numbers are in
[Measured results](#step-9--the-sedov-blast). What the case decided, and
where the paragraphs above needed amending:

- **The parameters.** `p_amb = 1/10⁵`, `r₀ = 1/16` in `D = 1, 2` and `1/8`
  in `D = 3`, `ρ_atm = p_atm = 1/10⁶` and `p_floor = 1/10⁸`. The
  atmosphere sits **six orders below `ρ₀`**, which is what the refinement
  criterion's global floor term needs (`CODE.md` measures it silencing
  `O(1)` noise six orders down at `τ = 0.0020` and five orders down at a
  marginal 0.0196); `p_atm` sits *below* `p_amb` so that a reset in
  undisturbed gas could never raise its energy, and `p_floor` two orders
  below `p_amb` so that the pressure floor can only fire on a state the
  initial data does not contain.
- **`r₀` spans eight cells at the cap and not three or four**, in `D = 1`
  and `D = 2`. The hot spot's sound speed goes as `r₀^{-D/2}`, so halving
  `r₀` doubles `λ` and halves the chunk the travelling margin admits — 100
  chunks instead of 40 — and the wider top hat is also what lets the centre
  flatten within an affordable `t_end`. `D = 3` uses four, where the cap is
  one level.
- **`speed_headroom = 2`, and the sentence about `λ_max` above is wrong.**
  "The early phase … is *also* where `λ_max` only decreases" is right about
  the blast and wrong about the first chunk, for the same reason it was
  wrong on Sod: the jump at `r₀` is a Riemann problem and the gas on the
  hot side of its contact moves, so `|v| + c_s` there exceeds the hot
  spot's own `c_s` before the discontinuity has resolved. Measured
  first-chunk growth **1.16537, 1.25164, 1.10396** in `D = 1, 2, 3`, and
  those are also the largest growths over each run. See
  [Time integration and the time step](#time-integration-and-the-time-step).
- **The measured `E₀` is what the law takes, and the ratios are
  1.00000, 1.03451, 1.04445.** In `D = 1` the cells that receive the
  deposition tile `V_1(r₀) = 2r₀` exactly, so the only difference from the
  nominal value is the ambient share of the top hat itself, `3.125e-6`.
- **`shock_radius` reads a density threshold of `3/2 · ρ₀` and is a host
  loop**, not the `firing_boxes` sweep this design first reached for. A
  per-block bounding box loses the correlation between dimensions: the
  shell crosses a block diagonally, so the box's outermost corner sits
  about `w²/(2 r_s)` beyond the outermost firing cell — 22% at the start of
  the `D = 2` fit range and 8% at its end — and a bias that *shrinks as the
  shock grows* lands directly on the slope being measured, costing about
  0.13 of the exponent. The loop is an oracle in the spirit of
  `reduce_to_grid` and runs once per chunk against hundreds of steps.
- **The planar `α` is twice the literature's, and that is a convention.**
  `α = σ_D ∫ …` with `σ_1 = 2`, matching the deposition volume `V_1 = 2r₀`
  this case uses, so `E₀` is the energy on *both* sides of a planar blast;
  Kamm & Timmes count one side. The cylindrical and spherical values agree
  with the recalled ones in every digit.
- **The refined region is a growing *disk*, not a shell** (amended). The
  prediction above — "a refined shell that grows while the interior
  coarsens", and `PLAN.md`'s "block count rising then falling behind the
  shock" — assumes the evacuated bubble is flat. It is not: the similarity
  solution's `G(λ) ∼ λ^{D/(γ−1)}` is a **steep density ramp**, and a
  Löhner indicator on `ρ` fires throughout it, correctly, because the ramp
  is under-resolved. Measured block history in `D = 2`: 40 → 88 → 112 → …
  → 196 over 40 chunks, **monotone**. Only at the very centre, where the
  ramp has flattened, does the criterion fall silent — at `t_end` the
  blocks within `r < 0.06` report `Coarsen` while those from 0.06 to 0.42
  fire — so the hollow opens from the inside out and opens late. The mesh
  still follows the blast (`tracking == 1`) and still saves cells (12544
  against 16384), which is what the milestone was for; what is corrected is
  the shape. A criterion that produced a shell would have to know that a
  monotone ramp behind a shock is not a feature, which is not something a
  second-difference indicator can be told.
- **The tracked mesh cannot measure the coarse-fine face at all, and that
  is a consequence of tracking rather than a defect.** `tracking == 1`
  means every strongly firing cell sits on a block at the cap, and the
  travelling margin then puts the refined region's boundary *ahead* of the
  shock by construction — so every coarse-fine face of a tracked run stands
  in gas the blast has not reached. Measured: `fixup = false`, `p = 1` and
  `reset = :step` give the same exponent, peak, cell count, mesh history
  and tracking as the run they are controls for, and the two reset cadences
  agree bit for bit. Every claim about the interface flux restriction, the
  prolongation order and the floor counts is therefore made on a **static**
  two-level mesh, `sedov_forest(refined = :center)`, where the blast starts
  inside the refined region and leaves it at `|x_d| = L/4`. That is the
  blast's `sod_forest(:middle)`.
- **`sedov_static` replaces the planned `uniform_sedov`.** What the case
  needed was a static-mesh driver, which `uniform_run` — the driver with
  the cap at zero — cannot be; the uniform control is `sedov_static` with
  `refined = false`.

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

run to `t = 1.5`. **(Checked against the paper in step 10.)** Every line
above — the four branches and their signs, `ρ_m`, `v_m`, the parameter
values, the perturbation, `γ`, `p` and `t_end` — is equations (1)–(5) of
arXiv:1111.1764 term for term; `test/kelvinhelmholtz_tests.jl` re-evaluates
them independently from the paper's own literals and the two agree **bit
for bit** over a grid of sample points. One thing below was wrong and is
corrected in place; it is flagged where it was. Step 6
uses the density ramp's **shape** — the four branches above, in one
dimension, at uniform pressure — as the smooth calibration profile for
the refinement criterion, and nothing else of the setup. Nothing there
depended on the transcription being faithful, and the one error was not in
the profile in any case, so the calibration table stands unamended.

Its diagnostics are the paper's: the **amplitude of the seeded mode**
`M(t)`, from the projections of `v_y` onto `sin(4πx)` and `cos(4πx)`
weighted by `e^{−4π|y − ¼|}` for `y < ½` and by `e^{−4π|(1−y) − ¼|}` for
`y ≥ ½`, so that **both interfaces are read, mirrored onto one another**
(equations (6)–(8); the first draft of this paragraph said "so that the
lower interface alone is read", which is wrong — **corrected in step 10**.
The two interfaces carry the same mode with opposite sign of `∂_y v_x`, and
the mirrored weight is what lets them add rather than cancel), and the
**maximum `y`-kinetic energy** `max ½ ρ v_y²` over the domain against time.

On a mesh whose cells are not all the same size the sums are
**area-weighted**, the paper's equations (14)–(17) rather than its
uniform-grid (6)–(9):

    s_i = V_y w_i sin(4πx_i) e_i     c_i = V_y w_i cos(4πx_i) e_i
    d_i = w_i e_i                    M   = 2 √((Σs_i/Σd_i)² + (Σc_i/Σd_i)²)

with `w_i` the cell's area. That is the form this package uses (decided in
step 10), because an adaptive mesh has cells of two sizes by construction:
the uniform sums would weight a coarse cell as though it were a fine one
and `M` would jump at every regrid. The two forms agree exactly on a
uniform mesh, where `w_i` cancels between numerator and denominator.

`M(t)` grows exponentially through the linear phase and eventually
saturates. Two upper bounds are quoted and the measured rate must stay
below both, neither being this problem's own: the paper's loose guide for
the **infinite-domain incompressible** flow is `M ∝ e^{4.384 t}` (Wang et
al. 2010, Eq. 18), with `max ½ρv_y² ∝ e^{2 × 4.384 t}` since that quantity
is quadratic in `v_y`; and the **sharp-interface** incompressible rate
`k Δv √(ρ₁ρ₂)/(ρ₁+ρ₂) = 4π √2/3 = 5.9238` for `k = 4π`. The ramp and the
compressibility both slow the real thing.
The quantitative reference is a **uniform fine run of this code**, as
for the shock cases; the paper's curves are the sanity check on their
shape, since the setup is theirs. Their Figure 4 is the Pencil Code
reference at 4096² and their Figure 7 puts every code's `M(t)` at 128²,
256² and 512² beside it: `M` starts at `0.01`, **dips and then plateaus
until `t ≈ 0.45`** before it takes off, and reaches a few tenths at
`t = 1.5` without a plateau at the end. That is the shape to match and not
a number.

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
  **(Measured in step 10: HLLC wins by a factor of two in linear
  resolution, and it becomes this case's default; the package-wide default
  stays HLLE. See "Riemann solver" and "Step 10" below.)**
- **The picture**: a filmstrip of `ρ` at four times, one heatmap per
  block with the block boundaries drawn and coloured by level, in the
  manner of TreeWave's `visualize2d.jl`; `M(t)` and the maximum
  `y`-kinetic energy against time with the uniform reference over them;
  block count against time. **(Implemented in step 11 as
  `bin/visualize2d.jl --case=kh`, exactly as described.** The diagnostics
  are the ones `kh_run` records, reached through the `observer`
  pass-through step 11 added to it, so the figure plots the run's own
  curves rather than a second computation of them. The four frames are
  `t = 0, 1/2, 1, 3/2` on one colour range, the fit window `2a ≤ M ≤ 6a`
  is shaded under `M(t)`, and the uniform fine run's 256 blocks are drawn
  across the block count as the mesh the 232 are being compared with.)
  **(Extended after step 11 with `--movie`, which was not part of what H5
  accepted and is recorded as an addition rather than folded in.** The
  design asked for four frames and four frames is what H5 was marked done
  on; the movie keeps all 301 the observer hands over instead. It exists
  because the strip cannot show *order* — that the mode decays before it
  grows, and that the refined region thickens with the rolls rather than
  travelling with them — which is the one thing this case is for. See
  "A movie" below.)

**What step 10 decided, beside the flux** — each measured rather than
assumed, and each recorded with its number under "Step 10" below:

- **`speed_headroom = 1`.** The flow is smooth and subsonic and carries no
  Riemann problem whose star region could outrun the initial data's
  `λ_max`, which is the mechanism that forces `2` on Sod and on Sedov.
  Measured worst growth over 300 chunks: **1.00031**. The entropy wave is
  the only other case at `1`.
- **`chunk = 1/200`, and it is the *margin* that sets it, not the CFL
  condition.** The derived buffer is `ceil(headroom·λ·chunk/h_cap) + 1`,
  and on this case the buffer is what decides how much of the box is
  refined: a 3-cell margin (`chunk = 1/200`) refines 128 of the 256
  possible finest blocks and a 7-cell one (`chunk = 1/64`) refines all 256,
  which is the uniform fine mesh under another name. **A feature that grows
  rather than travels gets nothing from a travelling margin** and pays the
  whole saving for it — which is new, the tube and the blast both having
  features that travel.
- **The growth window is a range of `M`, not of `t`**: from twice the
  seeded amplitude to six times it, `2a ≤ M ≤ 6a`. Stated that way it means
  the same thing at every resolution and under either flux, because the
  phase a run is in is a property of how far the mode has grown and not of
  the clock. `M` first *decays* while the ramp sheds the transient, so a
  window opened at `t = 0` would measure that instead.

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
        u  = solve(SSPRK33(), hydro_rhs!, u, t → stop; dt,
                   stage_limiter = reset_atmosphere!)   # the reset, per stage
        scatter!(U, u); fill_ghosts!(U, …; boundary); con2prim!(P, U)
        assert dt ≤ cfl · h / (D · max_signal_speed(p))      # the CFL check
        record totals of every conserved variable, the injection, floor
            counts by population — floor_hits and ghost_floor_hits from
            the recovery, reset_hits from the reset — block count
        observer(p, t, u)                            # before the regrid invalidates U
        flags = hydro_flags(P; refine_tol, coarsen_tol, cap)
        if regrid!(forest, (U => p.schedule, P => nothing, F_1 => nothing, …);
                   flags, buffer, boundary)
            p = HydroProblem(U, ops; …, prims = p.prims, fluxes = p.fluxes)
            u = statevector(U); gather!(u, U)
            reset_atmosphere!(u, nothing, p, t)       # the reset, per regrid
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
integral against its scale, the floor counts — `floor_hits` (owned cells
the recovery found unphysical at a chunk boundary), `reset_hits` (owned
cells the reset changed) and `ghost_hits` (ghost entries the recovery
floored), with `injection` beside them or `nothing` where it was not
measured — the mesh statistics per
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
| `src/sedov_reference.jl` | the similarity law `ξ₀`, its exponent, the energy integral's quadrature and the parametric profile, host `Float64` |
| `src/entropywave.jl`, `src/sod.jl`, `src/sedov.jl`, `src/kelvinhelmholtz.jl` | the four cases: initial data, parameters, references, per-case diagnostics |
| `src/benchmark.jl` | per-phase timings, TreeWave's format |
| `test/` | one `*_tests.jl` per case holding its unit, structural and physics claims together, plus `reset_tests.jl` for the atmosphere reset (which belongs to no case: its claims are about the floors, the integrator's hooks and the accounting), `type_tests.jl`, `threading_tests.jl`, `device_tests.jl` and the standalone `thread_workload.jl` |
| `.github/workflows/CI.yml` | the one workflow, two jobs: `test` runs the whole suite on every push, over the Julia × OS matrix, at one thread and at four; `viewer` instantiates `bin/` and renders every figure |
| `bin/visualize1d.jl` | the shock tube against the exact solution, per block, coloured by level, with `τ` and the conserved totals against time |
| `bin/visualize2d.jl` | the Kelvin–Helmholtz filmstrip and diagnostics; the Sedov filmstrip and radial scatter (`--case=`); either as a movie (`--movie`) |
| `bin/backend.jl`, `bin/Project.toml` | as in TreeWave; built in step 11 |
| `bin/benchmark.jl` | as in TreeWave, and it arrives with `src/benchmark.jl` in H6c — step 11 shipped the other two and not this one |

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
as a test that merely still passes. The tests take **3 m 44 at one thread
and 2 m 38 at four** locally as of step 10, on a quiet machine — the shear
layer added roughly a minute at one thread and thirty seconds at four to
step 9's 2 m 43 and 2 m 09, and the blast had
added 50 s and 30 s to step 8's 1 m 50 and 1 m 42 — and a CI
entry takes a few times that, a shared runner being slower and the rest of
it precompilation. `CI.yml`'s `timeout-minutes: 30` is the guard against a
runtime regression and has room, though less of it than before: the
one-thread entry is the one to watch, the Kelvin–Helmholtz file being the
most parallel work any one file holds and therefore the one that gains
least from a serial runner.

**Code coverage runs on the serial cells and must stay off the threaded
one** (amended in step 10b), and the reason for the exclusion is the whole
of the history below. `julia-actions/julia-runtest` turns coverage on by
default; what it costs on the threaded entry is a factor of a hundred.

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

Coverage is wanted — there is a badge in `README.md` and a
`CODECOV_TOKEN` in the repository's secrets — so the rule is not "off"
but **once, where it is read**. What it costs is measured on the whole
suite rather than on the sweep alone, at one thread, locally and back to
back: **4 m 01 without coverage and 12 m 38 with it, a factor of 3.14**,
both green at 11609 tests — the 5.7× sweep figure diluted by the parts of
the suite that are not kernel-heavy, and far from the hundred the
threaded entry would pay. Three conditions follow from that number, and
`CI.yml` spells out all three.

*One cell, not four, and the cheapest one.* Every serial cell executes
the same lines, so a second instrumented cell pays again to tell Codecov
what the first already said. Which cell, though, turned out to matter far
more than expected, and the reason is the Julia version rather than the
runner. Measured on this machine at one thread, back to back, the whole
suite:

| | Julia 1.11 | Julia 1.13 |
|---|---|---|
| no coverage | 8 m 55 | 4 m 01 |
| with coverage | 9 m 17 | 12 m 38 |
| factor | **1.04×** | **3.14×** |

So instrumentation is very nearly *free* on the floor version and costs a
factor of three on the current one — and under coverage 1.11 is faster in
absolute terms than 1.13, though it is 2.2× slower without it. On CI the
same three instrumented cells came back at 18 m 08 on Linux at 1.11,
23 m 22 on macOS at 1.13 and 42 m 02 on Linux at 1.13, which agrees in
ordering; the *factor* cannot be measured there at all, because 1.04×
sits far below the runners' own scatter.

On CI the choice has a second axis, the runner's architecture, and the
two together span a factor of 3.4. Instrumented, whole suite, one thread:

| | Julia 1.11 | Julia 1.13 |
|---|---|---|
| macOS arm64 | **12 m 12**, 17 m 03 | 23 m 22 |
| ubuntu x86 | 18 m 08, 19 m 52 | 42 m 02 |

macOS is 1.5–1.8× faster than Linux at either version and 1.11 is
1.9–2.3× faster than 1.13 on either architecture, so each effect is
reproduced across the other axis rather than resting on one draw — which
matters, given the ±70% scatter recorded below. Coverage therefore goes
on the **fastest cell of the four**: the `matrix` puts `coverage: true`
on the `version: "1.11"`, `macOS-latest` entry and the step reads
`matrix.coverage == true`. No file here contains a `VERSION` check or an
`@static`, so the lines reported are the same lines whichever cell
carries it, and Codecov cannot tell which one did.

*Only where it is read.* The badge reflects `main`, so instrumenting a
branch or a pull request buys nothing and spends the 3.14× on the cell
that decides how long anyone waits for a green tick. The condition is
`github.ref == 'refs/heads/main' || github.event_name ==
'workflow_dispatch'`; the dispatch arm is there so the upload path can be
exercised deliberately before it is relied on, because a reporting step
that runs nowhere reports nothing — the same failure mode as an SSPRK
stage limiter installed in a field nobody reads.

*Never the threaded cell*, which is the whole of the history above.

Measured end to end afterwards, against **14 m 20** for the same suite
before any of this (the last all-cells-green run under the old matrix)
and **24 m 39** for the straight mirror of TreeAMR's and TreeWave's
arrangement:

| | wall clock | critical path |
|---|---|---|
| ordinary push or pull request | ~12 m | the threaded cell |
| push to `main`, coverage collected | 18 m 02 | the instrumented cell |

So the everyday case is no worse than it was and one cell lighter, and
`main` pays about six minutes for its coverage. **The instrumented cell
is the critical path on a `main` push**, and has been under every
arrangement tried: 18 m 02 hosted on macOS at 1.11, 20 m 42 on Linux at
1.11, and 42 m 45 on Linux at the current release. A draft of this
paragraph predicted ~13 m and the threaded cell, from the 12 m 12 draw
alone; the next draw of that same cell was 17 m 03. Which cell is slowest
is exactly the kind of claim the variance note below says needs more than
one run behind it, and it has now been got wrong twice.

That correction is really a statement about variance, and it is the
caution to carry away from every CI number here, in the spirit of the one
about this machine being shared: GitHub's runners vary by more than most
of the effects being measured. The same instrumented cell, unchanged
configuration, came back at **18 m 08**, **11 m 24** and **19 m 52** on
three consecutive runs — a factor of 1.7 end to end, which swamps the
1.04× that instrumenting 1.11 costs in the first place; the cell that now
carries the coverage has itself come back at 12 m 12 and 17 m 03. The
orderings that decided where coverage goes (12–17 against 42 minutes, and
each of the two axes reproduced across the other) are far outside that
band and are safe; any comparison of two cells within a factor of two is
not, and should be read as the same measurement twice. The flat
`timeout-minutes: 30` has 1.8× of headroom over the slower of the two
draws, which is about what the uninstrumented cells had before.

Two more economies come from the same measurement, since `timeout` and
matrix size are both set by what a cell costs. The cap is a flat
**30 minutes**, and it is flat *because* coverage sits on the fastest
cell — every healthy run in the matrix is six to seventeen minutes, so
one number guards all four the same way. It was briefly per cell,
`${{ matrix.coverage == true && 40 || 30 }}`, while coverage lived on
Linux at the current release, where a **healthy** run measured 42 m 45
and a flat 30 would have failed it. If coverage is ever moved back to a
1.13 or a Linux cell, the per-cell form has to come back with it.
And the
matrix is spelled out one cell at a time instead of as a
`version × os` product, because the product's fourth combination is
redundant whichever one it is, and which one is left out follows from the
coverage choice above. Three axes, three serial cells: the floor on macOS
(which carries the coverage, and where a `TRACKED_1D`-style collision is
red while 1.13 is green), the current release on Linux, and the current
release on macOS (which carries the arm64 reduction claim above). The
pair left unbuilt is the floor on Linux: the floor is a property of the
Julia version rather than of the operating system, so one platform
exercises it. Pushes whose every path
matches `**.md` are skipped entirely — half the commits here are
`Record step N …` and a Markdown change cannot move a measured number —
while pull requests carry no such filter, so a PR always ends with a
check on it. Nothing about the suite's contents has to change; a
`D ≥ 2` sweep is seconds of arithmetic and belongs wherever the claim it
makes belongs.

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
    close the question and Sedov must. *(It did, in step 9, and with the
    opposite sign: on a strong shock crossing a coarse-fine face `p = 1`
    floors nothing where `p = 3` floors 4096 owned cells, for 0.47% of L1.
    The two rows measure different things — whether a prolongation ever
    spans a shock — and `p = 3` stays the default. See
    [Operator order](#operator-order).)*

  What step 7 wrote but did not exercise: every case but Sod and the
  entropy wave. (The atmosphere reset, which was the other item on that
  list, landed in step 8; see [Measured
  results](#step-8--the-atmosphere-reset).)
- **H4 — Sedov.** *(Done.)* Floors and the atmosphere reset, the `D = 2`
  and `D = 3` blasts, the similarity checks, the hook on edges and corners.
  *Accept:* the exponent and the jump; the reset idempotent and `U`/`P`
  consistent after it; the injection measured and the drift net of it
  at roundoff, `:stage` against `:step`; floor counts by population; the
  shell tracked to the
  uniform fine reference at fewer cells; block count rising then
  falling behind the shock; the `p = 1` / `p = 3` comparison and the
  ghost-floor count that decides the upstream question. The 3D adaptive
  run at test size. *(The reset, its accounting and both floor counts
  landed in step 8 and are measured there on cases where nothing fires;
  what is left for this milestone is the blast itself, which is the case
  where they do.)*

  What step 9 measured, all of it in
  [Measured results](#step-9--the-sedov-blast) and all of it identical at
  one and at four threads:

  - **The law, twice over.** `ξ₀ = 1.0327774677614250` for `γ = 7/5` in
    3D, which is Taylor's 1.033, from a parametrization derived rather
    than transcribed and checked against the one similarity equation it
    was not built from (residual `3.6e-14`). The measured exponents are
    **0.64146, 0.50443, 0.43766** against `2/3, 1/2, 2/5`, and the peak
    jumps **4.111, 3.765, 2.057** against the strong-shock 6, approached
    from below as a captured shock must.
  - **The tracked blast reproduces the uniform fine run to roundoff** —
    `8.3e-15` at 12544 cells against 16384 — with `tracking == 1` in
    every dimension, conservation at roundoff through the regrids, and a
    boundary that contributes exactly nothing, unlike Sod's.
  - **A tracked mesh cannot measure its own coarse-fine faces**, and that
    is the step's sharpest finding: tracking puts the refined region's
    boundary ahead of the shock, so `fixup = false`, `p = 1` and
    `reset = :step` come back identical to the run they control. The
    static `sedov_forest(:center)` mesh, where the blast leaves the
    refined region, is where every interface claim is made — and there
    the fixup buys `3.2e12` in mass in `D = 2` and `8.5e7` in `D = 3`.
  - **The floors fire, and neither rule nor place is what was predicted.**
    The evacuated interior never reaches `ρ_atm`, so the atmosphere rule
    never fires and every mass injection stays exactly zero; what fires is
    the *pressure floor*, driven by the interface flux restriction itself,
    4096 owned cells and 40 ghost entries in `D = 2` and 24504 and 4703 in
    `D = 3`.
  - **`p = 1` buys exact positivity for 0.47% of L1**, which closes the
    open question of [Operator order](#operator-order) with the opposite
    sign from Sod's row — and the two rows do not disagree, they measure
    whether a prolongation ever spans a shock.
  - **The injection is a bound under `:stage` and an equality under
    `:step`**, because SSPRK33's stages carry weights `1/6, 2/3, 1`;
    measured ratios `0.51984` and `0.52203`, and `|drift − injection|` of
    `4.4e-16` and `2.2e-16` under `:step`.
  - **The M2 ordering case in full**: zero mismatched entries out of
    1664, 72000 and 59360 stored outward-facing ghost entries on a 2D
    corner, a 3D edge and a 3D corner, with the fine corner block's
    density and energy bit-identical to the ambient.

  What the acceptance list got wrong, and it is one item: **the block count
  does not rise and then fall**. It rises monotonically, because the Sedov
  interior is a steep density ramp rather than a flat bubble and a Löhner
  indicator on `ρ` correctly fires throughout it; the refined region is a
  growing disk. The hollow opens only at the very centre and only late. The
  milestone is marked done on the strength of what that item was *for* — a
  mesh that follows a closed expanding surface, at a real saving — which is
  met; a criterion that would produce a shell is listed under
  [Possible extensions](#possible-extensions).
- **H5 — Kelvin–Helmholtz.** *(Done.)* The McNally setup, `M(t)` and the kinetic
  energy diagnostic, HLLC, the viewer. *Accept:* `M(t)` grows below the
  incompressible bound and converges toward the uniform fine run as the
  cap rises; conservation through regrids; HLLE against HLLC measured and
  the default chosen; the filmstrip rendered in CI.

  **Step 10 did the physics; the picture is what is left.** Done: the setup
  checked against the paper term for term, `mode_amplitude` in the
  area-weighted form, `max_y_kinetic_energy`, `growth_rate` and its window,
  the case and `kh_run`/`kh_uniform`, the measured growth rate 2.58036
  below both the `4.384` and the `5.9238` bounds, conservation to roundoff
  through the regrids with a five-to-six-order leak without the fixup, the
  cap sweep converging onto the uniform fine run, the `Float32` claim, and
  the HLLE/HLLC measurement with HLLC chosen as this case's default. Not
  done when step 10 closed: the **viewer and the filmstrip in CI**.
  **Step 11 built it and H5 is done.** `bin/visualize2d.jl --case=kh`,
  `bin/Project.toml` and the `viewer` job in `CI.yml` render the filmstrip,
  the two diagnostics against the uniform fine run, and the block count, on
  every push. The viewer adds no time stepping of its own: it reads
  `kh_run`'s observer through the pass-through step 11 gave it. The last
  clause of the acceptance — "the filmstrip rendered in CI" — is **closed
  on the first run of the job**, PR #1: the `viewer` job passed in 11 m 43,
  and the `kh_2d.png` it uploaded carries the same 232 blocks, the same
  `M(1.5) = 0.12346` and the same fitted 2.58036 as the local render.
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

### Step 8 — the atmosphere reset

`reset_atmosphere!` in the integrator's limiter hook and after every
regrid, its injection accounting, and the ghost population of the floor
count. The design is in [Floors and the
atmosphere](#floors-and-the-atmosphere); these are the numbers, from
`test/reset_tests.jl`. The synthetic state the first four claims are made
on is a uniform periodic mesh whose owned cells cycle through six
populations — healthy gas, `0 < ρ < ρ_atm`, `ρ = 0`, `ρ < 0`, `ρ = NaN`,
and a healthy density with `E < ½S²/ρ` — so that each of the two rules is
reached by every route it has.

**Idempotence is bit for bit, and the flag is not.** Applying the reset
twice gives a state identical to applying it once, `|twice − once|∞ = 0`
exactly, at `Float64` and at `Float32` in `D = 1, 2, 3`. The prediction
said "to roundoff, bit-for-bit not claimed", and the round trip it was
worried about is real — it just does not move the state:

| | cells | floored | second pass reports |
|---|---|---|---|
| `Float64`, `D = 1` | 16 | 13 | **2** |
| `Float64`, `D = 2` | 256 | 213 | 0 |
| `Float64`, `D = 3` | 512 | 426 | **85** |
| `Float32`, `D = 1, 2, 3` | 16 / 256 / 512 | 13 / 213 / 426 | 0 |

A pressure-floored cell recovers its internal energy through the
cancellation `E − ½S²/ρ`; where the kinetic energy dominates, the
recovered pressure lands a fraction of an ulp below `p_floor` and the rule
fires again — and writes `prim2con(eos, (ρ, v, p_floor))` from the same
`ρ` and `v`, which is the same arithmetic on the same numbers. So the
reset is a fixed point of the state and not of its own report.

**`U` and `P` agree, and healthy cells do not move.** `con2prim` of the
reset `U` reproduces the floored `P` to roundoff of the data's own scale
(not of `p_floor`, for the same cancellation reason), and every cell in
which no floor fired is **bit-identical** to what it was — which is the
property the next claim rests on.

**The injection is the state's own change.** On the synthetic state the
accumulated `injection` equals the hand-computed `Σ hᴰ (U_after −
U_before)` to roundoff of `Σ hᴰ (|U_after| + |U_before|)`, per variable,
and `reset_hits` equals the number of floored cells exactly. The momentum
entries are nonzero only because one population keeps its momentum: the
atmosphere rule discards it, the pressure floor does not. A state carrying
a `NaN` has no total, so its injection comes back `NaN` in that variable
and finite in the others — reported rather than hidden, and the state is
repaired all the same.

**Where nothing fires the reset costs exactly nothing.** On the tracked
shock tube (`D = 1`, `t_end = 1/50`, 49 steps, 1 mesh change) and on the
entropy wave through the driver (`D = 1`, `N = 8`, 50 steps):

| | injection | reset hits | ghost hits | owned floor hits |
|---|---|---|---|---|
| Sod, `reset = :stage` | **(0.0, 0.0, 0.0)** | 0 | 0 | 0 |
| Sod, `reset = :step` | **(0.0, 0.0, 0.0)** | 0 | 0 | 0 |
| entropy wave, `:stage` | **(0.0, 0.0, 0.0)** | 0 | 0 | 0 |

and the final state, the drift, the step count, the error and the mesh
history of both hooks are **bit-identical** to the `reset = :none` run's.
That is what makes every roundoff drift bound recorded above still the
same number now that `:stage` is the default and every run in the suite
calls the reset three times per step.

**The hooks are wired, and that is asserted rather than read.** One
`SSPRK33` step on a box half filled with gas and half with a vacuum a
hundred times below `ρ_atm = 1e-4`, at both types in `D = 1, 2`: the
minimum owned density comes out **1e-4** — the atmosphere exactly — under
`:stage` and under `:step`, and **1e-6** under `:none`, with the raw
recovered pressure `1.0e-5` against `-1.0`. The control matters because
the right-hand side floors `P` internally, so a `:none` run completes and
looks healthy from the outside while its *state* is untouched.

**The ghost population is counted over the stored extent.**
`ghost_floor_hits` equals the hand count of ghost *entries* whose
`con2prim` fired, at both types in `D = 1, 2`, with `floor_hits` equal to
the owned count from the same sweep — the split the Sedov milestone
reports against the ghost-exchange decision.

**What the default reset costs.** The whole suite is **11 282 tests in
1 m 42.9 at four threads and 1 m 55.8 at one**, against 11 149 in 1 m 42.3
and 1 m 36.8 on the same machine before the step — but most of that
difference is the new file's own twenty-odd evolutions and the
compilation they move earlier, and two runs of the same tree differ by
seconds, so the reset's price is better read off a controlled comparison.
The tracked tube of `driver_tests.jl` (`D = 1`, `t_end = 1/5`, 588 steps,
9 mesh changes), best of three:

| | 1 thread | 4 threads |
|---|---|---|
| `reset = :none` | 0.020 s | 0.150 s |
| `reset = :step` | 0.020 s | 0.169 s |
| `reset = :stage` | 0.022 s | 0.178 s |
| `:stage` with `accounting = true` | 0.027 s | 0.381 s |

So the per-stage reset costs **10% at one thread and 19% at four** of a
run that is nothing but driver, and `:step` costs a third of that — the
comparison Sedov will make on accuracy is cheap in either direction. (It
made it in step 9, and the answer is that the two give the same state to
roundoff, `:stage` making three times the repairs and reporting a bound
where `:step` reports an equality; see
[Step 9](#step-9--the-sedov-blast).) The
injection accounting costs **2.5× at four threads**, which is what six
extra whole-state reductions per step buy and exactly why it is a keyword
the tests turn on and the demos do not. (The four-thread column being
slower than the one-thread column is this mesh being tiny — 25 blocks of
8 cells — and is the launch overhead the benchmark of step 14 is for, not
anything the reset introduced.)

Every one of the 75 `@info` lines the suite printed before this step is
printed **byte-identically** after it; the 20 new ones are identical at
one thread and at four.

### Step 9 — the Sedov blast

`test/sedov_tests.jl`, 225 tests, `Float64`, `:minmod`, HLLE, the
conservative family at `p = 3` and the thresholds step 6 calibrated. The
tracked configurations are `roots = 4`, `N = 8`, `cap = 2` in `D = 1, 2`
and `roots = 4`, `N = 4`, `cap = 1` in `D = 3`, on `[−1/2, 1/2]^D` with
`γ = 7/5`, `ρ₀ = 1`, `p_amb = 10⁻⁵` and `E₀ = 1`.

**The similarity law.** Computed from the parametrization derived in
`src/sedov_reference.jl` rather than transcribed:

| `D` | `α` | `ξ₀ = α^{-1/(D+2)}` | recalled Kamm–Timmes `α` |
|---|---|---|---|
| 1 | 1.0774855847350489 | 0.9754301541925205 | 0.5386 — *half*, see below |
| 2 | 0.9840740168800447 | 1.0040216061302776 | 0.9840 |
| 3 | 0.8510718547582286 | **1.0327774677614250** | 0.8511 |

`ξ₀(7/5, 3) = 1.0328` is Taylor's classical 1.033, which is the check this
file named in advance; `ξ₀(5/3, 3) = 1.1516664179314904` is the other
widely quoted value. The planar `α` is **exactly twice** the recalled one,
which identifies the convention rather than an error: `σ_1 = 2` counts both
sides of `|x| < r`, matching this package's deposition volume `V_1 = 2r₀`,
and Kamm & Timmes count one. The literature values are **as recalled and
unverified against the report** — the public mirrors of its code are gone —
but the agreement in every recalled digit for `D = 2, 3` and to an exact
factor of two for `D = 1` is what the recall is worth. The derivation's own
check is the **momentum equation**, which is the one of the three the
closed form was *not* built from: worst residual `3.62e-14` over the whole
profile in all three geometries at `γ = 7/5` and `5/3`.

**The blast, tracked.** The exponent is fitted over the chunks with
`r_s ≥ 3r₀` (`2r₀` in `D = 3`):

| `D` | exponent | `2/(D+2)` | `r_s/r₀` reached | peak jump | steps / chunks / regrids | cells |
|---|---|---|---|---|---|---|
| 1 | **0.64146** | 0.66667 | 3.81 | 4.11089 | 222 / 25 / 2 | 112 |
| 2 | **0.50443** | 0.50000 | 5.34 | 3.76529 | 774 / 40 / 6 | 12544 |
| 3 | **0.43766** | 0.40000 | 2.72 | 2.05717 | 201 / 15 / 3 | 32768 |

`tracking == 1.0` in every dimension. The peak jump approaches the
strong-shock `(γ+1)/(γ−1) = 6` from below and never exceeds it, as a
captured shock must: the peak is the average over the cell the front sits
in, and it falls with the dimension because the same `h` resolves less of a
spherical shell. The `D = 1` and `D = 3` exponents are the loose ones and
the reason is in the `r_s/r₀` column — the law is the *asymptotic*
solution and those runs are fitted over a blast that has barely forgotten
its top hat.

**The measured `E₀`, and the deposition.** The initial-data cycle converges
in 4, 4 and 3 passes and puts the top hat at the cap:

| `D` | `p_hot` | `c_s` at `t = 0` | top hat, cells across at the cap | measured `E₀` |
|---|---|---|---|---|
| 1 | 3.2 | 2.11660 | 16 | **0.999996875** |
| 2 | 32.59493 | 6.75521 | 16 | **1.0345068127145072** |
| 3 | 48.89240 | 8.27341 | 8 | **1.0444541004175152** |

In `D = 1` the cells that receive the deposition tile `V_1(r₀) = 2r₀`
exactly, so the whole difference from the nominal `E₀ = 1` is the ambient
share of the top hat itself, `p_amb·2r₀/(γ−1) = 3.125e-6`. In `D = 2, 3` it
is which cell centres fall inside a circle and a sphere, and the ratio is
3.5% and 4.4%.

**The first chunk outgrows the speed it was sized from**, which corrects
"`λ_max` only decreases" in [Sedov blast wave](#sedov-blast-wave):

| `D` | `λ` at chunk 1 | `λ_end` at chunk 1 | growth | worst over the run | `λ` at `t_end` |
|---|---|---|---|---|---|
| 1 | 2.11660 | 2.46670 | **1.16537** | 1.16537 | 2.02621 |
| 2 | 6.75521 | 8.45480 | **1.25164** | 1.25164 | 4.26806 |
| 3 | 8.27341 | 9.13354 | **1.10396** | 1.10396 | 5.65649 |

The largest growth is the first chunk's in every dimension, `λ` falls
monotonically afterwards, and `speed_headroom = 2` covers all of it.

**The tracked mesh against the uniform ones, `D = 2`.** Reduced onto the
32² grid the three meshes share:

| run | cells | \|· − fine\| | peak | exponent |
|---|---|---|---|---|
| tracked | 12544 | **8.293829e-15** | 3.7652916 | 0.5044332 |
| uniform fine | 16384 | — | 3.7652916 | 0.5044332 |
| uniform coarse | 1024 | 1.118716e-1 | 2.2591980 | 0.5087455 |

The tracked run reproduces the uniformly fine run **to roundoff**, which is
a sharper statement than the tube's 1.0004 ratio and has a reason: outside
the refined region the gas is undisturbed, so the coarse blocks hold the
ambient exactly and there is nothing for them to get wrong. Conservation on
the tracked mesh is therefore the plain claim — drift
`(2.2e-16, 1.3e-17, 2.9e-17, 4.4e-16)` against roundoff bounds
`(1.4e-12, 3.2e-13, 3.2e-13, 1.4e-12)` over 774 steps and 6 mesh changes,
injection exactly `(0, 0, 0, 0)`. Unlike Sod's, the *boundary* contributes
nothing: the ambient is at rest and the two faces of each axis carry the
same pressure flux.

**What the tracked mesh cannot measure.** All three controls come back
identical to the run they control:

| run | exponent | peak | cells | floor / reset / ghost hits |
|---|---|---|---|---|
| `p = 3`, `:stage`, fixup | 0.5044332 | 3.7652916 | 12544 | 0 / 0 / 0 |
| `fixup = false` | 0.5044332 | 3.7652916 | 12544 | 0 / 0 / 0 |
| `p = 1` | 0.5044332 | 3.7652916 | 12544 | 0 / 0 / 0 |
| `reset = :step` | 0.5044332 | 3.7652916 | 12544 | 0 / 0 / 0 |

and `:step` is **bit-identical** to `:stage` in the final state and the
drift. The block count rises **monotonically**, 40 → 88 → 112 → … → 196
over 40 chunks, with 20 of the final 196 blocks below the cap; the refined
region is a growing *disk* and not a shell, for the reason recorded under
[Sedov blast wave](#sedov-blast-wave).

**The static two-level mesh, `refined = :center`, `D = 2`** — 28 blocks,
levels `[0, 1]`, 433 steps, the shock reaching `r_s = 0.34587` well past
the coarse-fine face at `1/4`, and every control at the same step count:

| run | mass drift | energy drift | reset hits | ghost hits | injection (E) | L1 vs fine |
|---|---|---|---|---|---|---|
| `p = 3`, `:stage` | 1.332e-15 | 1.2465883e-5 | 4096 | 40 | 2.3980402e-5 | 4.406604e-2 |
| `p = 3`, `:step` | 1.443e-15 | 1.2457381e-5 | 1384 | 40 | 1.2457381e-5 | 4.406604e-2 |
| `p = 3`, `:none` | 1.221e-15 | **1.554e-15** | 0 | 40 | — | — |
| `p = 1`, `:stage` | 1.665e-15 | **2.0e-15** | **0** | **0** | 0 | 4.427249e-2 |
| `fixup = false` | **4.237033e-3** | **2.9292753e-2** | 0 | 24 | 0 | 6.042174e-2 |
| uniform coarse | 6.66e-16 | 1.1e-16 | 0 | 0 | — | 9.536813e-2 |
| uniform fine | 0 | 1.1e-15 | 0 | 0 | — | — |

Roundoff bounds for the row above: `(7.7e-13, 1.8e-13, 1.8e-13, 8.0e-13)`.
Four things are in that table. The fixup buys **3.2e12** in mass and
**2350×** in energy on a mesh the shock crosses, and the uniform fine run
is **bit-identical** with the fixup and without it, so the leak is the
coarse-fine face's. The `:none` row shows the energy drift of the other
rows is *entirely* the reset's injection and not a leak. Under `:step` the
drift **equals** the injection to `4.44e-16` against a bound of `7.96e-13`;
under `:stage` it is `0.51984` of it, the SSPRK stage weights. And `p = 1`
floors nothing at all, for `0.47%` of L1.

**The same mesh in `D = 3`** — 120 blocks, levels `[0, 1]`, `N = 4`,
`r₀ = 1/8`, 133 steps, `r_s = 0.28082`, peak 1.93896:

| run | mass drift | energy drift | floor / reset / ghost hits |
|---|---|---|---|
| `p = 3`, `:stage` | 3.251e-11 | 4.4512289e-5 | 75 / 24504 / 4703 |
| `p = 3`, `:step` | 3.251e-11 | 4.4472762e-5 | 49 / 8232 / 4581 |
| `fixup = false` | **2.758779e-3** | **5.8954619e-2** | 0 / 0 / 4224 |
| uniform coarse | 2.773e-10 | 3.0e-14 | 0 / 0 / 0 |

The 3D coarse-fine face under a real shock, where the fixup averages
`2 × 2` fine faces: **8.5e7** in mass and **1324×** in energy. The mass
drift is *not* roundoff and is *not* the face's — the uniform mesh at the
same coarse spacing leaks eight times more, which is a strong blast's
numerical precursor reaching the Dirichlet boundary, the same trap Sod's
tube sets. Under `:step` the drift equals the injection to `2.22e-16`
against a bound of `2.47e-13`.

**Floor counts by population**, which is the measurement the design rests
on:

| run | owned (recovery) | owned (reset) | ghost |
|---|---|---|---|
| tracked `D = 2` | 0 | 0 | 0 |
| tracked `D = 3` | 0 | 288 | 1152 |
| `:center` `D = 2` | 0 | 4096 | 40 |
| `:center` `D = 3` | 75 | 24504 | 4703 |
| uniform `D = 2` | 0 | 0 | 0 |

The **atmosphere rule never fires**: the minimum density is `6.7e-2`, not
`10⁻⁶`, so every mass injection is exactly zero and every momentum
injection is roundoff (`-1.65e-24` in `D = 3` against a bound of
`2.2e-14`). What fires is the pressure floor, driven by the coarse-fine
face. See [Floors and the atmosphere](#floors-and-the-atmosphere).

**The corner and the edge of the Dirichlet box**, a few tens of steps with
the blast at the centre:

| mesh | blocks | steps | outward-facing ghost entries | mismatched | corner block's moved entries | worst \|ΔU\| |
|---|---|---|---|---|---|---|
| `D = 2` corner | 19 | 44 | 1664 | **0** | `[0, 0, 0, 0]` | 0 |
| `D = 3` edge | 92 | 40 | 72000 | **0** | `[0, 64, 64, 64, 0]` | 1.476e-54 |
| `D = 3` corner | 71 | 40 | 59360 | **0** | `[0, 64, 64, 64, 0]` | 1.297e-56 |

One ulp of the ambient energy density is `3.39e-21`, so the largest motion
is `10⁻³³` of it; the density and the energy are bit-identical in every
case and only the momenta, which start at exactly zero, move at all. See
[Boundaries](#boundaries).

**The suite.** 11507 tests in **2 m 43 at one thread and 2 m 09 at four**
(wall 2 m 51 and 2 m 14), against step 8's 11282 in 1 m 50 and 1 m 42 — so
the blast adds roughly 50 s and 30 s, which is what a case with a 3D
adaptive run and eleven two-dimensional evolutions costs. Run to run this
machine moves by about ten percent at four threads, so the delta is worth
one significant figure and no more. Every one of the 95 `@info`
lines the suite printed before this step is printed **byte-identically**
after it, and the 28 new ones are identical at one thread and at four.

### Step 10 — Kelvin–Helmholtz

The configuration everything below is measured on: `roots = 4`, `N = 8`,
`cap = 2` on the unit square (so 128² at the finest level),
`chunk = 1/200`, `t_end = 3/2` as the paper runs it, `limiter = :minmod`,
`p = 3`, `cfl = 2/5`, `speed_headroom = 1`, the step 6 thresholds
`refine_tol = 0.08` and `coarsen_tol = 0.02`, and — decided by the flux
measurement below — `riemann = :hllc`. 2700 steps in 300 chunks. The
controls (`fixup = false`, `Float32`) run to `t = 2/5` instead, which is
still inside the transient and is where their claims are visible; the
cap sweep and the flux comparison run to `3/2` with the headline pair.

**The transcription check.** `CODE.md`'s profiles, parameters,
perturbation, `γ = 5/3`, `p = 5/2` and `t = 1.5` are
McNally, Lyra & Passy's equations (1)–(5) **term for term**: the test
re-evaluates them from the paper's own literals and the worst difference
over 38 × 132 sample points is **exactly zero**. The one thing that was
wrong is the weighting of `M(t)`, which read "the lower interface alone"
and is in fact mirrored over both interfaces; it is corrected above, as is
the absence of the area-weighted form (14)–(17) that an adaptive mesh
needs. The initial data's closed-form integrals check the branch *signs*,
the `ρ_m` terms of the four branches cancelling exactly:

| | closed form | discrete, on the adapted mesh |
|---|---|---|
| `Σ h² ρ` | `(ρ₁+ρ₂)/2 = 3/2` | 1.4999999999999996 (1.3 ulp) |
| `Σ h² S_x` | −0.2125022699707237 | −0.2125079781452482 (2.69e-5 relative) |
| `Σ h² S_y` | 0 | −1.4e-19 |

The `S_x` difference is the midpoint rule's own error and not a defect: the
`e²` term of `ρ v_x` has width `L/2 = 1/80` against `h = 1/128`, where `ρ`
alone has width `L` and integrates to roundoff.

**The instability.** Measured on the tracked run, with the uniform fine
run beside it:

| | tracked, cap 2 | uniform fine 128² |
|---|---|---|
| cells | 14848 | 16384 |
| `M(0)` | 0.010000 (the seed, exactly) | 0.010000 |
| `M` minimum | 0.008063 at `t = 0.335` | — |
| `M(1.5)` | 0.1234566 (×12.346) | 0.1239814 |
| `max ½ρv_y²` at `t = 1.5` | 0.0373349 (×375.0) | 0.0375488 |
| growth rate over `2a ≤ M ≤ 6a` | **2.58036** | 2.58324 |
| kinetic-energy rate, same window | 5.42416 | 5.42662 |

The window is `t ∈ [0.735, 1.150]`, 84 of the 301 samples. The rate
**2.58036** lies below both bounds — Wang et al.'s `4.384` and the
sharp-interface `5.9238` — as it must, and the kinetic energy's rate is
**2.1021** times the mode's, which is the factor of two that says the two
diagnostics are measuring the same thing. The mesh does not set the rate:
the uniform fine run agrees to 0.1%.

**`M(t)` has not saturated by `t = 1.5`, and it is decelerating.** The
local logarithmic derivative falls from **3.349** at `t ≈ 0.5`, through the
window's 2.580, to **1.844** over the last twenty chunks, and `M` is still
at its maximum at `t_end`. So the run ends on the shoulder of the curve
rather than on its plateau, which is the reason the prediction above now
says "eventually saturates" rather than "saturates by `t_end`".

**Against the paper's curves, the first half of the shape is reproduced and
the second half is under-resolved.** The reference in their Figure 7 starts
at `0.01`, dips and plateaus until `t ≈ 0.45`, then takes off; this run
does the same, with its minimum of 0.008063 at `t = 0.335` and its take-off
at `t ≈ 0.45`, which is a corroboration and not a coincidence — the
transient is the seeded mode not being the eigenmode, and it is the paper's
too. What does **not** reproduce is the late curve: the reference reaches a
few tenths at `t = 1.5` and its local rate does not fall, while this run
reaches **0.1235** and its rate does. That is under-resolution rather than
saturation, and it is the expected place for it: 128² is the lowest
resolution the paper runs, the codes that sit on the reference there are
piecewise-parabolic (Enzo, Athena) or sixth-order (Pencil), and this scheme
is second-order MUSCL. The honest statement is that this package's
Kelvin–Helmholtz run has the right shape, the right bounds and the right
convergence with the cap, and an amplitude at `t = 1.5` below the published
reference's by something like a factor of two at its finest affordable
mesh.

**The mesh.** The initial-data cycle converges in **4 passes** onto 160
blocks — 128 at the cap and 32 at level 1, 10240 cells against the uniform
fine mesh's 16384 — and **every edge of every block at the cap lies within
`1/8` of `y = ¼` or `y = ¾`**, which is five ramp widths and a quarter of
the box in each strip. So the prediction's "two strips at `t = 0`" is
exactly right. The count then rises **160 → 196 (`t = 0.915`) → 220
(`t = 1.355`) → 232 (`t = 1.390`)**, monotonically, in 3 mesh changes, and
at `t_end` 224 of the 256 possible finest blocks are at the cap: the rolls
thicken the layer until the refined region reaches the middle of each slab,
and the run still never refines the whole box. `tracking == 1.0`
throughout, and the margin is **3 cells** at every regrid.

**The margin is what sizes the chunk here, and that is new.** The derived
buffer covers `speed_headroom · λ · chunk` at the cap's spacing; with
`λ = 2.5412` and `h_cap = 1/128`, `chunk = 1/200` gives 3 cells and
`chunk = 1/64` gives 7 — and 7 cells refines **all 256** finest blocks,
which is the uniform fine mesh under another name. A feature that *grows*
rather than travels gets nothing from a travelling margin and pays the
whole saving for it. On the tube and the blast the chunk was bounded from
above by `refinement_buffer` throwing; here it is bounded well below that,
by the saving going away quietly.

**The headroom.** Worst `λ_end/λ` over 300 chunks: **1.00031**, against
`speed_headroom = 1`. `λ` itself moves from 2.5412 to 2.6209 over the whole
run. This is the second case at `1` — the entropy wave is the other — and
for the same reason: there is no jump in the initial data, so there is no
star region absent from it.

**Conservation, and the leak.** There is no physical boundary here, so the
claim is the plain one; and unlike the blast, the coarse-fine faces carry
real flux, because the *whole domain* is in motion and the strips' edges
lie across the flow from `t = 0`:

| | mass | `S_x` | `S_y` | `E` |
|---|---|---|---|---|
| drift, 2700 steps, 3 regrids | 8.88e-16 | 2.78e-16 | 4.45e-18 | 1.78e-15 |
| roundoff bound | 7.19e-12 | 3.24e-12 | 4.01e-13 | 1.88e-11 |
| `fixup = false` (720 steps, `t = 2/5`) | 1.75e-7 | 1.93e-6 | 6.32e-18 | 9.14e-7 |
| its bound | 1.92e-12 | 8.64e-13 | 1.22e-14 | 5.00e-12 |
| ratio to the bound | **9.1e4** | **2.2e6** | 5.2e-4 | **1.8e5** |

with `floor_hits == reset_hits == ghost_hits == 0` and
`injection == (0, 0, 0, 0)` **exactly**, in both runs. So this is the case
step 9 wanted and could not have: a tracked mesh on which the interface
flux restriction is the difference between conservation and a leak.

**`S_y` does not leak, and that is measured rather than assumed.** Its
drift without the fixup is 6.32e-18 against a bound of 1.22e-14 — half a
thousandth of it, where the other three are five and six orders above
theirs. The coarse-fine faces are the horizontal edges of the two strips
and they span the whole of `x`; the `S_y` structure is the single mode
`sin(4πx)`, whose integral over `x` is zero, so the flux mismatch inherits
that zero mean. The other three integrals have no such symmetry protecting
them.

**The cap sweep**, reduced onto the 32² grid the four meshes share:

| | cells | L1 against the fine run | `\|ΔM\|/M` at `t_end` | mean `\|M − M_fine\|` |
|---|---|---|---|---|
| cap 0 (uniform coarse) | 1024 | 5.982e-2 | 0.9064 | 2.846e-2 |
| cap 1 | 4096 | 2.590e-2 | 0.3911 | 1.143e-2 |
| cap 2 | 14848 | **4.244e-4** | **0.004233** | 1.714e-4 |
| uniform fine 128² | 16384 | 0 | 0 | 0 |

All three columns fall monotonically with the cap, and the last step is a
factor of 60 in L1 and 90 in `M`. The adaptive run reaches the fine run's
answer at 91% of its cells — a smaller saving than Sedov's 77%, because a
shear layer occupies a band rather than a shell's neighbourhood, and the
honest statement is that this case is the one where adaptivity buys least.

**HLLE against HLLC** is in "Riemann solver" above, with the table and the
decision. The two numbers to keep here: the growth *rate* is **2.5804**
under HLLC and **1.1712** under HLLE at the same resolution, and HLLC at
64² is ahead of HLLE at 128².

**`Float32`**, to `t = 2/5` (720 steps), against the `Float64` run of the
same configuration: the **same** 160 blocks at levels `[1, 2]` at every one
of the 81 samples, the same step count, the same `tracking = 1.0` and the
same 4 cycle passes; `max |M₃₂ − M₆₄|/M = 1.857e-5`, which is **156 ulp of
`Float32`**, and `max |K₃₂ − K₆₄|/K = 2.552e-4`. The final state is *not*
claimed and is not asserted, for the reason the case description gives.
MultiFloats is skipped entirely: `sin` and `exp` are not implemented there,
so this case cannot run at `Float32x2` at all, and Sod and Sedov carry that
half of the precision study.

**The suite.** 11609 tests in **3 m 44 at one thread and 2 m 38 at four**
on a quiet machine, against step 9's 11507 in 2 m 43 and 2 m 09 — so the
shear layer adds roughly a minute at one thread and thirty seconds at four.
Runs taken while this (shared) machine was busy with other work reached
4 m 10 and 3 m 12, so the number carries one significant figure and the
comparison is only fair between measurements taken under the same load. The
gap between the two
thread counts is the largest in the package's history and is the case's
own shape: twelve two-dimensional evolutions, of which four run 2700 steps
on 128²-equivalent meshes, is the most parallel work any one file has held.
Every one of the 123 `@info` lines the suite printed before this step is
printed **byte-identically** after it, and the 11 new ones are identical at
one thread and at four. The clean-checkout check — a `git archive` tree
with no `Manifest.toml`, resolving TreeAMR from GitHub `main`, which is
what CI does and what the `[sources]` pin exists for — passes.

### Step 11 — the viewers and the figure job

The step that closes H5b and H5, and it is the first in this package whose
deliverable is a picture rather than a number. What it built:
`bin/Project.toml` (CairoMakie and SixelTerm in an environment of their
own, with `[sources]` for *both* packages), `bin/backend.jl` after
TreeWave's, `bin/visualize1d.jl` (the tracked tube) and
`bin/visualize2d.jl` (`--case=kh|sedov|both`), and the `viewer` job in
`CI.yml`.

**Every figure reproduces the recorded numbers**, which is the check that
the viewers are drawing the runs this document is about and not
configurations of their own. The tube renders 200 cells in 25 blocks over
588 steps, 40 chunks and 9 regrids at L1 `4.54e-3` and `tracking == 1` with
zero floor hits in either population — step 7's row. The blast renders
12544 cells in 196 blocks, measured `E₀ = 1.03451` against a nominal 1, a
fitted exponent of **0.50443** against `1/2` and a peak compression of
**3.7653** against the strong-shock 6 — step 9's row, to every digit it
records. The shear layer renders 14848 cells in 232 blocks against 16384
uniformly fine, with `M(0) = 0.0100` growing to `M(1.5) = 0.12346` at a
fitted **2.58036** — step 10's row.

**Three things the implementation settled:**

- **`kh_run` needed an `observer` pass-through, and the design did not say
  so** *(amended here)*. `evolve!` takes exactly one observer and `kh_run`
  installs it; the design had the viewer reading `kh_run`'s returned curves
  and never said how it would also get *frames*. A viewer that installed
  its own observer on `evolve!` would have displaced `kh_run`'s and taken
  `M` and `K` somewhere other than the one place this document says they
  may be taken. So `kh_run` gained `observer = nothing`, called after its
  own three diagnostics; a run that passes nothing is bit-identical to one
  from before the keyword. Asserted rather than assumed: the new testset in
  `test/kelvinhelmholtz_tests.jl` sees the hook called **81 times** on the
  `t = 2/5` control, at the same times and the same 160 blocks the case's
  own hook saw, with `Ms`, `Ks`, `nbs`, `ts`, the drift, the step count and
  the mesh history all equal — and every one of this file's recorded
  numbers is unchanged, `M → 0.1234566379057999` and the rate
  `2.5803559556363127` included. The suite goes from **11610 tests in
  4 m 13.6 to 11622 in 4 m 31.9** at one thread.
- **A tracked Sedov run floors nothing, and the figure has to say so.** The
  render reports `0` owned and `0` ghost floor hits, which is not a
  contradiction of step 9 but its central correction restated: tracking
  puts the refined region's boundary ahead of the shock, so a tracked run's
  coarse-fine faces stand in undisturbed ambient. The 4096 owned and 40
  ghost hits of step 9 are the *static* `sedov_forest(:center)` mesh's. The
  figure's title says which of the two it is showing, because a reader who
  knows the recorded number and sees a zero would otherwise conclude the
  floors had stopped working.
- **Two exported names collide with Makie, and one is new.** TreeAMR
  exports `scatter!` (the state-vector one) and Makie exports the plot
  recipe; TreeHydro exports `density` and Makie's `@recipe` exports
  `density`/`density!` too. Julia errors on any *use* of an ambiguous
  name, so `bin/` writes `CairoMakie.scatter!` in full and never writes
  `density` at all — it reads slot 1 of `P` directly. TreeWave met the
  first; the second is this package's own.

**Cost.** Locally, on a machine also running the suite: the tube renders in
**26 s**, the blast in **28 s**, and the shear layer in **53 s** — of which
roughly 20 s each is process start and `using CairoMakie`, and the rest is
the arithmetic (the shear layer runs *two* evolutions, the tracked one and
its uniform fine reference). A bare `visualize2d.jl`, which is
`--case=both`, does the pair in **66 s** rather than 81, that load being
paid once; CI still runs them as two steps, because a failure then names
which case failed. Instantiating `bin/` cold cost **98 s** of
precompilation on top of an already-warm depot; on a CI runner with nothing
cached, CairoMakie's stack is minutes and is expected to dominate the job.
**Measured on CI, cold, on the first run of the job** (PR #1, run
35367816414), and the prediction that precompilation would dominate is the
one thing it confirms:

| step | cold | warm |
|---|---|---|
| instantiate `bin/` | **7 m 31** | **5 s** |
| render the tube | 37 s | 35 s |
| render the tube at `Float32` | 36 s | 34 s |
| render Kelvin–Helmholtz | 1 m 49 | 1 m 31 |
| render Sedov | 48 s | 41 s |
| checkout, setup, cache, upload | ~20 s | ~28 s |
| **job total** | **11 m 43** | **3 m 57** |

So instantiating the environment is **64% of a cold run and 2% of a warm
one**: the cache is the whole story, and `julia-actions/cache` needs no
configuration to do it, its default key carrying the job name. The renders
are the same work either way, 3 m 50 cold against 3 m 21 warm, and cost
1.4–2.1× their local times — which is what a 2-vCPU runner does to this
arithmetic. The guard is the test job's `timeout-minutes: 30`, 2.5× a cold
draw and 7.6× a warm one.

**Two things this corrects.** The `--case=both` lever recorded above as the
fallback if the job ran long is the **wrong lever**: it saves one `using
CairoMakie`, which is ~20 s against a 451 s precompilation, under 3% of the
job. What actually governs this job's cost is the cache, and nothing in the
rendering. And the claim that the job "is not on the critical
path" because it runs beside the four test cells was **stated without a
cache state, which is the only thing that decides it**. Cold, the four
cells came in at 8 m 28, 9 m 52, 10 m 36 and 11 m 09 and the viewer's
11 m 43 was the *longest* job in the run. Warm, on the next run, the cells
came in at 7 m 55, 10 m 28, 12 m 34 and 10 m 37 and the viewer's 3 m 57 was
the *shortest* by a factor of two. So it is on the critical path only when
its cache is cold — a first run, or the first after a CairoMakie release
evicts the entry — and off it every other time. Note also the 4-thread cell
at 9 m 52 and then 12 m 34 on two consecutive runs of the same commit
range: 27%, and a reminder that none of these single draws is worth more
than an ordering.

**What was checked, beyond the figures looking right.** The suite is green
at 11622 tests both on the current release and — the check that matters for
a step which adds files — on the **floor version from a clean checkout**:
`git archive` into an empty tree, TreeAMR resolved from GitHub `main`,
`julia +1.11`, 6 m 32.2 and every number identical to the 1.13 run's. The
viewer environment was instantiated and rendered from that same clean tree,
which is the only thing that proves `bin/Project.toml`'s two `[sources]`
entries resolve — a local run, with a depot that already has everything,
proves nothing about it.

And **the device path works through the viewer**, which was not planned for
this step and is worth recording because it is the first time anything in
this package has run on one. `bin/visualize1d.jl --backend=metal
--type=f32` renders, and it renders the *same run*: 200 cells in 25 blocks,
588 steps over 40 chunks and 9 regrids, L1 `4.54e-3`, `tracking == 1`, zero
floor hits — every figure on it indistinguishable from the host `Float32`
render. That exercises `withbackend`'s `invokelatest`, and it exercises
`hostcopy` doing the thing it exists for rather than the CPU short-circuit
it usually takes. It is **not** the device milestone: H6c still owes the
per-phase table, the opt-in device tests and the benchmark, and nothing
here was measured for speed. It is one case, at one precision, saying the
plumbing is connected.

### A movie (added after step 11)

`--movie` on `bin/visualize2d.jl` keeps every frame the observer hands over
instead of the filmstrip's four and writes them as a video —
`bin/output/kh_2d.mp4` beside `kh_2d.png`, from the same run. The shear
layer's `chunk = 1/200` over `t_end = 3/2` is **301 observer calls**, so the
movie is 301 frames at 30 fps: ten seconds, 1.17 MB. It works for
`--case=sedov` too, the drawing code being shared.

This is an addition and not part of what H5 accepted, which is why it is
recorded here rather than folded into the step 11 entry. The strip shows
four states; the movie shows the *order*, which on this case is the whole
point — the seeded mode decays while the ramp sheds its transient, takes off
near `t = 0.5`, and only then rolls up, and the refined region thickens with
the rolls rather than travelling with them.

**It costs `bin/Project.toml` nothing.** `FFMPEG_jll` arrives as a
dependency of Makie, so `Makie.record` encodes `.mp4` and `.gif` with no new
entry — verified by encoding both before any of this was written.

**The figure is byte-identical whether or not a movie is asked for**, which
is the one property the design exists to protect: `keptframes` takes the
*union* of the filmstrip's four picks and the movie's, so `--movie` retains
more frames and draws the identical four. Checked with `cmp` on both cases,
not by inspection.

**The cost is superlinear in the frame count, and that was not expected.**
Measured on this machine, `--case=kh`:

| frames | wall clock | per extra frame |
|---|---|---|
| 2 (runs + figure only) | 57.8 s | — |
| 151 | 2 m 45 | 0.72 s |
| 301 | 7 m 31, and 7 m 23 on a second draw | 1.30 s |

Doubling the frames costs **3.66×**, not 2× — about `n^1.9`. The estimate
made before implementing was 1.9 min for 301 frames, from a synthetic
benchmark of the drawing alone; the real thing is four times that, and the
synthetic benchmark cannot be made to reproduce it. Three candidate
explanations were tested and **all three are wrong**: setting `ax.title`
per frame and the `Colorbar` cost nothing (0.28–0.34 s/frame with and
without), plots do not accumulate across `empty!(ax)` (the plot count is
flat over repeated cycles), and holding 301 frames live while drawing does
not slow drawing down on its own (0.24 s/frame with 301 live against 0.36
with 4). What is left is the combination — building 301 real snapshots
during the run and then drawing them — most plausibly garbage collection
marking a large live set once per frame, which would be `O(n²)`; that was
not confirmed and is recorded as unexplained rather than asserted.

The practical consequence is the useful part: **`--movie-frames=` is the
lever and not a convenience**. Halving the frames divides the *marginal*
cost by 3.66 and the total by 2.7, the runs and the figure being a fixed
58 s underneath: a 151-frame movie of the shear layer is a five-second
animation for 2 m 45 against 7 m 31. It is also what keeps the CI smoke test cheap — twelve frames
on the Sedov step, about five seconds, riding on the cheaper 2D case so
that the Kelvin–Helmholtz step goes on exercising the default no-movie
path. Both branches are covered for the cost of one.

One trap fixed while adding it: the artifact upload globbed
`bin/output/*.png`, so a movie rendered in CI would have been produced,
asserted non-empty by `test -s`, and then silently left out of the upload.
It is `bin/output/*` now.

**Why the job is ungated**, unlike the coverage step: coverage is a
*report* whose consumer reads only `main`, while this is a *check*, and a
pull request is exactly where a broken viewer should surface. `bin/` sits
outside `src/` and `test/` with its own environment and its own copy of the
TreeAMR pin, so nothing else in CI would notice it breaking — which is how
TreeWave's `bin/` went on building against a TreeAMR older than its own
tests until it failed on the removed `cell_center`. The four renders run as
four separate processes rather than one, because both scripts define
`main` and `const LEVELCOLORS` at top level in `Main` and Julia 1.11 — this
package's floor — refuses to redefine a `const`.

## Possible extensions

Not planned, listed because they are the obvious next questions:

- **A criterion that produces a Sedov *shell* rather than a disk**
  (measured in step 9, and deliberately not built). The Löhner indicator on
  `ρ` fires throughout the blast's interior, correctly — the similarity
  solution's `G(λ) ∼ λ^{D/(γ−1)}` is a steep, under-resolved ramp and not a
  flat bubble — so the refined region is a growing disk and the block count
  rises monotonically. Getting a shell means telling the criterion that a
  monotone ramp behind a shock does not need resolving, which a
  second-difference indicator cannot be told; it wants a different
  criterion (a shock detector, or a relative-amplitude threshold on the
  second difference) and a calibration of its own. The disk still tracks
  the blast and still saves cells, so this is an economy rather than a
  defect.
- **A limited, positivity-preserving prolongation** — `p = 3` accuracy with
  `p = 1`'s floor count. Step 9 put a number on what it would buy: on a
  strong shock crossing a coarse-fine face, `p = 1` floors nothing where
  `p = 3` floors 4096 owned cells and 40 ghost entries, and costs 0.47% of
  L1 for it. It is not a fixed-weight tensor-product stencil, so it is an
  upstream request rather than a keyword; it is still not made, because the
  repairs `p = 3` needs are ones the reset makes correctly and accounts for
  exactly.
- The Sedov radial profile *at a given radius*, making the radial-scatter
  figure quantitative. The parametric profile `(λ, G, V, Z)` came for free
  with the energy integral in step 9 and is in `sedov_profile`; what is
  missing is the inversion `λ ↦ V`, which is a root find nothing in the
  acceptance needs.
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
`con2prim` over the stored extent (was "may change, on the ghost floor
count"; step 9 measured that count and it does not change — see
[Floors and the atmosphere](#floors-and-the-atmosphere));
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
   decision already made. **(Measured in step 9.)** On the blast the two
   give the same final state to roundoff, `:stage` making three times the
   repairs and reporting a strictly less informative injection — a bound
   rather than an equality, because a stage's injection enters the step
   with that stage's SSPRK weight. `:stage` stays the default on the
   strength of what it was adopted for and what a blast in an ambient
   cannot show: a stage vector that the *next* stage's right-hand side
   never sees in an unphysical state, which is a star-in-a-vacuum
   property. See [Floors and the atmosphere](#floors-and-the-atmosphere).

Smaller defaults marked (proposed) in the text — the variable order,
point samples rather than cell averages for discontinuous initial data,
`λ_max` measured once per chunk — are implementation choices that the
first milestones will either confirm or amend in place. All three are now
settled: the variable order in step 1, the point samples in step 4 (and
generalized in step 7, where a case states its data pointwise because the
adaptation cycle changes `h` under it), and `λ_max` once per chunk in
steps 4 and 7 — confirmed, but only beside a `speed_headroom` factor and
a recheck that throws.
