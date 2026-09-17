# TreeHydro implementation plan

For the sessions that implement TreeHydro, one step at a time. The
**design** lives in `CODE.md` — read it first, in full; it is
authoritative, and this file is only the work breakdown: what each step
changes, what it must not change, and what it must measure and record.
`CLAUDE.md` has the mechanics and the traps. Delete this file when the
last milestone is marked *(Done.)* in `CODE.md`.

**Steps 0–10, 7b and 7c are done; step 11 is next.**

The steps map onto `CODE.md`'s milestones H0–H6, split so that every step
ends in a green test suite and a `CODE.md` update, and so that each is a
brief a single session can carry. The order is the dependency order.

## Ground rules, every step

- Read `CLAUDE.md` first, then the `CODE.md` sections the step names.
- **Work on a branch** named `claude/step-N-<slug>`, off `main`. Commit
  there in the style of TreeAMR's history (implement → measure → record,
  measured numbers in the commit body). **Do not merge into `main` and
  do not push**; report the branch and its commits. There is no remote
  yet; when there is one, the rule stands.
- **TreeAMR is pinned to GitHub `main`** through `[sources]`. The
  checkout at `~/src/jl/TreeAMR` is *not* what the tests see. If a step
  turns out to need something from TreeAMR, stop, describe exactly what
  and why, and report — do not edit that checkout and carry on.
- **Generic in `T` and in the backend from the first line.** Every driver
  takes `T` as a leading positional argument (default `Float64`) and
  `backend` as a keyword (default `CPU()`); no floating-point literal in
  an expression where `T` is in play (`T(7//5)`, `oftype(x[1], 2)`);
  every callback captures `isbits` only; per-block metadata a kernel
  reads goes through `to_backend`. Steps 12–14 add the *tests and
  measurements* of these properties, not the properties — TreeWave
  records that retrofitting them was a rewrite.
- Before and after each step: the full suite, once at the default thread
  count and once with `julia_args = ["--threads=4"]`.
- Spec-first: when the implementation shows `CODE.md` wrong or
  incomplete, amend it and say so in it ("amended in step N"), and
  replace each **(predicted)** the step measures with the measured
  number, marked **(measured in step N)**. Never loosen an existing
  assertion to get green; report the failure instead.
- Testset names are claims; each opens with a comment naming the failure
  mode it guards. Convergence rates, conservation drifts and mesh
  statistics are asserted as numbers with tolerances.
- **Every later step adds its physics claims to the suite**, which runs
  whole on every push, at one thread and at four. A convergence sweep, a
  table, a calibration or a tracked run goes in that case's `*_tests.jl`
  beside its unit claims; there is no second tier to put it in and none is
  wanted, because the arithmetic is seconds. What CI must not do is turn
  **code coverage** on: it costs a factor of a hundred under threads, and
  it is the whole reason the suite once looked too slow to run. See step 7c
  and "Testing" in `CODE.md`.
- One step at a time; the next starts from a green suite on `main`.

## Sharp edges to know before starting

The full list is in `CLAUDE.md`; these are the ones that bite while
writing kernels.

- **Three ghost widths.** `U` and `P` have `G = 2`, `F_d` has `G = 0`.
  Face `i` (in `1 … N+1`) lies between cells `i−1` and `i`; cell `i` is
  stored at `i + G_U`, the face at `i + G_F`. `coordinates` takes stored
  indices.
- **Three `map_blocks!` ranges.** Default and `closed = true`: the kernel
  adds `G[d]`. `stored = true`: the kernel's index *is* the stored index
  and adds nothing. `con2prim` is the `stored = true` kernel and nothing
  else is.
- **The RHS never mutates `u`.** The atmosphere reset is the SSPRK
  `stage_limiter!` (step 8) and a driver call after `regrid!`; the
  RHS-level floor touches `P` only.
- **`P` carries two diagnostic slots** beyond the `D + 2` primitives
  (amended here, and recorded in `CODE.md`'s "Field sets" in step 3):
  slot `D + 3` is
  the cell's signal speed `max_d(|v_d| + c_s)`, slot `D + 4` is the
  floor-hit flag, both written by the `con2prim` kernel. They exist
  because `block_mapreduce` maps a scalar function over one variable's
  values and cannot form `|v| + c_s` from three variables — so the kernel
  that has all of them writes the number once, and `λ_max` and the
  owned-cell floor count are plain `block_mapreduce` calls over one
  slot, deterministic upstream. The *ghost-cell* floor count needs the
  stored extent, which `block_mapreduce` does not offer: a one-item-per-
  block kernel summing slot `D + 4` over each block's stored cells, with
  the partials combined in block order (the same discipline).
- **The boundary hook goes to three places**: `fill_ghosts!`, `regrid!`,
  `adapt_to_initial_data!`.
- **Initial and boundary data use `AllVariables`**: `initial(x)` returns
  the primitive tuple, and the callback handed to TreeAMR is
  `AllVariables(x -> prim2con(eos, initial(x)))`.
- **Regrid pairs**: `(U => p.schedule, P => nothing, F_1 => nothing, …)`;
  afterwards rebuild `HydroProblem` passing the existing `prims` and
  `fluxes`, which `regrid!` already resized.
- **Per-direction launches unroll over `ntuple(Val(D)) do d … end`** so
  each launch gets a constant `Val(d)` (the Burgers pattern in TreeAMR's
  `test/burgers.jl`, which is the worked example for the whole RHS).
- **The limiter and the Riemann solver travel as `Val`s** (`Val(:mc)`,
  `Val(:hlle)`), never as `Symbol` arguments.
- **`OrdinaryDiffEqSSPRK`'s `SSPRK33(; stage_limiter!, step_limiter!)`**
  hooks have the signature `limiter!(u, integrator, p, t)`; `u` is the
  stage vector in state layout, `statearray(u, U)` gives the block view.

## Step 0 — Scaffolding (H0)

`CODE.md`: "File layout", milestone H0.

Changes:

- `Project.toml`: deps `TreeAMR`, `KernelAbstractions`,
  `OrdinaryDiffEqSSPRK`, `SciMLBase`; compat bounds; `julia = "1.11"`;
  the `[sources]` entry pinning TreeAMR to GitHub `main`, with TreeWave's
  comment on why it exists.
- `.gitignore` in TreeWave's image: `*~`, `*.swp`, `.DS_Store`,
  `/docs/build/`, `Manifest.toml`, `/bin/output/`, `TODO.md`.
- `src/TreeHydro.jl`: the module shell with its docstring and the
  includes that exist; `src/precision.jl` and `src/device.jl` ported
  from TreeWave's — same functions (`wrap`, `ceilint`, `floorint`,
  `tofloat64`; `to_backend`, `hostcopy`), same reasoning in the
  docstrings, with `hostcopy` written against the M8 `FieldSet`
  signature (`FieldSet{T}(fs.forest, fs.nvars; G = fs.G,
  centering = fs.centering)`).
- `test/Project.toml` (`Test`, `TreeAMR`, `KernelAbstractions`,
  `MultiFloats`, `OrdinaryDiffEqSSPRK`, `SciMLBase`),
  `test/runtests.jl`, `test/precision_tests.jl` (the Base-bridge claims
  at `Float64`, `Float32`, `Float32x2`), `test/prerequisite_tests.jl`:
  the pinned TreeAMR exports `AllVariables`, `map_blocks!` accepts
  `stored = true` and reaches the ghosts, `hostcopy` round-trips a
  staggered field set with per-dimension `G`.
- `.github/workflows/CI.yml` after TreeWave's (Julia 1.11 and release,
  Linux and macOS, one 4-thread entry), `.github/dependabot.yml`. No
  viewer job yet; there is no `bin/`.
- `README.md`: a short blurb with the status.

Accept: `Pkg.test()` green; a clean archive (`git archive HEAD | tar -x
-C /tmp/clean`) instantiates from the pin and passes; `CLAUDE.md`'s
"Current state" and "Commands" describe what now exists. Mark H0
*(Done.)* in `CODE.md`.

## Step 1 — Equation of state and floors (H1a)

`CODE.md`: "The equations", "Floors and the atmosphere" (the two rules
only; the reset is step 8).

Changes: `src/eos.jl` — `abstract type EquationOfState`, `IdealGas{T}`,
`pressure`, `internal_energy`, `soundspeed`, `prim2con(eos, P)`,
`con2prim(eos, floors, U) -> (P, hit)`; `src/floors.jl` — `Floors{T}`
(`ρ_atm`, `p_atm`, `p_floor`), `apply_floors(eos, floors, P) -> (P′,
hit)`, the one place both rules are written. States are `NTuple{D+2}` in
the order `(ρ, v₁…v_D, p)` / `(ρ, S₁…S_D, E)`; everything `isbits`, no
literals.

Accept: `con2prim ∘ prim2con` is the identity to roundoff on physical
states in `D = 1, 2, 3` at `Float64` and `Float32`; both floor rules
exercised with the flag correct, and `apply_floors` idempotent to
roundoff; `c_s² = γ p / ρ`; all of it callable from a trivial
KernelAbstractions kernel on `CPU()` (the `isbits` claim). Flip the
variable order from **(proposed)** to **(decided)** in `CODE.md`.

## Step 2 — Reconstruction and Riemann fluxes (H1b)

`CODE.md`: "Reconstruction", "Riemann solver".

Changes: `src/reconstruction.jl` — `slope(::Val{:none|:minmod|:mc}, a,
b)`, `face_states(lim, eos, floors, P₋₂, P₋₁, P₀, P₊₁) -> (P_L, P_R)`
componentwise, floored; `src/riemann.jl` — `physical_flux(eos, P,
::Val{d})`, `riemann_flux(::Val{:llf|:hlle|:hllc}, eos, P_L, P_R,
::Val{d})`, `signal_speed(eos, P)` = `max_d(|v_d| + c_s)`. HLLC is
*implemented* here beside the other two (its unit tests belong with
theirs) and *measured* in step 10, as `CODE.md` says.

Accept: consistency, `F(P, P)` equals the physical flux for all three;
direction generality, the flux in direction `d` equals the direction-1
flux with components permuted; upwinding, HLLE returns `F_L` when
`s_L > 0` and `F_R` when `s_R < 0`; HLLC reproduces the exact flux of a
stationary contact where HLLE does not; `:minmod` and `:mc` vanish at
extrema and `:none` is the centered slope; face states under a TVD
limiter lie between the neighbouring cell values. `D = 1, 2, 3`.

## Step 3 — The right-hand side and the entropy wave (H1c)

`CODE.md`: "Field sets", "The right-hand side", "Time integration and
the time step", "Entropy wave". TreeAMR's `test/burgers.jl` is the
worked example.

Changes: `src/evolution.jl` — `HydroProblem`, `con2prim_kernel!`
(`stored = true`, writes the `D + 2` primitives and the two diagnostic
slots), `flux_kernel!` (`closed = true`, fused reconstruction and
Riemann solve), `divergence_kernel!` (with the empty source slot named
in a comment), `hydro_rhs!`, `max_signal_speed(p)`, `hydro_dt`,
`convergence_rate`; `src/entropywave.jl` — `hydro_forest(Val(D), N;
roots, refined)` (TreeAMR's M3 hierarchy, as `burgers_forest`), the exact
cell-average fill on the host, `entropywave_errors(T, Val(D); N, ops, G,
limiter, riemann, fixup, refined, backend)`. `P` is `FieldSet` with
`nvars = D + 4`; record that in `CODE.md` ("Field sets" table) as an
amendment with the reason under "Sharp edges" above.

Accept: on the uniform periodic mesh in `D = 1, 2` (3D smoke): L1 and
L∞ rates ≥ 1.9 with `:none`, recorded with `:mc`; every conserved
integral constant to roundoff with and without the fixup (the uniform
control); the RHS is pure (two evaluations at the same `u` give
identical `du`) and never mutates `u`. Record the rate table.

## Step 4 — Sod and the exact Riemann solver (H1d)

`CODE.md`: "Sod shock tube", "Boundaries".

Changes: `src/exact_riemann.jl` — Toro's pressure iteration and sampling,
host `Float64`; `src/sod.jl` — the states, `γ = 7/5`, the Dirichlet
hook `boundary_by_coordinates(AllVariables(x -> prim2con(eos,
initial(x))))`, `sod_errors(T, Val(D); N, roots, direction, …)` on a
uniform mesh to `t = 0.2`, the arrival assertion (`t_end · λ_max`
against the distance to the nearest physical boundary), `direction` to
run the tube along any axis.

Accept: the exact solver reproduces Toro's tabulated star region for
Sod's data (`p* ≈ 0.30313`, `u* ≈ 0.92745`, `ρ*_L ≈ 0.42632`,
`ρ*_R ≈ 0.26557`); L1 rate against it in `[0.7, 1.05]`, recorded;
direction independence *bit for bit* — the `D = 2` tube along `y` equals
the tube along `x` transposed, and the `D = 2` planar tube's profile
equals the `D = 1` run; the arrival assertion fires when `t_end` is too
long.

## Step 5 — The static two-level mesh (H2)

`CODE.md`: "Conservation at coarse-fine faces", "Operator order",
"Boundaries" (the last paragraph).

Changes: `refined = true` paths of steps 3 and 4; a Sod forest whose
refined region touches the Dirichlet face; nothing new in `src/` beyond
what the tests need.

Accept: all `D + 2` integrals to roundoff with the fixup on the entropy
wave (`D = 1, 2, 3`) and Sod (`D = 1, 2`), a leak without it, the
uniform control conserving either way; the interface-order table for
the system on the entropy wave at `p = 1, 3, 5` in L∞ and L1 (predicted
1, 2, 2 and 2, 2, 2, with `p = 3` matching the unrefined control's
rate); the boundary-touching Sod run conserving. Record the table and
the drifts, replacing the predictions.

## Step 6 — The refinement criterion (H3a)

`CODE.md`: "The refinement criterion".

Changes: `src/refinement.jl` — `lohner` with the local *and* global
floor terms, `cell_tau` over `ρ` and `p` (slots 1 and `D + 2` of `P`),
`field_scales` for the global reference, `hydro_flags(P; refine_tol,
coarsen_tol, maxlevel_cap, …)` through two `firing_boxes` sweeps with
TreeWave's four marks, `refinement_buffer(forest, cap, travel)`.

Accept: on Sod and top-hat blast initial data (one RHS pass so `P` is
current): `τ` fires at the discontinuities and nowhere else; an
atmosphere region with noise does not fire *because of* the global term
(the negative control sets `ε_g = 0` and watches it fire); boxes bound
the firing cells; a block at the cap reports `(Keep, box)`; the buffer
throws when the travel exceeds a block width; the calibration table of
max `τ` against `h` on uniform meshes for the shock, the contact, the
rarefaction and the Kelvin–Helmholtz ramp. Record the table and the
thresholds it picks.

## Step 7 — The driver and the tracked shock tube (H3b)

`CODE.md`: "Regridding: one driver, restart per chunk", "Sod shock tube".

Changes: `src/driver.jl` — `HydroCase` (initial primitives, boundary or
`nothing`, periodicity, extents, EOS, floors, reference), `evolve!` as
the pseudo-code in `CODE.md` with `reset = :none` until step 8, the
per-chunk `λ_max` and the CFL recheck, the per-chunk record (totals,
floor counts, block count, tracking), `observer`; `reduce_to_grid` and
`l1_difference` after Burgers'; `uniform_sod`; the tracked Sod in
`D = 1, 2`.

Accept: the initial-data cycle converges on Sod; the tracked tube
matches the uniformly fine reference at fewer cells with the uniform
coarse mesh as control; all `D + 2` integrals to roundoff through the
regrids and a leak without the fixup; the `p = 1` against `p = 3` table
on Sod; the buffer-width table; the CFL recheck's assertion tested on
synthetic numbers. Record the tables.

## Step 7b — Fast CI (no milestone; the suite itself)

*(Its diagnosis was wrong and its split was undone in step 7c; see there.
The text below is kept as it was written.)*

`CODE.md`: "Testing: two tiers", "File layout".

Why: GitHub CI took **53 minutes** on the last push before this step. The
four single-thread jobs took 3 to 6 minutes each; the one **4-thread
Ubuntu job** took 53, all of it in the tests. The log's timestamps
localize it — the two-dimensional physics sweeps are **10–17× slower under
4 threads** on the 4-vCPU shared runner than under 1 thread on the same
kind of runner (entropy wave `D = 2` 13 s → 2 m 57; the `D = 2`
interface-order sweep 1 m 54 → **31 m 05**; the `D = 2`
`p = 1, fixup = false` sweep 34 s → 9 m 27; `D = 3` two-level 15 s →
1 m 17) — while one-dimensional runs cost seconds either way, and locally
on twelve cores four threads is *faster* than one. Coverage, on by default
in `julia-runtest` and consumed by nothing, cost nothing measurable. So
the threaded job must run a small suite, and the sweeps must leave the
default one.

Changes:

- `test/runtests.jl` reads `TREEHYDRO_TEST_LONG` and
  `TREEHYDRO_REGENERATE`, prints the tier and the thread count, includes
  the short files, then the long ones under the flag.
- The physics moved to `test/long/` **verbatim** — `entropywave_tests.jl`
  whole, and from `sod_tests.jl`, `interface_tests.jl`,
  `refinement_tests.jl` and `driver_tests.jl` the sweeps, tables,
  calibration, `D = 3` runs and fifteen tracked evolutions. No claim
  weakened, no number changed; the short files keep the unit and
  structural halves and the helpers both share.
- `test/references.jl` (`reference_outputs`, `write_references`,
  `compare_references`), `test/regression_tests.jl`, and
  `test/references/*.toml` — 25 reduced configurations over five studies,
  compared at `rtol = 1e-12` with exact equality for integers, levels and
  each table's `_config`, and a missing or extra key a failure. Only
  `Float64` is stored. `TOML` added to `test/Project.toml`.
- `.github/workflows/CI.yml`: `coverage: false`, `timeout-minutes: 30`,
  the measurement in a comment, the matrix unchanged; new
  `.github/workflows/Long.yml`, `workflow_dispatch` plus a weekly
  `schedule`, Ubuntu, Julia `"1"`, **one thread**,
  `TREEHYDRO_TEST_LONG=1`, `timeout-minutes: 120`, `coverage: false`.

Measured: short tier **47 s** at one thread and **47 s** at four (36 s of
it the unit files and their compilation), long tier **1 m 30 s**, against
the 1 m 35 s the undivided suite took. Every `@info` number the long tier
prints is byte-identical to the undivided suite's.

Accept: `Pkg.test()` green and under a minute; the long tier green with
every number in "Measured results" unchanged; a regeneration followed by a
short and a long run that both compare clean; the short tier bit-identical
at one and four threads.

## Step 7c — Undo the split (no milestone; the suite itself)

`CODE.md`: "Testing", "File layout".

Why: step 7b's diagnosis was wrong. Both CI jobs of the step 6 push ran
under `julia-runtest`'s defaults — `--code-coverage=@<package path>` and
`--check-bounds=yes`, visible in each log's "Precompiling for
configuration" line — so the runner's threading was never the variable
between them; coverage was on in both. Julia 1.13 compiles a coverage hit
into an **atomic read-modify-write on one global counter per source line**
(`visitLine` in `src/codegen.cpp`), and `src/coverage.cpp` packs 32
neighbouring lines into one 256-byte block; a KernelAbstractions CPU
launch is one task per thread over chunks of the same kernel, so every
thread executing a kernel line contends on the same cache line. The cost
therefore scales with parallel kernel work, which is exactly why it fell
on the `D ≥ 2` runs and left the `D = 1` ones alone. Reproduced locally on
twelve cores — no shared vCPUs, no oversubscription — on the `D = 2`
entropy-wave sweep at `N = (8, 16, 32)` with `--check-bounds=yes`: **1.83 s
at one thread and 0.79 s at four without coverage, 10.35 s and 79.13 s
with it**, so 5.7× at one thread and 100× at four, and the sign of
threading inverted. Step 7b set `coverage: false` in the same push as the
split, so the fast CI that followed did not distinguish the two changes;
with coverage off the whole two-tier suite ran locally in 1 m 45 at four
threads and 1 m 58 at one. The split bought nothing and cost the thing the
suite is for — the physics no longer ran on every push — so it is undone.

Changes:

- `test/sod_tests.jl`, `test/interface_tests.jl`,
  `test/refinement_tests.jl` and `test/driver_tests.jl` restored to their
  pre-split contents verbatim, and `test/long/entropywave_tests.jl` moved
  back to `test/entropywave_tests.jl`. The one test 7b genuinely added —
  that `evolve!` refuses a working type the case was not built at — is
  kept, as its own testset.
- `test/long/`, `test/regression_tests.jl`, `test/references.jl`,
  `test/references/*.toml` and the `TOML` dependency in
  `test/Project.toml` removed. Every claim the regression file made —
  conservation with the fixup, the leak without it, the momentum equal to
  its boundary flux, no floor hits, `tracking == 1`, the calibration's
  shape — is made at full size by the restored files.
- `test/runtests.jl` back to a plain include list in the original order,
  with no environment flags.
- `.github/workflows/Long.yml` deleted. `CI.yml` keeps its matrix,
  `coverage: false` and `timeout-minutes: 30`; the two comments now give
  the real reason for each.
- `CODE.md`'s "Testing: two tiers" replaced by "Testing", carrying the two
  measured tables, the mechanism, the rule, the correction of 7b, and 7b's
  one surviving finding: across machines Base's `@simd` reductions move
  order-one totals by 1–4 ulp, so bit-identity holds across thread counts
  and not across microarchitectures.

Measured: the whole suite **11 149 tests in 1 m 50 (1 m 55 wall) at one
thread and 1 m 42 (1 m 47 wall) at four** — four threads faster than one,
which is what coverage had been hiding. Every `@info` line the suite
prints is byte-identical to the two-tier suite's, all 75 of them; only
their order differs, and only because the include order is the original
one.

Accept: `Pkg.test()` green at one thread and at four, with the same
numbers; every number in "Measured results" still produced by a test that
runs on every push; no mention of a tier, a reference file or
`TREEHYDRO_TEST_LONG` left outside this file's history and `CODE.md`'s
account of it.

## Step 8 — The atmosphere reset (H4a)

`CODE.md`: "Floors and the atmosphere" (all of it).

Changes: `src/floors.jl` — `reset_atmosphere!(u, integrator, p, t)` as a
pointwise kernel over `statearray(u, U)`, the `reset = :stage | :step |
:none` keyword wiring `SSPRK33`'s hooks, the injection accounting
(per-variable totals before and after, `block_mapreduce` over `(U, u)`),
the post-regrid reset in `evolve!`, the ghost-population floor count.

Accept: idempotent to roundoff; `con2prim` of the reset `U` reproduces
the floored `P`; the injection is *exactly* zero on Sod and the entropy
wave (bit-identical totals); on a synthetic state with vacuum cells the
reset yields the atmosphere and the injection equals the hand-computed
`ΔU`; a step of `SSPRK33` on a state that needs flooring comes out
floored under `:stage` and `:step` and not under `:none`.

## Step 9 — Sedov (H4b)

`CODE.md`: "Sedov blast wave", "Boundaries" (edges and corners).

Changes: `src/sedov.jl` — the case with Dirichlet ambient on every face,
the measured `E₀`, `shock_radius` (the outermost firing box of a density
threshold), the exponent fit over the late chunks, the peak density
jump, `uniform_sedov`; `src/sedov_reference.jl` — the similarity law and
its exponent.

Accept: in `D = 2` and `D = 3` (small and short in 3D): the exponent
within tolerance of `2/(D+2)` once `r_s ≫ r₀`; the peak jump recorded
against 6; floor counts by population, nonzero, reported; the drift net
of the injection at roundoff, and a leak without the fixup; block count
rising then falling behind the shock; the adaptive run against the
uniform fine reference on a reduced grid; a refined region touching a
corner (2D) and an edge (3D) of the Dirichlet box; the `p = 1` against
`p = 3` and `:stage` against `:step` tables. Record everything; the
1D planar blast is a smoke test.

## Step 10 — Kelvin–Helmholtz and HLLC (H5a)

`CODE.md`: "Kelvin–Helmholtz instability", "Riemann solver".

Changes: `src/kelvinhelmholtz.jl` — the McNally setup **checked against
the paper** (arXiv:1111.1764; fix `CODE.md` where the transcription
differs and say so), `mode_amplitude`, the maximum `y`-kinetic energy,
the case, the uniform reference; the HLLE against HLLC measurement.

Accept: `M(t)` grows exponentially then saturates, with a rate below
the incompressible bound; conservation through regrids; the refined
region grows from two strips; the adaptive run approaches the uniform
fine run as the cap rises; HLLE against HLLC recorded and the default
flux for the case decided in `CODE.md`. Record.

## Step 11 — Viewers and the figure job (H5b)

`CODE.md`: "File layout" (`bin/`), the Kelvin–Helmholtz figure.

Changes: `bin/Project.toml` (CairoMakie, SixelTerm, `[sources]` for
both packages), `bin/backend.jl` after TreeWave's, `bin/visualize1d.jl`
(Sod against the exact solution, per block coloured by level, `τ`, the
totals against time), `bin/visualize2d.jl` (`--case=kh`: the filmstrip
with block outlines, `M(t)` and the kinetic energy with the reference,
block count; `--case=sedov`: filmstrip and radial scatter); the CI
figure job with its artifact upload.

Accept: every figure written in CI; the viewers contain no time-stepping
loop of their own (they use `observer`).

## Step 12 — Precision (H6a)

`CODE.md`: "Precision".

Changes: `test/type_tests.jl`; whatever `Float32` or `Float32x2` shakes
out of steps 1–10 (a leak shows as a returned `Float64`; a missing `Base`
method as a `MethodError`).

Accept: the type table in `CODE.md` measured — Sod and Sedov at
`Float32` reaching the same mesh and floor counts, errors agreeing to
~1%; Sod and Sedov at `Float32x2`; Kelvin–Helmholtz at `Float32`
agreeing on the mesh and `M(t)` through the linear phase. Record.

## Step 13 — Threads (H6b)

`CODE.md`: "Multi-threading".

Changes: `test/thread_workload.jl` (a tracked Sod in `D = 1` and a short
Kelvin–Helmholtz, digests per chunk, self-contained and fast),
`test/threading_tests.jl` (a subprocess at the other count, compared
character for character).

Accept: bit-identical digests at 1 and 4 threads on both runs.

## Step 14 — Device and benchmark (H6c)

`CODE.md`: "Running on a device".

Changes: `test/device_tests.jl` behind `TREEHYDRO_TEST_BACKEND`, run on
`CPU()` by default; `bin/benchmark.jl` and `src/benchmark.jl` after
TreeWave's. Metal is available on this machine: measure the per-phase
table at `Float32` on it, against 8 host threads, at a size where a
device pays (TreeWave used 29.4M cells).

Accept: the suite on Metal in `Float32`; the same mesh and floor counts
as the host; the RHS speedup on unified memory measured against the
prediction in `CODE.md`. Record the table.

## Step 15 — Review pass

Read `CODE.md` against the code once more. Mark H0–H6 *(Done.)*; update
`README.md`'s status and `CLAUDE.md`'s "Current state" and "Commands";
move anything still marked **(predicted)** to measured or to "Possible
extensions"; delete this file.
