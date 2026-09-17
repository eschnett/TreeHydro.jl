# Working notes for Claude in TreeHydro.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

## What this package is

The second sample application for
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl), and the
*conservative* one. TreeAMR is the mesh and contains no physics;
[TreeWave](https://github.com/eschnett/TreeWave.jl) is the scalar wave
equation, a second-order finite-difference scheme that needs nothing
special at coarse-fine faces; TreeHydro is Newtonian ideal hydrodynamics
with a shock-capturing finite-volume scheme, which needs everything
TreeAMR's M8 added — the conservative operator family, per-field-set
ghost widths, ghost-free face-centered flux fields, and the interface
flux restriction.

Two rules follow from `CODE.md` and govern every change here:

- **Every numerical method must have a GRMHD counterpart.** The package
  rehearses a relativistic MHD code. The table under "What has a GRMHD
  counterpart" in `CODE.md` is the checklist; if a method is not on it
  and has no counterpart, do not add it, however much better it would be
  for Sod's problem. No exact Riemann *flux*, no Roe, no operator
  splitting, no dual energy.
- **No mesh machinery.** If a change is about trees, ghost cells, or
  interpolation, it belongs upstream in TreeAMR. Two things this package
  needed have already gone there (`map_blocks!(…; stored = true)` and
  the `AllVariables` callback form); a limited, positivity-preserving
  prolongation is the next candidate, and `CODE.md` says what
  measurement would justify asking for it.

## Current state

**Scaffolding (H0), the scheme on a uniform mesh (H1) and the coarse-fine
faces on a static mesh (H2) are done.**
`CODE.md` is complete and reviewed. What exists: `Project.toml` with the
`[sources]` pin to TreeAMR's GitHub
`main`; `src/TreeHydro.jl`, the module shell; `src/precision.jl` (`wrap`,
`ceilint`, `floorint`, `tofloat64`) and `src/device.jl` (`to_backend`,
`hostcopy`, `hostcopy!`), both ported from TreeWave; `src/floors.jl`
(`Floors`, `apply_floors`, `in_atmosphere`, `atmosphere_state`) and
`src/eos.jl` (`EquationOfState`, `IdealGas`, `pressure`,
`internal_energy`, `soundspeed`, the state accessors `statedims`,
`density`, `velocity`, `momentum`, `pressure_of`, `energy`, and
`prim2con` / `con2prim`) from step 1; `src/reconstruction.jl` (`slope`
for `:none`, `:minmod` and `:mc`, and `face_states`) and `src/riemann.jl`
(`physical_flux`, `signal_speed`, and `riemann_flux` for `:llf`, `:hlle`
and `:hllc`) from step 2 — all `isbits`, pointwise, kernel-callable,
non-allocating and inferred; from step 3 `src/evolution.jl`
(`HydroProblem`, the three kernels `con2prim_kernel!`, `flux_kernel!` and
`divergence_kernel!`, `hydro_rhs!`, `update_primitives!`,
`max_signal_speed`, `floor_hits`, `hydro_dt`, `conserved_totals`,
`conserved_scales`, `hydro_solve!`, `convergence_rate`) and
`src/entropywave.jl` (`EntropyWave`, `hydro_forest`,
`fill_entropywave_averages!`, `entropywave_reference`,
`entropywave_errors`); and from step 4 `src/exact_riemann.jl`
(`ExactRiemann`, `exact_riemann`, `sample`, `max_signal_speed(::ExactRiemann)`
— host `Float64`, Toro ch. 4, a *reference* and not a flux) and
`src/sod.jl` (`SodTube`, `sod_state`, `sod_initial`, `sod_conserved`,
`sod_boundary`, `sod_forest`, `sod_reference`, `assert_no_arrival`,
`sod_errors`); and from step 5 almost nothing — `sod_forest` gained
`refined = :middle | :left` and `forest_levels(forest)` was added to
`src/evolution.jl`, the step being a measurement rather than a
construction. Tests: `test/precision_tests.jl`,
`test/prerequisite_tests.jl`, `test/eos_tests.jl`,
`test/riemann_tests.jl`, `test/evolution_tests.jl`,
`test/entropywave_tests.jl`, `test/exact_riemann_tests.jl`,
`test/sod_tests.jl` and `test/interface_tests.jl`; CI and a `README.md`.
The milestones are H0–H6 in `CODE.md`; H1 covered steps 1–4 and H2 step 5,
and `PLAN.md`'s step 6 (the refinement criterion, H3a) is next.

The measured numbers are in `CODE.md`'s "Measured results": the
entropy wave is second order in L1 and L∞ with `:none` in `D = 1, 2`
(rates 2.02 and 2.03), second order in L1 and **1.35 in L∞** with `:mc`
(the limiter clipping the smooth extrema, which is why the study runs
with `:none`), and all `D + 2` conserved integrals hold to a few ulp of
their own scale on the uniform mesh with the fixup and — bit-identically
— without it. Sod converges against the exact Riemann solution at an L1
rate of **0.903** with `:minmod` and **0.945** with `:mc` over
`N = 16 … 128`; the exact solver reproduces Toro's Table 4.3 for tests 1,
2 and 3 to the last digit the table prints; the tube gives the same answer
along every axis **bit for bit**, ghosts included, and the `D = 2` planar
tube equals the `D = 1` run with `S_y` exactly zero; and where the
boundary is physical the drift *is* the boundary flux, the momentum total
moving by exactly `(p_L − p_R)·t_end·A` while mass and energy do not move.
No floor has fired in any run of any case yet.

Step 5 added the two-level numbers, which are the ones the package exists
for. On the static two-level mesh every one of the `D + 2` integrals holds
to **0.003–0.011 ulp of its own scale per step** in `D = 1, 2, 3`, and the
identical run with `fixup = false` leaks by **`1e8`–`1e9` times** that
bound. The interface-order rule holds for the system unamended: L∞ rates
**0.963 / 2.034 / 2.037** at `p = 1, 3, 5` in `D = 1` and **0.925 / 2.041 /
2.037** in `D = 2`, against unrefined controls of 2.024 and 2.029, with
every L1 rate the scheme's own — and the negative control on the *rate*
reproduces Burgers': `p = 1` without the fixup falls from 1.966 to **1.111**
in L1 (1.900 to **1.192** in `D = 2`). On the two-level Sod tube the mass
and energy drift is 0.15–2.8 times the uniform mesh's at the same coarse
spacing (it is the *boundary's* numerical flux, not the coarse-fine
face's), the momentum equals the boundary flux to `5e-11` relative, and at
`N = 32` in `D = 1` all three are at true roundoff; without the fixup the
three are `6e3`–`8e7` times worse. The Dirichlet hook fills a *fine*
block's outer ghosts exactly, ghost rows across the tube included.

`floors.jl` is included *before* `eos.jl`: `con2prim` takes a `Floors` and
says so in its signature, and a signature is evaluated where the method is
defined.

## Commands

The full suite (about 80 s after step 5 — 1 m 18 s at one thread and 1 m
02 s at four, most of it the entropy wave's convergence studies, of which
the interface-order sweep in `D = 2` alone is 19 s), and the same at four
threads — `Pkg.test` does not inherit `-t`, so it has to be passed
explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The clean-checkout check, which is what the `[sources]` pin exists for: a
tree with no `Manifest.toml` resolves TreeAMR from GitHub and passes.
This is what CI does, and a local run that passes proves nothing about it:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Not yet real, and listed so the section can be filled in rather than
rewritten: the viewer (`julia --project=bin bin/visualize2d.jl --case=kh`)
arrives in step 11, the thread-independence test — which spawns its own
subprocess at another count either way — in step 13, and device tests,
opt-in behind `TREEHYDRO_TEST_BACKEND` (`metal`, `cuda`) in an environment
of your own that has the device package, in step 14. Neither this package
nor TreeAMR depends on a device package.

## Things that will bite

Carried over from TreeAMR and TreeWave where they apply here, plus what is
specific to a hydro code. Each is in `CODE.md` with its reason.

- **TreeAMR will be pinned to the GitHub `main`, not the local checkout.**
  Once H0 adds the `[sources]` entry, `~/src/jl/TreeAMR` is *not* what the
  tests see; an unpushed TreeAMR change is invisible here. Say so rather
  than editing that checkout and assuming the tests see it. The
  `[sources]` entry is also why the Julia floor is 1.11.
- **Three ghost widths, three off-by-`G`s.** The state `U` has `G = 2`,
  the primitives `P` have `G = 2`, the fluxes `F_d` have `G = 0`. Face `i`
  of a block (in `1 … N+1`) lies between cells `i−1` and `i`; cell `i` is
  stored at `i + G_U` in `U` and `i + G_P` in `P`, the face at `i + G_F`
  in `F_d`. `coordinates(fs, b, idx)` takes **stored** indices. Getting
  this wrong produces plots that look almost right.
- **`map_blocks!` has three ranges.** Default: owned, kernel adds `G[d]`.
  `closed = true`: the `N+1` faces, kernel adds `G[d]`. `stored = true`:
  every stored point including ghosts, and the kernel's index *is* the
  stored index — nothing to add. `con2prim!` is the `stored = true`
  kernel; a kernel written for one form is wrong under another.
- **The RHS never mutates `u`.** The atmosphere reset lives in the
  integrator's `stage_limiter!` hook and in the driver after `regrid!`,
  never inside the RHS. The RHS-level floor on `P` and the stage reset on
  `U` are two mechanisms on purpose (owned cells versus prolongated
  ghosts and face states); do not merge them, and keep the floor counts
  by population — they are a measurement the design depends on.
- **Face-state floor hits are not counted, and that is deliberate.**
  `face_states` floors both reconstructed states and returns the states
  alone. Under `:minmod` or `:mc` the floors cannot fire on physical cell
  states at all — the face value lies between the two neighbouring cell
  values — and under `:none` they can. The counts the design rests on are
  the ones in *cells*, owned from the stage reset and ghost from
  `con2prim`; a third count over face states, which are not cells and are
  rebuilt at every stage, would blur the ghost count that decides the
  upstream prolongation question. Do not add one to make the accounting
  look symmetric.
- **Conservation is claimed net of measured injection.** Where no cell is
  floored the totals before and after a reset are bit-identical and the
  injection is *exactly* zero; a test that sees a nonzero injection on
  Sod, the entropy wave or Kelvin–Helmholtz has found a bug, not a
  tolerance to loosen.
- **At a physical boundary the drift is not roundoff, and the momentum has
  no scale** (measured in step 5). Two traps in one place. The mass and
  energy totals of a Sod run move by the *numerical* foot of the
  rarefaction and the shock reaching the Dirichlet faces — `2.1e-11` of the
  mass at `N = 16` in `D = 1`, falling by four orders of magnitude per
  halving of `h` and reaching roundoff only at `N = 32` (`N = 32` in
  `D = 2` as well). So a conservation test written at `8 eps · scale ·
  nsteps` will fail on a coarse tube for a reason that has nothing to do
  with the coarse-fine face; compare against the *uniform* mesh of the same
  coarse spacing instead, or run fine enough that the boundary is clean.
  And `conserved_scales` gives the momentum **exactly zero** on Sod,
  because the initial state is at rest: its yardstick is the closed-form
  boundary flux `(p_L − p_R)·t_end·A`, not a norm of the state.
- **The boundary hook goes to three places**: `fill_ghosts!`, `regrid!`
  (it fills ghosts before its transfer) and `adapt_to_initial_data!`.
  Forgetting the second is the bug that arrives one chunk late.
- **Dirichlet-from-initial-data reflects once a wave arrives.** The
  driver asserts `t_end · λ_max` against the distance to the nearest
  physical boundary before the run. If that assertion fires, shorten
  `t_end` or enlarge the box; do not remove it. `assert_no_arrival` in
  `src/sod.jl` is the shock tube's copy of it, and it is written on the
  *characteristic* speed rather than on the shock's own travel, which is
  conservative by about 25% on Sod's data and conservative in the right
  direction.
- **The initial data's `max_signal_speed` is not a bound for a resolving
  discontinuity** (measured in step 4). A Riemann problem's fastest signal
  is not present at `t = 0`: Sod's initial data carries nothing above
  `c_L = 1.18322` and the gas behind its shock carries 2.19157, a factor
  of **1.8522**. So a `λ_max` measured at the start of a chunk can be
  exceeded *within* that chunk, and the end-of-chunk recheck is a detector
  rather than a guard — it fires after the damage. The driver therefore
  needs a headroom factor as a case parameter beside the recheck, and
  Sod's 1.8522 is the number that sizes it. `sod_errors` sidesteps this by
  taking `λ` from the exact solution, which no driver can do. A second,
  smaller term: the *discrete* state sits above the exact supremum at a
  discontinuity by about half a percent at `N = 16`, falling with `h`.
- **The CFL recheck at chunk end throws on purpose.** `λ_max` is measured
  once per chunk; if a chunk's fastest signal grew past the step it
  used, the fix is a shorter `chunk` or a smaller `cfl`, not deleting
  the check.
- **The Löhner floor here is local plus global, on positive fields.**
  TreeWave replaced Löhner's local floor with a global amplitude because
  its fields cross zero. `ρ` and `p` do not, and Sedov's density spans
  orders of magnitude, so the local term is right *and* an absolute term
  is needed against the atmosphere. Never put the velocity in the
  indicator. The box threshold is `coarsen_tol`, not `refine_tol`
  (TreeWave's lesson: keying it on `refine_tol` silently disables the
  travelling margin).
- **Don't name a keyword `maxlevel`.** It shadows TreeAMR's exported
  `maxlevel(forest)` inside the function body. Use `maxlevel_cap`.
- **A decimal literal in a `T` expression is a leak.** `T(7//5)`, not
  `1.4`; `T(1//2)`, not `0.5`. At `Float64` the two are bit-identical,
  which is what lets measured numbers stay put when a driver goes
  generic.
- **A callback must capture no `Type` and no host array.** Initial data,
  boundary and flagging callbacks become kernel arguments; write
  `oftype(x[1], 2)` rather than closing over `T`. The EOS and the floors
  are `isbits` structs for this reason.
- **Never thread anything a TreeAMR callback can reach**, and never
  accumulate into shared state in a loop of your own: bit-identity across
  thread counts is the invariant, and `test/threading_tests.jl` is the
  only thing that will report a violation.
- **The Kelvin–Helmholtz formulas in `CODE.md` were transcribed from
  memory** of McNally, Lyra & Passy (2012), ApJS 201:18. Check every one
  against the paper before recording a number. Also: a `Float32`
  Kelvin–Helmholtz run diverges from the `Float64` one late in the run
  by design (the instability amplifies roundoff), so assert the growth
  rate and the early chunks, not the final state; and MultiFloats cannot
  run it at all (`sin`, `exp` are not implemented), only Sod and Sedov.
- **The exact Riemann solver is a reference, not a flux.** It and the
  Sedov similarity code are host `Float64`, converted once at the
  comparison. For the Sedov law, `E₀` is the *measured* energy deposited
  on the adapted mesh at `t = 0`, not the nominal value.
- **Measured numbers go into `CODE.md`**, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes.

## Conventions

Match TreeAMR's, since the three packages are read together:

- 4-space indent, wrap at about 80–90 columns.
- `return` on the last line of any non-trivial function.
- `ntuple(d -> f(d), Val(D))` rather than comprehensions in
  kernel-adjacent code; velocities and fluxes are `NTuple{D}`s.
- Unicode in mathematical contexts (`ρ`, `γ`, `λ`, `∂ₜ`, `Σ`).
- Keyword-heavy driver signatures with no default for anything the caller
  must think about; `T` as a leading positional argument defaulting to
  `Float64`, `backend` a keyword defaulting to `CPU()`.
- `ArgumentError`s say *why*, not just what.
- Docstrings are prose-first: what it is, then why, pointing at `CODE.md`.
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.
- Spec-first: when the implementation shows `CODE.md` was wrong or
  incomplete, amend it and say so in it — "(amended in H3)", "(measured
  in H4)" — rather than diverging silently.

## Repository facts

- **`origin` is `git@github.com:eschnett/TreeHydro.jl.git`, and `main` tracks
  it.** Work on a branch, and do not push, open a pull request, or merge
  to `main` without being asked. Each step lands on `main` only after
  review.
- `TODO.md` is Erik's personal to-do list. **Do not modify it.**
  `TODO.md~` is an editor backup, not a file of this package. Both are
  kept out of the tree by `.gitignore`.
- `.gitignore` exists, in TreeWave's image: `Manifest.toml` everywhere,
  `bin/output/`, `docs/build/`, editor leftovers, `TODO.md`. No
  `Manifest.toml` is tracked — that is what makes the clean-checkout
  check above mean something. `CODE.md`, `PLAN.md` and this file are
  committed.
- Sibling checkouts: `~/src/jl/TreeAMR` (the mesh; read its `CLAUDE.md`
  and `CODE.md` for the API and its sharp edges) and `~/src/jl/TreeWave`
  (the other application; copy the *patterns* of its `precision.jl`,
  `device.jl`, `bin/backend.jl`, viewers and thread workload — do not
  depend on it).
