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

**Scaffolding (H0) done; H1 most of the way — the scheme runs, and the
first numbers are measured.** `CODE.md` is complete and reviewed. What
exists: `Project.toml` with the `[sources]` pin to TreeAMR's GitHub
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
non-allocating and inferred; and from step 3 `src/evolution.jl`
(`HydroProblem`, the three kernels `con2prim_kernel!`, `flux_kernel!` and
`divergence_kernel!`, `hydro_rhs!`, `update_primitives!`,
`max_signal_speed`, `floor_hits`, `hydro_dt`, `conserved_totals`,
`conserved_scales`, `hydro_solve!`, `convergence_rate`) and
`src/entropywave.jl` (`EntropyWave`, `hydro_forest`,
`fill_entropywave_averages!`, `entropywave_reference`,
`entropywave_errors`). Tests: `test/precision_tests.jl`,
`test/prerequisite_tests.jl`, `test/eos_tests.jl`,
`test/riemann_tests.jl`, `test/evolution_tests.jl` and
`test/entropywave_tests.jl`; CI and a `README.md`. The milestones are
H0–H6 in `CODE.md`; H1 (the scheme on a uniform mesh) is in progress, and
`PLAN.md` breaks it into steps 1–4, of which step 4 (Sod and the exact
Riemann solver) is next.

The first measured numbers are in `CODE.md`'s "Measured results": the
entropy wave is second order in L1 and L∞ with `:none` in `D = 1, 2`
(rates 2.02 and 2.03), second order in L1 and **1.35 in L∞** with `:mc`
(the limiter clipping the smooth extrema, which is why the study runs
with `:none`), and all `D + 2` conserved integrals hold to a few ulp of
their own scale on the uniform mesh with the fixup and — bit-identically
— without it.

`floors.jl` is included *before* `eos.jl`: `con2prim` takes a `Floors` and
says so in its signature, and a signature is evaluated where the method is
defined.

## Commands

The full suite (about 50 s after step 3, most of it the entropy wave's
convergence studies), and the same at four threads — `Pkg.test` does not
inherit `-t`, so it has to be passed explicitly:

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
- **The boundary hook goes to three places**: `fill_ghosts!`, `regrid!`
  (it fills ghosts before its transfer) and `adapt_to_initial_data!`.
  Forgetting the second is the bug that arrives one chunk late.
- **Dirichlet-from-initial-data reflects once a wave arrives.** The
  driver asserts `t_end · λ_max` against the distance to the nearest
  physical boundary before the run. If that assertion fires, shorten
  `t_end` or enlarge the box; do not remove it.
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
