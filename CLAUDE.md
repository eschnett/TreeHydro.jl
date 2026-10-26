# Working notes for Claude in TreeHydro.jl

**Contents**

- [Read first, and where things go](#read-first-and-where-things-go)
- [What this package is](#what-this-package-is)
- [Commands](#commands)
- [Things that will bite](#things-that-will-bite)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## Read first, and where things go

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics. The markdown
files and what goes in each:

- `README.md` — for users: what the package does, a short example, the
  status, and a pointer to `CODE.md`.
- `CLAUDE.md` — this file: what to read, the commands, short rules pointing
  at `CODE.md`, and the repository facts. Keep it short; no status
  narrative, file inventory, release history or detailed measurements (a
  rough duration that helps plan a run is fine).
- `CODE.md` — the current design and the reasons for it: the method, the
  code and its structure (its "File layout" has the file inventory, with
  the names to grep for), the interfaces and their contracts, the tests,
  current measurements that support a current decision ("Measured
  results"), known limitations, and open questions. Present tense, no
  dates or step numbers on decisions; mark only what is not settled, as
  **(open)** or **(proposed)**.
- `PLAN.md` — absent: nothing is planned. If work is planned, it goes
  there: steps or milestones, each with what it delivers and how one knows
  it is done.
- `HISTORY.md` — how the package got where it is: who decided what and
  when, alternatives tried or rejected and why, measurements of code that
  no longer exists or that later measurements replaced, upstream bugs since
  fixed, the milestone records with their steps, and the suite's timings.
  Ordered by topic, following `CODE.md`'s sections.
- `TODO.md` is Erik's personal to-do list. **Do not modify it**, do not
  move its content elsewhere, and do not move content into it.
  `TODO.md~` is an editor backup, not a file of this package.
- `showcase/README.md` (the zoom showcase's design, commands and numbers),
  `showcase/GOAL.md` (the request it answers) and `LICENSE.md` stand on
  their own.

The test between `CODE.md` and `HISTORY.md`: would someone changing the
code today need this? Then it belongs in `CODE.md`, rationale included;
where a reason lives in `HISTORY.md`, `CODE.md` links to it. Every fact
lives in one file; elsewhere, link to it.

**When the implementation shows `CODE.md` was wrong or incomplete**, fix
`CODE.md` to describe the current state, and record in `HISTORY.md` what
changed, when and why — rather than diverging silently, and without
amendment markers ("amended in step N", "(decided)") in `CODE.md`.

**Keep the tables of contents current**: `README.md`, `CLAUDE.md`,
`CODE.md` and `HISTORY.md` (and `PLAN.md`, when there is one) each have one
near the top, as links (`##` and `###` headings for `CODE.md`, `PLAN.md`
and `HISTORY.md`, `##` only for `README.md` and `CLAUDE.md`). A heading
added, renamed or removed changes it, and a renamed or moved heading needs
every reference to it updated — docstrings, comments, tests and the other
documents cite sections by heading.

## What this package is

The second sample application for
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl), and the
*conservative* one: Newtonian ideal hydrodynamics with a shock-capturing
finite-volume scheme, which needs everything TreeAMR's M8 added. The
package is complete; the status is at the top of `CODE.md`.

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
  prolongation is the next candidate, and `CODE.md` records the
  measurement that priced it (step 9) and why it was still not asked for.

## Commands

One suite, run whole, at every thread count; `CODE.md`'s "Testing" has
the discipline and the measurement behind it. Every claim in "Measured
results" comes from a test that runs here, so this is what to run before
recording a number. About 6 minutes at one thread and 5 at four on this
machine (the history is under "Suite timings" in `HISTORY.md`);
`test/checkpoint_tests.jl` alone, in a fresh process, is 51 s with its
compilation, 14 s of it the subprocess. **This machine is shared**: a
timing is worth comparing only against another taken back to back under
the same load. `Pkg.test` does not inherit `-t`, so the thread count has
to be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The last file, `test/threading_tests.jl`, spawns a subprocess at the other
thread count — four if the suite runs at one, one if it runs at more — and
compares twelve digest lines character for character; about twenty
seconds. **Do not set `JULIA_EXCLUSIVE=1` for a one-thread suite**: the
parent pins itself to one CPU and the subprocess inherits the mask
(TreeGeneralizedHarmonic met this on Symmetry).

And the checked run, which is a *different* claim rather than a slower
version of the same one: the three RHS kernels carry `@inbounds`, so their
indices are an assertion, and this is the only thing that falsifies it.
Because it overrides `@inbounds` package-wide it never runs the code the
package actually ships — so it does not replace a plain run, which is the
only one that can catch a wrong answer. Run both before recording a
number. It took **6 m 33** against 5 m 01 plain the last time it was
measured.

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
```

The clean-checkout check: a tree with no `Manifest.toml` resolves TreeAMR
from the General registry and passes. This is what CI does, and a local
run that passes proves nothing about it — a depot that already has
everything, and a `Manifest.toml` that pins what a fresh resolve would
have to choose, are exactly what it is checking around:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

There is no separate check under an older Julia: **1.13 is the floor**
of all the Tree* packages, and CI runs nothing else.

The MPI test by hand: the workload under `mpiexec -n 3` writes `n3.txt`,
`n2.txt`, `n1.txt` (a one-rank communicator) and `n0.txt` (serial) to the
directory it is given, and every line not starting with `~` or `#` must
agree across the four. It needs an environment with this package, TreeAMR
and MPI — the test environment, or a scratch one that `develop`s this
checkout; launch through `MPI.mpiexec()` with its environment set on the
command, since interpolating it into a larger command drops its library
paths (TreeAMR's finding). About 1 m 20:

```bash
julia --project=<env> -e 'using MPI; m = MPI.mpiexec(); run(setenv(`$m -n 3 $(Base.julia_cmd()) --threads=1 --project=<env> test/mpi_workload.jl /tmp/mpi`, m.env))'
```

The viewers, in their own environment so that CairoMakie never becomes a
dependency of the package. The first call instantiates it; the
`[sources]` entry for TreeHydro means no manual `Pkg.develop`:

```bash
julia --project=bin -e 'using Pkg; Pkg.instantiate()'
```

```bash
julia --project=bin bin/visualize1d.jl
```

```bash
julia --project=bin bin/visualize2d.jl --case=kh
```

`--case=sedov` and the default `--case=both` are the other two; every
script takes `--out=`, `--ops=`, `--type=f32|f64`, `--backend=`,
`--display` and `--no-display`. `visualize2d.jl` also takes **`--movie`**,
which writes a video beside the figure from the same run — every frame the
observer hands over rather than the filmstrip's four:

```bash
julia --project=bin bin/visualize2d.jl --case=kh --movie
```

with `--movie-frames=`, `--movie-fps=` and `--movie-format=mp4|gif` beside
it. The shear layer's 301 frames cost about **a minute all in**, most of it
the two evolutions rather than the encoding, which `--threads=auto` cannot
help with (`CODE.md`, "A movie").

`--speed-headroom=` and `--no-reference` are the two a deeper mesh needs:
the first because the CFL recheck has no margin at headroom 1 (see "Things
that will bite"), the second because the uniform fine reference quadruples
per level, supplies only the figure's dashed overlay curves, and is nothing
a movie needs.

`--cap=` and `--chunk=` open up the mesh, and **they move together**: halve
the chunk for each level added (`CODE.md`, "Raising the level cap"). A
non-default mesh writes its own filenames, so a `--cap=3` render cannot
overwrite what CI checks. PNGs land in `bin/output/`, which is gitignored,
and a terminal also gets them inline through SixelTerm — a pipe does not,
which is what `--no-display` makes explicit in CI. Roughly 26 s, 28 s and
53 s per figure, most of the first 20 s of each being `using CairoMakie`.

The benchmark, in the **package** environment and not `bin/`'s, one run
per thread count; `--scan=N:roots,…` runs several meshes in one process,
and the output is one tab-separated row per phase:

```bash
julia -t 4 --project=. bin/benchmark.jl --dim=3 --scan=16:8
```

`--case=sedov`, `--refined`, `--type=f32`, `--reps=`, `--steps=`,
`--driver` and `--backend=` are the rest. A device needs an environment of
your own with the device package in it (neither this package nor TreeAMR
depends on one), and so do the device tests, which are opt-in:

```bash
julia --project=/tmp/thgpu -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["Metal", "KernelAbstractions", "TreeAMR", "TreeIOHDF5", "MultiFloats"])'
```

(`TreeIOHDF5` for `test/checkpoint_tests.jl`; the device file alone, as
the Symmetry job runs it, does not need it.)

```bash
TREEHYDRO_TEST_BACKEND=metal julia --project=/tmp/thgpu test/runtests.jl
```

Without the variable `test/device_tests.jl` runs with the CPU standing in
for the device. The file alone, which is what a device change needs — about
1 m 30 on Metal, `Float32` only; on CUDA it runs `Float64` as well:

```bash
TREEHYDRO_TEST_BACKEND=metal julia --project=/tmp/thgpu -t 4 -e 'using Test, TreeAMR, TreeHydro; @testset "device" begin include("test/device_tests.jl") end'
```

On Symmetry, from a checkout of its own (rsync the tree to
a fresh directory; never into one whose jobs are running):

```bash
sbatch --partition=amdq --time=2:00:00 bin/symmetry_cpu.sh
```

```bash
sbatch bin/symmetry_gpu.sh
```

`TREEHYDRO_BENCH=0 sbatch bin/symmetry_gpu.sh` runs the device tests on the
H200 and stops before the scans, in about five minutes of the hour
(4 m 43 for job 564514, `Float32` and `Float64`).

Each builds a scratch environment under
`/mnt/beegfs/eschnetter/claude/treehydro-{cpu,gpu}` that `develop`s the
checkout (`TREEHYDRO_CPU_ENV`, `TREEHYDRO_GPU_ENV` override it) and writes
its TSVs to `/mnt/beegfs/eschnetter/claude/treehydro-bench-$SLURM_JOB_ID`.

The zoom showcase, in its own environment, simulation and rendering as two
separate steps; `showcase/README.md` has the knobs and the measured costs.
The smoke configuration is a quarter of a minute and the renderer a few
seconds more:

```bash
julia --project=showcase -e 'using Pkg; Pkg.instantiate()'
```

```bash
julia -t 4 --project=showcase showcase/simulate.jl --config=showcase/configs/smoke.toml --out=showcase/output/smoke
```

```bash
julia --project=showcase showcase/render.jl --frames=showcase/output/smoke
```

`--still=K` renders one frame as a PNG, and `--stop-after=K` /
`--restart=FILE` on the simulation are how a zoom target is chosen: run
the intro, look at its last frame, put the target in the configuration and
restart from the intro's checkpoint. The production run is one H200 job:

```bash
sbatch showcase/symmetry_showcase.sh
```

## Things that will bite

Each rule is explained in `CODE.md`, in the section named after it.

- **TreeAMR comes from the registry, not from the local checkout.**
  `~/src/jl/TreeAMR` is not what the tests see, and an **unreleased**
  TreeAMR change is invisible here: a change this package needs has to be
  tagged and registered before it can be used; say so rather than editing
  that checkout and assuming the tests see it ("File layout").
- **Three ghost widths, three off-by-`G`s**: `U` and `P` have `G = 2`, the
  fluxes `G = 0`, and `coordinates` takes stored indices ("Field sets").
- **`map_blocks!` has three ranges** — owned, `closed = true`,
  `stored = true` — and a kernel written for one is wrong under another
  ("The right-hand side").
- **The three RHS kernels carry `@inbounds`, and `@inbounds` does not
  reach into an `ntuple` closure**: write
  `ntuple(v -> @inbounds(prim[i..., v, b]), Val(D+2))`, not the annotation
  outside the closure, and settle it with `code_llvm`, not a stopwatch. Do
  not remove `check_bounds: 'yes'` from `CI.yml`, and do not add
  `@inbounds` anywhere else in `src/` without the same argument from
  `map_blocks!`'s contract and the ghost widths ("Bounds checking in the
  kernels").
- **The RHS never mutates `u`.** The reset lives in the integrator's two
  limiter hooks and after `regrid!`; the RHS-level floor on `P` and the
  reset on `U` are two mechanisms on purpose — do not merge them, and keep
  the floor counts by population ("Floors and the atmosphere").
- **The reset writes back only where a floor fired, and everything rests
  on that**; the bit-identity with `:none` in `test/reset_tests.jl` is
  what catches a reset that writes everywhere ("Floors and the
  atmosphere").
- **`:stage` is *both* of IMEXRungeKutta's hooks, and the two hooks are
  two functions**: `reset_stage!` as the stage limiter, `reset_atmosphere!`
  as the step limiter; the hooks are `init` keywords whose omission is
  silent ("Floors and the atmosphere").
- **`hydro_solve!` does not alias; `evolve!` does**, and must set
  `integ_prev = nothing` after any regrid that changed the mesh ("Time
  integration and the time step").
- **Keep per-face code free of data-dependent branches**, and benchmark on
  an *evolved* state ("Step 14").
- **A loop over `d` that indexes a tuple of arrays is not unrolled**;
  write explicit methods for `D = 1, 2, 3` ("Step 14").
- **The reset is a fixed point of the *state*, not of its own flag**: do
  not assert that the count goes to zero on a second pass ("Floors and the
  atmosphere").
- **Face-state floor hits are not counted, and that is deliberate**; do not
  add a third count to make the accounting look symmetric
  ("Reconstruction").
- **Conservation is claimed net of measured injection**: a test that sees
  a nonzero injection on Sod, the entropy wave or Kelvin–Helmholtz has
  found a bug, not a tolerance to loosen ("Floors and the atmosphere").
- **At a physical boundary the drift is not roundoff, and the momentum has
  no scale** — the trap of Sod's tube: compare against the uniform mesh of
  the same coarse spacing, or run fine enough that the boundary is clean;
  the momentum's yardstick is the closed-form boundary flux ("Sod shock
  tube").
- **The boundary hook goes to three places**: `fill_ghosts!`, `regrid!`
  and `adapt_to_initial_data!` ("Boundaries").
- **`speed_headroom = 1` throws on any discontinuous initial data**; every
  shock case needs a headroom above 1.8522 ("Time integration and the time
  step").
- **The regrid cadence is bounded by the level cap**: deepening the
  hierarchy by one level halves the admissible chunk ("Time integration
  and the time step", "Raising the level cap").
- **The tracking measure is taken before the regrid, and that is the
  point** ("Regridding: one driver, restart per chunk").
- **`evolve!` does not check that no wave reaches a Dirichlet boundary —
  the case does**, with `assert_no_arrival`; if it fires, shorten `t_end`
  or enlarge the box, do not remove it ("Boundaries").
- **The initial data's `max_signal_speed` is not a bound for a resolving
  discontinuity** ("Time integration and the time step").
- **The CFL recheck at chunk end throws on purpose**: the fix is a shorter
  `chunk` or a smaller `cfl`, not deleting the check ("Time integration and
  the time step").
- **The Löhner floor here is local plus global, on positive fields.** Never
  put the velocity in the indicator; the box threshold is `coarsen_tol`,
  not `refine_tol` ("The refinement criterion").
- **A shock fires forever, and it fires at 0.57 and not at 1**: a test that
  expects a shock case's depth to be an output has misread the criterion
  ("The refinement criterion").
- **A kink does *not* fire forever**; do not "fix" it by shrinking `ε`
  ("The refinement criterion").
- **A tracked mesh has no coarse-fine face worth measuring**: an interface
  claim goes on a **static** two-level mesh the feature crosses
  ("Sedov blast wave").
- **Sedov's chunk bound is tighter than Sod's**, and its first chunk
  outgrows its own speed ("Sedov blast wave").
- **`shock_radius` reads cells, not boxes**; turning it into a
  `firing_boxes` sweep would break the exponent ("Sedov blast wave").
- **On a blast it is the pressure floor that fires**: do not raise `ρ_atm`
  to make the atmosphere rule fire ("Floors and the atmosphere").
- **In `sedov_reference.jl`, never form `V − V₀`** ("Sedov blast wave").
- **The `E₀` the similarity law takes is the measured one** ("Sedov blast
  wave").
- **Every test file is `include`d into the same `Main`, so a top-level
  `const` name must be unique across files — and Julia will not tell you.**
  Name a file's shared runs with their case (`TRACKED_SEDOV_1D`); the
  source-text check in `runtests.jl` is the only guard, so keep it
  ("Testing").
- **A count taken as the ceiling of a float quotient gets one more at
  `Float32`**: any new count of intervals built from a float quotient wants
  `chunk_count`'s rule ("Step 12 — precision").
- **Don't name a keyword `maxlevel`.** It shadows TreeAMR's exported
  `maxlevel(forest)` inside the function body. Use `maxlevel_cap`.
- **A decimal literal in a `T` expression is a leak.** `T(7//5)`, not
  `1.4`; `T(1//2)`, not `0.5` ("Precision").
- **A callback must capture no `Type` and no host array**; write
  `oftype(x[1], 2)` rather than closing over `T` ("Running on a device",
  "Precision").
- **Never thread anything a TreeAMR callback can reach**, and never
  accumulate into shared state in a loop of your own; a new block-shaped
  launch goes through `TreeAMR.launch_by_owner!` and a new host loop over
  blocks through `TreeAMR.threaded_chunks` ("Multi-threading").
- **A number combined on the host from `block_mapreduce` or a host loop
  over `nblocks` is one rank's**: a new reduction goes through
  `mesh_mapreduce` or `rank_reduce`, a count of the mesh's blocks is
  `nleaves`, and a check that throws must throw on every rank ("Running
  distributed").
- **A rank may hold no blocks**, and every function must survive it
  ("Running distributed").
- **The reset's hit count is per rank while it accumulates**; do not make
  `local_floor_hits` global to tidy it ("Running distributed").
- **Anything decided from a clock or the file system is agreed** over the
  ranks, and only rank 0 deletes ("Running distributed").
- **An observer runs on every rank and sees that rank's blocks** ("Running
  distributed").
- **A `Float32` Kelvin–Helmholtz run diverges from the `Float64` one late
  in the run by design**: assert the growth rate and the early chunks, not
  the final state; MultiFloats cannot run it at all ("Kelvin–Helmholtz
  instability", "Step 10").
- **A feature that *grows* gets nothing from a travelling margin**: a
  Kelvin–Helmholtz-like case has to be checked for what fraction of the box
  it refines, not merely for whether the buffer was accepted
  ("Kelvin–Helmholtz instability").
- **HLLE is not a neutral baseline on a contact-dominated flow**:
  `kh_run`'s `riemann` defaults to `:hllc`; do not tidy it away, and do not
  change the package-wide default, which is HLLE ("Riemann solver").
- **A tracked run measured against a uniform fine run *with the same flux*
  measures the mesh, not the flux**; deciding a flux needs a resolution
  sweep, and any later "which method is better" comparison on an adaptive
  mesh has the same trap in it ("Riemann solver").
- **The exact Riemann solver is a reference, not a flux** ("Riemann
  solver").
- **Code coverage under threads is a 100× slowdown**: never put coverage
  on the threaded entry ("Testing").
- **Base's reductions are not bit-identical across machines**: a test that
  stores a number needs a relative tolerance of a few hundred ulp *and* an
  absolute floor, and a drift is asserted against its bound ("Testing").
- **At `speed_headroom = 1` the CFL recheck is protected by nothing but
  integer-step quantization**: a deeper cap needs a headroom above 1, and
  shortening the chunk is not the remedy ("Kelvin–Helmholtz instability").
- **And the two fixes fight each other: raising the headroom spends the
  buffer margin** ("Raising the level cap").
- **CI's artifact glob is extension-specific**: a new output *kind* in
  `bin/` means checking that glob and the `test -s` list, neither of which
  fails loudly ("Raising the level cap").
- **In `bin/`, two exported names collide with Makie**: write
  `CairoMakie.scatter!` in full, and never write `density` in `bin/`
  ("Step 11").
- **A viewer snapshot must *materialize* everything it keeps** ("Step 11").
- **`P` has `D + 4` variables and only `D + 2` of them are primitives**: a
  loop over `1:P.nvars` calling them primitives plots two diagnostics as
  physics ("Field sets").
- **`evolve!` records no time vector, and its histories are off by one
  against the observer** ("Regridding: one driver, restart per chunk").
- **A checkpoint is written before the regrid, and a restart regrids
  first, through the same function**, `regrid_chunk!`: do not inline it
  back into the loop, or give the restart a regrid of its own, and do not
  add anything to the loop's regrid outside it ("Checkpoint and restart").
- **The checkpoints are TreeIOHDF5's, a hard dependency**: do not add HDF5
  to `[deps]` ("Checkpoint and restart").
- **Every parameter that decides a number goes in the recipe, and every
  accumulator in the run state**; a new returned field goes in
  `CKPT_FIELDS` too, and moving a parameter between the recipe and the
  criterion is a design change that goes in `CODE.md` with its reason
  ("Checkpoint and restart").
- **A wall-time stop returns the checkpointed state, not an answer**: a
  caller reading `r.l1` must look at `r.finished` first ("Checkpoint and
  restart").
- **Rotation deletes matching files from earlier jobs**: two runs must
  never share a prefix ("Checkpoint and restart").
- **The observer's state is not checkpointed**: do not add checkpoint
  keywords to `kh_run` without restoring its `ts`, `Ms`, `Ks` and `nbs`
  from somewhere ("Checkpoint and restart").
- **Every field set over a reflecting forest needs its parity, and the
  parity is physics** ("Boundaries").
- **The half box is the full box to roundoff, not bit for bit, and the
  full box is why**; do not "fix" that into an equality ("Reflecting walls,
  measured", "Kelvin–Helmholtz instability").
- **The showcase's loop is a second loop, on purpose, and it must follow
  `evolve!`'s order**: a change to `evolve!`'s order is a change to
  `showcase/simulate.jl` too ("The zoom showcase").
- **The showcase window is sized for the zoom-out, not the zoom-in** ("The
  zoom showcase").
- **`floors.jl` is included *before* `eos.jl`**, and `checkpoint.jl`
  *after* `driver.jl` ("File layout").
- **`bin/backend.jl` may use only what both `bin/` and the package
  environment have**, which today is `KernelAbstractions` ("File layout").

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
  failure mode it guards. Convergence rates, conservation drifts and mesh
  statistics are asserted as numbers with tolerances, and **an existing
  assertion is never loosened to get green** — a failure is reported
  instead.
- Generic in `T` and in the backend from the first line of any new driver:
  TreeWave records that retrofitting either was a rewrite, and
  `test/type_tests.jl` and `test/device_tests.jl` are what would notice.
- **Measured numbers go into `CODE.md`**, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes. The test that produces one runs
  in the suite, on every push, like every other. A number a later
  measurement replaces moves to `HISTORY.md`.

## Repository facts

- **`origin` is `git@github.com:eschnett/TreeHydro.jl.git`, and `main` tracks
  it.** Work on a branch, and do not push, open a pull request, or merge
  to `main` without being asked. Every change lands on `main` only after
  review.
- `TODO.md` and `TODO.md~` are kept out of the tree by `.gitignore`, which
  exists in TreeWave's image: `Manifest.toml` everywhere, `bin/output/`,
  `docs/build/`, editor leftovers, `TODO.md`. No `Manifest.toml` is
  tracked — that is what makes the clean-checkout check above mean
  something. `CODE.md`, `HISTORY.md` and this file are committed.
- No generated file is tracked. Everything in the tree is written by hand
  and reviewed as such. `bin/output/` is gitignored and the viewers write
  PNGs there.
- **`bin/` has its own `Manifest.toml`**, so the viewers can resolve a
  different TreeAMR from the tests; grep for `TreeAMR =` across both
  `Project.toml`s rather than editing from memory ("File layout").
- One workflow with two jobs: `.github/workflows/CI.yml`. `test` runs the
  whole suite on every push that touches something other than Markdown, on
  three cells at Julia 1.13 — Linux, macOS (the arm64 reduction claim and
  the coverage) and Linux at 4 threads; coverage is collected on one cell,
  on `main` or a manual dispatch only. `viewer` instantiates `bin/` and
  renders all four figures, ungated, on pull requests too. Treat every CI
  timing as ±70%. `CI.yml`'s comments state the arrangement, not how it was
  arrived at; the reasons are in `CODE.md`'s "Testing" and the history in
  `HISTORY.md`.
- Sibling checkouts: `~/src/jl/TreeAMR` (the mesh; read its `CLAUDE.md`
  and `CODE.md` for the API and its sharp edges) and `~/src/jl/TreeWave`
  (the other application; copy the *patterns* of its `precision.jl`,
  `device.jl`, `bin/backend.jl`, viewers and thread workload — do not
  depend on it).
