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
  prolongation is the next candidate, and `CODE.md` records the
  measurement that priced it (step 9) and why it was still not asked for.

## Current state

**The package is complete: milestones H0–H6 are done, and step 15 — the
review pass that read `CODE.md` against the code once more — closed the
step plan, whose file (`PLAN.md`) it deleted.** Nothing is next. H7,
higher-order reconstruction, is optional and not planned. `CODE.md` is the
record: the design with every amendment marked where it was made, every
measured number in "Measured results" beside the prediction it confirms or
corrects, and under "Possible extensions" what was measured and deliberately
not built and what was never measured at all.

**Added since, on 2026-09-29: checkpoint and restart**, on TreeAMR 0.1.4's
M9a — `evolve!` writes checkpoints at chunk boundaries after the regrid and
restarts from them, bit-identically; see "Checkpoint and restart" in
`CODE.md`, and the entries below for the files.

**Added since, also on 2026-09-29: reflecting walls and the zoom
showcase.** TreeAMR's reflecting faces (M10, first-class, on every backend —
not a hook) are wired through: `HydroCase` carries `reflecting`, every
field set a run builds carries a parity, and `HydroCase(w::KelvinHelmholtz;
half = true)` runs the shear layer's lower half between two mirrors under
the `:mirrored` seed. `showcase/` uses it for 1080p movies that zoom
through the refinement levels — to ×256 and level 12 at best, with the
strain-matched clock; see "Reflecting faces" under "Boundaries"
and "The zoom showcase" in `CODE.md`, and `showcase/README.md`.

How the milestones map onto the steps, for reading `CODE.md`'s history: H0
was step 0; H1 steps 1–4; H2 step 5; H3 steps 6 and 7; H4 steps 8 and 9; H5
steps 10 and 11; and H6 steps 13, 14 and 12, in that order, taken together
with the move to **TreeAMR 0.1.3** (owner-based threading, from the
registry) and to **IMEXRungeKutta**, which replaced OrdinaryDiffEq for time
integration. Step 7b split the suite into two tiers and step 7c undid the
split, having found that what made CI slow was code coverage under threads
and not the runner.

What exists, file by file (the names are the ones to grep for; `CODE.md`'s
"File layout" has the one-line table):

- `Project.toml`: `TreeAMR = "0.1.4"` from the General registry (0.1.4
  since 2026-09-29, for its checkpoint and restart; 0.1.3 before, for the
  owner-based threading),
  `KernelAbstractions`, and `IMEXRungeKutta = "1.3"` (the first release
  that runs `Float32x2`) through the one `[sources]` entry left, since it
  is unregistered; `julia = "1.11"`, the floor of all the Tree* packages
  since 2026-09-25. **No HDF5**: TreeAMR's checkpoint functions live in its
  HDF5 extension and the caller loads it. `test/Project.toml` adds
  `MultiFloats`, `Random` and, since 2026-09-29, `HDF5 = "0.17"`, which is
  what loads that extension in the suite.
- `src/TreeHydro.jl`, the module shell and its exports.
- `src/precision.jl` (`wrap`, `ceilint`, `floorint`, `roundint`,
  `tofloat64`) and `src/device.jl` (`to_backend`, `hostcopy`, `hostcopy!`),
  both ported from TreeWave.
- `src/floors.jl`: `Floors`, `apply_floors`, `in_atmosphere`,
  `atmosphere_state`, and the reset — `ResetAccounting`,
  `reset_atmosphere!(u, integrator, p, t)` (the step limiter, and the call
  after every regrid) and `reset_stage!` (the stage limiter), a
  `map_blocks!` launch over the owned cells of `statearray(u, U)` writing
  back *only* where a floor fired.
- `src/eos.jl`: `EquationOfState`, `IdealGas`, `pressure`,
  `internal_energy`, `soundspeed`, the state accessors `statedims`,
  `density`, `velocity`, `momentum`, `pressure_of`, `energy`, and
  `prim2con` / `con2prim`.
- `src/reconstruction.jl` (`slope` for `:none`, `:minmod` and `:mc`, and
  `face_states`) and `src/riemann.jl` (`physical_flux`, `signal_speed`, and
  `riemann_flux` for `:llf`, `:hlle` and `:hllc`) — all `isbits`,
  pointwise, kernel-callable, non-allocating and inferred.
- `src/evolution.jl`: `HydroProblem`, the three kernels
  `con2prim_kernel!`, `flux_kernel!` and `divergence_kernel!` (the three
  that carry `@inbounds`), `hydro_rhs!`, `update_primitives!`,
  `max_signal_speed`, `floor_hits`, `ghost_floor_hits` (the package's one
  launch of its own, through `TreeAMR.launch_by_owner!`), `hydro_dt`,
  `conserved_totals` (field-set and state-vector forms),
  `conserved_scales`, `check_reset`, `forest_levels`,
  `convergence_rate`, and the parity tables `reflects`, `state_parity` and
  `flux_parity` (added 2026-09-29) that every field set over a reflecting
  forest is built with.
- `src/stepping.jl`: `state_partition`, `hydro_integrator` (IMEXRungeKutta's
  `SSPRK33`, stage arithmetic by block owner, the reset in both hooks) and
  `hydro_solve!`.
- `src/refinement.jl`: `lohner`, `cell_tau`, `indicator_scales`,
  `hydro_flags` and `refinement_buffer`; `hydro_flags` and
  `max_signal_speed` each have a `FieldSet` core and a `HydroProblem`
  forwarder.
- `src/driver.jl`: `HydroCase`, `evolve!` — the one loop, with `reset`
  (default `:stage`), `accounting`, the post-regrid reset, the
  `observer` hook and the checkpoint keywords (`checkpoint_path_prefix`,
  `checkpoint_every_chunks`, `checkpoint_interval_seconds`,
  `max_walltime_seconds`, `num_checkpoints_keep`,
  `checkpoint_hdf5_filters`, `checkpoint_sync_to_disk`, `restart_file`;
  it returns `finished`, `t`, `chunk`, `checkpoints_written` and
  `restart_file` beside the rest) — `uniform_run`, `check_cfl`,
  `chunk_count`, `tracked_share`, `reduce_to_grid` and `l1_difference`.
- `src/checkpoint.jl` (added 2026-09-29): `CHECKPOINT_APPLICATION` and
  `CHECKPOINT_VERSION`, `checkpointing_available`, `checkpoint_filename`,
  `checkpoint_files`, `latest_checkpoint` (the one export),
  `rotate_checkpoints!`, `plain_reals` / `from_plain_reals` (reals as
  limbs where they are not native), `run_recipe` / `check_recipe`,
  `run_state`, `save_run` / `load_run`, and `check_checkpoint_keywords`.
  Included *after* `driver.jl`, because `run_recipe`'s signature names a
  `HydroCase`; `evolve!` reaches it only at run time.
- `src/entropywave.jl` (`EntropyWave`, `hydro_forest`,
  `fill_entropywave_averages!`, `entropywave_reference`,
  `entropywave_errors`, `entropywave_primitive`, `HydroCase(::EntropyWave)`).
- `src/exact_riemann.jl` (`ExactRiemann`, `exact_riemann`, `sample`,
  `max_signal_speed(::ExactRiemann)` — host `Float64`, Toro ch. 4, a
  *reference* and not a flux) and `src/sod.jl` (`SodTube`, `sod_state`,
  `sod_initial`, `sod_conserved`, `sod_boundary`, `sod_forest` with
  `refined = :middle | :left`, `sod_reference`, `assert_no_arrival`,
  `sod_errors`, `HydroCase(::SodTube)`).
- `src/sedov_reference.jl` (`SedovSimilarity`, `sedov_alpha`,
  `sedov_exponent`, `sedov_radius`, `sedov_profile`, `exponent_fit` and an
  `adaptive_simpson` of its own — host `Float64`, the similarity law
  *derived* from the similarity equations rather than transcribed, a
  *reference* and not a method) and `src/sedov.jl` (`SedovBlast`,
  `sedov_state`, `ambient_state`, `sedov_initial`, `sedov_conserved`,
  `sedov_boundary`, `HydroCase(::SedovBlast)`, `sedov_forest` with
  `refined = :center | :corner | :edge`, `sedov_similarity`, `measured_E₀`,
  `shock_radius`, `peak_compression`, `assert_no_arrival(::SedovBlast, …)`
  and `sedov_static`).
- `src/kelvinhelmholtz.jl` (`KelvinHelmholtz` — `D = 2` only, and it
  refuses any other — `kh_state`, `kh_initial`, `kh_conserved`,
  `HydroCase(::KelvinHelmholtz)`, `mode_amplitude` and
  `max_y_kinetic_energy` (McNally's two diagnostics, host loops in block
  order read once per chunk through the observer), `growth_rate`, and the
  two measurement drivers `kh_run` and `kh_uniform`, which install that
  observer and are the only place either diagnostic can be taken; `kh_run`
  passes an `observer` of the caller's through after its own). Since
  2026-09-29 also `seed = :mcnally | :mirrored` and `HydroCase(w; half =
  true)`, the reflecting half box, which refuses McNally's seed.
- `src/benchmark.jl` (`benchmark_phases`, `benchmark_driver`).
- Tests, **one suite run whole**, included by `test/runtests.jl` in the
  dependency order: `precision_tests.jl`, `prerequisite_tests.jl`,
  `eos_tests.jl`, `riemann_tests.jl`, `evolution_tests.jl`,
  `reset_tests.jl`, `stepping_tests.jl`, `entropywave_tests.jl`,
  `exact_riemann_tests.jl`, `sod_tests.jl`, `interface_tests.jl`,
  `refinement_tests.jl`, `driver_tests.jl`, `sedov_tests.jl`,
  `kelvinhelmholtz_tests.jl`, `reflecting_tests.jl` (the parity tables,
  and the half box against the full box — to roundoff, not bit for bit,
  and why), `type_tests.jl` (every case at `Float64`,
  `Float32` and `Float32x2`), `checkpoint_tests.jl` (restart chains
  against the uninterrupted run, the rotation, the refusals; the first file
  to load HDF5, and it tests the refusal without it before it does, and
  reruns the standalone `test/restart_workload.jl` in a subprocess at the
  other thread count), `device_tests.jl` (every driver against the
  host; the CPU stands in unless `TREEHYDRO_TEST_BACKEND` names a device)
  and `threading_tests.jl`, which reruns the standalone
  `test/thread_workload.jl` in a subprocess at the other thread count.
  Before any of them, a testset refuses two files defining the same
  top-level `const` (see "Things that will bite").
- `bin/`: `Project.toml` (CairoMakie, SixelTerm and KernelAbstractions in
  an environment of their own, with a `[sources]` entry for TreeHydro),
  `backend.jl` (`resolvebackend`, `withbackend`, `checkprecision`, after
  TreeWave's; also included by the benchmark), `visualize1d.jl` (the
  tracked tube against the exact solution, per block coloured by level,
  with `τ` and the conserved totals against time), `visualize2d.jl`
  (`--case=kh|sedov|both`: the filmstrip with block outlines, and per case
  the two McNally diagnostics against the uniform fine run or the radial
  scatter against the similarity profile; `--movie`; `--cap=` and
  `--chunk=`), `benchmark.jl`, and the two Symmetry jobs
  `symmetry_cpu.sh` and `symmetry_gpu.sh`.
- `showcase/` (added 2026-09-29): the Kelvin–Helmholtz zoom movie, in an
  environment of its own like `bin/`'s (`Project.toml` with CairoMakie and
  HDF5 and a `[sources]` entry for TreeHydro). `kh_zoom.jl` (the
  configuration, `Geometry`, `Schedule`, `camera_centre`, `fold`, the
  interval folds and `block_cap` — pure, shared by both scripts),
  `simulate.jl` (the zoom-window loop, `evolve!`'s order from TreeHydro's
  pieces, one HDF5 frame file per movie frame, a checkpoint per new level),
  `render.jl` (the movie from the frame files, CairoMakie, no mesh),
  `configs/` (`smoke`, `pilot`, `production`), `symmetry_showcase.sh` and
  `README.md`. `showcase/output/` is gitignored.
- One workflow with two jobs, `CI.yml`'s `test` and `viewer`, and a
  `README.md`.

The headline numbers, each with the section of `CODE.md`'s "Measured
results" that has the rest: the entropy wave is second order in L1 and L∞
with `:none` (2.02 and 2.03), and 1.35 in L∞ with `:mc` (step 3); Sod
converges at an L1 rate of 0.903 against Toro's exact solution, bit for bit
along every axis (step 4); on a static two-level mesh every one of the
`D + 2` integrals holds to 0.003–0.011 ulp of its scale per step and the
run without the interface fixup leaks `1e8`–`1e9` times more, with the
interface-order rule holding unamended (step 5); the criterion's thresholds
are `refine_tol = 0.08` and `coarsen_tol = 0.02`, calibrated on the
McNally ramp (step 6); the tracked tube matches the uniform fine run's
error to a ratio of 1.0004 and 1.0000 at fewer cells (step 7); where
nothing floors, the reset changes nothing, bit for bit (step 8); Sedov's
exponents are 0.64146 / 0.50443 / 0.43766 against `2/3, 1/2, 2/5`, and
what fires is the pressure floor at a static coarse-fine face, never the
atmosphere rule (step 9); the shear layer grows at 2.58036, HLLC at half
the linear resolution ahead of HLLE at full (step 10); every run is
bit-identical at one and at four threads (step 13); Metal reproduces the
host `Float32` run bit for bit and an H200 runs a step at about 11× a
64-core node (step 14); and `Float32` and `Float32x2` rebuild the
`Float64` meshes and floor counts exactly (step 12); and a chain of
restarts is the uninterrupted run bit for bit, at another thread count too,
with a durable checkpoint of a 43.7 MB `D = 3` state costing 17 ms
("Checkpoint and restart, measured", 2026-09-29).

`floors.jl` is included *before* `eos.jl`: `con2prim` takes a `Floors` and
says so in its signature, and a signature is evaluated where the method is
defined.

## Commands

One suite, run whole, at every thread count; `CODE.md`'s "Testing" has
the discipline and the measurement behind it. Every claim in "Measured
results" comes from a test that runs here, so this is what to run before
recording a number. The last recorded timings, after checkpoint and
restart was added on 2026-09-29, on this machine: **12030 tests in 4 m 28
at one thread and 12064 in 3 m 47 at four** (after reflecting walls, the same day and under load: 12396 in 4 m 46 and 12430 in 4 m 10, 6 m 36 checked, 7 m 47 at 1.11; after step 12: 11856 in
4 m 26 and 11890 in 3 m 44) — the four-thread count is higher because the
ownership check has one assertion per block per thread, and about 50 s of
either is `test/type_tests.jl`, mostly compiling the `Float32x2` paths.
`test/checkpoint_tests.jl` alone, in a fresh process, is 51 s with its
compilation, 14 s of it the subprocess; inside the suite its cost is not
separable from the scatter. **This machine
is shared**: runs taken while something else was on it have come back a
fifth slower, so a timing is worth comparing only against another taken
back to back under the same load (the step-by-step history is in `CODE.md`,
and the `@inbounds` pass alone moved the suite by 15%, which is why the
older figures there are an ordering and not a target).
`Pkg.test` does not inherit `-t`, so the thread
count has to be passed explicitly:

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
number. After step 12 the checked run took **5 m 56** with the command as
written, against 4 m 26 plain; on 2026-09-29, **6 m 12** against 4 m 28.

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

The same check **under the floor version**, before a change is merged. CI
runs Julia 1.11 — the floor of all the Tree* packages since 2026-09-25 —
as well as the current release, and 1.11 is stricter in at least one way
that matters
here: it refuses to redefine a `const`, which 1.12 and later allow, so a
suite that is green at 1.13 can be red at the floor (see "Things that will
bite"). With `juliaup`:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia +1.11 --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

After step 12 the clean tree took **7 m 32** at 1.11 with the command as
written, against the release's 4 m 26 (on 2026-09-29, **7 m 27** against
4 m 28, with TreeAMR 0.1.4 and HDF5 0.17.4 resolved from the registry) — so budget for nearly twice the run
you have just done. 1.11 is also the *cheap* version under coverage, which
is the opposite way round and is why the instrumented cell is the floor
cell; see "Things that will bite".

The viewers, in their own environment so that CairoMakie never becomes a
dependency of the package. The first call instantiates it; the
`[sources]` entry for TreeHydro means no manual `Pkg.develop`, and it is
also why `bin/` needs **Julia 1.11**, which is the package's floor as
well:

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
the two evolutions rather than the encoding — the movie draws one heatmap
over a uniform fine grid and pushes into `Observable`s, which is twelve
times cheaper a frame than one heatmap per block rebuilt each frame, and
what `--threads=auto` cannot help with. See "A movie" in `CODE.md`.

`--speed-headroom=` and `--no-reference` are the two a deeper mesh needs:
the first because the CFL recheck has no margin at headroom 1 (see "Things
that will bite"), the second because the uniform fine reference quadruples
per level, supplies only the figure's dashed overlay curves, and is nothing
a movie needs.

`--cap=` and `--chunk=` open up the mesh, and **they move together**: the
buffer's margin is `speed_headroom · λ · chunk` at the cap's spacing, so halve
the chunk for each level added or the margin widens and more of the box is
refined. `--cap=5` at the default chunk throws out of `refinement_buffer`
naming the constraint. A non-default mesh writes its own filenames, so a
`--cap=3` render cannot overwrite what CI checks. PNGs land in `bin/output/`,
which is gitignored, and a terminal also gets them inline through SixelTerm —
a pipe does not, which is what `--no-display` makes explicit in CI. Roughly
26 s, 28 s and 53 s per figure, most of the first 20 s of each being
`using CairoMakie`. The `viewer` job in `CI.yml` runs all four renders on
every push and uploads them, because `bin/` is outside `src/` and `test/` and
nothing else would notice it breaking.

`--backend=metal --type=f32` renders too, and reproduces the host
`Float32` run exactly — measured in step 11 on this machine.

The benchmark (step 14), in the **package** environment and not `bin/`'s,
one run per thread count; `--scan=N:roots,…` runs several meshes in one
process, and the output is one tab-separated row per phase:

```bash
julia -t 4 --project=. bin/benchmark.jl --dim=3 --scan=16:8
```

`--case=sedov`, `--refined`, `--type=f32`, `--reps=`, `--steps=`,
`--driver` and `--backend=` are the rest. A device needs an environment of
your own with the device package in it (neither this package nor TreeAMR
depends on one), and so do the device tests, which are opt-in:

```bash
julia --project=/tmp/thgpu -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["Metal", "KernelAbstractions", "TreeAMR", "MultiFloats", "HDF5"])'
```

(`HDF5` since 2026-09-29, for `test/checkpoint_tests.jl`; the device file
alone, as the Symmetry job runs it, does not need it.)

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

The zoom showcase (added 2026-09-29), in its own environment, simulation
and rendering as two separate steps; `showcase/README.md` has the knobs and
the measured costs. The smoke configuration is a quarter of a minute and
the renderer a few seconds more:

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

Carried over from TreeAMR and TreeWave where they apply here, plus what is
specific to a hydro code. Each is in `CODE.md` with its reason.

- **TreeAMR comes from the registry, not from the local checkout**
  (amended when TreeAMR 0.1.1 was released, again at 0.1.3, and at 0.1.4).
  `Project.toml` has no `[sources]` entry for it: the compat bound is
  `TreeAMR = "0.1.4"` and a clean checkout resolves it from General. (A
  pin on its `main` came back on 2026-09-23 to see the owner-based
  threading before its release, and went again once 0.1.3 carried it.
  0.1.4, for checkpoint and restart, was used from the registry the day it
  was registered, with no pin at all.)
  The one `[sources]` entry left is **IMEXRungeKutta's**, which is not
  registered (compat `1.3` since step 12: 1.2's step count had no path for
  a MultiFloat, and the fix was made there, not worked around here); a path-tracked dependency's `[sources]` is honoured, so
  `bin/` and any scratch environment that `develop`s this package find it
  without an entry of their own. So `~/src/jl/TreeAMR` is
  still *not* what the tests see, and the bar is now higher than it was —
  it used to be that an unpushed change there was invisible here, and now
  an **unreleased** one is. A TreeAMR change this package needs has to be
  tagged and registered before it can be used; say so rather than editing
  that checkout and assuming the tests see it. Dropping the entry is also
  what lowered the Julia floor from 1.11 to 1.10, `[sources]` being a 1.11
  feature; it went back to 1.11 on 2026-09-25, with all the Tree*
  packages, so that unregistered dependencies can use `[sources]` again.
  `bin/Project.toml` still has one, for *this* package, which is
  unregistered — so the viewer environment has 1.11 as its floor and
  says so in its own `[compat]`.
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
- **The three RHS kernels carry `@inbounds`, and `@inbounds` does not
  reach into an `ntuple` closure** (measured after step 11). Two things in
  one place. The annotation propagates into an inlined callee only when
  that callee is marked `@propagate_inbounds`, and an anonymous closure is
  not — so `@inbounds ntuple(v -> prim[i..., v, b], Val(D+2))` is *inert*
  and `ntuple(v -> @inbounds(prim[i..., v, b]), Val(D+2))` is the one that
  works. Reading `D + 2` values that way is this package's characteristic
  idiom, so the mistake is made a tuple at a time, and it looks done. The
  IR is what settles it, not a stopwatch: `code_llvm` on the idiom carries
  5 bounds-error references unannotated, 5 with the annotation outside the
  closure and 0 with it inside, while the *timing* difference is 0.51 ms
  against a round-to-round scatter of 1.3 ms and came back inverted on the
  first pair of runs. And because the kernels' indices are now an
  assertion rather than a check, **`CI.yml` passes `check_bounds: 'yes'`
  explicitly** — it is already the action's default, and the point is that
  the package now depends on it. Do not remove that line, and do not add
  `@inbounds` anywhere else in `src/` without the same argument from
  `map_blocks!`'s contract and the ghost widths. The whole thing is worth
  19% of an RHS evaluation and 15% of the suite, and it changes not one
  bit; see "Bounds checking in the kernels" in `CODE.md`.
- **The RHS never mutates `u`.** The atmosphere reset lives in the
  integrator's two limiter hooks (`reset_stage!` and `reset_atmosphere!`,
  see the `:stage` bullet below) and in the driver after `regrid!`,
  never inside the RHS. The RHS-level floor on `P` and the stage reset on
  `U` are two mechanisms on purpose (owned cells versus prolongated
  ghosts and face states); do not merge them, and keep the floor counts
  by population — they are a measurement the design depends on.
- **The reset writes back only where a floor fired, and everything rests
  on that** (measured in step 8). A kernel that wrote
  `prim2con(con2prim(U))` unconditionally would be the same physics and
  would move every cell by a few ulp at every stage — and since `:stage`
  is the default, *every* run in this suite would then drift, turning
  every roundoff conservation bound in `CODE.md` into a tolerance and the
  `:none` comparison into an approximation. The reset's own tests would
  still pass. The claim that catches it is in `test/reset_tests.jl`: the
  tracked tube's state, drift, error, step count and mesh history under
  `:stage` and `:step` are **bit-identical** to the `:none` run's, and the
  injection is `0.0` rather than `≈ 0.0`.
- **`:stage` is *both* of IMEXRungeKutta's hooks, and the two hooks are
  two functions** (amended after step 11, replacing the step-8 note about
  OrdinaryDiffEq's `solve` keywords). IMEXRungeKutta's `SSPRK33` is in
  Butcher form: its stage limiter is called on a scratch copy of the two
  stage values per step the RHS reads beyond `uⁿ`, and *not* on the step's
  result, which is the step limiter's. So `hydro_integrator` passes
  `reset_stage!` as the stage limiter and `reset_atmosphere!` as the step
  limiter under `:stage`, and the step limiter alone under `:step`.
  Passing `reset_atmosphere!` as the stage limiter alone would leave every
  stored state unreset; passing it as both would count stage corrections
  as injection, which no total ever sees — a stage correction reaches
  `uⁿ⁺¹` only through the RHS, which conserves, so `reset_stage!` counts
  hits and no injection. The hooks are `init` keywords whose omission is
  silent, and `test/reset_tests.jl` asserts both the flooring and the hit
  counts (`:stage` 12 / 192 cells against `:step` 2 / 32 on the vacuum
  step), which is the only thing that would notice.
- **`hydro_solve!` does not alias; `evolve!` does** (amended after step
  11). `hydro_solve!` returns a new vector and leaves its argument alone —
  `sod.jl`, `sedov.jl` and the tests rely on that. `evolve!` builds its
  integrator with `alias_u0 = true` and hands each chunk's integrator the
  previous one's scratch (`reuse`), and must set `integ_prev = nothing`
  after any regrid that changed the mesh: the state vector has another
  length then, and IMEXRungeKutta refuses the scratch with an
  `ArgumentError` rather than silently resizing it.
- **A branch on the sign of a roundoff-level difference costs a third of
  the RHS** (measured in step 14). `:minmod` was `a * b ≤ 0 && return 0`
  and a ternary; on a state uniform to roundoff — the entropy wave's `v`
  and `p`, Sedov's ambient, the shear layer's pressure — that branch is a
  coin flip, and the flux kernel went from 3.54 ms on exact initial data to
  5.74 ms forty-eight steps later. It is two `ifelse`s now, which choose
  the same value in every case, NaN and signed zero included. **Keep
  per-face code free of data-dependent branches**, and benchmark on an
  *evolved* state: the exact initial data is the one state that hides this.
- **A loop over `d` that indexes a tuple of arrays is not unrolled**
  (measured in step 14). `for d in 1:D; fluxes[d][…]` in the divergence
  kernel took 6.35 ms against 1.38 ms written out per dimension, the same
  sums in the same order. An `ntuple` closure with `foldl` did *not* fix it
  inside a kernel (11.5 ms), nor did recursion on `Val` (65 ms); explicit
  methods for `D = 1, 2, 3` did. A loop over `v` into an `isbits` tuple is
  fine.
- **The reset is a fixed point of the *state*, not of its own flag**
  (measured in step 8). A pressure-floored cell whose kinetic energy
  dominates recovers `p` a fraction of an ulp below `p_floor` through the
  cancellation `E − ½S²/ρ`, so a second pass re-fires the rule — and
  writes the identical bits back, because `prim2con` is handed the same
  `ρ` and the same `v`. Idempotence of the state is bit for bit at both
  types in `D = 1, 2, 3`; a test that asserted the *count* went to zero on
  the second pass would fail at `Float64` in `D = 1` and `D = 3` and pass
  in `D = 2`, for no reason worth chasing.
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
  injection is *exactly* zero (measured in step 8 on Sod and the entropy
  wave, both hooks, `(0.0, 0.0, 0.0)`); a test that sees a nonzero
  injection on Sod, the entropy wave or Kelvin–Helmholtz has found a bug,
  not a tolerance to loosen. `accounting = false` returns `injection =
  nothing` and not a tuple of zeros, so that "not measured" cannot be read
  as "measured and zero" — and a state carrying a `NaN` has no total, so
  its injection comes back `NaN` in that one variable, which is the honest
  answer and not a defect.
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
  Forgetting the second is the bug that arrives one chunk late. All three
  are wired in `evolve!` as of step 7, and `test/driver_tests.jl` counts
  the outward-facing ghost entries that do not hold their boundary state
  on blocks the *cycle* created.
- **`speed_headroom = 1` throws on any discontinuous initial data**
  (measured in step 7). Not "may throw": Sod's first chunk ends at
  `λ_end = 1.9486` against a step sized for 1.25, a CFL number of 0.6236
  against the requested 0.4, and the recheck fires. Every shock case — Sod,
  Sedov's deposition, any restart from a jump — needs a headroom above the
  1.8522 growth step 4 measured, which is why `HydroCase(::SodTube)` uses
  2. A smooth case may use 1, and the entropy wave does; the recheck is
  what makes that a measurement rather than a hope.
- **The regrid cadence is bounded by the level cap, and the bound is
  tighter than it looks.** The derived margin covers
  `speed_headroom · λ · chunk` at the *cap's* spacing, and TreeAMR's
  recruitment reaches one ring of neighbours, so `refinement_buffer`
  throws once that travel exceeds one finest-level **block** width,
  `(L/roots)/2^cap` — which for the tracked tube at a cap of 2 is 1/32,
  and which the headroom doubles the demand on. `chunk = 1/50` is refused
  on Sod at `roots = 8, cap = 2`; `1/200` is what fits. Deepening the
  hierarchy by one level halves the admissible chunk, so a cap raised
  without shortening the chunk fails at the *buffer* rather than at the
  physics, which is a confusing place to meet it.
- **The tracking measure is taken before the regrid, and that is the
  point.** `tracked_share` asks what fraction of the strongly firing cells
  sit on blocks already at the cap, on the state at the *end* of a chunk —
  so it is a question about the mesh the *previous* regrid built. Taking it
  after the regrid would measure the criterion against itself and return 1
  always.
- **Dirichlet-from-initial-data reflects once a wave arrives**, and
  **`evolve!` does not check it — the case does** (settled in step 7,
  amending the first draft of this note). The check compares `t_end · λ`
  against the distance from the *feature* to the nearest physical
  boundary, and both of those are case knowledge: the driver knows neither
  where the feature is nor what the supremum of `λ` over all time will be,
  and the `λ` it can measure is the one that is not a bound. So
  `assert_no_arrival` stays in `src/sod.jl`, the tests call it beside the
  run, and every later case owes itself the same. If it fires, shorten
  `t_end` or enlarge the box; do not remove it. It is written on the
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
  carries `speed_headroom` as a case parameter beside the recheck (step 7),
  and Sod's 1.8522 is the number that sizes it. `sod_errors` sidesteps this by
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
- **A shock fires forever, and it fires at 0.57 and not at 1** (measured
  in step 6). Two halves of one trap. A captured discontinuity's `τ` does
  not fall with `h` — 0.5413, 0.5975, 0.5722, 0.5858 over a factor of
  eight — so the indicator can never terminate refinement on a shock and
  `maxlevel_cap` is what does, on *every* shock case rather than as a
  safety net; a test that expects a shock case's depth to be an output
  has misread the criterion. And the value is 0.57 and not the ≈ 1 a
  *sharp* jump scores, because capture spreads the jump over three or
  four cells: Löhner's canonical `τ > 0.8` would detect a discontinuity
  in the initial *data* and miss the shock the scheme is carrying. The
  thresholds are calibrated on the smooth features, which are the only
  ones whose refinement can terminate.
- **A kink does *not* fire forever, despite the argument that it should**
  (measured in step 6). The slope-jump-over-slope-sum ratio really is
  `h`-independent, but a numerically computed rarefaction head is not a
  kink: the scheme rounds the corner over a nearly fixed *physical*
  width, and the local `ε` term takes over the denominator once the first
  differences `h·s` fall below `ε·4|u|` (about `h·s = 0.04` for `O(1)`
  data). Measured head: 0.1736, 0.1050, 0.0569, 0.0286 — first order.
  This is the good outcome, and it is the reason a rarefaction is
  refinable to a finite depth; do not "fix" it by shrinking `ε`.
- **A tracked mesh has no coarse-fine face worth measuring, and that is a
  consequence of tracking** (measured in step 9). `tracking == 1` means
  every strongly firing cell sits on a block at the cap, and the travelling
  margin then puts the refined region's boundary *ahead* of the feature by
  construction — so a tracked run's coarse-fine faces stand in gas the
  solution has not reached, and the interface flux restriction, the
  prolongation and the floors all act on undisturbed ambient. On Sedov,
  `fixup = false`, `p = 1` and `reset = :step` came back with the *same*
  exponent, peak, cell count, mesh history and tracking as the run they
  control, and the two reset cadences were bit-identical. So an interface
  claim goes on a **static** two-level mesh the feature crosses —
  `sod_forest(:middle)` for the tube, `sedov_forest(:center)` for the blast
  — and a tracked run that "conserves" is not evidence that the fixup does
  anything. Sod's tracked tube *did* show a leak, and only because its
  Dirichlet boundary sits inside the refined region.
- **Sedov's chunk bound is tighter than Sod's, and the hot spot's `c_s` is
  why.** The derived margin covers `speed_headroom · λ · chunk` at the
  cap's spacing and must stay under one finest-level block width, and the
  blast's early `λ` is `sqrt(γ p_hot/ρ₀)` with `p_hot = (γ−1)E₀/V_D(r₀)` —
  **6.76** in `D = 2` and **8.27** in `D = 3`, against an `O(1)` wave
  speed. It also scales as `r₀^{-D/2}`, so halving `r₀` doubles `λ` *and*
  halves `h_cap`, tightening the admissible chunk by four. `chunk = 1/400`
  fits the `D = 2` configuration at `r₀ = 1/16` and `1/1000` is needed at
  `1/32`; a chunk that does not fit is refused by `refinement_buffer`
  naming the constraint, which is the guard working.
- **Sedov's first chunk outgrows its own speed too**, by **1.16537,
  1.25164 and 1.10396** in `D = 1, 2, 3` (measured in step 9). `CODE.md`
  said the hot spot's `c_s` is the maximum and `λ_max` only decreases; that
  is right about the blast and wrong about the first chunk, because the
  jump at `r₀` is a Riemann problem and its star region is not present at
  `t = 0`. `HydroCase(::SedovBlast)` uses `speed_headroom = 2`. Any case
  whose initial data has a jump in it needs a headroom above 1, and the
  recheck is a detector rather than a guard.
- **`shock_radius` reads cells, not boxes, and turning it into a
  `firing_boxes` sweep would break the exponent** (measured in step 9). A
  per-block bounding box loses the correlation between dimensions: the
  shell crosses a block diagonally, so the box's outermost corner sits
  about `w²/(2 r_s)` beyond the outermost firing cell — 22% at the start of
  the fit range and 8% at its end. A bias that *shrinks as the shock grows*
  lands directly on `d log r_s / d log t`, and it cost about **0.13** of an
  exponent of 0.5. The host loop is an oracle in the spirit of
  `reduce_to_grid` and runs once per chunk against hundreds of steps.
- **The accumulated injection is an equality under both cadences now**
  (amended after step 11: under IMEXRungeKutta only the step limiter's and
  the post-regrid correction reach a total, and only they are counted —
  the `:stage` drift equals its injection to `8.9e-16` in `D = 2`). What
  follows is the step-9 finding under OrdinaryDiffEq, kept for why the
  accounting is defined as it is. It *was* a bound under `:stage` and an
  equality under `:step` (measured in step 9). `SSPRK33`'s three stage vectors
  enter the step's result with weights `1/6`, `2/3` and `1`, and
  `reset_atmosphere!` is not told which stage it is in, so the accounting
  adds raw `Σ hᴰ ΔU` that reached the state scaled. Measured ratio of
  drift to injection **0.520**; under `:step` the two agree to `4.4e-16`.
  So a conservation claim "net of the injection" is an *equality* only
  under `:step`, and under `:stage` it is `|Δtotal| ≤ Σ injection`. Step 8
  could not see this, because every injection it measured was exactly zero.
- **On a blast it is the pressure floor that fires, not the atmosphere
  rule, and the coarse-fine face is what drives it** (measured in step 9).
  The evacuated interior never reaches `ρ_atm`: numerical diffusion holds
  the minimum density at `6.7e-2` against a floor at `10⁻⁶`, so no velocity
  is ever zeroed and every mass injection in this package is still exactly
  zero. What fires is the interface flux restriction replacing a coarse
  cell's flux with the average of its fine neighbours', in gas whose
  internal energy is `p_amb/(γ−1) = 2.5e-5`. Do not raise `ρ_atm` to make
  the atmosphere rule fire — it is six orders below the data because the
  refinement criterion's `ε_g` term needs it there.
- **In `sedov_reference.jl`, never form `V − V₀`** (measured in step 9).
  The similarity parameter and its centre value agree to the last bit long
  before `λ` is small — the substitution raises `w` to a power near 10 — so
  every formula is written in terms of `u = V − V₀` and `V` is only ever
  *formed*, as `V₀ + u`. Writing the difference instead produces a `NaN` at
  `λ ≈ 10⁻³`, and the adaptive quadrature then recurses to its depth cap on
  a panel it can never accept, which looks like a hang rather than an
  error.
- **The `E₀` the similarity law takes is the measured one**, not the
  nominal: which cell centres fall inside `r₀` is a property of the mesh,
  and the ratio is 1.0345 in `D = 2` and 1.0444 in `D = 3`. In `D = 1` the
  deposition tiles `2r₀` exactly and the ratio is 0.999996875, the missing
  `3.125e-6` being the ambient share of the top hat that `measured_E₀`
  subtracts along with the rest of the box.
- **Every test file is `include`d into the same `Main`, so a top-level
  `const` name must be unique across files — and only the floor version
  will tell you** (found in step 10b, when the merge of steps 8–10 went
  red on the two 1.11 entries of CI and green on the two 1.13 ones; the
  floor cell was 1.10 for a while, is 1.11 again, and both behave the
  same way). `driver_tests.jl`
  and `sedov_tests.jl` both defined `TRACKED_1D`, `TRACKED_2D`, `FINE_2D`,
  `COARSE_2D` and `NOFIX_2D`. Julia 1.12 and later quietly allow a `const`
  to be redefined, so every local run at 1.13 and the clean-checkout check
  passed; 1.11 throws `invalid redefinition of constant`, five minutes into
  otherwise green output. Name a file's shared runs with their case
  (`TRACKED_SEDOV_1D`, `TRACKED_KH`); `runtests.jl` now fails on any
  duplicate, from the source text, before anything is included; and a
  change that adds a test file runs once under the floor version before it
  is merged — the "Commands" section has the line. The floor is 1.11, and
  every Julia before 1.12 checks this.
- **A count taken as the ceiling of a float quotient gets one more at
  `Float32`** (found in step 12). `evolve!` counted its chunks as
  `ceilint(t_end / chunk)`, which is exact at `Float64` on every case here;
  at `Float32` the two-dimensional tube's `3//20 / 1//200` is
  `30.000002f0`, and the 31st chunk ran from `30 · chunk`, an ulp below
  `t_end`, to `t_end` — one step and one regrid more than the `Float64` run,
  so the two mesh histories differed in length and nothing else. The count
  is `chunk_count` now, IMEXRungeKutta's step-count rule: a quotient within
  a few ulp of an integer is that integer, and the last interval ends at
  `t_end` exactly. Any new count of intervals built from a float quotient
  wants the same, and only a `Float32` run will show it missing —
  `test/type_tests.jl` compares mesh histories exactly for that reason.
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
  only thing that will report a violation. **A new block-shaped launch
  goes through `TreeAMR.launch_by_owner!`** and a new host loop over
  blocks through `TreeAMR.threaded_chunks` (TreeAMR 0.1.3's rule; the
  ghost floor count is this package's one launch of its own and follows
  it). Both are unexported, and `test/prerequisite_tests.jl` checks by name
  that they and `threadchunks` still exist.
- **The Kelvin–Helmholtz formulas were transcribed from memory and the
  check found exactly one error** (step 10, closing the note that used to
  say "check every one before recording a number"). The *profiles* — the
  four branches, `ρ_m`, `v_m`, the parameters, the perturbation, `γ`, `p`
  and `t_end` — are McNally, Lyra & Passy (2012), ApJS 201:18
  (arXiv:1111.1764), equations (1)–(5), **term for term**; the test
  re-evaluates them from the paper's own literals and the worst difference
  is *exactly zero*. What was wrong was the **weighting of `M(t)`**:
  `CODE.md` said `e^{−4π|y − ¼|}` "so that the lower interface alone is
  read", and equations (6)–(8) use that for `y < ½` and
  `e^{−4π|(1−y) − ¼|}` for `y ≥ ½`, so **both** interfaces are read,
  mirrored. Reading one alone would have halved the signal and picked up
  the other interface's mode with the wrong sign. And a second omission
  rather than an error: on a mesh of unequal cells the sums are the
  **area-weighted** (14)–(17), `w_i = h_b²`, or `M` jumps at every regrid.
  Also, unchanged: a `Float32`
  Kelvin–Helmholtz run diverges from the `Float64` one late in the run
  by design (the instability amplifies roundoff), so assert the growth
  rate and the early chunks, not the final state — measured, to `t = 2/5`
  the mesh, the step count and the tracking are *equal* and `M(t)` agrees
  to 200 ulp of `Float32` (156 before the move to IMEXRungeKutta); and
  MultiFloats cannot
  run it at all (`sin`, `exp` are not implemented), only Sod and Sedov.
- **A feature that *grows* gets nothing from a travelling margin, and the
  margin is what decides how much of the box is refined** (measured in step
  10). On the tube and the blast the chunk is bounded from above by
  `refinement_buffer` *throwing*; on the shear layer it is bounded well
  below that, quietly, by the saving going away. `chunk = 1/200` derives a
  3-cell margin and refines 128 of the 256 possible finest blocks;
  `chunk = 1/64` derives 7 and refines **all 256**, which is the uniform
  fine mesh under another name and passes every test in the file. So a
  Kelvin–Helmholtz-like case has to be checked for what fraction of the box
  it refines, not merely for whether the buffer was accepted.
- **HLLE is not a neutral baseline on a contact-dominated flow, and the
  factor is two in linear resolution** (measured in step 10). `M(1.5)` on
  uniform meshes: HLLC `0.0116 / 0.0755 / 0.1240` at 32²/64²/128², HLLE
  `0.00066 / 0.0066 / 0.0357` — so HLLC at 64² is ahead of HLLE at 128²,
  and the two differ in the growth *rate* (2.5804 against 1.1712) and not
  merely in the amplitude. `kh_run`'s `riemann` defaults to `:hllc` and is
  the only place in the package that overrides `:hlle`; do not "tidy" it
  away, and do not change the package-wide default, which is HLLE because
  that is *the* GRMHD flux.
- **A tracked run measured against a uniform fine run *with the same flux*
  measures the mesh, not the flux** (measured in step 10, and it is how the
  HLLE/HLLC criterion written before the runs failed to decide). Both
  fluxes reproduce their own fine reference to under a percent — 0.42% and
  0.29% — while the two fine references themselves differ by 247%. What
  decides a flux is a **resolution sweep** under each, which is what the
  record now carries. Any later "which method is better" comparison on an
  adaptive mesh has the same trap in it.
- **The exact Riemann solver is a reference, not a flux.** It and the
  Sedov similarity code are host `Float64`, converted once at the
  comparison. For the Sedov law, `E₀` is the *measured* energy deposited
  on the adapted mesh at `t = 0`, not the nominal value.
- **Code coverage under threads is a 100× slowdown, and it looked like
  the runner** (measured in step 7c, correcting step 7b). Julia compiles a
  coverage hit into an *atomic* read-modify-write on one global counter per
  source line, and the counters of 32 neighbouring lines share a 256-byte
  block; a KernelAbstractions CPU launch is one task per thread over chunks
  of the *same* kernel, so every thread executing a kernel line contends on
  the same cache line. Measured locally on the `D = 2` entropy-wave sweep:
  1.83 s at one thread and 0.79 s at four with coverage off, **10.35 s and
  79.13 s** with it on — 5.7× at one thread, 100× at four, and the sign of
  threading inverted. On CI it turned a 5-minute serial job into a
  53-minute threaded one, the `D ≥ 2` segments 14–17× slower at four
  threads than at one. Step 7b read that same table as a fact about
  GitHub's 4-vCPU runners and split the suite; it was the coverage, which
  both jobs had on. Measured since, on the **whole suite** at one thread,
  locally and back to back: **4 m 01 without coverage and 12 m 38 with
  it**, a factor of **3.14**, both green at 11609 tests. So coverage is
  collected **once, where it is read, on the fastest cell of the four**:
  `coverage: true` on the floor version's `macOS-latest` entry alone
  (`version: "1.11"` when it was measured, `version: "1.10"` after
  TreeAMR's release, `version: "1.11"` again since 2026-09-25).
  Instrumented on CI, one thread, the four combinations measure **macOS
  1.11 12–17 m, ubuntu 1.11 18–20 m, macOS 1.13 23 m 22, ubuntu 1.13
  42 m 02** — macOS 1.5–1.8× faster than Linux at either version, 1.11
  1.9–2.3× faster than 1.13 on either architecture, each effect
  reproduced across the other axis. Most of the version column is
  instrumentation itself: measured locally, back to back, coverage cost
  **1.04× on 1.11 and 3.14× on 1.13** (8 m 55 → 9 m 17 against 4 m 01 →
  12 m 38). **Re-measured when the floor moved to 1.10**, the same way and
  at 11622 tests: **4 m 25.5 → 8 m 05.4 on 1.10 (1.83×)** against
  **3 m 32.3 → 12 m 38.3 on 1.13 (3.58×)**. So the *conclusion* holds —
  the floor cell runs instrumented in 8 m 05 against 12 m 38, a factor of
  1.56 — and the reason given for it does not: instrumentation is nearly
  free on **1.11**, not on floors, and 1.10 is the slower version
  uninstrumented. Do not repeat the 1.04× as though it were a property of
  whichever version is the floor — though since the floor is 1.11 again,
  it is once more the number that applies. With no `VERSION` check anywhere in
  `src/` or `test/` the lines reported are identical whichever cell
  carries it.
  The step's condition is `matrix.coverage == true && (github.ref ==
  'refs/heads/main' || github.event_name == 'workflow_dispatch')`, with
  `julia-processcoverage` and the Codecov upload under the same
  condition. The other serial cells run the same lines, so a second
  instrumented cell would pay 3.14× to say what the first said; a branch
  or PR is not what the badge reflects; and the dispatch arm exists so
  the upload path can be exercised on purpose, because a reporting step
  that runs nowhere reports nothing. **Never put coverage on the threaded
  entry.** A `D ≥ 2` sweep costs its local seconds and may go anywhere
  the claim belongs. `timeout-minutes: 30` is there so that a runtime
  regression fails loudly rather than billing an hour; it is a flat
  number only because the instrumented cell is the fast one, and moving
  coverage to a 1.13 or Linux cell needs the per-cell form back (there a
  *healthy* run measured 42 m 45).
- **Base's reductions are not bit-identical across machines, and that is
  measured** (step 7b, kept in step 7c). On all four CI runners — macOS
  arm64 on the same Julia 1.13 and the same packages as this machine
  included — the order-one conserved totals came back **1 to 4 ulp** off.
  Base's `sum` and `mapreduce` reduce under `@simd`, so the arrangement of
  their vectorized partial sums follows the CPU target, and every total and
  norm here goes through them via TreeAMR's `block_mapreduce`. Bit-identity
  across *thread counts* holds and is asserted; bit-identity across
  *microarchitectures* was never on offer. So a test that stores a number
  and compares against it needs a relative tolerance of a few hundred ulp
  **and** an absolute floor, and a claim about a *difference* of two
  order-one totals — every drift in this package — cannot be made
  relatively at all. Assert such a quantity against its bound instead,
  which is what the suite does and why it travels.
- **At `speed_headroom = 1` the CFL recheck is protected by nothing but
  integer-step quantization, and that protection shrinks as the mesh
  refines** (found after step 11, running the shear layer at `--cap=4`).
  `check_cfl` allows `8 eps` of slack, so a headroom of exactly 1 tolerates
  *no* growth of `λ` within a chunk except what the step rounding gives:
  the step taken is `chunk / ceil(chunk / dt_requested)`, which at cap 2 is
  8.34% under the step asked for and at cap 4 was **0.0083%** under it. The
  shear layer's `λ` grows 1.00033 within a chunk at every cap — that number
  is in `CODE.md` and always was — so cap 2 passes and cap 4 throws on the
  *same* physics. Remedy: `--speed-headroom=1.05`, which is 150× the
  growth for 5% more steps. **Shortening the chunk is not the remedy**,
  though it is the remedy for the buffer margin — it scales the growth down
  linearly and leaves the tolerance a lottery on where `chunk / dt` falls
  relative to an integer. Two constraints, two different fixes, and it is
  easy to reach for the wrong one.
- **And the two fixes fight each other: raising the headroom spends the
  buffer margin.** The margin is `ceil(speed_headroom · λ · chunk / h_cap)
  + 1`, so the headroom that fixes the CFL recheck widens the very quantity
  the margin bounds. Measured: `--cap=4 --chunk=1/200` derives exactly 8
  cells at headroom 1 — `N`, with nothing to spare — and
  `--speed-headroom=1.05` takes it to 9 and throws out of
  `refinement_buffer`. The margin is also derived from the `λ` *then
  current*, which on the shear layer reaches 2.6048 against the initial
  data's 2.5412, so any table computed from the initial `λ` is a lower
  bound. At cap 4 the combination that works is `--chunk=1/400`: a 5-cell
  margin, and room left for whatever headroom the recheck needs.
- **CI's artifact glob is extension-specific, and it silently drops what it
  does not match.** It was `path: bin/output/*.png` until the movie
  arrived, so an `.mp4` rendered in CI would have been produced, asserted
  non-empty by `test -s`, and then left out of the upload with no error
  anywhere. It is `bin/output/*` now. A new output *kind* in `bin/` means
  checking that glob and the `test -s` list, neither of which fails loudly.
- **In `bin/`, two exported names collide with Makie** (found in step 11).
  TreeAMR exports `scatter!` — the state-vector-into-field-set one — and
  Makie exports the plot recipe; TreeHydro exports `density` and Makie's
  `@recipe` exports `density`/`density!` too. Julia errors on any *use* of
  an ambiguous name, not on the `using`, so the failure arrives at the
  first plot call and names neither package helpfully. Write
  `CairoMakie.scatter!` in full, and never write `density` in `bin/` at
  all — the viewers read slot 1 of `P` directly. TreeWave met the first of
  these; the second is this package's own, and any new `bin/` script
  inherits both.
- **A viewer snapshot must *materialize* everything it keeps** (step 11).
  `hostcopy(fs)` returns `fs` **itself** on the CPU — deliberately, since
  the viewer only reads — and `regrid!` reuses and resizes the working
  array, so a frame that stored a view or a bare `Array(work)` comes back
  holding the *last* frame in every slot and the filmstrip shows four
  copies of the final one. `Float64.(interiorview(fs, b, v))` does the
  materializing and doubles as the one place a `Float32` run stops being
  one; everything below it is a figure. The same applies to the `τ` a
  viewer computes with `cell_tau`: take the number, not the array.
- **`P` has `D + 4` variables and only `D + 2` of them are primitives.**
  Slots `1 … D+2` are `(ρ, v₁…v_D, p)`; slot `D+3` is the cell's signal
  speed and `D+4` its floor-hit flag, both written by the `con2prim`
  kernel. A loop over `1:P.nvars` calling them primitives plots two
  diagnostics as physics.
- **`evolve!` records no time vector, and its histories are off by one
  against the observer.** `nblocks_history`, `λ_history` and
  `buffer_history` are per *chunk*, while the observer fires `nchunks + 1`
  times — once at `t = 0` and once per chunk — so zipping the two is a
  frame-shifted plot. Take `t` and the block count from your own observer,
  as `kh_run` does with `ts` and `nbs` and as both viewers do.
- **A checkpoint is written after the regrid, and never before** (added
  2026-09-29). There the integrator holds nothing but `(t, u)` and the next
  chunk begins from `u` alone, so a restart is the uninterrupted run bit for
  bit. A write moved before the regrid — to "save the state the observer
  saw", say — would make a restart replay the regrid *and* the post-regrid
  reset, and the chain test in `test/checkpoint_tests.jl` is what would
  notice. `u` is what is saved, not `U.work`: the post-regrid reset acts on
  `u`. The last chunk writes nothing — it has no regrid and is not a restart
  point.
- **HDF5 is the caller's to load, and `evolve!` checks it before the
  cycle** (added 2026-09-29). TreeAMR's `save_checkpoint` and
  `load_checkpoint` have methods only once `using HDF5` has loaded
  `TreeAMRHDF5Ext`; this package calls nothing but those and never names an
  HDF5 type, so **do not add HDF5 to `[deps]`** or as a weak dependency —
  it is in `test/Project.toml` only. A checkpoint keyword without it throws
  at the call. The test of that refusal runs only while the extension is
  *not* loaded, and a package cannot be unloaded, so `checkpoint_tests.jl`
  must stay the first file to load HDF5: a file above it that loaded HDF5
  would skip the test silently.
- **Every parameter that decides a number goes in the recipe, and every
  accumulator in the run state** (added 2026-09-29). A new `evolve!`
  keyword that changes the numbers and is not added to `run_recipe` lets a
  restart with a different value run as though it were the same run; a new
  returned accumulator not added to `run_state` comes back from a restart
  counted from the checkpoint rather than from `t = 0`. The chain test
  compares the fields listed in `CKPT_FIELDS`, so a new returned field goes
  there too. Only `t_end` may change on a restart, and the case's closures
  (`initial`, `boundary`, `reference`) cannot be compared at all — they are
  trusted.
- **A wall-time stop returns the checkpointed state, not an answer** (added
  2026-09-29). `finished = false`, `l1 = linf = nothing`, and `U`, `u` and
  `forest` are the state *after* the last chunk's regrid — which is what was
  saved — with `U` scattered and its ghosts filled. A caller reading `r.l1`
  must look at `r.finished` first. The limit is timed from the call and
  estimates the next chunk as the longest so far plus the longest write, so
  startup and compilation are the caller's margin below the queue's limit.
- **Rotation deletes matching files from earlier jobs** (added 2026-09-29).
  After each successful write, every `"<prefix>.it<digits>.h5"` in the
  prefix's directory but the file just written and the newest
  `num_checkpoints_keep − 1` others is removed, whichever job or run wrote
  it — which is the point in a job chain, and means **two runs must never
  share a prefix**: each would rotate the other's files away. The pattern is
  anchored, so `.h5.partial` files and longer prefixes are never touched.
  The file just written is recognised by its *name*, never by comparing
  path strings: `run//sedov` lists as `run/sedov`, and the first version,
  which compared paths, deleted the only checkpoint at
  `num_checkpoints_keep = 1` (found in review; the rotation testset has the
  doubled separator for that reason).
- **The observer's state is not checkpointed** (added 2026-09-29). A
  restart does not call the observer at `t = 0`, and hands it only the
  chunks it runs; whatever the observer accumulated in an earlier job is the
  caller's to keep. `kh_run`, whose McNally diagnostics are exactly such
  records, forwards no `evolve!` keyword, so it cannot be restarted by
  accident — do not add checkpoint keywords to it without restoring its
  `ts`, `Ms`, `Ks` and `nbs` from somewhere.
- **Every field set over a reflecting forest needs its parity, and the
  parity is physics** (added 2026-09-29). TreeAMR refuses a `FieldSet` over
  a forest with a reflecting face unless `parity` gives every variable
  `EvenParity` or `OddParity` in each such dimension. `state_parity(forest,
  nvars)` is the table for `U` (`D + 2`) and `P` (`D + 4`): slot `1 + d` odd
  in `d`, everything else even — the two diagnostic slots of `P` are
  scalars — and `flux_parity(forest, d)` the product rule for the fluxes.
  A new `FieldSet{T}(forest, …)` anywhere a reflecting forest can reach
  must pass one; on every other forest both return `nothing`. And a
  reflecting face is **not a hook**: the boundary hook never sees it, and
  a case whose every non-periodic face reflects has `boundary = nothing`.
- **The half box is the full box to roundoff, not bit for bit, and the
  full box is why** (measured 2026-09-29). HLLC's star state sums `A + X_L
  − X_R` in one order at a face and in another at its mirror image, so the
  periodic box drifts from its own mirror image by a few ulp per step,
  while the half box's mirrors are exact. `test/reflecting_tests.jl`
  asserts equal steps and meshes exactly and the states at a roundoff
  bound; do not "fix" that into an equality. The mirrored seed's envelope is
  evaluated at the mirror point and negated in the upper half for the same
  reason — `sin(2π(1 − y))` is not `−sin(2πy)` in floating point — and
  `mode_amplitude` reverses the upper half's `v_y` under that seed, or the
  two interfaces cancel.
- **The showcase's loop is a second loop, on purpose, and it must follow
  `evolve!`'s order** (decided 2026-09-29): step, CFL recheck, observe,
  regrid, rebuild the problem, `gather!`, `integ_prev = nothing`,
  `reset_atmosphere!`. It exists because `evolve!` fixes `chunk` and
  `maxlevel_cap` for a whole run and the zoom needs both per chunk — frame
  times that shrink with the zoom, and a cap that is a window about the
  camera. A change to `evolve!`'s order is a change to
  `showcase/simulate.jl` too.
- **The showcase window is sized for the zoom-out, not the zoom-in.** Level
  `ℓ` is allowed in a box covering the *largest* view that needs it — twice
  the height of the smallest — because the zoom-out reads the final mesh at
  every zoom, and a box sized for the current view would leave its outer
  part at the level below. The zoom-out stays centred on the anchor for the
  same reason; the pan home happens at zoom 1, where the floor level covers
  the whole box.
- **Measured numbers go into `CODE.md`**, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes. The test that produces one runs
  in the suite, on every push, like every other.

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
  instead (both carried over from the step plan's ground rules when step 15
  deleted it).
- Generic in `T` and in the backend from the first line of any new driver:
  TreeWave records that retrofitting either was a rewrite, and
  `test/type_tests.jl` and `test/device_tests.jl` are what would notice.
- Spec-first: when the implementation shows `CODE.md` was wrong or
  incomplete, amend it and say so in it — "(amended in H3)", "(measured
  in H4)" — rather than diverging silently.

## Repository facts

- **`origin` is `git@github.com:eschnett/TreeHydro.jl.git`, and `main` tracks
  it.** Work on a branch, and do not push, open a pull request, or merge
  to `main` without being asked. Every change lands on `main` only after
  review, as each step did.
- `TODO.md` is Erik's personal to-do list. **Do not modify it.**
  `TODO.md~` is an editor backup, not a file of this package. Both are
  kept out of the tree by `.gitignore`.
- `.gitignore` exists, in TreeWave's image: `Manifest.toml` everywhere,
  `bin/output/`, `docs/build/`, editor leftovers, `TODO.md`. No
  `Manifest.toml` is tracked — that is what makes the clean-checkout
  check above mean something. `CODE.md` and this file are committed;
  `PLAN.md`, the step-by-step work breakdown, was committed too until step
  15 deleted it with the last milestone done.
- No generated file is tracked. Everything in the tree is written by hand
  and reviewed as such; `test/references/*.toml`, which step 7b generated
  and step 7c removed, were the one exception and are gone. `bin/output/`
  is gitignored and the viewers write PNGs there.
- **There is no TreeAMR pin now and still two Manifests** (amended when
  TreeAMR 0.1.1 was released, at 0.1.3, and at 0.1.4). Both environments
  resolve TreeAMR from the registry — the root at `0.1.4`, `bin/` at a
  `0.1.3` bound that admits it, left alone because nothing in `bin/`
  checkpoints — so the `rev = "main"` entries are
  gone; what is left is a compat bound in each `Project.toml`, and
  `bin/` still has its own `Manifest.toml` (gitignored, like the root's),
  so `Pkg.update("TreeAMR")` at the root does not touch it. The viewers
  can therefore still resolve a *different* TreeAMR from the tests — an
  older one, if `bin/`'s Manifest is stale — which is what the `viewer`
  job in CI exists to catch. Grep for `TreeAMR =` across both files rather
  than editing from memory.
- **`bin/backend.jl` is `include`d by both viewers and by
  `bin/benchmark.jl`**, which runs against the *package* environment. So it
  may use only what both environments have, which today is
  `KernelAbstractions` — which is why that is a dependency of
  `bin/Project.toml` at all, and why a scratch environment that runs the
  benchmark must add it by name (the first Symmetry CPU job failed in nine
  seconds without it).
- One workflow with two jobs: `.github/workflows/CI.yml`. `test` runs the
  whole suite on every
  push that touches something other than Markdown, over **four cells
  spelled out one at a time** rather than a product — 1.11 on macOS (the
  floor, and the cell that carries the coverage), 1 on Linux, 1 on macOS
  (which carries the arm64 reduction claim) and 1 on Linux at 4 threads.
  The redundant fourth pair of the product is the floor on Linux; the
  floor is a property of the version, not of the OS. **Coverage is
  collected on one cell, on `main` or a manual dispatch only**, and never
  on the threaded entry, where it costs a factor of a hundred — see
  "Things that will bite". End to end, an ordinary push costs about
  **12 m** with the threaded cell on the critical path, and a push to
  `main` **18 m 02** with the *instrumented* cell on it, against
  **14 m 20** for the suite before any of this — the instrumented cell
  has been the critical path under every arrangement tried (18 m 02 on
  macOS at 1.11, 20 m 42 on Linux at 1.11, 42 m 45 on Linux at 1.13). And treat every CI timing as ±70%: one instrumented cell came
  back at 18 m 08, 11 m 24 and 19 m 52 on three consecutive runs, so only
  orderings wider than that carry a decision, and a claim about which
  cell is slowest needs more than one run behind it.
  `.github/workflows/CI.yml`'s comments state the arrangement, not how it
  was arrived at; the history lives in `CODE.md`'s "Testing".
  The second job, `viewer`, instantiates `bin/` and renders all four
  figures, and is **ungated** — it runs on pull requests too, because
  unlike coverage it is a check rather than a report, and `bin/` is the one
  part of the tree no test would notice breaking. It runs beside the four
  test cells rather than after them. **Measured on PR #1: 11 m 43 cold and
  3 m 57 warm**, the difference being entirely the instantiate step —
  7 m 31 against 5 s, 64% of a cold run and 2% of a warm one. The four
  renders are 3 m 50 / 3 m 21 either way. So **whether it is on the critical
  path depends only on its cache**: cold it was the longest of the five
  jobs (cells at 8 m 28, 9 m 52, 10 m 36, 11 m 09), warm it was the shortest
  by a factor of two (cells at 7 m 55, 10 m 28, 12 m 34, 10 m 37). It is
  therefore slow on a first run and after a CairoMakie release evicts the
  entry, and cheap otherwise. `timeout-minutes: 30`, as on the test job, is
  2.5× a cold draw. If it ever does run long the lever is the **cache** and
  never the renders: `--case=both` would save one `using CairoMakie`, about
  20 s against a 451 s precompilation.
- Sibling checkouts: `~/src/jl/TreeAMR` (the mesh; read its `CLAUDE.md`
  and `CODE.md` for the API and its sharp edges) and `~/src/jl/TreeWave`
  (the other application; copy the *patterns* of its `precision.jl`,
  `device.jl`, `bin/backend.jl`, viewers and thread workload — do not
  depend on it).
