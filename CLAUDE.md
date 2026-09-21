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

**Scaffolding (H0), the scheme on a uniform mesh (H1), the coarse-fine
faces on a static mesh (H2), regridding (H3), Sedov with the atmosphere
reset (H4) and Kelvin–Helmholtz with its viewers (H5) are done; precision
(H6a) is step 12 and is next.**
`CODE.md` is complete and reviewed. What exists: `Project.toml` with
`TreeAMR = "0.1.1"` from the General registry — TreeAMR was released, and
the `[sources]` pin to its GitHub `main` is gone, which also lowered the
Julia floor from 1.11 to 1.10; `src/TreeHydro.jl`, the module shell;
`src/precision.jl` (`wrap`, `ceilint`, `floorint`, `tofloat64`) and
`src/device.jl` (`to_backend`, `hostcopy`, `hostcopy!`), both ported from
TreeWave; `src/floors.jl`
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
construction; and from step 6 `src/refinement.jl` (`lohner`, `cell_tau`,
`indicator_scales`, `hydro_flags`, `refinement_buffer` — the criterion
alone, with no driver and no `regrid!` yet); and from step 7
`src/driver.jl` (`HydroCase`, `evolve!`, `uniform_run`, `check_cfl`,
`tracked_share`, `reduce_to_grid`, `l1_difference`), with
`HydroCase(::SodTube)` in `sod.jl` and `HydroCase(::EntropyWave)` plus
`entropywave_primitive` in `entropywave.jl`, and with `hydro_flags` and
`max_signal_speed` each split into a `FieldSet` core and a
`HydroProblem` forwarder; and from step 8 the atmosphere reset —
`ResetAccounting` and `reset_atmosphere!(u, integrator, p, t)` in
`floors.jl` (a `map_blocks!` launch over the owned cells of
`statearray(u, U)`, writing back *only* where a floor fired), and in
`evolution.jl` `ghost_floor_hits`, `check_reset`, the state-vector method
of `conserved_totals`, `hydro_solve!`'s `reset` keyword and
`HydroProblem`'s `accounting` one, with `evolve!` gaining `reset`
(defaulting to `:stage`), `accounting`, the post-regrid reset and the
three new return fields `reset_hits`, `ghost_hits` and `injection`; and
from step 9 `src/sedov_reference.jl` (`SedovSimilarity`, `sedov_alpha`,
`sedov_exponent`, `sedov_radius`, `sedov_profile`, `exponent_fit` and an
`adaptive_simpson` of its own — host `Float64`, the similarity law
*derived* from the similarity equations rather than transcribed, a
*reference* and not a method) and `src/sedov.jl` (`SedovBlast`,
`sedov_state`, `ambient_state`, `sedov_initial`, `sedov_conserved`,
`sedov_boundary`, `HydroCase(::SedovBlast)`, `sedov_forest` with
`refined = :center | :corner | :edge`, `sedov_similarity`, `measured_E₀`,
`shock_radius`, `peak_compression`, `assert_no_arrival(::SedovBlast, …)`
and `sedov_static`); and from step 10 `src/kelvinhelmholtz.jl`
(`KelvinHelmholtz` — `D = 2` only, and it refuses any other — `kh_state`,
`kh_initial`, `kh_conserved`, `HydroCase(::KelvinHelmholtz)`,
`mode_amplitude` and `max_y_kinetic_energy` (McNally's two diagnostics,
host loops in block order read once per chunk through the observer),
`growth_rate`, and the two measurement drivers `kh_run` and `kh_uniform`,
which install that observer and are the only place either diagnostic can
be taken).
Tests, **one suite run whole** since step 7c:
`test/precision_tests.jl`, `test/prerequisite_tests.jl`,
`test/eos_tests.jl`, `test/riemann_tests.jl`, `test/evolution_tests.jl`,
`test/reset_tests.jl`,
`test/entropywave_tests.jl`, `test/exact_riemann_tests.jl`,
`test/sod_tests.jl`, `test/interface_tests.jl`,
`test/refinement_tests.jl`, `test/driver_tests.jl`,
`test/sedov_tests.jl` and `test/kelvinhelmholtz_tests.jl`, included in that
order by `test/runtests.jl` — Sedov late, because the order is the
dependency order and the blast uses both the chunked driver and a static
`hydro_solve!` run, and Kelvin–Helmholtz last, being the only case whose
reference is a uniform fine run of this code rather than a closed form.
And from step 11 the viewers: `bin/Project.toml` (CairoMakie and SixelTerm
in an environment of their own, with a `[sources]` entry for TreeHydro —
it had one for TreeAMR too until that package was released),
`bin/backend.jl` (`resolvebackend`, `withbackend`, `checkprecision`, after
TreeWave's), `bin/visualize1d.jl` (the tracked tube against the exact
solution, per block coloured by level, with `τ` and the conserved totals
against time) and `bin/visualize2d.jl` (`--case=kh|sedov|both`: the
filmstrip with block outlines, and per case the two McNally diagnostics
against the uniform fine run or the radial scatter against the similarity
profile) — plus the one `src/` change the step needed, an `observer`
pass-through on `kh_run`.
`test/kelvinhelmholtz_tests.jl` gained the testset that asserts the
pass-through composes rather than replaces, which is the only thing in the
suite that exercises it — the other caller is `bin/`, and CI's `viewer` job
is what runs that. One workflow with two jobs, `CI.yml`'s `test` and
`viewer`, and a `README.md`.
The milestones are H0–H6 in `CODE.md`; H1 covered steps 1–4, H2 step 5,
H3a step 6, H3b step 7, H4a step 8, H4b step 9, H5a step 10 and H5b
step 11; step 7b split the suite
into a short tier and a
long one and step 7c undid the split, having found that what made CI slow
was code coverage under threads and not the runner; and `PLAN.md`'s step 12
(precision, H6a) is next.

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
No floor fires in any of these runs; the first that fire are Sedov's, in
step 9, and they are not the rule or the place the design expected.

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

Step 6 added the criterion and its calibration. Sod's initial data fires
in exactly the two cells straddling the diaphragm, at `τ = 0.9657` and
`0.9847`, and at **exactly zero** everywhere else; a synthetic atmosphere
six orders below the data with `O(1)` relative noise scores **0.0020**
with `ε_g = 1/1000` and **0.971** with `ε_g = 0`, which is the negative
control that pins the global floor term. Max `τ` on uniform meshes at
`h = 1/64 … 1/512`: a captured shock **0.5413, 0.5975, 0.5722, 0.5858**
(flat — it never resolves), the contact 0.19 / 0.22 / 0.18 / 0.13, the
rarefaction's head 0.1736 / 0.1050 / 0.0569 / 0.0286 (first order in `h`),
the smooth interior of the fan 0.0901 / 0.0400 / 0.0144 / 0.0048
(approaching second order), and the McNally density ramp 0.2535 / 0.1215
/ 0.0532 / 0.0210. The ramp picks the thresholds — **`refine_tol = 0.08`,
`coarsen_tol = 0.02`**, mid-plateau of the depth it then reaches — and
`ε = 1/100`, `ε_g = 1/1000` are the floors. All of it in `CODE.md`'s
"Step 6 — the refinement criterion".

Step 7 added the driver and the first mesh that moves. The tracked Sod
tube's L1 error against the exact solution is **1.0004** times the uniform
fine run's in `D = 1` and **1.0000** in `D = 2`, at 200 cells against 256
and 1472 against 2048, with the uniform coarse control at **3.678** and
**1.850**; reduced onto the common grid the tracked and fine runs differ
by `1.7e-5` where the coarse and fine runs differ by `1.1e-2`.
`tracking == 1.0` in both dimensions — every cell above `refine_tol` sat
on a block at the cap at every chunk — and the initial-data cycle
converges in 3 passes and 2. Through 9 and 3 mesh changes the mass and
energy drifts are `3.3e-16` / `1.6e-15` and `4.2e-17` / `1.9e-16`, and the
momentum equals its boundary flux to `8.6e-16` and `1.0e-16`;
`fixup = false` leaks `4.9e8`–`7.4e9` times more on **the same step count
and the same mesh history**. `speed_headroom = 1` throws in Sod's *first*
chunk (a CFL number of 0.6236 against 0.4); at 2 the run completes with
`λ` rising from 1.1832 to 2.2047. The buffer table is strictly ordered —
derived (6–7 cells) L1 4.540016e-3 at tracking 1.0, then 2: 1.0, 1:
0.9444, 0: 0.9048 — which is *neither* upstream finding. And `p = 1` on a
discontinuous solution is worse in every column (L1 4.701364e-3 against
4.540016e-3, tracking 0.9091, 216 cells against 200) with zero floor hits
either way, so the question stays open for Sedov. All of it in `CODE.md`'s
"Step 7 — the driver and the tracked shock tube".

Step 8 added the reset, and its two headline numbers are both zeros.
Applying it twice equals applying it once **bit for bit** at `Float64` and
`Float32` in `D = 1, 2, 3` — which `CODE.md` had predicted only to roundoff
— while the *flag* is not idempotent: a pressure-floored cell whose kinetic
energy dominates recovers a pressure a fraction of an ulp below `p_floor`
and re-fires (2 of 13 cells at `Float64` in `D = 1`, 85 of 426 in `D = 3`,
none at `Float32`), writing the identical bits back. And where nothing
fires the reset changes nothing at all: on the tracked tube under `:stage`
and `:step` and on the entropy wave, `injection == (0.0, 0.0, 0.0)`
exactly, `reset_hits == 0`, `ghost_hits == 0`, and the final state, drift,
error, step count and mesh history are bit-identical to the `reset = :none`
run's. One `SSPRK33` step on a half-vacuum box comes out at `ρ = ρ_atm`
under `:stage` and `:step` and stays at `ρ_atm/100` under `:none`, which is
what asserts the hook is wired at all. The whole suite costs **11 282 tests
in 1 m 42.9 at four threads** against 11 149 in 1 m 42.3 before the step,
and every one of the 75 `@info` lines it printed before is byte-identical.
All of it in `CODE.md`'s "Step 8 — the atmosphere reset".

Step 9 added the blast, and four of its findings correct the design.
`ξ₀(7/5, 3) = 1.0327774677614250` reproduces Taylor's 1.033 from a
parametrization derived here and checked against the one similarity
equation it was not built from (residual `3.6e-14`); the measured exponents
are **0.64146 / 0.50443 / 0.43766** against `2/3, 1/2, 2/5` and the peak
jumps **4.111 / 3.765 / 2.057** against the strong-shock 6. The tracked
mesh reproduces the uniform fine run **to roundoff** (`8.3e-15`) at 12544
cells against 16384, with `tracking == 1` everywhere and every drift at
roundoff — and, unlike Sod's, a boundary that contributes exactly nothing.
The corrections: **a tracked mesh cannot measure its own coarse-fine
faces**, since tracking puts the refined region's boundary ahead of the
shock, so `fixup = false`, `p = 1` and `reset = :step` all come back
*identical* to the run they control and every interface claim is made on
the static `sedov_forest(:center)` mesh instead (there the fixup buys
`3.2e12` in mass in `D = 2` and `8.5e7` in `D = 3`); **the atmosphere rule
never fires** — the bubble bottoms out at `ρ = 6.7e-2`, not `10⁻⁶` — and
what fires is the *pressure floor*, driven by the interface flux
restriction itself, 4096 owned cells and 40 ghost entries in `D = 2` and
24504 and 4703 in `D = 3`; **`p = 1` floors nothing there**, which buys
exact positivity for **0.47%** of L1 and closes the open question with the
opposite sign from Sod's row; and **the accumulated injection is a bound
under `:stage`** (measured ratio 0.520) and an equality under `:step`
(`4.4e-16` and `2.2e-16`), because SSPRK33's stages carry weights
`1/6, 2/3, 1`. The block count **rises monotonically** — the Sedov interior
is a steep ramp, so the refined region is a disk and not a shell. And the
M2 ordering case is finally exercised: zero mismatched entries out of 1664,
72000 and 59360 outward-facing ghost entries on a 2D corner, a 3D edge and
a 3D corner. All of it in `CODE.md`'s "Step 9 — the Sedov blast".

Step 10 added the shear layer, and its headline is a flux. The setup is
McNally, Lyra & Passy's **term for term** — the test re-evaluates equations
(1)–(5) from the paper's literals and the worst difference is *exactly
zero* — and the one transcription error was in `CODE.md`'s description of
the `M(t)` weighting, which read "the lower interface alone" and is in fact
mirrored over both. `M` grows from the seeded 0.0100 to **0.12346** at
`t = 1.5` at a fitted rate of **2.58036** over `2a ≤ M ≤ 6a`, below both
the `4.384` and the `5.9238` bounds, with the kinetic energy's rate
**2.1021** times it; the curve has **not saturated** by `t = 1.5` but is
decelerating, 3.349 → 2.580 → 1.844. Every one of the four integrals holds
at roundoff through three regrids, and `fixup = false` leaks by
`9.1e4`–`2.2e6` — this being the tracked mesh Sedov could not provide,
because the whole domain is in motion — while `S_y` alone does *not* leak,
being protected by the zero mean of `sin(4πx)` over the coarse-fine faces.
The cap sweep falls monotonically, L1 `5.98e-2 / 2.59e-2 / 4.24e-4` at
1024 / 4096 / 14848 cells against 16384 uniformly fine. **HLLE against
HLLC is not close**: `M(1.5)` on uniform meshes is `0.0116 / 0.0755 /
0.1240` under HLLC at 32²/64²/128² and `0.00066 / 0.0066 / 0.0357` under
HLLE, so **HLLC at half the linear resolution is ahead of HLLE at full
resolution**, and the growth *rates* are 2.5804 against 1.1712. HLLC
becomes this case's default (`kh_run`'s `riemann` keyword); the
package-wide default stays HLLE. `Float32` to `t = 2/5` reproduces the
mesh, the step count and the tracking exactly and `M(t)` to **156 ulp**.
All of it in `CODE.md`'s "Step 10 — Kelvin–Helmholtz".

`floors.jl` is included *before* `eos.jl`: `con2prim` takes a `Floors` and
says so in its signature, and a signature is evaluated where the method is
defined.

## Commands

One suite, run whole, at every thread count; `CODE.md`'s "Testing" has
the discipline and the measurement behind it. Every claim in "Measured
results" comes from a test that runs here, so this is what to run before
recording a number. About **3 m 44 at one thread and 2 m 38 at four**
(measured in step 10 on a quiet machine; it was 2 m 43 and
2 m 09 before the shear layer, which costs roughly a minute at one thread
and thirty seconds at four — twelve two-dimensional evolutions, four
of them 2700 steps on 128²-equivalent meshes, and the most parallel work
any one file holds, which is why the two thread counts diverge as much as
they do. The blast before it cost 50 s and 30 s). Every figure in this
paragraph **predates the `@inbounds` pass**, which took 15% off the suite
(4 m 23.9 against 3 m 44.8, measured back to back at one thread after step
11), so read them as an ordering rather than as a target. **This machine is
shared**, and runs taken while something else was on it came back at 4 m 10
and 3 m 12 — a fifth slower — so a timing is worth comparing only against
another taken under the same load. Step 11's one extra testset was measured
that way and not against the number above: **4 m 13.6 before and 4 m 31.9
after**, back to back at one thread on a machine that was also rendering
figures, 11610 tests against 11622.
`Pkg.test` does not inherit `-t`, so the thread
count has to be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

And the checked run, which is a *different* claim rather than a slower
version of the same one: the three RHS kernels carry `@inbounds`, so their
indices are an assertion, and this is the only thing that falsifies it.
Because it overrides `@inbounds` package-wide it never runs the code the
package actually ships — so it does not replace a plain run, which is the
only one that can catch a wrong answer. Run both before recording a
number. Measured at one thread: 3 m 44.8 plain, 5 m 19.9 checked, 11622
tests either way.

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

The same check **under the floor version**, before a step is merged. CI
runs Julia 1.10 — the floor since TreeAMR's release — as well as the
current release, and 1.10 is stricter in at least one way that matters
here: it refuses to redefine a `const`, which 1.12 and later allow, so a
suite that is green at 1.13 can be red at the floor (see "Things that will
bite"). With `juliaup`:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia +1.10 --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The floor costs **4 m 25.5 at one thread** against the release's 3 m 32.3,
measured back to back when the floor moved — so it is 1.25× the run you
have just done and not a different afternoon. Checked it costs **4 m 11.6**
there, which is *not* slower than its own plain run and is the one thing
about the move that inverted; see `CODE.md`. (It is also the *cheap* one
under coverage, which is the opposite way round and is why the
instrumented cell is the floor cell; see "Things that will bite".)

The viewers, in their own environment so that CairoMakie never becomes a
dependency of the package. The first call instantiates it; the
`[sources]` entry for TreeHydro means no manual `Pkg.develop`, and it is
also why `bin/` still needs **Julia 1.11** while the package itself runs
at 1.10:

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
buffer's margin is `speed_headroom · λ · chunk` at the cap's spacing, so
halve the chunk for each level added or the margin widens and more of the
box is refined. `--cap=5` at the default chunk throws out of
`refinement_buffer` naming the constraint. A non-default mesh writes its own
filenames, so a `--cap=3` render cannot overwrite what CI checks. PNGs land in `bin/output/`, which is
gitignored, and a terminal also gets them inline through SixelTerm — a
pipe does not, which is what `--no-display` makes explicit in CI. Roughly
26 s, 28 s and 53 s per figure, most of the first 20 s of each being
`using CairoMakie`. The `viewer` job in `CI.yml` runs all four renders on
every push and uploads them, because `bin/` is outside `src/` and `test/`
and nothing else would notice it breaking.

`--backend=metal --type=f32` renders too, and reproduces the host
`Float32` run exactly — measured in step 11 on this machine. That is the
plumbing working, not the device milestone: H6c still owes the per-phase
table and the opt-in device tests, and nothing was measured for speed.

Not yet real, and listed so the section can be filled in rather than
rewritten: the thread-independence test — which spawns its own
subprocess at another count either way — arrives in step 13, and device
tests, opt-in behind `TREEHYDRO_TEST_BACKEND` (`metal`, `cuda`) in an
environment of your own that has the device package, in step 14.
`bin/benchmark.jl` comes with `src/benchmark.jl` in step 14 as well.
Neither this package nor TreeAMR depends on a device package.

## Things that will bite

Carried over from TreeAMR and TreeWave where they apply here, plus what is
specific to a hydro code. Each is in `CODE.md` with its reason.

- **TreeAMR comes from the registry, not from the local checkout**
  (amended when TreeAMR 0.1.1 was released). `Project.toml` has no
  `[sources]` entry any more: the compat bound is `TreeAMR = "0.1.1"` and
  a clean checkout resolves it from General. So `~/src/jl/TreeAMR` is
  still *not* what the tests see, and the bar is now higher than it was —
  it used to be that an unpushed change there was invisible here, and now
  an **unreleased** one is. A TreeAMR change this package needs has to be
  tagged and registered before it can be used; say so rather than editing
  that checkout and assuming the tests see it. Dropping the entry is also
  what lowered the Julia floor from 1.11 to 1.10, `[sources]` being a 1.11
  feature. `bin/Project.toml` still has one, for *this* package, which is
  unregistered — so the viewer environment keeps 1.11 as its floor and
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
  integrator's `stage_limiter!` hook and in the driver after `regrid!`,
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
- **The SSPRK limiters are `solve` keywords, not constructor arguments**
  (amended in step 8). `SSPRK33(; stage_limiter! = f)` is what `CODE.md`
  was written against; `OrdinaryDiffEqCore` deprecated it in favour of
  `solve(prob, SSPRK33(); stage_limiter = f)`, it warns under
  `--depwarn=yes` (which `Pkg.test` passes), and once the deprecation
  completes the constructor's field would be **silently unread** — a
  positivity correction installed nowhere, reporting nothing. So
  `hydro_solve!` passes the keywords, and `test/reset_tests.jl` asserts
  that a step really does come out floored under `:stage` and `:step` and
  does not under `:none`, which is the only thing that would notice.
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
- **The accumulated injection is a bound under `:stage` and an equality
  under `:step`** (measured in step 9). `SSPRK33`'s three stage vectors
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
  floor cell is 1.10 now and behaves the same way). `driver_tests.jl`
  and `sedov_tests.jl` both defined `TRACKED_1D`, `TRACKED_2D`, `FINE_2D`,
  `COARSE_2D` and `NOFIX_2D`. Julia 1.12 and later quietly allow a `const`
  to be redefined, so every local run at 1.13 and the clean-checkout check
  passed; 1.11 throws `invalid redefinition of constant`, five minutes into
  otherwise green output. Name a file's shared runs with their case
  (`TRACKED_SEDOV_1D`, `TRACKED_KH`); `runtests.jl` now fails on any
  duplicate, from the source text, before anything is included; and a step
  that adds a test file runs once under the floor version before it is
  merged — the "Commands" section has the line. The floor is 1.10, and
  every Julia before 1.12 checks this.
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
  to 156 ulp of `Float32`; and MultiFloats cannot
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
  (`version: "1.11"` when it was measured, `version: "1.10"` since).
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
  whichever version is the floor. With no `VERSION` check anywhere in
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
- No generated file is tracked. Everything in the tree is written by hand
  and reviewed as such; `test/references/*.toml`, which step 7b generated
  and step 7c removed, were the one exception and are gone. `bin/output/`
  is gitignored and the viewers write PNGs there.
- **There is one TreeAMR pin now and still two Manifests** (amended when
  TreeAMR 0.1.1 was released). Both environments resolve TreeAMR from the
  registry, so the two `rev = "main"` entries that had to be kept in step
  are gone; what is left is a compat bound in each `Project.toml`, and
  `bin/` still has its own `Manifest.toml` (gitignored, like the root's),
  so `Pkg.update("TreeAMR")` at the root does not touch it. The viewers
  can therefore still resolve a *different* TreeAMR from the tests — an
  older one, if `bin/`'s Manifest is stale — which is what the `viewer`
  job in CI exists to catch. Grep for `TreeAMR =` across both files rather
  than editing from memory.
- **`bin/backend.jl` is `include`d by both viewers**, and step 14 will add
  a third script that runs against the *package* environment instead. So it
  may use only what both environments have, which today is
  `KernelAbstractions` — which is why that is a dependency of
  `bin/Project.toml` at all.
- One workflow with two jobs: `.github/workflows/CI.yml`. `test` runs the
  whole suite on every
  push that touches something other than Markdown, over **four cells
  spelled out one at a time** rather than a product — 1.10 on macOS (the
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
