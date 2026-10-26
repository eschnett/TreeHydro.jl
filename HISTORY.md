# TreeHydro.jl — History

How the package got where it is: who decided what and when, the
alternatives that were tried or rejected and why, measurements of code
that no longer exists or that later measurements replaced, upstream bugs
since fixed, and the milestone records. The current design, with the
reasons for it, is in [CODE.md](CODE.md). The sections follow `CODE.md`'s,
with the milestones first.

**Step numbers.** A step number — "step 9", "step 7b" — names a step of
the plan the package was built by, `PLAN.md`, deleted in step 15; the steps
and the milestones H0–H7 they belong to are listed under [Milestones and
steps](#milestones-and-steps).

**Contents**

- [Milestones and steps](#milestones-and-steps)
  - [H0 — Scaffolding and prerequisites](#h0--scaffolding-and-prerequisites)
  - [H1 — The scheme on a uniform mesh](#h1--the-scheme-on-a-uniform-mesh)
  - [H2 — Coarse-fine faces, static mesh](#h2--coarse-fine-faces-static-mesh)
  - [H3 — Regridding](#h3--regridding)
  - [H4 — Sedov](#h4--sedov)
  - [H5 — Kelvin–Helmholtz](#h5--kelvinhelmholtz)
  - [H6 — Precision, threads, device](#h6--precision-threads-device)
  - [H7 — Higher-order reconstruction](#h7--higher-order-reconstruction)
- [Goals and scope](#goals-and-scope)
- [Upstream prerequisites](#upstream-prerequisites)
- [The equations](#the-equations)
- [The scheme](#the-scheme)
  - [Field sets](#field-sets)
  - [Reconstruction](#reconstruction)
  - [Riemann solver](#riemann-solver)
  - [Time integration and the time step](#time-integration-and-the-time-step)
  - [Floors and the atmosphere](#floors-and-the-atmosphere)
  - [What has a GRMHD counterpart](#what-has-a-grmhd-counterpart)
- [The right-hand side](#the-right-hand-side)
- [Conservation at coarse-fine faces](#conservation-at-coarse-fine-faces)
- [Operator order](#operator-order)
- [Boundaries](#boundaries)
- [The cases](#the-cases)
  - [Entropy wave](#entropy-wave)
  - [Sod shock tube](#sod-shock-tube)
  - [Sedov blast wave](#sedov-blast-wave)
  - [Kelvin–Helmholtz instability](#kelvinhelmholtz-instability)
- [The refinement criterion](#the-refinement-criterion)
- [Regridding: one driver, restart per chunk](#regridding-one-driver-restart-per-chunk)
- [Checkpoint and restart](#checkpoint-and-restart)
- [Precision](#precision)
- [Multi-threading](#multi-threading)
- [Running on a device](#running-on-a-device)
- [Running distributed](#running-distributed)
- [Dependencies and the Julia floor](#dependencies-and-the-julia-floor)
  - [TreeAMR's release and the 1.10 floor](#treeamrs-release-and-the-110-floor)
  - [The 1.11 floor again (2026-09-25)](#the-111-floor-again-2026-09-25)
  - [The 1.13 floor (2026-10-16)](#the-113-floor-2026-10-16)
  - [IMEXRungeKutta and TreeAMR 0.1.3](#imexrungekutta-and-treeamr-013)
- [Testing and CI](#testing-and-ci)
  - [Suite timings](#suite-timings)
- [Measured results replaced](#measured-results-replaced)
- [Possible extensions](#possible-extensions)
- [Open questions](#open-questions)

## Milestones and steps

The package was built in milestones H0–H7, each with an acceptance test,
serial `Float64` correctness first, and the milestones in numbered steps
from a step plan, `PLAN.md`, committed beside `CODE.md` until step 15
deleted it with the last milestone done. H0 was step 0; H1 steps 1–4; H2
step 5; H3 steps 6 and 7; H4 steps 8 and 9; H5 steps 10 and 11; and H6
steps 13, 14 and 12, in that order, taken together with the move to
**TreeAMR 0.1.3** (owner-based threading, from the registry) and to
**IMEXRungeKutta**, which replaced OrdinaryDiffEq for time integration.
Steps 7b and 7c changed CI (see [Testing and CI](#testing-and-ci)), and
step 10b came with the merge of steps 8–10; step 15 was the review pass that read `CODE.md`
against the code once more and closed the step plan. Two of the plan's
ground rules — convergence rates, conservation drifts and mesh statistics
are asserted as numbers with tolerances, and an existing assertion is never
loosened to get green — were carried over into `CLAUDE.md`'s conventions
when step 15 deleted it. H0 to H6 are all done,
each on its acceptance or on an amendment to it recorded in its own entry;
H7 is optional and not planned (see "Possible extensions" in `CODE.md`).

H0 is the scaffolding; H1 the scheme on a uniform mesh (the equation of
state and floors, the reconstruction, the three Riemann fluxes, the
six-step right-hand side, the entropy wave, the exact Riemann solver and
Sod's tube); H2 the coarse-fine faces on a static mesh; H3 the refinement
criterion and the one evolve-and-regrid driver; H4 the atmosphere reset
and the Sedov blast; H5 Kelvin–Helmholtz and the viewers; and H6 threads,
the benchmark and the device, and precision.

The markers `CODE.md` used while the milestones ran: **(decided)** was a
decision taken in review; **(proposed)** was one the document made and
wanted confirmed; **(predicted)** was a number a milestone was to measure
and "Measured results" then recorded, as TreeAMR's and TreeWave's
`CODE.md` did; **(open)** pointed at "Open questions". Step 15 found none
of the three outstanding: every **(proposed)** had been confirmed or
amended in place, every **(predicted)** had its measured number beside it,
and no question was left open. The markers stood where they were, next to
what settled them, as the record of what was expected, until this file
took that record over; what each said is recorded under its topic below.

After step 15, on TreeAMR's releases: checkpoint and restart
(2026-09-29), reflecting walls and the zoom showcase (2026-09-29), runs
distributed over MPI (2026-10-02, on TreeAMR 0.1.6), TreeAMR 0.2, whose
checkpoints are the companion package TreeIOHDF5's (2026-10-09), and the
Julia floor raised to 1.13 (2026-10-16). Each is under its topic below.

### H0 — Scaffolding and prerequisites

Done. The two TreeAMR prerequisites landed on TreeAMR's `main`;
`Project.toml` with the `[sources]` pin to `main`, CI on 1.11 and release
at one and four threads, `CLAUDE.md`, the design document. *Accept:* a
clean clone instantiates, an empty test suite passes, and the pinned
TreeAMR provides both prerequisites. What H0 settled beyond that — the
suite is not empty, `hostcopy` is split — is under "Upstream
prerequisites" and "Running on a device" in `CODE.md`; its compat bounds
are under [Dependencies and the Julia
floor](#dependencies-and-the-julia-floor).

### H1 — The scheme on a uniform mesh

Done. EOS, `con2prim`, MUSCL with the three limiters, LLF, HLLE and HLLC,
SSPRK33, the six-step RHS with `D` flux sets, the exact Riemann solver and
the Dirichlet boundary hook — on a single-level forest, `D = 1, 2` (3D
smoke). *Accept:* the entropy wave converges at second order in L1 and L∞
with `:none`, and in L1 with `:mc` (amended in step 3: the first draft said
"near it" in both norms under a limiter, and the measurement says otherwise
— `:mc` clips the smooth extrema and its L∞ rate is 1.35); Sod against the
exact Riemann solution at an L1 rate in `[0.8, 1.0]`; direction
independence bit for bit; every conserved integral constant to roundoff
(every face is a same-level face, with or without the fixup — the
control).

What H1 measured, across steps 1–4, is under "Step 3" and "Step 4" in
`CODE.md`'s "Measured results", all of it identical at one and at four
threads; the `:mc` prediction was wrong and was amended rather than met.
The pieces H1 wrote but did not measure: HLLC, which is step 10's
comparison; the two-level `hydro_forest`, which is H2's; and the boundary
hook's other two call sites, `regrid!` and `adapt_to_initial_data!`, which
are H3's.

### H2 — Coarse-fine faces, static mesh

Done. The two-level `hydro_forest`, the two two-level `sod_forest`
configurations, the fixup, the boundary hook on a fine block. *Accept:*
conservation of all `D + 2` integrals to roundoff with the fixup and a leak
without, in `D = 1, 2, 3`, on the entropy wave and on Sod; the
interface-order table for the system (predicted L∞ 1, 2, 2 at
`p = 1, 3, 5`; L1 2, 2, 2); a two-level Sod whose refined region touches
the Dirichlet boundary.

What H2 measured is under "Step 5" in `CODE.md`'s "Measured results", all
of it identical at one and at four threads, the interface-order rule
predicted before it was run. Step 5 added almost no code — `sod_forest`'s
`refined` option and a one-line `forest_levels` — and the whole of it is
`test/interface_tests.jl`. What H2 wrote but did not decide: the `p = 1`
question, which the smooth wave cannot answer (it confirms the premise — an
integral norm does not see the interface defect — and leaves the answer to
the discontinuous cases of steps 7 and 9).

### H3 — Regridding

Done. The criterion, the buffer, `evolve!`, the initial-data cycle.
*Accept:* the cycle converges to a fixed hierarchy on the initial data it
is given — Sod's and the entropy wave's here; Sedov's and
Kelvin–Helmholtz's arrive with their cases in H4 and H5, and run through
this same cycle; the tracked Sod tube in `D = 1, 2` matches the uniformly
fine reference at fewer cells with the uniform coarse mesh as control,
conserves to roundoff through the regrids, and leaks without the fixup; the
`p = 1` against `p = 3` table for Sod; the buffer-width table.

What step 6 measured is under "Step 6" in `CODE.md`'s "Measured results",
with two predictions amended (a captured shock scores 0.57 rather than the
≈ 1 of a sharp jump; a kink does *not* fire forever). What step 7 measured,
the first step whose mesh *moves*, is under "Step 7": it closed the step-4
amendment on the headroom, and gave the first discontinuous row of the
`p = 1` question, against `p = 1` — which is exactly why Sod could not
close the question and Sedov had to. (It did, in step 9, and with the
opposite sign: on a strong shock crossing a coarse-fine face `p = 1` floors
nothing where `p = 3` floored 4096 owned cells, for 0.47% of L1. The two
rows measure different things — whether a prolongation ever spans a shock
— and `p = 3` stays the default.) What step 7 wrote but did not exercise:
every case but Sod and the entropy wave. (The atmosphere reset, which was
the other item on that list, landed in step 8.)

### H4 — Sedov

Done. Floors and the atmosphere reset, the `D = 2` and `D = 3` blasts, the
similarity checks, the hook on edges and corners. *Accept:* the exponent
and the jump; the reset idempotent and `U`/`P` consistent after it; the
injection measured and the drift net of it at roundoff, `:stage` against
`:step`; floor counts by population; the shell tracked to the uniform fine
reference at fewer cells; block count rising then falling behind the
shock; the `p = 1` / `p = 3` comparison and the ghost-floor count that
decides the upstream question. The 3D adaptive run at test size. (The
reset, its accounting and both floor counts landed in step 8 and were
measured there on cases where nothing fires; what was left for this
milestone was the blast itself, which is the case where they do.)

What step 9 measured is under "Step 9" in `CODE.md`'s "Measured results",
all of it identical at one and at four threads; its sharpest finding was
that a tracked mesh cannot measure its own coarse-fine faces, and its
floor counts and its injection-as-a-bound were taken under OrdinaryDiffEq
(see [Floors and the atmosphere](#floors-and-the-atmosphere)). What the
acceptance list got wrong, and it is one item: **the block count does not
rise and then fall**. It rises monotonically, because the Sedov interior is
a steep density ramp rather than a flat bubble and a Löhner indicator on
`ρ` correctly fires throughout it; the refined region is a growing disk.
The hollow opens only at the very centre and only late. The milestone is
marked done on the strength of what that item was *for* — a mesh that
follows a closed expanding surface, at a real saving — which is met; a
criterion that would produce a shell is listed under "Possible extensions"
in `CODE.md`.

### H5 — Kelvin–Helmholtz

Done. The McNally setup, `M(t)` and the kinetic energy diagnostic, HLLC,
the viewer. *Accept:* `M(t)` grows below the incompressible bound and
converges toward the uniform fine run as the cap rises; conservation
through regrids; HLLE against HLLC measured and the default chosen; the
filmstrip rendered in CI.

**Step 10 did the physics; the picture was what was left.** Done in step
10: the setup checked against the paper term for term, `mode_amplitude` in
the area-weighted form, `max_y_kinetic_energy`, `growth_rate` and its
window, the case and `kh_run`/`kh_uniform`, the measured growth rate
2.58036 below both the `4.384` and the `5.9238` bounds, conservation to
roundoff through the regrids with a five-to-six-order leak without the
fixup, the cap sweep converging onto the uniform fine run, the `Float32`
claim, and the HLLE/HLLC measurement with HLLC chosen as this case's
default. Not done when step 10 closed: the **viewer and the filmstrip in
CI**. **Step 11 built it and H5 is done.** `bin/visualize2d.jl --case=kh`,
`bin/Project.toml` and the `viewer` job in `CI.yml` render the filmstrip,
the two diagnostics against the uniform fine run, and the block count, on
every push. The viewer adds no time stepping of its own: it reads
`kh_run`'s observer through the pass-through step 11 gave it. The last
clause of the acceptance — "the filmstrip rendered in CI" — was **closed on
the first run of the job**, PR #1: the `viewer` job passed in 11 m 43, and
the `kh_2d.png` it uploaded carries the same 232 blocks, the same
`M(1.5) = 0.12346` and the same fitted 2.58036 as the local render.

### H6 — Precision, threads, device

Done. `T` and `backend` on every driver, the type table, the thread
workload, device tests, the benchmark. *Accept:* `Float32` reproduces the
Sod and Sedov meshes and floor counts; `Float32x2` runs Sod and Sedov;
digests identical across thread counts; the per-phase device table with the
RHS speedup on unified memory measured. Threads and device were done first,
as steps 13 and 14, ahead of precision: digests identical at one and four
threads, the device table on Metal and an H200, and scans on Symmetry.
**Precision was done in step 12, and H6 with it:** `Float32` and
`Float32x2` reproduce the Sod and Sedov meshes and floor counts exactly,
with errors inside the predicted 1% by a factor of eight or more. The
records are under "Step 13", "Step 14" and "Step 12" in `CODE.md`'s
"Measured results".

### H7 — Higher-order reconstruction

Optional, not planned and not started: the package is complete at H6. It
is under "Possible extensions" in `CODE.md`.

## Goals and scope

The goal of running in the caller's type, on the caller's backend and at
any thread count with bit-identical results gained its one qualification —
floating-point sums promised to roundoff only — after step 11, when TreeAMR
narrowed its M5 guarantee.

The scope first said:

> **Not a production code.** No checkpointing, no HDF5, no unit system.
> Output is what the viewer in `bin/` needs.

Amended 2026-09-29: checkpointing through TreeAMR 0.1.4's M9a, `evolve!`
writing checkpoints and restarting from them through TreeAMR's
`save_checkpoint` and `load_checkpoint`. Those two functions became
TreeIOHDF5's with TreeAMR 0.2, and TreeIOHDF5 a dependency of this package
on 2026-10-09 (see [Checkpoint and restart](#checkpoint-and-restart)).

It also said **"Dirichlet physical boundaries only"**, and of the others:
"Reflecting and outflow conditions read the interior and are CPU-only
upstream; nothing here needs them." Amended 2026-09-29: that was true of
TreeAMR's region-form hook and stale about reflection, which TreeAMR has
had since 0.1.3 as a first-class face; the package wired it through that
day, for the Kelvin–Helmholtz half box.

## Upstream prerequisites

Both prerequisites were decided in review, before H1, and landed on
TreeAMR's `main` in H0. `m8` was then merged, so both are in TreeAMR's
releases. Amended when TreeAMR 0.1.1 was released: the `[sources]` entry
that used to pin `main` went, and the compat bound became
`TreeAMR = "0.1.1"`; step 15 recorded the bound as `TreeAMR = "0.1.3"`, the
release with the owner-based threading. The later bounds are under
[Dependencies and the Julia floor](#dependencies-and-the-julia-floor).

## The equations

The variable order was decided in review and settled in step 1, which
implemented `src/eos.jl` and `src/floors.jl`; what writing them settled is
the list under "The equations" in `CODE.md`.

## The scheme

### Field sets

The three kinds of field set were decided in review. The table first gave
`P` `nvars = D+2`; step 3 amended it to `D+4`, adding the two diagnostic
slots. The paragraph on them first said the ghost-cell floor count "arrives
with the atmosphere reset"; it arrived in step 8, as `ghost_floor_hits`,
and its zero on the tracked shock tube and the entropy wave was measured
there.

The paragraph after it first said "deterministic upstream". After step 11,
when TreeAMR narrowed its M5 guarantee, that became two statements, only
one of them still a guarantee. The floor count was closed with
`round(Int, …)` until step 12, which MultiFloats does not provide; it is
`roundint` since. The same paragraph then said the totals "should stop
combining by hand": TreeAMR planned a global scalar form,
`mesh_mapreduce(f, op, init, fs[, u]; vars, weight)` — on one process
first, with the M7 `Allreduce` inside it later — and once it existed the
totals were to go through it, the field-set form already doing so through
`total_mass`. Step 15 recorded that it exists: TreeAMR 0.1.3 exports
`mesh_mapreduce` in both forms and its `total_mass` is that call, while the
state-vector form of `conserved_totals` and `conserved_scales` still
multiplied and summed their per-block partials by hand. After step 15 both
became that call too, and not one bit moved (the measurement is under
[Multi-threading](#multi-threading)).

That `P` holds primitives in its ghost cells was decided in review, and so
was option (a), "with the understanding that it may change" on the ghost
floor count. Step 9 measured that count and it does not change it. The
`con2prim` pass over the stored extent was, when this was written, "one
thing `map_blocks!` cannot launch today: it covers the owned or the closed
range, never the ghosts"; `map_blocks!(…; stored = true)` was decided as
the upstream prerequisite that changed that.

### Reconstruction

Primitive reconstruction, rather than conserved, was decided in review.
Step 2 implemented `src/reconstruction.jl`.

### Riemann solver

HLLE as the default, and HLLC as *the comparison, not the baseline*, were
decided in review: "the Kelvin–Helmholtz case is where HLLE's contact
diffusion shows, so HLLC is added at that milestone and measured against
HLLE there". Step 2 implemented `src/riemann.jl`, HLLC beside the other two,
to be measured in H5. Step 10 measured it on the shear layer, and HLLC
became the Kelvin–Helmholtz case's default while the package-wide default
stayed HLLE. The criterion that decision was made by was written down
before the HLLE runs; how two of its three clauses failed to decide is
under "Riemann solver" in `CODE.md`.

### Time integration and the time step

**The move to IMEXRungeKutta** came after step 11: the Tree* packages
moved to it for every time integration, explicit ones included — TreeAMR's
own tests and TreeGeneralizedHarmonic first — and this package followed,
replacing OrdinaryDiffEq's `SSPRK33`. Three things changed, and one of
them — where a stage limiter's correction goes — was a design change. The
by-owner stage arithmetic uses the placement TreeAMR 0.1.3 gives every
per-block pass; it is the Amdahl term this section used to say was "not
TreeHydro's to fix", fixed without a hand-written integrator: "The Amdahl
term TreeWave measured — the integrator's serial stage arithmetic capping a
threaded step at 3.6× — applied here unchanged until the move to
IMEXRungeKutta." The scratch hand-over is IMEXRungeKutta 1.2's `reuse`.
Every number the suite prints moved at roundoff (Shu–Osher and Butcher form
associate the same sums differently) and none moved beyond it except the
ones that depend on the limiter placement; both are recorded under
[IMEXRungeKutta and TreeAMR 0.1.3](#imexrungekutta-and-treeamr-013).

**`λ_max` once per chunk** was proposed, and confirmed in steps 4 and 7,
beside a headroom factor and a recheck that throws. The design first said:

> For the three test problems the speed is constant (Sod), nearly constant
> (Kelvin–Helmholtz) or decreasing (Sedov, after the deposition), so the
> per-chunk value is a bound in practice; the check is what makes that a
> fact rather than a hope.

Step 4 amended it — "constant for Sod" is wrong, and the driver needs a
headroom factor beside the recheck — and drew "two consequences, one taken
now and one recorded for step 7", adding: "The driver is not designed here;
the fact and the number are recorded so that it is not designed without
them." Step 7 implemented the headroom and closed that amendment. Step 9
amended the Sedov clause: "decreasing after the deposition" is right, and
the first chunk is not covered by it.

### Floors and the atmosphere

The two rules were decided in review, and so was option 1, the reset in the
integrator's own stage hook. It was written for OrdinaryDiffEq:

> Every SSPRK method in `OrdinaryDiffEqSSPRK` takes
> `stage_limiter!(u, integrator, p, t)` and
> `step_limiter!(u, integrator, p, t)`, called on each stage vector after
> it is formed and on the step's result — the hooks that exist for
> positivity-preserving limiters, which is what this is.

Step 8 amended it — the hooks had become `solve` keywords, not constructor
arguments:

> The signature and the semantics are exactly as above, but
> `OrdinaryDiffEqCore` has moved the two limiters into the solver options:
> `SSPRK33(; stage_limiter! = f)` is deprecated in favour of
> `solve(prob, SSPRK33(); stage_limiter = f)`, warns on every run under
> `--depwarn=yes` — which is what `Pkg.test` passes — and would, once the
> deprecation is completed, leave the constructor's field silently unread.
> A positivity correction that is installed nowhere and reports nothing is
> the one failure mode this must not have, so `hydro_solve!` passes the
> keywords to `solve` and `test/reset_tests.jl` asserts that a step really
> does come out floored under `:stage` and under `:step` and does not under
> `:none`. Read against the installed `OrdinaryDiffEqSSPRK`:
> `stage_limiter!` is applied to every stage of `SSPRK33` *including the
> last*, and `step_limiter!` once to the accepted step's result, which is
> what makes `:step` the cheaper cadence of the same thing rather than a
> different correction.

After step 11 the hooks became IMEXRungeKutta's, the same two keywords
with the same signature as `init` keywords, and `:stage` became both of
them — the same three calls per step as before, `:step` the step limiter
alone as before. For a reset that only ever fires on the pressure floor at
a coarse-fine face, as here, the move moved the static Sedov blast's L1 in
the eighth digit and its reset count by 1% (4096 → 4136).

The consistency claims were first predicted as:

> **`U` and `P` agree after a reset**: `con2prim` of the reset `U`
> reproduces the floored `P` to roundoff, and applying the reset twice
> equals applying it once to roundoff (an idempotence test; bit-for-bit
> would need the floor comparison to absorb the `prim2con`/`con2prim`
> round trip, and is not claimed).

Step 8 measured them and corrected the prediction in both directions: the
state is idempotent bit for bit, and the flag is not. Step 8 implemented
the reset, its accounting and the ghost floor count, and measured that
where nothing fires the injection is exactly zero. Step 15, after the move
to IMEXRungeKutta, recorded that only the step limiter measures an
injection. The note that the ghost count's block order is this package's
own discipline dates from after step 11, when TreeAMR narrowed its M5
guarantee to everything but floating-point sums; the count was closed with
`round(Int, …)` until step 12.

Step 9 measured the case where the floors finally fire, and the design was
wrong about which rule fires and about where. Under OrdinaryDiffEq the
static two-level mesh reset **4096** owned cells and **40** ghost entries
in `D = 2`, **24504** and **4703** in `D = 3`. And the accumulated
injection was not an equality under both cadences:

> It was an upper bound under `:stage` and an equality under `:step`. The
> accounting adds the raw `Σ hᴰ ΔU` of every call, but `SSPRK33`'s three
> stage vectors enter the step's result with weights `1/6`, `2/3` and `1`,
> so an injection into a stage reaches the state scaled by that stage's
> weight. Measured ratio of drift to accumulated injection: **0.51984** in
> `D = 2` and **0.52203** in `D = 3`. […] Step 8 could not see this,
> because on Sod and the entropy wave the injection was exactly zero and
> every weighting of zero is zero. The fix is not to weight the accounting
> — the hook is not told which stage it is in, and a `DiscreteCallback`
> would not be either — but to say what the number is: a bound with
> `:stage`, an equality with `:step`.

That finding is what made the accounting's semantics explicit. So, under
OrdinaryDiffEq, `:stage` bought three times the repairs (4096 against 1384
owned cells in `D = 2`, 24504 against 8232 in `D = 3`) for the same final
state to roundoff, and a strictly less informative injection; the
measurement recorded then was that on a blast the cheaper cadence is
indistinguishable in the answer "and better in the bookkeeping".
IMEXRungeKutta's Butcher form, after step 11, made the `:stage` injection
exact: the stage limiter's correction no longer reaches a total, so it is
no longer counted.

### What has a GRMHD counterpart

HLLC was implemented in step 2 and has been the Kelvin–Helmholtz default
since step 10. The reflecting face with a parity per variable was added on
2026-09-29.

## The right-hand side

Step 3 implemented `hydro_rhs!` in `src/evolution.jl`. Step 8 added the
tenth keyword of `HydroProblem`, `accounting`, which step 15 recorded. Step
3 also settled:

> **The boundary hook is passed through** `fill_ghosts!(U, schedule;
> boundary)` from step 3 on, although the only case here is periodic and
> hands it `nothing`: the hook is the part of the signature a Dirichlet
> case needs, and adding it in step 4 would mean touching the right-hand
> side again for it.

## Conservation at coarse-fine faces

The static two-level conservation was measured in step 5 and the uniform
control in step 3. Step 5 noted: "What step 5 does *not* cover, because it
has no driver yet, is a refined region that *follows* a shock with regrids
in between; that is step 7's". Step 7 measured the moving mesh. That the
momentum's scale can be zero too was an amendment of step 5.

## Operator order

The interface-order rule for the system was predicted, and measured in step
5 without amendment. The `p = 1` question was open, with **"Default `p = 3`
(decided), with `p = 1` measured beside it on all three shock cases and the
table recorded here."** — and the upstream request "is not made in advance
of the measurement". Step 5's entropy wave decided nothing ("and why the
question stays open"): "the question … is Sod in step 7 and Sedov in step
9, run at `p = 1` beside `p = 3` with the floor counts beside the errors.
Until then `p = 3` remains the default and the upstream request remains
unmade." Step 7 measured the first discontinuous row — "That last clause is
why the question is not closed here" — and Sedov "completes the table in
step 9". Step 9 closed the question, and `p = 3` stayed the default
(decided, unchanged).

Step 9's static-mesh row for `p = 3` was taken under OrdinaryDiffEq:

| `p` | L1 against the uniform fine run | reset hits | ghost hits | energy drift |
|---|---|---|---|---|
| 3 | **4.406604e-2** | 4096 | 40 | 1.24659e-5 |

Since IMEXRungeKutta it is 4.406604e-2, 4136, 40 and 1.24609e-5 — the reset
count moves because `:stage` now also resets the step's result, and L1
moves in the eighth digit; `p = 1` floors nothing and is unchanged to
roundoff.

## Boundaries

"Dirichlet wherever the physics does not require periodicity; periodic
only where it does" was decided in review, reversing the first draft,
which reached for periodic boundaries wherever the problem allowed them
because they cost no application code; two arguments overturned that
default. The Kelvin–Helmholtz half box and reflecting faces were added on
2026-09-29.

Step 3 put the `boundary` keyword in `HydroProblem`'s signature for the
shock tube, which step 4 implemented as the first downstream use of the
physical-boundary path. Step 5 implemented the Sod configuration with a
refined region touching the Dirichlet face (`refined = :left`). Step 7
measured the hook's other two call sites, `regrid!` and
`adapt_to_initial_data!`, wired in `evolve!` as of that step, and noted:
"What is still not exercised is the M2 ordering case in full, which needs
two physical faces meeting; that is Sedov's corner, in step 9." Step 9
implemented it.

`CLAUDE.md`'s note that `evolve!` does not check the arrival of a wave at a
Dirichlet boundary — the case does — was settled in step 7, amending the
first draft of that note.

## The cases

The cases table's Sedov row first said the refined region follows a closed
expanding *shell*; step 9 amended it to a growing disk. The all-variables
form of the coordinate callbacks was decided as an upstream prerequisite
while TreeAMR's callbacks were "today called once *per variable*". Point
samples at cell centers were decided in step 4, the first case to use them,
and generalized in step 7, where a case states its data pointwise because
the adaptation cycle changes `h` under it.

### Entropy wave

The design first said the entropy wave is where "the HLLE/HLLC comparison
has its first number". Step 15 recorded that it never did: the comparison
was made on the shear layer in step 10. Step 3 implemented the case and
measured it on the uniform mesh; step 5 ran the same study on the M3
two-level hierarchy.

### Sod shock tube

Step 4 implemented `src/exact_riemann.jl` and `src/sod.jl` and measured the
L1 rate (0.903), the direction independence (both halves hold, with no
differing entry anywhere) and the boundary flux on the uniform mesh. Step 5
gave `sod_forest` its `refined` option, the two static two-level
configurations; step 7 the tracked tube, measuring conservation through
the regrids and the match against the uniformly fine reference, and the
throw at `speed_headroom = 1` that turned the step-4 amendment into a
parameter.

### Sedov blast wave

The profile as an extension and not a milestone was decided in review;
step 11 drew it (`bin/visualize2d.jl --case=sedov`). The design's list of
what the case measures first included:

> **A refined shell that grows while the interior coarsens** — the Sedov
> mesh problem TreeWave could only imitate with the wave equation. Block
> count and coverage of the shell against time.

and the step plan said "block count rising then falling behind the shock"
(that plan was `PLAN.md`, deleted in step 15). Both assumed the evacuated
bubble is flat; step 9 measured a growing disk, monotone in block count.
The mesh still followed the blast and still saved cells, which is what the
milestone was for; what was corrected is the shape.

The design also planned that "the test runs it small and short (`N = 8`,
few roots, cap 1, a few chunks), and `bin/` runs it at demo size", and said
of the early phase that "it is *also* where `λ_max` only decreases, so the
per-chunk `dt` is a bound. Both facts are recorded here so that the first
chunk being slow is expected rather than investigated." Step 9 found that
right about the blast and wrong about the first chunk, for the same reason
it was wrong on Sod; step 15 recorded the test's real size (`N = 4`,
`roots = 4`, cap 1 over 15 chunks) and that `bin/` does not run `D = 3`.

Step 9 implemented `src/sedov_reference.jl` and `src/sedov.jl`, and
`sedov_static` replaced the planned `uniform_sedov`. It also measured why
`shock_radius` is a host loop and not the `firing_boxes` sweep the design
first reached for, and found that `V − V₀` must never be formed in the
similarity code.

### Kelvin–Helmholtz instability

The McNally setup was decided in review. Its formulas were transcribed from
memory, and step 10 checked them against the paper: the profiles came out
term for term, and the check found exactly one error, in the weighting of
`M(t)`. The first draft said `e^{−4π|y − ¼|}` "so that the lower interface
alone is read"; equations (6)–(8) use that for `y < ½` and
`e^{−4π|(1−y) − ¼|}` for `y ≥ ½`, so both interfaces are read, mirrored —
corrected in step 10, which also added the area-weighted form (14)–(17)
that an adaptive mesh needs (decided in step 10). Step 6 had used the
ramp's shape for its calibration before that check; nothing there depended
on the transcription being faithful, and the one error was not in the
profile in any case, so the calibration table stood unamended.

The design predicted for **HLLE against HLLC**: "at the same resolution
HLLC's rolls are visibly sharper and its `M(t)` closer to the fine
reference, because the shear layer *is* a contact. This is the measurement
that decides whether HLLC becomes the default or stays a switch." Step 10
measured it: HLLC wins by a factor of two in linear resolution, and it
became this case's default.

The picture was implemented in step 11 as `bin/visualize2d.jl --case=kh`,
exactly as described, through the `observer` pass-through step 11 added to
`kh_run`. `--movie` was added after step 11 — not part of what H5
accepted, and recorded as an addition rather than folded in: the design
asked for four frames and four frames is what H5 was marked done on.

What step 10 decided beside the flux — `speed_headroom = 1`, `chunk =
1/200` and the growth window — is under "Kelvin–Helmholtz instability" in
`CODE.md`; that `speed_headroom = 1` holds at the calibrated cap and not at
deeper ones was found after step 11, running the shear layer at `--cap=4`.
The half box was added on 2026-09-29.

## The refinement criterion

Step 6 implemented `src/refinement.jl` and calibrated the thresholds; where
the prediction needed amending — a captured shock scores 0.57 rather than
≈ 1, and a kink does not score `O(1)` forever (amended in step 6) — is
under "The refinement criterion" in `CODE.md`. Step 7 measured the
criterion driving a real `regrid!` and the buffer against a real
`λ_max · chunk`.

## Regridding: one driver, restart per chunk

The one time-stepping loop was decided in review. Step 7 implemented
`src/driver.jl`; the headroom as a case parameter closed the step-4
amendment. The design's loop sketch was amended in step 15 to match
`evolve!`: the step carries the case's `speed_headroom` since step 7, and
the loop runs over `chunk_count(t_end, chunk)` chunks since step 12 — it
was `while t < t_end` before. Amended 2026-10-01: the checkpoint is written
after the observer and before the regrid, and the regrid is one function,
`regrid_chunk!`, which a restart calls first. The viewers that get the
initial frame from the observer were step 11's.

## Checkpoint and restart

Added on 2026-09-29, on TreeAMR 0.1.4, whose M9a adds `save_checkpoint`
and `load_checkpoint` through a package extension on HDF5. The plan was
decided with Erik the same day, keyword names included; what the
implementation settled was marked as such: that reals are stored exactly;
that the file just written survives the rotation even when it is not the
newest (a run restarted from an older checkpoint while newer ones exist);
and that `checkpoint_interval_seconds = 0` writes at every boundary, the
plan having said "positive". The names, the rotation, the three triggers
and that the observer is not checkpointed were decided. The first version
of the rotation compared path strings, and deleted the only checkpoint at
`num_checkpoints_keep = 1`, since `run//sedov` lists as `run/sedov` (found
in review).

**Amended 2026-10-01**, Erik's decision to keep the checkpoint mechanisms
of TreeAMR's applications alike (TreeGeneralizedHarmonic did the same that
day): the checkpoint moved from after the regrid to before it, so that a
restart may change the regridding criterion and regrid with it first.
Until then the checkpoint was written *after the regrid and its reset*,
decided on 2026-09-29 as TreeAMR's M9a prescribes for a chunked driver;
the first version wrote after the regrid because a restart then has nothing
to replay, and `u` rather than `U.work` was saved because the post-regrid
reset acts on `u`. With the move: the last chunk became a restart point
too; the format went from version 1, `(; recipe, run)` after the regrid, to
version 2; a stopped run returns the state before the regrid (it was
after); and the recipe, decided on 2026-09-29 as the one thing besides
`t_end` that may not change, lost the six criterion parameters to
`run_criterion` — they were in the recipe until 2026-10-01 — so that two
things may change on a restart. The reflecting faces have been in the
recipe since 2026-09-29. Every bit-identity claim held unchanged when the
checkpoint moved before the regrid.

**Amended 2026-10-09**, on TreeAMR 0.2, which moved its checkpoint
functions and its HDF5 extension into the companion package TreeIOHDF5:
TreeIOHDF5 is a hard dependency of this package (Erik's decision, as for
TreeGeneralizedHarmonic and TreeWaveGR), so the caller no longer loads HDF5
and `evolve!` no longer refuses a checkpoint without it. Until 2026-10-09
HDF5 was the caller's to load: the functions had methods only once
`using HDF5` had loaded TreeAMR's extension, and `evolve!` refused a
checkpoint keyword without it at the call. A hard dependency has nothing
left to refuse, and the check and its test went.

## Precision

The `Float32` row of the precision table predicted errors agreeing to ~1%,
TreeWave's pattern; step 12 measured exact meshes and floors and errors
within 1.2e-3, inside the prediction by a factor of eight or more. The
Kelvin–Helmholtz `Float32` row was measured in step 10, at 156 ulp of `M`
before the move to IMEXRungeKutta and 200 since.

## Multi-threading

Step 13 added the static Sedov blast to the thread workload, beside the Sod
tube and the shear layer. After step 11, TreeAMR 0.1.3 (upstream's
`4be726e`) placed every per-block pass on the thread that owns the block,
and this package's one launch of its own, the ghost floor count, moved to
`launch_by_owner!`; the integrator's stage arithmetic moved to the same
ownership with IMEXRungeKutta. The narrowing of TreeAMR's M5 guarantee —
floating-point sums promised to roundoff only — also came after step 11.
On 2026-10-02, for MPI, the host loops in block order and the per-block
maxima and counts combined on the host became combined across ranks as
well; serially nothing moved.

The design then said the totals "should also stop combining by hand:
TreeAMR plans `mesh_mapreduce(f, op, init, fs[, u]; vars, weight)`, one
number for the mesh — `block_mapreduce`'s per-block values scaled by
`weight(key)` and combined over the local blocks, with the M7 `Allreduce`
inside it and nowhere else — and `conserved_totals` is that call with
`weight = key -> spacing(key)^D`. Once it exists the totals go through it,
which is how they turn global without a communicator ever appearing in this
package." Step 15 recorded that `mesh_mapreduce` is in TreeAMR 0.1.3 and
the field-set totals reach it through `total_mass`, the state-vector totals
and `conserved_scales` not yet. After step 15 they did, and the change was
bit-identical, measured so rather than argued from upstream's comment that
`mapreduce(identity, +, v)` is `sum(v)`'s pairwise reduction: the whole
suite, run through `test/runtests.jl` at four threads in two environments
differing only in this change, printed **150 `@info` lines on each side and
not one byte differed** — 38 of them report a drift, an injection, a leak
or a total, and 90 print some floating-point result to 12–17 digits — nor
did any other line of the log, the test summary's time excepted. The
`test/reset_tests.jl` check of gathered totals and of the scale of an
all-positive state was added then.

## Running on a device

The device speedup on unified memory was predicted before step 14
measured it: "Measured in H6; the number is the first thing anyone will
ask." The device tests were widened after step 14, closing the
acceptance's "the suite on Metal in `Float32`": the file had run two cases,
and since then runs every driver the suite measures with.

## Running distributed

Added 2026-10-02, on TreeAMR's M7: written against its `main`, which
`Project.toml` pinned for the day, and taken from the registry at
`TreeAMR = "0.1.6"` once that release carried the guard below. The
decisions under "Running distributed" in `CODE.md` were made in
implementing it, that day.

**An upstream bug, found by the empty rank** (2026-10-02). TreeAMR's
`AllVariables` boundary hook checks its callback's tuple length at "the first
owned point of block 1, which every field set has" — and over a distributed
forest a rank may have none, so the check reads a block that is not there and
throws on that rank alone, which the others then wait for. Its
`fill_by_coordinates!` has the guard (`nblocks(fs) == 0 && return fs`) and its
`cell_boundary!` does not; this package is the only caller of that form (its
boundary hook is `boundary_by_coordinates` over an `AllVariables`). The fix is
the same one line, verified here first against a scratch copy of TreeAMR's
`main` with it added; it went upstream the same day (TreeAMR's
`afb148d`, with its own MPI workload now running a rank without blocks
through every operation) and was released as **TreeAMR 0.1.6**, which
became this package's bound. 0.1.5, registered hours before, is M7 without
it, and was excluded by the bound for that reason. The workload's two-block
tube, which runs a whole run with rank 2 empty, is what found it.

The measurement under "Running distributed" in `CODE.md` was taken on
2026-10-02 against TreeAMR's `main` with the guard above, and the suite
again against the registered 0.1.6. Before the fix, `Pkg.test()` against
TreeAMR's `main` passed 12461 and errored in the one place the bug
predicts, the workload's two-block tube. Every serial number elsewhere in
`CODE.md` was unchanged: the suite's assertions on them were the same and
passed.

## Dependencies and the Julia floor

The current dependencies are under "File layout" in `CODE.md`. How they got
there, by release:

- **H0** pinned TreeAMR's `main` through `[sources]`, with CI on 1.11 and
  release at one and four threads. Its compat bounds:
  `KernelAbstractions = "0.9.42, 1"` and `SciMLBase = "3.50.1"` followed
  TreeWave's, `OrdinaryDiffEqSSPRK = "2.3.2"` the version TreeAMR's test
  environment resolved, `TreeAMR = "0.1.0"`, `julia = "1.11"` for the
  `[sources]` entry. `OrdinaryDiffEqSSPRK` and `SciMLBase` were
  dependencies from H0 and unused until H1c, so that the floor was fixed
  before anything relied on it.
- **TreeAMR 0.1.1** was registered: the `[sources]` entry went,
  `TreeAMR = "0.1.1"` came from the registry and the floor became
  `julia = "1.10"` (see [TreeAMR's release and the 1.10
  floor](#treeamrs-release-and-the-110-floor)).
- **2026-09-23 to TreeAMR 0.1.3**: `Project.toml` pinned TreeAMR's `main`
  through `[sources]` again, to see the owner-based threading before it
  was released; that pin went again once 0.1.3 carried it, and with it the
  contradiction with what the paragraph on the old arrangement (below)
  says.
- **2026-09-25**: the floor went back to 1.11 (see [The 1.11 floor
  again](#the-111-floor-again-2026-09-25)).
- **After step 11**: IMEXRungeKutta replaced `OrdinaryDiffEqSSPRK` and
  `SciMLBase`, at `IMEXRungeKutta = "1.2"`, with TreeAMR 0.1.3 (see
  [IMEXRungeKutta and TreeAMR 0.1.3](#imexrungekutta-and-treeamr-013)).
  Step 12 raised the bound to `IMEXRungeKutta = "1.3"`, the first release
  that runs `Float32x2`: 1.2's step count had no path for a MultiFloat, and
  the fix was made there, not worked around here. Step 15 recorded the
  bounds as `TreeAMR = "0.1.3"`, `KernelAbstractions = "0.9.42, 1"`,
  `IMEXRungeKutta = "1.3"`, the last through a `[sources]` entry, and
  `julia = "1.11"`; neither `OrdinaryDiffEqSSPRK` nor `SciMLBase` was a
  dependency any more.
- **2026-09-29**: the TreeAMR bound became `0.1.4`, the release with
  checkpoint and restart, used from the registry the day it was registered,
  with no pin at all; the tests added `HDF5 = "0.17"`, which loaded
  TreeAMR's extension, the package itself not depending on HDF5.
  `bin/Project.toml`'s `0.1.3` admitted 0.1.4 and was left alone, since
  nothing in `bin/` checkpoints.
- **2026-10-02**: the bound became `TreeAMR = "0.1.6"`, the release with
  MPI support (M7) and the empty-rank guard the distributed runs need; for
  the hours between M7's merge and that release `Project.toml` pinned
  TreeAMR's `main` again, to build the MPI port, and the entry went with
  the release. 0.1.5 was excluded on purpose — it is M7 without the
  empty-rank guard, and the MPI test fails on it. The tests added
  `MPI = "0.20"`, which loads TreeAMR's MPI extension. `CLAUDE.md` then
  said: "Both environments resolve TreeAMR from the registry — the root at
  `0.1.6`, `bin/` at a `0.1.3` bound that admits it, left alone because
  nothing in `bin/` checkpoints or runs distributed".
- **2026-10-09**: the bound became `TreeAMR = "0.2"`, here, in `bin/` and in
  `showcase/` — the release that moved the checkpoints into TreeIOHDF5; the
  package came to depend on `TreeIOHDF5 = "0.1.0"` through a `[sources]`
  entry pinning its `main`, the tests added it with an entry of their own
  and no longer added HDF5, and `showcase/` added it beside the HDF5 its
  frame files need. A scratch environment for the device tests adds
  TreeIOHDF5 for `test/checkpoint_tests.jl`, where from 2026-09-29 it added
  HDF5.
- **2026-10-16**: `julia = "1.13"`, in `Project.toml`, `bin/` and
  `showcase/` (see [The 1.13 floor](#the-113-floor-2026-10-16)).

The arrangement before TreeAMR 0.1.1, recorded when it ended because two of
this package's arrangements were built on it:

TreeAMR was unregistered, and was located through a `[sources]`
entry pinning `rev = "main"`, as TreeWave's does. Two things followed from
that and no longer do. The Julia floor was **1.11**, because `[sources]`
is a 1.11 feature; it became **1.10**, which was TreeAMR's own floor, and
`CI.yml`'s floor cell moved with it. *(Amended 2026-09-25: the floor is
**1.11** again, for all the Tree* packages; see "The 1.11 floor again"
below.)* And the tests saw whatever was on
TreeAMR's `main`, so an unpushed change there was invisible here; they now
see a released version, so an **unreleased** change is invisible here,
which is the higher bar and is worth knowing before a step plans on an
upstream fix. `bin/Project.toml` keeps a `[sources]` entry for *this*
package, which is unregistered, and therefore keeps 1.11 as its own floor
— stated there as a compat bound rather than left to be discovered, since
a 1.10 Pkg ignores `[sources]` and then fails to find TreeHydro at all.
*(Amended 2026-10-16: the floor is **1.13** for the package and for `bin/`
alike; see [The 1.13 floor](#the-113-floor-2026-10-16).)*

`bin/backend.jl`, which both viewers and the benchmark include, may use
only what both environments have; the first Symmetry CPU job failed in
nine seconds without `KernelAbstractions` added to its scratch environment
by name.

### TreeAMR's release and the 1.10 floor

TreeAMR 0.1.1 was registered in General, which ends the arrangement that
H0 built and that three of this package's notes were written around.
`Project.toml` loses its `[sources]` entry and gains
`TreeAMR = "0.1.1"`; `bin/Project.toml` loses its copy of the same entry
and keeps the one for *this* package, which is unregistered.

**What actually changed, as against what it looks like.** It looks like a
dependency bump. Three things move with it.

- **The bar for an upstream change rose.** The pin was `rev = "main"`, so
  a TreeAMR change was visible here as soon as it was *pushed*. It is now
  visible only once it is tagged and registered. The old note — "an
  unpushed TreeAMR change is invisible here" — understates the new
  situation rather than merely restating it, and `CLAUDE.md` and the
  step plan (`PLAN.md`, deleted in step 15) said so.
- **The Julia floor fell from 1.11 to 1.10.** `[sources]` is a 1.11
  feature and was the *only* reason the floor was 1.11; TreeAMR's own
  floor is 1.10, and that is now this package's. The suite is green there
  at **11622 tests**, the same count as at the current release, from a
  clean resolve with no `Manifest.toml`. Nothing in `src/` or `test/`
  needed changing for it, which is the outcome worth recording: the
  package had no 1.11-only construct in it, and nobody had checked.
- **`bin/` did not follow.** Its `[sources]` entry for TreeHydro cannot
  go anywhere, so the viewer environment keeps 1.11 — stated there as a
  `[compat]` bound, because a 1.10 Pkg does not reject `[sources]`, it
  *ignores* it, and then fails to find TreeHydro with an error about the
  wrong thing. The environment re-resolves to TreeAMR 0.1.1 and
  `bin/visualize1d.jl` renders unchanged.

**The measurements.** Whole suite, one thread, back to back on a quiet
machine, 11622 tests in every entry; the coverage column is under [Testing and
CI](#testing-and-ci) below and the argument it settles is which CI cell carries the
instrumentation:

| | Julia 1.10 | Julia 1.13 |
|---|---|---|
| no coverage | 4 m 25.5 | 3 m 32.3 |
| with coverage | 8 m 05.4 | 12 m 38.3 |

The floor is now the *slower* version uninstrumented — 1.25×, where 1.11
was 2.2× slower — and the cheaper one instrumented, by 1.56×. Coverage
therefore stays on the floor cell, for a reason re-measured on the version
that carries it.

**One thing that inverted, and is left as an observation.** CI passes
`check_bounds: 'yes'` on every cell, so what the floor cell actually runs
is the *checked* suite — and at 1.10 that costs nothing: **4 m 11.6
checked against 4 m 25.5 plain**, where at the current release it costs
1.4× (5 m 19.9 against 3 m 44.8, measured after step 11). Both
environments resolve the same KernelAbstractions 0.9.42, so that is not
it. One draw each, not investigated, and recorded because the `@inbounds`
saving in "Bounds checking in the kernels" in `CODE.md` was measured at the *release*
and there is now a reason to think it is not the same size at the floor.
It changes nothing: the annotation is a correctness assertion that CI
checks, and its cost is not what justifies it.

**What was checked.** The suite at 1.10 from a clean resolve (green,
11622); the same suite at 1.10 under `--check-bounds=yes`, which is what
CI's floor cell runs (green, 11622); the suite at the current release
(green, 11622); the viewer
environment re-resolved from a removed `Manifest.toml`, taking
TreeAMR 0.1.1 from the registry, and the tracked tube rendered from it.
The `[compat]` bound in `bin/` is not exercised by any of that — nothing
runs `bin/` at 1.10 — and is there to turn a confusing failure into a
clear one.

### The 1.11 floor again (2026-09-25)

The Julia floor went back from 1.10 to **1.11** (decided), in this package
and in TreeAMR, TreeWave and TreeGRRMHD at once, TreeGeneralizedHarmonic
having been at 1.11 throughout. The reason is the one the floor had
before TreeAMR's release: `[sources]` is a 1.11 key, and every package in
the family should be free to depend on an unregistered package — each
other, or a new upstream — through a `[sources]` entry without moving the
floor again, and without each package reaching the same conclusion on its
own. `CI.yml`'s floor cell is `version: "1.11"`; it keeps the coverage,
for the 1.04× measured on 1.11 under [Testing and CI](#testing-and-ci), which is the number
that applies once more. The 1.10 measurements in the section above are
left as the record of what was measured then.

### The 1.13 floor (2026-10-16)

The Julia floor went from 1.11 to **1.13** (decided by Erik on
2026-10-16), in this package and in all the Tree* packages at once:
supporting the versions before 1.13 slows development and is not useful
at the moment. `julia = "1.13"` in `Project.toml`, `bin/Project.toml` and
`showcase/Project.toml`. `CI.yml` runs 1.13 only — three cells, Linux,
macOS (arm64, with the coverage) and Linux at four threads — and its
`viewer` job runs 1.13 rather than the current release.

Nothing in `src/` or `test/` needed changing, there being no `VERSION`
check anywhere; only comments that named 1.11 as the floor did. Two
things the 1.11 cell did, and that go with it:

- **It caught a `const` defined in two test files**, which 1.11 refuses
  and 1.12 and later allow — the `TRACKED_1D` collision of step 10. The
  source-text check at the top of `runtests.jl`, added then, is now the
  only guard, and it stays.
- **It made coverage nearly free** (1.04× on 1.11 against 3.14× on 1.13,
  in [Testing and CI](#testing-and-ci)). Coverage moved to the macOS 1.13 cell, which
  measured 23 m 22 instrumented on CI against 12–17 m for the old 1.11
  cell, so a push to `main` is expected to take longer; the timeout is per
  cell again for the same reason.

The clean-checkout check at the older version, and the `julia +1.11`
line in `CLAUDE.md`'s "Commands", are gone. The measurements taken at
1.10 and 1.11 elsewhere in this document stay, as the record of what was
measured then.

### IMEXRungeKutta and TreeAMR 0.1.3

Two upstream changes taken together, because each alone would have moved
the same numbers: TreeAMR 0.1.3, whose owner-based threading places every
per-block pass on the thread that owns the block, and the move from
OrdinaryDiffEq's `SSPRK33` to IMEXRungeKutta's (see [Time
integration](CODE.md#time-integration-and-the-time-step) and [Floors and the
atmosphere](CODE.md#floors-and-the-atmosphere) for the design; `src/stepping.jl`
for the code). `Project.toml` loses `OrdinaryDiffEqSSPRK`, `SciMLBase` and
the `[sources]` pin on TreeAMR's `main`, and gains `IMEXRungeKutta = "1.2"`
with a `[sources]` entry of its own.

**How it was checked.** The whole suite was run on the tree before the
change and on the tree after it, at four threads, and the 136 `@info` lines
of each compared line by line. 75 differ, and in all but the lines listed
below the difference is in the last one to three of seventeen printed
digits — Shu–Osher and Butcher form compute the same `uⁿ⁺¹` with the sums
associated differently, and every run inherits that. No number `CODE.md`
records to its printed precision moved except these, and each moved for
the limiter placement and not for the arithmetic:

| run | before (OrdinaryDiffEq) | after (IMEXRungeKutta) |
|---|---|---|
| static Sedov `:center` `D = 2`, `:stage` reset hits | 4096 | **4136** |
| the same, energy drift | 1.24659e-5 | 1.24609e-5 |
| the same, injection | 2.39804e-5 (a bound; ratio 0.51984) | **1.24609e-5 (equal to the drift to 8.9e-16)** |
| the same, L1 against the fine run | 4.4066042e-2 | 4.4066041e-2 |
| static Sedov `:center` `D = 3`, floor / reset / ghost hits | 75 / 24504 / 4703 | **89 / 24576 / 4786** |
| the same, injection | 8.52677e-5 (a bound; ratio 0.52203) | **4.44841e-5 (equal to the drift to 3.6e-15)** |
| Kelvin–Helmholtz `Float32` to `t = 2/5`, `max |M₃₂ − M₆₄|/M` | 156 ulp | 200 ulp |

The `:step` columns — 1384 and 8232 hits, drift equal to injection — are
unchanged apart from the last digits, as they must be: the step limiter is
called on the same state at the same point in both integrators. The
`Float32` shear layer is an instability amplifying roundoff, so a
different association of the same sums is exactly what moves it; the mesh,
the step count and the tracking are still equal to the `Float64` run's.

**Tests: 11708 at four threads and 11674 at one**, against 11622 — the
integrator's own file (`test/stepping_tests.jl`), the two thread-identity
and device files, two TreeAMR internals checked by name, and the hit counts
above; the difference between the two thread counts is the ownership
check, which asserts one thing per block per thread and has nothing to
assert at one thread. Whole suite, back to back on a quiet machine:
**2 m 58.4 at one thread and 2 m 35.2 at four**, against 3 m 44.8 at one
thread after step 11. The share of that owed to the two kernel fixes of
step 14 rather than to the integrator was not separated; the
thread-identity subprocess costs twenty seconds of the four-thread run.

## Testing and CI

The "Testing" section of `CODE.md` was added in step 7b and rewritten in
step 7c. Step 7b split the suite into a short tier and a long one; step 7c
undid the split, having found that what made CI slow was code coverage
under threads and not the runner. Step 15 qualified "every physics claim in
'Measured results' comes from a test that runs on every push" with the
exceptions that section now names. The rule that coverage runs on the
serial cells and stays off the threaded one was amended in step 10b.

**The coverage finding on CI.**
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

**Which cell carries the coverage.** The rule first read:

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

With the runner's architecture as the second axis (the table under
"Testing" in `CODE.md`), coverage went on the **fastest cell of the four**:
the `matrix` put `coverage: true` on the floor version's `macOS-latest`
entry — `version: "1.11"` when this was measured, `version: "1.10"` after
TreeAMR's release moved the floor, and re-measured there (below) rather
than assumed; `version: "1.11"` again since 2026-09-25, where the original
measurement applies. Amended 2026-10-16: with the floor at 1.13 every cell
runs 1.13, and the coverage is on the `macOS-latest` cell at 1.13; the 1.11
column of that table is the record of what was measured then.

*(Re-measured when the floor moved to 1.10.)* Dropping the `[sources]`
entry moved the floor cell from 1.11 to 1.10, and the cell that carries
the coverage moved with it — so the table above was taken again, whole
suite, one thread, back to back on a quiet machine, 11622 tests in every
entry:

| | Julia 1.10 | Julia 1.13 |
|---|---|---|
| no coverage | 4 m 25.5 | 3 m 32.3 |
| with coverage | 8 m 05.4 | 12 m 38.3 |
| factor | **1.83×** | **3.58×** |

**The conclusion survives and the reason for it does not.** Instrumenting
the floor is no longer nearly free: 1.04× was a property of *1.11*, not of
floor versions, and 1.10 pays 1.83×. Nor is the floor the faster version
here — uninstrumented it is 1.25× *slower* than 1.13, where 1.11 was 2.2×
slower. What decides the cell is the instrumented column, and there 1.10
comes in at 8 m 05 against 12 m 38, a factor of **1.56**, which is the
same ordering the original measurement found and is far outside the local
back-to-back scatter. So `coverage: true` stays on the floor cell, and it
stays there for a measurement taken on the version that now carries it
rather than for an inherited one. The 1.11 column is left above as the
record of what was measured then; it was not re-taken, because no cell
runs 1.11 any more. *(Amended in step 15: the floor cell runs 1.11 again
since 2026-09-25, and the 1.11 column is once more the one that applies —
see [The 1.11 floor again](#the-111-floor-again-2026-09-25); and since
2026-10-16 no cell runs either, see [The 1.13
floor](#the-113-floor-2026-10-16).)*
And the 1.13 instrumented entry reading **12 m 38**
both times is a coincidence and not a number carried over: the
uninstrumented one moved from 4 m 01 to 3 m 32 over the same interval,
which is about what the `@inbounds` pass took off everything else.

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

On variance: the same instrumented cell's spread, a factor of 1.7 end to
end, swamps the 1.04× that instrumenting 1.11 costs in the first place; the
cell that then carried the coverage, macOS at 1.11, came back at 12 m 12
and 17 m 03, and the orderings that decided where coverage goes were 12–17
against 42 minutes. The flat `timeout-minutes: 30` then had 1.8× of
headroom over the slower of the two draws, which is about what the
uninstrumented cells had before. With coverage on macOS at 1.13 (23 m 22
measured) a push to `main` is expected to take longer; not yet measured.

Two more economies come from the same measurement, since `timeout` and
matrix size are both set by what a cell costs. The cap is a flat
**30 minutes**, and it is flat *because* coverage sits on the fastest
cell — every healthy run in the matrix is six to seventeen minutes, so
one number guards all four the same way. It was briefly per cell,
`${{ matrix.coverage == true && 40 || 30 }}`, while coverage lived on
Linux at the current release, where a **healthy** run measured 42 m 45
and a flat 30 would have failed it. If coverage is ever moved back to a
1.13 or a Linux cell, the per-cell form has to come back with it.
*(Amended 2026-10-16: it has, since every cell is now 1.13 — the cap is
`${{ matrix.coverage == true && 45 || 30 }}`, against the 23 m 22 measured
for the instrumented cell on macOS at 1.13.)*
And the
matrix is spelled out one cell at a time instead of as a
`version × os` product, because the product's fourth combination is
redundant whichever one it is, and which one is left out follows from the
coverage choice above. Three axes, three serial cells: the floor on macOS
(which carries the coverage, and where a `TRACKED_1D`-style collision is
red while 1.13 is green), the current release on Linux, and the current
release on macOS (which carries the arm64 reduction claim under "Testing" in `CODE.md`). The
pair left unbuilt is the floor on Linux: the floor is a property of the
Julia version rather than of the operating system, so one platform
exercises it. *(Amended 2026-10-16: with the floor at 1.13 the version
axis is gone, and the matrix is three cells, all at 1.13: Linux, macOS
(the arm64 reduction claim and the coverage), and Linux at four threads.
Nothing in CI now catches a `TRACKED_1D`-style collision at run time; the
source-text check at the top of `runtests.jl` is what does.)* Pushes whose
every path matches `**.md` were skipped because half the commits here were
`Record step N …`.

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
in step 7c; what step 7b is kept for is the cross-machine measurement under
"Testing" in `CODE.md`, which is a
real measurement and cost a red CI run to get.

The cross-machine measurement under "Testing" in `CODE.md` was taken when
7b's reference files were first taken to CI. TreeAMR's narrowing of its
promise on floating-point sums, which cites it, came after step 11.

**The `TRACKED_1D` collision** was found in step 10b, when the merge of
steps 8–10 went red on the two 1.11 entries of CI and green on the two 1.13
ones. `driver_tests.jl` and `sedov_tests.jl` both defined `TRACKED_1D`,
`TRACKED_2D`, `FINE_2D`, `COARSE_2D` and `NOFIX_2D`. Julia 1.12 and later
quietly allow a `const` to be redefined, the second file's value replacing
the first's, so every run at 1.13 passed; only 1.11, the floor then, threw
`invalid redefinition of constant`. The check in `runtests.jl` was added
then; with the floor at 1.13 since 2026-10-16, no CI cell checks this any
more.

### Suite timings

Each figure is wall clock on the development machine, which is shared:
runs taken while something else was on it have come back a fifth slower,
so a timing is worth comparing only against another taken back to back
under the same load, and the older figures are an ordering and not a
target.

- **Before step 8**: 11 149 tests in 1 m 42.3 at four threads and 1 m 36.8
  at one.
- **Step 8**: the whole suite is **11 282 tests in 1 m 42.9 at four
  threads and 1 m 55.8 at one** — but most of that difference is the new
  file's own twenty-odd evolutions and the compilation they move earlier,
  and two runs of the same tree differ by seconds, so the reset's price is
  better read off a controlled comparison (under "Step 8" in `CODE.md`).
  Every one of the 75 `@info` lines the suite printed before this step is
  printed **byte-identically** after it; the 20 new ones are identical at
  one thread and at four.
- **Step 9**: 11507 tests in **2 m 43 at one thread and 2 m 09 at four**
  (wall 2 m 51 and 2 m 14), against step 8's 11282 in 1 m 50 and 1 m 42 —
  so the blast adds roughly 50 s and 30 s, which is what a case with a 3D
  adaptive run and eleven two-dimensional evolutions costs. Run to run this
  machine moves by about ten percent at four threads, so the delta is worth
  one significant figure and no more. Every one of the 95 `@info` lines the
  suite printed before this step is printed **byte-identically** after it,
  and the 28 new ones are identical at one thread and at four.

  **Inconsistent:** this gives step 8's suite as 1 m 50 at one thread; the
  step 8 entry above gives 1 m 55.8.
- **Step 10**: 11609 tests in **3 m 44 at one thread and 2 m 38 at four**
  on a quiet machine, against step 9's 11507 in 2 m 43 and 2 m 09 — so the
  shear layer adds roughly a minute at one thread and thirty seconds at
  four. Runs taken while this (shared) machine was busy with other work
  reached 4 m 10 and 3 m 12, so the number carries one significant figure
  and the comparison is only fair between measurements taken under the
  same load. The gap between the two thread counts is the largest in the
  package's history and is the case's own shape: twelve two-dimensional
  evolutions, of which four run 2700 steps on 128²-equivalent meshes, is
  the most parallel work any one file has held. Every one of the 123
  `@info` lines the suite printed before this step is printed
  **byte-identically** after it, and the 11 new ones are identical at one
  thread and at four. The clean-checkout check — a `git archive` tree with
  no `Manifest.toml`, resolving TreeAMR from GitHub `main`, which is what
  CI does and what the `[sources]` pin exists for — passes.
- **Step 11**: adding `kh_run`'s `observer` keyword took the suite from
  **11610 tests in 4 m 13.6 to 11622 in 4 m 31.9** at one thread. The suite
  is green at 11622 tests both on the current release and — the check that
  matters for a step which adds files — on the floor version from a clean
  checkout: `git archive` into an empty tree, TreeAMR resolved from GitHub
  `main`, `julia +1.11`, 6 m 32.2 and every number identical to the 1.13
  run's. The viewer environment was instantiated and rendered from that
  same clean tree, which is the only thing that proves `bin/Project.toml`'s
  two `[sources]` entries resolve — a local run, with a depot that already
  has everything, proves nothing about it.
- **After step 11**, with the `@inbounds` pass, and with IMEXRungeKutta and
  TreeAMR 0.1.3: under "Bounds checking in the kernels" in `CODE.md` and
  [IMEXRungeKutta and TreeAMR 0.1.3](#imexrungekutta-and-treeamr-013)
  above.
- **Step 12**: **11856 tests in 4 m 26 at one thread and 11890 in 3 m 44 at
  four**, against 3 m 36 and 2 m 57 just before the step on the same
  machine. Most of the 50 s is compiling the `Float32x2` paths, which no
  other file exercises. The checked run took 5 m 56 with the command as
  written, against 4 m 26 plain. Step 15 recorded this as the last number:
  the four-thread count higher by the ownership check's one assertion per
  block per thread; the `@inbounds` pass had taken 15% off in between and
  the precision file put about 50 s back.
- **2026-09-29, with checkpoint and restart**: **12030 tests in 4 m 28 at
  one thread and 12064 in 3 m 47 at four**, `test/checkpoint_tests.jl` added
  after `type_tests.jl` — restart chains compared with the uninterrupted run
  field by field, the rotation, the refusals, and a restart in a subprocess
  at the other thread count through the standalone
  `test/restart_workload.jl`. Until 2026-10-09 it was also the first file
  to load HDF5, and tested the refusal of a checkpoint without it first;
  TreeIOHDF5 is a dependency since, and that refusal is gone. Back to back
  on the development machine under an unknown share of other load:
  **6 m 12** with `--check-bounds=yes`, and **7 m 27** for the clean tree at
  Julia 1.11 (TreeAMR 0.1.4 and HDF5 0.17.4 resolved from the registry).
  That is within the machine's scatter of step 12's 4 m 26 and 3 m 44, so
  the new file's cost inside the suite is not separated from it.
- **2026-09-29, with the reflecting walls**, on the development machine
  under other load: **12396 tests in 4 m 46 at one thread and 12430 in
  4 m 10 at four**, **6 m 36** with `--check-bounds=yes`, and **7 m 47**
  for the clean tree at Julia 1.11. The four-thread run overlapped a
  precompilation of the showcase's environment, so it is an upper bound.
- **2026-10-01, with the checkpoint moved before the regrid**, merged with
  the reflecting walls, on the development machine under an unknown share
  of other load: **12447 tests in 5 m 01 at one thread and 12481 in 4 m 28
  at four**, **6 m 33** with `--check-bounds=yes`, and **7 m 45** for the
  clean tree at Julia 1.11 — within the scatter of the walls' own 4 m 46,
  4 m 10, 6 m 36 and 7 m 47: the new testsets restart the tube, a second
  each.
- **2026-10-02, with MPI**: against the registered **0.1.6**, resolved
  fresh, `Pkg.test()` passes **12682 tests in 6 m 10 at one thread and
  12716 in 5 m 08 at four**, and a manifest-free copy on **Julia 1.11.9**,
  the floor then, 12682 in 9 m 32 — against 12447 in 5 m 01 and 12481 in
  4 m 28 on 2026-10-01, the difference being the MPI test and its launch,
  on a machine shared with another session's suite.

## Measured results replaced

What the "Measured results" of `CODE.md` used to record, where a later
measurement or a later integrator replaced it. The "Step N" names are
those sections'.

**Step 4.** "Under `:mc` the same sweep gives 0.945, which is the number to
beat when HLLC is measured." Step 15 recorded that HLLC was measured on the
shear layer and never on Sod.

**Step 8.** The reset's cost was read "against Sedov": "(It made it in step
9, and the answer is that the two give the same state to roundoff, `:stage`
making three times the repairs and reporting a bound where `:step` reports
an equality.)" The cost table itself was measured under OrdinaryDiffEq.

**Step 9**, under OrdinaryDiffEq, on the static two-level mesh. In
`D = 2`:

| run | mass drift | energy drift | reset hits | ghost hits | injection (E) | L1 vs fine |
|---|---|---|---|---|---|---|
| `p = 3`, `:stage` | 1.332e-15 | 1.2465883e-5 | 4096 | 40 | 2.3980402e-5 | 4.406604e-2 |

with "under `:stage` it is `0.51984` of it, the SSPRK stage weights" for
the drift against the injection. In `D = 3`:

| run | mass drift | energy drift | floor / reset / ghost hits |
|---|---|---|---|
| `p = 3`, `:stage` | 3.251e-11 | 4.4512289e-5 | 75 / 24504 / 4703 |

and the floor counts by population read `:center` `D = 2`: 0 / 4096 / 40,
and `:center` `D = 3`: 75 / 24504 / 4703. The energy ratios of the fixup
in `CODE.md`, **2350×** in `D = 2` and **1324×** in `D = 3`, were taken
against these drifts. Step 11's render of a tracked blast cited "the 4096
owned and 40 ghost hits of step 9". The before-and-after table is under
[IMEXRungeKutta and TreeAMR 0.1.3](#imexrungekutta-and-treeamr-013).

**Step 10.** The transcription check recorded: "The one thing that was
wrong is the weighting of `M(t)`, which read "the lower interface alone"
and is in fact mirrored over both interfaces; it is corrected above, as is
the absence of the area-weighted form (14)–(17) that an adaptive mesh
needs." The run ending on the shoulder of the curve "is the reason the
prediction above now says "eventually saturates" rather than "saturates by
`t_end`"". The `Float32` control, before the move to IMEXRungeKutta, which
associates the stage sums differently: `max |M₃₂ − M₆₄|/M = 1.857e-5`,
which is **156 ulp of `Float32`**, and `max |K₃₂ − K₆₄|/K = 2.552e-4`.

**Step 11.** "The step that closes H5b and H5, and it is the first in this
package whose deliverable is a picture rather than a number. What it built:
`bin/Project.toml` (CairoMakie and SixelTerm in an environment of their
own, with `[sources]` for *both* packages), `bin/backend.jl` after
TreeWave's, `bin/visualize1d.jl` (the tracked tube) and
`bin/visualize2d.jl` (`--case=kh|sedov|both`), and the `viewer` job in
`CI.yml`." It recorded three things the implementation settled; `kh_run`'s
`observer` pass-through was "amended here", the design not having said so,
and of the two names that collide with Makie, `density` was the new one.
That a viewer snapshot must materialize what it keeps was found in step 11
as well. The device path through the viewer was not planned for this step,
and was recorded because it was the first time anything in this package
had run on one: "It is **not** the device milestone: H6c still owes the
per-phase table, the opt-in device tests and the benchmark, and nothing
here was measured for speed." (All three were delivered in step 14.) "It
is one case, at one precision, saying the plumbing is connected."

**A movie**, added after step 11: an addition and not part of what H5
accepted, which is why it was recorded apart rather than folded into the
step 11 entry. **The cost was superlinear, and then it was not.** The first
implementation drew one `heatmap!` per block and rebuilt all of them every
frame behind an `empty!(ax)`. That cost 7 m 31 for 301 frames against
2 m 45 for 151 — doubling the frames cost **3.66×**, not 2× — and the
mechanism was never attributed: the per-frame `ax.title` and the
`Colorbar` were free (0.28–0.34 s/frame with and without), plots did not
accumulate across `empty!`, and holding 301 frames live did not slow
drawing on its own. The rewrite sidestepped it rather than explaining it:
the full Kelvin–Helmholtz movie went from **7 m 31 to 57 s**.

**Raising the level cap.** `bin/` once had its own copy of the TreeAMR pin;
since TreeAMR's release it has its own `Manifest.toml` and compat bound
(amended in step 15). The four renders ran as four separate processes
first because both scripts define `main` and `const LEVELCOLORS` at top
level in `Main` and Julia 1.11 — this package's floor then — refuses to
redefine a `const`. Amended 2026-10-16: at 1.13, the floor since, the
redefinition is allowed, and the separate processes stay.

**Bounds checking in the kernels** was added after step 11.

**Step 14.** `test/device_tests.jl` on Metal recorded the tracked Sod
tube's L1 as 4.1725063e-3 on both host and device; it is 4.172509e-3 since
step 12's chunk count. The device tests were widened after step 14,
closing the acceptance's "the suite on Metal in `Float32`": the file had
run two cases. The flux kernel's slowing as the state evolved was
`:minmod` written as `a * b ≤ 0 && return 0` and a ternary, until it became
two `ifelse`s.

**Step 12** was done last, after steps 13 and 14.

**Checkpoint and restart, measured** was added on 2026-09-29, with
checkpoint and restart, on TreeAMR 0.1.4 and HDF5.jl 0.17.4. Amended
2026-10-01, when the checkpoint moved before the regrid: every equality
held unchanged — the same tests, with only the counts of files written
amended for the last chunk's — and two more were added, from a finished
run and with another criterion.

**Reflecting walls, measured** was added on 2026-09-29, with reflecting
faces, on TreeAMR 0.1.4. A restart's recipe written before the field
existed first read as having none; amended 2026-10-01, such a recipe is a
version-1 file, which the move of the checkpoint before the regrid refuses,
so the reading went with it, and the test asserts the refusal instead.

**The zoom showcase** was added on 2026-09-29, its second loop, following
`evolve!`'s order, decided that day.

## Possible extensions

Struck from the list:

- ~~A hand-written SSPRK33 with kernel stage updates, shared with
  TreeWave.~~ Superseded after step 11: IMEXRungeKutta forms the stage
  combinations by block owner on the host and by one fused broadcast on a
  device, which is what this item was for; on a 64-core node it is worth
  2–2.7× on the step (see "Step 14" in `CODE.md`).
- ~~Reflecting boundaries, once the region-form hook has a device form
  upstream; the 2D Sedov in a quadrant is the case that would want them.~~
  Done 2026-09-29, and there was no hook to wait for: TreeAMR's reflecting
  faces are first-class and run on every backend. The Kelvin–Helmholtz half
  box uses them (see "Reflecting walls, measured" in `CODE.md`).

The criterion that would produce a Sedov shell was measured in step 9 and
deliberately not built; the HLLC item and the 3D Sedov render were added in
step 15. For the render, the design had said "`bin/` runs it at demo size";
the viewer of step 11 is `D = 2` only. Step 15 also collected two
observations recorded and not investigated, each with its section; the
second — why `--check-bounds=yes` cost nothing at Julia 1.10 where it costs
1.4× at the release ([TreeAMR's release and the 1.10
floor](#treeamrs-release-and-the-110-floor)) — has been moot since
2026-10-16, no CI cell running 1.10 or 1.11 any more.

## Open questions

What was decided in review, and what was once open. The design said:

> Decided in review: the ghost exchange in conserved variables with
> `con2prim` over the stored extent (was "may change, on the ghost floor
> count"; step 9 measured that count and it does not change — see Floors
> and the atmosphere); primitive reconstruction; HLLE as the baseline flux
> with HLLC as the Kelvin–Helmholtz comparison; the atmosphere reset of `U`
> in the integrator's stage hook, with the floor on `P` as the second line;
> Dirichlet boundaries wherever the physics does not require periodicity;
> one `evolve!` driver with cases as data; the two upstream prerequisites;
> default prolongation order `p = 3`, with `p = 1` measured beside it; the
> McNally smooth-ramp Kelvin–Helmholtz setup; Sedov acceptance on the
> exponent, the jump and a uniform reference, the full profile an
> extension; and the `[sources]` pin to TreeAMR's `main` (`m8` having been
> merged) — which TreeAMR's 0.1.1 release has since replaced with an
> ordinary registry dependency.

Still proposed, until step 15 found that neither was any more:

1. **Whether the entropy wave is a case** or only a test fixture; it was
   written as a case so that the interface-order table has a driver, and it
   does not appear in `bin/`. **Settled in step 7: it is a case.**
   `HydroCase(::EntropyWave)` runs it through `evolve!` on the periodic,
   hook-free path, and it is the case against which the driver's chunking
   and point-sampled initial data were measured (L1 ratios 0.9424 and
   0.9945 against the study, see "Step 7" in `CODE.md`). It is still not
   drawn in `bin/`, which is a statement about pictures and not about what
   it is.
2. **The reset cadence**, `:stage` against `:step`, is a measurement and
   not a question; it was listed so that it was not mistaken for a decision
   already made. **Measured in step 9.** On the blast the two give the same
   final state to roundoff, `:stage` making three times the repairs and
   reporting a strictly less informative injection — a bound rather than an
   equality, because a stage's injection enters the step with that stage's
   SSPRK weight. `:stage` stays the default on the strength of what it was
   adopted for and what a blast in an ambient cannot show: a stage vector
   that the *next* stage's right-hand side never sees in an unphysical
   state, which is a star-in-a-vacuum property. Step 15 recorded what the
   move to IMEXRungeKutta after step 11 changed: the injection is now an
   equality under both cadences, since the stage limiter's correction no
   longer reaches a total, and `:stage` resets 4136 cells on the static
   `D = 2` blast against `:step`'s 1384.

Smaller defaults marked (proposed) in the text — the variable order, point
samples rather than cell averages for discontinuous initial data, `λ_max`
measured once per chunk — were implementation choices that the first
milestones would either confirm or amend in place. All three were settled:
the variable order in step 1, the point samples in step 4 (and generalized
in step 7, where a case states its data pointwise because the adaptation
cycle changes `h` under it), and `λ_max` once per chunk in steps 4 and 7 —
confirmed, but only beside a `speed_headroom` factor and a recheck that
throws.

Step 15 recorded that nothing is open: the one question this package was
expected to take upstream — a limited, positivity-preserving prolongation —
was measured in step 9 and not made, with its price recorded.
