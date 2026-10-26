# TreeHydro.jl

[![CI](https://github.com/eschnett/TreeHydro.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeHydro.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/eschnett/TreeHydro.jl/graph/badge.svg?token=Z2KD14GPU6)](https://codecov.io/gh/eschnett/TreeHydro.jl)

`TreeHydro` solves the equations of Newtonian ideal hydrodynamics with a
high-resolution shock-capturing finite-volume scheme, as the
*conservative* sample application for
[TreeAMR](https://github.com/eschnett/TreeAMR.jl).

That is `∂ₜU + ∇·F(U) = 0` for `U = (ρ, S_i, E)` on an adaptively refined
mesh: MUSCL reconstruction of the primitive variables, an HLL-family
approximate Riemann solver, and SSPRK33 in time.
[TreeWave](https://github.com/eschnett/TreeWave.jl) shows the same mesh
under a second-order finite-difference scheme, which needs nothing special
at a coarse-fine face; this package needs everything TreeAMR's M8
milestone added — per-field-set ghost widths, ghost-free face-centered
flux fields, the conservative operator family, and the interface flux
restriction that makes the scheme conserve across refinement boundaries.

Four cases run, each measuring something the others cannot: a
smooth entropy wave (an exact solution, so the convergence order and the
interface-order rule can be measured), Sod's shock tube (against an exact
Riemann solver), the Sedov blast (a strong shock through the floors and
the atmosphere reset, in 3D), and the Kelvin–Helmholtz instability (a
contact-dominated flow, and refinement following a structure that grows).

Every numerical method is chosen to have a direct counterpart in a
relativistic MHD code, which is what the package rehearses; methods that
only work for Newtonian hydrodynamics are avoided even where they would be
better here.

**Contents**

- [Status](#status)
- [Testing](#testing)
- [Checkpoint and restart](#checkpoint-and-restart)
- [Running distributed](#running-distributed)
- [Pictures](#pictures)
- [Design and history](#design-and-history)

## Status

**Status: complete. All four cases run and all four are drawn, at any
thread count, on a GPU and at three float types — milestones H0 through H6
are done, and the design document has been read against the code a last
time.** Since then: checkpoint and restart, reflecting walls, and runs
distributed over MPI. Every measured result — the convergence orders, the
conservation at coarse-fine faces, the tracked shock tube, the Sedov blast,
the Kelvin–Helmholtz growth rate — is under "Measured results" in
[CODE.md](CODE.md#measured-results).

**And the same answer comes back at any thread count, on a GPU, and at a
lower precision.** Every run is bit-identical at one thread and at four; on
Apple's Metal every driver reproduces the host `Float32` run bit for bit,
and on an H200 at `Float64` to `1e-13`, at up to 715 million cell updates
per second. At `Float32` and at the software float `Float32x2` the tube and
the blast build the `Float64` mesh and floor the same cells exactly, with
their errors agreeing to `1.6e-5` and `1e-13` — which took one rounding
fix in the driver that only `Float32` could expose.

**Inconsistent:** `CODE.md` ("Step 14", the device tests widened) says
Metal reproduces the host bit for bit in everything except the shear
layer, whose seeded `sin(4πx)` differs in the last bit; the paragraph
above says every driver.

## Testing

There is one suite and it runs whole, on every push: the unit tests
and every physics claim the measured results in `CODE.md` rest on — the
convergence sweeps, the interface-order tables, the refinement
calibration, the tracked shock tube, the blast and its similarity law, the
shear layer and its growth rate.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The same numbers are expected at any thread count, to the last bit, and CI
runs the suite at one and at four; the suite's last file reruns three
short cases in a subprocess at the other count and compares their digests
character for character. `Pkg.test` does not inherit `-t`, so the count has to
be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

## Checkpoint and restart

**And a run can outlast a job.** `evolve!` writes checkpoints at chunk
boundaries, before the regrid, and restarts from them through
`save_checkpoint` and `load_checkpoint` from
[TreeIOHDF5](https://github.com/eschnett/TreeIOHDF5.jl), TreeAMR's
checkpoint package and a dependency of this one, regridding first; a restarted run,
or a chain of them, is the uninterrupted run bit for bit at any thread
count. A restart may move `t_end` and change the regridding criterion
(`refine_tol`, `coarsen_tol`, `maxlevel_cap`, `ε`, `ε_g`, `buffer`), which
its first regrid then uses and which it reports; with any other parameter
changed it is refused by name. One script serves every job of a chain — the first finds no
checkpoint and starts from the initial data, and each later one continues
where the previous one stopped:

```julia
using TreeAMR, TreeHydro

prefix = "run/sedov"                  # files run/sedov.it0000001234.h5
case = HydroCase(SedovBlast(Float64, Val(3); r₀ = 1 // 8); roots = 4)
r = evolve!(case, Val(3); N = 8,
            ops = Operators(family = Conservative, prolongation = 3, restriction = 2),
            t_end = 1 // 20, chunk = 1 // 600, limiter = :minmod,
            refine_tol = 2 // 25, coarsen_tol = 1 // 50, maxlevel_cap = 2,
            checkpoint_path_prefix = prefix,
            checkpoint_interval_seconds = 3600,      # one an hour
            max_walltime_seconds = 23.5 * 3600,      # stop inside a 24 h queue
            restart_file = latest_checkpoint(prefix))
r.finished || exit(3)                 # the job script resubmits on 3
```

The last two files are kept (`num_checkpoints_keep`), whichever job wrote
them. The wall-time limit is timed from the call, so leave room for Julia's
startup and compilation below the queue's own.

## Running distributed

**And a run can span many processes.** Pass a communicator and launch under
`mpiexec`; every rank makes the same call. The blocks are distributed by
TreeAMR, and the answer is the serial one — the state, the mesh, every step
and floor count bit for bit at any rank count, the conserved totals to
roundoff. MPI is the caller's to load. A checkpoint written at
one rank count restarts at any other, or serially:

```julia
using MPI, TreeAMR, TreeHydro
MPI.Init()
r = evolve!(case, Val(3); …, comm = MPI.COMM_WORLD)
```

This needs TreeAMR 0.1.6 or later; the package's compat bound is 0.2.

## Pictures

**And there are pictures.** `bin/` holds the viewers, in an environment of
their own so that CairoMakie is never a dependency of the package. They
contain no time-stepping loop: every frame and every curve comes through
the one driver's `observer` hook, which exists for exactly that. The first
call instantiates the environment, and the `[sources]` entry for this
package means no manual `Pkg.develop`:

```bash
julia --project=bin -e 'using Pkg; Pkg.instantiate()'
```

The shock tube against the exact Riemann solution, drawn one line per block
and coloured by refinement level, with the Löhner indicator that built the
mesh under it and the conserved integrals beside it:

```bash
julia --project=bin bin/visualize1d.jl
```

The shear layer rolling up, one heatmap per block with the block outlines
on top, with McNally's two diagnostics against the uniformly fine run and
the block count that shows what the refinement saved — and, under
`--case=sedov`, the blast's filmstrip and the radial scatter against the
similarity profile:

```bash
julia --project=bin bin/visualize2d.jl --case=kh
```

And `--movie` turns the 2D filmstrip into a video, from the same run and
without a new dependency — every frame the driver's observer hands over
rather than the four the strip draws, which for the shear layer is 301 of
them:

```bash
julia --project=bin bin/visualize2d.jl --case=kh --movie
```

Both take `--type=f32` and `--backend=cuda|metal`; no device package is a
dependency of this one. CI renders every figure on every push and uploads
them, because `bin/` sits outside `src/` and `test/` and nothing else would
notice it breaking.

## Design and history

See [CODE.md](CODE.md) for the design document — the equations, the
scheme, the cases, every measured result beside the prediction it confirms
or corrects, and the possible extensions that were measured or considered
and not built — and [HISTORY.md](HISTORY.md) for how the package got here.
