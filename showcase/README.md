# A Kelvin–Helmholtz zoom

A 1080p movie of a Kelvin–Helmholtz instability in which the camera zooms
through twenty refinement levels and stays crisp, then zooms back out to
show how far it went. It is built in two independent steps:

- `simulate.jl` runs the shear layer and writes one small file per movie frame;
- `render.jl` turns those files into a movie.

The simulation is the expensive step and can be restarted from a checkpoint.
The rendering can be redone with other colours or overlays without rerunning
anything. `GOAL.md` is the request this answers.

## How it works

**The half box.** The flow is McNally, Lyra & Passy's shear layer
(`KelvinHelmholtz` in `src/kelvinhelmholtz.jl`), with two changes:

- a thinner ramp, `L = 1/160` against their `1/40`, so that the layer has
  rolled into spirals of several turns by the time the zoom starts;
- the `:mirrored` seed, `v_y = a sin(4πx) sin(2πy)`. Under it the periodic
  box is its own mirror image in `y = 0` and `y = ½`.

So the simulation runs the lower half only, periodic in `x` and between two
reflecting walls: `HydroCase(w; half = true)`, which uses TreeAMR's
first-class reflecting faces. The camera looks at the reconstructed plane —
the full box, repeated periodically, with the upper half mirrored — and every
sampled point is folded back onto the half box (`fold` in `kh_zoom.jl`).

**The clock.** Simulated time advances at `V₀/Z` per movie second, where `Z`
is the zoom. This keeps the speed of features on screen constant.

*Is that true for Kelvin–Helmholtz?* Yes, for what the movie is about. An
inviscid vortex sheet with velocity jump `ΔU` has no length scale of its own.
A disturbance of size `ℓ` on it grows, and is carried across its own size, in
a time of order `ℓ/ΔU`. So slowing the clock by the zoom factor keeps apparent
speeds fixed. The roll-up cascade runs on the same clock: each generation of
billows winds its sheet thinner in a few `ℓ/ΔU`, so new detail can appear at
each level as the camera arrives. Three caveats:

1. **The bulk flow.** A camera fixed in space sees the gas stream through at
   constant pixel speed. So the camera's anchor is a tracer carried with the
   gas, advanced by Heun's method at frame cadence.
2. **Smooth regions.** These evolve on the large-scale strain time, which is
   order one, and appear to freeze at deep zoom.
3. **Contacts do not re-steepen after refinement.** A sheet gets thinner only
   by strain and roll-up. Whether detail keeps pace with the zoom is therefore
   a race within a factor of a few. The pilot is what shows it.

The step count per movie second is `V₀ n / (cfl/(D·headroom·λ))`, where `n` is
the number of finest cells across the view. It does not depend on the zoom,
because the step shrinks with the finest cell exactly as the clock slows.

**The window.** Refining the whole interface to twenty levels would double the
cells at every level: about 70 GB at ten levels past the full-box one, and out
of an H200's memory by eleven. Instead, level `ℓ` is allowed only inside a box
about the camera (`block_cap` in `kh_zoom.jl`):

- Each box covers the *largest* view that will ever need that level, with a
  margin (`window = 1.25`).
- Each box is twice the size of the next finer one, so each level holds about
  the same number of cells.
- Inside a box, TreeHydro's Löhner criterion still decides what refines.
- Everywhere, the whole box may refine to the level the full-box view needs.

Because the boxes are sized for the largest view per level, the final mesh
covers every view of the zoom-out. The zoom-out is taken from the frozen final
state and stays centred on the anchor until zoom 1, then pans home. The levels
come in as the zoom doubles, so the view always holds between `n_min` and
`2 n_min` finest cells across its height.

**The loop** in `simulate.jl` has the same order as `evolve!` and is built
from the same pieces:

1. step, then the CFL recheck;
2. write the frame;
3. regrid with the windowed flags;
4. rebuild the problem, then the post-regrid reset.

It is a second loop because `evolve!` fixes its chunk and its level cap for a
whole run. Here the chunk ends are the frame times and the cap is a window.
Each frame interval is split so that the regrid margin stays within half a
block.

**The frames.** Each frame is one HDF5 file `frames/fNNNNNN.h5` holding:

- `rho`: ρ as UInt16 over `rho_range`, sampled with 2×2 samples per pixel by
  TreeAMR's `interpolate` with `Lagrange(2)`, bilinear between cell centres,
  so a contact is never overshot;
- `minimap`: the full box;
- `blocks`: the outline of every block in view that is at least 3 px wide;
- attributes: the time, zoom, camera, levels, block and cell counts, and
  wall-clock time split into stepping, regridding and sampling.

`log.tsv` gets one line per frame, and `schedule.tsv` has the whole
prescribed schedule.

**Checkpoints** are TreeAMR's, written:

- at every new finest level;
- at the wall-time limit (`run.walltime_hours`), with a margin of three of
  the slowest frames;
- at the end.

A restart recomputes the schedule from the configuration and refuses one whose
frame times differ. The camera's tracking keys and the run keys may change.
A restart still in the intro takes its target from the new configuration,
which is how a target is chosen.

## Running it

The environment, once:

```bash
julia --project=showcase -e 'using Pkg; Pkg.instantiate()'
```

**Smoke**: three doublings at postage-stamp size in 15 s, then the movie in a
few seconds more:

```bash
julia -t 4 --project=showcase showcase/simulate.jl --config=showcase/configs/smoke.toml --out=showcase/output/smoke
```

```bash
julia --project=showcase showcase/render.jl --frames=showcase/output/smoke
```

**Choosing a target.**

1. Run the intro alone. For the pilot the last intro frame is 299:
   ```bash
   julia -t 6 --project=showcase showcase/simulate.jl --config=showcase/configs/pilot.toml --out=showcase/output/pilot --stop-after=299
   ```
2. Look at the frame:
   ```bash
   julia --project=showcase showcase/render.jl --frames=showcase/output/pilot --to=299 --still=299 --title=
   ```
3. Set `camera.target` in the configuration, in the reconstructed plane.
4. Continue from the checkpoint:
   ```bash
   julia -t 6 --project=showcase showcase/simulate.jl --config=showcase/configs/pilot.toml --out=showcase/output/pilot --restart=showcase/output/pilot/checkpoints/stop_f000299.h5
   ```

**The pilot** is the coarse movie. It uses the production's pacing and phases
at a quarter of its linear resolution, up to 12 levels, at 960×540, and runs
on a laptop: see "Measured" below.

**Cutting the zoom.** The zoom goes as deep as `max_level`, but past the
level where the flow's detail ends the view is a smooth blur. To end the movie
where the detail ends:

1. Read `log.tsv` for the last level the criterion created (column 6, the
   finest level, against column 5, the level the view needs), and look at the
   frames around it.
2. Choose the cut level `L`.
3. Copy the output directory, then restart it from `checkpoints/levelL′_*.h5`,
   where `L′ = L − 1`, with `mesh.max_level = L`. Until the last doubling's
   ease-out begins, the two schedules put every frame at the same time, which
   the restart checks. It then re-simulates one doubling with the zoom easing
   to a stop, the hold, and the zoom-out from the right state, so the movie
   has no jump.
4. The renderer reads only the frames the new schedule has.

**Aiming at a later time.** `camera.target_time` and `camera.drift` make
`target` the anchor's position at `target_time`. Before that time the anchor
moves at `drift` onto it; after it, the gas carries the anchor. A point can
therefore be picked in a frame from deep in an earlier run, as long as the
mesh up to that time did not depend on the camera.

**The strain-matched zoom** (option 1; `configs/pilot_strain.toml`). The
slowed clock gives each new level half the simulated time of the one before,
but the sheet thins by strain, at a fixed rate. The braid's strain was
measured on the first pilot's checkpoints at σ ≈ 4, with eigenvalues ±3.6 to
±4.1 from t = 1.5 to 2.73. So option 1 changes three things:

- **The clock.** `movie.clock_exponent = 0` stops the clock slowing with the
  zoom, and `movie.V_zoom = ln 2 / (σ T_level)` makes one doubling of the
  zoom last as long as the strain takes to halve a sheet.
- **The camera.** `camera.track = "stagnation"` re-finds the braid's
  stagnation point near the anchor every frame. A tracer leaves a hyperbolic
  point exponentially, as fast as the zoom closes in.
- **The regrid margin.** The steps per frame now grow with the zoom, so
  `scheme.buffer_speed = 0.75` makes the margin cover the gas speed, which is
  what carries the contacts, instead of the sound speed.

The price is that the steps grow with the zoom: the deepest doubling costs as
much as all the others together.

**Production** is one H200 job on Symmetry. Submit it from a checkout of its
own: rsync the tree to a fresh directory, never into one whose jobs are
running.

```bash
sbatch showcase/symmetry_showcase.sh
```

**Queues.**

- **`h200debugq`** is for getting started, once. Never chain jobs there.
- **`h200q`** is for a run that continues from a checkpoint.
- **`h200preq`** runs on the debug-reserved nodes when they are idle. A debug
  job can preempt it, and a preempted job is requeued, so submit it with
  `SHOWCASE_RESTART=latest`: the requeued job then continues from its newest
  checkpoint, or starts fresh if there is none yet.
- **Memory.** The script requests 64 GB of host memory. The default is 21 GB
  per CPU, 341 GB in all, which the group's memory limit holds pending.

It builds a scratch environment with CUDA under
`/mnt/beegfs/eschnetter/claude/treehydro-showcase`, runs the simulation on the
device, renders on the node's 16 host cores, and leaves everything in
`/mnt/beegfs/eschnetter/claude/treehydro-showcase-$SLURM_JOB_ID`. The script's
header has the variables for a pilot on `h200debugq`, a restart, and a
render-only rerun.

## The knobs

| key | pilot | production | what it does |
|---|---|---|---|
| `physics.L` | 1/160 | 1/160 | ramp width; McNally's is 1/40 |
| `mesh.N`, `mesh.roots` | 16, 2 | 32, 2 | block size, root blocks along x |
| `mesh.n_min` | 128 | 512 | finest cells across the view height, at least; cost ∝ n³ |
| `mesh.max_level` | 12 | 16, then cut back | the finest level the zoom reaches |
| `mesh.window` | 1.25 | 1.25 | margin of each level's box over its largest view |
| `movie.width`, `height`, `supersample` | 960, 540, 1 | 1920, 1080, 2 | the frame |
| `movie.V0` | 0.15 | 0.15 | simulated time per movie second at zoom 1 |
| `movie.intro`, `T_level`, `hold` | 10, 5, 4 | 10, 5, 5 | seconds: before the zoom, per doubling, at the deepest zoom |
| `movie.T_out`, `final` | 0.75, 4 | 0.75, 4 | seconds per doubling of the zoom-out; the last hold |
| `camera.target` | a vortex core | the pilot's braid point | the anchor when the zoom starts, or the guess for `target_mode` |
| `camera.target_mode` | fixed | braid | `braid` (pressure maximum on ρ ≈ 1.5) or `core` (pressure minimum) nearest the guess, found at the zoom start |
| `camera.lock_gain` | 0 | 0 | a pull toward the density gradients near the anchor |
| `camera.track` | true | true | `true` (a tracer), `false`, or `"stagnation"` |
| `movie.clock_exponent`, `V_zoom` | 1, V0 | 1, V0 | the clock during the zoom, `dt/dτ = V · 2^(−α ζ)` |
| `scheme.buffer_speed` | `"signal"` | `"signal"` | the speed the regrid margin covers |

The renderer's options are in its header: `--colormap=` (default `lipari`),
`--mesh=zoomout|always|none`, `--title=`, `--still=K`, `--from=`/`--to=`,
`--crf=`, `--no-minimap`, `--no-text`.

## Measured

**The pilot, 2026-09-29**, on the development machine (Apple M3 Pro, CPU,
Float64). It used `configs/pilot.toml`: `L = 1/160`, levels 3 → 12 over nine
doublings, 2198 frames, of which 1831 are simulated from `t = 0` to
`2.7293`. The intro was run once to its checkpoint and then continued twice
from it, side by side at five threads each. One run zoomed into a vortex
core at (0.3730, 0.2520). The other zoomed into the braid's stagnation point
at (0.5605, 0.2363). Both points were located from the intro's last frame as
the pressure minimum and the pressure maximum on ρ ≈ 1.5.

| | braid | core |
|---|---|---|
| intro, 300 frames | 40 s | 40 s |
| zoom, hold, zoom-out | 13.2 min | 10.2 min |
| steps | 24473 | 20358 |
| mesh changes | 698 | 856 |
| peak blocks, cells | 2612, 669k | 2198, 563k |
| finest level reached | **9** | **7** |
| host memory | 2.2 GB | 2.1 GB |
| floor, reset, ghost hits; injection | 0, 0, 0; exactly 0 | 0, 0, 0; exactly 0 |

Other costs:

- The frames were 1.0 MB each at 960×540, 2.5 GB per run.
- The checkpoints were 1–22 MB each.
- Rendering took 47 s for all 2198 frames, about 0.02 s per frame.
- A restart reproduces the uninterrupted frames bit for bit: the smoke run
  was stopped at frame 40 and continued, and all 103 frames and anchors
  matched.

**What the pilot found: the flow runs out of detail before the mesh runs
out of levels.**

- The machinery works: the camera stays on its target, the window, the
  clock and the frames behave as designed, and the zoom-out shows the nested
  levels.
- But the Löhner criterion stopped asking for new levels at level 9 on the
  braid and level 7 on the core, against the 12 the zoom went to. Beyond
  zoom ~64, the view shows a smooth ramp at fewer than `n_min` cells: blurry
  because the solution there *is* smooth, not because the mesh is coarse.
- Each level was first needed at these times:

  | level | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|
  | t | 2.189 | 2.459 | 2.595 | 2.662 | 2.696 | 2.713 |

  That is, each level gets half the simulated time of the one before (0.27,
  0.135, 0.068, …). So the clock that keeps pixel speeds constant gives
  level 9 about 0.016 time units.
- A sheet thins only by strain or roll-up, at an order-one rate. A contact
  smeared at one resolution does not re-steepen when a finer one arrives.
  So no sheet thinner than the one level 9 resolved ever forms.
- The self-similarity argument above holds for the *growth* of a
  disturbance on a sheet. It fails for the *thinning* of the sheet, and the
  thinning is what feeds each new level.

**The uncut production run, 2026-09-29** (Symmetry job 565872, one H200):

- Configuration: `configs/production.toml`, levels 4 → 16 over twelve
  doublings, 2746 frames, the automatic braid target.
- The simulation took **24.8 min**: 88328 steps, 1957 mesh changes, at most
  5837 blocks and 5.98M cells, with no floor hit and exactly zero injection.
- The frames are 12 GB, 4.4 MB each.
- The host peaked at 6.2 GB. The device reported 141 GB used, which is
  CUDA's caching pool keeping what it once allocated, not the working set.
- **The finest level was 9, reached at zoom ×32** (t = 2.6962). From ×64 on,
  the view needed levels 10 to 16 and the criterion created none of them, as
  the pilot predicted. Beyond level 9 each frame took 1–9 steps, so the dead
  part of the zoom cost almost nothing.
- The full-box view at the end resolves secondary billows along every braid,
  which the pilot's resolution did not.
- The automatic braid point was taken at t = 1.5, and by the deep zoom the
  gas had carried the anchor onto the smooth edge of a blob. So its frames
  are a smooth curve by ×16.

That is what `configs/production_cut.toml` corrects. The zoom stops at level
10 (×64). The target is a secondary billow's rolled-up tip, picked in frame
780 of the uncut run at t = 2.5947. Up to that time, the last before level 7
first appears, the mesh covers the whole box whatever the camera does, so the
flow — and the point — are the same in any run.

**The cut production run, 2026-09-29** (Symmetry job 565876, one H200,
`configs/production_cut.toml`):

- 1711 frames, 57 s of movie.
- The simulation took **20.0 min**: 92621 steps, 1075 mesh changes, up to
  9.93M cells, with no floor hit and exactly zero injection.
- The render took 233 s, and the movie is 42 MB.
- **Every level kept pace with the view, up to level 10 at ×64**:

  | level | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|
  | first needed, t | 2.189 | 2.459 | 2.595 | 2.662 | 2.696 | 2.715 |

  The view holds a secondary billow's spiral at ×16 and a multi-turn spiral
  at ×64.
- Aimed at a secondary billow, the flow carries one level more detail than
  the smooth blob edge did.

**The strain-matched pilot, 2026-09-30** (Symmetry job 566127, one H200 on
`h200preq`, `configs/pilot_strain.toml`):

- 128–256 cells across a 960×540 view, levels 2 → 11 over nine doublings,
  t from 1.5 to 3.40 during the zoom.
- 691182 steps, 3046 mesh changes, up to 1.63M cells, with no floor hit and
  exactly zero injection. The render took 73 s.
- **Every level kept pace with the view to the end, at ×512** — against ×32
  under the slowed clock at the same resolution. The sheet stays a crisp
  contact at every zoom.
- **But at the stagnation point there is one stretched contact and little
  else.** For most of the zoom the view is a straight line, curving only in
  the last doubling. The zoom-out shows the braid's many layers.
- **Cost.** The steps grow as `2^ℓ`: the last doubling took about half of
  them, at 1200–2400 steps a frame.
- **Preemption.** The job was preempted and requeued three times during the
  level-11 hold. Each time it resumed from the level checkpoint, which is
  what `run.checkpoint_minutes` (every 10 min, the newest two kept) now
  bounds.

**The strain-matched billow zoom, 2026-09-30** (`configs/production_strain*.toml`,
Symmetry, one H200 per job, all in
`/mnt/beegfs/eschnetter/claude/treehydro-showcase-strainprod`):

- **Stage A** (job 566175, `h200debugq`, under 5 min): the intro to t = 2.1
  and the zoom to ×4.
- **Stage B** (job 566190, `h200q`): restarted from the ×2 checkpoint,
  re-aimed at the secondary billow in stage A's frame 640, which the camera
  reached exactly (0.5737, 0.2843). Every level kept pace to level 10 at ×65.
  But the camera rode the billow's centre, and by ×64 that centre had mixed
  into a smooth blob. The job was cancelled at ×67 after 48 min, 1180 steps
  a frame and 9.6M cells.
- **The cut** (job 566206, `h200q`, 51 min including the render): restarted
  from B's ×16 checkpoint with `max_level = 9`, so the zoom eases to ×32 and
  holds there. The movie is 1553 frames and 46 MB.
- **What it shows:** crisp spirals at ×8, stretched sheets around a smooth
  core at ×32. At ×27 in the zoom-out there are fresh tertiary spirals beside
  the billow: the next target, if the zoom goes deeper.

**The strain-matched zoom into a tertiary spiral, 2026-09-30/10-01**
(`configs/production_strain_c.toml`, Symmetry job 566253, `h200q`, one H200,
`/mnt/beegfs/eschnetter/claude/treehydro-showcase-tertiary`):

- **Setup.** Restarted from stage B's ×16 checkpoint. The camera glides onto
  the multi-turn spiral that the ×32 cut showed beside the billow at
  t = 3.385, then follows the gas.
- **The target.** The spiral formed under that run's mesh, but it was there
  in this one too. At ×74 the view holds a chain of three crisp tertiary
  billows.
- **Deeper.** They merge into layered turbulent mixing: crisp at ×131 and
  softer by ×256.
- **Levels.** Every level kept pace through all eight doublings, to level 12
  at ×256 (t = 3.735). Up to 17.8M cells.
- **Cost.** 1571215 steps and 6038 mesh changes; 10 h 23 min in all, of which
  stepping took 10.1 h and regridding 7 min. **The 5 s hold at level 12
  alone cost 5.7 h** (4726 steps a frame) — twice the estimate, and the
  first thing to shorten.
- **Result.** No floor hit and exactly zero injection. The render took 279 s,
  and the movie is 2071 frames and 71 MB.

On the core, the diffused centre is smooth, and the criterion's hysteresis
band (0.08 against 0.02, a factor of 4) equals the `h²` scaling of `τ` at a
smooth extremum. So level-8 blocks there refine and coarsen on alternate
frames: a cost in regrids and nothing else.
