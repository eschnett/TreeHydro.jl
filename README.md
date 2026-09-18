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

**Status: all four cases run and all four are drawn — milestones H1, H2,
H3, H4 and H5 are done.** What exists is
the module shell, the `Base` bridges for software floating-point types,
the host-copy helpers, the tests that say the pinned TreeAMR still
provides what the scheme is written against, the ideal-gas equation of
state, the conversions between the conserved and primitive states, the two
floor rules and the reset that imposes them on the conserved state from
the integrator's own limiter hook, the MUSCL reconstruction with its three
slope limiters, the
LLF, HLLE and HLLC fluxes, the six-step right-hand side over the mesh with
its three kernels, SSPRK33 in time, the Löhner refinement criterion, the
one chunked evolve-and-regrid driver, and four cases on it. The entropy
wave is an exact solution of the nonlinear system, and it
measures second order in L1 and L∞ in one and two dimensions. Sod's shock
tube runs against Toro's exact Riemann solution through a Dirichlet
boundary — the first use of TreeAMR's physical-boundary hook by any
downstream package — at an L1 rate of 0.903, with the tube giving the same
answer along every axis *bit for bit*.

**The central claim is measured.** On a two-level mesh in one, two and
three dimensions, every one of the `D + 2` conserved integrals holds to a
hundredth of one ulp of its own scale per step — and the identical run with
TreeAMR's interface flux restriction switched off, one line of the
right-hand side, leaks by **a factor of `1e8` to `1e9`**. The prolongation
order behaves as the interface-order rule predicts for a flux divergence:
L∞ rates of 0.96, 2.03 and 2.04 at orders 1, 3 and 5 against an unrefined
control of 2.02.

**The refinement criterion is a Löhner indicator on `ρ` and `p`**, and it
is calibrated rather than guessed: max `τ` on uniform meshes at four
spacings says that a captured shock scores 0.57 at every one of them —
so the criterion never resolves a discontinuity, and the level cap is
what stops it — while the shear layer's smooth ramp falls by more than a
factor of two per halving, which is what picks the thresholds. An
atmosphere six orders below the data scores 0.0020 with the indicator's
global floor term and 0.97 without it.

**And the mesh follows the shock.** There is exactly one time-stepping
loop, `evolve!`, and a case is data it takes: the tracked shock tube
matches the uniformly fine reference's L1 error to within 0.04% in one
dimension and 0.00% in two, at 200 cells against 256 and 1472 against
2048, with the uniform coarse mesh as the control at 3.68 and 1.85 times
the error. Every cell the indicator fires strongly on sits on a
finest-level block at every chunk, and all `D + 2` integrals hold to
roundoff across the regrids while the same run without the interface
fixup leaks by a factor of `1e9`. A time step sized from the signal speed
at the *start* of a chunk is not safe on a shock tube — a Riemann
problem's fastest signal is not in its initial data — so the driver carries
a per-case headroom factor and rechecks the condition at the end of every
chunk, loudly: with the headroom at 1, Sod throws in its first chunk.

**And where the gas runs out, the atmosphere is imposed on the state.**
The reset is a pointwise `con2prim` / floors / `prim2con` pass run from
`SSPRK33`'s own stage limiter and again after every regrid — GRMHD
practice in the integrator's vocabulary, and the right-hand side still
never mutates its state. It writes back *only* the cells a floor fired in,
which is what lets the conservation results above stand unchanged with the
reset on by default: on the tracked tube and the entropy wave the measured
injection is exactly `(0, 0, 0)`, no cell is floored in either population,
and the final state is bit-identical to a run with the reset switched off.
Applying it twice equals applying it once bit for bit, at `Float64` and
`Float32` in one, two and three dimensions.

**The Sedov blast is where the floors finally fire, and where two
Dirichlet faces meet.** The shock expands as the similarity law says it
must — measured exponents 0.641, 0.504 and 0.438 against `2/3`, `1/2` and
`2/5`, with the constant `ξ₀ = 1.0328` for `γ = 7/5` in 3D reproducing
Taylor's own 1.033 — and the captured density jump approaches the
strong-shock limit of 6 from below. The tracked mesh reproduces the
uniformly fine run *to roundoff* at 12544 cells against 16384. Three
findings corrected the design. A mesh that tracks a shock keeps its
coarse-fine faces in undisturbed gas, so a tracked run cannot measure the
interface flux restriction at all, and every such claim is made on a
static mesh the blast leaves; the evacuated interior never reaches the
atmosphere density, so it is the *pressure* floor that fires and the
coarse-fine face that drives it, 4096 cells in two dimensions and 24504 in
three; and a piecewise-constant prolongation floors nothing at all there,
which buys exact positivity for 0.47% of the L1 error. Every outward-facing
ghost entry of a fine block wedged into a corner of the box holds its
boundary state exactly — 1664, 72000 and 59360 of them across a 2D corner,
a 3D edge and a 3D corner — which is the one exchange path no downstream
package had run.

**And the Kelvin–Helmholtz shear layer chooses the flux.** McNally, Lyra &
Passy's smooth-ramp setup — checked against the paper term for term — is
the one case with no closed form, and the one whose feature *grows* rather
than travels: the refined region starts as two strips within five ramp
widths of the interfaces and thickens with the rolls, monotonically, at
91% of the uniform fine mesh's cells and the same answer. The seeded mode
grows from 0.0100 to 0.1235 at a fitted rate of 2.58, below both the
incompressible bounds of 4.38 and 5.92; every conserved integral holds to
roundoff through the regrids while the same run without the interface
fixup leaks by five to six orders of magnitude — this being the tracked
mesh the blast could not provide, since here the whole domain is in motion.
And HLLE against HLLC is not close: the shear layer *is* a contact, HLLE's
two-wave average is what smears it, and HLLC at half the linear resolution
is further along than HLLE at full resolution. HLLC becomes this case's
default; the package-wide default stays HLLE, which is *the* GRMHD flux.

There is one test suite and it runs whole, on every push: the unit tests
and every physics claim the measured results above rest on — the
convergence sweeps, the interface-order tables, the refinement
calibration, the tracked shock tube, the blast and its similarity law, the
shear layer and its growth rate. About four minutes.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The same numbers are expected at any thread count, and CI runs the suite
at one and at four. `Pkg.test` does not inherit `-t`, so the count has to
be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

**And there are pictures.** `bin/` holds the viewers, in an environment of
their own so that CairoMakie is never a dependency of the package. They
contain no time-stepping loop: every frame and every curve comes through
the one driver's `observer` hook, which exists for exactly that. The first
call instantiates the environment, and the `[sources]` entries mean no
manual `Pkg.develop`:

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

See [CODE.md](CODE.md) for the design document — the equations, the
scheme, the cases, and the measured results as they arrive — and
[PLAN.md](PLAN.md) for the work breakdown.
