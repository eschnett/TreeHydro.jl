# TreeHydro.jl

[![CI](https://github.com/eschnett/TreeHydro.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeHydro.jl/actions/workflows/CI.yml)

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

Four cases are planned, each measuring something the others cannot: a
smooth entropy wave (an exact solution, so the convergence order and the
interface-order rule can be measured), Sod's shock tube (against an exact
Riemann solver), the Sedov blast (a strong shock through the floors and
the atmosphere reset, in 3D), and the Kelvin–Helmholtz instability (a
contact-dominated flow, refinement following a structure that grows, and
the picture).

Every numerical method is chosen to have a direct counterpart in a
relativistic MHD code, which is what the package rehearses; methods that
only work for Newtonian hydrodynamics are avoided even where they would be
better here.

**Status: the mesh now follows the solution — milestones H1, H2 and H3
are done; the atmosphere reset is next.** What exists is
the module shell, the `Base` bridges for software floating-point types,
the host-copy helpers, the tests that say the pinned TreeAMR still
provides what the scheme is written against, the ideal-gas equation of
state, the conversions between the conserved and primitive states, the two
floor rules, the MUSCL reconstruction with its three slope limiters, the
LLF, HLLE and HLLC fluxes, the six-step right-hand side over the mesh with
its three kernels, SSPRK33 in time, the Löhner refinement criterion, the
one chunked evolve-and-regrid driver, and two cases on it. The entropy
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
Still missing: the atmosphere reset, and the Sedov and Kelvin–Helmholtz
cases.

The tests come in two tiers. The **short** one is the default and is what
CI runs on every push: the unit tests, plus a reduced configuration of
every physics study compared against reference outputs committed under
`test/references/` to roundoff. About 45 seconds.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The **long** one adds `test/long/`: the convergence sweeps, the
interface-order tables, the refinement calibration and the tracked shock
tube — every claim the measured results above rest on. A minute and a
half, and it runs weekly rather than on every push, because the
two-dimensional sweeps are an order of magnitude slower on a shared
four-thread CI runner than on one thread.

```bash
TREEHYDRO_TEST_LONG=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

The references are regenerated only on request, and only by a run that
passed the physics claims first — so moved numbers arrive as a reviewed
diff and not as a silent rewrite.

```bash
TREEHYDRO_TEST_LONG=1 TREEHYDRO_REGENERATE=1 \
  julia --project=. -e 'using Pkg; Pkg.test()'
```

See [CODE.md](CODE.md) for the design document — the equations, the
scheme, the cases, and the measured results as they arrive — and
[PLAN.md](PLAN.md) for the work breakdown.
