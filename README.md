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

**Status: the scheme conserves across coarse-fine faces — milestones H1
and H2 are done, and the refinement criterion is next.** What exists is
the module shell, the `Base` bridges for software floating-point types,
the host-copy helpers, the tests that say the pinned TreeAMR still
provides what the scheme is written against, the ideal-gas equation of
state, the conversions between the conserved and primitive states, the two
floor rules, the MUSCL reconstruction with its three slope limiters, the
LLF, HLLE and HLLC fluxes, the six-step right-hand side over the mesh with
its three kernels, SSPRK33 in time, and two cases on a static two-level
mesh. The entropy wave is an exact solution of the nonlinear system, and it
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
control of 2.02. Still missing: the refinement criterion, the driver, the
atmosphere reset, and the Sedov and Kelvin–Helmholtz cases.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

See [CODE.md](CODE.md) for the design document — the equations, the
scheme, the cases, and the measured results as they arrive — and
[PLAN.md](PLAN.md) for the work breakdown.
