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

**Status: design complete; the scaffolding and the per-cell physics of
the scheme are done; nothing solves anything yet.** What exists is the
module shell, the `Base` bridges for software floating-point types, the
host-copy helpers, the tests that say the pinned TreeAMR still provides
what the scheme is written against, and — from steps 1 and 2 — the
ideal-gas equation of state, the conversions between the conserved and
primitive states, the two floor rules, the MUSCL reconstruction with its
three slope limiters, and the LLF, HLLE and HLLC fluxes. They are
pointwise functions of `isbits` tuples, so they run in a kernel on any
backend; what is still missing is the right-hand side that calls them
over a mesh, which is step 3.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

See [CODE.md](CODE.md) for the design document — the equations, the
scheme, the cases, and the measured results as they arrive — and
[PLAN.md](PLAN.md) for the work breakdown.
