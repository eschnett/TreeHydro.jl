"""
    TreeHydro

Newtonian ideal hydrodynamics with a high-resolution shock-capturing
finite-volume scheme, as the *conservative* sample application for
[TreeAMR](https://github.com/eschnett/TreeAMR.jl):

    ∂ₜ ρ   + ∂_d (ρ v_d)             = 0
    ∂ₜ S_i + ∂_d (S_i v_d + p δ_id)  = 0
    ∂ₜ E   + ∂_d ((E + p) v_d)       = 0

TreeAMR supplies the mesh and contains no physics.
[TreeWave](https://github.com/eschnett/TreeWave.jl) shows that mesh under
a second-order finite-difference scheme, which needs nothing special at a
coarse-fine face; this package shows it under a finite-volume scheme,
which needs everything TreeAMR's M8 milestone added — the conservative
operator family, per-field-set ghost widths, ghost-free face-centered
flux fields, and the interface flux restriction that makes the scheme
conserve across refinement boundaries.

Every numerical method here is chosen to have a direct counterpart in a
relativistic MHD code, which is what the package rehearses: primitive
recovery as an explicit fallible step, an atmosphere reset, HLL-family
fluxes, MUSCL reconstruction of primitives, SSPRK33 in time. Methods that
only work for Newtonian hydrodynamics are avoided even where they would be
better here.

Every driver takes the floating-point type it computes in as a leading
positional argument and the KernelAbstractions backend it runs on as a
keyword, so the same study runs at `Float32` on a device as at `Float64`
on the host, and the answer is bit-identical at any thread count.

*Status: milestone H1 done. The scheme runs on a uniform mesh: the
equation of state, the two state conversions and the floors, the MUSCL
reconstruction with its three limiters, the three Riemann solvers, the
six-step right-hand side with its three kernels, SSPRK33 in time, the
entropy wave, which measures second order and conservation to roundoff,
and Sod's shock tube against the exact Riemann solution, with the
Dirichlet boundary hook. The coarse-fine faces, the refinement criterion
and the driver are still to come.*

See `CODE.md` in the package root for the design document — what each
piece is for and why it is that way — and `PLAN.md` for the work
breakdown.
"""
module TreeHydro

using TreeAMR

using KernelAbstractions: Backend, CPU, allocate, get_backend
using KernelAbstractions: @kernel, @index, @Const
# Unused before step 3 and depended on from step 0, so that the Julia floor
# the two of them set is fixed before anything relies on it.
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ODEProblem, solve

# Devices
export hostcopy

# Floors and the atmosphere
export Floors, apply_floors

# Equation of state and the two state conversions
export EquationOfState, IdealGas
export pressure, internal_energy, soundspeed
export statedims, density, velocity, momentum, pressure_of, energy
export prim2con, con2prim

# Reconstruction
export slope, face_states

# The physical flux and the three approximate Riemann solvers
export physical_flux, signal_speed, riemann_flux

# The right-hand side and what a driver needs around it
export HydroProblem, hydro_rhs!, update_primitives!
export max_signal_speed, floor_hits, hydro_dt
export conserved_totals, conserved_scales, hydro_solve!, convergence_rate

# The entropy wave: the mesh, the exact cell averages, the study
export EntropyWave, hydro_forest
export fill_entropywave_averages!, entropywave_reference, entropywave_errors

# The exact Riemann solution: the shock tube's reference, and not a flux
export ExactRiemann, exact_riemann, sample

# Sod's shock tube: the case, its Dirichlet boundary, the study
export SodTube, sod_initial, sod_conserved, sod_boundary
export sod_forest, sod_reference, assert_no_arrival, sod_errors

include("precision.jl")
include("device.jl")
# `floors.jl` before `eos.jl`: `con2prim` takes a `Floors` and says so in
# its signature, and a signature is evaluated where the method is defined.
# The reverse dependency — `apply_floors` reading a state through the
# accessors `eos.jl` defines — is resolved when it is called, not when it
# is compiled.
include("floors.jl")
include("eos.jl")
# The two halves of the flux kernel of step 3, in the order it calls them:
# `face_states` builds the pair of primitive states at a face, and
# `riemann_flux` turns that pair into the flux through it.
include("reconstruction.jl")
include("riemann.jl")
# The right-hand side that calls all of the above over a mesh, and the
# first case to run on it.
include("evolution.jl")
include("entropywave.jl")
# The shock tube and the host `Float64` reference it is judged against. The
# solver comes first because the case reads it: `λ` and the reference both
# come out of one `ExactRiemann`.
include("exact_riemann.jl")
include("sod.jl")

end # module TreeHydro
