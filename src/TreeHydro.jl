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

*Status: milestones H1, H2 and H3 done, and the atmosphere reset with
them. The scheme runs and conserves on
a mesh that follows the solution: the equation of state, the two state
conversions and the floors, the MUSCL reconstruction with its three
limiters, the three Riemann solvers, the six-step right-hand side with its
three kernels, SSPRK33 in time with the reset in its limiter hook, the
interface flux restriction that makes
a coarse-fine face conserve, the Löhner refinement criterion with its
calibrated thresholds, and the one chunked evolve-and-regrid driver that a
case is data for — with the entropy wave measuring second order and
conservation to roundoff, and Sod's shock tube tracked against the exact
Riemann solution through the Dirichlet boundary hook, matching the
uniformly fine reference at fewer cells. The Sedov and Kelvin–Helmholtz
cases are still to come, and Sedov is the first case in which a floor
actually fires.*

See `CODE.md` in the package root for the design document — what each
piece is for and why it is that way — and `PLAN.md` for the work
breakdown.
"""
module TreeHydro

using TreeAMR

using KernelAbstractions: Backend, CPU, allocate, get_backend, synchronize
using KernelAbstractions: @kernel, @index, @Const
# Unused before step 3 and depended on from step 0, so that the Julia floor
# the two of them set is fixed before anything relies on it.
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ODEProblem, solve

# Devices
export hostcopy

# Floors and the atmosphere: the two rules, the reset of the conserved
# state that applies them from the integrator's limiter hook, and the
# host-side record of what it injected
export Floors, apply_floors
export ResetAccounting, reset_atmosphere!

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
export max_signal_speed, floor_hits, ghost_floor_hits, hydro_dt
export conserved_totals, conserved_scales, hydro_solve!
export forest_levels, convergence_rate

# The refinement criterion: the indicator, its two global references, the
# flag vector `regrid!` takes, and the buffer width around what fired
export lohner, cell_tau, indicator_scales, hydro_flags, refinement_buffer

# The driver: a case as data, the one evolve-and-regrid loop, the uniform
# reference it is judged against, and the three measurements around them
export HydroCase, evolve!, uniform_run
export check_cfl, tracked_share, reduce_to_grid, l1_difference

# The entropy wave: the mesh, the exact cell averages, the study
export EntropyWave, hydro_forest
export fill_entropywave_averages!, entropywave_reference, entropywave_errors

# The exact Riemann solution: the shock tube's reference, and not a flux
export ExactRiemann, exact_riemann, sample

# Sod's shock tube: the case, its Dirichlet boundary, the study
export SodTube, sod_initial, sod_conserved, sod_boundary
export sod_forest, sod_reference, assert_no_arrival, sod_errors

# The Sedov–Taylor similarity law: the blast's reference, and not a method
export SedovSimilarity, sedov_alpha, sedov_exponent, sedov_radius
export sedov_profile, exponent_fit

# The Sedov blast: the case, its Dirichlet boundary on every face, and the
# three things a run of it is read for
export SedovBlast, sedov_state, ambient_state, sedov_initial, sedov_conserved
export sedov_boundary, sedov_forest, sedov_similarity
export measured_E₀, shock_radius, peak_compression, sedov_static

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
# The refinement criterion reads the primitive set a `HydroProblem` holds
# and takes the problem itself, so it follows the file that defines one.
# It is the mesh's other half of the driver, and nothing in the cases
# below needs it.
include("refinement.jl")
# The one time-stepping loop, and the case struct that is its only
# argument. It follows the criterion because it calls it, and precedes the
# cases because each of them constructs a `HydroCase` of its own — which is
# the direction the dependency has to run if the driver is to know nothing
# case-specific.
include("driver.jl")
include("entropywave.jl")
# The shock tube and the host `Float64` reference it is judged against. The
# solver comes first because the case reads it: `λ` and the reference both
# come out of one `ExactRiemann`.
include("exact_riemann.jl")
include("sod.jl")
# The blast and the similarity law it is judged against, in the same order and
# for the same reason: the case reads the reference — `λ`'s bound, the arrival
# check and the exponent all come out of one `SedovSimilarity`.
include("sedov_reference.jl")
include("sedov.jl")

end # module TreeHydro
