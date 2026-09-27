# Running on a device (step 14, H6c): a whole run on a device reproduces
# the host run at the same type.
#
# The mesh decides where the work runs from where the storage is, so every
# driver takes a `backend` beside its `T`, and the integrator follows the
# state: a device state takes IMEXRungeKutta's broadcast path, since
# `state_partition` hands it no partition. What only a device can check is
# that a run there builds the same mesh, takes the same steps and floors the
# same cells as the host run — a device that flagged or floored differently
# would still run, and just build a different mesh. That needs a device
# package, which is deliberately not a dependency of this package or of
# TreeAMR; add one to an environment of your own and name it:
#
#     julia --project=/tmp/thgpu -e 'using Pkg; Pkg.develop(path = ".");
#                                     Pkg.add(["Metal", "MultiFloats"])'
#     TREEHYDRO_TEST_BACKEND=metal julia --project=/tmp/thgpu test/runtests.jl
#
# Unset, the file runs its comparisons with the CPU standing in for the
# device, which is what CI does: the claims are then trivially exact, and
# what is exercised is the plumbing — the backend keyword reaching every
# field set, and the state vector coming back on the backend it was asked
# for.

using KernelAbstractions: CPU, Backend, get_backend

const DEVICE_NAME = lowercase(get(ENV, "TREEHYDRO_TEST_BACKEND", ""))

if DEVICE_NAME == "cuda"
    using CUDA
elseif DEVICE_NAME == "metal"
    using Metal
elseif !isempty(DEVICE_NAME)
    error("TREEHYDRO_TEST_BACKEND must be \"cuda\" or \"metal\", got " *
          "\"$DEVICE_NAME\"")
end

# The device, or the CPU standing in for it. `Float32` either way: it is
# the type every device runs and the one Metal is limited to.
const DEVICE = let
    device = if DEVICE_NAME == "cuda"
        CUDA.functional() ? CUDABackend() : nothing
    elseif DEVICE_NAME == "metal"
        Metal.functional() ? MetalBackend() : nothing
    end
    if device === nothing && !isempty(DEVICE_NAME)
        @info "TREEHYDRO_TEST_BACKEND=$DEVICE_NAME is not functional here; " *
              "the CPU stands in for it"
    end
    device === nothing ? CPU() : device
end

const DEVICE_OPS = Operators(family=Conservative, prolongation=3, restriction=2)

device_sod(backend) =
    evolve!(Float32, HydroCase(SodTube(Float32, Val(1)); roots=(8,)), Val(1); N=8,
            ops=DEVICE_OPS, t_end=1 // 20, chunk=1 // 200, limiter=:minmod,
            refine_tol=2 // 25, coarsen_tol=1 // 50, maxlevel_cap=2,
            accounting=true, backend=backend)

device_sedov(backend) =
    sedov_static(Float32, Val(2); N=8, ops=DEVICE_OPS, roots=4, r₀=1 // 16,
                 t_end=1 // 20, refined=:center, backend=backend)

@testset "A run on $(nameof(typeof(DEVICE))) reproduces the host run at Float32" begin
    # Guards a device path that runs and answers differently: a kernel that
    # reads a host array (which the CPU forgives), a reduction whose
    # device form flags a different cell, a limiter hook that does not reach
    # the device state, or an integrator that takes the host partition for a
    # device array. Mesh histories, step counts and every floor count are
    # compared **exactly**; the states only to a tolerance, because a device
    # contracts `a*b + c` into an `fma` where the host may not.
    host, dev = device_sod(CPU()), device_sod(DEVICE)
    @test get_backend(dev.u) == DEVICE
    @test dev.nblocks_history == host.nblocks_history
    @test dev.nsteps == host.nsteps && dev.nregrids == host.nregrids
    @test dev.floor_hits == host.floor_hits == 0
    @test dev.injection == host.injection == (0.0f0, 0.0f0, 0.0f0)
    @test abs(dev.l1 - host.l1) ≤ host.l1 / 100
    @info "tracked Sod at Float32 on $(nameof(typeof(DEVICE))): L1 $(dev.l1) " *
          "against the host's $(host.l1), $(dev.nsteps) steps, blocks " *
          "$(dev.nblocks_history)"

    # The blast: the one run where both limiter hooks fire, on the device's
    # stage values.
    host, dev = device_sedov(CPU()), device_sedov(DEVICE)
    @test get_backend(dev.u) == DEVICE
    @test dev.nsteps == host.nsteps
    @test host.reset_hits > 0
    @test dev.reset_hits == host.reset_hits
    @test dev.ghost_hits == host.ghost_hits
    @test dev.floor_hits == host.floor_hits
    @test abs(dev.r_s - host.r_s) ≤ host.r_s / 100
    @test abs(dev.peak - host.peak) ≤ host.peak / 100
    @info "static Sedov at Float32 on $(nameof(typeof(DEVICE))): " *
          "$(dev.reset_hits) resets against the host's $(host.reset_hits), " *
          "$(dev.ghost_hits) ghost hits against $(host.ghost_hits), r_s " *
          "$(dev.r_s) against $(host.r_s), peak $(dev.peak) against $(host.peak)"
end
