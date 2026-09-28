# Running on a device (step 14, H6c): every driver the suite measures with
# runs on a device and reproduces the host run at the same type.
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
#                                     Pkg.add(["Metal", "KernelAbstractions",
#                                              "TreeAMR", "MultiFloats"])'
#     TREEHYDRO_TEST_BACKEND=metal julia --project=/tmp/thgpu test/runtests.jl
#
# Unset, the file runs its comparisons with the CPU standing in for the
# device, which is what CI does: the claims are then trivially exact, and
# what is exercised is the plumbing — the backend keyword reaching every
# field set, and the state vector coming back on the backend it was asked
# for.
#
# The file covers each *driver* once rather than each claim of the suite
# again (widened after step 14, which had the tracked tube and the static
# blast only): `entropywave_errors`, `sod_errors`, `sedov_static`,
# `evolve!` on all three cases and `kh_run`, in `D = 1, 2, 3`, on uniform,
# static two-level and tracked meshes. The physics claims themselves are
# the host suite's; what this file claims is that the device answers them
# the same way. See "Step 14" in `CODE.md`.

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

# The device, or the CPU standing in for it.
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

const DEVICE_LABEL = string(nameof(typeof(DEVICE)))

# `Float32` everywhere: it is the type every device runs and the only one
# Metal has. `Float64` as well on CUDA, which is the type the host suite
# makes its claims at — so there the device answers *those* runs, not
# merely their single-precision shadows. Not on the CPU stand-in, where a
# second type would double CI's cost for a comparison that is exact by
# construction.
const DEVICE_TYPES = DEVICE_NAME == "cuda" && DEVICE isa Backend && !(DEVICE isa CPU) ?
                     (Float32, Float64) : (Float32,)

const DEVICE_OPS = Operators(family=Conservative, prolongation=3, restriction=2)

# The agreement asked of a continuous result — an error norm, a shock
# radius, a mode amplitude: `≈` at its default for the run's type,
# `sqrt(eps(T))`, and never at the type of the number compared (the shear
# layer's `M` comes back as a `Float64` from a `Float32` run). Integers
# (steps, blocks, floor counts) and mesh histories are compared exactly and
# never get one. A device may contract `a*b + c` into an `fma` where the
# host does not, and its `sin` and `exp` are not Julia's — Metal's `Float32`
# `sin` differs from the host's in the last bit on 42% of arguments, which
# is what makes the shear layer's seeded `S_y` differ where nothing else
# does. An *error* norm is a difference of order-one states, so it carries a
# relative noise of `eps(T)/err` and not `eps(T)`: on CUDA at `Float32` the
# two-level entropy wave's L1 moved by 4.3e-5 of itself (measured after step
# 14), which is why the tolerance is `sqrt(eps)` rather than a few `eps`.
device_close(dev, host, ::Type{T}) where {T} = isapprox(dev, host; rtol=sqrt(eps(T)))

# Two drifts of a conserved total agree to the roundoff bound the host suite
# puts on a drift, `8 eps(T) · scale · nsteps`. A drift is a difference of
# two order-one totals, so a relative tolerance on it would be meaningless —
# the reason given under "Base's reductions" in `CLAUDE.md`, which the
# device's reductions share. A total whose own scale is zero — the momentum
# across a planar tube, which the host keeps exactly zero — borrows the
# largest scale of the state: CUDA writes `1.5e-36` there where the host
# writes `0.0` (measured after step 14), and a yardstick of zero would call
# that a leak.
function device_drift_bound(r, v, ::Type{T}) where {T}
    scale = r.scales[v] > 0 ? r.scales[v] : maximum(r.scales)
    return 8 * eps(T) * scale * r.nsteps
end

function device_same_drift(dev, host, ::Type{T}) where {T}
    return all(v -> abs(dev.drift[v] - host.drift[v]) ≤
                    device_drift_bound(host, v, T), eachindex(host.drift))
end

# The stored cells of a final state whose pressure, recovered from `U` as
# `con2prim` recovers it, lies within `k` ulp of `p_floor` on either side:
# the cells whose floor flag is a coin flip that an `fma` can turn. They are
# a real population and not a curiosity — a cell the reset put *at* the floor
# recovers a pressure a fraction of an ulp away from it (step 8's finding
# that the flag is not idempotent), and on the three-dimensional blast they
# are 48 owned cells and 240 ghost entries at `Float32`. Returned as
# `(owned, ghost)`.
function borderline_floor_cells(r, ::Val{D}; k=4) where {D}
    U, G, N = r.U.work, r.U.G, size(r.U.work)[1:D] .- 2 .* r.U.G
    γ, p_floor = r.w.eos.γ, r.w.floors.p_floor
    T = eltype(U)
    owned = ghost = 0
    for b in axes(U, D + 2), I in CartesianIndices(size(U)[1:D])
        ρ = U[I, 1, b]
        S² = sum(d -> U[I, 1 + d, b]^2, 1:D)
        p = (γ - 1) * (U[I, D + 2, b] - S² / (2ρ))
        abs(p - p_floor) ≤ k * eps(T) * p_floor || continue
        all(d -> G[d] < I[d] ≤ G[d] + N[d], 1:D) ? (owned += 1) : (ghost += 1)
    end
    return (owned, ghost)
end

device_sod(::Type{T}, backend) where {T} =
    evolve!(T, HydroCase(SodTube(T, Val(1)); roots=(8,)), Val(1); N=8,
            ops=DEVICE_OPS, t_end=1 // 20, chunk=1 // 200, limiter=:minmod,
            refine_tol=2 // 25, coarsen_tol=1 // 50, maxlevel_cap=2,
            accounting=true, backend=backend)

device_sedov(::Type{T}, backend; D=2, N=8, t_end=1 // 20, r₀=1 // 16,
             refined=:center, reset=:stage) where {T} =
    sedov_static(T, Val(D); N=N, ops=DEVICE_OPS, roots=4, r₀=r₀, t_end=t_end,
                 refined=refined, reset=reset, backend=backend)

@testset "A run on $DEVICE_LABEL reproduces the host run at $T" for T in DEVICE_TYPES
    # Guards a device path that runs and answers differently: a kernel that
    # reads a host array (which the CPU forgives), a reduction whose
    # device form flags a different cell, a limiter hook that does not reach
    # the device state, or an integrator that takes the host partition for a
    # device array. Mesh histories, step counts and every floor count are
    # compared **exactly**; the states only to a tolerance, because a device
    # contracts `a*b + c` into an `fma` where the host may not.
    host, dev = device_sod(T, CPU()), device_sod(T, DEVICE)
    @test get_backend(dev.u) == DEVICE
    @test dev.nblocks_history == host.nblocks_history
    @test dev.nsteps == host.nsteps && dev.nregrids == host.nregrids
    @test dev.floor_hits == host.floor_hits == 0
    @test dev.injection == host.injection == (zero(T), zero(T), zero(T))
    @test device_close(dev.l1, host.l1, T)
    @info "tracked Sod at $T on $DEVICE_LABEL: L1 $(dev.l1) " *
          "against the host's $(host.l1), $(dev.nsteps) steps, blocks " *
          "$(dev.nblocks_history)"

    # The blast: the one run where both limiter hooks fire, on the device's
    # stage values.
    host, dev = device_sedov(T, CPU()), device_sedov(T, DEVICE)
    @test get_backend(dev.u) == DEVICE
    @test dev.nsteps == host.nsteps
    @test host.reset_hits > 0
    @test dev.reset_hits == host.reset_hits
    @test dev.ghost_hits == host.ghost_hits
    @test dev.floor_hits == host.floor_hits
    @test device_close(dev.r_s, host.r_s, T)
    @test device_close(dev.peak, host.peak, T)
    @info "static Sedov at $T on $DEVICE_LABEL: " *
          "$(dev.reset_hits) resets against the host's $(host.reset_hits), " *
          "$(dev.ghost_hits) ghost hits against $(host.ghost_hits), r_s " *
          "$(dev.r_s) against $(host.r_s), peak $(dev.peak) against $(host.peak)"
end

@testset "The coarse-fine faces conserve on $DEVICE_LABEL at $T" for T in DEVICE_TYPES
    # Guards the interface flux restriction on a device: the one kernel of
    # TreeAMR's M8 this package exists to exercise, and the one whose
    # absence a device run would not notice — the run completes, the error
    # moves a little, and the totals leak. On the static two-level mesh in
    # `D = 2` and `D = 3`, the fixup run's drift is inside the host suite's
    # roundoff bound on the device, and the `fixup = false` control leaks
    # there by the same amount it leaks on the host, which is what says the
    # restriction is the thing the device ran and not a no-op on both.
    for (D, N, t_end) in ((2, 8, 1 // 4), (3, 4, 1 // 8))
        common = (; N=N, ops=DEVICE_OPS, roots=4, refined=true, limiter=:none,
                  t_end=t_end)
        host = entropywave_errors(T, Val(D); common...)
        dev = entropywave_errors(T, Val(D); common..., backend=DEVICE)
        @test dev.levels == host.levels == [0, 1]
        @test (dev.nsteps, dev.nblocks) == (host.nsteps, host.nblocks)
        @test dev.floor_hits == host.floor_hits == 0
        @test device_close(dev.l1, host.l1, T)
        @test device_close(dev.linf, host.linf, T)
        @test all(v -> dev.drift[v] ≤ device_drift_bound(dev, v, T), 1:(D + 2))

        host_nofix = entropywave_errors(T, Val(D); common..., fixup=false)
        dev_nofix = entropywave_errors(T, Val(D); common..., fixup=false,
                                       backend=DEVICE)
        @test device_same_drift(dev_nofix, host_nofix, T)
        # At `Float64` the control's leak clears the bound by the host
        # suite's millionfold; at `Float32` the bound is `eps` larger and the
        # leak is not, so the separation is a `Float64` claim only.
        T == Float64 && @test all(v -> dev_nofix.drift[v] >
                                       1e6 * device_drift_bound(dev, v, T), 1:(D + 2))
        @info "entropy wave, two-level D = $D, at $T on $DEVICE_LABEL: " *
              "L1 $(dev.l1) against $(host.l1), drift $(dev.drift); " *
              "without the fixup $(dev_nofix.drift) against $(host_nofix.drift)"
    end

    # Sod across the coarse-fine face, with the Dirichlet hook: the drift is
    # the boundary's own numerical flux here and not roundoff (see "At a
    # physical boundary" in `CLAUDE.md`), so the claim is that the device's
    # drift *is* the host's.
    common = (; N=8, ops=DEVICE_OPS, roots=(4, 1), refined=:middle, t_end=1 // 20)
    host = sod_errors(T, Val(2); common...)
    dev = sod_errors(T, Val(2); common..., backend=DEVICE)
    @test dev.nsteps == host.nsteps && dev.floor_hits == host.floor_hits == 0
    @test device_close(dev.l1, host.l1, T)
    @test device_close(dev.λ_final, host.λ_final, T)
    @test device_same_drift(dev, host, T)
    @info "two-level Sod, D = 2, at $T on $DEVICE_LABEL: L1 $(dev.l1) against " *
          "$(host.l1), drift $(dev.drift) against $(host.drift)"
end

@testset "A tracked mesh on $DEVICE_LABEL builds the host's mesh at $T" for T in DEVICE_TYPES
    # Guards the regrid on a device: `hydro_flags`' two `firing_boxes`
    # sweeps, the boundary hook reaching `regrid!` and
    # `adapt_to_initial_data!` on a device field set, and the problem rebuilt
    # after each mesh change around the device arrays `regrid!` resized. A
    # device that flagged one cell differently would build a different mesh
    # and still run, which is why the mesh history is compared exactly and
    # first.
    drivers = (; ops=DEVICE_OPS, limiter=:minmod, refine_tol=2 // 25,
               coarsen_tol=1 // 50, maxlevel_cap=2, accounting=true)

    # The planar tube on a two-dimensional mesh, where the boxes have a
    # second extent to get wrong.
    sod2(backend) = evolve!(T, HydroCase(SodTube(T, Val(2)); roots=(8, 1)), Val(2);
                            N=8, t_end=1 // 20, chunk=1 // 200, drivers...,
                            backend=backend)
    host, dev = sod2(CPU()), sod2(DEVICE)
    @test get_backend(dev.u) == DEVICE
    @test dev.nblocks_history == host.nblocks_history
    @test (dev.nsteps, dev.nregrids, dev.passes) ==
          (host.nsteps, host.nregrids, host.passes)
    @test dev.tracking == host.tracking
    @test device_close(dev.l1, host.l1, T)
    @test device_same_drift(dev, host, T)

    # The blast through the chunked driver, with the Dirichlet hook on every
    # face and a mesh that grows at every regrid.
    sedov2(backend) = evolve!(T, HydroCase(SedovBlast(T, Val(2); r₀=1 // 16); roots=4),
                              Val(2); N=8, t_end=1 // 50, chunk=1 // 400,
                              drivers..., backend=backend)
    host, dev = sedov2(CPU()), sedov2(DEVICE)
    @test dev.nblocks_history == host.nblocks_history
    @test length(unique(host.nblocks_history)) > 1   # the mesh really moved
    @test (dev.nsteps, dev.nregrids) == (host.nsteps, host.nregrids)
    @test (dev.floor_hits, dev.reset_hits, dev.ghost_hits) ==
          (host.floor_hits, host.reset_hits, host.ghost_hits)
    @test device_same_drift(dev, host, T)
    @info "tracked Sedov, D = 2, at $T on $DEVICE_LABEL: blocks " *
          "$(dev.nblocks_history) against the host's $(host.nblocks_history), " *
          "$(dev.nsteps) steps"

    # The shear layer through `kh_run`, whose two diagnostics are host loops
    # the observer takes once per chunk: on a device they read a state that
    # has to come down first, and `M(t)` at every sample says it did. Its
    # initial `S_y` is `sin(4πx)`, which is where a device's `sin` shows —
    # the only case here whose *initial data* differs from the host's.
    kh(backend) = kh_run(T, Val(2); N=8, ops=DEVICE_OPS, chunk=1 // 200,
                         maxlevel_cap=2, refine_tol=2 // 25, coarsen_tol=1 // 50,
                         t_end=1 // 10, backend=backend)
    host, dev = kh(CPU()), kh(DEVICE)
    @test dev.nbs == host.nbs
    @test dev.r.nsteps == host.r.nsteps
    @test all(device_close.(dev.Ms, host.Ms, T))
    @test all(device_close.(dev.Ks, host.Ks, T))
    @test dev.r.injection == host.r.injection == (zero(T), zero(T), zero(T), zero(T))
    @test all(v -> dev.r.drift[v] ≤ device_drift_bound(dev.r, v, T), 1:4)
    @info "Kelvin–Helmholtz to t = 1/10 at $T on $DEVICE_LABEL: M $(dev.Ms[end]) " *
          "against the host's $(host.Ms[end]), worst relative difference over " *
          "the samples $(maximum(abs.(dev.Ms .- host.Ms) ./ host.Ms))"
end

@testset "The floors fire on the same cells on $DEVICE_LABEL in D = 3 at $T" for T in DEVICE_TYPES
    # Guards the floor populations in three dimensions, where the blast
    # crosses a coarse-fine face in every direction at once and the
    # prolongation fills edge and corner ghosts: the ghost floor count is
    # this package's one launch of its own, and on a device it is the count
    # most likely to see a different set of ghosts. Under both reset
    # cadences, since `:step` leaves the stage values unreset and so floors
    # a different population — measured after step 14 at `Float32`, 8520
    # resets and 4224 ghost hits under `:stage` against 2856, 4464 and 48
    # owned cells floored in the RHS under `:step`. (The `:edge` mesh at this
    # size floors ghosts alone, 160 of them, and no owned cell.)
    #
    # The owned cells the *reset* writes back are compared exactly. The two
    # flag counts — the RHS's owned floor hits and the ghost entries — are
    # not, and that is measured rather than conceded: on CUDA they came back
    # 23 and 4339 against the host's 48 and 4464 (`Float32`, `:step`), and
    # every difference lay inside the population of cells whose recovered
    # pressure is within 4 ulp of `p_floor`, 48 and 240 there; where that
    # population was empty (`Float64`, `:step`) the counts were equal. So the
    # claim is that a device floors the same cells *except* those, which is
    # as exact as a flag defined by a comparison at the floor can be. Metal,
    # which contracts nothing here, matched all of them exactly.
    for reset in (:stage, :step)
        host = device_sedov(T, CPU(); D=3, N=4, t_end=1 // 50, r₀=1 // 8,
                            refined=:center, reset=reset)
        dev = device_sedov(T, DEVICE; D=3, N=4, t_end=1 // 50, r₀=1 // 8,
                           refined=:center, reset=reset)
        @test host.reset_hits > 0 && host.ghost_hits > 0
        @test (dev.nsteps, dev.reset_hits) == (host.nsteps, host.reset_hits)
        border = max.(borderline_floor_cells(host, Val(3)),
                      borderline_floor_cells(dev, Val(3)))
        @test abs(dev.floor_hits - host.floor_hits) ≤ border[1]
        @test abs(dev.ghost_hits - host.ghost_hits) ≤ border[2]
        @test device_close(dev.r_s, host.r_s, T)
        @test device_same_drift(dev, host, T)
        @info "static Sedov, D = 3, reset = :$reset, at $T on " *
              "$DEVICE_LABEL: $(dev.reset_hits) resets, $(dev.floor_hits) owned " *
              "and $(dev.ghost_hits) ghost floor hits against $(host.reset_hits), " *
              "$(host.floor_hits) and $(host.ghost_hits); borderline cells " *
              "(owned, ghost) $border"
    end
end
