# The time integrator (added with the move to IMEXRungeKutta):
# `src/stepping.jl`, `SSPRK33` with its stage arithmetic by block owner.
#
# What the rest of the suite cannot say about it: every convergence rate,
# every conservation bound and the thread workload already run *through*
# it, so they claim its results; what they do not claim is that the
# partition is the ownership partition (a partition that covers the state
# but hands a block's entries to another thread is invisible in every
# value), that the by-owner path is bitwise its own broadcast — which is
# the path a device state takes — and that the scratch `evolve!` hands from
# one chunk to the next changes no bit.

# IMEXRungeKutta through the package's own binding: it is a dependency of
# the package and not of the test environment.
const IRK = TreeHydro.IRK

@kernel function stepping_owner_kernel!(tid)
    I = @index(Global, NTuple)
    tid[I[end]] = Threads.threadid()
end

# A two-level entropy wave: smooth, floors nowhere, and has coarse-fine
# faces, so the fixup is on the path.
function stepping_wave(::Type{T}, ::Val{D}; N=8, roots=4) where {T,D}
    w = EntropyWave(T, Val(D))
    forest = hydro_forest(Val(D), N; roots=roots, L=w.L, refined=true, T=T)
    U = FieldSet{T}(forest, D + 2; G=2)
    p = HydroProblem(U, evolution_ops(); eos=w.eos, floors=w.floors,
                     limiter=:mc, riemann=:hlle)
    fill_entropywave_averages!(U, w)
    u = statevector(U)
    gather!(u, U)
    update_primitives!(p, u)
    dt = hydro_dt(forest, T(2 // 5), max_signal_speed(p), Val(D))
    return (U=U, p=p, u=u, dt=dt)
end

# The broadcast integrator: the same problem, tableau and hooks, with
# `partition = nothing`.
function stepping_broadcast(p, u, t1, nsteps)
    integ = TreeHydro.hydro_integrator(p, u, zero(t1), t1, nsteps; partition=nothing)
    IRK.solve!(integ)
    return integ.u
end

@testset verbose = true "The integrator (IMEXRungeKutta's SSPRK33 by owner)" begin
    T = Float64

    # Guards a partition that covers the state exactly — so IMEXRungeKutta
    # accepts it and every result is still right — but gives a block's
    # entries to another thread than the one `map_blocks!` runs the block
    # on, which is the cross-core traffic the ownership exists to remove.
    @testset "the partition is TreeAMR's block ownership: D=$D" for D in (1, 2)
        s = stepping_wave(T, Val(D))
        part = TreeHydro.state_partition(s.U, s.u)
        @test length(part) == Threads.nthreads()
        @test reduce(vcat, collect.(part)) == 1:length(s.u)
        L = s.U.forest.N^D * s.U.nvars
        @test all(r -> isempty(r) || (first(r) - 1) % L == 0 && length(r) % L == 0,
                  part)
        tid = zeros(Int, nblocks(s.U))
        map_blocks!(stepping_owner_kernel!, s.U, tid)
        offset = Threads.threadpoolsize(:interactive)
        if Threads.nthreads() > 1
            for (c, r) in enumerate(part), b in unique(cld.(r, L))
                @test tid[b] == offset + c
            end
        end
        # A device state takes the broadcast path, which a partition would
        # be refused on; anything but an `Array` stands for it here.
        @test TreeHydro.state_partition(s.U, view(s.u, :)) === nothing
        @test_throws DimensionMismatch TreeHydro.state_partition(s.U, s.u[1:(end - 1)])
    end

    # Guards a stage arithmetic that differs by path — a partition that
    # drops, doubles or reorders an entry — on a smooth two-level run and
    # on one where both limiter hooks fire in every step.
    @testset "by owner is bitwise the broadcast: D=$D" for D in (1, 2)
        s = stepping_wave(T, Val(D))
        nsteps = 5
        owner = hydro_solve!(s.p, s.u, zero(T), nsteps * s.dt, nsteps)
        @test isequal(owner, stepping_broadcast(s.p, s.u, nsteps * s.dt, nsteps))
        @test all(isfinite, owner) && owner != s.u

        v = vacuum_setup(T, Val(D); roots=4)
        update_primitives!(v.p, v.u)
        dt = hydro_dt(v.forest, T(2 // 5), max_signal_speed(v.p), Val(D))
        hits0 = v.p.accounting.hits
        owner = hydro_solve!(v.p, v.u, zero(T), 3dt, 3)
        @test v.p.accounting.hits > hits0          # the hooks did fire
        @test isequal(owner, stepping_broadcast(v.p, v.u, 3dt, 3))
    end

    # Guards `hydro_solve!` stepping its argument in place — the callers in
    # `sod.jl`, `sedov.jl` and the tests reuse it — and a step count the
    # integrator derived differently from the one the driver asked for.
    @testset "hydro_solve! leaves its input alone and takes the steps it is told" begin
        s = stepping_wave(T, Val(2))
        before = copy(s.u)
        out = hydro_solve!(s.p, s.u, zero(T), 3s.dt, 3)
        @test s.u == before && out !== s.u
        integ = TreeHydro.hydro_integrator(s.p, s.u, zero(T), 3s.dt, 3)
        @test integ.nsteps == 3 && integ.u !== s.u
        integ = TreeHydro.hydro_integrator(s.p, s.u, zero(T), 3s.dt, 3; alias_u0=true)
        @test integ.u === s.u
        @test_throws "nsteps must be at least 1" TreeHydro.hydro_integrator(
            s.p, s.u, zero(T), s.dt, 0)
    end

    # Guards the scratch reuse (IMEXRungeKutta 1.2's `reuse`) in `evolve!`:
    # an integrator that takes over the previous chunk's scratch must not
    # allocate it again — the point of passing it — and must step exactly
    # as one with fresh scratch does, since no scratch value carries over
    # between steps. And another mesh's state must be refused, not silently
    # given scratch of the wrong length.
    @testset "the next chunk takes the previous chunk's scratch" begin
        s = stepping_wave(T, Val(2))
        w = copy(s.u)
        i1 = TreeHydro.hydro_integrator(s.p, w, zero(T), 2s.dt, 2; alias_u0=true)
        IRK.solve!(i1)
        fresh = hydro_solve!(s.p, w, 2s.dt, 4s.dt, 2)
        mk() = TreeHydro.hydro_integrator(s.p, w, 2s.dt, 4s.dt, 2; alias_u0=true,
                                          reuse=i1)
        mk()
        nbytes = sizeof(w)
        @test @allocated(mk()) < nbytes
        @test @allocated(TreeHydro.hydro_integrator(s.p, w, 2s.dt, 4s.dt, 2;
                                                    alias_u0=true)) > 3nbytes
        i2 = mk()
        IRK.solve!(i2)
        @test isequal(i2.u, fresh)
        other = stepping_wave(T, Val(2); roots=2)
        @test_throws ArgumentError TreeHydro.hydro_integrator(
            other.p, other.u, zero(T), other.dt, 1; alias_u0=true, reuse=i1)
    end
end
