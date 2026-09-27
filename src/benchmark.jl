# Where a hydrodynamic step's time goes, and how much of it threads (step
# 14, H6c; after TreeWave's `src/benchmark.jl`).
#
# TreeAMR's own `bench/threads.jl` measures the mesh — scatter, ghost fill,
# a toy right-hand side, the norm. This measures the application on top of
# it: a whole `SSPRK33` step through IMEXRungeKutta, split into the passes of
# `hydro_rhs!`, the two limiter hooks, the integrator's own stage arithmetic
# (by owner and by broadcast), and the once-per-chunk diagnostics a driver
# pays between steps. `scatter` and `fill_ghosts` are kept as the phases
# common to both tables, so the two can be read against each other.
#
# Nothing here prints: `src/` stays free of I/O, and `bin/benchmark.jl`
# formats what these return.

"""
    best(f, reps; backend = CPU())

The shortest of `reps` timings of `f`, after one warm-up call. The minimum
is what a scaling study wants: noise only ever adds time, so the fastest
run is the one least contaminated by everything else on the machine.

Each timing ends with a `synchronize`, which is a no-op on the CPU and the
difference between a measurement and a fiction on a device, where a launch
and a broadcast both return before the work is done. The passes that go
through the mesh synchronize on their own; the integrator's stage
arithmetic is a broadcast and does not.
"""
function best(f, reps; backend::Backend=CPU())
    f()
    synchronize(backend)
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed begin
                    f()
                    synchronize(backend)
                end)
    end
    return t
end

# The two meshes the phase table runs on. `:wave` is the entropy wave on a
# periodic box — smooth, no floor ever fires, no physical boundary — which is
# the simple setup a block-size scan wants; `:sedov` is the blast on its
# Dirichlet box, where the boundary hook runs and the floors fire, which is
# what a production step pays. `refined = true` refines the middle sub-box
# (the centre, for the blast) once, so the coarse-fine faces are on the path.
function benchmark_mesh(::Type{T}, ::Val{D}, case::Symbol; N, roots,
                        refined::Bool) where {T,D}
    if case === :wave
        w = EntropyWave(T, Val(D))
        forest = hydro_forest(Val(D), N; roots=roots, L=w.L, refined=refined, T=T)
        return w, forest, HydroCase(w; roots=roots), nothing
    elseif case === :sedov
        w = SedovBlast(T, Val(D))
        forest = sedov_forest(Val(D), N; roots=roots, L=w.L,
                              refined=refined ? :center : false, T=T)
        return w, forest, HydroCase(w; roots=roots), sedov_boundary(w)
    end
    throw(ArgumentError(
        "case must be :wave or :sedov, got :$case: the phase table runs on the " *
        "entropy wave, which is the simple periodic mesh a scan wants, or on " *
        "the Sedov blast, which has the boundary hook and the floors."))
end

"""
    benchmark_phases([T], ::Val{D}; N, roots, case = :wave, refined = false,
                     reps = 5, steps = 4, limiter = :minmod, riemann = :hlle,
                     cfl = 2//5, ops, backend = CPU())

Seconds per phase of a hydrodynamic step on a `roots^D` forest of `N^D`
blocks, uniform or with its middle refined once. Returns
`(sizes = (; nblocks, cells, statebytes, workbytes), timings = [name => seconds, …])`
with the phases in the order a step visits them; the caller formats.

The phases, and why each is here:

- `step` — one `SSPRK33` step through [`hydro_integrator`](@ref) with the
  stage arithmetic by owner and the default `:stage` reset, as the cost of
  `steps` steps divided by `steps`, on an integrator that takes over the
  previous one's scratch as [`evolve!`](@ref) does, so that allocating the
  scratch is not in the number. This is the number that matters.
- `step_broadcast` — the same with `partition = nothing`: the integrator's
  broadcast path, which is what a device takes and what OrdinaryDiffEq's
  serial stage arithmetic amounted to. The gap to `step` is what the
  ownership partition buys.
- `rhs` — one [`hydro_rhs!`](@ref). Three make up the bulk of a step.
- `scatter`, `fill_ghosts`, `con2prim`, `flux`, `fixup`, `divergence` —
  the passes of `hydro_rhs!` in its own order, `flux` and `fixup` summed
  over the `D` directions. They add up to `rhs` less launch overhead.
- `reset` — one atmosphere reset kernel with its hit count, as each limiter
  hook calls it; `:stage` calls it three times per step.
- `integrator` — `step − 3·rhs − 3·reset`, the stage combinations and the
  integrator's own bookkeeping, derived rather than timed: it is the serial
  term a threaded step would be capped by if it did not thread.
- `max_signal_speed`, `floor_hits`, `ghost_floor_hits`, `conserved_totals`,
  `hydro_flags` — what [`evolve!`](@ref) pays once per chunk between steps.
- `problem` — rebuilding the [`HydroProblem`](@ref) with its ghost and
  interface schedules, which `evolve!` pays after every regrid that changes
  the mesh.
"""
function benchmark_phases(::Type{T}, ::Val{D}; N, roots, case::Symbol=:wave,
                          refined::Bool=false, reps=5, steps=4, limiter=:minmod,
                          riemann=:hlle, cfl=2 // 5,
                          ops=Operators(family=Conservative, prolongation=3,
                                        restriction=2),
                          backend::Backend=CPU()) where {T,D}
    bestof(f, n=reps) = best(f, n; backend=backend)

    w, forest, hcase, boundary = benchmark_mesh(T, Val(D), case; N=N, roots=roots,
                                                refined=refined)
    U = FieldSet{T}(forest, D + 2; G=2, backend=backend)
    p = HydroProblem(U, ops; eos=hcase.eos, floors=hcase.floors, limiter=limiter,
                     riemann=riemann, boundary=boundary)
    fill_by_coordinates!(conserved_initial(hcase), U)
    u = statevector(U)
    gather!(u, U)
    du = similar(u)
    update_primitives!(p, u)
    dt = hydro_dt(forest, T(cfl), hcase.speed_headroom * max_signal_speed(p), Val(D))
    t1 = steps * dt

    # One integrator per timing, each taking over the scratch of the one
    # before, as `evolve!` builds one per chunk. The state is stepped in
    # place, so every repetition starts where the last one ended; on the
    # blast that moves the shock a few cells over the whole table, which
    # changes nothing a timing can see.
    function stepper(partition)
        prev = Ref{Any}(nothing)
        return () -> begin
            integ = hydro_integrator(p, u, zero(T), t1, steps; alias_u0=true,
                                     reuse=prev[], partition=partition)
            IRK.solve!(integ)
            prev[] = integ
            nothing
        end
    end
    # Each stepper is called once before `best` sees it, so that `best`'s own
    # warm-up already takes the `reuse` path the timed calls take; otherwise
    # the first timed call compiles it.
    owner, bcast = stepper(state_partition(U, u)), stepper(nothing)
    owner()
    bcast()
    t_step = bestof(owner) / steps
    t_bcast = bestof(bcast) / steps

    t_rhs = bestof(() -> hydro_rhs!(du, u, p, zero(T)))
    t_scatter = bestof(() -> scatter!(U, u))
    t_ghosts = bestof(() -> fill_ghosts!(U, p.schedule; boundary=boundary))
    con2prim!() = map_blocks!(con2prim_kernel!, p.P, p.P.work, U.work, p.eos,
                              p.floors, p.valD; stored=true)
    t_con2prim = bestof(con2prim!)
    t_flux = bestof() do
        ntuple(Val(D)) do d
            map_blocks!(flux_kernel!, p.fluxes[d], p.fluxes[d].work, p.P.work,
                        p.eos, p.floors, p.valD, p.valGP, p.valGF, Val(d),
                        p.limiter, p.solver; closed=true)
            nothing
        end
    end
    t_fixup = bestof() do
        ntuple(d -> (restrict_interfaces!(p.fluxes[d], p.ischeds[d]); nothing),
               Val(D))
    end
    t_div = bestof() do
        map_blocks!(divergence_kernel!, U, statearray(du, U),
                    map(f -> f.work, p.fluxes), p.spacings, p.valD, p.valGF)
    end
    # On a copy: the reset writes back where a floor fires, and the state
    # the diagnostics below read should be the one the step left. A
    # `statevector` filled by `gather!`, not `copy(u)`: `copy` first-touches
    # every page from the calling thread, which on a NUMA node puts the whole
    # copy in one domain and timed the reset at 12× on 64 threads where the
    # owner-placed arrays a real run resets are not so placed (measured on
    # Symmetry in step 14).
    v = statevector(U)
    gather!(v, U)
    t_reset = bestof(() -> reset_stage!(v, nothing, p, zero(T)))

    update_primitives!(p, u)
    t_λ = bestof(() -> max_signal_speed(p))
    t_floor = bestof(() -> floor_hits(p))
    t_ghostfloor = bestof(() -> ghost_floor_hits(p))
    t_totals = bestof(() -> conserved_totals(U))
    t_flags = bestof() do
        hydro_flags(p; refine_tol=T(2 // 25), coarsen_tol=T(1 // 50),
                    maxlevel_cap=maxlevel(forest) + 1)
    end
    t_problem = bestof(max(2, reps ÷ 2)) do
        HydroProblem(U, ops; eos=hcase.eos, floors=hcase.floors, limiter=limiter,
                     riemann=riemann, boundary=boundary, prims=p.P, fluxes=p.fluxes)
    end

    sizes = (nblocks=nblocks(U), cells=nblocks(U) * N^D, statebytes=sizeof(u),
             workbytes=sizeof(U.work) + sizeof(p.P.work) +
                       sum(f -> sizeof(f.work), p.fluxes))
    timings = ["step" => t_step, "step_broadcast" => t_bcast, "rhs" => t_rhs,
               "scatter" => t_scatter, "fill_ghosts" => t_ghosts,
               "con2prim" => t_con2prim, "flux" => t_flux, "fixup" => t_fixup,
               "divergence" => t_div, "reset" => t_reset,
               "integrator" => t_step - 3 * t_rhs - 3 * t_reset,
               "max_signal_speed" => t_λ, "floor_hits" => t_floor,
               "ghost_floor_hits" => t_ghostfloor,
               "conserved_totals" => t_totals, "hydro_flags" => t_flags,
               "problem" => t_problem]
    return (sizes=sizes, timings=timings)
end
benchmark_phases(::Val{D}; kwargs...) where {D} = benchmark_phases(Float64, Val(D); kwargs...)

"""
    benchmark_driver([T], ::Val{D}; reps = 2, backend = CPU())

Wall time of a whole tracked Sedov blast through [`evolve!`](@ref) — the
initial-data cycle, every chunk and every regrid — at the configuration
`test/sedov_tests.jl` tracks (`roots = 4`, `N = 8`, a level cap of 2 in
`D = 2`; `N = 4` and a cap of 1 in `D = 3`). Returns
`(seconds, nsteps, nregrids, nblocks, cell_updates)`. `cell_updates` is the
sum over chunks of the cells of that chunk's mesh times the mean step count
per chunk — the numerator for a mesh whose size changes, approximate only in
taking the steps as evenly spread, which on a blast whose `λ` falls they
nearly are.

The phase table cannot see what lies between steps — regridding, the
schedules rebuilt after it, the criterion, the host-side diagnostics — and
on a small adaptive mesh that is where a threaded run's time goes.
"""
function benchmark_driver(::Type{T}, ::Val{D}; reps=2,
                          backend::Backend=CPU()) where {T,D}
    cfg = D == 3 ? (N=4, cap=1, r₀=1 // 8, chunk=1 // 300, t_end=1 // 20) :
          (N=8, cap=2, r₀=1 // 16, chunk=1 // 400, t_end=1 // 10)
    w = SedovBlast(T, Val(D); r₀=cfg.r₀)
    run() = evolve!(T, HydroCase(w; roots=4), Val(D); N=cfg.N,
                    ops=Operators(family=Conservative, prolongation=3,
                                  restriction=2),
                    t_end=cfg.t_end, chunk=cfg.chunk, limiter=:minmod,
                    refine_tol=2 // 25, coarsen_tol=1 // 50,
                    maxlevel_cap=cfg.cap, backend=backend)
    r = run()
    seconds = best(run, reps; backend=backend)
    nchunks = length(r.nblocks_history)
    per_chunk = r.nsteps / nchunks                  # steps are near-uniform
    cell_updates = round(Int, per_chunk * sum(r.nblocks_history) * cfg.N^D)
    return (seconds=seconds, nsteps=r.nsteps, nregrids=r.nregrids,
            nblocks=r.nblocks, cell_updates=cell_updates)
end
benchmark_driver(::Val{D}; kwargs...) where {D} = benchmark_driver(Float64, Val(D); kwargs...)
