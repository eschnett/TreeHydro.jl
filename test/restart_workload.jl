# The workload behind the test that a restart at another thread count is
# bit-identical (added 2026-09-29, with checkpoint and restart).
#
# Run as a standalone script —
#
#     julia -t N --project=<an environment with HDF5> test/restart_workload.jl FILE
#
# — it restarts the run below from the checkpoint `FILE` and prints a digest
# of everything the restarted run returns: the state vector, the mesh, the
# floor counts, the injection, the drift and the four histories. The parent,
# `checkpoint_tests.jl`, prints the same digest of its own *uninterrupted*
# run at its own thread count, and the two must agree character for
# character. So the claim is not only that a checkpoint loads on any thread
# count — TreeAMR's claim — but that the run *continued* there is the run
# that never stopped.
#
# The run is the one two-dimensional configuration in the suite that is
# cheap, tracked, and floors cells through `evolve!`: a Sedov blast at
# `maxlevel_cap = 2` with `buffer = 0`, so that the shock reaches the
# coarse-fine faces its margin would otherwise keep it from — 328 owned
# resets, 152 ghost entries and a nonzero energy injection by `t = 1/50`,
# measured when this was written — and so every accumulator the file carries
# is exercised, not only the state.
#
# Like `thread_workload.jl` it may use nothing outside `Base`, TreeAMR, HDF5
# and the package, and it prints with `repr`, which round-trips a `Float64`
# exactly, so that a difference in the last bit is a difference in the text.

using HDF5
using TreeAMR
using TreeHydro

restart_digest(u::AbstractVector) = string(hash(Array(u)); base=16, pad=16)

"""The run a restart continues: a tracked `D = 2` blast that floors cells."""
restart_run(; kwargs...) =
    evolve!(HydroCase(SedovBlast(Float64, Val(2); r₀=1 // 16); roots=4), Val(2);
            N=8, ops=Operators(family=Conservative, prolongation=3, restriction=2),
            t_end=1 // 50, chunk=1 // 400, limiter=:minmod, refine_tol=2 // 25,
            coarsen_tol=1 // 50, maxlevel_cap=2, buffer=0, accounting=true,
            kwargs...)

"""Everything a finished run returns, as lines to compare character for character."""
function restart_lines(r)
    return [string("finished=", r.finished, " t=", repr(r.t), " chunk=", r.chunk,
                   " u=", restart_digest(r.u),
                   " leaves=", string(hash(repr(r.forest.leaves)); base=16)),
            string("nsteps=", r.nsteps, " regrids=", r.nregrids, " floors=",
                   r.floor_hits, " resets=", r.reset_hits, " ghosts=", r.ghost_hits),
            string("injection=", repr(r.injection), " drift=", repr(r.drift)),
            string("scales=", repr(r.scales), " totals0=", repr(r.totals0),
                   " totals=", repr(r.totals)),
            string("tracking=", repr(r.tracking), " λ_initial=", repr(r.λ_initial)),
            string("blocks=", r.nblocks_history, " buffers=", r.buffer_history),
            string("λ=", repr(r.λ_history), " λ_end=", repr(r.λ_end_history))]
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    foreach(println, restart_lines(restart_run(; restart_file=only(ARGS))))
end
