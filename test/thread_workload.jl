# The workload behind the thread-count independence test (step 13).
#
# Run as a standalone script —
#
#     julia -t N --project=. test/thread_workload.jl
#
# — it prints a digest of everything three short runs produce: the state
# vector at every chunk, the floor counts, the fastest signal, the conserved
# totals and the mesh. Two runs at different thread counts must print the
# same lines, character for character. That is the property the code has:
# every kernel runs through TreeAMR's `map_blocks!` and writes its own
# cells, every reduction goes through `block_mapreduce` and is combined on
# the host in block order, the integrator's stage arithmetic writes each
# block's entries on the block's owner, and nothing in this package threads
# a loop of its own.
#
# TreeAMR since 0.1.3 promises floating-point *sums* only to roundoff
# across thread counts (its CPU fold is still ordered, so today they are
# bit-identical too). The totals and the injection below are such sums, and
# they are the only lines `CODE.md`'s "Multi-threading" allows an ulp
# tolerance on, should upstream's fold ever change; every other line is
# exact by construction.
#
# It drives the package's own `evolve!` and `sedov_static` rather than a
# loop of its own — the observer is how a watcher reads a run without one —
# and it may use nothing outside `Base`, TreeAMR and the package, because it
# has to run from the root environment as well as from `test/`. Hence
# `hash` rather than `sha256`, and `repr`, which round-trips a `Float64`
# exactly, so a difference in the last bit is a difference in the text.

using TreeAMR
using TreeHydro

digest(u::AbstractVector) = string(hash(Array(u)); base=16, pad=16)

workload_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

"""
A tracked Sod tube in `D = 1`, a short Kelvin–Helmholtz on its tracked mesh,
and a static two-level Sedov blast in `D = 2`, reduced to printed lines.

The three cover everything that threads. Sod gives the initial-data
adaptation, the Dirichlet hook in all three places and a regrid every chunk;
the shear layer is the two-dimensional tracked mesh with the whole domain in
motion and McNally's host-side diagnostics; and the blast is the one run
where the floors fire — both limiter hooks, the owned and the ghost counts,
and a nonzero injection.
"""
function thread_digests()
    lines = String[]

    watch(tag) = (p, t, u) -> push!(lines, string(
        tag, " t=", repr(t), " u=", digest(u), " blocks=", nblocks(p.U),
        " λ=", repr(max_signal_speed(p)), " floors=", floor_hits(p),
        " ghosts=", ghost_floor_hits(p), " totals=", repr(conserved_totals(p.U))))

    sod = evolve!(HydroCase(SodTube(Float64, Val(1)); roots=(8,)), Val(1); N=8,
                  ops=workload_ops(), t_end=1 // 50, chunk=1 // 200,
                  limiter=:minmod, refine_tol=2 // 25, coarsen_tol=1 // 50,
                  maxlevel_cap=2, accounting=true, observer=watch("sod"))
    push!(lines, string("sod nsteps=", sod.nsteps, " regrids=", sod.nregrids,
                        " tracking=", repr(sod.tracking), " l1=", repr(sod.l1),
                        " drift=", repr(sod.drift), " injection=",
                        repr(sod.injection), " history=", sod.nblocks_history))

    kh = kh_run(Float64, Val(2); N=8, ops=workload_ops(), chunk=1 // 200,
                maxlevel_cap=2, refine_tol=2 // 25, coarsen_tol=1 // 50,
                t_end=3 // 200, observer=watch("kh"))
    push!(lines, string("kh nsteps=", kh.r.nsteps, " M=", repr(kh.Ms),
                        " K=", repr(kh.Ks), " injection=", repr(kh.r.injection),
                        " history=", kh.r.nblocks_history))

    sedov = sedov_static(Float64, Val(2); N=8, ops=workload_ops(), roots=4,
                         r₀=1 // 16, t_end=1 // 20, refined=:center)
    push!(lines, string("sedov nsteps=", sedov.nsteps, " u=", digest(sedov.u),
                        " floors=", sedov.floor_hits, " resets=", sedov.reset_hits,
                        " ghosts=", sedov.ghost_hits, " injection=",
                        repr(sedov.injection), " r_s=", repr(sedov.r_s),
                        " peak=", repr(sedov.peak)))
    return lines
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && foreach(println, thread_digests())
