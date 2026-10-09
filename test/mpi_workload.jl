# The workload behind the test that a distributed run is the serial run
# (added 2026-10-02, on TreeAMR's M7).
#
# Run under MPI —
#
#     mpiexec -n 3 julia --threads=1 --project=<env> test/mpi_workload.jl OUTDIR
#
# — it runs every case below three times in one launch, so that one
# compilation serves three rank counts: over all three ranks; then over a
# communicator of ranks 0 and 1, while rank 2 runs the same cases over a
# communicator of itself alone and then once more with no communicator at
# all. Each group's first rank writes the group's lines to `OUTDIR/n<k>.txt`
# (`n0.txt` being the serial run), and `mpi_tests.jl` compares the four
# files with the serial lines it computes in its own process.
#
# Two kinds of line. A line that does not start with `~` is a claim of
# **bit-identity**: the state vector gathered in curve order, the leaves,
# every step, chunk and floor count, every speed, the tracking measure,
# every maximum — none of which is a floating-point sum, and all of which
# TreeAMR's exchange and this package's reductions make independent of the
# rank count. A line that starts with `~` holds sums — the conserved totals,
# the drift, the injection, an L1 norm, McNally's `M` — which a partition
# reassociates, and is compared to roundoff. A line that starts with `#`
# depends on the rank count (the part files of a checkpoint) and is not
# compared at all; the workload asserts what it must say itself.
#
# Like `thread_workload.jl` it may use nothing outside `Base`, TreeAMR,
# MPI and the package. Its digests are a fold of `hash` over **every**
# element, not `hash` of the vector, which samples a long one.

using MPI
using TreeAMR
using TreeHydro

# Every element, in curve order: over a distributed forest the ranks' state
# vectors concatenated in rank order are the serial state vector, because
# each rank holds a contiguous run of the curve.
function mpi_digest(comm, u::AbstractVector)
    full = TreeAMR.allgatherv(comm, Array(u))
    return string(foldl((h, x) -> hash(x, h), full; init=UInt(0x5eed)); base=16,
                  pad=16)
end

mpi_leaves(forest) =
    string(foldl((h, k) -> hash(k, h), forest.leaves; init=UInt(0x1eaf)); base=16,
           pad=16)

mpi_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

# The observer every tracked run is watched through: at `t = 0` and at every
# chunk, the state, the mesh, the speed and the floor counts — all exact —
# and the totals, which are sums.
function mpi_watch(lines, tag)
    return (p, t, u) -> begin
        comm = p.U.forest.comm
        push!(lines, string(tag, " t=", repr(t), " u=", mpi_digest(comm, u),
                            " leaves=", mpi_leaves(p.U.forest),
                            " blocks=", nleaves(p.U.forest),
                            " λ=", repr(max_signal_speed(p)),
                            " floors=", floor_hits(p), " ghosts=", ghost_floor_hits(p)))
        push!(lines, string("~", tag, " t=", repr(t), " totals=",
                            repr(conserved_totals(p.U))))
    end
end

# Everything `evolve!` returns that is a claim about the run, split by kind.
function mpi_result(lines, tag, r)
    comm = r.forest.comm
    push!(lines, string(tag, " finished=", r.finished, " t=", repr(r.t),
                        " chunk=", r.chunk, " u=", mpi_digest(comm, r.u),
                        " leaves=", mpi_leaves(r.forest)))
    push!(lines, string(tag, " nsteps=", r.nsteps, " regrids=", r.nregrids,
                        " passes=", r.passes, " floors=", r.floor_hits,
                        " resets=", r.reset_hits, " ghosts=", r.ghost_hits,
                        " tracking=", repr(r.tracking)))
    push!(lines, string(tag, " blocks=", r.nblocks_history, " buffers=",
                        r.buffer_history, " λ=", repr(r.λ_history), " λ_end=",
                        repr(r.λ_end_history), " λ_initial=", repr(r.λ_initial),
                        " linf=", repr(r.linf)))
    push!(lines, string("~", tag, " drift=", repr(r.drift), " scales=",
                        repr(r.scales), " totals=", repr(r.totals),
                        " injection=", repr(r.injection), " l1=", repr(r.l1)))
    return r
end

mpi_sod_case() = HydroCase(SodTube(Float64, Val(1)); roots=(8,))

const MPI_SOD = (N=8, ops=mpi_ops(), t_end=1 // 50, chunk=1 // 200, limiter=:minmod,
                 refine_tol=2 // 25, coarsen_tol=1 // 50, maxlevel_cap=2,
                 accounting=true)

# The tracked blast `restart_workload.jl` restarts: two-dimensional, floors
# owned and ghost cells, and injects energy, so every accumulator a
# checkpoint carries is exercised.
mpi_blast_case() = HydroCase(SedovBlast(Float64, Val(2); r₀=1 // 16); roots=4)

const MPI_BLAST = (N=8, ops=mpi_ops(), t_end=1 // 100, chunk=1 // 400,
                   limiter=:minmod, refine_tol=2 // 25, coarsen_tol=1 // 50,
                   maxlevel_cap=2, buffer=0, accounting=true)

"""
The cases, over `comm` (`nothing` for the serial run), as lines; `dir` is
where this group writes its checkpoints and `from` the directory of the
three-rank group's, which the others restart from.

- **Sod in `D = 1`, tracked**: the initial-data cycle, the Dirichlet hook in
  its three places, a regrid every chunk that moves the partition.
- **Sod on two blocks**: at three ranks one rank holds no block for the
  whole run — the reductions, the integrator, the reset and the ghost count
  on an empty rank.
- **The shear layer, tracked**, and its **reflecting half box**: the
  two-dimensional exchange with the whole domain in motion, McNally's
  diagnostics, and mirrored transfers between ranks, where the parity
  factor's `−0` is what a pack applied too early would lose.
- **The blast on the static `:center` mesh**: the floors fire, both
  populations and the injection, across a coarse-fine face.
- **The entropy wave on its two-level mesh**: the interface restriction.
- **The tracked blast with checkpoints**, uninterrupted, and restarted from
  the three-rank group's checkpoint at chunk 2: a checkpoint written at one
  rank count continued at another is the run that never stopped.
- **A wall-time stop**, decided from each rank's own clock and agreed.
- **Rotation**, with one I/O process per rank so that a checkpoint has part
  files, and only the newest index and its own parts left behind.
- **A refusal on some ranks only**, which must be a refusal on all.
"""
function mpi_cases(comm, dir, from)
    lines = String[]
    rank = TreeAMR.commrank(TreeAMR.communicator(comm))

    mpi_result(lines, "sod", evolve!(mpi_sod_case(), Val(1); MPI_SOD..., comm=comm,
                                     observer=mpi_watch(lines, "sod")))

    mpi_result(lines, "sod2",
               uniform_run(HydroCase(SodTube(Float64, Val(1)); roots=(2,)), Val(1);
                           N=16, ops=mpi_ops(), t_end=1 // 50, chunk=1 // 100,
                           limiter=:minmod, accounting=true, comm=comm,
                           observer=mpi_watch(lines, "sod2")))

    kh = kh_run(Float64, Val(2); N=8, ops=mpi_ops(), chunk=1 // 200, maxlevel_cap=2,
                refine_tol=2 // 25, coarsen_tol=1 // 50, t_end=3 // 200, comm=comm,
                observer=mpi_watch(lines, "kh"))
    mpi_result(lines, "kh", kh.r)
    push!(lines, string("kh K=", repr(kh.Ks), " blocks=", kh.nbs))
    push!(lines, string("~kh M=", repr(kh.Ms)))

    half = kh_run(Float64, Val(2); N=8, ops=mpi_ops(), chunk=1 // 200,
                  maxlevel_cap=2, refine_tol=2 // 25, coarsen_tol=1 // 50,
                  t_end=3 // 200, half=true, seed=:mirrored, comm=comm,
                  observer=mpi_watch(lines, "half"))
    mpi_result(lines, "half", half.r)
    push!(lines, string("half K=", repr(half.Ks), " blocks=", half.nbs))
    push!(lines, string("~half M=", repr(half.Ms)))

    sedov = sedov_static(Float64, Val(2); N=8, ops=mpi_ops(), roots=4, r₀=1 // 16,
                         t_end=1 // 20, refined=:center, comm=comm)
    push!(lines, string("sedov nsteps=", sedov.nsteps, " u=",
                        mpi_digest(sedov.forest.comm, sedov.u), " floors=",
                        sedov.floor_hits, " resets=", sedov.reset_hits, " ghosts=",
                        sedov.ghost_hits, " r_s=", repr(sedov.r_s), " peak=",
                        repr(sedov.peak), " λ_end=", repr(sedov.λ_end)))
    push!(lines, string("~sedov drift=", repr(sedov.drift), " injection=",
                        repr(sedov.injection)))

    ew = entropywave_errors(Val(2); N=8, ops=mpi_ops(), refined=true, t_end=1 // 20,
                            comm=comm)
    push!(lines, string("entropy nsteps=", ew.nsteps, " linf=", repr(ew.linf),
                        " floors=", ew.floor_hits, " levels=", ew.levels))
    push!(lines, string("~entropy l1=", repr(ew.l1), " drift=", repr(ew.drift)))

    # The checkpointed blast, uninterrupted. Every chunk is written, none
    # deleted, so that the other groups find chunk 2's file in `from`; one
    # I/O process per rank, so that the three-rank group's checkpoint is an
    # index and three part files, which the others then read.
    prefix = joinpath(dir, "blast")
    mpi_result(lines, "blast",
               evolve!(mpi_blast_case(), Val(2); MPI_BLAST..., comm=comm,
                       checkpoint_path_prefix=prefix, checkpoint_every_chunks=1,
                       num_checkpoints_keep=100, checkpoint_io=:all,
                       checkpoint_sync_to_disk=false))
    # And continued from the three-rank group's checkpoint at chunk 2: these
    # lines must be the uninterrupted run's above, whichever group wrote it.
    second = TreeHydro.checkpoint_files(joinpath(from, "blast"))[2][2]
    mpi_result(lines, "blast",
               evolve!(mpi_blast_case(), Val(2); MPI_BLAST..., comm=comm,
                       restart_file=second))

    # Stopped by the wall clock after its first chunk, on every rank: a
    # limit no chunk fits under, so that the decision is the same whatever
    # each clock reads — and agreed, so that it would be the same if not.
    stopped = evolve!(mpi_sod_case(), Val(1); MPI_SOD..., comm=comm,
                      checkpoint_path_prefix=joinpath(dir, "stop"),
                      max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
    push!(lines, string("stop finished=", stopped.finished, " chunk=", stopped.chunk,
                        " u=", mpi_digest(stopped.forest.comm, stopped.u),
                        " written=", length(stopped.checkpoints_written)))

    # Rotation with part files: one I/O process per rank, one file kept.
    rot = joinpath(dir, "rot")
    evolve!(mpi_sod_case(), Val(1); MPI_SOD..., comm=comm,
            checkpoint_path_prefix=rot, checkpoint_every_chunks=1,
            num_checkpoints_keep=1, checkpoint_io=:all, checkpoint_sync_to_disk=false)
    if rank == 0
        names = readdir(dir)
        indexes = filter(n -> occursin(r"^rot\.it\d+\.h5$", n), names)
        parts = filter(n -> occursin(r"^rot\.it\d+\.h5\.[0-9a-f]{32}\.\d+\.h5$", n),
                       names)
        stale = filter(n -> !startswith(n, only(indexes) * "."), parts)
        push!(lines, string("rotation indexes=", length(indexes), " stale=",
                            length(stale)))
        push!(lines, string("# rotation parts=", length(parts)))
    end

    # A restart file that exists on rank 0 and not elsewhere: the ranks that
    # refuse it and the one that does not must all stop, and nobody wait. A
    # group of one has no other rank to disagree with, and says so.
    if TreeAMR.commsize(TreeAMR.communicator(comm)) > 1
        bad = rank == 0 ? second : joinpath(dir, "absent.h5")
        refused = try
            evolve!(mpi_blast_case(), Val(2); MPI_BLAST..., comm=comm,
                    restart_file=bad)
            false
        catch err
            err isa ArgumentError || rethrow()
            true
        end
        push!(lines, "refusal agreed=$refused")
    else
        push!(lines, "refusal agreed=true")
    end
    return lines
end

# The lines of a group, written by its first rank.
function write_lines(path, lines)
    open(path, "w") do io
        foreach(l -> println(io, l), lines)
    end
    return nothing
end

function mpi_main(out)
    MPI.Init()
    world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(world)
    MPI.Comm_size(world) == 3 || error("the workload runs at three ranks")
    ckpt(n) = mkpath(joinpath(out, "ckpt$n"))
    rank == 0 && foreach(ckpt, 0:3)
    MPI.Barrier(world)

    lines = mpi_cases(world, ckpt(3), ckpt(3))
    rank == 0 && write_lines(joinpath(out, "n3.txt"), lines)

    # Ranks 0 and 1 together, and rank 2 alone; the three-rank group's
    # checkpoints are complete, its collective calls having returned.
    sub = MPI.Comm_split(world, rank < 2 ? 0 : 1, rank)
    n = MPI.Comm_size(sub)
    lines = mpi_cases(sub, ckpt(n), ckpt(3))
    MPI.Comm_rank(sub) == 0 && write_lines(joinpath(out, "n$n.txt"), lines)
    if rank == 2
        write_lines(joinpath(out, "n0.txt"), mpi_cases(nothing, ckpt(0), ckpt(3)))
    end
    MPI.Barrier(world)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        mpi_main(only(ARGS))
    catch err
        # One rank's exception would leave the others waiting in their next
        # collective: take the job down instead.
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
end
