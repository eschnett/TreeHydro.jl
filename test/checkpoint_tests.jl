# Checkpoint and restart (added 2026-09-29, on TreeAMR 0.1.4's M9a).
#
# The claim is one sentence: **a restarted run, or a chain of restarts, is
# the uninterrupted run, bit for bit, at any thread count** — its state, its
# mesh, and every number `evolve!` returns about it — **unless it changes
# the regridding criterion, which its first regrid then uses** (amended
# 2026-10-01, when the checkpoint moved before the regrid). Everything below
# is either that claim on another configuration or a refusal that keeps a
# restart from silently becoming a different run.
#
# The failure modes, one per testset:
#
#   * a checkpoint taken anywhere but before the regrid, a restart whose
#     replayed regrid is not the loop's, or an accumulator left out of the
#     run state — the state would continue and the *answer* (a drift, a
#     floor count, a history) would not, and only a comparison of every
#     returned field notices;
#   * a restart with a changed criterion that does not regrid with it before
#     its next step, or does not say so;
#   * a finished run on a chunk boundary that cannot be continued, or is
#     continued into another run than the longer one;
#   * rotation deleting the wrong file, or a name that sorts wrongly;
#   * a real that does not round-trip at `Float32x2`, whose scalars TreeAMR's
#     plain data refuse;
#   * a two-dimensional restart whose interface schedules or ghosts are not
#     rebuilt as an uninterrupted run rebuilds them;
#   * a restart whose continuation depends on the thread count;
#   * a reflecting forest whose walls or parities do not survive a load, and
#     a file from before the recipe recorded them that no longer reads;
#   * a restart with another parameter, or from another application's file,
#     running instead of refusing.
#
# Every comparison is `==` or `isequal`, never `≈`: roundoff-level agreement
# is what a restart that replayed something differently would give.

# Guards the one refusal that has to be tested *before* HDF5 is loaded: a
# job script that forgot `using HDF5` must fail at the call, in a second,
# and not at the first write hours in. Once HDF5 is loaded in a session it
# cannot be unloaded, so this runs only where the suite reaches this file
# without it — which is every ordinary run, since no file above loads it.
if Base.get_extension(TreeAMR, :TreeAMRHDF5Ext) === nothing
    @testset "Checkpointing without HDF5 is refused at the call" begin
        case = HydroCase(SodTube(Float64, Val(1)); roots=(8,))
        err = try
            evolve!(case, Val(1); N=8,
                    ops=Operators(family=Conservative, prolongation=3, restriction=2),
                    t_end=1 // 50, chunk=1 // 200, limiter=:minmod,
                    refine_tol=2 // 25, coarsen_tol=1 // 50, maxlevel_cap=2,
                    checkpoint_path_prefix=joinpath(mktempdir(), "sod"),
                    checkpoint_every_chunks=1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("using HDF5", err.msg)
        @test !TreeHydro.checkpointing_available()
    end
end

# Each top-level statement of an included file runs in the latest world, so
# every testset below sees the methods the extension adds here.
using HDF5
using MultiFloats: Float32x2

include("restart_workload.jl")

const CKPT_OPS = Operators(family=Conservative, prolongation=3, restriction=2)

# The tracked tube of `driver_tests.jl`'s `SOD1D`, shortened to ten chunks.
const CKPT_SOD = (N=8, ops=CKPT_OPS, t_end=1 // 20, chunk=1 // 200, limiter=:minmod,
                  refine_tol=2 // 25, coarsen_tol=1 // 50, maxlevel_cap=2,
                  accounting=true)

ckpt_sod_case(::Type{T}=Float64) where {T} = HydroCase(SodTube(T, Val(1)); roots=(8,))

ckpt_sod(; kwargs...) = evolve!(ckpt_sod_case(), Val(1); CKPT_SOD..., kwargs...)

# Every field `evolve!` returns that is a claim about the run — everything
# but the three that say how this *call* went (`checkpoints_written`,
# `restart_file`, `criterion_changed`) and the `U` whose working array is
# compared through `u`.
const CKPT_FIELDS = (:u, :drift, :scales, :totals0, :totals, :floor_hits, :reset_hits,
                     :ghost_hits, :injection, :nsteps, :nchunks, :nregrids, :passes,
                     :converged, :nblocks, :nblocks_history, :buffer_history,
                     :levels, :cells, :tracking, :λ_initial, :λ_history,
                     :λ_end_history, :h, :l1, :linf, :finished, :t, :chunk)

function ckpt_same(a, b)
    same = true
    for k in CKPT_FIELDS
        isequal(getproperty(a, k), getproperty(b, k)) && continue
        same = false
        @info "restarted run differs" field = k restarted = getproperty(a, k) uninterrupted =
            getproperty(b, k)
    end
    return same && a.forest.leaves == b.forest.leaves
end

# A wall-time limit no chunk fits in: every call stops after the first chunk
# boundary that can write, so a chain of these restarts once per chunk.
const CKPT_STOP_NOW = 1e-9

"""
Run to the end as a chain of one-chunk jobs, each from the latest file, and
return every call's result with, for each call that stopped, the checkpoint
it wrote as `load_checkpoint` reads it — read at once, since the rotation of
the next call deletes it.
"""
function ckpt_chain(run, prefix; types=(), observer=nothing, kwargs...)
    calls = Any[]
    files = Any[]
    while isempty(calls) || !last(calls).finished
        r = run(; checkpoint_path_prefix=prefix, max_walltime_seconds=CKPT_STOP_NOW,
                checkpoint_sync_to_disk=false, restart_file=latest_checkpoint(prefix),
                observer=observer, kwargs...)
        push!(calls, r)
        r.finished || push!(files, load_checkpoint(only(r.checkpoints_written);
                                                   types=types))
    end
    return calls, files
end

function ckpt_refusal(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

const CKPT_REFERENCE_TS = Float64[]
const CKPT_REFERENCE = ckpt_sod(; observer=(p, t, u) -> push!(CKPT_REFERENCE_TS, t))

@testset "A chain of restarts is the uninterrupted run, bit for bit" begin
    # Guards the whole design: a checkpoint written anywhere but before the
    # regrid, a restart whose first regrid is not the one the loop would have
    # made there, a run-state field left out, or a restart that rebuilds
    # something the uninterrupted run carries over. Each of those continues
    # the state and changes an answer, and only a comparison of every
    # returned field says so.
    dir = mktempdir()
    prefix = joinpath(dir, "sod")
    ts = Float64[]
    calls, files = ckpt_chain(ckpt_sod, prefix; observer=(p, t, u) -> push!(ts, t))
    ref = CKPT_REFERENCE
    # One chunk per call, then the last call finishes.
    @test length(calls) == ref.nchunks == 10
    @test ckpt_same(last(calls), ref)
    # The claim is about a mesh that moves, and the floors are counted.
    @test ref.nregrids ≥ 1 && length(unique(ref.nblocks_history)) > 1
    for (i, r) in enumerate(calls[1:(end - 1)])
        # A wall-time stop: not finished, no answer at `t_end`, one chunk
        # further than the call before, and holding the state it wrote.
        @test !r.finished && r.chunk == i && r.l1 === nothing && r.linf === nothing
        @test r.t == CKPT_REFERENCE_TS[i + 1]
        @test length(r.checkpoints_written) == 1
        @test r.u == files[i].fieldsets["U"].state
        @test r.forest.leaves == files[i].forest.leaves
        @test files[i].data.run.chunk == i
        # Stopped before the chunk's regrid: on the mesh the chunk ran on,
        # with that regrid's margin not yet recorded.
        @test r.nblocks == ref.nblocks_history[i]
        @test r.buffer_history == ref.buffer_history[1:(i - 1)]
        @test r.criterion_changed == (i == 1 ? nothing : Symbol[])
        @test r.restart_file == (i == 1 ? nothing : only(calls[i - 1].checkpoints_written))
    end
    # The observer is not called at `t = 0` by a restart, so the chain's
    # calls, concatenated, are the uninterrupted run's.
    @test ts == CKPT_REFERENCE_TS
    @test first(ts) == 0 && count(iszero, ts) == 1
    # And the directory holds the default two files, the latest the ninth
    # chunk's: the wall-time limit never stops the last chunk, and nothing
    # else here would write there.
    @test length(TreeHydro.checkpoint_files(prefix)) == 2
    @test isempty(last(calls).checkpoints_written)
end

@testset "Rotation keeps the newest n files, named by iteration" begin
    # Guards the names and the rotation: a name that did not sort by
    # iteration, a rotation that deleted the file just written or kept too
    # many, or a pattern that matched a partial file or another prefix.
    dir = mktempdir()
    prefix = joinpath(dir, "sod")
    # An earlier job's file is rotated away like any other; a partial file
    # and another prefix's file are not this prefix's and are left alone.
    stale = prefix * ".it0000000000.h5"
    strangers = (prefix * ".it0000000001.h5.partial", prefix * "x.it0000000001.h5",
                 joinpath(dir, "other.it0000000001.h5"))
    foreach(touch, (stale, strangers...))
    r = ckpt_sod(; checkpoint_path_prefix=prefix, checkpoint_interval_seconds=0,
                 num_checkpoints_keep=3, checkpoint_sync_to_disk=false)
    @test ckpt_same(r, CKPT_REFERENCE)
    # A file at every chunk boundary, the last included: `t_end = 1/20` is ten
    # chunks of `1/200` exactly in `Float64`, so a longer run would have
    # regridded there.
    @test length(r.checkpoints_written) == r.nchunks == 10
    kept = TreeHydro.checkpoint_files(prefix)
    @test [path for (_, path) in kept] == r.checkpoints_written[(end - 2):end]
    @test !isfile(stale) && all(isfile, strangers)
    @test latest_checkpoint(prefix) == last(r.checkpoints_written)
    for (iteration, path) in kept
        saved = load_checkpoint(path).data.run
        # The name is the step count, and the file is the chunk boundary's.
        @test saved.nsteps == iteration
        @test path == TreeHydro.checkpoint_filename(prefix, iteration)
        @test saved.chunk in 8:10
    end
    @test issorted(first.(kept)) && allunique(first.(kept))

    # A chunk count writes at its multiples only.
    prefix2 = joinpath(dir, "every")
    r2 = ckpt_sod(; checkpoint_path_prefix=prefix2, checkpoint_every_chunks=2,
                  num_checkpoints_keep=10, checkpoint_sync_to_disk=false)
    @test [load_checkpoint(f).data.run.chunk for f in r2.checkpoints_written] ==
          [2, 4, 6, 8, 10]
    @test all(isfile, r2.checkpoints_written)
    @test latest_checkpoint(joinpath(dir, "none")) === nothing

    # A prefix spelled with a doubled separator lists its files under the
    # normalized path; the file just written must still be recognised as
    # itself, or `num_checkpoints_keep = 1` deletes the only checkpoint.
    prefix3 = joinpath(dir, "sub") * "//sod"
    mkpath(joinpath(dir, "sub"))
    r3 = ckpt_sod(; t_end=1 // 50, checkpoint_path_prefix=prefix3,
                  checkpoint_interval_seconds=0, num_checkpoints_keep=1,
                  checkpoint_sync_to_disk=false)
    @test length(r3.checkpoints_written) == 4
    @test isfile(last(r3.checkpoints_written))
    @test length(TreeHydro.checkpoint_files(prefix3)) == 1
end

@testset "A Float32x2 run restarts exactly" begin
    # Guards the limb path twice: the run state's reals, which TreeAMR's
    # plain data refuse as MultiFloat scalars and `plain_reals` stores as
    # limbs, and the field set, which TreeAMR stores as limbs itself.
    T = Float32x2
    run32(; kwargs...) = evolve!(ckpt_sod_case(T), Val(1); CKPT_SOD..., t_end=1 // 50,
                                 kwargs...)
    ref = run32()
    calls, files = ckpt_chain(run32, joinpath(mktempdir(), "sod32x2"); types=(T,))
    @test length(calls) == ref.nchunks == 4
    @test ckpt_same(last(calls), ref)
    @test eltype(last(calls).u) === T && last(calls).tracking isa T
    # And the reals really are limbs in the file, two `Float32` per value,
    # not a rounded `Float64`.
    @test files[1].data.run.drift isa Matrix{Float32}
    @test size(files[1].data.run.drift) == (2, 3)
    @test eltype(files[1].fieldsets["U"].state) === T
end

@testset "A two-dimensional run that floors cells restarts exactly" begin
    # Guards what a one-dimensional tube cannot: the interface schedules and
    # the corner ghosts rebuilt after a load, and every floor population —
    # owned resets, ghost hits, the injection — carried across it. The blast
    # with `buffer = 0` reaches its coarse-fine faces and floors there.
    ref = restart_run()
    @test ref.reset_hits > 0 && ref.ghost_hits > 0 && ref.injection[end] > 0
    @test ref.nregrids ≥ 1
    calls, _ = ckpt_chain(restart_run, joinpath(mktempdir(), "sedov"))
    @test length(calls) == ref.nchunks
    @test ckpt_same(last(calls), ref)
end

@testset "A restart at another thread count is the uninterrupted run" begin
    # Guards a continuation that depends on the thread count — a load placed
    # differently, a schedule rebuilt in another order — which no in-process
    # test can see. The subprocess restarts from this process's file, and
    # its digest must be this process's uninterrupted run's, character for
    # character.
    ref = restart_run()
    prefix = joinpath(mktempdir(), "sedov")
    r = restart_run(; checkpoint_path_prefix=prefix, checkpoint_every_chunks=4,
                    checkpoint_sync_to_disk=false)
    # Writing a checkpoint changes nothing about the run that writes it.
    @test restart_lines(r) == restart_lines(ref)
    # Chunks 4 and 8, the last on a chunk boundary; the subprocess continues
    # from the first.
    @test length(r.checkpoints_written) == 2
    file = first(r.checkpoints_written)
    other = Threads.nthreads() == 1 ? max(2, min(4, Sys.CPU_THREADS)) : 1
    script = joinpath(@__DIR__, "restart_workload.jl")
    project = Base.active_project()
    out = read(`$(Base.julia_cmd()) --threads=$other --project=$project $script $file`,
               String)
    lines = split(chomp(out), '\n')
    @test lines == restart_lines(ref)
    for (a, b) in zip(lines, restart_lines(ref))
        a == b || @info "restart differs at $other thread(s)" subprocess = a here = b
    end
end

# One checkpoint of the tube, at chunk 2 of 10 (`t = 1/100`), for the
# refusals below: the first of the five a run writes every two chunks, the
# last chunk's included — all kept, or the rotation would delete it.
const CKPT_FILE = first(ckpt_sod(; checkpoint_path_prefix=joinpath(mktempdir(), "sod"),
                                 checkpoint_every_chunks=2, num_checkpoints_keep=5,
                                 checkpoint_sync_to_disk=false).checkpoints_written)

@testset "A restart with another parameter is refused, naming it" begin
    # Guards the recipe: a restart with another chunk, limiter, block size or
    # Riemann solver would run — and be a different experiment that looks
    # like the old one. Each is refused with the parameter's name, and only
    # its name. (`maxlevel_cap` was the fourth until 2026-10-01; it is the
    # criterion's now, which a restart may change.)
    for (k, v) in ((:chunk, 1 // 100), (:limiter, :mc), (:N, 16), (:riemann, :hllc))
        err = ckpt_refusal(() -> ckpt_sod(; restart_file=CKPT_FILE, (k => v,)...))
        @test err isa ArgumentError
        @test occursin("`$k`", err.msg)
        others = filter(!=(k), (:chunk, :limiter, :N, :riemann))
        @test !any(o -> occursin("`$o`", err.msg), others)
    end
    # Two at once are named at once, so a job script is fixed in one round.
    err = ckpt_refusal(() -> ckpt_sod(; restart_file=CKPT_FILE, chunk=1 // 100,
                                      limiter=:mc))
    @test err isa ArgumentError && occursin("`chunk`", err.msg) &&
          occursin("`limiter`", err.msg)
    # A `Float32` case from a `Float64` file.
    err = ckpt_refusal(() -> evolve!(ckpt_sod_case(Float32), Val(1); CKPT_SOD...,
                                     restart_file=CKPT_FILE))
    @test err isa ArgumentError && occursin("`float_type`", err.msg)
    # A `t_end` at or before the checkpoint's time, `1/100`.
    for t_end in (1 // 200, 1 // 100)
        err = ckpt_refusal(() -> ckpt_sod(; restart_file=CKPT_FILE, t_end=t_end))
        @test err isa ArgumentError && occursin("lies at or before", err.msg)
    end
    # The same parameters spelled differently are the same run, and the
    # same criterion: nothing is reported as changed.
    r = ckpt_sod(; restart_file=CKPT_FILE, refine_tol=0.08, chunk=0.005)
    @test ckpt_same(r, CKPT_REFERENCE)
    @test r.criterion_changed == Symbol[]
end

@testset "A restart with another criterion regrids with it first" begin
    # Guards the one thing the checkpoint's place before the regrid is for: a
    # restart that changes the regridding criterion must regrid with the new
    # one *before its next step*, where the uninterrupted run regridded with
    # the old, and say so. A checkpoint after the regrid, or a restart that
    # stepped before regridding, would run the next chunk on the old mesh.
    # Chunk 3 is where the tube's mesh first moves, from 12 blocks to 16.
    ref = CKPT_REFERENCE
    @test ref.nblocks_history[3:4] == [12, 16]
    prefix = joinpath(mktempdir(), "crit")
    file = first(ckpt_sod(; checkpoint_path_prefix=prefix, checkpoint_every_chunks=3,
                          num_checkpoints_keep=4,
                          checkpoint_sync_to_disk=false).checkpoints_written)
    @test load_checkpoint(file).data.run.chunk == 3
    r = @test_logs (:info, r"`coarsen_tol` is 0\.02 in the checkpoint and 0\.05") ckpt_sod(;
        restart_file=file, coarsen_tol=1 // 20)
    @test r.criterion_changed == [:coarsen_tol]
    # Everything up to the checkpoint is the uninterrupted run's ...
    @test r.nblocks_history[1:3] == ref.nblocks_history[1:3]
    @test r.buffer_history[1:2] == ref.buffer_history[1:2]
    @test r.λ_history[1:3] == ref.λ_history[1:3]
    # ... and chunk 3's regrid, made by the restart before its first step, is
    # another: the wider coarsening band leaves 14 blocks where the old
    # criterion made 16, and the run goes on from there to another answer.
    @test r.nblocks_history[4] == 14
    @test r.nregrids == ref.nregrids + 1
    @test r.finished && r.nsteps == ref.nsteps && r.l1 != ref.l1
    # The margin too: `buffer = 0` from the same file records the restart's
    # own width at chunk 3's regrid, which then refines nothing.
    r0 = ckpt_sod(; restart_file=file, buffer=0)
    @test r0.criterion_changed == [:buffer]
    @test r0.buffer_history[1:3] == [ref.buffer_history[1:2]; 0]
    @test r0.nblocks_history[4] == 12
    # A run from the initial data has no criterion to change.
    @test ref.criterion_changed === nothing
end

@testset "A larger t_end continues the run, a finished one too" begin
    # Guards the other parameter a restart may change: a run to `t_end₁`,
    # continued from a checkpoint to `t_end₂`, is the run to `t_end₂` — from
    # one inside it, and (since 2026-10-01) from the one its *last* chunk
    # wrote, `t_end₁ = 1/50` being four chunks of `1/200` exactly: the
    # finished run never regridded there, and its continuation regrids first.
    prefix = joinpath(mktempdir(), "short")
    short = ckpt_sod(; t_end=1 // 50, checkpoint_path_prefix=prefix,
                     checkpoint_interval_seconds=0, checkpoint_sync_to_disk=false)
    @test short.finished && short.nchunks == 4
    @test length(short.checkpoints_written) == 4
    inner, last_file = short.checkpoints_written[3:4]
    @test load_checkpoint(inner).data.run.chunk == 3
    saved = load_checkpoint(last_file)
    @test latest_checkpoint(prefix) == last_file && saved.data.run.chunk == 4
    # The last chunk's file is the finished run's own answer, unregridded.
    @test saved.fieldsets["U"].state == short.u
    @test saved.forest.leaves == short.forest.leaves
    for file in (inner, last_file)
        r = ckpt_sod(; restart_file=file)
        @test ckpt_same(r, CKPT_REFERENCE)
    end
    # Where `t_end` is not a whole number of chunks the last chunk is no
    # restart point: a continuation's chunks would not line up with it.
    prefix2 = joinpath(mktempdir(), "ragged")
    ragged = ckpt_sod(; t_end=7 // 400, checkpoint_path_prefix=prefix2,
                      checkpoint_interval_seconds=0, num_checkpoints_keep=4,
                      checkpoint_sync_to_disk=false)
    @test ragged.nchunks == 4
    @test [load_checkpoint(f).data.run.chunk for f in ragged.checkpoints_written] ==
          [1, 2, 3]
end

# The reflecting half box of the shear layer, cut to four chunks: the one
# configuration whose forest has mirrored faces, which the file must carry
# and the restart must rebuild the parities of (added 2026-09-29).
ckpt_half_kh(; kwargs...) =
    evolve!(HydroCase(KelvinHelmholtz(Float64, Val(2); seed=:mirrored); half=true),
            Val(2); N=8, ops=CKPT_OPS, t_end=1 // 50, chunk=1 // 200,
            limiter=:minmod, riemann=:hllc, refine_tol=2 // 25, coarsen_tol=1 // 50,
            maxlevel_cap=1, accounting=true, kwargs...)

@testset "A reflecting half box restarts exactly, and records its walls" begin
    # Guards the mirrored faces across a load: the forest's reflecting flags
    # and every field set's parity come back from the file, and the ghost
    # schedule rebuilt from them mirrors as the uninterrupted run's did. A
    # parity lost on the way would be refused by TreeAMR at the first field
    # set, and a wall lost would be a different run — which the recipe names.
    ref = ckpt_half_kh()
    calls, files = ckpt_chain(ckpt_half_kh, joinpath(mktempdir(), "halfkh"))
    @test length(calls) == ref.nchunks
    @test ckpt_same(last(calls), ref)
    @test first(files).data.recipe.reflecting == ((false, false), (true, true))
    @test first(files).forest.reflecting == ((false, false), (true, true))

    # A recipe written before the field existed is a version-1 file — the
    # format went to 2 on 2026-10-01, after the walls — and is refused for
    # its version, which names the reason, before its missing field could
    # be read as anything (amended 2026-10-01; it used to read as "none").
    ck = load_checkpoint(CKPT_FILE)
    old = Base.structdiff(ck.data.recipe, NamedTuple{(:reflecting,)})
    @test !haskey(old, :reflecting)
    path = joinpath(mktempdir(), "old.h5")
    U, u = ck.fieldsets["U"].fieldset, ck.fieldsets["U"].state
    save_checkpoint(path, ck.forest; fieldsets=("U" => (U, u),),
                    application="TreeHydro.jl" => 1,
                    data=(; recipe=old, run=ck.data.run), sync=false)
    err = ckpt_refusal(() -> ckpt_sod(; restart_file=path))
    @test err isa ArgumentError && occursin("after the regrid", err.msg)
end

@testset "Another application's file, or a future version's, is refused" begin
    # Guards `load_run`'s own checks, which TreeAMR leaves to the
    # application: its group's name and its format version. Without them a
    # file from a newer TreeHydro would be read with this version's meaning.
    r = CKPT_REFERENCE
    dir = mktempdir()
    # Version 1, before 2026-10-01, was written after the regrid, and is
    # refused with the reason: read as version 2 it would be regridded twice.
    for (application, needle) in (("TreeHydro.jl" => 99, "format version 99"),
                                  ("TreeHydro.jl" => 1, "after the regrid"),
                                  ("SomethingElse" => 1, "\"SomethingElse\""))
        path = joinpath(dir, "foreign.h5")
        save_checkpoint(path, r.forest; fieldsets=("U" => (r.U, r.u),),
                        application=application, data=(;), sync=false)
        err = ckpt_refusal(() -> ckpt_sod(; restart_file=path))
        @test err isa ArgumentError && occursin(needle, err.msg)
    end
end

@testset "Every inconsistent checkpoint keyword is refused before the run" begin
    # Guards a job script that would write nowhere, never write, or be told
    # to keep no file: each is refused at the call rather than discovered at
    # the first chunk boundary.
    dir = mktempdir()
    prefix = joinpath(dir, "sod")
    refused(; kwargs...) = ckpt_refusal(() -> ckpt_sod(; kwargs...)) isa ArgumentError
    # A trigger with nowhere to write.
    @test refused(; checkpoint_every_chunks=1)
    @test refused(; checkpoint_interval_seconds=60)
    @test refused(; max_walltime_seconds=3600)
    # A prefix that nothing would ever make write.
    @test refused(; checkpoint_path_prefix=prefix)
    # Counts of at least one, times that are times.
    @test refused(; checkpoint_path_prefix=prefix, checkpoint_every_chunks=0)
    @test refused(; checkpoint_path_prefix=prefix, checkpoint_interval_seconds=-1)
    @test refused(; checkpoint_path_prefix=prefix, max_walltime_seconds=0)
    @test refused(; checkpoint_path_prefix=prefix, checkpoint_every_chunks=1,
                  num_checkpoints_keep=0)
    # A prefix with no stem, or in a directory that is not there.
    @test refused(; checkpoint_path_prefix=dir * "/", checkpoint_every_chunks=1)
    @test refused(; checkpoint_path_prefix=joinpath(dir, "missing", "sod"),
                  checkpoint_every_chunks=1)
    # A restart file that is not there.
    @test refused(; restart_file=joinpath(dir, "missing.h5"))
    # And nothing was written by any of them.
    @test isempty(readdir(dir))
end
