# Thread-count independence (step 13, H6b).
#
# There is no switch to test. TreeAMR's kernels and host-side passes thread
# themselves and place every block on the thread that owns it; the
# integrator's stage arithmetic follows the same ownership; and this package
# adds no threaded loop of its own. What has to be guarded is the
# *invariant* that makes that safe to rely on — the answer does not move
# when the thread count does, to the last bit — because nothing else in the
# suite would notice if it did (`CODE.md`, "Multi-threading"; `CLAUDE.md`,
# "Never thread anything a TreeAMR callback can reach").
#
# Every assertion here is exact equality, never `≈`. Roundoff-level
# agreement is what a *reassociated* sum gives, and a reassociated sum is
# exactly the bug.

include("thread_workload.jl")

@testset "A run is bit-identical across thread counts" begin
    # The acceptance test. The thread count is a command-line argument to
    # Julia and cannot be changed from inside a running session, so the
    # comparison is against a subprocess started at a different count. A
    # reduction partitioned by thread rather than by block, or a stage
    # combination written from a thread other than the block's owner with a
    # different association, would pass every other test in this suite and
    # fail here.
    reference = thread_digests()
    # Sod: the setup and four chunks, then its summary; the shear layer:
    # the setup and three chunks, then its summary; the blast's one line.
    @test length(reference) == 5 + 1 + 4 + 1 + 1
    # A run that regridded, and a run in which the floors fired — without
    # both, the digests would be a claim about a quiet static mesh.
    @test !occursin("regrids=0", reference[6])
    @test !occursin("resets=0", reference[end])

    other = Threads.nthreads() == 1 ? max(2, min(4, Sys.CPU_THREADS)) : 1
    script = joinpath(@__DIR__, "thread_workload.jl")
    # The *active* project, not `test/`: under `Pkg.test` the tests run in a
    # sandbox and `test/Project.toml` has no manifest of its own.
    project = Base.active_project()
    out = read(`$(Base.julia_cmd()) --threads=$other --project=$project $script`,
               String)
    lines = split(chomp(out), '\n')
    @test lines == reference
    # Said line by line where they differ, so that a failure names the run
    # and the chunk rather than two long outputs.
    for (a, b) in zip(lines, reference)
        a == b || @info "digest differs at $other thread(s)" subprocess = a here = b
    end
end
