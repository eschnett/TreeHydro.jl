# Rank-count independence (added 2026-10-02, on TreeAMR's M7).
#
# The claim extends the thread one: **a run distributed over MPI is the
# serial run** — its state, its mesh, its step and chunk counts, every floor
# count, every speed, every maximum, bit for bit at any rank count — **and
# its sums agree to roundoff**, exactly at one rank. TreeAMR makes the
# exchange, the regrid and its own reductions so; what can break it here is
# this package's, and each failure mode is one a serial suite cannot see:
#
#   * a number combined on the host from per-block values — the CFL speed,
#     a floor count, the indicator's scales, a diagnostic — that is a rank's
#     own over a distributed forest: the ranks then size different steps, or
#     build different meshes, and the run hangs or diverges;
#   * a decision taken from a rank's own clock or file system — when to
#     checkpoint, when to stop, whether a file exists — that sends some ranks
#     into a collective and the others past it;
#   * a run state that differs between ranks, which TreeAMR's checkpoint
#     refuses on every rank;
#   * a rank with no blocks, on which a reduction or a launch or a check
#     reads a block that is not there;
#   * a checkpoint that does not restart at another rank count, or restarts
#     into another run, or a rotation that leaves part files behind.
#
# Every line of the workload that is not a sum is compared with `==`; a
# line of sums is compared number by number to roundoff, and its text
# exactly.

isdefined(@__MODULE__, :take_mpi_job!) || include("mpi_jobs.jl")

# The numbers in a line, and the line with each of them replaced by `#`,
# so that two lines of sums are the same line exactly when their skeletons
# are equal and their numbers agree.
const MPI_NUMBER = r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+|NaN|Inf)"

mpi_skeleton(line) = replace(line, MPI_NUMBER => "#")
mpi_numbers(line) = [parse(Float64, m.match) for m in eachmatch(MPI_NUMBER, line)]

# Roundoff: the sums here are totals of order one, their drifts and
# injections differences of such totals, an L1 norm and McNally's `M`. A
# partition reassociates each sum once per rank boundary, which is a few
# ulp of the total; `atol` is a hundred ulp of a total of order one, and
# so also the bound on a drift, which is a difference of two of them and
# has no relative accuracy at all ("Things that will bite").
mpi_agree(a, b) = isapprox(a, b; rtol=1e-12, atol=1e-14)

@testset "A distributed run is the serial run, at every rank count" begin
    files = finish_mpi_workload(take_mpi_job!())
    serial = files[0]
    # Every case reached its end, the serial run included.
    @test length(serial) ≥ 60
    @test any(startswith("blast finished=true"), serial)
    # The cases are what they claim to be: the tube regridded, the blast
    # floored cells of both populations, the wall-clock stop stopped and
    # wrote its checkpoint, and the rotation kept one index and no stale
    # part — and a refusal on some ranks was a refusal on all.
    @test !occursin("regrids=0", only(filter(startswith("sod nsteps="), serial)))
    sedov = only(filter(startswith("sedov "), serial))
    @test !occursin("resets=0", sedov) && !occursin("ghosts=0", sedov)
    stop = only(filter(startswith("stop "), serial))
    @test occursin("finished=false chunk=1", stop) && occursin("written=1", stop)

    for n in 1:3
        lines = files[n]
        @test length(lines) == length(serial)
        @test only(filter(startswith("rotation "), lines)) == "rotation indexes=1 stale=0"
        @test only(filter(startswith("refusal "), lines)) == "refusal agreed=true"
        # The part files of the last checkpoint kept: one per rank, the
        # rotation's I/O processes, and none serially or at one rank.
        @test only(filter(startswith("# rotation parts="), lines)) ==
              "# rotation parts=$(n == 1 ? 0 : n)"
        for (a, b) in zip(serial, lines)
            startswith(a, "#") && (@test startswith(b, "#"); continue)
            if startswith(a, "~")
                ok = mpi_skeleton(a) == mpi_skeleton(b) &&
                     all(map(mpi_agree, mpi_numbers(a), mpi_numbers(b)))
                # At one rank nothing is reassociated, and the sums are the
                # serial bits too.
                n == 1 && (ok &= a == b)
            else
                ok = a == b
            end
            @test ok
            ok || @info "a line differs at $n rank(s)" serial = a distributed = b
        end
    end
end
