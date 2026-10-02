# The `mpiexec` job of the MPI test (added 2026-10-02), as a helper rather
# than a test: `runtests.jl` includes it before the first test, and
# `mpi_tests.jl` includes it if it was not. TreeAMR's `test/mpi_jobs.jl` is
# the pattern.
#
# The job runs `mpi_workload.jl` under `mpiexec -n 3` and is
# compilation-bound — about 80 s of wall clock on this machine, most of it
# three ranks compiling the drivers. Where the machine has room for three
# more processes beside the suite it starts here, at the start of the suite,
# and compiles on otherwise idle cores while the rest runs; elsewhere — a CI
# runner — nothing starts here, and `mpi_tests.jl` runs it where it stands.

using MPI: MPI

const MPI_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")

# Whether the job may run beside the suite: three single-threaded ranks,
# each compiling the workload. `TREEHYDRO_TEST_MPI_CONCURRENT=0` or `1`
# decides it by hand.
function mpi_concurrent()
    choice = get(ENV, "TREEHYDRO_TEST_MPI_CONCURRENT", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.CPU_THREADS >= 8 && Sys.total_memory() >= 24 * 2^30
end

# The workload under `mpiexec -n 3`, started and not waited for. The
# command is built from `MPI.mpiexec()` with its environment set on it:
# interpolating `mpiexec()` into a larger command drops the library paths
# it carries (TreeAMR's finding). The *active* project, not `test/`: under
# `Pkg.test` the tests run in a sandbox.
function launch_mpi_workload(; timeout=1800)
    out = mktempdir()
    mpi = MPI.mpiexec()
    project = Base.active_project()
    cmd = `$mpi -n 3 $(Base.julia_cmd()) --threads=1 --project=$project
           $MPI_WORKLOAD $out`
    err = IOBuffer()
    proc = run(pipeline(setenv(cmd, mpi.env); stdout=devnull, stderr=err); wait=false)
    return (; proc, out, err, start=time(), timeout)
end

# The job's four files of lines, with the deadline from its start: a rank
# waiting for a message never sent would otherwise hang the suite.
function finish_mpi_workload(job)
    (; proc, out, err, start, timeout) = job
    left = max(1.0, timeout - (time() - start))
    if timedwait(() -> process_exited(proc), left) !== :ok
        kill(proc)
        error("the MPI workload did not finish in $timeout s; its stderr:\n" *
              String(take!(err)))
    end
    success(proc) || error("the MPI workload failed; its stderr:\n" *
                           String(take!(err)))
    return Dict(n => readlines(joinpath(out, "n$n.txt")) for n in 0:3)
end

const MPI_JOB = Ref{Any}(nothing)

function start_mpi_job!()
    mpi_concurrent() || return nothing
    job = launch_mpi_workload()
    MPI_JOB[] = job
    atexit(() -> process_running(job.proc) && kill(job.proc))
    return nothing
end

function take_mpi_job!()
    job = MPI_JOB[]
    MPI_JOB[] = nothing
    return job === nothing ? launch_mpi_workload() : job
end
