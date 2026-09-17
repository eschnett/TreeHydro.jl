using Test
using TreeAMR
using TreeHydro

# One suite, run whole. Every claim `CODE.md`'s "Measured results" records
# comes from a test that runs here, at every thread count and on every push:
# the numbers are expected to be identical at one thread and at four, and
# `Pkg.test` does not inherit `-t`, so the thread count has to be passed
# explicitly — see "Commands" in `CLAUDE.md`.
#
# The one thing CI must not turn on is code coverage. Julia implements a
# coverage hit as an atomic increment of one counter per source line, and
# threads running the same kernel contend on the same cache line, which
# costs a factor of 100 on the two-dimensional runs here. See "Testing" in
# `CODE.md` for the measurement.
@info "Running the tests on $(Threads.nthreads()) thread(s)"

@testset "TreeHydro.jl" begin
    include("precision_tests.jl")
    include("prerequisite_tests.jl")
    include("eos_tests.jl")
    include("riemann_tests.jl")
    include("evolution_tests.jl")
    include("reset_tests.jl")
    include("entropywave_tests.jl")
    include("exact_riemann_tests.jl")
    include("sod_tests.jl")
    include("interface_tests.jl")
    include("refinement_tests.jl")
    include("driver_tests.jl")
end
