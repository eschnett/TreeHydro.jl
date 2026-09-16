using Test
using TreeAMR
using TreeHydro

# The suite is expected to pass, with identical numbers, at any thread
# count; CI runs it at one and at four. `Pkg.test` does not inherit `-t`,
# so the thread count has to be passed explicitly — see `CLAUDE.md`.
@info "Running the tests on $(Threads.nthreads()) thread(s)"

@testset "TreeHydro.jl" begin
    include("precision_tests.jl")
    include("prerequisite_tests.jl")
    include("eos_tests.jl")
    include("riemann_tests.jl")
end
