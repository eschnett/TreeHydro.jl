using Test
using TreeAMR
using TreeHydro

# The suite runs in two tiers, and which one runs is an environment
# variable. See "Testing: two tiers" in `CODE.md`.
#
#   * **Short** — nothing set, and this is what `Pkg.test()` and CI do: the
#     unit tests and the cheap structural claims. Tens of seconds.
#   * **Long** — `TREEHYDRO_TEST_LONG=1`. The short tier *and* `test/long/`:
#     the convergence sweeps, the interface-order tables, the calibration,
#     the tracked runs — every claim `CODE.md`'s "Measured results" cites.
#     Minutes.
#
# Why the split exists at all is a measurement and not a preference: on
# GitHub's 4-vCPU shared runners the two-dimensional sweeps are 10–17 times
# slower at four threads than at one, which turned a 6-minute serial job
# into a 53-minute threaded one. The sweeps therefore may not enter the
# short tier — see "Things that will bite" in `CLAUDE.md`.

"""Whether an environment variable is set to something that means yes."""
env_flag(name) = lowercase(strip(get(ENV, name, ""))) in ("1", "true", "yes", "on")

const TEST_LONG = env_flag("TREEHYDRO_TEST_LONG")

# The suite is expected to pass, with identical numbers, at any thread
# count; CI runs it at one and at four. `Pkg.test` does not inherit `-t`,
# so the thread count has to be passed explicitly — see `CLAUDE.md`.
@info "Running the $(TEST_LONG ? "long" : "short") tier on " *
      "$(Threads.nthreads()) thread(s)"

@testset "TreeHydro.jl" begin
    # The short tier, and always first: the long files share its helpers.
    include("precision_tests.jl")
    include("prerequisite_tests.jl")
    include("eos_tests.jl")
    include("riemann_tests.jl")
    include("evolution_tests.jl")
    include("exact_riemann_tests.jl")
    include("sod_tests.jl")
    include("interface_tests.jl")
    include("refinement_tests.jl")
    include("driver_tests.jl")

    if TEST_LONG
        include("long/entropywave_tests.jl")
        include("long/sod_tests.jl")
        include("long/interface_tests.jl")
        include("long/refinement_tests.jl")
        include("long/driver_tests.jl")
    end
end
