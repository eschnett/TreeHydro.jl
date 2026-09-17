using Test
using TreeAMR
using TreeHydro

# The suite runs in two tiers, and which one runs is an environment
# variable. See "Testing: two tiers" in `CODE.md` for the discipline and
# "Commands" in `CLAUDE.md` for the three command lines.
#
#   * **Short** — nothing set, and this is what `Pkg.test()` and CI do. The
#     unit tests as they are, plus `regression_tests.jl`: a reduced
#     configuration of every physics study, compared against the references
#     committed under `test/references/` to roundoff. Tens of seconds.
#   * **Long** — `TREEHYDRO_TEST_LONG=1`. The short tier *and* `test/long/`:
#     the convergence sweeps, the interface-order tables, the calibration,
#     the tracked runs — every claim `CODE.md`'s "Measured results" cites.
#     Minutes.
#   * **Regenerate** — `TREEHYDRO_TEST_LONG=1 TREEHYDRO_REGENERATE=1`. The
#     long tier, and then the reference files are rewritten from the
#     reduced configurations. The comparison is skipped for that run, since
#     it would be comparing the numbers against themselves.
#
# Why the split exists at all is a measurement and not a preference: on
# GitHub's 4-vCPU shared runners the two-dimensional sweeps are 10–17 times
# slower at four threads than at one, which turned a 6-minute serial job
# into a 53-minute threaded one. The sweeps therefore may not enter the
# short tier — see "Things that will bite" in `CLAUDE.md`.
#
# Regenerating is deliberate on purpose. The long tier runs its physics
# claims *first* and rewrites the references only if they pass, so a long
# run on another machine cannot quietly move the committed numbers; a
# regeneration shows up in `git diff` and is reviewed like any other change.

"""Whether an environment variable is set to something that means yes."""
env_flag(name) = lowercase(strip(get(ENV, name, ""))) in ("1", "true", "yes", "on")

const TEST_LONG = env_flag("TREEHYDRO_TEST_LONG")
const REGENERATE_REFERENCES = env_flag("TREEHYDRO_REGENERATE")

REGENERATE_REFERENCES && !TEST_LONG && error(
    "TREEHYDRO_REGENERATE=1 was set without TREEHYDRO_TEST_LONG=1. The " *
    "references are rewritten only after the long tier's physics claims " *
    "have passed — regenerating them from a run that never checked the " *
    "convergence rates, the conservation tables or the calibration would " *
    "record whatever the code happens to do. Set both.")

# The suite is expected to pass, with identical numbers, at any thread
# count; CI runs it at one and at four. `Pkg.test` does not inherit `-t`,
# so the thread count has to be passed explicitly — see `CLAUDE.md`.
@info "Running the $(TEST_LONG ? "long" : "short") tier on " *
      "$(Threads.nthreads()) thread(s)" *
      (REGENERATE_REFERENCES ? ", regenerating the references afterwards" : "")

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
    include("regression_tests.jl")

    if TEST_LONG
        include("long/entropywave_tests.jl")
        include("long/sod_tests.jl")
        include("long/interface_tests.jl")
        include("long/refinement_tests.jl")
        include("long/driver_tests.jl")

        if REGENERATE_REFERENCES
            @testset "The references are rewritten from the reduced \
                      configurations" begin
                # After the physics, never before it, and from the very
                # results `regression_tests.jl` made its claims on — so what
                # lands in the file is what was just checked.
                for study in REFERENCE_STUDIES
                    outputs = REGRESSION_OUTPUTS[study]
                    path = write_references(reference_path(study), outputs)
                    @info "wrote $(length(outputs)) configurations to " *
                          "$(relpath(path, @__DIR__))"
                    @test isfile(path)
                end
            end
        end
    end
end
