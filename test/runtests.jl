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
    # Every test file below is `include`d into this one module, so a top-level
    # name defined in two of them is one name. Julia 1.11 refuses to redefine
    # a `const` and 1.12 and later quietly allow it, so a collision passes
    # every local run at the newer version and fails only the 1.11 entries of
    # CI — which is how the blast's `TRACKED_1D` reached `main` in step 10.
    # Checked from the source text, once, before anything is included, so
    # that it fails at every version and not only at the floor.
    @testset "No two test files define the same top-level constant" begin
        files = filter(f -> endswith(f, "_tests.jl"), readdir(@__DIR__))
        defined = Dict{String,Vector{String}}()
        for f in files, line in eachline(joinpath(@__DIR__, f))
            m = match(r"^const\s+([^\s=]+)", line)
            m === nothing && continue
            push!(get!(defined, m.captures[1], String[]), f)
        end
        duplicates = sort([name => fs for (name, fs) in defined if length(fs) > 1])
        @test isempty(duplicates)
        isempty(duplicates) || @info "duplicated top-level constants" duplicates
    end

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
    # Sedov last, because the order is the dependency order and the blast is
    # the only case that uses *both* the chunked driver and a static
    # `hydro_solve!` run — the one to say what a tracked mesh measures and the
    # other to say what it cannot.
    include("sedov_tests.jl")
    # Kelvin–Helmholtz after it, for the same reason: it is the last case, it
    # needs the driver and the observer and nothing else, and it is the only
    # one whose reference is a uniform fine run of this code rather than a
    # closed form — so every part it rests on has been asserted above it.
    include("kelvinhelmholtz_tests.jl")
end
