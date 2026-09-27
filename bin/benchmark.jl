# Thread-scaling and device measurement (step 14, H6c).
#
#     julia -t N --project=. bin/benchmark.jl
#
# Note the `--project=.`: this is the one script in `bin/` that runs
# against the *package* environment rather than `bin/Project.toml`. It
# needs no CairoMakie, and a compute node should not build Cairo to time a
# flux divergence. `bin/backend.jl` is shared with the viewers and uses only
# KernelAbstractions, which both environments have.
#
# Prints a header row, then one tab-separated row per phase:
#
#     threads backend type case D N roots refined blocks cells phase seconds
#
# for the thread count Julia was started with. Run it once per thread count
# and read the columns against each other; `bin/symmetry_cpu.sh` does that
# on a Symmetry node. `cell_updates_per_second` is a row of its own, from
# the `step` phase.
#
# `--scan=N:roots,N:roots,…` runs the phase table once per pair in one
# process, which is how the block-size scan is taken without paying the
# compilation per point. Every pair shares the header.
#
# `--backend=cuda` or `--backend=metal` runs the same phases on a device,
# in the same format. The device package is *not* a dependency of
# TreeHydro, so the script needs an environment that has it:
#
#     julia --project=/tmp/thgpu -e 'using Pkg; Pkg.develop(path = ".");
#                                     Pkg.add("Metal")'
#     julia --project=/tmp/thgpu bin/benchmark.jl --backend=metal --type=f32
#
# `--type=f32` is not optional on a device without hardware fp64. See
# "Step 14 — benchmark and device" in `CODE.md` for the tables.

using TreeHydro

include(joinpath(@__DIR__, "backend.jl"))

const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)
const TYPENAMES = Dict(Float32 => "f32", Float64 => "f64")

function parse_scan(s)
    return map(split(s, ',')) do pair
        n, r = split(pair, ':')
        (parse(Int, n), parse(Int, r))
    end
end

function main(args)
    dim = 3
    scan = [(16, 8)]
    case = :wave
    refined = false
    reps = 5
    steps = 4
    driver = false
    T = Float64
    backendname = "cpu"
    for a in args
        if startswith(a, "--dim=")
            dim = parse(Int, a[7:end])
        elseif startswith(a, "--n=")
            scan = [(parse(Int, a[5:end]), scan[1][2])]
        elseif startswith(a, "--roots=")
            scan = [(scan[1][1], parse(Int, a[9:end]))]
        elseif startswith(a, "--scan=")
            scan = parse_scan(a[8:end])
        elseif startswith(a, "--case=")
            case = Symbol(a[8:end])
        elseif a == "--refined"
            refined = true
        elseif startswith(a, "--reps=")
            reps = parse(Int, a[8:end])
        elseif startswith(a, "--steps=")
            steps = parse(Int, a[9:end])
        elseif startswith(a, "--type=")
            tag = a[8:end]
            haskey(FLOATTYPES, tag) ||
                error("unknown --type=$tag; expected f32 or f64")
            T = FLOATTYPES[tag]
        elseif startswith(a, "--backend=")
            backendname = a[11:end]
        elseif a == "--driver"
            driver = true
        else
            error("unknown argument $a; expected --dim=, --n=, --roots=, --scan=, \
                   --case=wave|sedov, --refined, --reps=, --steps=, --type=, \
                   --backend=, --driver")
        end
    end
    dim in (1, 2, 3) || error("--dim must be 1, 2 or 3; got $dim")

    threads = Threads.nthreads()
    println(join(("threads", "backend", "type", "case", "D", "N", "roots", "refined",
                  "blocks", "cells", "phase", "seconds"), '\t'))
    return withbackend(backendname, T) do backend
        for (n, roots) in scan
            r = benchmark_phases(T, Val(dim); N=n, roots=roots, case=case,
                                 refined=refined, reps=reps, steps=steps,
                                 backend=backend)
            prefix = join((threads, backendname, TYPENAMES[T], case, dim, n, roots,
                           refined, r.sizes.nblocks, r.sizes.cells), '\t')
            for (phase, s) in r.timings
                println(prefix, '\t', phase, '\t', s)
            end
            step = last(first(filter(x -> first(x) == "step", r.timings)))
            println(prefix, '\t', "cell_updates_per_second", '\t', r.sizes.cells / step)
            println(prefix, '\t', "workbytes", '\t', r.sizes.workbytes)
            flush(stdout)
        end
        if driver
            d = benchmark_driver(T, Val(dim); backend=backend)
            prefix = join((threads, backendname, TYPENAMES[T], "sedov_driver", dim, "-",
                           4, true, d.nblocks, "-"), '\t')
            println(prefix, '\t', "evolve", '\t', d.seconds)
            println(prefix, '\t', "cell_updates_per_second", '\t',
                    d.cell_updates / d.seconds)
        end
        return nothing
    end
end

main(ARGS)
