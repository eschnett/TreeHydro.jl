# The stored reference outputs: the short tier's regression net.
#
# The suite runs in two tiers (see "Testing: two tiers" in `CODE.md`). The
# **long** tier holds the physics — the convergence sweeps, the tables, the
# calibration, the tracked runs — and is what every number in `CODE.md`'s
# "Measured results" comes from. It costs minutes. The **short** tier, which
# is what `Pkg.test()` runs and what CI runs, cannot afford any of that and
# must still notice when the numerics move, so it runs a *reduced
# configuration* of each study — coarser `N`, fewer roots, a shorter
# `t_end`, fewer chunks — and compares its named outputs against the numbers
# committed under `test/references/`, to roundoff.
#
# That is a different kind of test from the rest of the suite and it is
# worth being explicit about what it can and cannot say. It cannot say that
# the scheme is second order; only the long tier says that. What it says is
# that *this* code produces *these* numbers, so any change that moves them —
# a reordered sum, a different slope limiter branch, an upstream change in
# TreeAMR's prolongation — shows up as a failing comparison with the moved
# number printed beside the stored one, rather than as a silence.
#
# Three rules follow, and they are the whole discipline:
#
#   * **The references are committed and reviewed like code.** They are
#     regenerated only by `TREEHYDRO_TEST_LONG=1 TREEHYDRO_REGENERATE=1`,
#     which runs the physics claims *first* and rewrites the files only if
#     they pass. A regeneration shows up in `git diff` as changed numbers,
#     and the commit that does it says why they moved.
#   * **Only `Float64` outputs go in.** `Float32` and `Float32x2` stay
#     unit-level: a MultiFloats value's last bits depend on whether the
#     platform has an `fma`, and a reference file that moved with the
#     hardware would be a liability rather than a net.
#   * **Every stored key is compared, and a missing or extra key fails.** A
#     study that gains an output must be regenerated to gain it here too,
#     rather than quietly not being compared.
#
# ## Why the numbers are expected to be the same on another machine
#
# The files are generated on Apple silicon and compared on GitHub's Linux
# x86-64 and macOS arm64 runners. Bit-identity across the three is what is
# actually expected: Julia's `sin`, `cos`, `exp`, `log` and `^` are
# pure-Julia and therefore platform-independent, `sqrt` is correctly rounded
# by IEEE 754, Julia does not contract `a*b + c` into an `fma` unless asked,
# and TreeAMR's reductions are order-fixed, so there is nothing left for the
# hardware to disagree about. **The stated tolerance is nevertheless
# `rtol = 1e-12`** — roundoff, not physics — so that a last-bit difference,
# should one ever appear, does not break CI; what CI reports is then the
# agreement actually observed. A comparison that fails at `1e-12` is a
# change in the numerics and not a platform difference.
#
# This file is deliberately self-contained: it computes its own primed
# problems and its own indicator table rather than borrowing
# `refinement_tests.jl`'s, because what it computes is *committed data* and
# has to be readable in one place and stay put when a test file is
# rearranged.

using TOML

"""The five studies, one TOML file each — one file per study reads better in a diff."""
const REFERENCE_STUDIES = (:entropywave, :interface, :sod, :refinement, :driver)

const REFERENCE_DIR = joinpath(@__DIR__, "references")

"""The committed file holding `study`'s reduced configurations."""
reference_path(study::Symbol) = joinpath(REFERENCE_DIR, "$(study).toml")

"""
Roundoff, not physics: the references are the *same* computation, so the
only thing they may differ by is the last bits of a floating-point number.
See the note on cross-platform bit-identity at the top of this file.
"""
const REFERENCE_RTOL = 1e-12

"""
The absolute floor of the comparison, below which two numbers are both
"zero to roundoff". It is far below any scale the package computes in and
exists only so that an exactly-zero output — the transverse momentum drift,
for one — compares equal to itself without dividing by zero.
"""
const REFERENCE_ATOL = 1e-300

# The conservative family at the interface order the scheme wants, as every
# study in the suite runs it. `p` is the prolongation order, which the
# interface study varies and nothing else does.
reference_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

# Step 6's calibrated thresholds and floors, quoted rather than reinvented.
const REFERENCE_REFINE_TOL = 2 // 25            # 0.08
const REFERENCE_COARSEN_TOL = 1 // 50           # 0.02

"""
    reference_outputs(study::Symbol) -> Dict{String,Dict{String,Any}}

Run `study`'s reduced configurations and return, per configuration name,
the named outputs the reference file holds: floats, integers, booleans and
vectors of those, plus a `_config` sub-table recording the configuration
itself so that the file is self-describing and a changed configuration
cannot pass as a changed number.

One call per study runs everything that study stores, which is what keeps
the short tier inside its budget: [`regression_tests.jl`](@ref) makes its
qualitative claims on the same results it compares.
"""
function reference_outputs(study::Symbol)
    study === :entropywave && return entropywave_references()
    study === :interface && return interface_references()
    study === :sod && return sod_references()
    study === :refinement && return refinement_references()
    study === :driver && return driver_references()
    throw(ArgumentError(
        "there is no reference study $(repr(study)): the studies are " *
        "$(join(map(repr, REFERENCE_STUDIES), ", ")). A new study needs an " *
        "entry here, a file under test/references/, and a regeneration " *
        "(TREEHYDRO_TEST_LONG=1 TREEHYDRO_REGENERATE=1) to create it."))
end

"""
    write_references(path, dict)

Write `dict` to `path` as TOML, sorted, with a header saying what the file
is and how it is made. A `Float64` prints in Julia's shortest round-tripping
form, which TOML's float syntax accepts and `TOML.parse` returns bit for bit.
"""
function write_references(path, dict)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Generated by TREEHYDRO_TEST_LONG=1 TREEHYDRO_REGENERATE=1")
        println(io, "# julia --project=. -e 'using Pkg; Pkg.test()'.")
        println(io, "#")
        println(io, "# Reduced configurations of the physics studies and the")
        println(io, "# outputs the short tier compares against, to roundoff")
        println(io, "# (rtol = 1e-12). Committed and reviewed like code: a")
        println(io, "# changed number here is a changed numerical result, and")
        println(io, "# the commit that changes it says why. Each table's")
        println(io, "# `_config` sub-table records the configuration it came")
        println(io, "# from. See test/references.jl.")
        println(io)
        TOML.print(io, dict; sorted=true)
    end
    return path
end

"""
    compare_references(path, dict; rtol = REFERENCE_RTOL)

Compare the outputs in `dict` against the ones stored in `path`, asserting
every one of them. Floats are compared with `isapprox(…; rtol, atol =
$(REFERENCE_ATOL))`; integers, booleans, strings and the `_config` sub-table
are compared **exactly**, since a step count, a block count or a level list
that moved by one is a different run and not a rounding.

A configuration present in one and not the other fails, and so does a key
present in one table and not the other: a study that gains an output must
be regenerated, or the new output would be the one thing not under test.
"""
function compare_references(path, dict; rtol=REFERENCE_RTOL)
    isfile(path) || error(
        "there is no reference file at $path. Generate it with " *
        "TREEHYDRO_TEST_LONG=1 TREEHYDRO_REGENERATE=1 julia --project=. -e " *
        "'using Pkg; Pkg.test()', read the diff, and commit it.")
    stored = TOML.parsefile(path)
    @test sort(collect(keys(stored))) == sort(collect(keys(dict)))
    for name in sort(collect(keys(dict)))
        haskey(stored, name) || continue
        @testset "$name" begin
            compare_table(stored[name], dict[name]; rtol=rtol)
        end
    end
    return nothing
end

"""One configuration's table: every key of it, in both directions."""
function compare_table(stored::AbstractDict, computed::AbstractDict; rtol)
    @test sort(collect(keys(stored))) == sort(collect(keys(computed)))
    for key in sort(collect(keys(stored)))
        haskey(computed, key) || continue
        @test reference_equal(key, stored[key], computed[key]; rtol=rtol)
    end
    return nothing
end

"""
Whether a stored value and a computed one agree: `rtol` for floats, exact
equality for everything else, elementwise through a vector, and recursively
through the `_config` sub-table.

It prints what moved rather than only that something did — a comparison
whose failure says `false` is one nobody can act on.
"""
function reference_equal(key, stored, computed; rtol)
    if stored isa AbstractDict && computed isa AbstractDict
        length(stored) == length(computed) || return failed(key, stored, computed)
        return all(k -> haskey(computed, k) &&
                        reference_equal("$key.$k", stored[k], computed[k]; rtol=rtol),
                   keys(stored))
    elseif stored isa AbstractVector && computed isa AbstractVector
        length(stored) == length(computed) || return failed(key, stored, computed)
        return all(i -> reference_equal("$key[$i]", stored[i], computed[i]; rtol=rtol),
                   eachindex(stored))
    elseif stored isa AbstractFloat && computed isa AbstractFloat
        isapprox(stored, computed; rtol=rtol, atol=REFERENCE_ATOL) ||
            return failed(key, stored, computed)
        return true
    else
        typeof(stored) == typeof(computed) && stored == computed ||
            return failed(key, stored, computed)
        return true
    end
end

function failed(key, stored, computed)
    @warn "reference mismatch at $key" stored computed
    return false
end

# ---------------------------------------------------------------------------
# The reduced configurations.
#
# Each is the study it stands for with the resolution taken out of it: the
# same case, the same operators, the same limiter and solver, run on a mesh
# small enough that the *threaded* CI runner cannot spend minutes on it (see
# "Things that will bite" in `CLAUDE.md`). Every `D ≥ 2` configuration is at
# `N = 8` and a few dozen steps.

"""What an `entropywave_errors` or `sod_errors` result stores."""
function study_outputs(r)
    return Dict{String,Any}("l1" => r.l1, "linf" => r.linf,
                            "drift" => collect(r.drift),
                            "scales" => collect(r.scales),
                            "floor_hits" => r.floor_hits, "h" => r.h,
                            "nsteps" => r.nsteps, "nblocks" => r.nblocks,
                            "levels" => collect(r.levels))
end

"""The same, plus the four signal speeds only the shock tube has."""
function tube_outputs(r)
    out = study_outputs(r)
    out["lambda"] = r.λ
    out["lambda_initial"] = r.λ_initial
    out["lambda_ratio"] = r.λ_ratio
    out["lambda_final"] = r.λ_final
    return out
end

"""A table's `_config`, from the keywords that produced it."""
config(; kwargs...) = Dict{String,Any}(String(k) => v for (k, v) in kwargs)

# The entropy wave on the **uniform** mesh: the convergence study of step 3
# at its two smallest `N` in `D = 1` and its smallest in `D = 2`, with both
# limiters. What the long tier fits a rate through, the short tier pins one
# point of.
function entropywave_references()
    out = Dict{String,Any}()
    for (name, D, N, limiter, t_end) in (("d1_none_n8", 1, 8, :none, 1 // 4),
                                         ("d1_none_n16", 1, 16, :none, 1 // 4),
                                         ("d1_mc_n8", 1, 8, :mc, 1 // 4),
                                         ("d2_none_n8", 2, 8, :none, 1 // 8),
                                         ("d2_mc_n8", 2, 8, :mc, 1 // 8))
        r = entropywave_errors(Val(D); N=N, ops=reference_ops(), roots=4,
                               refined=false, limiter=limiter, t_end=t_end)
        out[name] = merge(study_outputs(r),
                          Dict("_config" => config(; D=D, N=N, roots=4,
                                                   t_end=string(t_end),
                                                   limiter=String(limiter),
                                                   riemann="hlle", p=3, fixup=true,
                                                   refined="false")))
    end
    return out
end

# The coarse-fine face: the entropy wave on the static two-level mesh with
# the fixup and without it — the leak *is* an output, and a reference that
# stored only the conserving run would not notice a fixup that started
# working by accident — and the interface-order runs at `p = 1` and `p = 3`
# at one `N`, plus the two two-level Sod tubes each way.
function interface_references()
    out = Dict{String,Any}()
    for (name, D, p, fixup, t_end) in (("ew_d1_p3", 1, 3, true, 1 // 8),
                                       ("ew_d1_p3_nofixup", 1, 3, false, 1 // 8),
                                       ("ew_d1_p1", 1, 1, true, 1 // 8),
                                       ("ew_d2_p3", 2, 3, true, 1 // 16),
                                       ("ew_d2_p3_nofixup", 2, 3, false, 1 // 16))
        r = entropywave_errors(Val(D); N=8, ops=reference_ops(p), roots=4,
                               refined=true, limiter=:none, fixup=fixup,
                               t_end=t_end)
        out[name] = merge(study_outputs(r),
                          Dict("_config" => config(; D=D, N=8, roots=4,
                                                   t_end=string(t_end),
                                                   limiter="none", riemann="hlle",
                                                   p=p, fixup=fixup,
                                                   refined="true")))
    end
    for (name, refined, fixup) in (("sod_d1_middle", :middle, true),
                                   ("sod_d1_middle_nofixup", :middle, false),
                                   ("sod_d1_left", :left, true),
                                   ("sod_d1_left_nofixup", :left, false))
        r = sod_errors(Val(1); N=8, ops=reference_ops(), roots=(4,),
                       refined=refined, limiter=:minmod, fixup=fixup, t_end=1 // 5)
        out[name] = merge(tube_outputs(r),
                          Dict("_config" => config(; D=1, N=8, roots=[4],
                                                   t_end="1//5", limiter="minmod",
                                                   riemann="hlle", p=3, fixup=fixup,
                                                   refined=String(refined))))
    end
    return out
end

# Sod's tube on the uniform mesh: the convergence study of step 4 at one
# `N` below its coarsest, both limiters in `D = 1`, and the planar tube in
# `D = 2` at half the end time.
function sod_references()
    out = Dict{String,Any}()
    for (name, D, roots, limiter, t_end) in (("d1_minmod_n8", 1, (4,), :minmod,
                                              1 // 5),
                                             ("d1_mc_n8", 1, (4,), :mc, 1 // 5),
                                             ("d2_minmod_n8", 2, (4, 1), :minmod,
                                              1 // 10))
        r = sod_errors(Val(D); N=8, ops=reference_ops(), roots=roots,
                       limiter=limiter, t_end=t_end)
        out[name] = merge(tube_outputs(r),
                          Dict("_config" => config(; D=D, N=8, roots=collect(roots),
                                                   t_end=string(t_end),
                                                   limiter=String(limiter),
                                                   riemann="hlle", p=3, fixup=true,
                                                   refined="false")))
    end
    return out
end

# The refinement criterion's calibration at **one** spacing. The long tier
# runs the four-row table that chose the thresholds; what the short tier
# pins is the row at `h = 1/64` — Sod's four feature scores after a short
# evolution, and the McNally ramp's score on initial data.
function refinement_references()
    T = Float64
    out = Dict{String,Any}()

    t_end = T(1 // 10)
    r = reference_sod_taus(16; roots=4, t_end=t_end)
    w, sol = r.w, exact_riemann(r.w)
    x₀ = w.x₀
    head = x₀ + sol.head_L * t_end
    tail = x₀ + sol.tail_L * t_end
    contact = x₀ + sol.u★ * t_end
    shock = x₀ + sol.head_R * t_end
    win = T(3 // 100)                             # three cells at h = 1/64
    fan = (head + (tail - head) * 3 // 10, head + (tail - head) * 7 // 10)
    table = reference_taus(r.p)
    maxwin(lo, hi) = maximum((c.τ for c in table if lo ≤ c.x[1] ≤ hi); init=zero(T))
    out["sod_tau_h64"] = Dict{String,Any}(
        "h" => minimum_spacing(T, r.forest),
        "tau_shock" => maxwin(shock - win, shock + win),
        "tau_contact" => maxwin(contact - win, contact + win),
        "tau_head" => maxwin(head - win, head + win),
        "tau_tail" => maxwin(tail - win, tail + win),
        "tau_fan" => maxwin(fan[1], fan[2]),
        "_config" => config(; D=1, N=16, roots=[4], t_end="1//10",
                            limiter="minmod", riemann="hlle", p=3, fixup=true,
                            refined="false"))

    ramp = reference_ramp_tau(16; roots=4)
    out["ramp_tau_h64"] = Dict{String,Any}(
        "h" => ramp.h, "tau_max" => ramp.τ,
        "_config" => config(; D=1, N=16, roots=[4], t_end="0", limiter="minmod",
                            riemann="hlle", p=3, fixup=true, refined="false"))
    return out
end

# The tracked shock tube through the driver, at a fifth of the `t_end` and
# one level of refinement instead of two in `D = 1`, and a tenth of it in
# `D = 2` — a handful of chunks, but enough that the mesh is rebuilt under
# the solution at least once, which is the path the whole file exists for.
# Each dimension stores its tracked run and the two uniform references it is
# judged against.
const REFERENCE_SOD1D = (roots=(8,), N=8, cap=1, chunk=1 // 200, t_end=1 // 20)
const REFERENCE_SOD2D = (roots=(8, 1), N=8, cap=1, chunk=1 // 200, t_end=1 // 25)

function driver_references()
    out = Dict{String,Any}()
    for (D, cfg) in ((1, REFERENCE_SOD1D), (2, REFERENCE_SOD2D))
        case() = HydroCase(SodTube(Float64, Val(D)); roots=cfg.roots)
        common = (; N=cfg.N, ops=reference_ops(), t_end=cfg.t_end, chunk=cfg.chunk,
                  limiter=:minmod)
        tracked = evolve!(case(), Val(D); common...,
                          refine_tol=REFERENCE_REFINE_TOL,
                          coarsen_tol=REFERENCE_COARSEN_TOL, maxlevel_cap=cfg.cap)
        fine = uniform_run(case(), Val(D); common..., roots=cfg.roots .* 2^cfg.cap)
        coarse = uniform_run(case(), Val(D); common..., roots=cfg.roots)

        # The tracking claim without the exact solution in it: both runs
        # reduced onto the grid the three meshes have in common.
        M = cfg.roots .* cfg.N
        reduced_fine = reduce_to_grid(fine.U, M)
        conf(kind) = config(; D=D, N=cfg.N, roots=collect(cfg.roots),
                            t_end=string(cfg.t_end), chunk=string(cfg.chunk),
                            limiter="minmod", riemann="hlle", p=3, fixup=true,
                            cap=(kind === :tracked ? cfg.cap : 0),
                            refined=String(Symbol(kind)))
        out["d$(D)_tracked"] = merge(
            driver_outputs(tracked),
            Dict("l1_tracked_minus_fine" =>
                     l1_difference(reduce_to_grid(tracked.U, M), reduced_fine),
                 "l1_coarse_minus_fine" =>
                     l1_difference(reduce_to_grid(coarse.U, M), reduced_fine),
                 "_config" => conf(:tracked)))
        out["d$(D)_fine"] = merge(driver_outputs(fine), Dict("_config" => conf(:fine)))
        out["d$(D)_coarse"] = merge(driver_outputs(coarse),
                                    Dict("_config" => conf(:coarse)))
    end
    return out
end

"""What an `evolve!` record stores: the errors, the conservation, the mesh
history and the speeds, which between them touch every branch of the loop."""
function driver_outputs(r)
    return Dict{String,Any}("l1" => r.l1, "linf" => r.linf,
                            "drift" => collect(r.drift),
                            "scales" => collect(r.scales),
                            "floor_hits" => r.floor_hits, "h" => r.h,
                            "nsteps" => r.nsteps, "nchunks" => r.nchunks,
                            "nregrids" => r.nregrids, "passes" => r.passes,
                            "converged" => r.converged, "nblocks" => r.nblocks,
                            "cells" => r.cells, "tracking" => r.tracking,
                            "levels" => collect(r.levels),
                            "nblocks_history" => copy(r.nblocks_history),
                            "buffer_history" => copy(r.buffer_history),
                            "lambda_initial" => r.λ_initial,
                            "lambda_history" => copy(r.λ_history),
                            "lambda_end_history" => copy(r.λ_end_history))
end

# ---------------------------------------------------------------------------
# The two pieces of machinery the refinement study needs, written out here
# for the reason given at the top of the file.

"""
A `HydroProblem` over `forest` whose primitives hold `initial(x)` — the
primitive tuple — with the ghosts filled and `P` current.

The sequence every driver runs before flagging: fill, gather,
[`update_primitives!`](@ref). The indicator's stencil reaches one cell past
each block face, so skipping the ghost fill would measure zeros at every
block boundary.
"""
function reference_problem(::Val{D}, forest, initial; eos, floors, boundary=nothing,
                           limiter=:minmod, T=Float64) where {D}
    U = FieldSet{T}(forest, D + 2; G=2)
    p = HydroProblem(U, reference_ops(); eos=eos, floors=floors, limiter=limiter,
                     riemann=:hlle, boundary=boundary)
    fill_by_coordinates!(AllVariables(x -> prim2con(eos, initial(x))), U)
    u = statevector(U)
    gather!(u, U)
    update_primitives!(p, u)
    return (p=p, u=u)
end

"""Every owned cell as `(x, τ)`, through the same [`cell_tau`](@ref) the
predicates call, so that the table and the criterion cannot drift apart."""
function reference_taus(p::HydroProblem{T,D}; ε=T(1 // 100),
                        ε_g=T(1 // 1000)) where {T,D}
    P = p.P
    N = P.forest.N
    scales = indicator_scales(P)
    refs = (T(scales[1]), T(scales[2]))
    return [begin
                idx = ntuple(d -> Tuple(c)[d] + P.G[d], D)
                (x=coordinates(T, P, b, idx),
                 τ=cell_tau(P.work, idx, b, refs, T(ε), T(ε_g), Val(D)))
            end
            for b in 1:nblocks(P) for c in CartesianIndices(ntuple(_ -> N, D))]
end

"""Sod's tube on a uniform mesh, evolved to `t_end`, with `P` current."""
function reference_sod_taus(N; roots=4, t_end=1 // 10, limiter=:minmod, T=Float64)
    w = SodTube(T, Val(1))
    forest = sod_forest(Val(1), N; roots=(roots,), L=w.L, T=T)
    r = reference_problem(Val(1), forest, x -> TreeHydro.sod_state(w, x); eos=w.eos,
                          floors=w.floors, boundary=sod_boundary(w), limiter=limiter)
    λ = T(max_signal_speed(exact_riemann(w)))
    dt = hydro_dt(forest, T(2 // 5), λ, Val(1))
    u = hydro_solve!(r.p, r.u, zero(T), T(t_end), ceil(Int, T(t_end) / dt))
    update_primitives!(r.p, u)
    return (p=r.p, w=w, forest=forest)
end

"""
The McNally density ramp on initial data: ρ from 1 to 2 over an exponential
ramp of width `L = 1/40` at uniform pressure, which is the Kelvin–Helmholtz
shear layer's density signature and the feature the criterion is meant to
resolve *and stop*. Only the ramp's shape is used; step 10 checks the setup
against the paper.
"""
function reference_ramp_tau(N; roots=4, T=Float64)
    ρ₁, ρ₂, Lr = one(T), T(2), T(1 // 40)
    ρm = (ρ₁ - ρ₂) / 2
    function ramp(x)
        y = x[1]
        ρ = y < T(1 // 4) ? ρ₁ - ρm * exp((y - T(1 // 4)) / Lr) :
            y < T(1 // 2) ? ρ₂ + ρm * exp((T(1 // 4) - y) / Lr) :
            y < T(3 // 4) ? ρ₂ + ρm * exp((y - T(3 // 4)) / Lr) :
            ρ₁ - ρm * exp((T(3 // 4) - y) / Lr)
        return (ρ, zero(T), T(5 // 2))
    end
    forest = Forest((roots,); N=N, periodic=(true,), extents=((zero(T), one(T)),))
    # Floors far below anything the ramp contains, so that nothing here
    # measures the floors by accident.
    floors = Floors{T}(; ρ_atm=T(1 // 10^12), p_atm=T(1 // 10^12),
                       p_floor=T(1 // 10^12))
    r = reference_problem(Val(1), forest, ramp; eos=IdealGas(T(5 // 3)),
                          floors=floors)
    return (h=minimum_spacing(T, forest), τ=maximum(c.τ for c in reference_taus(r.p)))
end
