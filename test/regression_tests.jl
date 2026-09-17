# The regression net: every physics study at a reduced configuration,
# compared against its committed reference to roundoff.
#
# This is the short tier's substitute for the long tier's sweeps, and it is
# a different kind of claim from everything else in the suite. The sweeps
# say the scheme is second order and that the fixup is what conserves; those
# stay in `test/long/` and run on demand. What this file says is narrower
# and cheaper: *these* reduced runs produce *these* numbers, to `1e-12`, so
# a change anywhere in the scheme — a reordered sum, a limiter branch, an
# upstream change in TreeAMR's prolongation — surfaces here in seconds with
# the moved number printed beside the stored one.
#
# The numeric comparison is not the whole of it. Beside each study's numbers
# are the *claims* that cost nothing once the run has happened — conservation
# to roundoff where it is owed, a leak where the fixup is off, the momentum
# equal to its closed-form boundary flux, no floor firing, the mesh tracking
# the shock — because a reference file records what the code did and a claim
# records what it is supposed to do, and only the second one notices when a
# reference was regenerated from a bug.
#
# The runs are done once, at the top, and read by every testset below: each
# `reference_outputs` call is the whole of one study, and doing it twice
# would double the tier this file exists to keep short.

include("references.jl")

# Every study's reduced configurations, computed once. The regeneration at
# the end of the long tier writes *these* results, so that the file that is
# committed is the one that was just compared and claimed against.
const REGRESSION_OUTPUTS = Dict{Symbol,Any}(
    study => reference_outputs(study) for study in REFERENCE_STUDIES)

"""The `8 eps · scale · nsteps` bound the conservation claims are made at."""
roundoff_bound(out, v) = 8 * eps(Float64) * out["scales"][v] * out["nsteps"]

"""A time read back out of a `_config`, where `"1//5"` is how a `Rational` is
written: TOML has no rational, and a decimal would not say what was meant."""
function config_time(s::AbstractString)
    parts = split(s, "//")
    return length(parts) == 1 ? parse(Float64, parts[1]) :
           parse(Float64, parts[1]) / parse(Float64, parts[2])
end

"""
The tube's cross-section, from the root counts: `sod_forest` makes its
blocks cubes, so every transverse extent is `roots[d]/roots[1]` of the
tube's own length.
"""
tube_area(roots, D) = prod(ntuple(d -> d == 1 ? 1.0 : roots[d] / roots[1], D))

@testset "The reduced runs reproduce the stored references to roundoff: $study" for
        study in REFERENCE_STUDIES
    # The failure this guards is the one a test suite cannot otherwise
    # catch: a change that keeps every inequality in the suite true and
    # moves the answer. Every stored key is compared and a missing or extra
    # key fails, so a study that gains an output cannot silently skip it.
    #
    # Skipped only while regenerating, which is the one run whose job is to
    # write these numbers rather than to check them.
    if REGENERATE_REFERENCES
        @info "TREEHYDRO_REGENERATE is set: not comparing $study against " *
              "$(relpath(reference_path(study), @__DIR__))"
        @test REGENERATE_REFERENCES                      # the tier is what it says
    else
        compare_references(reference_path(study), REGRESSION_OUTPUTS[study])
    end
end

@testset "The reduced entropy wave conserves and floors nothing" begin
    # The claims of step 3 at the reduced size. They cost nothing once the
    # runs have happened and they are what says a regenerated reference was
    # regenerated from a scheme that still works: every one of the `D + 2`
    # integrals constant to a few ulp of its own scale, on the uniform mesh
    # where the fixup has nothing to do, and no floor anywhere.
    for (name, out) in REGRESSION_OUTPUTS[:entropywave]
        D = out["_config"]["D"]
        @test out["levels"] == [0]
        @test out["nblocks"] == 4^D
        @test out["floor_hits"] == 0
        @test isfinite(out["l1"]) && isfinite(out["linf"])
        for v in 1:(D + 2)
            @test out["drift"][v] ≤ roundoff_bound(out, v)
        end
    end
end

@testset "The reduced two-level runs conserve, and leak without the fixup" begin
    # Step 5's central claim at the reduced size, and the reason the leaking
    # runs are stored at all: the reference file records the leak, so a
    # fixup that quietly started running in the control would move a stored
    # number, and the inequality here says the two runs are still on
    # opposite sides of the bound.
    out = REGRESSION_OUTPUTS[:interface]
    for (name, control) in (("ew_d1_p3", "ew_d1_p3_nofixup"),
                            ("ew_d2_p3", "ew_d2_p3_nofixup"))
        r, c = out[name], out[control]
        D = r["_config"]["D"]
        @test r["levels"] == [0, 1]
        @test r["nblocks"] > 4^D                    # the mesh really is two-level
        # One line of `hydro_rhs!` apart, so everything else about the two
        # runs is the same and the drift is the only thing to attribute.
        @test (r["nsteps"], r["nblocks"]) == (c["nsteps"], c["nblocks"])
        @test r["floor_hits"] == c["floor_hits"] == 0
        for v in 1:(D + 2)
            bound = roundoff_bound(r, v)
            @test r["drift"][v] ≤ bound
            @test c["drift"][v] > 1e6 * bound
            @test c["drift"][v] > 1e-8 * c["scales"][v]
        end
    end
end

@testset "The reduced tubes move the momentum by the boundary flux" begin
    # Step 4's claim, which is an equality with a closed form and therefore
    # sharper than "the totals moved": until a wave reaches a Dirichlet face
    # both sides of it are at rest in that face's own state, so the only
    # nonzero Euler flux there is the pressure and
    # `ΔS = (p_L − p_R)·t_end·A` exactly, with `Δρ = ΔE = 0`. It fails for a
    # forgotten hook, for a reflecting boundary, and for a run long enough
    # for a wave to arrive.
    #
    # The mass and energy bound is `1e-5` of the scale and not the `1e-8`
    # the study makes it at, and the reason is a measurement rather than a
    # concession: what remains in those two rows is the *numerical* foot of
    # the rarefaction reaching the Dirichlet face, which falls by four
    # orders of magnitude per halving of `h` — 2.1e-11 of the mass at
    # `h = 1/64` and 2.4e-7 at the `h = 1/32` these reduced runs use. The
    # sharp form of the claim needs the finer mesh and is the long tier's;
    # the numbers themselves are pinned to roundoff by the comparison above.
    for (study, names) in ((:sod, ("d1_minmod_n8", "d1_mc_n8", "d2_minmod_n8")),
                           (:interface, ("sod_d1_middle", "sod_d1_left")))
        for name in names
            out = REGRESSION_OUTPUTS[study][name]
            D = out["_config"]["D"]
            w = SodTube(Float64, Val(D))
            t_end = config_time(out["_config"]["t_end"])
            area = tube_area(out["_config"]["roots"], D)
            expected = (w.p_L - w.p_R) * t_end * area
            @test out["floor_hits"] == 0
            @test out["drift"][2] ≈ expected rtol = 1e-5
            @test out["drift"][1] ≤ 1e-5 * out["scales"][1]
            @test out["drift"][D + 2] ≤ 1e-5 * out["scales"][D + 2]
            # The transverse momentum is zero for all time and *exactly* so:
            # both transverse faces of every cell see identical states, so
            # their flux difference is an exact zero.
            D == 2 && @test out["drift"][3] == 0
        end
    end
    # And the two leaking two-level tubes are on the other side of it.
    for (name, control) in (("sod_d1_middle", "sod_d1_middle_nofixup"),
                            ("sod_d1_left", "sod_d1_left_nofixup"))
        r = REGRESSION_OUTPUTS[:interface][name]
        c = REGRESSION_OUTPUTS[:interface][control]
        @test r["levels"] == [0, 1]
        @test (r["nsteps"], r["nblocks"]) == (c["nsteps"], c["nblocks"])
        @test c["drift"][1] > 1e3 * r["drift"][1]
        @test c["drift"][3] > 1e3 * r["drift"][3]
    end
end

@testset "The reduced calibration keeps the shock above every smooth feature" begin
    # The one row of step 6's table the short tier can afford, and the shape
    # of the calibration rather than its slope: a captured shock scores far
    # above `refine_tol` — it is what the level cap and not the indicator
    # has to stop — while the McNally ramp, the feature whose refinement
    # *must* terminate, sits above the threshold at `h = 1/64` and is the
    # column that picked it.
    sod = REGRESSION_OUTPUTS[:refinement]["sod_tau_h64"]
    ramp = REGRESSION_OUTPUTS[:refinement]["ramp_tau_h64"]
    @test sod["h"] == ramp["h"] == 1 / 64
    @test 0.4 < sod["tau_shock"] < 0.8
    @test sod["tau_shock"] > 5 * Float64(REFERENCE_REFINE_TOL)
    @test sod["tau_contact"] > Float64(REFERENCE_REFINE_TOL)
    @test sod["tau_shock"] > sod["tau_head"] > sod["tau_fan"]
    @test ramp["tau_max"] > Float64(REFERENCE_REFINE_TOL)
end

@testset "The reduced tracked tube tracks, converges and conserves" begin
    # Step 7's claims at a handful of chunks: the initial-data cycle
    # converges, the mesh reaches the cap and is rebuilt under the solution
    # at least once, every cell whose indicator fires sits on a block
    # already at the cap, and the adaptive run is the fine run's answer on
    # fewer cells. A run that refined nothing would match the *coarse*
    # control and cost nothing, which is what the second half guards.
    out = REGRESSION_OUTPUTS[:driver]
    for D in (1, 2)
        tracked, fine, coarse = out["d$(D)_tracked"], out["d$(D)_fine"],
                                out["d$(D)_coarse"]
        for r in (tracked, fine, coarse)
            @test r["converged"]
            @test r["floor_hits"] == 0
        end
        @test tracked["levels"] == collect(0:tracked["_config"]["cap"])
        @test tracked["nregrids"] ≥ 1
        @test tracked["tracking"] == 1.0
        @test tracked["cells"] < fine["cells"]
        @test tracked["l1"] ≤ 1.3 * fine["l1"]
        # Without the exact solution in it: reduced onto the grid the three
        # meshes have in common, the tracked run is far closer to the fine
        # one than the coarse run is.
        @test tracked["l1_tracked_minus_fine"] < tracked["l1_coarse_minus_fine"] / 10
        # Periodic nowhere and Dirichlet at both ends, so the conservation
        # claim through the regrids is the boundary flux again — the mesh
        # being rebuilt under the solution may not move a conserved total —
        # and the transverse momentum is exactly zero.
        w = SodTube(Float64, Val(D))
        expected = (w.p_L - w.p_R) * config_time(tracked["_config"]["t_end"]) *
                   tube_area(tracked["_config"]["roots"], D)
        @test tracked["drift"][2] ≈ expected rtol = 1e-5
        D == 2 && @test tracked["drift"][3] == 0
    end
end
