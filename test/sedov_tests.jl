# The Sedov blast: the strong shock, the case where a floor finally fires, and
# the case where two Dirichlet faces meet.
#
# Everything before this file either had no floor to exercise (the entropy wave,
# Sod) or no physical boundary that met another (the tube is periodic across
# itself). Here the interior of the blast evacuates, the prolongation across a
# coarse-fine face has a strong shock to cross, and every one of the `2D` faces
# is Dirichlet, so a fine block in a corner has ghost regions that only two
# faces together can fill.
#
# The failure modes this file guards, one per testset:
#
#   * a similarity constant that is not the literature's, which would put a
#     systematic error into every radius the law predicts — `ξ₀ ≈ 1.033` for
#     `γ = 7/5` in 3D is Taylor's own number and is the check `CODE.md` names;
#   * a deposition that the initial-data cycle does not resolve, or an `E₀`
#     read from the nominal value rather than from what the mesh received;
#   * a blast that does not follow `r_s ∝ t^{2/(D+2)}`, or a captured shock
#     whose density jump is not below the strong-shock limit of 6;
#   * a coarse-fine face that leaks under a real shock, with `fixup = false`
#     as the negative control — and the *quiet* failure that the tracked mesh
#     cannot make that measurement at all, because it keeps the shock at the
#     cap and leaves its coarse-fine faces standing in undisturbed gas;
#   * an unlimited `p = 3` prolongation producing states the floors have to
#     repair, which is the positivity half of the `p = 1` question that Sod
#     could not answer;
#   * a reset whose injection is not what the drift says it is;
#   * the boundary hook missing an edge or a corner region, which no profile
#     plot would show.
#
# The numbers recorded in the comments below are in `CODE.md` under "Step 9" in
# "Measured results"; a changed number is a regression, and a test that merely
# still passes is not.

sedov_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

# Step 6's calibrated thresholds, quoted rather than reinvented.
const SEDOV_REFINE_TOL = 2 // 25             # 0.08
const SEDOV_COARSEN_TOL = 1 // 50            # 0.02

# The tracked configurations. `r₀` spans eight cells at the cap in `D = 1, 2`
# and four in `D = 3` — `CODE.md` asks for "several", and eight is what the
# two-dimensional run wants for two measured reasons: the hot spot's sound
# speed goes as `r₀^{-D/2}`, so halving `r₀` doubles `λ` and halves the chunk
# the travelling margin admits (100 chunks instead of 40), and the wider top
# hat is what lets the centre flatten within an affordable `t_end`.
#
# `chunk` is bounded by the cap, as it is on the tube, and *more* tightly here:
# the derived margin covers `speed_headroom · λ · chunk` at the cap's spacing
# and must stay under one finest-level block width, and Sedov's early `λ` is
# the hot spot's sound speed — 6.76 in `D = 2`, 8.27 in `D = 3` — rather than
# an `O(1)` wave speed.
const SEDOV1D = (roots=4, N=8, cap=2, r₀=1 // 16, chunk=1 // 200, t_end=1 // 8)
const SEDOV2D = (roots=4, N=8, cap=2, r₀=1 // 16, chunk=1 // 400, t_end=1 // 10)
const SEDOV3D = (roots=4, N=4, cap=1, r₀=1 // 8, chunk=1 // 300, t_end=1 // 20)

# The static two-level configurations, where the *shock itself* crosses a
# coarse-fine face. `sedov_forest(:center)` refines the `2^D` root blocks
# around the origin, so the blast starts inside the refined region and leaves
# it at `|x_d| = 1/4`.
const STATIC2D = (N=8, roots=4, r₀=1 // 16, t_end=1 // 10)
const STATIC3D = (N=4, roots=4, r₀=1 // 8, t_end=1 // 30)

"""
The tracked blast, with the shock radius and the peak compression recorded per
chunk **through the observer** — which is the only way to take them, since the
state is scattered and `P` current exactly there and nowhere else a caller can
reach.
"""
function tracked_sedov(::Val{D}, cfg; p=3, fixup=true, reset=:stage,
                       from=3, kwargs...) where {D}
    w = SedovBlast(Float64, Val(D); r₀=cfg.r₀)
    case = HydroCase(w; roots=cfg.roots)
    ts, rs, peaks = Float64[], Float64[], Float64[]
    function watch(pr, t, u)
        push!(ts, t)
        push!(rs, shock_radius(pr.P, w))
        push!(peaks, peak_compression(pr.P, w))
    end
    r = evolve!(case, Val(D); N=cfg.N, ops=sedov_ops(p), t_end=cfg.t_end,
                chunk=cfg.chunk, limiter=:minmod, refine_tol=SEDOV_REFINE_TOL,
                coarsen_tol=SEDOV_COARSEN_TOL, maxlevel_cap=cfg.cap,
                fixup=fixup, reset=reset, accounting=true, observer=watch,
                kwargs...)
    E₀ = measured_E₀(r, w)
    return (; r, w, ts, rs, peaks, E₀,
            exponent=exponent_fit(ts, rs; from=from * w.r₀),
            peak=peaks[end], r_s=rs[end])
end

"""
The same blast on a uniform mesh with the root brick scaled by `scale` —
`2^cap` for the *fine* reference, which then shares the tracked run's finest
spacing, and `1` for the *coarse* control, which shares its coarsest.
"""
function uniform_sedov(::Val{D}, cfg; scale=1, from=3) where {D}
    w = SedovBlast(Float64, Val(D); r₀=cfg.r₀)
    case = HydroCase(w; roots=cfg.roots)
    ts, rs, peaks = Float64[], Float64[], Float64[]
    function watch(pr, t, u)
        push!(ts, t)
        push!(rs, shock_radius(pr.P, w))
        push!(peaks, peak_compression(pr.P, w))
    end
    r = uniform_run(case, Val(D); N=cfg.N, ops=sedov_ops(), t_end=cfg.t_end,
                    chunk=cfg.chunk, limiter=:minmod, roots=case.roots .* scale,
                    accounting=true, observer=watch)
    return (; r, w, ts, rs, peaks, E₀=measured_E₀(r, w),
            exponent=exponent_fit(ts, rs; from=from * w.r₀),
            peak=peaks[end], r_s=rs[end])
end

"""
How many stored entries of the blast's **outward-facing** ghost regions do not
hold the conserved ambient state — zero, or the Dirichlet hook did not reach
a region, and the corner and edge regions are the ones it would miss.

An entry counts as outward-facing if it lies in the outer ghost range of *any*
dimension in which its block sits against a physical face, so a corner entry
— outward in two dimensions at once, and reachable by neither face alone — is
included. `driver_tests.jl` asks the same question of the tube, where the
condition can only ever hold in one dimension.
"""
function sedov_ghost_mismatches(U::FieldSet{T,D}, w::SedovBlast{T,D}) where {T,D}
    amb = prim2con(w.eos, ambient_state(w))
    forest = U.forest
    N = forest.N
    cons = Array(U.work)
    bad, seen = 0, 0
    for b in 1:nblocks(U)
        ext = block_extent(forest, blockkey(U, b))
        for idx in CartesianIndices(size(cons)[1:D])
            outward = false
            for d in 1:D
                i, G = Tuple(idx)[d], U.G[d]
                ext[d][1] == forest.extents[d][1] && i ≤ G && (outward = true)
                ext[d][2] == forest.extents[d][2] && i > G + N && (outward = true)
            end
            outward || continue
            for v in 1:(D + 2)
                seen += 1
                cons[Tuple(idx)..., v, b] == amb[v] || (bad += 1)
            end
        end
    end
    return (bad=bad, seen=seen)
end

"""
The owned cells of the **fine blocks in the low corner of the box**, against
the ambient state they started in: how many entries of each conserved variable
moved at all, and by how much.

The shock has not arrived there, so a correct exchange leaves the corner
block exactly as it was. A tangential prolongation reading the wrong ghosts
would show here and nowhere else.
"""
function corner_block_motion(U::FieldSet{T,D}, w::SedovBlast{T,D}) where {T,D}
    amb = prim2con(w.eos, ambient_state(w))
    forest = U.forest
    N = forest.N
    cons = Array(U.work)
    moved = zeros(Int, D + 2)
    worst = zeros(T, D + 2)
    blocks = 0
    for b in 1:nblocks(U)
        k = blockkey(U, b)
        level(k) > 0 || continue
        ext = block_extent(forest, k)
        all(d -> ext[d][1] == forest.extents[d][1], 1:D) || continue
        blocks += 1
        for idx in CartesianIndices(ntuple(d -> (U.G[d] + 1):(U.G[d] + N), D))
            for v in 1:(D + 2)
                x = cons[Tuple(idx)..., v, b]
                x == amb[v] && continue
                moved[v] += 1
                worst[v] = max(worst[v], abs(x - amb[v]))
            end
        end
    end
    return (moved=moved, worst=worst, blocks=blocks, ambient=amb)
end

# The runs, computed once. About thirty seconds between them, and every
# testset below reads these rather than running its own, so that the tables
# really are the same runs compared against each other.
const TRACKED_1D = tracked_sedov(Val(1), SEDOV1D)

const TRACKED_2D = tracked_sedov(Val(2), SEDOV2D)
const FINE_2D = uniform_sedov(Val(2), SEDOV2D; scale=2^SEDOV2D.cap)
const COARSE_2D = uniform_sedov(Val(2), SEDOV2D)
const NOFIX_2D = tracked_sedov(Val(2), SEDOV2D; fixup=false)
const ORDER1_2D = tracked_sedov(Val(2), SEDOV2D; p=1)
const STEP_2D = tracked_sedov(Val(2), SEDOV2D; reset=:step)

const TRACKED_3D = tracked_sedov(Val(3), SEDOV3D; from=2)

# The static two-level runs. Every control takes the *same* step count as the
# run it is a control for, or the difference between them would not be the
# thing being measured.
static_sedov(::Val{D}, cfg; p=3, kwargs...) where {D} =
    sedov_static(Val(D); N=cfg.N, ops=sedov_ops(p), roots=cfg.roots,
                 r₀=cfg.r₀, t_end=cfg.t_end, kwargs...)

const CENTER_2D = static_sedov(Val(2), STATIC2D; refined=:center)
const CENTER_2D_STEP = static_sedov(Val(2), STATIC2D; refined=:center,
                                    reset=:step, nsteps=CENTER_2D.nsteps)
const CENTER_2D_NONE = static_sedov(Val(2), STATIC2D; refined=:center,
                                    reset=:none, nsteps=CENTER_2D.nsteps)
const CENTER_2D_P1 = static_sedov(Val(2), STATIC2D; refined=:center, p=1,
                                  nsteps=CENTER_2D.nsteps)
const CENTER_2D_NOFIX = static_sedov(Val(2), STATIC2D; refined=:center,
                                     fixup=false, nsteps=CENTER_2D.nsteps)
const STATIC_2D_COARSE = static_sedov(Val(2), STATIC2D; nsteps=CENTER_2D.nsteps)
const STATIC_2D_FINE = sedov_static(Val(2); N=STATIC2D.N, ops=sedov_ops(),
                                    roots=2 * STATIC2D.roots, r₀=STATIC2D.r₀,
                                    t_end=STATIC2D.t_end,
                                    nsteps=CENTER_2D.nsteps)
const STATIC_2D_FINE_NOFIX = sedov_static(Val(2); N=STATIC2D.N, ops=sedov_ops(),
                                          roots=2 * STATIC2D.roots,
                                          r₀=STATIC2D.r₀, t_end=STATIC2D.t_end,
                                          fixup=false, nsteps=CENTER_2D.nsteps)

const CENTER_3D = static_sedov(Val(3), STATIC3D; refined=:center)
const CENTER_3D_STEP = static_sedov(Val(3), STATIC3D; refined=:center,
                                    reset=:step, nsteps=CENTER_3D.nsteps)
const CENTER_3D_NOFIX = static_sedov(Val(3), STATIC3D; refined=:center,
                                     fixup=false, nsteps=CENTER_3D.nsteps)
const STATIC_3D_COARSE = static_sedov(Val(3), STATIC3D; nsteps=CENTER_3D.nsteps)

# The boundary meshes, run for a few tens of steps — long enough that the
# interior near the blast has moved and short enough that the corner has not
# heard of it, which is what makes the claim about the exchange.
const CORNER_2D = sedov_static(Val(2); N=8, ops=sedov_ops(), roots=4,
                               r₀=1 // 16, t_end=1 // 100, refined=:corner)
const EDGE_3D = sedov_static(Val(3); N=4, ops=sedov_ops(), roots=4, r₀=1 // 8,
                             t_end=1 // 100, refined=:edge)
const CORNER_3D = sedov_static(Val(3); N=4, ops=sedov_ops(), roots=4,
                               r₀=1 // 8, t_end=1 // 100, refined=:corner)

@testset "The similarity law reproduces Taylor's constant and its own exponent" begin
    # The failure mode is a silent one: a wrong `ξ₀` puts a constant factor
    # into every radius the law predicts, and since the *exponent* is what the
    # runs are judged on, nothing downstream would notice. So the constant is
    # checked against the literature and the parametrization it comes from is
    # checked against the equation it solves.
    #
    # Measured (γ = 7/5):
    #   D = 1   α = 1.0774855847350489   ξ₀ = 0.9754301541925205
    #   D = 2   α = 0.9840740168800447   ξ₀ = 1.0040216061302776
    #   D = 3   α = 0.8510718547582286   ξ₀ = 1.0327774677614250
    # and γ = 5/3 spherical gives ξ₀ = 1.1516664179314904.
    sims = [SedovSimilarity(7 // 5, D) for D in 1:3]

    # Taylor's classical value, which is the check `CODE.md` names.
    @test sims[3].ξ₀ ≈ 1.033 atol = 5e-4
    @test SedovSimilarity(5 // 3, 3).ξ₀ ≈ 1.1517 atol = 5e-4

    # The planar value is **twice** the literature's, and the factor is the
    # deposition convention rather than an error: `σ_1 = 2` counts both sides
    # of `|x| < r`, which is the volume `sedov_state` deposits into, and Kamm
    # & Timmes count one. Their recalled α — 0.5386, 0.9840, 0.8511 — then
    # agree in every digit recalled.
    @test sims[1].α / 2 ≈ 0.5386 atol = 5e-4
    @test sims[2].α ≈ 0.9840 atol = 5e-4
    @test sims[3].α ≈ 0.8511 atol = 5e-4
    @test all(d -> sims[d].ξ₀ ≈ sims[d].α^(-1 / (d + 2)), 1:3)

    # The exponent needs no constant at all: it follows from the dimensions.
    @test sedov_exponent(1) == 2 / 3
    @test sedov_exponent(2) == 1 / 2
    @test sedov_exponent(3) == 2 / 5
    @test all(d -> sedov_exponent(sims[d]) == 2 / (d + 2), 1:3)

    # And the law scales as it says: `r_s ∝ (E₀ t²)^{1/(D+2)}`, so doubling
    # `E₀` and quadrupling `t` each multiply it by a known factor.
    for (d, sim) in enumerate(sims)
        @test sedov_radius(sim, 0.0, 1.0) == 0.0
        @test sedov_radius(sim, 0.1, 2.0) ≈
              2^(1 / (d + 2)) * sedov_radius(sim, 0.1, 1.0)
        @test sedov_radius(sim, 0.4, 1.0) ≈
              4^(2 / (d + 2)) * sedov_radius(sim, 0.1, 1.0)
        @test sedov_radius(sim, 0.1, 1.0, 4.0) ≈
              4^(-1 / (d + 2)) * sedov_radius(sim, 0.1, 1.0)
    end

    # The derivation, not the transcription: the parametric `(λ, G, V, Z)`
    # satisfies the momentum equation of the similarity system,
    #   (V−ν) dV/dlnλ + V² − V + (Z dlnG/dlnλ + 2Z + dZ/dlnλ)/γ = 0,
    # which is the one of the three the closed form was *not* built from —
    # `Z(V)` came from the energy integral and `λ(V)`, `G(V)` from continuity
    # and entropy. Measured worst residual over the profile: 3.6e-14.
    worst = 0.0
    for sim in sims, f in 0.01:0.03:1.0
        u = f * sim.u₂
        λ, G, V, Z = sedov_profile(sim, u)
        dl = TreeHydro.sedov_dlnλ(sim, u)
        dlnZdV = 2 / V + 1 / (u + sim.dν) + sim.γ / (-sim.γ * u)
        res = (V - sim.ν) / dl + V^2 - V +
              (Z * TreeHydro.sedov_dlnG(sim, u) / dl + 2Z + Z * dlnZdV / dl) / sim.γ
        worst = max(worst, abs(res))
    end
    @test worst < 1e-12
    # The shock itself, where the profile must be the Rankine–Hugoniot state.
    for sim in sims
        λ, G, V, Z = sedov_profile(sim, sim.u₂)
        @test λ ≈ 1
        @test G ≈ (sim.γ + 1) / (sim.γ - 1)
        @test V ≈ 2 * sim.ν / (sim.γ + 1)
    end

    @test_throws "needs γ > 1" SedovSimilarity(1.0, 3)
    @test_throws "D = 1, 2 or 3" SedovSimilarity(7 // 5, 4)
    @test_throws "at least two samples" exponent_fit([1.0, 2.0], [0.1, 0.2];
                                                     from=1.0)

    @info "Sedov similarity: α = " *
          "$(sims[1].α), $(sims[2].α), $(sims[3].α) and ξ₀ = " *
          "$(sims[1].ξ₀), $(sims[2].ξ₀), $(sims[3].ξ₀) in D = 1, 2, 3 at " *
          "γ = 7/5; ξ₀ = $(SedovSimilarity(5//3, 3).ξ₀) at γ = 5/3 in D = 3; " *
          "worst momentum-equation residual $worst"
end

@testset "The deposition is resolved and the energy the law takes is the measured one" begin
    # Two failure modes. A top hat the initial-data cycle does not put at the
    # cap is a delta function the scheme cannot represent, and the run would
    # merely look diffusive; and an `E₀` read off the nominal value rather
    # than off the mesh puts a systematic error into every radius the law
    # predicts, since which cell centres fall inside `r₀` is a property of the
    # mesh and not of the case.
    #
    # Measured: the cycle converges in 2 passes in D = 2 and the deposition
    # sits at the cap; measured E₀ / nominal is 1.00000 in D = 1 (the planar
    # top hat is a whole number of cells), 1.03451 in D = 2 and 1.04445 in
    # D = 3.
    for (D, run, expected) in ((1, TRACKED_1D, 1.0), (2, TRACKED_2D, 1.03451),
                               (3, TRACKED_3D, 1.04445))
        r, w = run.r, run.w
        @test r.converged
        @test 1 ≤ r.passes ≤ 8
        cfg = D == 1 ? SEDOV1D : D == 2 ? SEDOV2D : SEDOV3D
        @test maxlevel(r.forest) == cfg.cap
        # The blocks holding the deposition are at the cap: every leaf whose
        # extent overlaps the top hat.
        holding = [k for k in r.forest.leaves
                   if all(d -> block_extent(r.forest, k)[d][1] < w.r₀ &&
                                   block_extent(r.forest, k)[d][2] > -w.r₀, 1:D)]
        @test !isempty(holding)
        @test all(k -> level(k) == maxlevel(r.forest), holding)

        # The measured energy is the one the law is read with, and it is not
        # the nominal one — in `D = 2, 3` because the cells whose centres fall
        # inside `r₀` do not tile `V_D(r₀)`, and in `D = 1`, where they tile it
        # exactly, because the ambient subtracted over the whole box includes
        # the top hat's own share, `p_amb V_1(r₀)/(γ−1) = 3.125e-6`.
        @test run.E₀ ≈ expected rtol = 1e-4
        @test run.E₀ != w.E₀
        @test 0.95 ≤ run.E₀ / w.E₀ ≤ 1.1
        @info "Sedov deposition, D = $D: cycle converged in $(r.passes) " *
              "passes onto $(r.nblocks_history[1]) blocks at levels " *
              "$(r.levels); measured E₀ = $(run.E₀), a ratio of " *
              "$(run.E₀ / w.E₀) to the nominal; p_hot = $(w.p_hot), " *
              "c_s = $(soundspeed(w.eos, w.ρ₀, w.p_hot)), the top hat " *
              "$(round(Int, 2 * w.r₀ / r.h)) cells across at the cap"
    end
end

@testset "The tracked blast follows the similarity law: D=$D" for (D, run, tol) in
                                                                  ((1, TRACKED_1D, 0.08),
                                                                   (2, TRACKED_2D, 0.05),
                                                                   (3, TRACKED_3D, 0.08))
    # The failure mode is a blast that expands at the wrong rate, which a
    # profile plot does not show and which a conservation test cannot see at
    # all. The exponent is the check that needs no constant: `log r_s` against
    # `log t` has slope `2/(D+2)` whatever `γ` and `ξ₀` are.
    #
    # **The tolerances, and why they differ.** The radius is quantized at the
    # cell spacing — `shock_radius` returns the outermost *cell centre* above
    # the density threshold — so each sample carries ±h/2, which is ±1.2% of
    # `r_s` at the end of the D = 2 run and ±2% at the start of the fit. That
    # alone admits about 0.03 of slope. On top of it the law is the
    # *asymptotic* solution and the fit runs over chunks with `r_s ≥ 3r₀`
    # (`2r₀` in D = 3), which the runs reach only by a factor of 3.8, 5.3 and
    # 2.7 — so the D = 1 and D = 3 runs are fitted over a blast that has
    # barely forgotten its top hat and are allowed 0.08.
    #
    # Measured:  D = 1  0.64150 against 2/3   r_s/r₀ 3.81  peak 4.11090
    #            D = 2  0.50443 against 1/2   r_s/r₀ 5.31  peak 3.76529
    #            D = 3  0.43766 against 2/5   r_s/r₀ 2.72  peak 2.05717
    r, w = run.r, run.w
    ν = sedov_exponent(D)
    @test abs(run.exponent - ν) ≤ tol
    @test r.tracking == 1.0

    # The arrival check is the *case's* and not the driver's, and a run of
    # this file is where it is called: the driver knows neither where the
    # feature is nor what λ will become, and this case knows both in closed
    # form.
    @test assert_no_arrival(w, r.forest, Float64(run.ts[end]), run.E₀) === nothing

    # The post-shock density jump. The strong-shock limit is
    # `(γ+1)/(γ−1) = 6` and a captured shock reaches it only from below — the
    # peak is the average over the cell the front sits in — so the upper bound
    # is the physics, with one percent of slack for an overshoot the limiter
    # is supposed to prevent, and the lower bound is `shock_radius`'s own
    # threshold: a peak at or below `3/2` would mean no shock was found and
    # every radius above would be a zero.
    @test 3 // 2 < run.peak ≤ 6 * (1 + 1 // 100)
    @test run.r_s == maximum(run.rs)            # the blast only ever expands

    @info "Sedov tracked, D = $D: exponent $(run.exponent) against " *
          "$(ν) over the chunks with r_s ≥ $((D == 3 ? 2 : 3) * w.r₀) " *
          "(r_s reaching $(run.r_s), $(run.r_s / w.r₀) times r₀), peak " *
          "compression $(run.peak) against the strong-shock 6; $(r.nsteps) " *
          "steps in $(r.nchunks) chunks with $(r.nregrids) mesh changes, " *
          "$(r.cells) cells at levels $(r.levels), tracking $(r.tracking)"
end

@testset "The blast's first chunk outgrows the speed it was sized from" begin
    # `CODE.md` said the hot spot's sound speed at t = 0 is the maximum and
    # `λ_max` only decreases. That is right about the blast and wrong about
    # the first chunk, for exactly the reason it was wrong on Sod: the jump at
    # `r₀` is a Riemann problem, and the gas on the hot side of its contact
    # moves, so `|v| + c_s` there exceeds the hot spot's own `c_s` before the
    # discontinuity has resolved.
    #
    # Measured first-chunk growth λ_end/λ:  D = 1  1.16537
    #                                       D = 2  1.25164
    #                                       D = 3  1.10396
    # and the largest growth over the whole run, against the headroom of 2:
    #   D = 1  1.32100   D = 2  1.44736   D = 3  1.28618
    for (D, run) in ((1, TRACKED_1D), (2, TRACKED_2D), (3, TRACKED_3D))
        r = run.r
        growth = r.λ_end_history ./ r.λ_history
        @test growth[1] > 1                    # it grows, and CODE.md said not
        @test maximum(growth) ≤ 2              # the headroom covers it
        # λ falls after the first chunks, which is the half of the sentence
        # that was right.
        @test r.λ_end_history[end] < r.λ_end_history[1]
        @info "Sedov λ growth, D = $D: first chunk $(r.λ_history[1]) → " *
              "$(r.λ_end_history[1]), a factor of $(growth[1]); worst over " *
              "the run $(maximum(growth)) against speed_headroom = 2; λ " *
              "ending at $(r.λ_end_history[end])"
    end
end

@testset "The tracked blast matches the uniformly fine reference at fewer cells: D=2" begin
    # What would pass without the claim: a run that refines everything matches
    # the fine reference and costs as much, and a run that refines nothing
    # matches the coarse control. So the claim is two-sided, and the coarse
    # control is what says the difference is worth anything.
    #
    # Measured (reduced onto the 32² grid the three meshes share):
    #   |tracked − fine| = 8.294e-15 at 12544 cells against 16384
    #   |coarse  − fine| = 1.119e-1  at  1024 cells
    # The tracked run reproduces the fine run to **roundoff**, which is a
    # sharper statement than the tube's 1.0004 and has a reason: outside the
    # refined region the gas is undisturbed, so the coarse blocks hold the
    # ambient exactly and there is nothing for them to get wrong.
    M = SEDOV2D.roots * SEDOV2D.N
    gt = reduce_to_grid(TRACKED_2D.r.U, M)
    gf = reduce_to_grid(FINE_2D.r.U, M)
    gc = reduce_to_grid(COARSE_2D.r.U, M)
    tracked_fine = l1_difference(gt, gf)
    coarse_fine = l1_difference(gc, gf)

    @test TRACKED_2D.r.cells < FINE_2D.r.cells
    @test tracked_fine < coarse_fine / 1e6
    @test tracked_fine ≤ 1e-12
    # And the two agree on what they are measuring, not merely on the state.
    @test TRACKED_2D.peak ≈ FINE_2D.peak rtol = 1e-3
    # The radius is quantized at the cell spacing the two runs share, so the
    # comparison carries one cell. Measured, they are equal.
    @test TRACKED_2D.r_s ≈ FINE_2D.r_s atol = TRACKED_2D.r.h
    @test COARSE_2D.peak < TRACKED_2D.peak      # the control is worse

    @info "Sedov tracked, D = 2: $(TRACKED_2D.r.cells) cells against " *
          "$(FINE_2D.r.cells) uniformly fine and $(COARSE_2D.r.cells) " *
          "uniformly coarse; reduced onto the common $(M)² grid " *
          "|tracked − fine| = $tracked_fine against |coarse − fine| = " *
          "$coarse_fine; peak $(TRACKED_2D.peak), $(FINE_2D.peak) and " *
          "$(COARSE_2D.peak); exponent $(TRACKED_2D.exponent), " *
          "$(FINE_2D.exponent) and $(COARSE_2D.exponent)"
end

@testset "The tracked mesh keeps its coarse-fine faces in undisturbed gas: D=2" begin
    # The measurement that says what the tracked blast **cannot** answer, and
    # it is the reason the tables below are taken on a static mesh instead.
    #
    # `tracking == 1` means every strongly firing cell sits on a block at the
    # cap; the travelling margin then puts the refined region's boundary ahead
    # of the shock by construction, so every coarse-fine face of a tracked run
    # stands in gas the blast has not reached. Nothing crosses it, the
    # interface flux restriction has nothing to restrict, the prolongation has
    # nothing but the ambient to prolongate, and no cell is ever unphysical.
    #
    # Measured: `fixup = false`, `p = 1` and `reset = :step` give the *same*
    # exponent, peak, cell count, mesh history and tracking as the run they
    # are controls for, and the two reset cadences agree **bit for bit**.
    for (tag, run) in (("fixup = false", NOFIX_2D), ("p = 1", ORDER1_2D),
                       ("reset = :step", STEP_2D))
        @test run.r.cells == TRACKED_2D.r.cells
        @test run.r.nsteps == TRACKED_2D.r.nsteps
        @test run.r.nblocks_history == TRACKED_2D.r.nblocks_history
        @test run.r.tracking == TRACKED_2D.r.tracking
        @test run.exponent ≈ TRACKED_2D.exponent rtol = 1e-9
        @test run.peak ≈ TRACKED_2D.peak rtol = 1e-9
        @test (run.r.floor_hits, run.r.reset_hits, run.r.ghost_hits) == (0, 0, 0)
        @test run.r.injection == ntuple(_ -> 0.0, 4)
        @info "Sedov tracked, D = 2, $tag: exponent $(run.exponent), peak " *
              "$(run.peak), $(run.r.cells) cells, floor/reset/ghost hits " *
              "$(run.r.floor_hits)/$(run.r.reset_hits)/$(run.r.ghost_hits), " *
              "injection $(run.r.injection)"
    end
    # The reset cadence is not merely equivalent here, it is the same bits:
    # the reset writes back only the cells a floor fired in, and none did.
    @test STEP_2D.r.u == TRACKED_2D.r.u
    @test STEP_2D.r.drift == TRACKED_2D.r.drift
    # Conservation on the tracked mesh is therefore the plain claim, with the
    # injection exactly zero rather than zero to a tolerance. The ambient is
    # at rest and the two faces of each axis carry the same pressure flux, so
    # before the shock arrives the Dirichlet boundary contributes exactly
    # nothing — measured: every drift at or below its roundoff bound, where
    # Sod's mass and energy drifts at the same stage were the boundary's own
    # numerical flux and were not.
    for v in 1:4
        @test TRACKED_2D.r.drift[v] ≤
              8 * eps(Float64) * TRACKED_2D.r.scales[v] * TRACKED_2D.r.nsteps
    end
    @info "Sedov tracked, D = 2: drift $(TRACKED_2D.r.drift) against roundoff " *
          "bounds $(ntuple(v -> 8 * eps(Float64) * TRACKED_2D.r.scales[v] * TRACKED_2D.r.nsteps, 4)) " *
          "over $(TRACKED_2D.r.nsteps) steps and $(TRACKED_2D.r.nregrids) mesh " *
          "changes, injection $(TRACKED_2D.r.injection)"
end

@testset "The refined region follows the blast as a disk, not as a shell: D=2" begin
    # `CODE.md` predicted "a refined shell that grows while the interior
    # coarsens", and the block count "rising then falling behind the shock".
    # Measured, it does neither, and the reason is physics rather than a
    # defect: the Sedov interior is a **steep density ramp** — the similarity
    # solution's `G ∼ λ^{D/(γ−1)}` — not a flat bubble, so the Löhner
    # indicator on `ρ` fires throughout it and the refined region is a growing
    # *disk*. Only at the very centre, where the ramp has flattened, does the
    # criterion fall silent, and that is where the hollow opens first.
    #
    # Measured block history: 40 → 88 → 112 → … → 196, monotone, over 40
    # chunks; at t_end the blocks within r < 0.06 report Coarsen and those
    # between 0.06 and 0.42 fire.
    r = TRACKED_2D.r
    @test issorted(r.nblocks_history)           # rising, and never falling
    @test r.nblocks_history[end] > r.nblocks_history[1]
    @test r.cells < FINE_2D.r.cells             # it is still a saving

    # It is still a hierarchy at the end and not a uniform mesh: some blocks
    # are below the cap, and they are the ones the blast has not reached.
    below = count(k -> level(k) < maxlevel(r.forest), r.forest.leaves)
    @test below > 0
    @test r.levels == [1, 2]

    @info "Sedov tracked, D = 2: block count $(r.nblocks_history[1]) → " *
          "$(r.nblocks_history[end]) over $(r.nchunks) chunks, monotone " *
          "(the interior fires because the Sedov density ramp is steep, so " *
          "the refined region is a disk and not a shell); $below of " *
          "$(r.nblocks) blocks are below the cap at t_end, at levels " *
          "$(r.levels)"
end

@testset "A strong shock crossing a coarse-fine face conserves, and leaks without the fixup: D=2" begin
    # The claim the package exists to make, on the one configuration of this
    # case that can make it: the static `:center` mesh, where the blast starts
    # inside the refined region and leaves it at |x_d| = 1/4. The tracked mesh
    # cannot — see the testset above.
    #
    # Measured (433 steps, 28 blocks, levels [0, 1], r_s = 0.34587):
    #   with the fixup     mass 1.33e-15   energy drift 1.24659e-5
    #                      = the reset's injection, to roundoff
    #   without it         mass 4.237e-3   energy 2.929e-2
    #   reset = :none      mass 1.22e-15   energy 1.554e-15  (nothing injected)
    # so the fixup buys 3.2e12 in mass and the leak is not the reset's.
    T = Float64
    rb(r, v) = 8 * eps(T) * r.scales[v] * r.nsteps
    r, c = CENTER_2D, CENTER_2D_NOFIX
    @test r.levels == [0, 1]
    @test r.nsteps == c.nsteps == CENTER_2D_NONE.nsteps
    @test r.r_s > 1 // 4                        # the shock really did leave

    # With the fixup: mass and momentum at roundoff, and the energy drift is
    # the reset's injection and nothing else — which the `:none` control
    # proves by having neither.
    @test r.drift[1] ≤ rb(r, 1)
    @test r.drift[2] ≤ rb(r, 2) && r.drift[3] ≤ rb(r, 3)
    @test all(v -> CENTER_2D_NONE.drift[v] ≤ rb(CENTER_2D_NONE, v), 1:4)
    @test CENTER_2D_NONE.injection == (0.0, 0.0, 0.0, 0.0)

    # Without it, every conserved integral leaks by ten orders of magnitude
    # more than roundoff, on the same mesh and the same step count.
    @test c.drift[1] > 1e9 * max(r.drift[1], rb(r, 1))
    @test c.drift[4] > 1e9 * rb(r, 4)
    # And the single-level control at the same spacing is **bit-identical**
    # with the fixup and without it: the leak is the coarse-fine face's and
    # not the scheme's.
    @test STATIC_2D_FINE.levels == [0]
    @test STATIC_2D_FINE.u == STATIC_2D_FINE_NOFIX.u
    @test STATIC_2D_FINE.drift == STATIC_2D_FINE_NOFIX.drift
    @test STATIC_2D_COARSE.drift[1] ≤ rb(STATIC_2D_COARSE, 1)

    @info "Sedov :center, D = 2 ($(r.nblocks) blocks, $(r.nsteps) steps, " *
          "r_s = $(r.r_s)): with the fixup drift $(r.drift) against roundoff " *
          "bounds $(ntuple(v -> rb(r, v), 4)); without it $(c.drift); with " *
          "reset = :none $(CENTER_2D_NONE.drift)"
end

@testset "The reset's injection is the energy drift, and the cadence is what the bookkeeping sees" begin
    # Two claims and one amendment. Where the floors fire, the drift is
    # *claimed net of* the measured injection — and under `reset = :step`,
    # which resets once per step on the step's own result, the two agree to
    # roundoff, which is the equality `CODE.md` predicted.
    #
    # Under `reset = :stage` they do **not**, and the reason is the method
    # rather than the measurement: `SSPRK33`'s three stages enter the step's
    # result with weights 1/6, 2/3 and 1, so an injection into a stage vector
    # reaches the state multiplied by that stage's weight, while the
    # accounting adds the raw `Σ hᴰ ΔU` of each call. The accumulated
    # injection is therefore an **upper bound** on what arrived, not an
    # equality. Step 8 could not see this: on Sod and the entropy wave the
    # injection was exactly zero, so every weighting of it was too.
    #
    # Measured (2D :center, 433 steps):
    #   :stage  reset hits 4096   injection 2.39804e-5   drift 1.24659e-5
    #   :step   reset hits 1384   injection 1.245738e-5  drift 1.245738e-5
    #           |drift − injection| = 4.44e-16 against a roundoff bound of
    #           7.96e-13
    # and in 3D (133 steps): :stage 24504 hits and 8.52677e-5 against
    # 4.45123e-5; :step 8232 hits and 4.447276224328611e-5 against
    # 4.447276224350816e-5, a difference of 2.22e-16.
    T = Float64
    for (D, stage, step) in ((2, CENTER_2D, CENTER_2D_STEP),
                             (3, CENTER_3D, CENTER_3D_STEP))
        E = D + 2
        bound = 8 * eps(T) * step.scales[E] * step.nsteps
        @test step.reset_hits > 0               # the floors fire here
        @test stage.reset_hits > step.reset_hits
        # `:step` is the exact statement.
        @test abs(step.drift[E] - step.injection[E]) ≤ bound
        # `:stage` is the bound, and it is not tight.
        @test stage.drift[E] ≤ stage.injection[E]
        @test stage.drift[E] > stage.injection[E] / 3
        # The injection is energy only, because it is the *pressure* floor
        # that fires and not the atmosphere rule: the pressure floor keeps ρ
        # and v and changes E alone. The mass injection is therefore **exactly
        # zero** — `apply_floors` returns ρ unchanged and `prim2con` writes it
        # back bit for bit — and the momentum injection is roundoff rather
        # than zero, because the same round trip recomputes `S` as `ρ (S/ρ)`,
        # which is not the bits it started from. Measured in `D = 3`:
        # `-1.65e-24` against a roundoff bound of `2.2e-14`.
        @test stage.injection[1] == 0
        @test all(v -> abs(stage.injection[v]) ≤
                       8 * eps(T) * stage.scales[v] * stage.nsteps, 2:(D + 1))
        @info "Sedov :center, D = $D: reset = :stage floored " *
              "$(stage.reset_hits) owned cells and reports an injection of " *
              "$(stage.injection[E]) against a drift of $(stage.drift[E]) " *
              "(a ratio of $(stage.drift[E] / stage.injection[E]), the SSPRK " *
              "stage weights); reset = :step floored $(step.reset_hits) and " *
              "reports $(step.injection[E]) against $(step.drift[E]), a " *
              "difference of $(abs(step.drift[E] - step.injection[E])) " *
              "against a roundoff bound of $bound"
    end
end

@testset "p = 1 keeps the floors quiet where p = 3 does not, and that is the answer Sod could not give" begin
    # The open question of "Operator order", closed. Step 5 measured the
    # smooth case — `p = 1` costs a full order in L∞ and nothing in L1 — and
    # step 7 measured a discontinuous one on Sod, where `p = 1` was worse in
    # every column and bought nothing, *because no floor fired at all*. The
    # positivity half of the argument needed a prolongation acting across a
    # strong shock, and this is it.
    #
    # Measured (2D :center, 433 steps, L1 against the uniformly fine run on
    # the 32² grid the meshes share):
    #   p = 3   L1 4.406604e-2   floor/reset/ghost 0/4096/40   energy drift 1.24659e-5
    #   p = 1   L1 4.427249e-2   floor/reset/ghost 0/   0/ 0   energy drift 1.998e-15
    #   fixup=false (p = 3)      L1 6.042174e-2
    #   uniform coarse           L1 9.536813e-2
    # So `p = 1` buys **exact positivity** — nothing floored, nothing
    # injected, conservation at roundoff — for 0.47% of L1. `p = 3` stays the
    # default, because the interface-order rule is what it is there for and
    # the accuracy is better; the number a positivity-critical run would trade
    # is now measured rather than assumed.
    M = STATIC2D.roots * STATIC2D.N
    gf = reduce_to_grid(STATIC_2D_FINE.U, M)
    l1(r) = l1_difference(reduce_to_grid(r.U, M), gf)
    l1_p3, l1_p1 = l1(CENTER_2D), l1(CENTER_2D_P1)
    T = Float64

    @test CENTER_2D.reset_hits > 0 && CENTER_2D.ghost_hits > 0
    @test (CENTER_2D_P1.reset_hits, CENTER_2D_P1.ghost_hits) == (0, 0)
    @test CENTER_2D_P1.drift[4] ≤ 8 * eps(T) * CENTER_2D_P1.scales[4] *
                                  CENTER_2D_P1.nsteps
    @test CENTER_2D_P1.injection == (0.0, 0.0, 0.0, 0.0)
    # The accuracy price, which is what makes it a trade rather than a win.
    @test l1_p1 > l1_p3
    @test l1_p1 < 1.02 * l1_p3
    # Both beat the coarse control and the run without the fixup.
    @test l1(CENTER_2D_NOFIX) > 1.2 * l1_p3
    @test l1(STATIC_2D_COARSE) > 2 * l1_p3

    @info "Sedov :center, D = 2, p = 3: L1 $l1_p3 against the uniformly fine " *
          "run, floor/reset/ghost hits $(CENTER_2D.floor_hits)/" *
          "$(CENTER_2D.reset_hits)/$(CENTER_2D.ghost_hits), injection " *
          "$(CENTER_2D.injection[4]), peak $(CENTER_2D.peak)"
    @info "Sedov :center, D = 2, p = 1: L1 $l1_p1 against the uniformly fine " *
          "run, floor/reset/ghost hits $(CENTER_2D_P1.floor_hits)/" *
          "$(CENTER_2D_P1.reset_hits)/$(CENTER_2D_P1.ghost_hits), injection " *
          "$(CENTER_2D_P1.injection[4]), peak $(CENTER_2D_P1.peak) — a " *
          "$(round(100 * (l1_p1 / l1_p3 - 1); digits=2))% price in L1 for " *
          "exact positivity"
    @info "Sedov :center, D = 2, controls: L1 $(l1(CENTER_2D_NOFIX)) without " *
          "the fixup and $(l1(STATIC_2D_COARSE)) on the uniform coarse mesh"
end

@testset "The 3D coarse-fine face carries a real shock, and the fixup is what conserves" begin
    # The 3D face is where the fixup averages `2 × 2` fine faces, and this is
    # the first time a *shock* crosses one: `interface_tests.jl` makes the same
    # claim in 3D on the smooth entropy wave, and step 5's Sod runs reach only
    # `D = 2`.
    #
    # Measured (133 steps, 120 blocks, levels [0, 1], r_s = 0.28082):
    #   with the fixup     mass 3.251e-11   energy 4.45123e-5 = injection
    #   without it         mass 2.759e-3    energy 5.895e-2
    #   uniform control    mass 2.773e-10
    # The mass drift is **not** roundoff and is not the face's: the uniform
    # mesh at the same coarse spacing leaks eight times more, which is the
    # numerical precursor of a strong blast reaching the Dirichlet boundary —
    # the trap `CLAUDE.md` records for Sod's tube, met again here.
    T = Float64
    r, c, u = CENTER_3D, CENTER_3D_NOFIX, STATIC_3D_COARSE
    rb(x, v) = 8 * eps(T) * x.scales[v] * x.nsteps
    @test r.levels == [0, 1]
    @test r.nsteps == c.nsteps == u.nsteps
    @test r.r_s > 1 // 4                        # the shock left the fine region
    # Mass against the uniform mesh of the same coarse spacing, not against
    # roundoff, and the momenta against roundoff since nothing injects them.
    @test r.drift[1] ≤ max(rb(r, 1), 10 * u.drift[1])
    @test all(v -> r.drift[v] ≤ rb(r, v), 2:4)
    # The leak, on the same mesh and the same step count.
    @test c.drift[1] > 1e6 * max(r.drift[1], rb(r, 1))
    @test c.drift[5] > 1e2 * r.drift[5]

    @info "Sedov :center, D = 3 ($(r.nblocks) blocks, $(r.nsteps) steps, " *
          "r_s = $(r.r_s), peak $(r.peak)): with the fixup drift $(r.drift), " *
          "without it $(c.drift), uniform mesh at the same coarse spacing " *
          "$(u.drift); floor/reset/ghost hits $(r.floor_hits)/" *
          "$(r.reset_hits)/$(r.ghost_hits) against $(c.floor_hits)/" *
          "$(c.reset_hits)/$(c.ghost_hits)"
end

@testset "The floors fire where the design says they would, and not where it said they would" begin
    # `CODE.md` expected Sedov to be "the first case in which a floor actually
    # fires", and put the firing in the **evacuated interior**, where the
    # similarity solution's density falls six orders of magnitude. Measured,
    # that is not where it fires and not what fires.
    #
    #   * The interior never reaches `ρ_atm = 10⁻⁶`. Numerical diffusion
    #     refills the bubble long before: the minimum density is 6.7e-2 in the
    #     tracked D = 2 run, so the *atmosphere* rule never fires on this case
    #     at any size the tests can afford, and the injection into mass and
    #     momentum is exactly zero everywhere below.
    #   * What fires is the **pressure floor**, and what makes it fire is the
    #     coarse-fine face: the interface flux restriction replaces a coarse
    #     cell's flux with the average of its fine neighbours', and in gas
    #     whose internal energy is `p_amb/(γ−1) = 2.5e-5` that correction can
    #     take it below `p_floor`. On the uniform meshes and on the tracked
    #     mesh — whose faces stand in quiet gas — nothing fires at all.
    #
    # Measured floor/reset/ghost hits:
    #   tracked D = 2      0/    0/   0        uniform D = 2   0/0/0
    #   tracked D = 3      0/  288/1152
    #   :center D = 2      0/ 4096/  40
    #   :center D = 3     75/24504/4703
    @test (TRACKED_2D.r.floor_hits, TRACKED_2D.r.reset_hits,
           TRACKED_2D.r.ghost_hits) == (0, 0, 0)
    @test (FINE_2D.r.floor_hits, FINE_2D.r.reset_hits, FINE_2D.r.ghost_hits) ==
          (0, 0, 0)
    @test (STATIC_2D_COARSE.reset_hits, STATIC_2D_COARSE.ghost_hits) == (0, 0)
    # Where there *is* a coarse-fine face with a shock on it, all three
    # populations are nonzero — which is the measurement the upstream
    # prolongation question was waiting for.
    @test CENTER_2D.ghost_hits > 0
    @test CENTER_3D.ghost_hits > 0
    @test TRACKED_3D.r.ghost_hits > 0
    # And no mass is ever injected, because the atmosphere rule never fires:
    # only the pressure floor does, and it changes `E` alone. The momentum
    # injection is roundoff rather than exactly zero, for the `ρ (S/ρ)` round
    # trip the testset above records.
    for r in (CENTER_2D, CENTER_3D, CENTER_2D_STEP, CENTER_3D_STEP)
        n = length(r.injection)
        @test r.injection[1] == 0
        @test all(v -> abs(r.injection[v]) ≤
                       8 * eps(Float64) * r.scales[v] * r.nsteps, 2:(n - 1))
    end
    @test TRACKED_3D.r.injection[1] == 0
    @test all(v -> abs(TRACKED_3D.r.injection[v]) ≤
                   8 * eps(Float64) * TRACKED_3D.r.scales[v] *
                   TRACKED_3D.r.nsteps, 2:4)

    @info "Sedov floor counts (owned recovery / reset / ghost): tracked D = 2 " *
          "$(TRACKED_2D.r.floor_hits)/$(TRACKED_2D.r.reset_hits)/$(TRACKED_2D.r.ghost_hits), " *
          "tracked D = 3 $(TRACKED_3D.r.floor_hits)/$(TRACKED_3D.r.reset_hits)/$(TRACKED_3D.r.ghost_hits), " *
          ":center D = 2 $(CENTER_2D.floor_hits)/$(CENTER_2D.reset_hits)/$(CENTER_2D.ghost_hits), " *
          ":center D = 3 $(CENTER_3D.floor_hits)/$(CENTER_3D.reset_hits)/$(CENTER_3D.ghost_hits), " *
          "uniform D = 2 $(FINE_2D.r.floor_hits)/$(FINE_2D.r.reset_hits)/$(FINE_2D.r.ghost_hits); " *
          "the atmosphere rule never fires — every mass injection is exactly " *
          "zero and every momentum injection is roundoff"
end

@testset "A refined region may touch a corner and an edge of the Dirichlet box: $tag" for (tag, res) in
                                                                                          (("D=2 corner", CORNER_2D),
                                                                                           ("D=3 edge", EDGE_3D),
                                                                                           ("D=3 corner", CORNER_3D))
    # The M2 ordering case in full, which "Boundaries" in `CODE.md` has been
    # waiting for since step 4 and which no configuration in this package could
    # produce before: a prolongation reaching **tangentially** into ghosts the
    # physical-boundary hook wrote needs two physical faces meeting, and a
    # shock tube periodic across itself has none. Every face here is Dirichlet,
    # so a fine block in the low corner has ghost regions that only two faces
    # together can fill.
    #
    # Measured: zero mismatched entries out of 1664 (D = 2 corner), 72000
    # (D = 3 edge) and 59360 (D = 3 corner) stored outward-facing entries, and
    # the corner block's interior untouched — ρ and E **bit-identical** to the
    # ambient, the three momenta moving by at most 1.5e-54, which is 1e-34 of
    # one ulp of the ambient energy density and is the pressure flux failing to
    # cancel exactly in 3D where it cancels exactly in 2D.
    D = ndims(res.U.work) - 2
    @test res.levels == [0, 1]
    g = sedov_ghost_mismatches(res.U, res.w)
    @test g.seen > 0
    @test g.bad == 0

    m = corner_block_motion(res.U, res.w)
    @test m.blocks ≥ 1
    amb_E = m.ambient[end]
    @test m.moved[1] == 0                      # ρ bit-identical
    @test m.moved[end] == 0                    # E bit-identical
    # No entry moves by as much as one ulp of the ambient energy density.
    @test all(v -> m.worst[v] ≤ eps(amb_E), 1:(D + 2))
    # The shock never got there, which is what makes this about the exchange.
    @test res.r_s < 1 // 4
    # And the mesh conserves to roundoff while it does it.
    @test all(v -> res.drift[v] ≤
                   8 * eps(Float64) * res.scales[v] * res.nsteps, 1:(D + 2))

    @info "Sedov $tag ($(res.nblocks) blocks, $(res.nsteps) steps): " *
          "$(g.bad) of $(g.seen) stored outward-facing ghost entries differ " *
          "from the conserved ambient; the $(m.blocks) fine corner block(s) " *
          "moved $(m.moved) entries per variable, worst |ΔU| $(m.worst) " *
          "against one ulp of the ambient energy $(eps(amb_E)); drift " *
          "$(res.drift)"
end

@testset "The planar blast runs, and assert_no_arrival refuses a run that would reflect" begin
    # The 1D smoke test, and the guard the case owns.
    #
    # Measured (D = 1, 222 steps, 25 chunks, 2 mesh changes, 112 cells):
    #   exponent 0.64150 against 2/3, peak 4.11090, r_s 0.23830, E₀ exactly 1
    r = TRACKED_1D.r
    @test r.converged
    @test maxlevel(r.forest) == SEDOV1D.cap
    @test (r.floor_hits, r.reset_hits, r.ghost_hits) == (0, 0, 0)
    for v in 1:3
        @test r.drift[v] ≤ 8 * eps(Float64) * r.scales[v] * r.nsteps
    end
    # The planar top hat `|x| < r₀` is a whole number of cells at every
    # spacing that divides `r₀`, so the cells that receive the deposition tile
    # `V_1(r₀) = 2r₀` *exactly* — the one dimension in which they do. What is
    # left between the measured energy and the nominal one is then only the
    # ambient share of the top hat itself, `p_amb·2r₀/(γ−1) = 3.125e-6`, which
    # `measured_E₀` subtracts along with the rest of the box. Measured:
    # 0.999996875.
    @test TRACKED_1D.E₀ ≈ 1 - Float64(TRACKED_1D.w.p_amb) * 2 *
                              Float64(TRACKED_1D.w.r₀) /
                              (Float64(TRACKED_1D.w.eos.γ) - 1) rtol = 1e-12

    @info "Sedov tracked, D = 1: exponent $(TRACKED_1D.exponent) against " *
          "$(2/3), peak $(TRACKED_1D.peak), r_s $(TRACKED_1D.r_s), E₀ " *
          "$(TRACKED_1D.E₀), $(r.nsteps) steps in $(r.nchunks) chunks with " *
          "$(r.nregrids) mesh changes, $(r.cells) cells, drift $(r.drift)"

    # The arrival check: a `t_end` at which the law puts the shock at the box's
    # face is refused *before* the run, not diagnosed after it. At `t = 1` the
    # law puts the D = 2 shock at 1.004, twice the half-width.
    w = SedovBlast(Float64, Val(2))
    forest = sedov_forest(Val(2), 8; roots=4, L=w.L)
    @test assert_no_arrival(w, forest, 1 // 10, 1.0) === nothing
    @test_throws "reaches the Dirichlet boundary before t_end" assert_no_arrival(
        w, forest, 1, 1.0)
    # And it is the *energy* that decides it, not only the time: the same
    # `t_end` with a thousand times the energy reaches the face.
    @test_throws "reaches the Dirichlet boundary" assert_no_arrival(w, forest,
                                                                    1 // 10, 1e3)

    # The mesh constructor refuses what it cannot build, naming the reason.
    @test_throws "needs D ≥ 3" sedov_forest(Val(2), 8; roots=4, refined=:edge)
    @test_throws "must be false, :none, :center" sedov_forest(Val(2), 8;
                                                              roots=4,
                                                              refined=:middle)
    @test_throws "selected no root block" sedov_forest(Val(2), 8; roots=2,
                                                       refined=:center)
    @test_throws "same root count in every dimension" HydroCase(
        SedovBlast(Float64, Val(2)); roots=(4, 2))
    @test_throws "deposition radius must be inside the box" SedovBlast(
        Float64, Val(2); r₀=1)
    @test_throws "threshold above 1" shock_radius(TRACKED_1D.r.U,
                                                  TRACKED_1D.w; threshold=1)
end
