# **Long tier.** This file runs only under `TREEHYDRO_TEST_LONG=1`; see
# "Testing: two tiers" in `CODE.md`. It is where the physics is claimed and
# where the numbers `CODE.md` records come from, and it may take minutes.
# The short tier pins reduced configurations of the same studies against
# committed references in `test/regression_tests.jl`.
#
# The driver: the one evolve-and-regrid loop, and the first mesh in this
# package that *moves*.
#
# Everything before this file ran on a mesh that was fixed before the first
# step — uniform in steps 3 and 4, statically two-level in step 5 — and step
# 6 measured the criterion without ever letting it change anything. Here the
# criterion drives `regrid!`, the shock tube's refined region follows its
# three waves, and the claims are about what that buys and what it costs.
#
# The failure modes this file guards, one per testset:
#
#   * the initial-data cycle never converging, or converging to a mesh that
#     does not hold the diaphragm at the finest level, or losing the
#     Dirichlet hook on the blocks the cycle created — the hook's second and
#     third call sites (`regrid!` and `adapt_to_initial_data!`) are
#     exercised for the first time here, and forgetting either is a bug that
#     arrives one chunk late;
#   * a refined region that does not actually follow the waves, which looks
#     like a run that merely costs less rather than one that is wrong;
#   * a coarse-fine face that leaks when it is rebuilt every chunk, with
#     `fixup = false` as the negative control;
#   * a buffer too narrow for the motion between regrids;
#   * the prolongation order on a *discontinuous* solution, which the
#     entropy wave of step 5 could not answer;
#   * a time step sized from a signal that grows within the chunk, which is
#     what `speed_headroom` and the end-of-chunk recheck exist for.
#
# The numbers recorded in the comments below are in `CODE.md` under "Step 7"
# in "Measured results"; a changed number is a regression, and a test that
# merely still passes is not.
#
# The three refusals, the CFL recheck's arithmetic and the observer stay in
# the short tier, in `test/driver_tests.jl`, which also defines `driver_ops`,
# the tolerances, `SOD1D`, `sod_case` and `tracked_sod`.

const SOD2D = (roots=(8, 1), N=8, cap=1, chunk=1 // 200, t_end=3 // 20)
"""
The same tube on a uniform mesh with the root brick scaled by `scale`
— `2^cap` for the *fine* reference, which then shares the tracked run's
finest spacing, and `1` for the *coarse* control, which shares its coarsest.
Same chunk, so the two measure `λ_max` at the same cadence.
"""
function uniform_sod(::Val{D}, cfg; scale=1, p=3, fixup=true) where {D}
    return uniform_run(sod_case(Val(D), cfg.roots), Val(D); N=cfg.N,
                       ops=driver_ops(p), t_end=cfg.t_end, chunk=cfg.chunk,
                       limiter=:minmod, roots=cfg.roots .* scale, fixup=fixup)
end

"""
How many stored entries of the tube's **outward-facing** ghost regions do
not hold the conserved boundary state they are supposed to hold for all
time — zero, or the Dirichlet hook did not reach a block the adaptation
cycle or a regrid created.

`interface_tests.jl` asks the same question of a *statically* refined mesh;
each test file here carries its own oracles, and what makes this one a
different claim is that the blocks it inspects did not exist when the run
started. Ghost rows across the tube are included, because TreeAMR fills the
edge regions of a physical face unconditionally and a hook that skipped
them would still leave a plausible profile.
"""
function adapted_ghost_mismatches(U::FieldSet{T,D}, w::SodTube{T,D,DIR}) where {T,D,DIR}
    zeros_ = ntuple(_ -> zero(T), Val(D))
    states = (prim2con(w.eos, (w.ρ_L, zeros_..., w.p_L)),
              prim2con(w.eos, (w.ρ_R, zeros_..., w.p_R)))
    forest = U.forest
    lo, hi = forest.extents[DIR]
    cons = Array(U.work)
    N, G = forest.N, U.G[DIR]
    bad = 0
    for b in 1:nblocks(U)
        ext = block_extent(forest, blockkey(U, b))[DIR]
        for idx in CartesianIndices(size(cons)[1:D]), v in 1:(D + 2)
            i = Tuple(idx)[DIR]
            side = ext[1] == lo && i ≤ G ? 1 : ext[2] == hi && i > G + N ? 2 : 0
            side == 0 && continue
            cons[Tuple(idx)..., v, b] == states[side][v] || (bad += 1)
        end
    end
    return bad
end

"""The closed-form boundary flux the momentum total must move by."""
function sod_boundary_flux(::Val{D}, cfg) where {D}
    w = SodTube(Float64, Val(D))
    case = sod_case(Val(D), cfg.roots)
    area = prod(ntuple(d -> d == 1 ? 1.0 : case.extents[d][2] - case.extents[d][1],
                       D))
    return (w.p_L - w.p_R) * Float64(cfg.t_end) * area
end

# The runs, computed once. Eleven evolutions in `D = 1` and four in `D = 2`,
# about two seconds of the suite between them; every testset below reads
# these rather than running its own, so that the tables really are the same
# runs compared against each other.
const TRACKED_1D = tracked_sod(Val(1), SOD1D)
const FINE_1D = uniform_sod(Val(1), SOD1D; scale=2^SOD1D.cap)
const COARSE_1D = uniform_sod(Val(1), SOD1D)
const NOFIX_1D = tracked_sod(Val(1), SOD1D; fixup=false)
const COARSE_NOFIX_1D = uniform_sod(Val(1), SOD1D; fixup=false)
const BUFFER_1D = [(b, tracked_sod(Val(1), SOD1D; buffer=b)) for b in (2, 1, 0)]
const ORDER_1D = [(p, tracked_sod(Val(1), SOD1D; p=p)) for p in (1, 5)]

const TRACKED_2D = tracked_sod(Val(2), SOD2D)
const FINE_2D = uniform_sod(Val(2), SOD2D; scale=2^SOD2D.cap)
const COARSE_2D = uniform_sod(Val(2), SOD2D)
const NOFIX_2D = tracked_sod(Val(2), SOD2D; fixup=false)

@testset "The initial-data cycle converges on Sod and holds the diaphragm at the cap: D=$D" for (D, cfg) in
                                                                                                ((1, SOD1D),
                                                                                                 (2, SOD2D))
    # Three failure modes in one place. A cycle that never converges leaves
    # a hierarchy that is still growing when the evolution starts, and the
    # run would look merely expensive; a cycle that converges to the *root*
    # mesh — which is what a criterion whose ghosts are stale produces,
    # since Sod's jump sits exactly on a block face — makes every tracking
    # claim below pass on a mesh that never refined; and a cycle that
    # forgets the boundary hook fills the fresh blocks' outer ghosts with
    # whatever the allocation left there, which no profile plot would show.
    r = D == 1 ? TRACKED_1D : TRACKED_2D
    w = SodTube(Float64, Val(D))

    # The driver does **not** make this check and cannot: it compares
    # `t_end · λ` against the distance from the *feature* to the nearest
    # physical boundary, and the driver knows neither where the feature is
    # nor the supremum of λ over all time — the λ it can measure is
    # precisely the one that is not a bound. So the case owns it, and a run
    # of this file's is where it is called.
    @test assert_no_arrival(w, r.forest, Float64(cfg.t_end),
                            max_signal_speed(exact_riemann(w))) === nothing

    @test r.converged
    @test 1 ≤ r.passes ≤ 8                      # measured: 3 in D = 1 and 2
    # The depth is an *output* — but on a discontinuity it is the cap that
    # binds, because a captured shock's τ does not fall with h (step 6).
    @test r.levels == collect(0:(cfg.cap))
    @test maxlevel(r.forest) == cfg.cap

    # Sod's diaphragm lies exactly on a root-block face, so "inside a
    # finest-level block" is the claim that every block touching it is at
    # the cap.
    touching = [k for k in r.forest.leaves
                if any(≈(w.x₀), block_extent(r.forest, k)[1])]
    @test !isempty(touching)
    @test all(k -> level(k) == cfg.cap, touching)

    # And the hook reached them: the outer ghost regions along the tube hold
    # the two conserved boundary states exactly, ghost rows across the tube
    # included, on blocks that did not exist when the run started.
    @test adapted_ghost_mismatches(r.U, w) == 0
end

@testset "The tracked tube matches the uniformly fine reference at fewer cells: D=1" begin
    # What would pass without the claim: a run that refines nothing matches
    # the *coarse* control exactly and costs nothing, and a run that refines
    # everything matches the fine reference and costs more than it. So the
    # claim is two-sided — the error is the fine run's and the cell count is
    # not — and the coarse control is what says the difference between the
    # two errors is worth anything at all.
    #
    # Measured (L1 against the exact Riemann solution, t_end = 1/5):
    #   tracked 4.540016e-3 at  200 cells   ratio to fine 1.0004
    #   fine    4.538238e-3 at  256 cells
    #   coarse  1.669060e-2 at   64 cells   ratio to fine 3.6778
    @test TRACKED_1D.l1 ≤ 1.3 * FINE_1D.l1
    @test COARSE_1D.l1 ≥ 1.5 * FINE_1D.l1
    @test TRACKED_1D.cells < FINE_1D.cells
    @test TRACKED_1D.cells == 200 && FINE_1D.cells == 256 && COARSE_1D.cells == 64

    # The same claim without the exact solution in it: reduced onto the
    # coarse grid the two meshes have in common, the tracked run and the
    # fine run are three orders of magnitude closer to each other than the
    # coarse run is to either. Measured 1.745e-5 against 1.073e-2.
    M = SOD1D.roots .* SOD1D.N
    tracked_fine = l1_difference(reduce_to_grid(TRACKED_1D.U, M),
                                 reduce_to_grid(FINE_1D.U, M))
    coarse_fine = l1_difference(reduce_to_grid(COARSE_1D.U, M),
                                reduce_to_grid(FINE_1D.U, M))
    @test tracked_fine < coarse_fine / 100

    # Every cell whose indicator exceeded refine_tol sat on a block already
    # at the cap, at every one of the 40 chunks. This is the tracking claim
    # proper: a shock never stops firing, so a mesh that fell behind would
    # show up here and nowhere else.
    @test TRACKED_1D.tracking == 1.0
    @test TRACKED_1D.nchunks == 40
    @test TRACKED_1D.nregrids == 9              # the mesh actually changed
    @test TRACKED_1D.floor_hits == 0

    @info "Sod tracked, D = 1: L1 $(TRACKED_1D.l1) at $(TRACKED_1D.cells) cells " *
          "against $(FINE_1D.l1) at $(FINE_1D.cells) uniformly fine (ratio " *
          "$(round(TRACKED_1D.l1 / FINE_1D.l1; digits=4))) and $(COARSE_1D.l1) " *
          "at $(COARSE_1D.cells) uniformly coarse (ratio " *
          "$(round(COARSE_1D.l1 / FINE_1D.l1; digits=4))); reduced onto the " *
          "coarse grid |tracked − fine| = $tracked_fine against " *
          "|coarse − fine| = $coarse_fine; tracking $(TRACKED_1D.tracking) over " *
          "$(TRACKED_1D.nchunks) chunks and $(TRACKED_1D.nregrids) mesh changes"
end

@testset "The tracked tube matches the uniformly fine reference at fewer cells: D=2" begin
    # The planar tube, one level deep and to t = 3/20, which is where the
    # three waves still occupy less than half the box: at t = 1/5 they
    # occupy 59% of it and an adaptive mesh saves 9% of the cells instead of
    # 28%, which is a fact about Sod's problem and not about the driver.
    #
    # Measured: tracked 6.215185e-3 at 1472 cells (ratio to fine 1.0000),
    # fine 6.215186e-3 at 2048, coarse 1.149851e-2 at 512 (ratio 1.8501);
    # reduced onto the common coarse grid, 2.219e-7 against 5.292e-3.
    @test TRACKED_2D.l1 ≤ 1.3 * FINE_2D.l1
    @test COARSE_2D.l1 ≥ 1.5 * FINE_2D.l1
    @test TRACKED_2D.cells < FINE_2D.cells
    @test TRACKED_2D.cells == 1472 && FINE_2D.cells == 2048 && COARSE_2D.cells == 512

    M = SOD2D.roots .* SOD2D.N
    tracked_fine = l1_difference(reduce_to_grid(TRACKED_2D.U, M),
                                 reduce_to_grid(FINE_2D.U, M))
    coarse_fine = l1_difference(reduce_to_grid(COARSE_2D.U, M),
                                reduce_to_grid(FINE_2D.U, M))
    @test tracked_fine < coarse_fine / 100

    @test TRACKED_2D.tracking == 1.0
    @test TRACKED_2D.nregrids ≥ 1
    @test TRACKED_2D.floor_hits == 0

    @info "Sod tracked, D = 2: L1 $(TRACKED_2D.l1) at $(TRACKED_2D.cells) cells " *
          "against $(FINE_2D.l1) at $(FINE_2D.cells) uniformly fine (ratio " *
          "$(round(TRACKED_2D.l1 / FINE_2D.l1; digits=4))) and $(COARSE_2D.l1) " *
          "at $(COARSE_2D.cells) uniformly coarse (ratio " *
          "$(round(COARSE_2D.l1 / FINE_2D.l1; digits=4))); reduced onto the " *
          "coarse grid |tracked − fine| = $tracked_fine against " *
          "|coarse − fine| = $coarse_fine; tracking $(TRACKED_2D.tracking) over " *
          "$(TRACKED_2D.nchunks) chunks and $(TRACKED_2D.nregrids) mesh changes"
end

# The conservation claim for a mesh that is rebuilt while the solution
# crosses it, in the form step 5 established and for the reason it
# established it: with a physical boundary the domain integral is *not*
# constant, so the mass and energy bound is the larger of roundoff and ten
# times what the uniform mesh at the same coarse spacing drifts by — the
# boundary's own numerical flux is what remains there — and the momentum's
# yardstick is the closed form `(p_L − p_R)·t_end·A` it must equal, since
# Sod starts at rest and `Σ hᴰ |S|` gives it no scale.
#
# The run without the fixup is compared against the run *with* it and not
# against that bound: the leak is what is being measured, and inflating the
# yardstick would measure the yardstick.
function test_tracked_conservation(r, control, uniform, expected, D)
    T = Float64
    mass, energy, mom = 1, D + 2, 2
    roundoff(v) = 8 * eps(T) * r.scales[v] * r.nsteps
    bound(v) = max(roundoff(v), 10 * uniform.drift[v])
    Δmom(x) = abs(x.drift[mom] - expected)
    mom_roundoff = 8 * eps(T) * expected * r.nsteps

    @test r.floor_hits == control.floor_hits == 0
    @test r.nsteps == control.nsteps
    # Recorded rather than asserted as an equality in the brief's sense: the
    # leak could in principle move the criterion and give the control a
    # different mesh history. Measured, it does not — the two runs regrid
    # identically in both dimensions.
    @test r.nregrids == control.nregrids
    @test r.nblocks_history == control.nblocks_history

    @info "Sod tracked, D = $D: with the fixup, mass $(r.drift[mass]), energy " *
          "$(r.drift[energy]), momentum $(r.drift[mom]) against the boundary " *
          "flux $expected (uniform mesh at the same coarse spacing: mass " *
          "$(uniform.drift[mass]), energy $(uniform.drift[energy])); without " *
          "it, mass $(control.drift[mass]), energy $(control.drift[energy]), " *
          "momentum $(control.drift[mom]) — over $(r.nsteps) steps and " *
          "$(r.nregrids) mesh changes, the same in both runs"

    @test r.drift[mass] ≤ bound(mass)
    @test r.drift[energy] ≤ bound(energy)
    @test Δmom(r) ≤ max(mom_roundoff, 10 * Δmom(uniform))
    @test r.drift[mom] ≈ expected rtol = 1e-9

    @test control.drift[mass] > 1e3 * max(r.drift[mass], roundoff(mass))
    @test control.drift[energy] > 1e3 * max(r.drift[energy], roundoff(energy))
    @test Δmom(control) > 1e3 * max(Δmom(r), mom_roundoff)
    # The transverse momentum is zero for all time and *exactly* so: both
    # transverse faces of every cell see identical states, so their flux
    # difference is an exact zero — with the fixup and without it.
    if D == 2
        @test r.drift[3] == 0
        @test control.drift[3] == 0
    end
    return nothing
end

@testset "Conservation survives the regrids, and fails without the fixup: D=$D" for (D, cfg, r, control, uniform) in
                                                                                    ((1, SOD1D,
                                                                                      TRACKED_1D,
                                                                                      NOFIX_1D,
                                                                                      COARSE_1D),
                                                                                     (2, SOD2D,
                                                                                      TRACKED_2D,
                                                                                      NOFIX_2D,
                                                                                      COARSE_2D))
    # Step 5 made this claim on a mesh that was fixed before the run; here
    # the mesh is rebuilt under the solution nine times in D = 1 and three
    # in D = 2, and every new fine block is prolongated from its parent
    # while every coarsened one is restricted from its children. A transfer
    # that was not conservative, or a coarse-fine face whose fluxes were not
    # made to agree, shows up here as a drift that grows with the regrid
    # count.
    #
    # Measured in D = 1 (t_end = 1/5, 588 steps, 9 regrids):
    #   mass 3.3e-16, energy 1.6e-15, momentum 0.18 to 8.6e-16 absolute;
    #   without the fixup 1.385e-6, 1.793e-6 and 6.350e-6 — 4.2e9, 1.2e9
    #   and 7.4e9 times worse, on the same mesh history and the same step
    #   count. In D = 2 (t_end = 3/20, 431 steps, 3 regrids): 4.2e-17,
    #   1.9e-16 and 1.0e-16 against 4.31e-8, 1.51e-7 and 5.12e-8 — 1.0e9,
    #   7.8e8 and 4.9e8 times worse.
    test_tracked_conservation(r, control, uniform, sod_boundary_flux(Val(D), cfg), D)

    # The control that gives the claim its meaning, as step 3's did: on a
    # single-level mesh there is no coarse-fine face for the fixup to act
    # on, so the run with it and the run without it are **bit-identical** —
    # the leak above is the face's and not the driver's.
    if D == 1
        @test COARSE_1D.drift == COARSE_NOFIX_1D.drift
        @test COARSE_1D.l1 == COARSE_NOFIX_1D.l1
        @test COARSE_1D.u == COARSE_NOFIX_1D.u
    end
end

@testset "The derived buffer width is what tracks the waves: D=1" begin
    # The failure mode is the quiet one: a refined region that the feature
    # leaves between one regrid and the next still produces a plausible
    # profile, only a slightly worse one, so nothing short of the tracking
    # measure reports it. The derived width is
    # `refinement_buffer(forest, cap, speed_headroom · λ · chunk)`, which
    # here is 6 cells on the first regrid and 7 on every later one.
    #
    # Measured (L1 against the exact solution, and the tracking measure):
    #   buffer  derived (6–7)  4.540016e-3   1.0000   200 cells   9 regrids
    #   buffer  2              4.540964e-3   1.0000   176 cells  14 regrids
    #   buffer  1              4.546217e-3   0.9444   168 cells  13 regrids
    #   buffer  0              4.552096e-3   0.9048   168 cells  12 regrids
    #
    # Which upstream finding this reproduces: **neither**. TreeAMR measured
    # a margin narrower than the motion coming out *slightly worse* than no
    # margin at all, and TreeWave measured it coming out no worse. Here the
    # three widths are strictly ordered — every cell of margin buys both
    # tracking and accuracy, monotonically — because the feature is a shock
    # that fires at every resolution and a partially covered shock is
    # partially resolved, where TreeAMR's and TreeWave's narrow margins were
    # measured on features their criteria could resolve away.
    @test unique(TRACKED_1D.buffer_history) == [6, 7]
    @test TRACKED_1D.tracking == 1.0

    @info "Sod tracked, D = 1, buffer = derived " *
          "$(unique(TRACKED_1D.buffer_history)): L1 $(TRACKED_1D.l1), tracking " *
          "$(TRACKED_1D.tracking), $(TRACKED_1D.cells) cells, " *
          "$(TRACKED_1D.nregrids) mesh changes"
    for (b, r) in BUFFER_1D
        @info "Sod tracked, D = 1, buffer = $b: L1 $(r.l1), tracking " *
              "$(r.tracking), $(r.cells) cells, $(r.nregrids) mesh changes"
        @test r.floor_hits == 0
        @test r.nsteps == TRACKED_1D.nsteps
        # Every narrower margin is at least as bad in both measures.
        @test r.l1 ≥ TRACKED_1D.l1
        @test r.tracking ≤ TRACKED_1D.tracking
    end
    narrow = last(first(BUFFER_1D))              # buffer = 2
    tight = last(BUFFER_1D[2])                   # buffer = 1
    none = last(last(BUFFER_1D))                 # buffer = 0
    # Two cells still tracks and costs 24 cells of mesh; one does not, and
    # none is worse again. A zero margin is measurably worse in the error
    # *and* loses tracking, which is the claim.
    @test narrow.tracking == 1.0
    @test tight.tracking < 1.0
    @test none.tracking < tight.tracking
    @test none.l1 > tight.l1 > narrow.l1
    # And the derived width's error is inside the bound the tracked claim
    # is made at.
    @test TRACKED_1D.l1 ≤ 1.3 * FINE_1D.l1

    # The guard itself — a cadence too slow for the cap refused by
    # `refinement_buffer` naming the constraint, rather than silently
    # producing a margin that cannot be recruited — costs no evolution at
    # all and is asserted in the short tier, in `test/driver_tests.jl`.
end

@testset "p = 1 against p = 3 on a discontinuous solution: D=1" begin
    # The first *discontinuous* entry of the open question in "Operator
    # order". Step 5 measured the smooth case and settled nothing: on the
    # entropy wave `p = 1` costs a full order in L∞ and nothing at all in
    # L1, which confirms the premise (an integral norm does not see the
    # interface defect) rather than answering the question. The question is
    # what happens in L1 on a solution with a shock in it, where the
    # scheme's own rate is 0.9 and positivity is a live concern.
    #
    # Measured (tracked tube, t_end = 1/5):
    #   p = 1   L1 4.701364e-3   tracking 0.9091   216 cells   0 floor hits
    #   p = 3   L1 4.540016e-3   tracking 1.0000   200 cells   0 floor hits
    #   p = 5   L1 4.539988e-3   tracking 1.0000   200 cells   0 floor hits
    #
    # Asserted here: only that both run and conserve. The question stays
    # open until Sedov (step 9) completes the table — Sod is the case where
    # no floor fires at all, so it cannot speak to the positivity half of
    # the argument, which is the half `p = 1` was proposed for.
    @info "Sod tracked, D = 1, p = 3: L1 $(TRACKED_1D.l1), tracking " *
          "$(TRACKED_1D.tracking), $(TRACKED_1D.cells) cells, " *
          "$(TRACKED_1D.floor_hits) floor hits"
    for (p, r) in ORDER_1D
        @info "Sod tracked, D = 1, p = $p: L1 $(r.l1), tracking $(r.tracking), " *
              "$(r.cells) cells, $(r.floor_hits) floor hits"
        @test r.floor_hits == 0
        @test r.converged
        @test r.drift[1] ≤ max(8 * eps(Float64) * r.scales[1] * r.nsteps,
                               10 * COARSE_1D.drift[1])
        @test r.drift[3] ≤ max(8 * eps(Float64) * r.scales[3] * r.nsteps,
                               10 * COARSE_1D.drift[3])
        @test r.drift[2] ≈ sod_boundary_flux(Val(1), SOD1D) rtol = 1e-9
    end
    p1 = last(first(ORDER_1D))
    p5 = last(last(ORDER_1D))
    # Recorded, not decided: on this case the lower order is the worse one
    # in every column, and it buys nothing in the floor count because there
    # is no floor count to buy.
    @test p1.l1 > TRACKED_1D.l1
    @test p1.tracking < 1.0
    @test p5.l1 ≈ TRACKED_1D.l1 rtol = 1e-3
end

@testset "The CFL recheck fires when the signal outgrows the step it was sized for" begin
    # The second of the two halves of the step-4 amendment, and the one that
    # needs a run. (The first is arithmetic — the recheck is a pure function
    # of five numbers — and is asserted in the short tier, in
    # `test/driver_tests.jl`.) This is the reason the headroom is a case
    # parameter at all: a Riemann problem's fastest signal is not in its
    # initial data, so a driver that measured λ once per chunk and believed
    # it would run the first chunk of *every* shock case at nearly twice the
    # CFL number it asked for.

    # The measurement that sizes the parameter. Sod's initial data
    # carries nothing above c_L = 1.18322 and its post-shock gas carries
    # 2.19157, a factor of 1.8522; with `speed_headroom = 1` the very first
    # chunk ends at λ_end = 1.9486 against a step sized for 1.25, a CFL
    # number of 0.6236 against the requested 0.4.
    @test_throws "in chunk 1" evolve!(sod_case(Val(1), SOD1D.roots;
                                               speed_headroom=1), Val(1);
                                      N=SOD1D.N, ops=driver_ops(), t_end=1 // 200,
                                      chunk=1 // 200, limiter=:minmod,
                                      refine_tol=DRIVER_REFINE_TOL,
                                      coarsen_tol=DRIVER_COARSEN_TOL,
                                      maxlevel_cap=SOD1D.cap)
    # The same run at the case's own headroom of 2 completes, and its
    # end-of-chunk speeds stay inside it: measured λ from 1.1832 to 2.2047
    # over the run, never above 2 × the λ the chunk was sized from.
    @test all(zip(TRACKED_1D.λ_history, TRACKED_1D.λ_end_history)) do (λ, λ_end)
        λ_end ≤ 2 * λ
    end
    @test TRACKED_1D.λ_initial ≈ 1.1832159566199232
    @test maximum(TRACKED_1D.λ_end_history) < 2.21
end

@testset "The entropy wave runs through the driver on the periodic, no-boundary path" begin
    # A smoke test of the two paths Sod cannot reach — a case with no
    # boundary hook at all, and `uniform_run`, which is the driver with the
    # cap at zero — and a check that the chunked loop reproduces the
    # convergence study's own number.
    #
    # It is *not* bit for bit, and two things separate them. The driver
    # takes chunked steps, so it uses a slightly different step count (50
    # against 47 at N = 8); and a case states its initial data as a pure
    # `x -> P`, which is a **point sample**, where `entropywave_errors`
    # fills the exact cell average — a relative difference of `(kh)²/24` in
    # the amplitude, which is `O(h²)` and therefore the same order as the
    # error being compared.
    #
    # Measured, driver against `entropywave_errors` at the same N:
    #   D = 1, N =  8   5.219843e-4 against 5.538618e-4   ratio 0.9424
    #   D = 1, N = 16   1.343281e-4 against 1.350740e-4   ratio 0.9945
    #   D = 2, N =  8   1.249213e-3 against 1.325157e-3   ratio 0.9427
    # The gap closes like `h²`, as the explanation above says it must.
    for (D, N, tol) in ((1, 8, 0.10), (1, 16, 0.03), (2, 8, 0.10))
        case = HydroCase(EntropyWave(Float64, Val(D)); roots=4)
        @test case.boundary === nothing
        @test all(case.periodic)
        @test case.speed_headroom == 1
        r = uniform_run(case, Val(D); N=N, ops=driver_ops(), t_end=1 // 4,
                        chunk=1 // 20, limiter=:none)
        e = entropywave_errors(Val(D); N=N, ops=driver_ops(), t_end=1 // 4)
        @test r.levels == [0]
        @test r.nregrids == 0
        @test r.floor_hits == 0
        @test r.l1 ≈ e.l1 rtol = tol
        @test r.linf ≈ e.linf rtol = tol
        @info "entropy wave through the driver, D = $D, N = $N: L1 $(r.l1) in " *
              "$(r.nsteps) chunked steps against $(e.l1) in $(e.nsteps) of " *
              "entropywave_errors' (ratio $(round(r.l1 / e.l1; digits=4)))"
        # Periodic everywhere, so this one really does conserve to roundoff.
        for v in 1:(D + 2)
            @test r.drift[v] ≤ 8 * eps(Float64) * r.scales[v] * r.nsteps
        end
    end
end
