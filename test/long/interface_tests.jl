# **Long tier.** This file runs only under `TREEHYDRO_TEST_LONG=1`; see
# "Testing: two tiers" in `CODE.md`. It is where the physics is claimed and
# where the numbers `CODE.md` records come from, and it may take minutes.
# The short tier pins reduced configurations of the same studies against
# committed references in `test/regression_tests.jl`.
#
# Coarse-fine faces: the claim the package exists to make.
#
# Everything measured so far lives on a single-level mesh, where every face
# is a same-level face and `restrict_interfaces!` has nothing to do — the
# entropy wave's two runs, with the fixup and without it, are bit-identical
# there. This file puts a coarse-fine face in the way and measures what
# changes, for a *system* of `D + 2` conservation laws rather than for
# Burgers' one. Four claims, in the order TreeAMR's M8b states them:
#
#   * **The fixup is what conserves.** On the static two-level mesh, with
#     the fixup every one of the `D + 2` integrals holds to roundoff and
#     without it every one of them leaks by ten orders of magnitude. The
#     two runs differ in one line of `hydro_rhs!` and in nothing else —
#     same mesh, same step count, same everything — which is what makes
#     this a measurement of that line.
#   * **The interface-order rule carries over to the system.** A flux
#     divergence takes one derivative, so the prolongation order must
#     exceed the scheme's by one: rates 1, 2, 2 in L∞ for `p = 1, 3, 5`,
#     and 2, 2, 2 in L1, with `p = 3` landing on the unrefined control's
#     own rate. Predicted in `CODE.md` before it was run.
#   * **Sod conserves across a coarse-fine face the shock crosses.** With
#     a physical boundary the integrals are *not* constant, so the claim is
#     the one step 4 set up: the momentum total moves by the closed-form
#     boundary flux and the mass and energy totals do not move, and the
#     difference between two runs sharing that same boundary flux is the
#     leak.
#   * **A refined region may touch the Dirichlet face.** The hook fills the
#     outer ghosts of *fine* blocks too, and a coarse-fine face beside a
#     physical one conserves.
#
# The one cheap claim these four rest on — that the two-level Sod forests
# really are two-level, asserted from the block extents rather than from a
# run — stays in the short tier, in `test/interface_tests.jl`.
#
# The norm is part of the result, and both are asserted in both directions:
# the interface defect an order-`p` prolongation leaves sits on the
# coarse-fine face and nowhere else, so a volume-weighted L1 norm — which
# multiplies it by the shrinking measure of the region it occupies —
# converges at the scheme's own rate for every `p`, and only L∞ exposes the
# rule. TreeAMR measured the same thing on Burgers and records *why* under
# "Operators" in its `CODE.md`: the defect stays local because the fixup
# makes it zero-mean, and the negative control on the *rate* below is what
# pins that on conservation rather than on the flux-divergence form.

# The conservative family at the interface order the scheme wants. Every
# run in this file that is not about the order itself uses it, so the
# refined runs differ from `entropywave_tests.jl`'s and `sod_tests.jl`'s in
# the mesh and in nothing else.
interface_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

@testset "The fixup is what conserves at a coarse-fine face: D=$D" for D in (1, 2, 3)
    # The claim in its cheapest and most direct form, and the only one that
    # runs in 3D: the M3 static two-level mesh, the smooth entropy wave, no
    # limiter and no regridding, so the interface flux restriction is the
    # *single* difference between the two runs. In 3D a coarse-fine face
    # carries four fine faces, so that is also where the fixup's tangential
    # average is a 2×2 rather than a single cell.
    #
    # `D + 2` integrals rather than Burgers' one, and the momentum
    # components are the new case: each is measured against its own scale
    # `Σ hᴰ |U_v|`, because a component whose total is zero by symmetry
    # would otherwise be compared against nothing at all.
    T = Float64
    N, t_end = D == 3 ? (4, 1 // 8) : (8, 1 // 4)
    roots = 4
    common = (; N=N, ops=interface_ops(), roots=roots, refined=true,
              limiter=:none, t_end=t_end)
    r = entropywave_errors(T, Val(D); common...)
    control = entropywave_errors(T, Val(D); common..., fixup=false)

    # The mesh really is the two-level one: more blocks than the uniform
    # box has, and both levels present. Without this the drift assertions
    # below would pass on a mesh with no coarse-fine face in it, which is
    # exactly how a conservation test passes for the wrong reason.
    @test r.nblocks > roots^D
    @test r.levels == [0, 1]
    @test isfinite(r.l1) && isfinite(r.linf)
    # The two runs differ in one line, so everything else about them is the
    # same and the drift is the only thing left to attribute.
    @test (r.nsteps, r.nblocks) == (control.nsteps, control.nblocks)
    @test r.floor_hits == control.floor_hits == 0

    bounds = ntuple(v -> 8 * eps(T) * r.scales[v] * r.nsteps, Val(D + 2))
    for v in 1:(D + 2)
        # `c · eps(T) · Σ hᴰ|U_v| · nsteps`, `CODE.md`'s form, measured far
        # inside it — the drift is a fraction of one ulp of the scale
        # whatever the step count.
        @test r.drift[v] ≤ bounds[v]
        # The negative control leaks, and by a margin that is not a matter
        # of taste: at least a millionfold above the bound the fixup run
        # meets, and at least `1e-8` of the variable's own scale so that a
        # tiny scale cannot make a tiny leak look like a large ratio.
        # Measured separations are 1e8 … 1e9 and 1e-5 … 1e-4 of the scale.
        @test control.drift[v] > 1e6 * bounds[v]
        @test control.drift[v] > 1e-8 * control.scales[v]
    end
    @info "entropy wave, D = $D, two-level ($(r.nblocks) blocks, " *
          "$(r.nsteps) steps): worst drift with the fixup " *
          "$(round(maximum(r.drift[v] / (eps(T) * r.scales[v] * r.nsteps)
                           for v in 1:(D + 2)), sigdigits=3)) ulp of the scale " *
          "per step; without it $(round(maximum(control.drift[v] /
                                                control.scales[v]
                                                for v in 1:(D + 2)),
                                        sigdigits=3)) of the scale"
end

@testset "The interface-order rule holds for a system: D=$D" for D in (1, 2)
    # `CODE.md`'s prediction for the system, measured. A flux divergence
    # divides an `O(hᵖ)` ghost error by `h` once, so the interface caps the
    # global rate at `p`; against a second-order scheme that is 1 at
    # `p = 1` and 2 from `p = 3` on, and raising `p` further buys nothing.
    # The restriction order never enters — conservative restriction is the
    # exact volume average, exact for any field — so only the prolongation
    # is varied.
    #
    # `:none` is the limiter, as in the order study on the uniform mesh: a
    # TVD limiter clips at the sine's smooth extrema and would put its own
    # first-order footprint on top of the interface's.
    #
    # Both norms, in both directions. L∞ is where the rule shows; L1 is
    # second order at every `p` including 1, because the defect sits on the
    # coarse-fine face and the volume weight shrinks with it. Picking one
    # norm and believing it is the easy mistake, which is why the L1 column
    # is asserted too — and why the negative control at the end matters.
    T = Float64
    Ns = D == 1 ? (8, 16, 32, 64) : (8, 16, 32)

    function rates(p; refined=true, fixup=true)
        hs, l1s, linfs = Float64[], Float64[], Float64[]
        for N in Ns
            r = entropywave_errors(T, Val(D); N=N, ops=interface_ops(p), roots=4,
                                   refined=refined, limiter=:none, fixup=fixup)
            push!(hs, r.h); push!(l1s, r.l1); push!(linfs, r.linf)
            @test isfinite(r.l1) && isfinite(r.linf)
            @test r.levels == (refined ? [0, 1] : [0])
            @test r.floor_hits == 0
        end
        # A rate fitted through a non-monotone sequence can be anything.
        @test issorted(linfs; rev=true)
        return (linf=convergence_rate(hs, linfs), l1=convergence_rate(hs, l1s))
    end

    # The unrefined control: with no coarse-fine face the scheme is second
    # order, so whatever the refined runs lose is the interface's doing and
    # not the scheme's.
    control = rates(3; refined=false)
    @test control.linf > 1.65

    refined = Dict(p => rates(p) for p in (1, 3, 5))
    for p in (1, 3, 5)
        @info "entropy wave, D = $D, two-level, p = $p, N = $Ns: L∞ rate " *
              "$(round(refined[p].linf, digits=3)), L1 rate " *
              "$(round(refined[p].l1, digits=3))"
    end
    @info "entropy wave, D = $D, unrefined control, p = 3: L∞ rate " *
          "$(round(control.linf, digits=3)), L1 rate " *
          "$(round(control.l1, digits=3))"

    # Order 1 costs a full order in L∞, exactly as the rule says.
    @test refined[1].linf ≈ 1.0 atol = 0.25
    for p in (3, 5)
        # Not "second order" in the abstract: the *same* rate as the
        # unrefined control, which is the claim that the interface has
        # stopped being what limits it.
        @test refined[p].linf ≈ control.linf atol = 0.15
        @test refined[p].linf > 1.65
    end
    # And the L1 rate never sees any of this, `p = 1` included.
    for p in (1, 3, 5)
        @test refined[p].l1 > 1.65
    end

    # The negative control on the *rate*, and it is a claim about
    # conservation and not about the shape of the operator. With the fixup
    # the interface defect is a dipole — the fine cell loses exactly what
    # the coarse cell gains — and a first-order hyperbolic operator carries
    # a zero-mean residual nowhere. Without it the residual has net mass
    # `O(h)` per unit time, and the equation transports that downstream as
    # an `O(h)` plateau, which a volume-weighted norm does see. Burgers
    # measured the L1 rate falling from 1.98 to 1.12 in `D = 1` and from
    # 1.80 to 1.25 in `D = 2`; the system reproduces it (1.97 → 1.11 and
    # 1.90 → 1.19, recorded in "Measured results" in `CODE.md`), while L∞
    # never depended on the fixup at all.
    leaky = rates(1; fixup=false)
    @info "entropy wave, D = $D, two-level, p = 1, fixup = false: L∞ rate " *
          "$(round(leaky.linf, digits=3)), L1 rate $(round(leaky.l1, digits=3)) " *
          "against $(round(refined[1].l1, digits=3)) with the fixup"
    @test leaky.l1 < 1.5
    @test leaky.l1 < refined[1].l1 - 0.4
    @test leaky.linf ≈ 1.0 atol = 0.25
end

# Sod's conserved-integral claim, which is not the entropy wave's: a
# Dirichlet face lets mass, momentum and energy out of the box, so the
# totals *must* move and the question is by how much and in which row.
#
# Until a wave arrives both sides of each boundary face are in that face's
# own initial state — at rest — so the only nonzero component of the Euler
# flux there is the pressure, and
#
#     ΔS = (p_L − p_R) · t_end · A,     Δρ = ΔE = 0
#
# exactly. What the discrete run adds to that is the *numerical* foot of
# the rarefaction and of the shock reaching the boundary, an `O(hᵏ)`
# discretization effect that is 2.1e-11 of the mass at `N = 16` in `D = 1`
# and roundoff by `N = 32` (step 4 measured it on the uniform mesh). It is
# a property of the boundary and not of the coarse-fine face, so the bound
# on the run *with* the fixup is the larger of roundoff and ten times what
# the **uniform mesh at the same coarse spacing** drifts by: where the
# boundary's own flux has reached roundoff the roundoff term binds and the
# claim is the entropy wave's plain one, and where it has not the claim is
# that a coarse-fine face added nothing to it. The measured ratios of the
# refined drift to the uniform one are 0.15 … 2.8.
#
# The run *without* the fixup is then compared against the run with it and
# not against that bound: the leak is what is being measured, and inflating
# the yardstick by the factor of ten above would measure the yardstick.
function test_boundary_conservation(r, control, uniform, expected, ::Type{T},
                                    D) where {T}
    mass, energy, mom = 1, D + 2, 2
    roundoff(v) = 8 * eps(T) * r.scales[v] * r.nsteps
    bound(v) = max(roundoff(v), 10 * uniform.drift[v])
    # The momentum has no scale of its own — `Σ hᴰ |S|` is zero, since the
    # initial state is at rest — so its yardstick is the boundary flux it
    # is claimed to equal.
    Δmom(x) = abs(x.drift[mom] - expected)
    mom_roundoff = 8 * eps(T) * expected * r.nsteps
    @test r.floor_hits == control.floor_hits == 0
    @test (r.nsteps, r.nblocks) == (control.nsteps, control.nblocks)

    # With the fixup: the momentum total moves by the boundary flux, and
    # mass and energy move by no more than the boundary's own numerics.
    @test r.drift[mass] ≤ bound(mass)
    @test r.drift[energy] ≤ bound(energy)
    @test Δmom(r) ≤ max(mom_roundoff, 10 * Δmom(uniform))
    @test r.drift[mom] ≈ expected rtol = 1e-5
    # Without it: every one of the three departs by at least a thousandfold
    # — measured 6.3e3 … 7.7e7 — and the momentum departs from a number
    # with a closed form, which is the sharpest of the three.
    @test control.drift[mass] > 1e3 * max(r.drift[mass], roundoff(mass))
    @test control.drift[energy] > 1e3 * max(r.drift[energy], roundoff(energy))
    @test control.drift[mass] > 1e-5 * control.scales[mass]
    @test control.drift[energy] > 1e-5 * control.scales[energy]
    @test Δmom(control) > 1e3 * max(Δmom(r), mom_roundoff)
    @test Δmom(control) > 1e-4 * expected
    # The transverse momentum is zero for all time and *exactly* so: both
    # transverse faces of every cell see identical states, so their flux
    # difference is an exact zero — with the fixup and without it.
    if D == 2
        @test r.drift[3] == 0
        @test control.drift[3] == 0
    end
    return nothing
end

# The conserved states the two Dirichlet faces hold for all time, and the
# number of stored entries in the outward-facing ghost regions along the
# tube that do not hold them. Zero, or the hook did not reach a fine block.
function outer_ghost_mismatches(U::FieldSet{T,D}, w::SodTube{T,D,DIR}) where {T,D,DIR}
    zeros_ = ntuple(_ -> zero(T), Val(D))
    U_L = prim2con(w.eos, (w.ρ_L, zeros_..., w.p_L))
    U_R = prim2con(w.eos, (w.ρ_R, zeros_..., w.p_R))
    forest = U.forest
    lo, hi = forest.extents[DIR]
    cons = Array(U.work)
    N, G = forest.N, U.G[DIR]
    bad = 0
    for b in 1:nblocks(U)
        ext = block_extent(forest, blockkey(U, b))[DIR]
        # The whole stored slab of the block, ghost rows across the tube
        # included: TreeAMR fills the edge and corner regions of a
        # Dirichlet face unconditionally, and a hook that skipped them
        # would still leave a plausible profile.
        for idx in CartesianIndices(size(cons)[1:D]), v in 1:(D + 2)
            i = Tuple(idx)[DIR]
            if ext[1] == lo && i ≤ G
                cons[Tuple(idx)..., v, b] == U_L[v] || (bad += 1)
            elseif ext[2] == hi && i > G + N
                cons[Tuple(idx)..., v, b] == U_R[v] || (bad += 1)
            end
        end
    end
    return bad
end
@testset "Sod conserves across a coarse-fine face the shock crosses: D=$D" for
        D in (1, 2)
    # The failure this guards against is the one the entropy wave cannot
    # produce: a *shock* passing through a coarse-fine face. The refined
    # region is the middle half of the tube, so the diaphragm starts inside
    # it and the shock — travelling at 1.7522 — leaves it at `t = 0.1427`,
    # well before `t_end = 0.2`. A refined region the solution never left
    # would make every assertion below pass for the wrong reason, so the
    # crossing is asserted from the exact solution rather than assumed.
    #
    # Two uniform runs stand beside the refined one and are not decoration:
    # the run at the same *coarse* spacing says how much of the drift is
    # the boundary's own (see `test_boundary_conservation`), and the run at
    # the same *finest* spacing is what the L1 error is judged against.
    T = Float64
    N, roots, area = 16, (D == 1 ? (4,) : (4, 1)), (D == 1 ? 1.0 : 0.25)
    t_end = 1 // 5
    w = SodTube(T, Val(D))
    common = (; ops=interface_ops(), roots=roots, limiter=:minmod, t_end=t_end)
    r = sod_errors(T, Val(D); N=N, refined=:middle, common...)
    control = sod_errors(T, Val(D); N=N, refined=:middle, fixup=false, common...)
    uniform = sod_errors(T, Val(D); N=N, common...)
    fine = sod_errors(T, Val(D); N=2N, common...)

    # The mesh is two-level, and it is the middle half of the tube that is
    # refined.
    @test r.levels == [0, 1]
    @test r.nblocks > uniform.nblocks
    @test r.h ≈ fine.h
    @test uniform.levels == [0]

    # The shock did cross the coarse-fine face at `3L/4`, from the exact
    # solution: it starts at the diaphragm, inside the refined box, and
    # ends outside it.
    sol = exact_riemann(w)
    x_shock = w.x₀ + sol.head_R * T(t_end)
    @test w.x₀ < 3 * w.L / 4 < x_shock
    @info "Sod, D = $D, refined = :middle: the shock crosses x = " *
          "$(3 * w.L / 4) at t = $(round((3 * w.L / 4 - w.x₀) / sol.head_R,
                                         digits=4)) and ends at " *
          "$(round(x_shock, digits=4)); $(r.nblocks) blocks, $(r.nsteps) steps"

    expected = (w.p_L - w.p_R) * T(t_end) * area
    test_boundary_conservation(r, control, uniform, expected, T, D)
    @info "Sod, D = $D, :middle: with the fixup, mass $(r.drift[1]), energy " *
          "$(r.drift[D + 2]), momentum $(r.drift[2]) against the boundary flux " *
          "$expected (uniform mesh at the same coarse spacing: mass " *
          "$(uniform.drift[1]), energy $(uniform.drift[D + 2])); without it, " *
          "mass $(control.drift[1]), energy $(control.drift[D + 2]), momentum " *
          "$(control.drift[2])"

    # A sanity bound on the error and not the tracked-shock claim of step 7:
    # the two-level run resolves the shock's neighbourhood at the fine
    # spacing and the rest of the tube at the coarse one, so it should sit
    # between the two uniform runs and near the fine one. Measured 1.34
    # times the uniform fine error in `D = 1` and 1.18 in `D = 2`.
    @test r.l1 < uniform.l1
    @test r.l1 < 1.5 * fine.l1
    @info "Sod, D = $D, :middle: L1 $(r.l1) against $(fine.l1) uniform at the " *
          "same finest spacing (ratio $(round(r.l1 / fine.l1, digits=3))) and " *
          "$(uniform.l1) uniform at the coarse one"
end

@testset "The two-level Sod drift is roundoff where the boundary's own is" begin
    # The sharp form of the claim above, and the one `CODE.md`'s step-4 note
    # promised. The residual mass and energy drift in the runs above is the
    # numerical foot of the rarefaction and the shock reaching the Dirichlet
    # faces, which falls by four orders of magnitude per halving of `h`; by
    # `N = 32` in `D = 1` it is gone, and then all three integrals obey the
    # entropy wave's plain bound — `8 eps · scale · nsteps` — on a mesh with
    # a coarse-fine face in it, with the momentum measured against the
    # closed-form boundary flux rather than against a scale that is zero
    # because the initial state is at rest.
    #
    # `D = 2` needs `N = 32` for the same statement and costs eight times as
    # much to make it; the number is recorded in `CODE.md` instead.
    T = Float64
    t_end = 1 // 5
    w = SodTube(T, Val(1))
    expected = (w.p_L - w.p_R) * T(t_end)
    for refined in (:middle, :left)
        r = sod_errors(T, Val(1); N=32, ops=interface_ops(), roots=(4,),
                       refined=refined, limiter=:minmod, t_end=t_end)
        @test r.levels == [0, 1]
        @test r.floor_hits == 0
        @test r.drift[1] ≤ 8 * eps(T) * r.scales[1] * r.nsteps
        @test r.drift[3] ≤ 8 * eps(T) * r.scales[3] * r.nsteps
        @test abs(r.drift[2] - expected) ≤ 8 * eps(T) * expected * r.nsteps
        @info "Sod, D = 1, N = 32, refined = $refined: mass $(r.drift[1]), " *
              "energy $(r.drift[3]), |ΔS − flux| $(abs(r.drift[2] - expected)) " *
              "against bounds $(8 * eps(T) * r.scales[1] * r.nsteps), " *
              "$(8 * eps(T) * r.scales[3] * r.nsteps), " *
              "$(8 * eps(T) * expected * r.nsteps)"
    end
end

@testset "A refined region may touch the Dirichlet face: D=$D" for D in (1, 2)
    # The configuration "Boundaries" in `CODE.md` asks for, and the one
    # thing step 4 could not exercise: the physical-boundary hook filling
    # the outer ghosts of a *fine* block, with a coarse-fine face at the
    # diaphragm and the boundary state on one side of the refined region.
    # The hook runs between the copy/restriction phase and the prolongation
    # sweep, and a mesh whose refined region touches a Dirichlet face is
    # where an ordering mistake there would show.
    #
    # **What this does not exercise**, and it is worth being explicit: the
    # M2 ordering case in full is a prolongation reaching *tangentially*
    # into hook-filled ghosts, which needs two physical faces meeting at an
    # edge or a corner. The tube is periodic across itself, so its only
    # physical faces are the two ends and they never meet. That case is
    # Sedov's, in step 9, where every face is Dirichlet. What is exercised
    # here is the hook on a fine block — including the ghost rows *across*
    # the tube, which TreeAMR fills unconditionally — and a coarse-fine
    # face at `x₀` with the boundary state on one side of it.
    T = Float64
    N, roots, area = 16, (D == 1 ? (4,) : (4, 1)), (D == 1 ? 1.0 : 0.25)
    t_end = 1 // 5
    w = SodTube(T, Val(D))
    common = (; ops=interface_ops(), roots=roots, limiter=:minmod, t_end=t_end)
    r = sod_errors(T, Val(D); N=N, refined=:left, common...)
    control = sod_errors(T, Val(D); N=N, refined=:left, fixup=false, common...)
    uniform = sod_errors(T, Val(D); N=N, common...)

    @test r.levels == [0, 1]
    @test r.nblocks > uniform.nblocks
    # The block against the low boundary really is a fine one — otherwise
    # the ghost claim below would be step 4's claim again.
    U = r.U
    lo = U.forest.extents[1][1]
    lowblocks = [b for b in 1:nblocks(U)
                 if block_extent(U.forest, blockkey(U, b))[1][1] == lo]
    @test !isempty(lowblocks)
    @test all(b -> level(blockkey(U, b)) == 1, lowblocks)

    # The claim is made *after* the run, as step 4's was: at `t = 0` every
    # ghost holds the initial state whether the hook ran or not, because the
    # interior does too.
    @test outer_ghost_mismatches(U, w) == 0

    expected = (w.p_L - w.p_R) * T(t_end) * area
    test_boundary_conservation(r, control, uniform, expected, T, D)
    @info "Sod, D = $D, refined = :left ($(r.nblocks) blocks, $(r.nsteps) " *
          "steps, $(length(lowblocks)) fine block(s) on the Dirichlet face): " *
          "with the fixup, mass $(r.drift[1]), energy $(r.drift[D + 2]), " *
          "momentum $(r.drift[2]) against $expected; without it, mass " *
          "$(control.drift[1]), energy $(control.drift[D + 2]), momentum " *
          "$(control.drift[2])"
end
