# The Kelvin–Helmholtz shear layer: the contact-dominated case, the one with no
# exact solution, and the one that decides between HLLE and HLLC.
#
# Everything before this file is a shock case or a smooth advected wave. Here
# the feature is a *contact* — the density jumps across the layer and the
# pressure does not — it **grows** rather than travels, and the whole domain is
# in motion, so a coarse-fine face anywhere in the box carries real flux. That
# last point is what lets the interface fixup be measured on the tracked mesh
# itself, which step 9 found it could not be on the blast.
#
# The failure modes this file guards, one per testset:
#
#   * a setup that is not McNally, Lyra & Passy's — a sign in `ρ_m`, a branch
#     boundary, a missing mirror in the weighting — which no plot would show
#     and which would make every number below incomparable with the paper's;
#   * an instability that does not grow exponentially, or that grows faster
#     than the incompressible bounds, which would mean the seeded mode is not
#     what is being measured;
#   * a refined region that is not the two strips, or one that does not follow
#     the rolls as they thicken;
#   * a coarse-fine face that leaks under a smooth flow, where the leak is
#     small in absolute terms and the roundoff claim is correspondingly
#     sharper, with `fixup = false` as the negative control;
#   * an adaptive run that does not approach the uniform fine run as the cap
#     rises, which is the only quantitative reference this case has;
#   * a flux choice made by assertion rather than by measurement.
#
# The numbers recorded in the comments below are in `CODE.md` under "Step 10"
# in "Measured results"; a changed number is a regression, and a test that
# merely still passes is not.

kh_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

# Step 6's calibrated thresholds, quoted rather than reinvented — and this is
# the case they were calibrated *on*: `refinement_tests.jl`'s smooth profile is
# this setup's density ramp in one dimension.
const KH_REFINE_TOL = 2 // 25            # 0.08
const KH_COARSEN_TOL = 1 // 50           # 0.02

# The configuration. `roots = 4`, `N = 8`, `cap = 2` is 128² at the finest
# level on the unit square, which is the resolution McNally's Figure 4 shows a
# converging `M(t)` at, and `t_end = 3/2` is the paper's own end time.
#
# **`chunk = 1/200` is set by the buffer and not by the physics**, and it is
# the tightest binding in this file. The derived margin covers
# `speed_headroom · λ · chunk` at the cap's spacing; with `λ = 2.5412` and
# `h_cap = 1/128` a chunk of `1/200` gives a **3-cell** margin, and the margin
# is what decides how much of the box is refined: measured at `t = 0`, a
# 3-cell margin refines 128 of 256 possible finest blocks and a 7-cell margin
# (`chunk = 1/64`) refines all 256, which is the uniform fine mesh under
# another name. The feature here *grows* rather than travels, so a margin
# sized for travel buys nothing and costs the whole saving.
const KH = (roots=4, N=8, cap=2, chunk=1 // 200, t_end=3 // 2)

# The controls run to `t = 2/5` rather than to `3/2`, and the reason is the
# clock: this file already costs about a minute of arithmetic and the leak, the
# floor counts and the `Float32` agreement are all visible in the first fifth
# of the run — the mesh has its coarse-fine faces from `t = 0`, and the
# instability has not yet left its linear phase at `2/5`.
const KH_SHORT = 2 // 5

# **The case's default flux, decided in step 10 by the measurement in the
# HLLE/HLLC testset below** and recorded in `CODE.md` under "Riemann solver".
# The package-wide default in `HydroProblem` and `evolve!` stays `:hlle`, which
# is *the* GRMHD flux; this is the one case that overrides it, and `kh_run`'s
# own `riemann` default is the same symbol.
const KH_FLUX = :hllc

# The window the growth rate is fitted over, stated as a range of `M` and not
# of `t` so that it means the same thing at every resolution and under either
# flux: **from twice the seeded amplitude to six times it**. `M(0)` is the
# seed's own amplitude `a = 1/100` exactly, the mode first *decays* while the
# ramp sheds the transient (measured: down to 0.00806 at `t = 0.335`), and the
# curve bends over well before `t_end`. Below `2a` the fit would measure the
# transient and above `6a` the saturation.
const KH_WINDOW = (from=2 // 100, to=6 // 100)

"""
The tracked shear layer, with `M(t)` and the maximum `y`-kinetic energy
recorded per chunk through the observer — which [`kh_run`](@ref) does, because
the observer is the only place the state is scattered and `P` current.
"""
tracked_kh(flux=KH_FLUX; cap=KH.cap, t_end=KH.t_end, T=Float64, kwargs...) =
    kh_run(T, Val(2); N=KH.N, ops=kh_ops(), chunk=KH.chunk, maxlevel_cap=cap,
           refine_tol=KH_REFINE_TOL, coarsen_tol=KH_COARSEN_TOL, t_end=t_end,
           roots=KH.roots, riemann=flux, kwargs...)

"""
The same layer on a uniform mesh with the root brick scaled by `scale`:
`2^cap = 4` is the **fine** reference at the tracked run's own finest spacing,
`1` is the coarse control at its coarsest, and `2` is the middle rung of the
resolution sweep the flux comparison is decided on.
"""
uniform_kh(flux=KH_FLUX; scale, t_end=KH.t_end, T=Float64) =
    kh_uniform(T, Val(2); N=KH.N, ops=kh_ops(), chunk=KH.chunk, t_end=t_end,
               roots=KH.roots, scale=scale, riemann=flux)

# The runs, computed once and shared by every testset below, so that the
# tables really are the same runs compared against each other. About a minute
# between them; the two `t_end = 3/2` pairs are half of it.
const TRACKED_KH = tracked_kh()
const FINE_KH = uniform_kh(; scale=2^KH.cap)
const CAP1_KH = tracked_kh(; cap=1)
const COARSE_KH = uniform_kh(; scale=1)
const MID_KH = uniform_kh(; scale=2)

const TRACKED_KH_E = tracked_kh(:hlle)
const FINE_KH_E = uniform_kh(:hlle; scale=2^KH.cap)
const MID_KH_E = uniform_kh(:hlle; scale=2)
const COARSE_KH_E = uniform_kh(:hlle; scale=1)

const SHORT_KH = tracked_kh(; t_end=KH_SHORT)
const SHORT_KH_NOFIX = tracked_kh(; t_end=KH_SHORT, fixup=false)
const SHORT_KH_32 = tracked_kh(; t_end=KH_SHORT, T=Float32)

# The grid the four meshes of the cap sweep share: the coarsest of them is the
# uniform `32²` control, so that is the common ground and `reduce_to_grid`
# refuses anything finer.
const KH_GRID = KH.roots * KH.N

"""
How far the outermost edge of any block at the refinement cap lies from the
nearer of the two interfaces at `y = ¼` and `y = ¾` — the width of the band the
refined region occupies, in physical units.
"""
function kh_refined_band(forest, cap)
    worst = 0.0
    for k in forest.leaves
        level(k) == cap || continue
        e = block_extent(forest, k)
        for y in (e[2][1], e[2][2])
            worst = max(worst, min(abs(y - 0.25), abs(y - 0.75)))
        end
    end
    return worst
end

@testset "The setup is McNally, Lyra and Passy's, term by term" begin
    # The failure mode is silent and total: a sign in `ρ_m`, a branch boundary
    # at the wrong `y`, or a perturbation with the wrong wavenumber gives a
    # shear layer that rolls up and looks right in every plot, and no number
    # below would be comparable with the paper's. So the profile is checked
    # against equations (1)–(5) **evaluated independently here**, with the
    # paper's own literals rather than the case's fields.
    #
    # Checked in step 10 against arXiv:1111.1764: `CODE.md`'s transcription of
    # the profiles, the parameters, the perturbation, `γ = 5/3`, `p = 5/2` and
    # `t = 1.5` is the paper's in every term. The one thing that was wrong is
    # the weighting of `M(t)`, and it is corrected in `CODE.md` and in
    # `mode_amplitude` — see the testset on the growth rate.
    #
    # Measured: the discrete initial totals against the closed-form integrals
    # are mass 1.4999999999999996 against 3/2 (1.3 ulp) and
    # S_x −0.2125079781452482 against −0.2125022699707237 (2.69e-5 relative,
    # which is the midpoint rule's own error on the `e²` term of `ρ v_x`,
    # whose width is L/2); `Σ S_y = −1.4e-19`.
    w = TRACKED_KH.w
    # The parameters travel into kernels inside the initial-data closure, so
    # they have to be `isbits`, as the equation of state and the floors are.
    @test isbits(w)
    @test w.ρ₁ == 1 && w.ρ₂ == 2
    @test w.v₁ == 1 // 2 && w.v₂ == -1 // 2
    @test w.ρ_m == -1 // 2 && w.v_m == 1 // 2       # (2) and (4); ρ_m is negative
    # `w.L` and `w.a` are `Float64(1//40)` and `Float64(1//100)`, which are not
    # the rationals themselves — neither denominator is a power of two — so the
    # comparison is against the converted value and not against `1//40`.
    @test w.p₀ == 5 // 2
    @test w.L == Float64(1 // 40) && w.a == Float64(1 // 100)
    @test w.eos.γ == Float64(5 // 3)

    # Equations (1), (3) and (5), written out from the paper with its own
    # numbers. Both branches of each ramp, at eight `y` per branch.
    Lp, ρm, vm = 0.025, -0.5, 0.5
    function paper(x, y)
        if 0.25 > y ≥ 0
            e = exp((y - 0.25) / Lp)
            ρ, vx = 1.0 - ρm * e, 0.5 - vm * e
        elseif 0.5 > y ≥ 0.25
            e = exp((-y + 0.25) / Lp)
            ρ, vx = 2.0 + ρm * e, -0.5 + vm * e
        elseif 0.75 > y ≥ 0.5
            e = exp(-(0.75 - y) / Lp)
            ρ, vx = 2.0 + ρm * e, -0.5 + vm * e
        else
            e = exp(-(y - 0.75) / Lp)
            ρ, vx = 1.0 - ρm * e, 0.5 - vm * e
        end
        return (ρ, vx, 0.01 * sin(4π * x), 2.5)
    end
    worst = 0.0
    for x in 0.0:(1 / 37):1.0, y in 0.0:(1 / 131):1.0
        worst = max(worst, maximum(abs.(kh_state(w, (x, y)) .- paper(x, y))))
    end
    @test worst == 0.0                              # bit for bit, not merely close

    # The profile is continuous at all three branch boundaries. At `¼` and `¾`
    # it is continuous in its derivative too and the jump is bounded by the
    # slope `|ρ_m|/L = 20`; at `½` the two branches agree **exactly**, both
    # being `ρ₂ + ρ_m e^{−1/(4L)}`.
    δ = 1e-8
    for y in (0.25, 0.5, 0.75)
        lo, hi = kh_state(w, (0.3, y - δ)), kh_state(w, (0.3, y + δ))
        @test all(abs.(hi .- lo) .≤ 40 * δ)
    end
    # The mean of the two slabs at the interfaces, which is what a smooth ramp
    # between 1 and 2 has to give.
    @test kh_state(w, (0.3, 0.25))[1] ≈ 1.5
    @test kh_state(w, (0.3, 0.75))[1] ≈ 1.5
    @test kh_state(w, (0.3, 0.25))[2] ≈ 0.0 atol = 1e-15
    # Equation (5): the seed is one mode, four wavelengths across the box.
    @test kh_state(w, (0.125, 0.1))[3] ≈ 0.01
    @test kh_state(w, (0.375, 0.1))[3] ≈ -0.01
    @test kh_state(w, (0.25, 0.1))[3] ≈ 0.0 atol = 1e-17
    @test all(x -> kh_state(w, (x, 0.1))[4] == 2.5, 0.0:0.05:1.0)

    # The closed-form integrals of the initial data. The `ρ_m` terms of the
    # four branches cancel **exactly** — two enter with `+ρ_m` and two with
    # `−ρ_m` over equal intervals — so the mass is `(ρ₁+ρ₂)/2` whatever `L` is,
    # which is a check on the branch *signs* and not merely on the arithmetic.
    E = exp(-1 / (4 * Lp))
    mass = (w.ρ₁ + w.ρ₂) / 2
    I1 = w.ρ₁ * w.v₁ / 4 - (w.ρ₁ * vm + ρm * w.v₁) * Lp * (1 - E) +
         ρm * vm * (Lp / 2) * (1 - E^2)
    I2 = w.ρ₂ * w.v₂ / 4 + (w.ρ₂ * vm + ρm * w.v₂) * Lp * (1 - E) +
         ρm * vm * (Lp / 2) * (1 - E^2)
    Sx = 2 * (I1 + I2)
    t0 = TRACKED_KH.r.totals0
    @test t0[1] ≈ mass rtol = 1e-12                 # measured 1.3 ulp
    @test t0[2] ≈ Sx rtol = 1e-4                    # measured 2.69e-5
    @test abs(t0[3]) ≤ 1e-15                        # ∫sin(4πx) dx = 0

    # What the case refuses, and why each refusal is worth having.
    @test_throws "stated for D = 2" KelvinHelmholtz(Float64, Val(1))
    @test_throws "stated for D = 2" KelvinHelmholtz(Float64, Val(3))
    @test_throws "needs ρ₁ ≠ ρ₂" KelvinHelmholtz(Float64, Val(2); ρ₂=1)
    @test_throws "needs v₁ ≠ v₂" KelvinHelmholtz(Float64, Val(2); v₂=1 // 2)
    @test_throws "ramp width must be positive" KelvinHelmholtz(Float64, Val(2);
                                                               L=0)
    @test_throws "same root count in both" HydroCase(w; roots=(4, 2))
    @test_throws "at least two samples" growth_rate([0.0, 1.0], [1.0, 2.0];
                                                    from=10.0)
    @test_throws "one amplitude per time" growth_rate([0.0], [1.0, 2.0]; from=0)

    @info "Kelvin–Helmholtz setup: ρ₁ = $(w.ρ₁), ρ₂ = $(w.ρ₂), v₁ = $(w.v₁), " *
          "v₂ = $(w.v₂), ρ_m = $(w.ρ_m), v_m = $(w.v_m), p = $(w.p₀), " *
          "L = $(w.L), a = $(w.a), γ = $(w.eos.γ); worst |kh_state − paper| " *
          "over 38×132 points = $worst; initial totals $(t0) against the " *
          "closed forms mass $mass and S_x $Sx"
end

@testset "M(t) grows exponentially through the linear phase, below both bounds" begin
    # The failure mode is an instability that is not the seeded one: a growth
    # rate above the incompressible bound means the measurement is picking up
    # grid noise rather than the mode, and one far below it means the scheme
    # has diffused the layer away. Both bounds are recorded, and the measured
    # rate must lie under both.
    #
    # The two bounds, and neither is this problem's own. The paper's loose
    # guide for the **infinite-domain incompressible** flow is
    # `M ∝ exp(4.384 t)` (Wang et al. 2010, Eq. 18), with
    # `max ½ρv_y² ∝ exp(2 × 4.384 t)`; the **sharp-interface** incompressible
    # result is `k Δv √(ρ₁ρ₂)/(ρ₁+ρ₂) = 4π·√2/3 = 5.9239` for `k = 4π`. The
    # ramp and the compressibility both slow the real thing, and this run is
    # periodic and compressible, which is why the paper says its own reference
    # solution is the quantitative one.
    #
    # Measured (tracked, cap 2, HLLC, 2700 steps in 300 chunks):
    #   M: 0.0100 → 0.12346, a factor of 12.346; minimum 0.008063 at t = 0.335
    #   K: 9.956e-5 → 0.037335, a factor of 375.0
    #   rate over 2a ≤ M ≤ 6a (t ∈ [0.735, 1.150], 84 samples): 2.58036
    #   K rate over the same window: 5.42416, a ratio of 2.1021 to the mode's
    #   uniform fine 128²: 2.58324 and 5.42662 — the mesh does not move it
    # and `M(t)` has **not** saturated by t = 1.5: the local rate falls from
    # 3.349 at t ≈ 0.5 through the window's 2.58 to 1.844 over the last 20
    # chunks, so the curve is bending over and the run ends on its shoulder.
    #
    # Against the paper's own curves (their Figure 7, every code at 128², 256²
    # and 512² beside the 4096² reference): the reference starts at 0.01, dips
    # and plateaus until t ≈ 0.45, then takes off — and so does this, minimum
    # 0.008063 at t = 0.335 and take-off at t ≈ 0.45. What does not reproduce
    # is the late curve: the reference reaches a few tenths with its rate still
    # rising, this run reaches 0.1235 with its rate falling. That is
    # under-resolution and not saturation — 128² is the lowest resolution the
    # paper runs, the codes that sit on the reference there are
    # piecewise-parabolic or sixth-order, and this scheme is second-order
    # MUSCL. Asserted below is the shape and the bounds, not the amplitude.
    x, f = TRACKED_KH, FINE_KH
    a = Float64(x.w.a)
    rate = growth_rate(x.ts, x.Ms; from=KH_WINDOW.from, to=KH_WINDOW.to)
    rate_f = growth_rate(f.ts, f.Ms; from=KH_WINDOW.from, to=KH_WINDOW.to)
    # The kinetic-energy window is the *same* window, carried over through the
    # samples `M` selects, since the two diagnostics are read at the same times.
    win = [i for i in eachindex(x.Ms) if KH_WINDOW.from ≤ x.Ms[i] ≤ KH_WINDOW.to]
    krate = growth_rate(x.ts[win], x.Ks[win]; from=0.0)

    @test x.Ms[1] ≈ a rtol = 1e-12          # M(0) is the seed's own amplitude
    @test rate < 4.384                      # Wang et al. 2010, Eq. 18
    @test rate < 4π * sqrt(2) / 3           # the sharp-interface bound, 5.9239
    @test rate > 2                          # and it really is growing
    @test krate < 2 * 4.384                 # ½ρv_y² is quadratic in v_y
    @test krate / rate ≈ 2 rtol = 1e-1      # measured 2.1021
    @test x.Ms[end] / x.Ms[1] > 10          # measured 12.346
    @test x.Ks[end] / x.Ks[1] > 100         # measured 375.0
    # The mesh is not what sets the rate: the uniform fine run at the same
    # finest spacing agrees to a tenth of a percent.
    @test rate ≈ rate_f rtol = 1e-2
    # Not saturated, and decelerating: the local rate at the end is well below
    # the window's, and `M` is still at its maximum at `t_end`.
    tail = (log(x.Ms[end]) - log(x.Ms[end - 20])) / (x.ts[end] - x.ts[end - 20])
    @test tail < rate
    @test x.Ms[end] == maximum(x.Ms)
    # The paper's shape in the first half: `M` dips below the seed and does
    # not take off until `t ≈ 0.45`. Measured: minimum 0.008063 at t = 0.335,
    # and still 0.008460 at t = 0.45.
    @test minimum(x.Ms) < x.Ms[1]
    @test x.ts[argmin(x.Ms)] < 0.5
    @test x.Ms[findfirst(≥(0.45), x.ts)] < x.Ms[1]
    # The mesh tracks it: every strongly firing cell sat on a block at the cap
    # at every chunk, and the hierarchy is still a hierarchy at the end.
    @test x.r.tracking == 1.0
    @test x.r.levels == [1, 2]

    @info "Kelvin–Helmholtz tracked (cap $(KH.cap), $(KH_FLUX), $(x.r.nsteps) " *
          "steps in $(x.r.nchunks) chunks): M $(x.Ms[1]) → $(x.Ms[end]) " *
          "(×$(x.Ms[end] / x.Ms[1]), minimum $(minimum(x.Ms)) at " *
          "t = $(x.ts[argmin(x.Ms)])), max ½ρv_y² $(x.Ks[1]) → $(x.Ks[end]) " *
          "(×$(x.Ks[end] / x.Ks[1])); growth rate $rate over " *
          "$(length(win)) samples with $(KH_WINDOW.from) ≤ M ≤ " *
          "$(KH_WINDOW.to), t ∈ [$(x.ts[win[1]]), $(x.ts[win[end]])], against " *
          "the bounds 4.384 and $(4π * sqrt(2) / 3); kinetic-energy rate " *
          "$krate, a ratio of $(krate / rate); uniform fine rate $rate_f; " *
          "local rate over the last 20 chunks $tail, so M has not saturated"
end

@testset "The refined region is two strips at t = 0 and grows with the rolls" begin
    # `CODE.md` predicted "two strips at t = 0, then rolls, then the whole
    # layer" — the criterion's behaviour under a feature whose footprint
    # changes shape, which neither the travelling pulse nor the expanding
    # shell tests. The failure mode is a refined region that is not the layer:
    # a criterion firing on the uniform slabs would refine everything and the
    # case would measure nothing about adaptivity at all.
    #
    # Measured at t = 0 (cycle converged in 4 passes, 3-cell margin): 128 blocks
    # at the cap and 32 at level 1, 10240 cells against the uniform fine mesh's
    # 16384, and **every edge of every block at the cap lies within 1/8 of
    # y = ¼ or y = ¾** — five ramp widths, the band being 1/4 of the box in
    # each strip. Then 160 → 196 (t = 0.915) → 220 (t = 1.355) → 232
    # (t = 1.390) blocks, **monotone**, in 3 mesh changes.
    x = TRACKED_KH
    # At `t_end` the rolls have thickened the layer until the refined region
    # reaches the middle of each slab — the band measure saturates at its
    # largest possible value, `1/4` — and the mesh is still not the uniform
    # fine one: 224 of the 256 possible finest blocks, and 232 leaves in all.
    capped = count(k -> level(k) == KH.cap, x.r.forest.leaves)
    @test x.r.nblocks < (KH.roots * 2^KH.cap)^2
    @test capped < (KH.roots * 2^KH.cap)^2
    @test x.r.cells < FINE_KH.r.cells
    @test issorted(x.nbs)                    # rising, and never falling
    @test x.nbs[end] > x.nbs[1]
    @test x.r.nregrids ≥ 1
    @test x.r.converged && 1 ≤ x.r.passes ≤ 8
    @test unique(x.r.buffer_history) == [3]

    # And at `t = 0` it really is two strips. The cycle is re-run here rather
    # than read off the tracked run, because `evolve!` returns the mesh at
    # `t_end` and the claim is about the mesh the *cycle* built.
    r0 = tracked_kh(; t_end=KH.chunk)
    band0 = kh_refined_band(r0.r.forest, KH.cap)
    fine0 = count(k -> level(k) == KH.cap, r0.r.forest.leaves)
    @test band0 ≤ 1 / 8                      # measured exactly 1/8 = 5 L
    @test fine0 == 128
    @test r0.r.levels == [1, 2]
    @test r0.r.cells == 10240

    @info "Kelvin–Helmholtz mesh: at t = 0 the cycle converged in " *
          "$(r0.r.passes) passes onto $(r0.r.nblocks) blocks at levels " *
          "$(r0.r.levels) ($(fine0) at the cap, $(r0.r.cells) cells against " *
          "$(FINE_KH.r.cells) uniformly fine), every cap-level block edge " *
          "within $(band0) of an interface = $(band0 / Float64(x.w.L)) ramp " *
          "widths; the count then rises $(x.nbs[1]) → $(x.nbs[end]) " *
          "monotonically over $(x.r.nchunks) chunks and $(x.r.nregrids) mesh " *
          "changes, ending with $capped of $((KH.roots * 2^KH.cap)^2) possible " *
          "finest blocks and a band of $(kh_refined_band(x.r.forest, KH.cap)); " *
          "margin $(unique(x.r.buffer_history)) cells"
end

@testset "Conservation through the regrids, and a leak without the fixup" begin
    # The plain claim: there is no physical boundary here, so nothing is
    # allowed to move at all. On the blast this measurement was impossible on
    # a tracked mesh — tracking puts the refined region's boundary ahead of the
    # shock, so its coarse-fine faces stand in undisturbed gas (step 9) — and
    # here it is possible, because the *whole domain* is in motion and the
    # layer's strips have coarse-fine faces along them from `t = 0`.
    #
    # Measured (tracked, cap 2, 2700 steps, 3 mesh changes):
    #   drift (8.88e-16, 2.78e-16, 4.45e-18, 1.78e-15)
    #   bound (7.19e-12, 3.24e-12, 4.01e-13, 1.88e-11)
    #   floor/reset/ghost hits 0/0/0, injection exactly (0, 0, 0, 0)
    # and without the fixup, on the same mesh and the same 720 steps at
    # t = 2/5, the leak is 9.1e4 (mass), 2.2e6 (S_x) and 1.8e5 (energy) times
    # the roundoff bound.
    #
    # **`S_y` does not leak**, and that is measured rather than assumed: its
    # drift without the fixup is 6.3e-18 against a bound of 1.2e-14, half a
    # thousandth of it. The coarse-fine faces here are the horizontal edges of
    # the two strips and they span the whole of `x`; the `S_y` structure is the
    # single mode `sin(4πx)`, whose integral over `x` is zero, so the flux
    # mismatch inherits that zero mean. The other three have no such symmetry
    # to protect them.
    T = Float64
    x = TRACKED_KH
    rb(r, v) = 8 * eps(T) * r.scales[v] * r.nsteps
    for v in 1:4
        @test x.r.drift[v] ≤ rb(x.r, v)
    end
    @test (x.r.floor_hits, x.r.reset_hits, x.r.ghost_hits) == (0, 0, 0)
    @test x.r.injection == (0.0, 0.0, 0.0, 0.0)

    a, c = SHORT_KH, SHORT_KH_NOFIX
    @test a.r.nsteps == c.r.nsteps            # the same steps, or it is not a control
    @test a.nbs == c.nbs                      # and the same mesh history
    @test all(v -> a.r.drift[v] ≤ rb(a.r, v), 1:4)
    for v in (1, 2, 4)
        @test c.r.drift[v] > 1e4 * rb(a.r, v)
    end
    @test c.r.drift[3] ≤ rb(c.r, 3)           # S_y is the exception, by symmetry
    # Nothing is floored in either run: the density stays in [1, 2] and the
    # pressure near 5/2, so a floor hit here would be a bug upstream of the
    # floors rather than the floors doing their job.
    @test (c.r.floor_hits, c.r.reset_hits, c.r.ghost_hits) == (0, 0, 0)
    @test c.r.injection == (0.0, 0.0, 0.0, 0.0)

    @info "Kelvin–Helmholtz conservation (tracked, $(x.r.nsteps) steps, " *
          "$(x.r.nregrids) mesh changes): drift $(x.r.drift) against roundoff " *
          "bounds $(ntuple(v -> rb(x.r, v), 4)), floor/reset/ghost hits " *
          "$(x.r.floor_hits)/$(x.r.reset_hits)/$(x.r.ghost_hits), injection " *
          "$(x.r.injection)"
    @info "Kelvin–Helmholtz fixup control (t = $(Float64(KH_SHORT)), " *
          "$(a.r.nsteps) steps): with the fixup $(a.r.drift), without it " *
          "$(c.r.drift) against bounds $(ntuple(v -> rb(a.r, v), 4)) — ratios " *
          "$(ntuple(v -> c.r.drift[v] / rb(a.r, v), 4)), S_y protected by the " *
          "zero mean of sin(4πx) over the coarse-fine faces"
end

@testset "The adaptive run approaches the uniform fine run as the cap rises" begin
    # What would pass without the claim: a run that refines everything matches
    # the fine reference and costs as much, and a run that refines nothing
    # matches the coarse control. So the sweep is the claim, and the cap is the
    # only thing that varies across it.
    #
    # Measured (t = 1.5, HLLC, reduced onto the 32² grid the four meshes share):
    #   cap 0 (uniform coarse)   1024 cells   L1 5.98e-2   |ΔM|/M 0.9064
    #   cap 1                    4096 cells   L1 2.59e-2   |ΔM|/M 0.3911
    #   cap 2                   14848 cells   L1 4.24e-4   |ΔM|/M 0.004233
    #   uniform fine 128²       16384 cells   L1 0          |ΔM|/M 0
    # Both columns fall monotonically, and the mean distance of the whole
    # `M(t)` curve from the fine run's falls with them: 2.85e-2, 1.14e-2,
    # 1.71e-4.
    gf = reduce_to_grid(FINE_KH.r.U, KH_GRID)
    sweep = (("cap 0", COARSE_KH), ("cap 1", CAP1_KH), ("cap 2", TRACKED_KH))
    l1s = [l1_difference(reduce_to_grid(x.r.U, KH_GRID), gf) for (_, x) in sweep]
    dMs = [abs(x.Ms[end] - FINE_KH.Ms[end]) / FINE_KH.Ms[end] for (_, x) in sweep]
    curves = [sum(abs.(x.Ms .- FINE_KH.Ms)) / length(x.Ms) for (_, x) in sweep]

    @test issorted(l1s; rev=true)            # the L1 difference falls with the cap
    @test issorted(dMs; rev=true)            # and so does the M(t_end) difference
    @test issorted(curves; rev=true)         # and the whole curve with it
    @test l1s[end] < l1s[1] / 100            # and the last step is a large one
    @test TRACKED_KH.r.cells < FINE_KH.r.cells
    # The cell counts are the sweep's other axis, and they rise with it.
    @test COARSE_KH.r.cells < CAP1_KH.r.cells < TRACKED_KH.r.cells

    for (i, (tag, x)) in enumerate(sweep)
        @info "Kelvin–Helmholtz cap sweep, $tag: $(x.r.cells) cells in " *
              "$(x.r.nblocks) blocks at levels $(x.r.levels), $(x.r.nsteps) " *
              "steps; against the uniform fine run on the $(KH_GRID)² grid " *
              "L1 $(l1s[i]), |ΔM|/M $(dMs[i]), mean |M − M_fine| $(curves[i]); " *
              "M_end $(x.Ms[end]) against $(FINE_KH.Ms[end])"
    end
end

@testset "HLLE against HLLC on a contact-dominated flow, and the flux decided" begin
    # The measurement `CODE.md`'s "Riemann solver" defers to this milestone,
    # and the criterion was written down **before** the HLLE runs were made:
    #
    #   1. the two uniform fine references must agree in M(t_end) to within 5%,
    #      or "the fine reference" is itself flux-dependent and the two
    #      tracked-against-fine comparisons are comparisons of different things;
    #   2. HLLC becomes this case's default if its tracked run is closer to its
    #      own fine reference than HLLE's is, in M(t_end) and in the
    #      reduced-grid L1, by a factor above 2 in at least one of the two;
    #   3. the difference between the two fine references is recorded, and if
    #      HLLC's exceeds HLLE's by more than 2% the contact diffusion CODE.md
    #      predicted is visible at this resolution;
    #   4. the package-wide default stays :hlle whatever the numbers say.
    #
    # **Criterion 1 failed and criterion 2 measured the wrong thing**, which is
    # the honest record of it. The two fine references differ by 247%, not 5%;
    # and criterion 2 comes out 0.42% against 0.29% — both tiny, HLLE nominally
    # closer — because tracked-against-same-flux-fine measures the *mesh* and
    # not the flux, and the mesh is equally good under either. Criterion 3 is
    # overwhelming, so the decision is made on it and on the **resolution
    # sweep** that criterion 3 implies, which is the amendment:
    #
    #   M(t = 1.5), uniform      32²        64²        128²
    #     HLLC                   0.011608   0.075489   0.123981
    #     HLLE                   0.000662   0.006599   0.035717
    #     ratio                  17.53      11.44      3.471
    #
    # HLLC at **half the linear resolution** is further along than HLLE at full
    # resolution — 0.075489 at 64² against 0.035717 at 128², a factor of 2.11 —
    # and the same at the rung below, 0.011608 at 32² against 0.006599 at 64².
    # So on this flow HLLE costs at least a factor of two in linear resolution,
    # which is four in cells and eight in work. The shear layer *is* a contact,
    # HLLE's two-wave average is what smears it, and that is the whole of the
    # difference between the two solvers.
    #
    # **Decided: HLLC is this case's default flux** (`kh_run`'s `riemann`
    # keyword, and `KH_FLUX` here). The package-wide default in `HydroProblem`
    # and `evolve!` stays `:hlle` — it is *the* GRMHD flux and the baseline the
    # comparison is against.
    #
    # Growth rates over the same window: HLLC 2.58036 tracked and 2.58324 fine,
    # HLLE 1.17122 and 1.16794. The two solvers do not merely differ in
    # amplitude; they differ in the rate the linear phase grows at.
    gc, ge = FINE_KH, FINE_KH_E
    refs = abs(gc.Ms[end] - ge.Ms[end]) / ge.Ms[end]
    @test refs > 0.05                        # criterion 1 fails, by a lot

    trel(t, f) = abs(t.Ms[end] - f.Ms[end]) / f.Ms[end]
    l1(t, f) = l1_difference(reduce_to_grid(t.r.U, KH_GRID),
                             reduce_to_grid(f.r.U, KH_GRID))
    rc, re = trel(TRACKED_KH, gc), trel(TRACKED_KH_E, ge)
    lc, le = l1(TRACKED_KH, gc), l1(TRACKED_KH_E, ge)
    # Criterion 2: both are under a percent, so it decides nothing — which is
    # the point being recorded, not a tolerance being loosened.
    @test rc < 0.01 && re < 0.01
    @test lc < 1e-3 && le < 1e-3

    # Criterion 3 and the resolution sweep, which is what decides.
    @test gc.Ms[end] > 3 * ge.Ms[end]        # measured 3.471 at 128²
    @test MID_KH.Ms[end] > FINE_KH_E.Ms[end] # HLLC at 64² beats HLLE at 128²
    @test COARSE_KH.Ms[end] > MID_KH_E.Ms[end]   # and at 32² beats it at 64²
    # The ratio narrows with resolution, as two consistent fluxes must.
    ratios = [COARSE_KH.Ms[end] / COARSE_KH_E.Ms[end],
              MID_KH.Ms[end] / MID_KH_E.Ms[end],
              FINE_KH.Ms[end] / FINE_KH_E.Ms[end]]
    @test issorted(ratios; rev=true)
    @test ratios[end] > 1                    # and HLLC is ahead at every rung
    # The rates differ too, and the mesh does not move either of them.
    rate_c = growth_rate(gc.ts, gc.Ms; from=KH_WINDOW.from, to=KH_WINDOW.to)
    rate_e = growth_rate(ge.ts, ge.Ms; from=KH_WINDOW.from, to=KH_WINDOW.to)
    @test rate_c > 2 * rate_e
    @test rate_e < 4.384                     # both stay under the bound
    @test KH_FLUX == :hllc                   # the decision, where it is made

    @info "Kelvin–Helmholtz flux comparison at t = $(Float64(KH.t_end)): " *
          "uniform M(t_end) at 32²/64²/128² is " *
          "$(COARSE_KH.Ms[end])/$(MID_KH.Ms[end])/$(FINE_KH.Ms[end]) under " *
          "HLLC and $(COARSE_KH_E.Ms[end])/$(MID_KH_E.Ms[end])/" *
          "$(FINE_KH_E.Ms[end]) under HLLE — ratios $(ratios); HLLC at 64² " *
          "($(MID_KH.Ms[end])) is ahead of HLLE at 128² ($(FINE_KH_E.Ms[end]))"
    @info "Kelvin–Helmholtz flux comparison, tracked against its own fine " *
          "reference: HLLC |ΔM|/M $rc and L1 $lc, HLLE $re and $le — both " *
          "under a percent, so the comparison measures the mesh and not the " *
          "flux; the two fine references differ by $(refs) in M(t_end) and " *
          "$(l1_difference(reduce_to_grid(gc.r.U, KH_GRID), reduce_to_grid(ge.r.U, KH_GRID))) " *
          "in L1; growth rates $rate_c against $rate_e. Decided: HLLC is the " *
          "Kelvin–Helmholtz default, the package-wide default stays HLLE"
end

@testset "Float32 reproduces the mesh and the linear phase, and not the rest" begin
    # `CODE.md` says the claim at `Float32` is the mesh statistics of the first
    # chunks and `M(t)` through the linear phase, and **not** the final state:
    # the instability amplifies roundoff exponentially, so two precisions
    # diverge in detail while agreeing in what is being measured. This is the
    # one case in the package where "the same mesh at both precisions" is a
    # statement with a time limit on it, and the run is stopped at `t = 2/5` —
    # still inside the transient, before the mode takes off — so that the
    # claim is made where it holds.
    #
    # Measured (t = 2/5, 720 steps, tracked cap 2):
    #   the same 160 blocks at levels [1, 2] at every chunk, the same step
    #   count, tracking 1.0 at both types
    #   max relative |M₃₂ − M₆₄| over the run 1.857e-5 = 156 ulp of Float32
    #   max relative |K₃₂ − K₆₄| over the run 2.552e-4
    #   |M₃₂ − M₆₄| / M at t_end 2.029e-6
    # The tolerances below are a thousand and ten thousand ulp of `Float32`,
    # which is roughly ten times a *linear* accumulation of one ulp over the
    # 720 steps — room for the amplification without room for a divergence.
    #
    # MultiFloats is skipped entirely and deliberately: `sin` and `exp` are not
    # implemented there, so this case cannot run at `Float32x2` at all. Sod and
    # Sedov carry that half of the precision study.
    a, b = SHORT_KH, SHORT_KH_32
    @test b.r.nsteps == a.r.nsteps
    @test b.nbs == a.nbs
    @test b.r.levels == a.r.levels
    @test b.r.nblocks == a.r.nblocks
    @test b.r.tracking == a.r.tracking
    @test b.r.passes == a.r.passes
    @test (b.r.floor_hits, b.r.reset_hits, b.r.ghost_hits) == (0, 0, 0)

    n = min(length(a.Ms), length(b.Ms))
    dM = maximum(abs.(b.Ms[1:n] .- a.Ms[1:n]) ./ a.Ms[1:n])
    dK = maximum(abs.(b.Ks[1:n] .- a.Ks[1:n]) ./ a.Ks[1:n])
    @test dM ≤ 1000 * eps(Float32)           # measured 156 ulp
    @test dK ≤ 10000 * eps(Float32)          # measured 2144 ulp
    # And the `Float32` run's own conserved integrals hold to *its* roundoff,
    # which is a different and much looser bound than the `Float64` one.
    for v in 1:4
        @test b.r.drift[v] ≤ 8 * eps(Float32) * b.r.scales[v] * b.r.nsteps
    end

    @info "Kelvin–Helmholtz Float32 (t = $(Float64(KH_SHORT)), $(b.r.nsteps) " *
          "steps): the same $(b.r.nblocks) blocks at levels $(b.r.levels) at " *
          "every one of $(length(b.nbs)) samples, tracking $(b.r.tracking); " *
          "max relative |M₃₂ − M₆₄| $dM = $(dM / eps(Float32)) ulp of Float32 " *
          "and |K₃₂ − K₆₄| $dK; M_end $(b.Ms[end]) against $(a.Ms[end]); " *
          "drift $(b.r.drift) against bounds " *
          "$(ntuple(v -> 8 * eps(Float32) * b.r.scales[v] * b.r.nsteps, 4))"
end
