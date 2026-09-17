# The refinement criterion: the Löhner indicator on `ρ` and `p`, and the
# per-block verdict it reduces to.
#
# These testsets are claims about the *indicator*, not about a run: no mesh
# changes here, and nothing is regridded. That the criterion actually
# steers a moving mesh is step 7's claim, measured against a uniform mesh
# of the same finest spacing.
#
# Every one of them prepares its data the way a driver will — build a
# `HydroProblem`, fill `U`, call `update_primitives!` so that `P` is
# current *with its ghosts* — and then reads `P`. The stencil reaches one
# cell past each block face, so a test that skipped the ghost fill would
# be measuring zeros at every block boundary and would still pass most of
# what is below, which is why the preparation is a shared helper rather
# than written out four times.
#
# The calibration that *chose* the tolerances — max `τ` against `h` on
# uniform meshes, four spacings for Sod's four features and for the McNally
# ramp — is four evolutions and lives in `test/long/refinement_tests.jl`
# with the rest of the long tier; the short tier pins its `h = 1/64` row in
# `regression_tests.jl` instead. The helpers below are shared with it: the
# short tier is always included first (see `runtests.jl`).

# The conservative family at the interface order the scheme wants — the
# same operators `interface_tests.jl` runs on, so the two-level mesh below
# is the one measured there.
refine_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

# The calibrated defaults, in one place, so that a testset that is not
# about the thresholds does not quietly invent its own.
const REFINE_TOL = 2 // 25            # 0.08
const COARSEN_TOL = 1 // 50           # 0.02

"""
A `HydroProblem` over `forest` whose primitives hold `initial(x)` — the
primitive tuple — with the ghosts filled and `P` current.

The two-line sequence every driver runs before flagging: fill, gather,
[`update_primitives!`](@ref). `boundary` is the physical-boundary hook or
`nothing`.
"""
function primed_problem(::Val{D}, forest, initial; eos, floors, boundary=nothing,
                        limiter=:minmod, T=Float64) where {D}
    U = FieldSet{T}(forest, D + 2; G=2)
    p = HydroProblem(U, refine_ops(); eos=eos, floors=floors, limiter=limiter,
                     riemann=:hlle, boundary=boundary)
    fill_by_coordinates!(AllVariables(x -> prim2con(eos, initial(x))), U)
    u = statevector(U)
    gather!(u, U)
    update_primitives!(p, u)
    return (p=p, u=u)
end

"""Sod's tube on a mesh of `sod_forest`'s, evolved to `t_end` if positive."""
function primed_sod(::Val{D}, N; roots=4, refined=false, t_end=0, cfl=2 // 5,
                    limiter=:minmod, T=Float64) where {D}
    w = SodTube(T, Val(D))
    rs = ntuple(d -> d == 1 ? roots : 1, D)
    forest = sod_forest(Val(D), N; roots=rs, L=w.L, refined=refined, T=T)
    r = primed_problem(Val(D), forest, x -> TreeHydro.sod_state(w, x); eos=w.eos,
                       floors=w.floors, boundary=sod_boundary(w), limiter=limiter)
    T(t_end) > 0 || return (r..., w=w, forest=forest)
    λ = T(max_signal_speed(exact_riemann(w)))
    dt = hydro_dt(forest, T(cfl), λ, Val(D))
    u = hydro_solve!(r.p, r.u, zero(T), T(t_end), ceil(Int, T(t_end) / dt))
    update_primitives!(r.p, u)
    return (p=r.p, u=u, w=w, forest=forest)
end

"""
Every owned cell as `(b, i, x, τ)`: its block, its **interior** index
`1:N`, its position and its indicator.

The host walk `firing_boxes` makes in a kernel, written out so that a test
can say *which* cells fired rather than only how many. It calls the same
[`cell_tau`](@ref) the predicates do, so the two cannot drift apart.
"""
function tau_table(p::HydroProblem{T,D}; ε=T(1 // 100), ε_g=T(1 // 1000),
                   scales=indicator_scales(p.P)) where {T,D}
    P = p.P
    N = P.forest.N
    refs = (T(scales[1]), T(scales[2]))
    return [begin
                i = ntuple(d -> Tuple(c)[d], D)
                idx = ntuple(d -> i[d] + P.G[d], D)
                (b=b, i=i, x=coordinates(T, P, b, idx),
                 τ=cell_tau(P.work, idx, b, refs, T(ε), T(ε_g), Val(D)))
            end
            for b in 1:nblocks(P) for c in CartesianIndices(ntuple(_ -> N, D))]
end

"""The Löhner indicator of variable `v` along dimension `d` at one owned cell."""
function var_tau(p::HydroProblem{T,D}, b, i, v, d, u_ref; ε=T(1 // 100),
                 ε_g=T(1 // 1000)) where {T,D}
    P = p.P
    idx = ntuple(e -> i[e] + P.G[e], D)
    u0 = P.work[idx..., v, b]
    up = P.work[Base.setindex(idx, idx[d] + 1, d)..., v, b]
    um = P.work[Base.setindex(idx, idx[d] - 1, d)..., v, b]
    return lohner(um, u0, up, u_ref; ε=T(ε), ε_g=T(ε_g))
end

"""The bare `RegridFlag` of a mark, with or without a box."""
bareflag(m) = m isa Tuple ? m[1] : m
"""The box a mark reports, or `nothing` for a bare flag."""
markbox(m) = m isa Tuple ? m[2] : nothing

"""The indices of the blocks whose extent satisfies `pred`."""
blocks_where(P, pred) =
    [b for b in 1:nblocks(P) if pred(block_extent(P.forest, blockkey(P, b)))]

# Floors far below anything these testsets contain, so that nothing here
# measures the floors by accident. The atmosphere testset builds its own
# "atmosphere" eight orders above them on purpose: what it is about is the
# indicator's global term, not the floor rules.
quiet_floors(T=Float64) = Floors{T}(; ρ_atm=T(1 // 10^12), p_atm=T(1 // 10^12),
                                    p_floor=T(1 // 10^12))

@testset "τ fires at Sod's discontinuity and nowhere else: D=$D" for D in (1, 2)
    # The failure mode on both sides. A criterion whose floor is wrong
    # fires on the flat states as well and refines the whole tube — which
    # is what TreeWave measured when its floor was scale-free — and one
    # whose stencil never crosses a block face misses the jump entirely,
    # since Sod's diaphragm sits exactly on the face between two root
    # blocks. Both come out as "every block reports something" versus
    # "no block reports anything", so the claim has to be the *exact* set
    # of firing cells, not a count.
    T = Float64
    N = 8
    r = primed_sod(Val(D), N; roots=4)
    P = r.p.P
    x₀ = r.w.x₀

    # ρ and p are 1 on the left and 1/8, 1/10 on the right, so the global
    # references are the left state's.
    @test indicator_scales(P) == (1.0, 1.0)

    table = tau_table(r.p)
    fired = [c for c in table if c.τ > T(REFINE_TOL)]
    h = minimum_spacing(T, r.forest)
    # In D = 2 the jump is a plane, so a whole column of cells fires.
    @test length(fired) == (D == 1 ? 2 : 2 * N)
    # Every one of them is a cell that touches the diaphragm.
    @test all(c -> abs(c.x[1] - x₀) ≈ h / 2, fired)
    # A one-cell jump is Löhner's canonical case and scores nearly 1.
    @test all(c -> c.τ > 0.95, fired)
    # And nothing else comes anywhere near: the flat states score exactly 0.
    @test maximum(c.τ for c in table if !(c in fired)) == 0.0

    flags = hydro_flags(r.p; refine_tol=REFINE_TOL, coarsen_tol=COARSEN_TOL,
                        maxlevel_cap=2)
    @test length(flags) == nblocks(P)
    @test all(b -> level(blockkey(P, b)) == 0, 1:nblocks(P))

    # The two blocks that meet at the diaphragm report exactly the cell
    # that fired — the last one of the left block and the first one of the
    # right — and every other block is a bare `Keep` at level 0.
    left = only(blocks_where(P, ext -> ext[1][2] ≈ x₀))
    right = only(blocks_where(P, ext -> ext[1][1] ≈ x₀))
    @test bareflag(flags[left]) === Refine
    @test bareflag(flags[right]) === Refine
    @test markbox(flags[left]) == ntuple(d -> d == 1 ? (N:N) : (1:N), D)
    @test markbox(flags[right]) == ntuple(d -> d == 1 ? (1:1) : (1:N), D)
    for b in 1:nblocks(P)
        b in (left, right) && continue
        @test flags[b] === Keep
    end
end

@testset "A top-hat pressure fires, which ρ alone would never see" begin
    # Sedov's initial data in miniature: energy deposited as pressure in a
    # ball, in a uniform ambient gas at rest. Nothing about ρ changes
    # anywhere, so a criterion reading the density alone — which is the
    # obvious one to write, and the one a shock-tube test would not catch —
    # returns identically zero on the case the package exists to refine.
    # The Sedov case itself is step 9's; this is the indicator's half of it.
    T = Float64
    D, N, roots = 2, 8, 4
    eos = IdealGas(T(7 // 5))
    r₀, p_in, p_amb = T(1 // 5), T(10), T(1 // 10)
    centre = ntuple(_ -> T(1 // 2), D)
    tophat(x) = (one(T), ntuple(_ -> zero(T), D)...,
                 sqrt(sum(ntuple(d -> (x[d] - centre[d])^2, D))) < r₀ ? p_in : p_amb)

    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    r = primed_problem(Val(D), forest, tophat; eos=eos, floors=quiet_floors())

    scales = indicator_scales(r.p.P)
    @test scales == (1.0, p_in)
    table = tau_table(r.p)
    fired = [c for c in table if c.τ > T(REFINE_TOL)]
    @test !isempty(fired)

    # Every firing cell straddles the edge of the ball, to within the
    # staircase a sphere makes on a Cartesian mesh: a cell fires only if
    # its three-point stencil crosses the jump.
    h = minimum_spacing(T, forest)
    radius(c) = sqrt(sum(ntuple(d -> (c.x[d] - centre[d])^2, D)))
    @test all(c -> abs(radius(c) - r₀) < 2 * h, fired)

    # And the density sees none of it. Not "sees little": the ambient and
    # the deposit have the same ρ, so every second difference of ρ is
    # exactly zero, in every cell and every direction.
    @test maximum(var_tau(r.p, c.b, c.i, 1, d, scales[1])
                  for c in table, d in 1:D) == 0.0
    # The pressure is where all of it lives.
    @test all(c -> maximum(var_tau(r.p, c.b, c.i, D + 2, d, scales[2])
                           for d in 1:D) == c.τ, fired)
end

@testset "ρ catches what p cannot, and p what ρ cannot" begin
    # Two states that differ in exactly one variable each. The criterion
    # has to read both, and the test is written so that each half fails
    # outright — not merely scores lower — if one variable is dropped.
    T = Float64
    D, N, roots = 1, 16, 4
    eos = IdealGas(T(7 // 5))
    h = one(T) / (roots * N)
    forest() = Forest((roots,); N=N, periodic=(true,),
                      extents=((zero(T), one(T)),))
    left(x) = x[1] < T(1 // 2)
    # The box is periodic, so a step at x = 1/2 has a partner at the wrap:
    # **four** cells straddle a jump, two at each of the two of them.
    atjump(c) = min(abs(c.x[1] - T(1 // 2)), c.x[1], one(T) - c.x[1]) ≈ h / 2

    # A pure contact: ρ jumps, p and v are uniform. This is the entropy
    # wave's discontinuous limit and the Kelvin–Helmholtz shear layer's
    # density signature, and the pressure is flat across it by definition.
    contact = primed_problem(Val(D), forest(),
                             x -> (left(x) ? T(2) : T(1), zero(T), one(T));
                             eos=eos, floors=quiet_floors())
    cscales = indicator_scales(contact.p.P)
    ctable = tau_table(contact.p)
    cfired = [c for c in ctable if c.τ > T(REFINE_TOL)]
    @test length(cfired) == 4
    @test all(atjump, cfired)
    @test maximum(var_tau(contact.p, c.b, c.i, D + 2, 1, cscales[2])
                  for c in ctable) == 0.0

    # A pure pressure jump at rest: only `E` differs between the two
    # states, so `con2prim` returns the same ρ everywhere and the density
    # indicator is exactly zero. This is the shock's and the rarefaction's
    # signature, and the half a density criterion misses.
    jump = primed_problem(Val(D), forest(),
                          x -> (one(T), zero(T), left(x) ? one(T) : T(1 // 10));
                          eos=eos, floors=quiet_floors())
    jscales = indicator_scales(jump.p.P)
    jtable = tau_table(jump.p)
    jfired = [c for c in jtable if c.τ > T(REFINE_TOL)]
    @test length(jfired) == 4
    @test all(atjump, jfired)
    @test maximum(var_tau(jump.p, c.b, c.i, 1, 1, jscales[1])
                  for c in jtable) == 0.0
end

@testset "The atmosphere does not fire, and without the global term it does" begin
    # The failure mode this exists for. An evacuated region — Sedov's
    # bubble — is numerical dust: its cells sit near the floor and differ
    # from one another by `O(1)` *relative* amounts at a negligible
    # *absolute* level. Löhner's local floor scales with those values and
    # therefore cannot tell the dust from a feature; it scores it at
    # τ ≈ 1 and the criterion refines the vacuum. The `ε_g · u_ref` term
    # is the whole of the fix, so the negative control here is not
    # decoration: it is the measurement that the term is what does it.
    T = Float64
    D, N, roots = 1, 16, 4                       # h = 1/64, 64 cells
    eos = IdealGas(T(7 // 5))
    h = T(1 // 64)
    atm = T(1 // 10^6)                           # six orders below the data
    # A quarter of the domain holds O(1) gas — which is what sets the
    # global references — and the rest alternates between the atmosphere
    # and twice it, cell by cell. The alternation is written as a function
    # of position so that the ghost cells agree with the owned ones.
    noisy(x) = isodd(floor(Int, x[1] / h)) ? T(2) : one(T)
    function state(x)
        x[1] < T(1 // 4) && return (one(T), zero(T), one(T))
        return (atm * noisy(x), zero(T), atm * noisy(x))
    end

    forest = Forest((roots,); N=N, periodic=(true,),
                    extents=((zero(T), one(T)),))
    r = primed_problem(Val(D), forest, state; eos=eos, floors=quiet_floors())
    @test indicator_scales(r.p.P) == (1.0, 1.0)
    @test floor_hits(r.p) == 0                   # the floors are not what is tested

    # Well inside the noisy region, away from both edges of it.
    inside(c) = T(3 // 10) < c.x[1] < T(19 // 20)
    quiet = [c.τ for c in tau_table(r.p) if inside(c)]
    loud = [c.τ for c in tau_table(r.p; ε_g=0) if inside(c)]
    @test !isempty(quiet)
    @info "atmosphere six orders below the data, O(1) relative noise: max τ " *
          "$(round(maximum(quiet), sigdigits=3)) with ε_g = 1/1000, " *
          "$(round(maximum(loud), sigdigits=3)) with ε_g = 0"

    # With the global term the dust is below even `coarsen_tol` — it does
    # not so much as hold its block against coarsening.
    @test maximum(quiet) < T(COARSEN_TOL)
    # Without it, the same cells score what a genuine discontinuity does.
    @test maximum(loud) > 0.9
    @test count(>(T(REFINE_TOL)), loud) == length(loud)
end

@testset "A box bounds the firing cells and is not their union" begin
    # `firing_boxes` reports a bounding box, and a driver dilates *that*.
    # A top hat six cells wide fires on its two edges and not in its
    # middle, so the box is strictly larger than the set of firing cells —
    # which is the property that makes the reported region convex and
    # closes the notch a vanishing second difference would otherwise leave
    # in the middle of a feature.
    T = Float64
    N = 16
    eos = IdealGas(T(7 // 5))
    forest = Forest((1,); N=N, periodic=(true,), extents=((zero(T), one(T)),))
    # Cells 5 … 10 of 16 hold the deposit; cells 4 and 11 are the ambient
    # ones next to it.
    inside(x) = T(4 // 16) < x[1] < T(10 // 16)
    r = primed_problem(Val(1), forest, x -> (one(T), zero(T),
                                             inside(x) ? T(10) : T(1 // 10));
                       eos=eos, floors=quiet_floors())

    fired = [c.i[1] for c in tau_table(r.p) if c.τ > T(COARSEN_TOL)]
    @test sort(fired) == [4, 5, 10, 11]
    flags = hydro_flags(r.p; refine_tol=REFINE_TOL, coarsen_tol=COARSEN_TOL,
                        maxlevel_cap=2)
    @test bareflag(only(flags)) === Refine
    @test markbox(only(flags)) == (4:11,)
end

@testset "The four marks are the four cases, on a two-level mesh" begin
    # All four in one call, which is the point: a mesh with a feature on
    # its finest blocks and quiet blocks at both levels produces
    # `(Refine, box)`, `(Keep, box)`, a bare `Coarsen` and a bare `Keep`,
    # and each of the four is wrong in its own way if the reduction is.
    # The one that is easy to get wrong is `(Keep, box)`: a block at the
    # cap has stopped being under-resolved and must still report its
    # footprint, or the margin that travels with the feature does not
    # exist. Keying the box on `refine_tol` instead of `coarsen_tol` is
    # exactly the mistake that removes it.
    T = Float64
    N = 8
    r = primed_sod(Val(1), N; roots=4, refined=:middle)
    P = r.p.P
    x₀ = r.w.x₀
    @test forest_levels(r.forest) == [0, 1]

    atdiaphragm = blocks_where(P, ext -> ext[1][1] ≈ x₀ || ext[1][2] ≈ x₀)
    @test length(atdiaphragm) == 2
    @test all(b -> level(blockkey(P, b)) == 1, atdiaphragm)

    # Below the cap: the two blocks holding the jump ask to refine, the
    # other two level-1 blocks are quiet and ask to coarsen, and the two
    # level-0 blocks are quiet and stay put without recruiting anyone.
    below = hydro_flags(r.p; refine_tol=REFINE_TOL, coarsen_tol=COARSEN_TOL,
                        maxlevel_cap=2)
    for b in 1:nblocks(P)
        l = level(blockkey(P, b))
        if b in atdiaphragm
            @test bareflag(below[b]) === Refine
            @test !isempty(only(markbox(below[b])))
        elseif l > 0
            @test below[b] === Coarsen
        else
            @test below[b] === Keep
        end
    end

    # At the cap: the same two blocks report `(Keep, box)` — the same box,
    # since the box is the `coarsen_tol` sweep's and that sweep did not
    # change — and nothing else moves.
    atcap = hydro_flags(r.p; refine_tol=REFINE_TOL, coarsen_tol=COARSEN_TOL,
                        maxlevel_cap=1)
    for b in atdiaphragm
        @test bareflag(atcap[b]) === Keep
        @test markbox(atcap[b]) == markbox(below[b])
    end
    @test [bareflag(m) for m in atcap] != [bareflag(m) for m in below]
    for b in 1:nblocks(P)
        b in atdiaphragm && continue
        @test atcap[b] === below[b]
    end

    # A `(Keep, box)` is a dilation source and a bare `Keep` is not, which
    # is what makes the margin travel: buffering the marks holds blocks
    # that had asked to coarsen. The box here is the single cell against
    # the diaphragm, so the margin has to reach a whole block width to
    # cross the *other* face of its own block and recruit anything.
    @test count(==(Coarsen), buffered_flags(r.forest, atcap, N)) <
          count(==(Coarsen), [bareflag(m) for m in atcap])

    # The dead band is not optional.
    @test_throws ArgumentError hydro_flags(r.p; refine_tol=1 // 10,
                                           coarsen_tol=1 // 10, maxlevel_cap=2)
    @test_throws ArgumentError hydro_flags(r.p; refine_tol=1 // 10,
                                           coarsen_tol=1 // 5, maxlevel_cap=2)
end

@testset "The buffer covers the travel and refuses more than a block width" begin
    # TreeAMR's measured guidance: the margin must *exceed* the motion it
    # covers, because a narrower one costs cells and still loses the
    # feature. Recruitment reaches one ring of neighbours, so a feature may
    # not cross a whole finest-level block between regrids, and that is
    # refused by name rather than truncated silently.
    forest = Forest((8,); N=8, periodic=(true,), extents=((0.0, 1.0),))
    for travel in (0.005, 0.01, 0.02)
        cells = refinement_buffer(forest, 2, travel)
        @test cells > travel / spacing(forest, 2)
        @test cells <= forest.N
    end
    # Exactly the spacing still gets a margin wider than the travel.
    @test refinement_buffer(forest, 2, spacing(forest, 2)) == 2

    @test_throws ArgumentError refinement_buffer(forest, 2, 0.05)
    @test_throws ArgumentError refinement_buffer(forest, 6, 0.02)
end
