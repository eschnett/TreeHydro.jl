# Reflecting walls: TreeAMR's mirrored faces (M10), which the ghost schedule
# fills itself with a sign per variable, wired through this package's field
# sets as a parity table — and the Kelvin–Helmholtz half box they make
# possible, the lower half of the shear layer between two mirrors.
#
# The failure modes this file guards, one per testset:
#
#   * a parity table with a sign in the wrong slot — a wall that reverses the
#     tangential momentum, or the pressure's flux — which would still run,
#     and would be some other wall than a mirror;
#   * a field set built without its parity on one of the paths a run takes,
#     which TreeAMR refuses at the constructor, so the first case that needed
#     it would fail there rather than here;
#   * a half box that is not the full box's lower half: a seed that is not
#     odd in the mirror, a wall in the wrong place, a mesh that refines
#     differently on the two sides;
#   * a wall that leaks mass or energy through the face it mirrors;
#   * a half box accepted under a seed it does not stand for.
#
# The numbers in the comments are in `CODE.md` under "Reflecting walls,
# measured"; a changed number is a regression.

const REFL_OPS = Operators(family=Conservative, prolongation=3, restriction=2)

# The shear layer of `kelvinhelmholtz_tests.jl` — the same blocks, cadence and
# thresholds — under the mirrored seed and cut short at `t = 1/5`, inside the
# phase where the seeded mode is still shedding its transient: the claim is
# about the wall, and a roundoff difference is still roundoff there rather
# than an instability's amplification of one.
const REFL_KH = (N=8, roots=4, chunk=1 // 200, t_end=1 // 5,
                 refine_tol=2 // 25, coarsen_tol=1 // 50)

refl_kh(; half, cap) =
    kh_run(Float64, Val(2); N=REFL_KH.N, ops=REFL_OPS, chunk=REFL_KH.chunk,
           maxlevel_cap=cap, refine_tol=REFL_KH.refine_tol,
           coarsen_tol=REFL_KH.coarsen_tol, t_end=REFL_KH.t_end,
           roots=REFL_KH.roots, seed=:mirrored, half=half)

const HALF_KH_UNIFORM = refl_kh(; half=true, cap=0)
const FULL_KH_UNIFORM = refl_kh(; half=false, cap=0)
const HALF_KH_TRACKED = refl_kh(; half=true, cap=2)
const FULL_KH_TRACKED = refl_kh(; half=false, cap=2)

# Every owned cell of a conserved set, keyed by its centre. The centres of
# the half box are the full box's lower half *exactly* — the same root
# blocks, the same `block_origin` arithmetic — so the key is the cell.
function refl_cells(U)
    cells = Dict{NTuple{2,Float64},NTuple{4,Float64}}()
    N = U.forest.N
    for b in 1:nblocks(U)
        for idx in CartesianIndices(ntuple(e -> (U.G[e] + 1):(U.G[e] + N), 2))
            x = coordinates(Float64, U, b, Tuple(idx))
            cells[(x[1], x[2])] = ntuple(v -> U.work[Tuple(idx)..., v, b], 4)
        end
    end
    return cells
end

# The blocks of a forest as `(level, extent)`, the part of the mesh a mirror
# must reproduce; on the full box only those of its lower half.
refl_blocks(forest; below=Inf) =
    Set((level(k), block_extent(forest, k)) for k in forest.leaves
        if block_extent(forest, k)[2][2] ≤ below)

# The worst difference per variable between the half box and the full box's
# lower half, and the worst difference per variable between the full box's
# lower half and its own mirror image — `S_y` reversed, since it is the
# normal momentum.
function refl_gaps(half, full)
    h, f = refl_cells(half.r.U), refl_cells(full.r.U)
    gap = zeros(4)
    asym = zeros(4)
    for (x, v) in h
        gap .= max.(gap, abs.(v .- f[x]))
        w = f[(x[1], 1 - x[2])]
        asym .= max.(asym, abs.(f[x] .- (w[1], w[2], -w[3], w[4])))
    end
    return (; gap, asym, n=length(h), nfull=length(f))
end

# The roundoff bound on a difference of two states that took the same
# `nsteps` steps: the form `kelvinhelmholtz_tests.jl` puts on a drift, with the
# largest value of the variable in place of an integral.
refl_bound(r, v) = 8 * eps(Float64) * r.r.nsteps *
                   maximum(abs(c[v]) for c in values(refl_cells(r.r.U)))

@testset "The parity of every hydro variable is the mirror's" begin
    # A mirror normal to `e` reverses the `e` component of a vector and
    # nothing else, so exactly the slot `1 + e` is odd in dimension `e` — in
    # the conserved state and in the primitive set, whose two diagnostic
    # slots are scalars. A flux takes the product of its variable's parity
    # and its face normal's: through a face normal to `d`, the mass flux is
    # odd across a wall normal to `d` and `ρ v_d² + p` is even.
    for D in 1:3
        forest = Forest{Float64}(ntuple(_ -> 2, D); N=8,
                                 reflecting=ntuple(_ -> (true, true), D))
        for nvars in (D + 2, D + 4)
            par = state_parity(forest, nvars)
            @test length(par) == nvars
            for v in 1:nvars, e in 1:D
                @test par[v][e] == (v == 1 + e ? OddParity : EvenParity)
            end
        end
        for d in 1:D
            par = flux_parity(forest, d)
            @test length(par) == D + 2
            for v in 1:(D + 2), e in 1:D
                odd = (v == 1 + e) ⊻ (d == e)
                @test par[v][e] == (odd ? OddParity : EvenParity)
            end
            @test par[1][d] == OddParity          # the mass flux through the wall
            @test par[1 + d][d] == EvenParity     # ρ v_d² + p, the pressure's
        end
        @test_throws ArgumentError state_parity(forest, D + 3)

        # A forest with no reflecting face asks for none, and the field sets
        # of every other case are built exactly as before.
        plain = Forest{Float64}(ntuple(_ -> 2, D); N=8)
        @test state_parity(plain, D + 2) === nothing
        @test flux_parity(plain, 1) === nothing
    end
end

@testset "A run over a reflecting forest builds every field set it needs" begin
    # TreeAMR refuses a field set over a reflecting forest without a parity,
    # so each path that builds one would fail at its constructor: `evolve!`'s
    # `U`, the adaptation cycle's scratch primitives, the problem's `P` and
    # fluxes after every regrid. The tracked run above took all of them; here
    # the parities they were built with are the table's.
    r = HALF_KH_TRACKED.r
    forest = r.forest
    @test forest.reflecting == ((false, false), (true, true))
    @test forest.periodic == (true, false)
    @test r.U.parity == state_parity(forest, 4)
    U = FieldSet{Float64}(forest, 4; G=2, parity=state_parity(forest, 4))
    p = HydroProblem(U, REFL_OPS; eos=IdealGas(5 // 3),
                     floors=Floors{Float64}(; ρ_atm=1e-8, p_atm=1e-8, p_floor=1e-8),
                     limiter=:minmod, riemann=:hllc)
    @test p.P.parity == state_parity(forest, 6)
    @test all(d -> p.fluxes[d].parity == flux_parity(forest, d), 1:2)
    # And `hostcopy`'s device path, which builds a host field set of the same
    # layout: over a reflecting forest that layout includes the parity.
    host = FieldSet{Float64}(U.forest, U.nvars; G=U.G, centering=U.centering,
                             parity=U.parity)
    @test TreeHydro.hostcopy!(host, U).parity == U.parity
end

@testset "The half box is the full box's lower half, to roundoff" begin
    # The claim is that two mirrors at `y = 0` and `y = ½` stand for the upper
    # half of the periodic box, under a seed that makes the full box its own
    # mirror image. What cannot be claimed is bit-identity, and the reason is
    # measured rather than assumed: **the full box is not exactly its own
    # mirror image**. Its initial data is (the seed is written to be odd
    # exactly), but the HLLC star state sums `A + X_L − X_R` in one order at a
    # face and its mirror image in another, and floating-point addition does
    # not reassociate — so the full box drifts from its own mirror image by a
    # few ulp per step while the half box, whose mirror is exact, does not.
    # The half box differs from the full box by as much as the full box
    # differs from itself, and both sit two orders below the roundoff bound.
    #
    # Measured (t = 1/5), worst over the four variables:
    #   uniform 32², 120 steps: gap 4.0e-15, the full box's own asymmetry 3.6e-15
    #   tracked cap 2, 360 steps: gap 7.5e-15, own asymmetry 6.4e-15
    # against roundoff bounds of 4.3e-13 and 1.3e-12 in the density; `M`
    # agrees to 4.2e-17 and 3.9e-16. A parity with a sign in the wrong slot
    # would put an O(1) difference at the walls instead.
    for (half, full, label) in ((HALF_KH_UNIFORM, FULL_KH_UNIFORM, "uniform"),
                                (HALF_KH_TRACKED, FULL_KH_TRACKED, "tracked"))
        # The same steps and the same mesh, exactly: `λ_max` and the
        # indicator's references are maxima, which do not reassociate.
        @test half.r.nsteps == full.r.nsteps
        @test half.r.nchunks == full.r.nchunks
        @test refl_blocks(half.r.forest) == refl_blocks(full.r.forest; below=0.5)
        @test 2 * half.r.nblocks == full.r.nblocks
        @test half.nbs .* 2 == full.nbs

        g = refl_gaps(half, full)
        @test 2 * g.n == g.nfull
        for v in 1:4
            @test g.gap[v] ≤ refl_bound(half, v)
            @test g.asym[v] ≤ refl_bound(full, v)
        end
        # `M` reads the lower interface directly and the upper one through
        # its mirror, `v_y` reversed, so the two boxes report one amplitude.
        a = 1 // 100
        @test all(abs.(half.Ms .- full.Ms) .≤
                  8 * eps(Float64) * half.r.nsteps * a)
        @info "Kelvin–Helmholtz half box against the full box ($label, " *
              "$(half.r.nsteps) steps, $(half.r.nblocks) blocks): worst gap " *
              "$(g.gap), the full box's own mirror asymmetry $(g.asym), " *
              "bounds $(ntuple(v -> refl_bound(half, v), 4)), worst M difference " *
              "$(maximum(abs.(half.Ms .- full.Ms)))"
    end
end

@testset "A reflecting wall conserves mass, x momentum and energy" begin
    # A mirror has no flux of mass or energy through it — the mirrored state
    # has the interior's density and energy and the reversed normal velocity,
    # so the two states' mass fluxes cancel in the Riemann solver — and none
    # of tangential momentum either. The normal momentum is **not**
    # conserved and is not asserted: a wall pushes on the gas with its
    # pressure, and the two walls' pressures differ as soon as the layer
    # moves.
    for x in (HALF_KH_UNIFORM, HALF_KH_TRACKED)
        rb(v) = 8 * eps(Float64) * x.r.scales[v] * x.r.nsteps
        for v in (1, 2, 4)
            @test x.r.drift[v] ≤ rb(v)
        end
        @test (x.r.floor_hits, x.r.reset_hits, x.r.ghost_hits) == (0, 0, 0)
        @test x.r.injection == (0.0, 0.0, 0.0, 0.0)
    end
end

@testset "A half box is refused under a seed it does not stand for" begin
    # McNally's seed is even in `y = ½`, where the velocity normal to the
    # mirror must be odd: the wall would force `v_y = 0` on a line where the
    # paper's flow has none, which is a different problem that looks like
    # this one. The refusal says so.
    err = try
        HydroCase(KelvinHelmholtz(Float64, Val(2)); half=true)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("seed = :mirrored", err.msg)
    @test_throws ArgumentError HydroCase(KelvinHelmholtz(Float64, Val(2);
                                                         seed=:mirrored);
                                         roots=3, half=true)
    @test_throws ArgumentError KelvinHelmholtz(Float64, Val(2); seed=:other)

    # The mirrored seed is odd in both mirror lines and is McNally's mode at
    # the lower interface.
    w = KelvinHelmholtz(Float64, Val(2); seed=:mirrored)
    m = KelvinHelmholtz(Float64, Val(2))
    for x in range(0, 1; length=17), y in (1 // 16, 3 // 16, 5 // 16, 7 // 16)
        s = kh_state(w, (x, Float64(y)))
        t = kh_state(w, (x, Float64(1 - y)))
        @test s[3] == -t[3]
        @test (s[1], s[2], s[4]) == (t[1], t[2], t[4])
    end
    @test kh_state(w, (0.125, 0.25)) == kh_state(m, (0.125, 0.25))
    @test kh_state(w, (0.3, 0.0))[3] == 0.0

    # A case that is both periodic and reflecting in one dimension, and one
    # with an outer face and no hook, are refused where the case is built.
    eos = IdealGas(5 // 3)
    floors = Floors{Float64}(; ρ_atm=1e-8, p_atm=1e-8, p_floor=1e-8)
    base = (; initial=x -> (1.0, 0.0, 0.0, 1.0), eos=eos, floors=floors,
            extents=((0, 1), (0, 1)), roots=2)
    @test_throws ArgumentError HydroCase(Float64, Val(2); base...,
                                         periodic=(true, true),
                                         reflecting=((false, false), (true, true)))
    @test_throws ArgumentError HydroCase(Float64, Val(2); base...,
                                         periodic=(true, false),
                                         reflecting=((false, false), (true, false)))
    @test HydroCase(Float64, Val(2); base..., periodic=(true, false),
                    reflecting=((false, false), (true, true))).boundary === nothing
end
