# The right-hand side: the three kernels, the six steps, and the two
# properties the integrator depends on.
#
# Everything here runs on the **uniform** periodic mesh
# (`hydro_forest(…; refined = false)`), where every face is a same-level
# face: the coarse-fine machinery is step 5's, and a claim about the flux
# kernel's indexing should not have to be read through an interface
# restriction.
#
# The claims are chosen so that each one fails for a *different* mistake,
# and the two that matter most are about indices rather than about physics:
#
#   * the primitives in every *stored* cell catch a `con2prim` launch
#     written over the owned range — the ghosts would hold conserved zeros,
#     the recovery would hand back the atmosphere, and the run would look
#     almost right at the block interiors;
#   * the flux against a host evaluation from the same four stored
#     primitive tuples catches an off-by-`G` between the three ghost
#     widths, which no convergence rate would name.
#
# Then purity (the integrator evaluates the same `u` more than once per
# step and the atmosphere reset must not live here), the semi-discrete
# order (cheap and sharp: the scheme's own truncation error, with no time
# stepping to blur it), the exact zero on a uniform state, and the layout
# refusals.

using KernelAbstractions: CPU, get_backend

const EVOLUTION_DIMS = (1, 2)

# The operators the conservative family wants for a second-order scheme:
# prolongation of order 3, one more than the scheme's own, as TreeAMR's
# interface-order rule has it. Nothing here refines, so only the family
# matters; it is spelled out so that the refined runs of step 5 differ from
# these in the mesh and in nothing else.
evolution_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

# A uniform periodic mesh carrying the entropy wave, its problem and its
# state vector: the smooth data every claim below is made on. `roots = 2`
# keeps the block count at the smallest number that still has a periodic
# neighbor on both sides of every block.
function evolution_setup(::Type{T}, ::Val{D}; N=8, roots=2, G=2, limiter=:none,
                         riemann=:hlle, fixup=true) where {T,D}
    w = EntropyWave(T, Val(D))
    forest = hydro_forest(Val(D), N; roots=roots, L=w.L, refined=false, T=T)
    U = FieldSet{T}(forest, D + 2; G=G)
    p = HydroProblem(U, evolution_ops(); eos=w.eos, floors=w.floors,
                     limiter=limiter, riemann=riemann, fixup=fixup)
    fill_entropywave_averages!(U, w)
    u = statevector(U)
    gather!(u, U)
    return (w=w, forest=forest, U=U, p=p, u=u)
end

# The exact time derivative of the exact cell averages, in the state
# vector's layout — the reference the semi-discrete residual is measured
# against.
#
# `∂ₜρ = −Σ_d v_d ∂_d ρ = −(Σ_d v_d) a k cos(k Σ_d x_d)` for the entropy
# wave, and since `v` and `p` are constant the momentum and energy rows are
# `v_i ∂ₜρ` and `½ v² ∂ₜρ`. The *average* of that over a cell carries the
# same damping factor as the average of `ρ` itself, so this is the exact
# derivative of the numbers the scheme actually stores and not an `O(h²)`
# approximation to it — which it would have to be, to measure an `O(h²)`
# residual with it.
function analytic_rhs(U::FieldSet{T,D}, w::EntropyWave{T,D}, t::T) where {T,D}
    forest = U.forest
    N = forest.N
    k = TreeHydro.wavenumber(w)
    vsum = sum(w.v)
    v² = sum(w.v .^ 2)
    host = zeros(T, statelength(U))
    arr = reshape(host, ntuple(_ -> N, D)..., U.nvars, nblocks(U))
    for b in 1:nblocks(U)
        damp = TreeHydro.average_damping(w, spacing(T, forest, blockkey(U, b)))
        for idx in CartesianIndices(ntuple(_ -> N, D))
            x = coordinates(T, U, b, ntuple(d -> Tuple(idx)[d] + U.G[d], D))
            s = sum(ntuple(d -> x[d] - w.v[d] * t, Val(D)))
            dρ = -vsum * w.a * k * damp * cos(k * s)
            arr[Tuple(idx)..., 1, b] = dρ
            for d in 1:D
                arr[Tuple(idx)..., 1 + d, b] = w.v[d] * dρ
            end
            arr[Tuple(idx)..., D + 2, b] = v² * dρ / 2
        end
    end
    u = statevector(U)
    copyto!(u, host)
    return u
end

# The volume-weighted L∞ residual of one right-hand-side evaluation on the
# initial data, and the spacing it belongs to.
function residual_error(::Type{T}, ::Val{D}; N, roots=4) where {T,D}
    s = evolution_setup(T, Val(D); N=N, roots=roots, limiter=:none)
    du = similar(s.u)
    hydro_rhs!(du, s.u, s.p, zero(T))
    err = du .- analytic_rhs(s.U, s.w, zero(T))
    return (linf=volume_weighted_norm(s.U, err; p=Inf), h=minimum_spacing(T, s.forest))
end

@testset "The primitives are current in every stored cell: D=$D" for
        D in (1, 2, 3)
    # Guards the launch, not the arithmetic: `con2prim` itself is tested in
    # `eos_tests.jl`, and what can go wrong here is the range. A pass over
    # the owned cells alone leaves the ghosts holding conserved *zeros*,
    # and a zero density is the atmosphere — so the floor-hit slot would be
    # `1` in every ghost cell and the reconstruction at a block's own
    # boundary face would read vacuum. Both failure modes are caught below,
    # the first by the cell-by-cell comparison over the whole stored
    # extent, the second by the flag count being zero on data whose density
    # never leaves [4/5, 6/5].
    #
    # Bit for bit, not to a tolerance: the kernel and the host loop call
    # the same function on the same numbers.
    T = Float64
    N = D == 3 ? 4 : 8
    s = evolution_setup(T, Val(D); N=N, roots=2)
    update_primitives!(s.p, s.u)
    cons, prim = Array(s.U.work), Array(s.p.P.work)
    stored = ntuple(d -> size(cons, d), D)
    @test stored == ntuple(_ -> N + 2 * s.U.G[1], D)

    worst = zero(T)
    flags = zero(T)
    for b in 1:nblocks(s.U), idx in CartesianIndices(stored)
        c = Tuple(idx)
        Ucell = ntuple(v -> cons[c..., v, b], D + 2)
        Pcell, _ = con2prim(s.w.eos, s.w.floors, Ucell)
        for v in 1:(D + 2)
            worst = max(worst, abs(prim[c..., v, b] - Pcell[v]))
        end
        worst = max(worst, abs(prim[c..., D + 3, b] - signal_speed(s.w.eos, Pcell)))
        flags += prim[c..., D + 4, b]
    end
    @test worst == 0
    @test flags == 0
    @test floor_hits(s.p) == 0
    # The signal speed is a real speed and not a leftover zero: the wave
    # carries |v| + c_s ≈ 2.3 per direction.
    @test max_signal_speed(s.p) > 1
end

@testset "The flux kernel is face_states then riemann_flux: D=$D, $lim/$rs" for
        D in EVOLUTION_DIMS, (lim, rs) in ((:none, :hlle), (:mc, :hllc))
    # Guards the off-by-`G`. Face `I[d]` of the closed range lies between
    # cells `I[d]−1` and `I[d]`; cell `i` of `P` is stored at `i + G_P` and
    # the face at `I + G_F`, and the two ghost widths differ (2 and 0), so
    # a kernel that added the wrong one reads a stencil shifted by two
    # cells and still stays in bounds. The faces chosen are the two
    # boundary faces of a block — `1` and `N+1`, the ones whose stencils
    # reach into the ghosts — and three interior ones.
    #
    # Bit for bit: the host evaluation below is the kernel's body with the
    # indices written out by hand.
    T = Float64
    N = 8
    s = evolution_setup(T, Val(D); N=N, roots=2, limiter=lim, riemann=rs)
    du = similar(s.u)
    hydro_rhs!(du, s.u, s.p, zero(T))
    prim = Array(s.p.P.work)
    GP = s.p.P.G
    for d in 1:D
        flux = Array(s.p.fluxes[d].work)
        GF = s.p.fluxes[d].G
        for b in (1, nblocks(s.U)), i in (1, 2, N ÷ 2, N, N + 1), j in (1, N)
            I = ntuple(e -> e == d ? i : j, D)
            c = ntuple(e -> I[e] + GP[e], D)
            m1 = Base.setindex(c, c[d] - 1, d)
            m2 = Base.setindex(c, c[d] - 2, d)
            p1 = Base.setindex(c, c[d] + 1, d)
            stencil = map((m2, m1, c, p1)) do idx
                ntuple(v -> prim[idx..., v, b], D + 2)
            end
            P_L, P_R = face_states(Val(lim), s.w.eos, s.w.floors, stencil...)
            F = riemann_flux(Val(rs), s.w.eos, P_L, P_R, Val(d))
            f = ntuple(e -> I[e] + GF[e], D)
            @test ntuple(v -> flux[f..., v, b], D + 2) == F
        end
    end
end

@testset "The right-hand side is pure and never mutates u: D=$D" for
        D in EVOLUTION_DIMS
    # Two claims the integrator rests on. `SSPRK33` evaluates the
    # right-hand side three times per step from three different stage
    # vectors, so a right-hand side that left something of one evaluation
    # behind in `P` or in a flux set would make the *second* step differ
    # from the first on identical data. And the atmosphere reset belongs in
    # the stage-limiter hook, not here: a right-hand side that floored `u`
    # in place would be a scheme whose conserved integral changed outside
    # the flux divergence, which is the one thing the conservation claim
    # cannot survive.
    T = Float64
    s = evolution_setup(T, Val(D); N=8, roots=2, limiter=:mc)
    before = copy(s.u)
    du1, du2 = similar(s.u), similar(s.u)
    hydro_rhs!(du1, s.u, s.p, zero(T))
    hydro_rhs!(du2, s.u, s.p, zero(T))
    @test du1 == du2
    @test s.u == before
end

@testset "The semi-discrete residual is second order" begin
    # The scheme's own truncation error, with no time stepping to blur it
    # and no reference solution to approximate: the entropy wave's exact
    # time derivative is closed form, so this measures the space
    # discretization alone and does it in a fraction of a second. A first
    # order reconstruction, a flux written at the wrong face or a
    # divergence divided by the wrong spacing all show here as a rate, and
    # the full convergence study in `entropywave_tests.jl` would only
    # repeat the finding an order of magnitude more slowly.
    T = Float64
    Ns = (8, 16, 32)
    rs = [residual_error(T, Val(1); N=N) for N in Ns]
    errs = [r.linf for r in rs]
    rate = convergence_rate([r.h for r in rs], errs)
    @info "semi-discrete residual, D = 1, :none: L∞ rate $(round(rate, digits=3))"
    @test rate ≥ 1.9
    @test issorted(errs; rev=true)
end

@testset "A uniform state has exactly zero right-hand side: D=$D" for
        D in (1, 2, 3)
    # The discrete statement that the scheme is *conservative in form*:
    # every face of every block computes the same flux from the same
    # constant state, so every difference in the divergence cancels
    # identically — not to roundoff, exactly, because it is the same
    # floating-point number subtracted from itself. A stencil that reached
    # one cell too far, a transverse index that drifted, or a flux written
    # at `I` and read at `I+1` would leave a nonzero residual at the block
    # boundaries and nowhere else, which is invisible in any norm that is
    # not exact. The velocity is nonzero so that the advective terms have
    # to cancel too.
    T = Float64
    N = D == 3 ? 4 : 8
    w = EntropyWave(T, Val(D); a=0, v=3 // 10)
    forest = hydro_forest(Val(D), N; roots=2, L=w.L, refined=false, T=T)
    U = FieldSet{T}(forest, D + 2; G=2)
    p = HydroProblem(U, evolution_ops(); eos=w.eos, floors=w.floors, limiter=:none)
    fill_entropywave_averages!(U, w)     # a = 0: the constant state
    u = statevector(U)
    gather!(u, U)
    du = similar(u)
    hydro_rhs!(du, u, p, zero(T))
    @test all(iszero, du)
end

@testset "HydroProblem refuses a layout the kernels cannot index: D=$D" for
        D in EVOLUTION_DIMS
    # Each of these is a mistake that would otherwise be found as a wrong
    # answer rather than as an error: `G = 1` reads a stencil that runs off
    # the ghosts (in bounds, wrong numbers); a primitive set with another
    # ghost width makes the `con2prim` pass write cells the flux kernel
    # does not read; a limiter or a solver that has no method would be a
    # MethodError from inside a kernel launch, with the compiler's stack
    # trace rather than a sentence.
    T = Float64
    forest = hydro_forest(Val(D), 8; roots=2, refined=false, T=T)
    ops = evolution_ops()
    w = EntropyWave(T, Val(D))
    kw = (eos=w.eos, floors=w.floors)
    U = FieldSet{T}(forest, D + 2; G=2)

    @test_throws "needs G >= 2 in every dimension" HydroProblem(
        FieldSet{T}(forest, D + 2; G=1), ops; limiter=:none, kw...)
    @test_throws "the conserved state is cell-centered" HydroProblem(
        FieldSet{T}(forest, D + 2; G=2, centering=facecentered(D, 1)), ops;
        limiter=:none, kw...)
    @test_throws "must have the conserved state's layout" HydroProblem(
        U, ops; limiter=:none, prims=FieldSet{T}(forest, D + 4; G=3), kw...)
    @test_throws "two diagnostic slots" HydroProblem(
        U, ops; limiter=:none, prims=FieldSet{T}(forest, D + 2; G=2), kw...)
    @test_throws "a flux set carries no ghosts" HydroProblem(
        U, ops; limiter=:none,
        fluxes=ntuple(d -> FieldSet{T}(forest, D + 2; G=1,
                                       centering=facecentered(D, d)), D), kw...)
    @test_throws "no default limiter" HydroProblem(U, ops; kw...)
    @test_throws "limiter must be one of" HydroProblem(U, ops; limiter=:superbee,
                                                       kw...)
    @test_throws "riemann must be one of" HydroProblem(U, ops; limiter=:none,
                                                       riemann=:roe, kw...)
    # And the one it must accept: every admissible pair, built on the CPU
    # backend, which is the whole of the device claim this step makes.
    for lim in (:none, :minmod, :mc), rs in (:llf, :hlle, :hllc)
        p = HydroProblem(U, ops; limiter=lim, riemann=rs, kw...)
        @test p.limiter === Val(lim)
        @test p.solver === Val(rs)
        @test get_backend(p.P.work) isa CPU
    end
end
