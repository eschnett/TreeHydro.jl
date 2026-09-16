# Reconstruction and the three Riemann fluxes.
#
# This is the whole of step (3) of the right-hand side except the indexing:
# the flux kernel of step 3 reads four primitive states around a face,
# calls `face_states` on them and hands the pair to `riemann_flux`. What is
# tested here is therefore every number that kernel will compute, on
# stencils handed over directly, where a failure names a formula rather
# than a convergence rate that is slightly off.
#
# The claims are chosen so that each one fails for a *different* mistake.
# Consistency catches a flux that is not a flux of the state it was given;
# direction generality catches an `ntuple` over `d` with a preferred axis;
# upwinding catches a wave-speed estimate with the wrong sign; the
# stationary contact is the claim that says what HLLC is *for*, and it is
# asserted together with HLLE's failure of it, since a solver that passed
# both would not be HLLE. The limiter claims are about total variation and
# the face-state claims about the range the reconstruction may leave.
#
# As in `eos_tests.jl`: the inputs come from a seeded generator and the
# assertions do not — every claim follows from the ranges the states are
# drawn from (`ρ ∈ [1/2, 2]`, `|v| ≤ 1`, `p ∈ [1/2, 2]`), so a fixed seed
# is a convenience and not the thing being tested. Three types for the
# three faults they catch: `Float64` the baseline, `Float32` the leak
# detector (a stray `Float64` operand widens the result, so every testset
# asserts the element type of what came back), `Float32x2` the software
# float no hardware fast path can serve. Everything here is `sqrt`, `abs`,
# `min` and `max`, all of which MultiFloats provides, so all three run all
# of it.

using KernelAbstractions: @kernel, @index, @Const, CPU, synchronize
using MultiFloats: Float32x2
using Random: MersenneTwister

const RIEMANN_FLOATTYPES = (Float64, Float32, Float32x2)
const RIEMANN_DIMS = (1, 2, 3)

# Roundoff, in the only unit that means the same thing at every type. A
# flux is a `prim2con`, a sound speed and an HLL average — a few tens of
# operations, some of which (the star state of HLLC at a near-stationary
# contact) divide a difference by a difference — so 64 ulp is the round
# number above the honest bound, as it is in `eos_tests.jl`.
riemann_rtol(::Type{T}) where {T} = 64 * eps(T)

riemann_gas(::Type{T}) where {T} = IdealGas(T(7 // 5))

# Far below every state drawn below, so that a physical face state never
# trips them: the floors are exercised on purpose in their own testset and
# must be invisible everywhere else.
riemann_floors(::Type{T}) where {T} =
    Floors{T}(; ρ_atm=T(1 // 10^6), p_atm=T(1 // 10^6), p_floor=T(1 // 10^8))

# A physical primitive state: ρ ∈ [1/2, 2], v_d ∈ [−1, 1], p ∈ [1/2, 2].
# Integers over a fixed denominator, so the three types see the same
# numbers and a disagreement between them is the code's.
function riemann_prim(rng, ::Type{T}, ::Val{D}) where {T,D}
    ρ = T(rand(rng, 500:2000) // 1000)
    v = ntuple(_ -> T(rand(rng, -1000:1000) // 1000), Val(D))
    p = T(rand(rng, 500:2000) // 1000)
    return (ρ, v..., p)
end

# Two fluxes agree to roundoff, component by component, judged on the
# scale of the *flux* rather than of each component: a momentum component
# whose exact value is zero must be compared against something, and its
# own magnitude is not it.
function test_flux_approx(F::NTuple{M,T}, G::NTuple{M,T}, rtol) where {M,T}
    scale = max(maximum(abs, F), maximum(abs, G))
    for v in 1:M
        @test isapprox(F[v], G[v]; rtol=rtol, atol=rtol * scale)
    end
    return nothing
end

# `@allocated` measures the expression as compiled, so the arguments have
# to reach it through a function rather than as captured globals; each is
# called once to compile before the measurement is believed.
face_states_allocs(lim, eos, floors, a, b, c, e) =
    @allocated face_states(lim, eos, floors, a, b, c, e)
riemann_flux_allocs(solver, eos, P_L, P_R, dir) =
    @allocated riemann_flux(solver, eos, P_L, P_R, dir)

# The flux kernel of step 3 with the mesh taken out: a batch of four-cell
# stencils in, one flux per stencil out. `eos`, `floors`, the limiter and
# the solver are kernel *arguments*, which is the `isbits` claim that lets
# the real kernel pass them on a device.
@kernel function riemann_flux_kernel!(fluxes, @Const(stencils), eos, floors,
                                      lim, solver, ::Val{M},
                                      ::Val{d}) where {M,d}
    i = @index(Global)
    P₋₂ = ntuple(v -> stencils[v, 1, i], Val(M))
    P₋₁ = ntuple(v -> stencils[v, 2, i], Val(M))
    P₀ = ntuple(v -> stencils[v, 3, i], Val(M))
    P₊₁ = ntuple(v -> stencils[v, 4, i], Val(M))
    P_L, P_R = face_states(lim, eos, floors, P₋₂, P₋₁, P₀, P₊₁)
    F = riemann_flux(solver, eos, P_L, P_R, Val(d))
    for v in 1:M
        fluxes[v, i] = F[v]
    end
end

@testset "Every solver reduces to the physical flux on one state: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # Consistency, the property that makes a numerical flux a flux of
    # *these* equations at all: where the two face states agree there is no
    # Riemann problem, and the answer is the exact Euler flux. It is the
    # single sharpest test of a flux function — a transposed slot, a
    # missing pressure term, a dissipation coefficient applied to the wrong
    # difference all survive every other claim here and fail this one.
    # LLF and HLLC reproduce it exactly (their dissipation multiplies a
    # difference that is identically zero); HLLE forms an average whose
    # weights sum to one, so it is to roundoff.
    eos = riemann_gas(T)
    rtol = riemann_rtol(T)
    rng = MersenneTwister(2100 + D)
    for _ in 1:6
        P = riemann_prim(rng, T, Val(D))
        for d in 1:D
            F = physical_flux(eos, P, Val(d))
            @test F isa NTuple{D + 2,T}
            for solver in (Val(:llf), Val(:hlle), Val(:hllc))
                G = riemann_flux(solver, eos, P, P, Val(d))
                @test G isa NTuple{D + 2,T}
                test_flux_approx(G, F, rtol)
            end
        end
    end
end

@testset "The flux has no preferred direction: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # Guards the one place a stray asymmetry can hide: the `ntuple` over
    # the momentum components and the `Base.setindex` that puts the
    # pressure — and, in HLLC, the contact speed — into the normal slot.
    # Permute the velocity components of *both* states by a cyclic shift
    # and read the flux in the direction that shift sent `d` to: the mass
    # and energy fluxes must be unchanged and the momentum fluxes must come
    # back permuted the same way. A `d` hard-wired anywhere, or a pressure
    # added to slot 2 rather than slot `1 + d`, fails this in `D = 2` and
    # `3` and is invisible in `D = 1`.
    eos = riemann_gas(T)
    rtol = riemann_rtol(T)
    # `perm[i]` is the component the permuted state's `i` reads, so the
    # permuted state's direction `dp` carries the original's `perm[dp]`.
    perm = ntuple(i -> mod1(i + 1, D), D)
    permute(P) = (density(P), ntuple(i -> velocity(P)[perm[i]], D)...,
                  pressure_of(P))
    rng = MersenneTwister(2200 + D)
    for _ in 1:6
        P_L = riemann_prim(rng, T, Val(D))
        P_R = riemann_prim(rng, T, Val(D))
        Q_L, Q_R = permute(P_L), permute(P_R)
        for d in 1:D
            dp = findfirst(==(d), perm)
            for solver in (Val(:llf), Val(:hlle), Val(:hllc))
                F = riemann_flux(solver, eos, P_L, P_R, Val(d))
                G = riemann_flux(solver, eos, Q_L, Q_R, Val(dp))
                want = (F[1], ntuple(i -> F[1 + perm[i]], D)..., F[D + 2])
                @test G isa NTuple{D + 2,T}
                test_flux_approx(G, want, rtol)
            end
        end
    end
end

@testset "A supersonic face takes its flux from one side: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # Guards the sign of Davis's wave-speed estimates and the branches that
    # read them. Where both states move right faster than sound no signal
    # can reach the face from the right, and the flux is the left state's
    # exact flux — *exactly*, not to roundoff, since both solvers return
    # that tuple rather than an average that happens to equal it. A
    # swapped `min` and `max`, or a `≤` where a `≥` belongs, turns the
    # upwind side into the downwind one and produces a scheme that is
    # unconditionally unstable, which a convergence study would report as a
    # `NaN` ten steps in.
    eos = riemann_gas(T)
    for d in 1:D
        # Normal velocity 5, sound speeds near 1: supersonic on both sides
        # with a wide margin, so the claim is about the branch and not
        # about a borderline comparison.
        fast = T(5)
        P_L = (T(1), ntuple(i -> i == d ? fast : T(1 // 4), D)..., T(1))
        P_R = (T(1 // 2), ntuple(i -> i == d ? fast - 1 : -T(1 // 4), D)...,
               T(1 // 2))
        F_L = physical_flux(eos, P_L, Val(d))
        F_R = physical_flux(eos, P_R, Val(d))
        @test riemann_flux(Val(:hlle), eos, P_L, P_R, Val(d)) === F_L
        @test riemann_flux(Val(:hllc), eos, P_L, P_R, Val(d)) === F_L

        # And the mirror image: both states moving left faster than sound,
        # where the face sees the right state alone.
        Q_L = (density(P_L), ntuple(i -> -velocity(P_L)[i], D)...,
               pressure_of(P_L))
        Q_R = (density(P_R), ntuple(i -> -velocity(P_R)[i], D)...,
               pressure_of(P_R))
        G_R = physical_flux(eos, Q_R, Val(d))
        @test riemann_flux(Val(:hlle), eos, Q_L, Q_R, Val(d)) === G_R
        @test riemann_flux(Val(:hllc), eos, Q_L, Q_R, Val(d)) === G_R
        @test F_L isa NTuple{D + 2,T}
        @test G_R isa NTuple{D + 2,T}
    end
end

@testset "HLLC resolves a stationary contact and HLLE does not: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # The claim that decides why HLLC exists, and the reason it is measured
    # against HLLE on the Kelvin–Helmholtz instability in step 10: a shear
    # layer *is* a contact.
    #
    # Two states at rest with equal pressures and unequal densities are a
    # stationary contact discontinuity — an exact steady solution of the
    # Euler equations — so the exact flux is `(0, p δ_id, 0)`: no mass
    # crosses the face, and only the pressure pushes. HLLC's middle wave is
    # that contact and it reproduces the flux to roundoff. HLLE has no
    # middle wave, so its two-wave average must smear the density jump
    # across the face, and its mass flux comes out as the closed form
    # asserted below — nonzero, and of the size of the jump itself rather
    # than of a roundoff.
    eos = riemann_gas(T)
    rtol = riemann_rtol(T)
    floors = riemann_floors(T)
    ρ_L, ρ_R, p = T(1), T(1 // 8), T(1)
    for d in 1:D
        P_L = (ρ_L, ntuple(_ -> zero(T), D)..., p)
        P_R = (ρ_R, ntuple(_ -> zero(T), D)..., p)
        exact = ntuple(v -> v == 1 + d ? p : zero(T), D + 2)

        F_hllc = riemann_flux(Val(:hllc), eos, P_L, P_R, Val(d))
        @test F_hllc isa NTuple{D + 2,T}
        test_flux_approx(F_hllc, exact, rtol)

        # HLLE's mass flux, in closed form. Both states are at rest, so
        # Davis gives `s_R = −s_L = max(c_L, c_R)` and the whole average
        # collapses to its dissipative term, `c (ρ_L − ρ_R) / 2`.
        c = max(soundspeed(eos, ρ_L, p), soundspeed(eos, ρ_R, p))
        F_hlle = riemann_flux(Val(:hlle), eos, P_L, P_R, Val(d))
        @test isapprox(density(F_hlle), c * (ρ_L - ρ_R) / 2; rtol=rtol)
        # Nonzero by a wide margin, which is the half of the claim that
        # says HLLC bought something: the mass flux here is ≈ 1.46 in the
        # units of the problem, against zero to roundoff for HLLC.
        @test abs(density(F_hlle)) > one(T)

        # The face states of this stencil are the cell states under a TVD
        # limiter — one side of each slope is flat, so the slope vanishes —
        # so the same contact reached through the reconstruction says the
        # same thing, which is how the flux kernel of step 3 will see it.
        # `:none` is excluded because it is not a limiter: its centered
        # slope extrapolates straight across the jump, which is the whole
        # reason the other two exist.
        for lim in (Val(:minmod), Val(:mc))
            Q_L, Q_R = face_states(lim, eos, floors, P_L, P_L, P_R, P_R)
            @test Q_L === P_L
            @test Q_R === P_R
        end
    end
end

@testset "LLF is at least as diffusive as HLLE: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # Guards the wave-speed estimates against each other, which no single
    # solver's own tests can: LLF applies the *fastest* speed to every
    # characteristic field, HLLE applies the slowest and the fastest to
    # their own sides, so LLF's departure from the plain average
    # `(F_L + F_R)/2` must be the larger of the two. A `λ` built from the
    # normal velocity without its absolute value, or Davis speeds that had
    # lost a `c_s`, would invert this.
    #
    # Sod's states with the left state given a rightward velocity. The
    # standard pair at rest is the degenerate case where the two solvers
    # coincide — `s_R = −s_L = λ` exactly — so the moving pair is what
    # makes this a strict inequality rather than an equality that would
    # pass whatever the code did. The comparison is on the mass flux,
    # whose diffusive term is the density jump, the largest in the problem.
    eos = riemann_gas(T)
    for d in 1:D
        P_L = (T(1), ntuple(i -> i == d ? T(1 // 2) : T(1 // 5), D)..., T(1))
        P_R = (T(1 // 8), ntuple(i -> i == d ? zero(T) : -T(1 // 5), D)...,
               T(1 // 10))
        F_L = physical_flux(eos, P_L, Val(d))
        F_R = physical_flux(eos, P_R, Val(d))
        central = ntuple(v -> (F_L[v] + F_R[v]) / 2, D + 2)
        F_llf = riemann_flux(Val(:llf), eos, P_L, P_R, Val(d))
        F_hlle = riemann_flux(Val(:hlle), eos, P_L, P_R, Val(d))
        @test abs(density(F_llf) - density(central)) >
              abs(density(F_hlle) - density(central))
        @test F_llf isa NTuple{D + 2,T}
    end
end

@testset "The limiters limit what their names say: T=$T" for
        T in RIEMANN_FLOATTYPES
    # Guards each of the three slopes and the two properties that make the
    # two limited ones total-variation-diminishing: they vanish where the
    # one-sided differences disagree in sign — the extremum, where an
    # unlimited slope creates a new one — and they never exceed the
    # steepness that would let a face value leave the range of its
    # neighbours. `:mc`'s whole reason for existing is that it is the
    # centered slope wherever that is not too steep, so that is asserted
    # both ways: it *is* the centered slope where the centered slope is the
    # smallest of the three, and it is `2a` or `2b` where it is not.
    #
    # Symmetry and oddness are asserted with `==` and not `===` because a
    # limiter that returns `zero(a)` returns `+0.0` where the negation of
    # its mirror image is `−0.0`; the two are equal numbers and distinct
    # bit patterns, and nothing downstream can tell them apart.
    lims = (Val(:none), Val(:minmod), Val(:mc))
    rng = MersenneTwister(2300)
    pairs = [(T(rand(rng, -2000:2000) // 1000), T(rand(rng, -2000:2000) // 1000))
             for _ in 1:24]
    # The cases a random draw will not produce: an extremum with one
    # side flat, a symmetric extremum, and two equal differences.
    append!(pairs, [(zero(T), one(T)), (one(T), zero(T)), (one(T), -one(T)),
                    (one(T), one(T)), (-T(3 // 2), -T(3 // 2))])
    for (a, b) in pairs
        centered = (a + b) / 2
        @test slope(Val(:none), a, b) === centered
        for lim in lims
            σ = slope(lim, a, b)
            @test σ isa T
            @test slope(lim, b, a) == σ                       # symmetric
            @test slope(lim, -a, -b) == -σ                    # odd
        end
        σm = slope(Val(:minmod), a, b)
        σc = slope(Val(:mc), a, b)
        if a * b ≤ zero(T)
            # An extremum: both limiters give a piecewise-constant cell,
            # which is what stops the reconstruction creating a new
            # extremum where the data had one.
            @test σm == zero(T)
            @test σc == zero(T)
        else
            @test abs(σm) == min(abs(a), abs(b))              # the smaller one
            @test σm == (abs(a) < abs(b) ? a : b)
            @test abs(σc) ≤ 2 * min(abs(a), abs(b))           # the TVD bound
            @test abs(σc) ≥ abs(σm)                           # less diffusive
            @test σc * a > zero(T)                            # the common sign
            @test σc == (abs(centered) ≤ 2 * min(abs(a), abs(b)) ?
                         centered : 2 * (abs(a) < abs(b) ? a : b))
        end
    end
    # Two named cases, so that the branch each one exercises is on the
    # record rather than left to the draw: the centered slope wins where it
    # is shallow, and `2a` wins where it is not.
    @test slope(Val(:mc), T(1), T(3 // 2)) === T(5 // 4)       # centered
    @test slope(Val(:mc), T(1 // 8), T(2)) === T(1 // 4)       # 2a
end

@testset "A TVD limiter keeps face states between the cells: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # The property the whole scheme's robustness rests on, and the reason
    # `face_states` needs no floor-hit count under `:minmod` or `:mc`: both
    # reconstructed states at a face lie, componentwise, in the closed
    # interval spanned by the two cells that face separates. Positive
    # densities and pressures on both sides therefore give a positive
    # density and pressure at the face, so the floors cannot fire on a
    # physical stencil — which is what makes their hits here uncounted
    # rather than unnoticed.
    #
    # Random stencils are non-monotone almost surely, which is the case a
    # limiter is *for*; monotone ones are appended on purpose, since that
    # is the case where the limit is attained and an off-by-a-factor-of-two
    # in the `σ/2` would be visible.
    eos = riemann_gas(T)
    floors = riemann_floors(T)
    rtol = riemann_rtol(T)
    M = D + 2
    rng = MersenneTwister(2400 + D)
    stencils = [ntuple(_ -> riemann_prim(rng, T, Val(D)), 4) for _ in 1:8]
    # Monotone in every component: each state is the previous one plus a
    # fixed positive step, and the reversed stencil for the other sign.
    up = ntuple(k -> ntuple(v -> T(1 + k) * T(1 // 3), Val(M)), 4)
    push!(stencils, up)
    push!(stencils, reverse(up))
    for lim in (Val(:minmod), Val(:mc)), st in stencils
        P_L, P_R = face_states(lim, eos, floors, st...)
        @test P_L isa NTuple{M,T}
        @test P_R isa NTuple{M,T}
        for v in 1:M
            lo = min(st[2][v], st[3][v])
            hi = max(st[2][v], st[3][v])
            # A ulp of slack at each end: the limit is attained exactly in
            # exact arithmetic, and `P₋₁ + (P₀ − P₋₁)` is `P₀` only to
            # roundoff.
            slack = rtol * max(abs(lo), abs(hi))
            @test lo - slack ≤ P_L[v] ≤ hi + slack
            @test lo - slack ≤ P_R[v] ≤ hi + slack
        end
    end
end

@testset "An unlimited face state that leaves the gas is floored: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # The other half of the same claim, and the reason `face_states` calls
    # `apply_floors` at all. `:none` is not a limiter: on a stencil whose
    # pressure rises steeply just past the face, the centered slope
    # extrapolates the right state to a *negative* pressure, from which
    # `soundspeed` would return a `NaN` and the flux would poison the
    # block. The claim is that what comes back is exactly the floored
    # unlimited reconstruction — the floors applied, and nothing else
    # quietly changed on the way.
    eos = riemann_gas(T)
    floors = riemann_floors(T)
    M = D + 2
    small, big = T(1 // 100), T(10)
    flat = ntuple(_ -> zero(T), D)
    # `P_R = P₀ − (P₊₁ − P₋₁)/4`, so a pressure of 1/100 at the face and 10
    # one cell past it puts the reconstructed pressure near −5/2.
    st = ((one(T), flat..., small), (one(T), flat..., small),
          (one(T), flat..., small), (one(T), flat..., big))
    σ_L = ntuple(v -> slope(Val(:none), st[2][v] - st[1][v], st[3][v] - st[2][v]),
                 Val(M))
    σ_R = ntuple(v -> slope(Val(:none), st[3][v] - st[2][v], st[4][v] - st[3][v]),
                 Val(M))
    raw_L = ntuple(v -> st[2][v] + σ_L[v] / 2, Val(M))
    raw_R = ntuple(v -> st[3][v] - σ_R[v] / 2, Val(M))
    @test pressure_of(raw_R) < zero(T)          # the stencil really does this
    @test pressure_of(raw_L) ≥ floors.p_floor   # and the left state does not

    P_L, P_R = face_states(Val(:none), eos, floors, st...)
    # Bit for bit: `face_states` is exactly this arithmetic followed by
    # exactly this floor, so a tolerance here would hide a reordering.
    @test P_L === first(apply_floors(eos, floors, raw_L))
    @test P_R === first(apply_floors(eos, floors, raw_R))
    @test P_L === raw_L                          # nothing fired on the left
    @test pressure_of(P_R) === floors.p_floor    # and the floor fired on the right
    @test density(P_R) === density(raw_R)        # only the pressure moved
    @test P_R isa NTuple{M,T}

    # And the flux built from the floored pair is a finite number, which is
    # the whole point: the unfloored one would carry a `NaN` sound speed
    # into every cell the divergence touches.
    for solver in (Val(:llf), Val(:hlle), Val(:hllc)), d in 1:D
        @test all(isfinite, riemann_flux(solver, eos, P_L, P_R, Val(d)))
    end
end

@testset "The signal speed is the fastest wave in the cell: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # Guards the number that sets the time step of the whole hierarchy. It
    # is `max_d (|v_d| + c_s)` and not `|v| + c_s`: a missing `abs` would
    # make a leftward-moving cell report a small speed and the run would go
    # unstable where the flow reversed, which is the failure that looks
    # like a bug in the initial data.
    eos = riemann_gas(T)
    rng = MersenneTwister(2500 + D)
    for _ in 1:8
        P = riemann_prim(rng, T, Val(D))
        c = soundspeed(eos, density(P), pressure_of(P))
        λ = signal_speed(eos, P)
        @test λ isa T
        @test λ === maximum(abs.(velocity(P)) .+ c)
        @test λ ≥ c                                   # a cell at rest still speaks
    end
    # A cell whose fastest direction is not the first, so that the maximum
    # is a maximum and not `ntuple`'s first entry.
    v = ntuple(i -> T(i) / 2, D)
    P = (one(T), v..., one(T))
    @test signal_speed(eos, P) === abs(v[D]) + soundspeed(eos, one(T), one(T))
end

@testset "Reconstruction and the flux run inside a kernel: T=$T, D=$D" for
        T in RIEMANN_FLOATTYPES, D in RIEMANN_DIMS
    # The `isbits` claim made real, for the two functions the flux kernel
    # of step 3 is made of. `eos`, `floors`, the limiter `Val` and the
    # solver `Val` are kernel *arguments*, not values a host closure
    # reaches around the launch for, which is how the real kernel will pass
    # them on a device; anything holding a `Type`, an array or an abstract
    # field would fail to launch rather than fail a comparison.
    #
    # Bit for bit against the host loop, not to roundoff: the same
    # arithmetic on the same values has one answer, and a tolerance would
    # hide exactly the reassociation that costs bit-identity across
    # backends and thread counts.
    #
    # The inference and allocation checks belong here for the same reason:
    # a per-cell function that allocates cannot run in a kernel at all, and
    # one whose return type is not inferred allocates.
    eos = riemann_gas(T)
    floors = riemann_floors(T)
    M = D + 2
    lim = Val(:mc)
    solver = Val(:hlle)
    d = D                                   # the last direction, not the first
    rng = MersenneTwister(2600 + D)
    inputs = [ntuple(_ -> riemann_prim(rng, T, Val(D)), 4) for _ in 1:8]
    n = length(inputs)

    stencils = Array{T,3}(undef, M, 4, n)
    for i in 1:n, k in 1:4, v in 1:M
        stencils[v, k, i] = inputs[i][k][v]
    end
    fluxes = Matrix{T}(undef, M, n)

    riemann_flux_kernel!(CPU(), 4)(fluxes, stencils, eos, floors, lim, solver,
                                   Val(M), Val(d); ndrange=n)
    synchronize(CPU())

    for i in 1:n
        P_L, P_R = face_states(lim, eos, floors, inputs[i]...)
        F = riemann_flux(solver, eos, P_L, P_R, Val(d))
        for v in 1:M
            @test fluxes[v, i] === F[v]
        end
    end

    # Written out rather than splatted: `@inferred` inspects the call
    # expression it is handed, and a `...` in it is not one it can read.
    a, b, c, e = inputs[1]
    @test (@inferred face_states(lim, eos, floors, a, b, c, e)) isa
          Tuple{NTuple{M,T},NTuple{M,T}}
    face_states_allocs(lim, eos, floors, a, b, c, e)     # compile, then measure
    @test face_states_allocs(lim, eos, floors, a, b, c, e) == 0
    P_L, P_R = face_states(lim, eos, floors, a, b, c, e)
    for sol in (Val(:llf), Val(:hlle), Val(:hllc))
        @test (@inferred riemann_flux(sol, eos, P_L, P_R, Val(d))) isa NTuple{M,T}
        riemann_flux_allocs(sol, eos, P_L, P_R, Val(d))
        @test riemann_flux_allocs(sol, eos, P_L, P_R, Val(d)) == 0
    end
    @test (@inferred physical_flux(eos, P_L, Val(d))) isa NTuple{M,T}
    @test (@inferred signal_speed(eos, P_L)) isa T
end
