# The equation of state, the two state conversions, and the floors.
#
# This is the whole of the physics at step 1, and everything the scheme
# adds later sits on top of it: the reconstruction floors its face states
# with the same function, the flux kernel recovers primitives with the
# same `con2prim`, and the atmosphere reset is `con2prim`, `apply_floors`,
# `prim2con` in a row. A sign error here would be found again in every
# convergence table in the package, as a rate that is nearly right.
#
# The inputs are drawn from a seeded generator; the assertions are not.
# Every claim below follows from the *ranges* the states are drawn from —
# `ρ > 0`, `p > 0`, `|v| ≤ 1` — and would hold for any draw, which is why
# a fixed seed is a convenience here and not the thing being tested.
#
# Three types, for the three faults they each catch (see
# `precision_tests.jl`): `Float64` is the baseline, `Float32` is the leak
# detector — a stray `Float64` operand widens the result, so a returned
# `Float64` names the leak — and `Float32x2` is the software float that no
# hardware fast path can serve, which is what finds a `Base` method
# MultiFloats lacks. Tolerances are stated in units of `eps(T)` so that
# the same assertion means the same thing at all three.

using KernelAbstractions: @kernel, @index, CPU, synchronize
using MultiFloats: Float32x2
using Random: MersenneTwister

const EOS_FLOATTYPES = (Float64, Float32, Float32x2)
const EOS_DIMS = (1, 2, 3)

# Roundoff, in the only unit that means the same thing at every type. The
# round trip is a handful of operations, two of which (the kinetic energy
# in and out of `E`) cancel against a quantity of their own size, so a few
# tens of ulp is the honest bound and 64 is the round number above it.
eos_rtol(::Type{T}) where {T} = 64 * eps(T)

# Sod's gas, since it is the one with a number in `CODE.md`. Written as a
# rational converted to `T`, never as a decimal literal: at `Float64` the
# two are bit-identical, and that is what lets a measured number stay put
# when the driver goes generic.
eos_gas(::Type{T}) where {T} = IdealGas(T(7 // 5))

# Floors far below the states drawn below, so that a physical state never
# trips them by accident and a test that expects `hit == false` is
# asserting the rules and not the margin.
eos_floors(::Type{T}) where {T} =
    Floors{T}(; ρ_atm=T(1 // 10^6), p_atm=T(1 // 10^6), p_floor=T(1 // 10^8))

# A physical primitive state: ρ ∈ [1/2, 2], v_d ∈ [−1, 1], p ∈ [1/2, 2].
# Drawn as integers over a fixed denominator rather than with `rand(T)`,
# so that the three types see the *same* numbers and a disagreement
# between them is the code's and not the generator's.
function random_prim(rng, ::Type{T}, ::Val{D}) where {T,D}
    ρ = T(rand(rng, 500:2000) // 1000)
    v = ntuple(_ -> T(rand(rng, -1000:1000) // 1000), Val(D))
    p = T(rand(rng, 500:2000) // 1000)
    return (ρ, v..., p)
end

# The round trip, written once so that the kernel below and the host loop
# it is compared against cannot drift apart.
eos_roundtrip(eos, floors, P) = con2prim(eos, floors, prim2con(eos, P))

@kernel function eos_roundtrip_kernel!(prims, hits, @Const(states), eos, floors,
                                       ::Val{M}) where {M}
    i = @index(Global)
    P = ntuple(v -> states[v, i], Val(M))
    Q, hit = eos_roundtrip(eos, floors, P)
    for v in 1:M
        prims[v, i] = Q[v]
    end
    hits[i] = hit
end

@testset "con2prim recovers what prim2con made: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # Guards a sign, a factor of two or a transposed slot in either
    # conversion, and the leak that a `Float64` constant in one of them
    # would be. Nothing else in the package checks that `E` and `p` are
    # the same physical state written twice; every later test would see an
    # error here only as a convergence rate that is slightly off.
    eos = eos_gas(T)
    floors = eos_floors(T)
    rtol = eos_rtol(T)
    rng = MersenneTwister(1100 + D)
    for _ in 1:8
        P = random_prim(rng, T, Val(D))
        U = prim2con(eos, P)
        Q, hit = con2prim(eos, floors, U)
        @test U isa NTuple{D + 2,T}
        @test Q isa NTuple{D + 2,T}
        @test !hit                                  # a physical state floors nothing
        @test density(U) === density(P)             # ρ is the same slot, untouched
        for v in 1:(D + 2)
            @test isapprox(Q[v], P[v]; rtol=rtol)
        end
    end
end

@testset "prim2con recovers what con2prim made: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # The other direction, which is not the same claim: the recovery
    # subtracts the kinetic energy from `E` and the conversion adds it
    # back, so a round trip through the *conserved* state is where a
    # cancellation would show. The state vector is conserved, so this is
    # the direction the evolution actually travels.
    eos = eos_gas(T)
    floors = eos_floors(T)
    rtol = eos_rtol(T)
    rng = MersenneTwister(1200 + D)
    for _ in 1:8
        U = prim2con(eos, random_prim(rng, T, Val(D)))   # physical by construction
        P, hit = con2prim(eos, floors, U)
        V = prim2con(eos, P)
        @test V isa NTuple{D + 2,T}
        @test !hit
        for v in 1:(D + 2)
            @test isapprox(V[v], U[v]; rtol=rtol)
        end
    end
end

@testset "A cell below the atmosphere density is replaced whole: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # Guards the atmosphere rule and the order it is applied in. The three
    # densities below are the three ways a cell can be vacuum — thin,
    # empty, and negative — and the last two are the reason the test comes
    # *before* the division: `S/ρ` at `ρ = 0` is an `Inf` that no later
    # comparison can undo, and the velocity it would leave behind is what
    # sets the time step of the whole hierarchy.
    eos = eos_gas(T)
    floors = eos_floors(T)
    atm = (floors.ρ_atm, ntuple(_ -> zero(T), Val(D))..., floors.p_atm)
    for ρ in (T(1 // 10^9), zero(T), -one(T))
        U = (ρ, ntuple(d -> T(d), Val(D))..., T(5))
        P, hit = con2prim(eos, floors, U)
        @test hit
        @test P === atm                              # exactly, not to roundoff
        @test all(iszero, velocity(P))               # the velocity really is gone
        @test all(isfinite, P)                       # no Inf from a division
        @test P isa NTuple{D + 2,T}
    end

    # And a NaN density is vacuum too, which is a claim about how the
    # comparison is written rather than about the rule: `ρ < ρ_atm` is
    # false for a NaN, so the naive spelling would pass it through as
    # healthy gas.
    U = (T(NaN), ntuple(d -> T(d), Val(D))..., T(5))
    P, hit = con2prim(eos, floors, U)
    @test hit
    @test P === atm
    @test all(isfinite, P)
end

@testset "A NaN anywhere in a conserved state is flagged: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # Guards every floor comparison against being written the natural way
    # round. A NaN fails `<` and `≥` alike, so a rule spelled `p < p_floor`
    # would report a state full of NaNs as needing no floor at all — the
    # one outcome the floors exist to make impossible. `hit` is the claim;
    # what the state contains afterwards is documented on `apply_floors`.
    eos = eos_gas(T)
    floors = eos_floors(T)
    nan = T(NaN)
    for slot in 1:(D + 2)
        U = ntuple(v -> v == slot ? nan : one(T), Val(D + 2))
        _, hit = con2prim(eos, floors, U)
        @test hit
    end
end

@testset "A cell with no internal energy left is floored: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # Guards the second rule and its boundary with the first. The state is
    # built so that the recovery *must* produce a negative internal
    # energy: the kinetic energy ½S·S/ρ = D/2 exceeds the total energy
    # E = 1/10, which is what a strong shock prolongated into a fine ghost
    # cell does for real. The claim is that ρ and v come back untouched —
    # only `E` moves — and that this is not the atmosphere: the cell has
    # gas in it.
    eos = eos_gas(T)
    floors = eos_floors(T)
    ρ = one(T)
    v = ntuple(_ -> one(T), Val(D))
    U = (ρ, v..., T(1 // 10))
    P, hit = con2prim(eos, floors, U)
    @test hit
    @test density(P) === ρ                            # exactly
    @test velocity(P) === v                           # exactly: S/ρ at ρ = 1
    @test pressure_of(P) === floors.p_floor
    @test density(P) ≥ floors.ρ_atm                   # gas, not atmosphere
    @test P isa NTuple{D + 2,T}

    # The same rule reached directly, on a primitive state whose pressure
    # is merely too small rather than negative.
    Q, qhit = apply_floors(eos, floors, (ρ, v..., T(1 // 10^12)))
    @test qhit
    @test Q === (ρ, v..., floors.p_floor)
end

@testset "Applying the floors twice is applying them once: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # Guards the property the atmosphere reset of step 8 rests on: a
    # floored state is a fixed point, so a reset that runs after every
    # Runge-Kutta stage cannot walk a cell away from where the previous
    # one put it. It holds bit for bit and not merely to roundoff, because
    # both rules *select* a state rather than computing one — and because
    # `Floors` refuses a `p_atm` below `p_floor`, which is the one
    # configuration in which the atmosphere state would itself be floored.
    eos = eos_gas(T)
    floors = eos_floors(T)
    rng = MersenneTwister(1300 + D)
    states = (random_prim(rng, T, Val(D)),                          # untouched
              (T(1 // 10^9), ntuple(_ -> one(T), Val(D))..., one(T)),   # atmosphere
              (one(T), ntuple(_ -> one(T), Val(D))..., T(1 // 10^12)))  # floored
    for P in states
        P1, hit1 = apply_floors(eos, floors, P)
        P2, hit2 = apply_floors(eos, floors, P1)
        @test P2 === P1
        @test !hit2                                   # already floored, nothing to do
        @test hit1 == (P1 !== P)
    end
end

@testset "A floored state survives the conserved round trip: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # The other half of what step 8 needs: `con2prim` of the conserved
    # state built from a floored primitive one reproduces that primitive
    # state, so `U` and `P` agree after a reset.
    #
    # Floors of the same order as the state, deliberately. The recovery
    # forms `ε = (E − ½S·S/ρ)/ρ`, so a pressure floor many orders below
    # the kinetic energy is recovered through a cancellation that costs
    # relative accuracy in proportion — a property of the Newtonian
    # recovery, not of the floors, and not what this test is about.
    eos = eos_gas(T)
    floors = Floors{T}(; ρ_atm=T(1 // 10), p_atm=T(1 // 4), p_floor=T(1 // 8))
    rtol = eos_rtol(T)
    half = T(1 // 2)

    # The atmosphere state, whose pressure sits well above the floor, so
    # the flag is a claim and not a coin flip: nothing fires the second
    # time round.
    P′, hit = apply_floors(eos, floors,
                           (T(1 // 100), ntuple(_ -> one(T), Val(D))..., one(T)))
    @test hit
    Q, qhit = con2prim(eos, floors, prim2con(eos, P′))
    @test !qhit
    for v in 1:(D + 2)
        @test isapprox(Q[v], P′[v]; rtol=rtol)
    end

    # A pressure-floored state, whose pressure sits *on* the floor. The
    # values come back to roundoff; whether the floor fires again is a
    # question about the last ulp of a round trip through `E` and is not
    # asserted — which is why idempotence is claimed above on
    # `apply_floors`, where it is exact, and not here.
    P′, hit = apply_floors(eos, floors,
                           (one(T), ntuple(_ -> half, Val(D))..., T(1 // 100)))
    @test hit
    @test pressure_of(P′) === floors.p_floor
    Q, _ = con2prim(eos, floors, prim2con(eos, P′))
    for v in 1:(D + 2)
        @test isapprox(Q[v], P′[v]; rtol=rtol)
    end
end

@testset "The equation of state is the ideal gas law: T=$T" for T in EOS_FLOATTYPES
    # Guards the three functions the rest of the scheme closes through,
    # and the two directions of the closure against each other. `c_s²` is
    # the one that matters beyond this file: it sets the time step through
    # `|v| + c_s` and both wave-speed estimates of every HLL-family flux,
    # and a factor of `γ` misplaced in it would show up as a run that is
    # merely slightly unstable.
    eos = eos_gas(T)
    rtol = eos_rtol(T)
    rng = MersenneTwister(1400)
    for _ in 1:8
        ρ = T(rand(rng, 500:2000) // 1000)
        p = T(rand(rng, 500:2000) // 1000)
        ε = internal_energy(eos, ρ, p)
        @test ε isa T
        @test pressure(eos, ρ, ε) isa T
        @test soundspeed(eos, ρ, p) isa T
        @test isapprox(pressure(eos, ρ, ε), p; rtol=rtol)
        @test isapprox(internal_energy(eos, ρ, pressure(eos, ρ, ε)), ε; rtol=rtol)
        @test isapprox(soundspeed(eos, ρ, p)^2, eos.γ * p / ρ; rtol=rtol)
    end
end

@testset "The parameters are isbits and refuse what is not a gas: T=$T" for
        T in EOS_FLOATTYPES
    # Guards the property every kernel launch and every callback depends
    # on — an `isbits` equation of state and an `isbits` set of floors are
    # what let them be captured and passed to a device — and the four
    # parameter mistakes that would otherwise surface as a NaN in the
    # middle of a run rather than at the call that made them.
    eos = eos_gas(T)
    floors = eos_floors(T)
    @test isbits(eos)
    @test isbits(floors)
    @test eos isa EquationOfState
    @test eos.γ === T(7 // 5)

    @test_throws "needs γ > 1" IdealGas(one(T))
    @test_throws "needs γ > 1" IdealGas{T}(1 // 2)
    @test_throws "needs γ > 1" IdealGas(-T(5 // 3))

    @test_throws "positive ρ_atm" Floors{T}(; ρ_atm=zero(T), p_atm=one(T),
                                            p_floor=one(T))
    @test_throws "positive p_atm" Floors{T}(; ρ_atm=one(T), p_atm=-one(T),
                                            p_floor=one(T))
    @test_throws "positive p_floor" Floors{T}(; ρ_atm=one(T), p_atm=one(T),
                                              p_floor=zero(T))
    @test_throws "p_atm ≥ p_floor" Floors{T}(; ρ_atm=one(T), p_atm=T(1 // 10^6),
                                             p_floor=one(T))
end

@testset "The conversions run inside a kernel: T=$T, D=$D" for
        T in EOS_FLOATTYPES, D in EOS_DIMS
    # The `isbits` claim made real. `eos` and `floors` are kernel
    # *arguments* here, not values a host closure reaches around the
    # launch for, which is how the right-hand side and the atmosphere
    # reset will pass them on a device; a struct that held a `Type`, an
    # array or an abstract field would fail to launch rather than fail a
    # comparison.
    #
    # Bit for bit against the host loop, not to roundoff: the same
    # arithmetic on the same values has one answer, and a tolerance here
    # would hide exactly the kind of difference — a reassociation, a
    # widened intermediate — that costs bit-identity across backends and
    # thread counts.
    eos = eos_gas(T)
    floors = eos_floors(T)
    M = D + 2
    rng = MersenneTwister(1500 + D)
    inputs = [random_prim(rng, T, Val(D)) for _ in 1:6]
    push!(inputs, (T(1 // 10^9), ntuple(_ -> one(T), Val(D))..., one(T)))
    # At rest, so that the pressure this one comes back with is its own
    # and not what is left of `E` after the kinetic energy is subtracted:
    # the point here is that the kernel takes the flooring branch, not how
    # accurately a tiny pressure survives a cancellation.
    push!(inputs, (one(T), ntuple(_ -> zero(T), Val(D))..., T(1 // 10^12)))
    n = length(inputs)

    states = Matrix{T}(undef, M, n)
    for i in 1:n, v in 1:M
        states[v, i] = inputs[i][v]
    end
    prims = similar(states)
    hits = Vector{Bool}(undef, n)

    eos_roundtrip_kernel!(CPU(), 4)(prims, hits, states, eos, floors, Val(M);
                                    ndrange=n)
    synchronize(CPU())

    for i in 1:n
        Q, hit = eos_roundtrip(eos, floors, inputs[i])
        @test hits[i] == hit
        for v in 1:M
            @test prims[v, i] === Q[v]
        end
    end
    # Both floor rules were exercised, and the physical states were not:
    # a kernel that agreed with the host because neither ever floored
    # would be no evidence at all.
    @test count(hits) == 2
end
