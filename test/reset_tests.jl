# The atmosphere reset: the one place in this package where something other
# than the flux divergence changes the evolved state.
#
# Everything before this file conserves because every stage's `du` is a
# difference of fluxes and sums to zero. The reset is the exception, and it
# is an exception the package rehearses on purpose: a star in a large vacuum
# region needs the atmosphere imposed on `U` itself, or the velocities where
# there is no gas run away and set the time step of the whole hierarchy. So
# the claims here are about what that exception costs and about the fact
# that, where it does not fire, it costs *exactly* nothing.
#
# The failure modes, one per testset:
#
#   * a reset that does not reach a fixed point — applying it twice differs
#     from applying it once, which would make the per-stage and the per-step
#     hook two different schemes rather than two cadences of one;
#   * a reset that leaves `U` and `P` disagreeing, so that the next
#     right-hand side reconstructs from primitives the state does not hold;
#   * an injection that is not what it says it is, which would make every
#     drift on Sedov unattributable;
#   * a reset that writes back where nothing fired — the quiet one. It
#     would move every cell by a few ulp per stage and turn the roundoff
#     conservation claims of the entropy wave, Sod and Kelvin–Helmholtz
#     into tolerances, and no test of the reset *itself* would notice;
#   * a hook that is never called, so that the reset exists and does
#     nothing. `SSPRK33`'s limiter hooks moved out of the algorithm
#     constructor upstream, and a version that ignored the old form would
#     fail here and nowhere else;
#   * a ghost count taken over the owned range, which would report zero for
#     the population the count exists to measure.
#
# The numbers recorded in the comments are in `CODE.md` under "Floors and
# the atmosphere"; a changed number is a regression.

# The conservative family at the interface order the scheme wants, as every
# other test file runs it.
reset_ops() = Operators(family=Conservative, prolongation=3, restriction=2)

"""
Floors whose atmosphere pressure sits three orders above the pressure floor.

That gap is not decoration. A reset atmosphere cell is written as
`prim2con(eos, (ρ_atm, 0…, p_atm))` and read back by the next `con2prim`; if
`p_atm` sat *at* `p_floor`, the round trip could land a fraction of an ulp
below it and the pressure rule would fire on a cell the reset had just
placed — which is the configuration `Floors` refuses outright when `p_atm <
p_floor` and the one this leaves room around.
"""
reset_floors(::Type{T}) where {T} =
    Floors{T}(; ρ_atm=T(1 // 10^4), p_atm=T(1 // 10^5), p_floor=T(1 // 10^8))

"""
One conserved cell state per population the floors distinguish, chosen so
that each reaches [`apply_floors`](@ref) by a different route:

    0  healthy gas                          nothing fires
    1  0 < ρ < ρ_atm                        the atmosphere rule
    2  ρ = 0                                the atmosphere, before the division
    3  ρ < 0                                the same, from the other side
    4  ρ = NaN                              the comparison written as a negation
    5  healthy ρ, E < ½S²/ρ                 the pressure floor

Classes 2 to 4 are the ones that say the atmosphere test comes *before*
`S/ρ` is formed: each of them would produce an `Inf` or a `NaN` velocity if
it did not. Class 5 is the only one where the momentum survives the reset,
which is what makes the injection tuple's middle entries nonzero.
"""
function synthetic_cell(::Type{T}, ::Val{D}, cls) where {T,D}
    eos = IdealGas(T(7 // 5))
    if cls == 0
        v = ntuple(d -> T(3 // 10) + T(d) / 20, Val(D))
        return prim2con(eos, (T(6 // 5), v..., T(9 // 10)))
    elseif cls == 1
        return (T(1 // 10^6), ntuple(_ -> T(1 // 10), Val(D))..., T(1 // 2))
    elseif cls == 2
        return (zero(T), ntuple(_ -> T(1 // 10), Val(D))..., T(1 // 2))
    elseif cls == 3
        return (-one(T), ntuple(_ -> T(1 // 10), Val(D))..., T(1 // 2))
    elseif cls == 4
        return (T(NaN), ntuple(_ -> T(1 // 10), Val(D))..., T(1 // 2))
    else
        ρ = T(6 // 5)
        return (ρ, ntuple(_ -> ρ * T(1 // 2), Val(D))..., T(1 // 100))
    end
end

"""
A uniform periodic mesh whose owned cells cycle through the six populations
of [`synthetic_cell`](@ref), with its problem, its state vector and the
accounting record the reset writes into.

The cycle is assigned in a host loop in block order, so the mesh is the same
at every thread count and the hand-computed answers below are the same
numbers the reductions see.
"""
function synthetic_reset_setup(::Type{T}, ::Val{D}; N=8, roots=2, measure=true,
                               classes=(0, 1, 2, 3, 4, 5)) where {T,D}
    forest = hydro_forest(Val(D), N; roots=roots, refined=false, T=T)
    U = FieldSet{T}(forest, D + 2; G=2)
    eos = IdealGas(T(7 // 5))
    floors = reset_floors(T)
    acc = ResetAccounting{T}(D + 2; measure=measure)
    p = HydroProblem(U, reset_ops(); eos=eos, floors=floors, limiter=:minmod,
                     accounting=acc)
    u = statevector(U)
    arr = statearray(u, U)
    host = zeros(T, size(arr))
    k = 0
    for b in 1:nblocks(U), idx in CartesianIndices(ntuple(_ -> N, D))
        cell = synthetic_cell(T, Val(D), classes[k % length(classes) + 1])
        k += 1
        for v in 1:(D + 2)
            host[Tuple(idx)..., v, b] = cell[v]
        end
    end
    copyto!(arr, host)
    return (forest=forest, U=U, p=p, u=u, eos=eos, floors=floors, acc=acc)
end

"""Every owned cell of a state vector as a conserved tuple, on the host."""
function owned_states(U::FieldSet{T,D}, u) where {T,D}
    arr = Array(statearray(u, U))
    N = U.forest.N
    return [ntuple(v -> arr[Tuple(idx)..., v, b], D + 2)
            for b in 1:nblocks(U) for idx in CartesianIndices(ntuple(_ -> N, D))]
end

@testset "The reset reaches a fixed point in one application: T=$T, D=$D" for
        T in (Float64, Float32), D in (1, 2, 3)
    # A reset that had to be applied twice would not be a reset: the stage
    # hook and the step hook would be two different schemes rather than two
    # cadences of one, and the state handed to the next right-hand side
    # would depend on how many times the integrator happened to call the
    # limiter. `CODE.md` predicted "to roundoff, bit-for-bit not claimed",
    # because the floor comparisons would have to absorb the
    # prim2con/con2prim round trip for the stronger claim.
    #
    # Measured in step 8, and the prediction is corrected in both
    # directions. The **state** is idempotent bit for bit, at `Float64` and
    # at `Float32`, in D = 1, 2 and 3. The **flag** is not: a
    # pressure-floored cell recovers its internal energy through the
    # cancellation `E − ½S²/ρ`, and where the kinetic energy dominates that
    # lands a fraction of an ulp below `p_floor`, so the rule fires a second
    # time — and writes `prim2con(eos, (ρ, v, p_floor))` from the same `ρ`
    # and the same `v`, which is the same arithmetic on the same numbers and
    # therefore the same bits. Measured second-pass counts on this state:
    # 2 of 13 at `Float64` in D = 1, none in D = 2, 85 of 426 in D = 3, and
    # none at `Float32` in any dimension. So the reset is a fixed point of
    # the state and not of its own report, which is why the counts are taken
    # per call rather than read off the flag slot afterwards.
    s = synthetic_reset_setup(T, Val(D); N=(D == 3 ? 4 : 8))
    reset_atmosphere!(s.u, nothing, s.p, zero(T))
    once = copy(s.u)
    hits_once = s.acc.hits
    reset_atmosphere!(s.u, nothing, s.p, zero(T))
    twice = copy(s.u)
    hits_twice = s.acc.hits - hits_once

    scale = maximum(abs, once)
    worst = maximum(abs.(twice .- once))
    @test worst == 0                              # measured in step 8
    @test twice == once
    @test scale > 0                               # not a state of all zeros
    # Every cell the second pass reports is one the first pass floored: the
    # rules may re-report, they may not spread.
    @test hits_twice ≤ hits_once
    @info "reset idempotence, T = $T, D = $D: $hits_once of " *
          "$(length(once) ÷ (D + 2)) cells floored, second pass reported " *
          "$hits_twice, |twice − once|∞ = $worst against a state scale of $scale"
end

@testset "con2prim of the reset state is the floored P, and healthy cells do not move: T=$T, D=$D" for
        T in (Float64, Float32), D in (1, 2)
    # The consistency the whole design rests on. The reset writes `U`; the
    # next right-hand side recovers `P` from it and reconstructs face states
    # from *that*. If the two disagreed, the scheme would be advancing a
    # state nobody had floored.
    #
    # The second half is the quiet claim and the important one: a cell where
    # no floor fired is **bit-identical** afterwards, because the kernel
    # writes back only where `hit` came back true. That is what makes the
    # injection exactly zero on a run that floors nowhere, and it is the
    # property a kernel written as an unconditional `prim2con(con2prim(U))`
    # would lose without failing anything else.
    s = synthetic_reset_setup(T, Val(D))
    before = owned_states(s.U, s.u)
    reset_atmosphere!(s.u, nothing, s.p, zero(T))
    after = owned_states(s.U, s.u)

    worst = zero(T)
    scaleP = zero(T)
    untouched = 0
    floored = 0
    for (Ub, Ua) in zip(before, after)
        Pb, hit = con2prim(s.eos, s.floors, Ub)
        Pa, _ = con2prim(s.eos, s.floors, Ua)
        for v in 1:(D + 2)
            worst = max(worst, abs(Pa[v] - Pb[v]))
            scaleP = max(scaleP, abs(Pb[v]))
        end
        if hit
            floored += 1
        else
            untouched += 1
            Ua == Ub || (untouched -= 1)       # a healthy cell that moved
        end
    end
    # The recovery agrees to roundoff *of the data's own scale*, which is
    # what the claim means: a pressure-floored cell recovers its internal
    # energy through the cancellation `E − ½S²/ρ`, so the agreement in `p`
    # is roundoff of `E` and not of `p_floor`.
    @test worst ≤ 64 * eps(T) * scaleP
    @test untouched == count(U -> !con2prim(s.eos, s.floors, U)[2], before)
    @test floored == s.acc.hits
    @test floored > 0 && untouched > 0
end

@testset "The injection is the hand-computed Σ hᴰ ΔU and the count is the cells: T=$T, D=$D" for
        T in (Float64, Float32), D in (1, 2)
    # What the accounting is *for*: on Sedov the drift is claimed as fixup
    # roundoff plus a measured injection, and a measured injection that did
    # not equal the state's own change would make the claim unfalsifiable.
    # So it is checked against the same sum taken by hand, cell by cell, on
    # the host.
    #
    # The `NaN` population is left out of *this* state on purpose and gets
    # its own claim at the foot of the testset: a state holding a `NaN` has
    # no total, so the honest injection into it is a `NaN` and not a number.
    # That is a property of the measurement rather than a defect of it — an
    # injection that came back `NaN` has said that the state was broken
    # before the floors saw it, which is what the flag says too.
    s = synthetic_reset_setup(T, Val(D); classes=(0, 1, 2, 3, 5))
    before = owned_states(s.U, s.u)
    reset_atmosphere!(s.u, nothing, s.p, zero(T))
    after = owned_states(s.U, s.u)

    # One spacing: the mesh is uniform, which is what lets the hand sum be a
    # single weighted total rather than a per-block one.
    h = minimum_spacing(T, s.forest)
    w = h^D
    hand = zeros(T, D + 2)
    scale = zeros(T, D + 2)
    for (Ub, Ua) in zip(before, after)
        for v in 1:(D + 2)
            hand[v] += w * (Ua[v] - Ub[v])
            scale[v] += w * (abs(Ua[v]) + abs(Ub[v]))
        end
    end
    for v in 1:(D + 2)
        @test abs(s.acc.injection[v] - hand[v]) ≤ 64 * eps(T) * scale[v]
    end
    @test s.acc.hits == count(U -> con2prim(s.eos, s.floors, U)[2], before)
    # The middle entries are nonzero only because one population keeps its
    # momentum: the atmosphere rule discards it, the pressure floor does not.
    @test any(v -> !iszero(hand[1 + v]), 1:D)

    # And the `NaN` population, whose injection is a `NaN` and says so — in
    # the variable the `NaN` lived in and in no other. The cells of that
    # population carry a `NaN` density and a finite momentum and energy, so
    # the mass total has no value before the reset while the other `D + 1`
    # do, and the accounting reports exactly that rather than a zero.
    n = synthetic_reset_setup(T, Val(D); classes=(0, 4))
    reset_atmosphere!(n.u, nothing, n.p, zero(T))
    @test isnan(n.acc.injection[1])
    @test all(v -> isfinite(n.acc.injection[v]), 2:(D + 2))
    @test n.acc.hits > 0
    # The *state* is repaired all the same: nothing finite is left broken.
    @test all(U -> all(isfinite, U), owned_states(n.U, n.u))
end

@testset "The ghost population is counted over the stored extent: T=$T, D=$D" for
        T in (Float64, Float32), D in (1, 2)
    # `block_mapreduce` reduces a block's interior, which is exactly the
    # range this count must *not* use: the whole point of the population
    # split is that a prolongated ghost can be unphysical where the owned
    # cell it came from was not. A ghost count written over the owned range
    # would report zero for the one population it exists to measure and
    # nothing would say so.
    #
    # Ghost *entries*, not cells: a physical cell that is a ghost of two
    # blocks counts twice, because what is measured is how often the
    # recovery meets an unphysical ghost.
    s = synthetic_reset_setup(T, Val(D); measure=false)
    update_primitives!(s.p, s.u)
    cons = Array(s.p.U.work)
    N, G = s.forest.N, s.U.G
    stored = ntuple(d -> size(cons, d), D)

    hand_ghost = 0
    hand_owned = 0
    for b in 1:nblocks(s.U), idx in CartesianIndices(stored)
        c = Tuple(idx)
        Uc = ntuple(v -> cons[c..., v, b], D + 2)
        _, hit = con2prim(s.eos, s.floors, Uc)
        hit || continue
        if all(d -> G[d] < c[d] ≤ G[d] + N, 1:D)
            hand_owned += 1
        else
            hand_ghost += 1
        end
    end
    @test ghost_floor_hits(s.p) == hand_ghost
    @test floor_hits(s.p) == hand_owned
    @test hand_ghost > 0                          # the count is not trivially 0
end

# The state a step has to survive: blocks of gas alternating with blocks of
# vacuum a hundred times below `ρ_atm`, so that the interior of a vacuum
# block is still vacuum after one step while its edge is not.
function vacuum_setup(::Type{T}, ::Val{D}; N=8, roots=2) where {T,D}
    forest = hydro_forest(Val(D), N; roots=roots, refined=false, T=T)
    U = FieldSet{T}(forest, D + 2; G=2)
    eos = IdealGas(T(7 // 5))
    floors = reset_floors(T)
    p = HydroProblem(U, reset_ops(); eos=eos, floors=floors, limiter=:minmod)
    u = statevector(U)
    arr = statearray(u, U)
    host = zeros(T, size(arr))
    zeros_ = ntuple(_ -> zero(T), Val(D))
    gas = prim2con(eos, (one(T), zeros_..., one(T)))
    vac = prim2con(eos, (floors.ρ_atm / 100, zeros_..., floors.p_atm / 100))
    for b in 1:nblocks(U), idx in CartesianIndices(ntuple(_ -> N, D))
        cell = isodd(b) ? vac : gas
        for v in 1:(D + 2)
            host[Tuple(idx)..., v, b] = cell[v]
        end
    end
    copyto!(arr, host)
    return (forest=forest, U=U, p=p, u=u, eos=eos, floors=floors)
end

@testset "One SSPRK33 step comes out floored under :stage and :step and not under :none: T=$T, D=$D" for
        T in (Float64, Float32), D in (1, 2)
    # The wiring, and the reason it is a test rather than a reading of the
    # integrator's source. The limiter hooks moved out of `SSPRK33`'s
    # constructor and into `solve`'s keywords upstream (amended in step 8),
    # and the failure mode of getting that wrong is **silence**: the reset
    # would be installed nowhere, every cell would keep whatever the flux
    # divergence gave it, and every other claim in this file — which calls
    # `reset_atmosphere!` directly — would still pass.
    #
    # `:none` is the control and it is what says the step itself does not do
    # the flooring: the right-hand side floors `P` internally, so the run
    # completes, and the *state* is left below `ρ_atm` where it started.
    s = vacuum_setup(T, Val(D))
    update_primitives!(s.p, s.u)
    λ = max_signal_speed(s.p)
    dt = hydro_dt(s.forest, T(2 // 5), λ, Val(D))

    ρ_atm, p_floor = s.floors.ρ_atm, s.floors.p_floor
    results = map((:stage, :step, :none)) do reset
        u = hydro_solve!(s.p, copy(s.u), zero(T), dt, 1; reset=reset)
        states = owned_states(s.U, u)
        ρ_min = minimum(U -> density(U), states)
        # The raw recovery, *before* the floors: `con2prim`'s own output is
        # floored by construction and would make the claim vacuous.
        p_min = minimum(states) do U
            ρ = density(U)
            ρ ≥ ρ_atm || return -one(T)
            ε = (energy(U) - sum(momentum(U) .^ 2) / ρ / 2) / ρ
            return pressure(s.eos, ρ, ε)
        end
        (reset=reset, ρ_min=ρ_min, p_min=p_min)
    end
    for r in results
        @info "one SSPRK33 step on a vacuum state, T = $T, D = $D, " *
              "reset = :$(r.reset): min ρ $(r.ρ_min) against ρ_atm $ρ_atm, " *
              "min recovered p $(r.p_min) against p_floor $p_floor"
    end
    for r in results[1:2]
        @test r.ρ_min ≥ ρ_atm
        # To roundoff: a pressure-floored cell recovers its internal energy
        # through `E − ½S²/ρ`, so it may land a few ulp of `E` below.
        @test r.p_min ≥ p_floor * (1 - 1024 * eps(T))
    end
    @test results[3].ρ_min < ρ_atm
end

# A short tracked shock tube, in the configuration `driver_tests.jl` uses
# with its `t_end` shortened: the point here is the reset and not the tube,
# and three runs of the full length would be three times the arithmetic for
# the same answer.
reset_sod_run(; reset, accounting=false) =
    evolve!(HydroCase(SodTube(Float64, Val(1)); roots=(8,)), Val(1); N=8,
            ops=reset_ops(), t_end=1 // 50, chunk=1 // 200, limiter=:minmod,
            refine_tol=2 // 25, coarsen_tol=1 // 50, maxlevel_cap=2,
            reset=reset, accounting=accounting)

@testset "Where no floor fires the reset changes exactly nothing" begin
    # The claim the conservation results of steps 3 to 7 now depend on. With
    # `reset = :stage` the default, every run in this suite calls
    # `reset_atmosphere!` three times per step; if the reset wrote back where
    # nothing had fired, every one of those runs would move by a few ulp per
    # stage and every roundoff drift bound in `CODE.md` would have become a
    # tolerance. So it is asserted the strongest way available: the final
    # state, the drift, the step count and the mesh history are **bit
    # identical** to the `reset = :none` run's, and the injection is
    # `0.0` and not `≈ 0.0`.
    #
    # Measured in step 8: injection exactly (0.0, 0.0, 0.0), 0 reset hits and
    # 0 ghost hits on the tracked tube under both hooks.
    none = reset_sod_run(; reset=:none)
    stage = reset_sod_run(; reset=:stage, accounting=true)
    step = reset_sod_run(; reset=:step, accounting=true)
    @test none.injection === nothing            # not measured is not zero
    for r in (stage, step)
        @test r.injection == ntuple(_ -> 0.0, 3)
        @test r.reset_hits == 0
        @test r.ghost_hits == 0
        @test r.floor_hits == 0
        @test r.u == none.u
        @test r.drift == none.drift
        @test r.nsteps == none.nsteps
        @test r.nblocks_history == none.nblocks_history
        @test r.l1 == none.l1
    end
    @info "Sod tracked, D = 1, t_end = 1/50: reset injection " *
          "$(stage.injection) under :stage and $(step.injection) under :step, " *
          "$(stage.reset_hits) reset hits, $(stage.ghost_hits) ghost hits over " *
          "$(stage.nsteps) steps and $(stage.nregrids) mesh changes; the state " *
          "is bit-identical to the reset = :none run's"

    # And the periodic case with no boundary hook, where the tube's two
    # Dirichlet faces cannot be the reason nothing fires.
    wave = uniform_run(HydroCase(EntropyWave(Float64, Val(1)); roots=4), Val(1);
                       N=8, ops=reset_ops(), t_end=1 // 4, chunk=1 // 20,
                       limiter=:none, reset=:stage, accounting=true)
    @test wave.injection == ntuple(_ -> 0.0, 3)
    @test wave.reset_hits == 0
    @test wave.ghost_hits == 0
    @test wave.floor_hits == 0
    @info "entropy wave through the driver, D = 1, N = 8, reset = :stage: " *
          "injection $(wave.injection), $(wave.reset_hits) reset hits, " *
          "$(wave.ghost_hits) ghost hits over $(wave.nsteps) steps"
end

@testset "An unknown reset is refused by name, wherever it is passed" begin
    # The keyword takes three symbols and a typo is a run that silently does
    # something else — `:stages` would be a scheme with no reset at all. Both
    # entry points refuse it, and `evolve!` refuses it *before* it adapts a
    # mesh rather than at the first chunk.
    s = vacuum_setup(Float64, Val(1))
    @test_throws "reset must be one of" hydro_solve!(s.p, copy(s.u), 0.0, 1.0e-4,
                                                     1; reset=:stages)
    @test_throws "reset must be one of" reset_sod_run(; reset=:stages)
end
